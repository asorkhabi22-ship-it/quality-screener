"""Universe construction: scraping/fallback, ticker->CIK mapping, share classes, SIC flags."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import requests
from fakes import FakeResponse, FakeSession

from quality_screener import universe
from quality_screener.edgar import EdgarError, Submissions

TICKER_MAP = pd.DataFrame(
    {
        "cik": ["0001067983", "0001652044", "0001652044", "0000320193", "0000072971", "0000753308"],
        "ticker": ["BRK-B", "GOOGL", "GOOG", "AAPL", "WFC", "NEE"],
        "title": ["Berkshire", "Alphabet", "Alphabet", "Apple", "Wells Fargo", "NextEra"],
    }
)


def test_normalize_ticker_uses_sec_share_class_format() -> None:
    assert universe.normalize_ticker(" brk.b ") == "BRK-B"
    assert universe.normalize_ticker("BF-B") == "BF-B"


def test_map_tickers_handles_share_classes_and_drops_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    constituents = pd.DataFrame({"ticker": ["BRK.B", "GOOGL", "GOOG", "ZZZZ"]})
    with caplog.at_level(logging.WARNING):
        mapped = universe.map_tickers_to_ciks(constituents, TICKER_MAP)
    assert mapped["ticker"].tolist() == ["BRK-B", "GOOGL", "GOOG"]
    assert mapped["cik"].tolist() == ["0001067983", "0001652044", "0001652044"]
    assert "ZZZZ" in caplog.text


def test_collapse_share_classes_keeps_one_row_per_cik() -> None:
    mapped = universe.map_tickers_to_ciks(pd.DataFrame({"ticker": ["GOOGL", "GOOG", "AAPL"]}), TICKER_MAP)
    firms = universe.collapse_share_classes(mapped)
    assert firms["cik"].is_unique
    alphabet = firms.set_index("cik").loc["0001652044"]
    assert (alphabet["ticker"], alphabet["all_tickers"]) == ("GOOGL", "GOOGL,GOOG")


@pytest.mark.parametrize(
    ("sic", "sector"),
    [
        (3571, "Manufacturing"),
        (7372, "Services"),
        (6021, "Finance, Insurance & Real Estate"),
        (4911, "Transportation, Communications & Utilities"),
        (5331, "Retail Trade"),
        (1850, None),  # gap between divisions
        (None, None),
        (float("nan"), None),
    ],
)
def test_sic_to_sector(sic: float | None, sector: str | None) -> None:
    assert universe.sic_to_sector(sic) == sector


class FakeEdgar:
    """Minimal EdgarClient stand-in: fixed ticker map and SIC per CIK."""

    def __init__(self, sics: dict[str, int | None], missing: set[str] = frozenset()) -> None:
        self.sics = sics
        self.missing = missing

    def get_ticker_map(self) -> pd.DataFrame:
        return TICKER_MAP

    def get_submissions(self, cik: str) -> Submissions:
        if cik in self.missing:
            raise EdgarError("HTTP 404")
        return Submissions(cik, "x", self.sics[cik], "desc", "1231", pd.DataFrame())


def test_build_universe_flags_financials_and_utilities_but_keeps_them() -> None:
    constituents = pd.DataFrame(
        {
            "ticker": ["AAPL", "WFC", "NEE", "GOOGL", "GOOG", "BRK.B"],
            "name": ["Apple", "Wells", "NextEra", "Alphabet A", "Alphabet C", "Berkshire"],
            "gics_sector": ["IT", "Financials", "Utilities", "Comm", "Comm", "Financials"],
        }
    )
    client = FakeEdgar(
        {"0000320193": 3571, "0000072971": 6021, "0000753308": 4911, "0001652044": 7370},
        missing={"0001067983"},
    )
    raw = universe.build_universe(client, constituents)
    assert list(raw.columns) == universe.UNIVERSE_COLUMNS
    built = raw.set_index("ticker")

    assert len(built) == 5  # GOOG folded into GOOGL; nothing dropped for being a financial
    assert built.loc["WFC", "is_financial"] and built.loc["WFC", "excluded_from_main"]
    assert built.loc["NEE", "is_utility"] and built.loc["NEE", "excluded_from_main"]
    assert not built.loc["AAPL", "excluded_from_main"]
    # unknown SIC stays missing instead of being treated as "not financial"
    assert pd.isna(built.loc["BRK-B", "sic"])
    assert pd.isna(built.loc["BRK-B", "is_financial"])
    assert pd.isna(built.loc["BRK-B", "sector"])


CONSTITUENTS_HTML = """
<table id="constituents">
  <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>CIK</th></tr>
  <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>320193</td></tr>
  <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>1067983</td></tr>
</table>
"""


def test_parse_constituents_table() -> None:
    table = universe.parse_constituents_table(CONSTITUENTS_HTML, min_rows=2)
    assert list(table.columns) == ["ticker", "name", "gics_sector"]
    assert table["ticker"].tolist() == ["AAPL", "BRK.B"]


def test_parse_rejects_suspiciously_short_table() -> None:
    with pytest.raises(ValueError, match="expected >= 450"):
        universe.parse_constituents_table(CONSTITUENTS_HTML)


def test_fetch_constituents_falls_back_to_csv(tmp_path: Path) -> None:
    fallback = tmp_path / "sp500.csv"
    pd.DataFrame({"ticker": ["AAPL"], "name": ["Apple"], "gics_sector": ["IT"]}).to_csv(fallback, index=False)
    session = FakeSession({universe.WIKIPEDIA_URL: [requests.ConnectionError("offline")]})
    got = universe.fetch_constituents(session, fallback_csv=fallback)  # type: ignore[arg-type]
    assert got["ticker"].tolist() == ["AAPL"]


def test_fetch_constituents_falls_back_on_http_error(tmp_path: Path) -> None:
    fallback = tmp_path / "sp500.csv"
    pd.DataFrame({"ticker": ["MSFT"], "name": ["Microsoft"], "gics_sector": ["IT"]}).to_csv(fallback, index=False)

    class Raising(FakeResponse):
        def raise_for_status(self) -> None:
            raise requests.HTTPError("503")

    session = FakeSession({universe.WIKIPEDIA_URL: [Raising(503)]})
    got = universe.fetch_constituents(session, fallback_csv=fallback)  # type: ignore[arg-type]
    assert got["ticker"].tolist() == ["MSFT"]


def test_load_or_build_universe_reads_saved_file(tmp_path: Path) -> None:
    saved = pd.DataFrame({"cik": ["0000320193"], "ticker": ["AAPL"]})
    path = tmp_path / "universe.parquet"
    saved.to_parquet(path, index=False)
    got = universe.load_or_build_universe(client=None, path=path)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(got, saved)
