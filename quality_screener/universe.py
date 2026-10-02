"""Company universe: current S&P 500 constituents mapped to SEC CIKs, with SIC codes, a
SIC-division sector, and flags for industries where the scores are not meaningful.

SURVIVORSHIP BIAS - read this before trusting any backtest built on this universe.
The universe is *today's* S&P 500 list, applied to every year back to 2010. Firms that went
bankrupt, were acquired, or were dropped from the index during the sample are missing, and
those are disproportionately the low-quality firms that ended badly. Leaving them out flatters
the low-quality quintile, which biases any high-minus-low quality spread measured here toward
zero. Today's list also contains firms that only joined the index later because they grew;
putting them in the 2010 sample selects on future success. Treat results as indicative only.

TODO: add delisted and removed firms (historical S&P 500 membership changes, their CIKs and
EDGAR data, and delisting returns) so the universe is point-in-time.

Financials (SIC 6000-6999) and utilities (SIC 4900-4999) are flagged, not dropped: Altman Z
and Piotroski F were built for industrial firms and are not meaningful for them, so they are
excluded from the main tests but kept in the data.

The fallback CSV (``data/sp500_constituents.csv``) is a snapshot of the Wikipedia table taken
on ``FALLBACK_SNAPSHOT_DATE`` with ``write_fallback_snapshot``.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import requests
from tqdm import tqdm

from quality_screener import config
from quality_screener.edgar import EdgarError

if TYPE_CHECKING:
    from quality_screener.edgar import EdgarClient

log = logging.getLogger(__name__)

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK_CSV = config.DATA_DIR / "sp500_constituents.csv"
FALLBACK_SNAPSHOT_DATE = "2026-10-02"
UNIVERSE_PATH = config.PROCESSED_DIR / "universe.parquet"
MIN_CONSTITUENTS = 450  # sanity floor: a smaller scraped table means the page layout changed

FINANCIAL_SIC = (6000, 6999)
UTILITY_SIC = (4900, 4999)

# Divisions of the Standard Industrial Classification Manual (1987), as used by SEC/EDGAR.
SIC_DIVISIONS: tuple[tuple[int, int, str], ...] = (
    (100, 999, "Agriculture, Forestry & Fishing"),
    (1000, 1499, "Mining"),
    (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"),
    (4000, 4999, "Transportation, Communications & Utilities"),
    (5000, 5199, "Wholesale Trade"),
    (5200, 5999, "Retail Trade"),
    (6000, 6799, "Finance, Insurance & Real Estate"),
    (7000, 8999, "Services"),
    (9100, 9729, "Public Administration"),
    (9900, 9999, "Nonclassifiable"),
)

UNIVERSE_COLUMNS = [
    "cik", "ticker", "all_tickers", "name", "title", "gics_sector", "sic", "sic_description",
    "sector", "fiscal_year_end", "is_financial", "is_utility", "excluded_from_main",
]


def parse_constituents_table(html: str, min_rows: int = MIN_CONSTITUENTS) -> pd.DataFrame:
    """Extract ``ticker``, ``name``, ``gics_sector`` from the Wikipedia constituents table."""
    table = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
    out = table.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "gics_sector"})
    out = out[["ticker", "name", "gics_sector"]]
    if len(out) < min_rows:
        raise ValueError(f"constituents table has {len(out)} rows, expected >= {min_rows}")
    return out.reset_index(drop=True)


def fetch_constituents(
    session: requests.Session | None = None, fallback_csv: Path = FALLBACK_CSV
) -> pd.DataFrame:
    """Current S&P 500 constituents from Wikipedia, or the CSV snapshot if the scrape fails."""
    session = session if session is not None else requests.Session()
    try:
        resp = session.get(WIKIPEDIA_URL, headers={"User-Agent": config.USER_AGENT}, timeout=30)
        resp.raise_for_status()
        return parse_constituents_table(resp.text)
    except (requests.RequestException, ValueError, KeyError) as exc:
        log.warning("Wikipedia scrape failed (%s); using snapshot %s", exc, fallback_csv)
        return pd.read_csv(fallback_csv, dtype=str)


def write_fallback_snapshot(path: Path = FALLBACK_CSV) -> pd.DataFrame:
    """Scrape Wikipedia now and overwrite the fallback CSV (update FALLBACK_SNAPSHOT_DATE too)."""
    resp = requests.get(WIKIPEDIA_URL, headers={"User-Agent": config.USER_AGENT}, timeout=30)
    resp.raise_for_status()
    constituents = parse_constituents_table(resp.text)
    constituents.to_csv(path, index=False)
    return constituents


def normalize_ticker(ticker: str) -> str:
    """Share-class tickers use '.' on Wikipedia (BRK.B) and '-' at SEC (BRK-B); use SEC's form."""
    return ticker.strip().upper().replace(".", "-")


def map_tickers_to_ciks(constituents: pd.DataFrame, ticker_map: pd.DataFrame) -> pd.DataFrame:
    """Attach ``cik`` and SEC ``title`` to each ticker; tickers SEC does not know are dropped."""
    left = constituents.assign(ticker=constituents["ticker"].map(normalize_ticker))
    right = ticker_map.assign(ticker=ticker_map["ticker"].map(normalize_ticker))
    right = right.drop_duplicates("ticker")[["ticker", "cik", "title"]]
    merged = left.merge(right, on="ticker", how="left", validate="many_to_one")
    unmapped = merged.loc[merged["cik"].isna(), "ticker"].tolist()
    if unmapped:
        log.warning("%d tickers missing from SEC ticker map, dropped: %s", len(unmapped), unmapped)
    return merged.dropna(subset=["cik"]).reset_index(drop=True)


def collapse_share_classes(firms: pd.DataFrame) -> pd.DataFrame:
    """One row per CIK: fundamentals are filed per registrant, not per share class.

    The first ticker (table order) becomes ``ticker``; all of them go into ``all_tickers``.
    """
    all_tickers = firms.groupby("cik", sort=False)["ticker"].agg(",".join).rename("all_tickers")
    return firms.drop_duplicates("cik").join(all_tickers, on="cik").reset_index(drop=True)


def sic_to_sector(sic: float | int | None) -> str | None:
    """Map a 4-digit SIC code to its SIC Manual (1987) division; None if unknown or unmapped."""
    if sic is None or pd.isna(sic):
        return None
    for low, high, division in SIC_DIVISIONS:
        if low <= sic <= high:
            return division
    return None


def flag_sic_range(sic: pd.Series, bounds: tuple[int, int]) -> pd.Series:
    """Nullable boolean: True inside ``bounds`` (inclusive), NA where SIC is unknown."""
    return sic.astype("Int64").between(*bounds).astype("boolean")


def add_sic_and_flags(firms: pd.DataFrame, client: EdgarClient) -> pd.DataFrame:
    """Pull SIC and fiscal year end from submissions; add sector and industry flags."""
    records = [_sic_record(cik, client) for cik in tqdm(firms["cik"], desc="submissions")]
    out = firms.join(pd.DataFrame(records, index=firms.index))
    out["sic"] = out["sic"].astype("Int64")
    out["sector"] = out["sic"].map(sic_to_sector)
    out["is_financial"] = flag_sic_range(out["sic"], FINANCIAL_SIC)
    out["is_utility"] = flag_sic_range(out["sic"], UTILITY_SIC)
    out["excluded_from_main"] = out["is_financial"] | out["is_utility"]
    return out


def _sic_record(cik: str, client: EdgarClient) -> dict[str, object]:
    try:
        sub = client.get_submissions(cik)
    except EdgarError as exc:
        log.warning("no submissions for CIK %s (%s); SIC left missing", cik, exc)
        return {"sic": None, "sic_description": None, "fiscal_year_end": None}
    return {
        "sic": sub.sic,
        "sic_description": sub.sic_description or None,
        "fiscal_year_end": sub.fiscal_year_end or None,
    }


def build_universe(client: EdgarClient, constituents: pd.DataFrame | None = None) -> pd.DataFrame:
    """Constituents -> CIKs -> one row per firm -> SIC, sector and exclusion flags."""
    if constituents is None:
        constituents = fetch_constituents()
    mapped = map_tickers_to_ciks(constituents, client.get_ticker_map())
    firms = collapse_share_classes(mapped)
    return add_sic_and_flags(firms, client)[UNIVERSE_COLUMNS]


def load_or_build_universe(client: EdgarClient, path: Path = UNIVERSE_PATH) -> pd.DataFrame:
    """Read the saved universe, building and saving it first if it does not exist yet."""
    if path.exists():
        return pd.read_parquet(path)
    universe = build_universe(client)
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(path, index=False)
    log.info("saved %d firms to %s", len(universe), path)
    return universe
