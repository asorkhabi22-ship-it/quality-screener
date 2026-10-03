"""Tests for prices: parsing yfinance output, batching and caching, split adjustment, month-end
returns, gap detection, market-cap timing and the share-count check. No network access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quality_screener import prices

DATES = pd.bdate_range("2020-01-01", "2020-01-10")


def bars(close: list[float], dates: pd.DatetimeIndex = DATES, **fields: list[float]) -> pd.DataFrame:
    """One ticker's yf.download bars; extra fields by Yahoo name (e.g. Stock_Splits=...)."""
    frame = pd.DataFrame({"Open": close, "Close": close, "Adj Close": close, "Volume": 1e3},
                         index=pd.DatetimeIndex(dates, name="Date"))
    for name, values in fields.items():
        frame[name.replace("_", " ")] = values
    return frame


class FakeDownloader:
    """Stands in for yf.download(group_by='ticker'): tickers it does not know come back as
    all-NaN columns, like Yahoo's 'possibly delisted' failures. ``flaky`` tickers fail once."""

    def __init__(self, known: dict[str, pd.DataFrame], flaky: set[str] | None = None) -> None:
        self.known = known
        self.flaky = set(flaky or ())
        self.calls: list[list[str]] = []

    def __call__(self, tickers: list[str], start: str) -> pd.DataFrame:
        self.calls.append(list(tickers))
        out = {}
        for t in tickers:
            if t in self.known and t not in self.flaky:
                out[t] = self.known[t]
            else:
                self.flaky.discard(t)
                out[t] = bars([np.nan] * len(DATES))
        return pd.concat(out, axis=1)


def daily_frame(ticker: str, dates: pd.DatetimeIndex, close: list[float],
                adj: list[float] | None = None, splits: list[float] | None = None) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "ticker": ticker, "date": pd.DatetimeIndex(dates).astype("datetime64[ns]"),
        "close": close, "adj_close": adj if adj is not None else close,
        "dividends": 0.0, "splits": splits if splits is not None else [0.0] * n, "volume": 1e3,
    })


def with_factors(*frames: pd.DataFrame) -> pd.DataFrame:
    daily = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"], ignore_index=True)
    return prices.add_split_factor(daily)


# --- parsing, batching, caching ------------------------------------------------------------


def test_tidy_daily_drops_empty_rows_and_keeps_missing_actions_as_nan() -> None:
    raw = bars([10.0, np.nan, 12.0], dates=DATES[:3])  # no Dividends / Stock Splits columns
    out = prices.tidy_daily(raw, "AAA")
    assert out.columns.tolist() == prices.DAILY_COLUMNS
    assert out["close"].tolist() == [10.0, 12.0]
    assert out["splits"].isna().all() and out["dividends"].isna().all()
    assert out["date"].dtype == "datetime64[ns]"


def test_download_in_batches_caches_and_second_run_makes_no_calls(tmp_path: Path) -> None:
    tickers = [f"T{i:03d}" for i in range(120)]
    fake = FakeDownloader({t: bars([1.0] * len(DATES)) for t in tickers})
    missing = prices.download_prices(tickers, cache_dir=tmp_path, download=fake, sleep=lambda s: None)
    assert missing == []
    assert [len(c) for c in fake.calls] == [50, 50, 20]
    assert len(list(tmp_path.glob("*.parquet"))) == 120

    again = prices.download_prices(tickers, cache_dir=tmp_path, download=fake, sleep=lambda s: None)
    assert again == [] and len(fake.calls) == 3  # everything served from the cache


def test_empty_tickers_are_retried_then_reported_and_not_cached(tmp_path: Path) -> None:
    fake = FakeDownloader({"AAA": bars([1.0] * 8), "BBB": bars([2.0] * 8)}, flaky={"BBB"})
    missing = prices.download_prices(
        ["AAA", "BBB", "ZZZ"], cache_dir=tmp_path, download=fake, sleep=lambda s: None
    )
    assert missing == ["ZZZ"]
    assert fake.calls == [["AAA", "BBB", "ZZZ"], ["BBB", "ZZZ"]]
    assert not prices.cache_path("ZZZ", tmp_path).exists()
    assert prices.load_daily(["AAA", "BBB", "ZZZ"], tmp_path)["ticker"].unique().tolist() == ["AAA", "BBB"]


def test_load_daily_without_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prices.load_daily(["AAA"], tmp_path)


# --- split adjustment and calendar ----------------------------------------------------------


def test_raw_close_undoes_yahoo_split_adjustment() -> None:
    """AAPL-style 4:1 split on day 3: Yahoo shows 25, 26 for days 1-2, but 100 and 104 traded."""
    a = daily_frame("AAA", DATES[:4], [25.0, 26.0, 27.0, 28.0], splits=[0.0, 0.0, 4.0, 0.0])
    b = daily_frame("BBB", DATES[:4], [5.0, 6.0, 7.0, 8.0], splits=[0.0, 2.0, 0.0, 0.5])
    out = with_factors(a, b).set_index(["ticker", "date"])
    assert out.loc["AAA", "split_factor"].tolist() == [4.0, 4.0, 1.0, 1.0]
    assert out.loc["AAA", "close_raw"].tolist() == [100.0, 104.0, 27.0, 28.0]
    # 2:1 split on day 2, then a 1:2 reverse split on day 4: day 1's factor is 2 * 0.5
    assert out.loc["BBB", "split_factor"].tolist() == [1.0, 0.5, 0.5, 1.0]
    assert (out["price_factor"] == out["split_factor"]).all()  # true splits move both


def test_spin_off_ratio_restores_price_but_not_share_count() -> None:
    """ABT 2013-01-02: Yahoo writes the AbbVie spin-off as a 2.084 'split'. The traded price is
    close * 2.084 before it, but ABT's share count did not change."""
    abt = daily_frame("ABT", DATES[:3], [31.0, 32.0, 33.0], splits=[0.0, 2.084, 0.0])
    out = with_factors(abt)
    assert out["price_factor"].tolist() == pytest.approx([2.084, 1.0, 1.0])
    assert out["split_factor"].tolist() == [1.0, 1.0, 1.0]
    assert out["close_raw"].iloc[0] == pytest.approx(31.0 * 2.084)
    assert (out["close_split_adj"] == out["close_raw"]).all()


def test_listed_mixed_event_uses_its_share_count_ratio() -> None:
    """GOOGL 2014-04-03: Yahoo ratio 1.998 (price), but one Class C share per share = 2.0."""
    dates = pd.DatetimeIndex(["2014-04-02", "2014-04-03"])
    out = with_factors(daily_frame("GOOGL", dates, [565.0, 570.0], splits=[0.0, 1.998]))
    assert out["price_factor"].tolist() == pytest.approx([1.998, 1.0])
    assert out["split_factor"].tolist() == pytest.approx([2.0, 1.0])


def test_simple_split_ratios() -> None:
    ratios = pd.Series([2.0, 0.333333, 1.5, 50.0, 2.084, 1.398, 1.3348, 0.4725])
    assert prices.is_simple_split(ratios).tolist() == [True] * 4 + [False] * 4


def test_trading_calendar_is_benchmark_dates() -> None:
    daily = with_factors(daily_frame("SPY", DATES[:3], [1.0] * 3), daily_frame("AAA", DATES, [1.0] * 8))
    assert prices.trading_calendar(daily).tolist() == DATES[:3].tolist()
    with pytest.raises(ValueError):
        prices.trading_calendar(daily[daily["ticker"] == "AAA"])


def test_next_and_previous_trading_day() -> None:
    cal = pd.DatetimeIndex(pd.bdate_range("2020-03-02", "2020-03-31")).astype("datetime64[ns]")
    dates = pd.Series(pd.to_datetime(["2020-03-13", "2020-03-14", "2020-03-16", "2020-03-31"]))
    nxt = prices.next_trading_day(dates, cal)
    assert nxt[:3].tolist() == [pd.Timestamp("2020-03-16"), pd.Timestamp("2020-03-16"),
                                pd.Timestamp("2020-03-17")]
    assert pd.isna(nxt.iloc[3])  # past the end of the calendar
    prev = prices.previous_trading_day(pd.DatetimeIndex(["2020-03-16", "2020-03-31"]), cal)
    assert prev.tolist() == [pd.Timestamp("2020-03-13"), pd.Timestamp("2020-03-30")]


# --- monthly series and coverage -------------------------------------------------------------


def test_monthly_total_returns_hand_calculated() -> None:
    cal = pd.bdate_range("2020-01-01", "2020-03-31").astype("datetime64[ns]")
    adj = np.linspace(90.0, 100.0, len(cal))
    jan, feb, mar = (cal.month == m for m in (1, 2, 3))
    adj[jan.nonzero()[0][-1]] = 100.0
    adj[feb.nonzero()[0][-1]] = 110.0
    adj[mar.nonzero()[0][-1]] = 99.0
    x = daily_frame("XXX", cal, list(adj), adj=list(adj))
    gappy = cal[(cal <= "2020-01-15") | mar]  # stops mid-January, no February at all
    y = daily_frame("YYY", gappy, [10.0] * len(gappy))
    monthly = prices.monthly_prices(with_factors(x, y), cal).set_index(["ticker", "t"])

    xr = monthly.loc["XXX", "ret"]
    assert np.isnan(xr.iloc[0])
    assert xr.iloc[1] == pytest.approx(0.10)  # 110 / 100 - 1
    assert xr.iloc[2] == pytest.approx(-0.10)  # 99 / 110 - 1
    assert monthly.loc["XXX"].index.tolist() == [
        pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-31")]
    jan_y = monthly.loc[("YYY", pd.Timestamp("2020-01-31"))]
    assert jan_y["date"] == pd.Timestamp("2020-01-15")  # last own trade, keyed to month end
    assert np.isnan(monthly.loc[("YYY", pd.Timestamp("2020-03-31")), "ret"])  # Feb missing


def test_price_coverage_flags_gaps_and_early_ends() -> None:
    cal = pd.bdate_range("2020-01-01", periods=60).astype("datetime64[ns]")
    full = daily_frame("AAA", cal, [1.0] * 60)
    holey = cal[np.r_[0:20, 32:60]]  # 12 trading days missing
    short = cal[:50]  # ends 10 trading days early
    cov = prices.price_coverage(
        with_factors(full, daily_frame("BBB", holey, [1.0] * 48), daily_frame("CCC", short, [1.0] * 50)),
        cal,
    ).set_index("ticker")
    assert pd.isna(cov.loc["AAA", "issue"]) and cov.loc["AAA", "max_gap_days"] == 0
    assert cov.loc["BBB", "issue"] == "gap" and cov.loc["BBB", "max_gap_days"] == 12
    assert cov.loc["BBB", "missing_days"] == 12
    assert cov.loc["CCC", "issue"] == "ends early" and cov.loc["CCC", "trading_days_to_end"] == 10


# --- market cap ----------------------------------------------------------------------------

CAL = pd.bdate_range("2020-08-03", "2020-10-30").astype("datetime64[ns]")
FIRMS = pd.DataFrame({"cik": ["1", "2", "3"], "ticker": ["AAA", "BBB", "CCC"], "name": ["A", "B", "C"]})


def panel_rows(**cols: list[object]) -> pd.DataFrame:
    out = pd.DataFrame(cols)
    for c in ("availability_date", "first_filed", "period_end"):
        if c in out:
            out[c] = pd.to_datetime(out[c]).astype("datetime64[us]")
    return out


def test_market_cap_uses_close_on_first_trading_day_after_availability() -> None:
    close = np.where(CAL == pd.Timestamp("2020-08-14"), 999.0, 10.0)  # Friday availability
    daily = with_factors(daily_frame("AAA", CAL, list(close)))
    panel = panel_rows(cik=["1"], period_end=["2020-06-30"], fiscal_year=[2020],
                       availability_date=["2020-08-14"], first_filed=["2020-08-14"],
                       shares_out=[100e6])
    out = prices.add_market_cap(panel, FIRMS, daily, CAL)
    assert out.loc[0, "market_cap_date"] == pd.Timestamp("2020-08-17")  # Monday, not Friday
    assert out.loc[0, "market_cap"] == pytest.approx(10.0 * 100e6)


def test_market_cap_survives_a_split_between_filing_and_first_trade() -> None:
    """Filed Friday 2020-08-28 with 100M pre-split shares; 4:1 split ex-date Monday 08-31, the
    first trading day after filing. Naive close_raw * shares_out would give 130 * 100M."""
    close = np.where(CAL < pd.Timestamp("2020-08-31"), 500.0 / 4, 130.0)
    splits = np.where(CAL == pd.Timestamp("2020-08-31"), 4.0, 0.0)
    daily = with_factors(daily_frame("AAA", CAL, list(close), splits=list(splits)))
    panel = panel_rows(cik=["1"], period_end=["2020-06-30"], fiscal_year=[2020],
                       availability_date=["2020-08-28"], first_filed=["2020-08-28"],
                       shares_out=[100e6])
    out = prices.add_market_cap(panel, FIRMS, daily, CAL)
    assert out.loc[0, "shares_out_adj"] == pytest.approx(400e6)
    assert out.loc[0, "market_cap"] == pytest.approx(130.0 * 400e6)


def test_market_cap_ignores_a_spin_off_between_filing_and_first_trade() -> None:
    """Filed Friday 2020-08-28; spin-off (Yahoo ratio 2.0849) ex Monday 08-31. Shares do not
    change, so market cap = the traded price on 08-31 * shares_out."""
    close = np.where(CAL < pd.Timestamp("2020-08-31"), 60.0 / 2.0849, 31.0)
    ratio = np.where(CAL == pd.Timestamp("2020-08-31"), 2.0849, 0.0)
    daily = with_factors(daily_frame("AAA", CAL, list(close), splits=list(ratio)))
    panel = panel_rows(cik=["1"], period_end=["2020-06-30"], fiscal_year=[2020],
                       availability_date=["2020-08-28"], first_filed=["2020-08-28"],
                       shares_out=[100e6])
    out = prices.add_market_cap(panel, FIRMS, daily, CAL)
    assert out.loc[0, "shares_out_adj"] == pytest.approx(100e6)
    assert out.loc[0, "market_cap"] == pytest.approx(31.0 * 100e6)


def test_market_cap_is_nan_without_shares_price_or_timely_trade() -> None:
    late_listing = CAL[CAL >= "2020-10-01"]
    daily = with_factors(daily_frame("AAA", CAL, [10.0] * len(CAL)),
                         daily_frame("CCC", late_listing, [10.0] * len(late_listing)))
    panel = panel_rows(
        cik=["1", "2", "3"], availability_date=["2020-08-14"] * 3, first_filed=["2020-08-14"] * 3,
        period_end=["2020-06-30"] * 3, fiscal_year=[2020] * 3, shares_out=[0.0, 100e6, 100e6],
    )
    out = prices.add_market_cap(panel, FIRMS, daily, CAL)
    assert out["market_cap"].isna().all()  # zero shares / no prices / first trade weeks later
    assert out.loc[2, "shares_out_adj"] == pytest.approx(100e6)  # factor before listing = first day's


# --- share-count check ---------------------------------------------------------------------


def test_share_count_errors_use_only_earlier_rows() -> None:
    """x1000 scale slips are rejected and the next normal year accepted again; a pre-spin-off
    shell's 100 shares are rejected; a lasting >20x jump (big merger) is rejected once, then
    accepted when the next filing confirms the new level; a count from >2 years ago is no
    reference."""
    cases = {
        "slip": [90e6, 89e9, 91e6],  # PKG FY2025-style scale slip in the middle year
        "shell": [100.0, 450e6, 455e6],  # GEHC-style shell before the spin-off
        "merger": [10e6, 300e6, 310e6],  # genuine level change
        "gaps": [90e6, np.nan, 88e6],
        "old_reference": [10e6, np.nan, np.nan, 225e6],  # BE: pre-IPO count, years of gaps
    }
    frames = [
        pd.DataFrame({"cik": cik, "period_end": pd.date_range("2020-12-31", periods=len(v), freq="YE"),
                      "shares": v})
        for cik, v in cases.items()
    ]
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=0)  # order must not matter
    errors = prices.share_count_errors(df["cik"], df["period_end"], df["shares"])
    flagged = df[errors].groupby("cik")["period_end"].apply(lambda d: d.dt.year.tolist()).to_dict()
    assert flagged == {"slip": [2021], "shell": [2020], "merger": [2021]}


def test_market_cap_is_nan_for_rejected_share_counts() -> None:
    daily = with_factors(daily_frame("AAA", CAL, [10.0] * len(CAL)))
    panel = panel_rows(
        cik=["1", "1"], period_end=["2019-06-30", "2020-06-30"], fiscal_year=[2019, 2020],
        availability_date=["2020-08-03", "2020-08-14"], first_filed=["2020-08-03", "2020-08-14"],
        shares_out=[100e6, 100e9],
    )
    out = prices.add_market_cap(panel, FIRMS, daily, CAL)
    assert out["market_cap"].tolist()[0] == pytest.approx(1e9)
    assert np.isnan(out.loc[1, "market_cap"]) and np.isnan(out.loc[1, "shares_out_adj"])
    assert out.loc[1, "shares_out_adj_reported"] == pytest.approx(100e9)


def test_check_shares_compares_latest_year_and_flags_big_gaps() -> None:
    panel = panel_rows(
        cik=["1", "1", "2", "3"], fiscal_year=[2023, 2024, 2024, 2024],
        period_end=["2023-12-31", "2024-12-31", "2024-12-31", "2024-12-31"],
        first_filed=["2024-02-01", "2025-02-01", "2025-02-01", "2025-02-01"],
        shares_out=[50.0, 100.0, 100.0, 100.0], shares_out_adj=[50.0, 100.0, np.nan, 100.0],
        shares_out_adj_reported=[50.0, 100.0, 100.0, 100.0],
    )
    current = pd.DataFrame({"ticker": ["AAA", "BBB"], "shares": [110.0, 200.0]})
    out = prices.check_shares(panel, FIRMS, current).set_index("ticker")
    assert out.loc["AAA", "fiscal_year"] == 2024 and not out.loc["AAA", "mismatch"]
    assert out.loc["BBB", "mismatch"] and out.loc["BBB", "ratio"] == pytest.approx(0.5)
    assert out.loc["BBB", "rejected"] and not out.loc["AAA", "rejected"]
    assert pd.isna(out.loc["CCC", "mismatch"])  # no Yahoo count -> unknown, not "ok"


def test_fetch_current_shares_caches_successes_only(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(ticker: str) -> float | None:
        calls.append(ticker)
        if ticker == "BAD":
            raise RuntimeError("404")
        return {"AAA": 10.0, "BBB": 20.0}.get(ticker)

    path = tmp_path / "shares.parquet"
    got = prices.fetch_current_shares(["AAA", "BBB", "BAD"], path=path, fetch=fetch, sleep=lambda s: None)
    assert dict(zip(got["ticker"], got["shares"])) == {"AAA": 10.0, "BBB": 20.0}
    prices.fetch_current_shares(["AAA", "BBB", "BAD"], path=path, fetch=fetch, sleep=lambda s: None)
    assert calls == ["AAA", "BBB", "BAD", "BAD"]  # only the failure is asked again
