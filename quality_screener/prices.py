"""Daily prices from Yahoo Finance (yfinance), month-end returns, market capitalisation and the
trading calendar used to find the first trading day after a filing.

What Yahoo's columns mean (checked on AAPL around its 2020-08-31 4:1 split):
- ``Close`` is NOT the price that traded. AAPL's 2020-08-28 close shows as ~$125, but the
  price that day was ~$499: history is divided by every later "Stock Splits" ratio.
- ``Adj Close`` is also adjusted for dividends, so it is the total-return series.
- ``Stock Splits`` holds true splits (exact ratios: 4.0, 0.333333) but ALSO spin-offs, written
  as the price ratio across the ex-date (ABT 2013-01-02 2.084 for AbbVie, EBAY 2015-07-20
  2.376 for PayPal). A spin-off changes the price, not the number of shares.

So two factors, each the product of ratios with ex-date after day d (up to the download):
- ``price_factor`` over every "Stock Splits" event:  close_raw(d) = close(d) * price_factor(d)
- ``split_factor`` over true share-count changes only (``share_count_ratio``): exact simple
  ratios, plus the hand-checked mixed events in ``SHARE_COUNT_EVENTS``.

``close_split_adj`` = close_raw / split_factor is the traded price in today's share-count
basis, and ``shares_out_adj = shares_out * split_factor(filed)`` turns the count reported in a
10-K into the same basis. Then at any later date d

    market_cap(d) = close_split_adj(d) * shares_out_adj
                  = close_raw(d) * shares_out * (true splits in (filed, d]),

which is close_raw(d) * shares_out whenever no true split falls between the filing and d, and
stays right when one does (CRSP's cumulative adjustment factors do the same job).

Caching: one parquet per ticker in ``data/raw/prices``. Yahoo rescales history whenever a new
dividend or split arrives, so the cached download, not a fresh one, is the reproducible
input. Delete a file (or pass ``refresh=True``) to refetch it.

Survivorship bias: tickers are today's S&P 500 (see ``universe``); Yahoo has no prices for
firms that were delisted, so delisting returns are missing too.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from quality_screener import config

log = logging.getLogger(__name__)

PRICES_DIR = config.RAW_DIR / "prices"
YAHOO_SHARES_PATH = config.RAW_DIR / "yahoo_shares.parquet"
MONTHLY_PATH = config.PROCESSED_DIR / "prices_monthly.parquet"
CALENDAR_PATH = config.PROCESSED_DIR / "trading_calendar.parquet"
COVERAGE_REPORT = config.REPORTS_DIR / "price_coverage.csv"
SHARES_CHECK_REPORT = config.REPORTS_DIR / "shares_check.csv"

BENCHMARK = "SPY"
START_DATE = f"{config.START_YEAR}-01-01"
BATCH_SIZE = 50
MAX_GAP_TRADING_DAYS = 10  # longer runs of missing trading days are reported as gaps
ENDS_EARLY_TRADING_DAYS = 5  # history ending this many trading days before the calendar does
MCAP_TOLERANCE = pd.Timedelta(days=7)  # max wait for the first trade after availability
SHARES_TOLERANCE = 0.20
MIN_SHARES = 1e6  # pre-spin-off shells report 100 shares; S&P 500 firms have millions
MAX_SHARE_JUMP = 20.0  # XBRL scale slips are x1000; real mergers and IPOs stay below this
SHARE_REFERENCE_AGE = pd.Timedelta(days=800)  # a count older than ~2 years is no reference

# Yahoo records true splits exactly; anything else in "Stock Splits" is a price ratio.
SIMPLE_SPLIT_RATIOS = np.array(sorted(
    {float(n) for n in range(2, 101)} | {1 / n for n in range(2, 101)}
    | {3 / 2, 5 / 4, 4 / 3, 5 / 2, 2 / 3, 4 / 5, 3 / 4, 2 / 5}
))
SPLIT_RATIO_TOLERANCE = 1e-4
# Events whose Yahoo ratio mixes a share-count change with a spin-off, or changed the count by
# an unusual ratio: (ticker, ex-date) -> change in shares per old share.
SHARE_COUNT_EVENTS: dict[tuple[str, str], float] = {
    ("MSI", "2011-01-04"): 1 / 7,  # 1-for-7 reverse split with the Motorola Mobility spin-off
    ("LDOS", "2013-09-30"): 1 / 4,  # 1-for-4 reverse split with the SAIC spin-off
    ("GOOGL", "2014-04-03"): 2.0,  # one Class C share distributed per Class A/B share
    ("WTW", "2016-01-05"): 1 / 2.649,  # Willis share consolidation at the Towers Watson merger
    ("JCI", "2016-09-06"): 0.955,  # Tyco share consolidation at the Johnson Controls merger
    ("HLT", "2017-01-04"): 1 / 3,  # 1-for-3 reverse split with the Park / HGV spin-offs
    ("DD", "2019-06-03"): 1 / 3,  # 1-for-3 reverse split with the Corteva spin-off
    ("HON", "2026-06-29"): 1 / 2,  # Yahoo share count 634M -> 317M with the Aerospace spin-off
}

YAHOO_FIELDS = {
    "Close": "close",
    "Adj Close": "adj_close",
    "Dividends": "dividends",
    "Stock Splits": "splits",
    "Volume": "volume",
}
DAILY_COLUMNS = ["ticker", "date", *YAHOO_FIELDS.values()]

Downloader = Callable[[list[str], str], pd.DataFrame]
SharesFetcher = Callable[[str], float | None]


# --- download and cache -------------------------------------------------------------------


def yahoo_download(tickers: list[str], start: str) -> pd.DataFrame:
    """Daily bars for ``tickers`` from ``start`` to today, columns (ticker, field)."""
    import yfinance as yf

    return yf.download(
        tickers, start=start, auto_adjust=False, actions=True, group_by="ticker",
        progress=False, threads=True, multi_level_index=True,
    )


def as_days(values: pd.Series | pd.Index) -> pd.Series | pd.DatetimeIndex:
    """Dates as tz-naive midnight datetime64[ns], so joins never fail on a dtype mismatch."""
    if isinstance(values, pd.Series):
        return pd.to_datetime(values).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    return pd.DatetimeIndex(values).tz_localize(None).normalize().astype("datetime64[ns]")


def tidy_daily(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One ticker's yfinance bars -> ``DAILY_COLUMNS``, rows without a close dropped. Missing
    action columns stay NaN (no dividend or split recorded)."""
    out = bars.reindex(columns=list(YAHOO_FIELDS)).rename(columns=YAHOO_FIELDS)
    out = out[out["close"].notna()].astype("float64")
    out.insert(0, "date", as_days(out.index))
    out.insert(0, "ticker", ticker)
    return out.reset_index(drop=True)[DAILY_COLUMNS]


def split_download(raw: pd.DataFrame | None, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    """yf.download output (ticker, field columns) -> tidy frame per ticker that has data."""
    if raw is None or raw.empty:
        return {}
    present = set(raw.columns.get_level_values(0))
    frames = {t: tidy_daily(raw[t], t) for t in tickers if t in present}
    return {t: f for t, f in frames.items() if not f.empty}


def cache_path(ticker: str, cache_dir: Path = PRICES_DIR) -> Path:
    return cache_dir / f"{ticker}.parquet"


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _download_batches(
    tickers: list[str], start: str, download: Downloader, batch_size: int,
    cache_dir: Path, pause: float, sleep: Callable[[float], None],
) -> set[str]:
    """Download in batches, cache every ticker that returned data; returns those tickers."""
    got: set[str] = set()
    for i in range(0, len(tickers), batch_size):
        if i:
            sleep(pause)
        batch = tickers[i : i + batch_size]
        for ticker, frame in split_download(download(batch, start), batch).items():
            _write_atomic(frame, cache_path(ticker, cache_dir))
            got.add(ticker)
    return got


def download_prices(
    tickers: Sequence[str],
    *,
    start: str = START_DATE,
    cache_dir: Path = PRICES_DIR,
    download: Downloader = yahoo_download,
    batch_size: int = BATCH_SIZE,
    refresh: bool = False,
    retries: int = 1,
    pause: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Make sure every ticker's daily history is cached; returns the tickers with no data.

    Only tickers without a cache file (all of them with ``refresh``) are requested, in batches
    of ``batch_size``. Tickers that come back empty are retried ``retries`` more times (Yahoo
    rate limits show up as empty results) and are never cached, so a later run tries again.
    """
    wanted = list(dict.fromkeys(tickers))
    todo = [t for t in wanted if refresh or not cache_path(t, cache_dir).exists()]
    for attempt in range(retries + 1):
        if not todo:
            break
        if attempt:
            sleep(pause * 10)
        got = _download_batches(todo, start, download, batch_size, cache_dir, pause, sleep)
        todo = [t for t in todo if t not in got]
    if todo:
        log.warning("%d tickers returned no price data: %s", len(todo), todo)
    return todo


def load_daily(tickers: Sequence[str], cache_dir: Path = PRICES_DIR) -> pd.DataFrame:
    """Cached daily prices for ``tickers`` (those with a cache file), with ``split_factor`` and
    ``close_raw`` added; sorted by ticker and date."""
    paths = [cache_path(t, cache_dir) for t in dict.fromkeys(tickers)]
    frames = [pd.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError(f"no cached prices in {cache_dir}; run scripts/fetch_prices.py")
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = as_days(daily["date"])
    return add_split_factor(daily.sort_values(["ticker", "date"], ignore_index=True))


# --- split adjustment and the trading calendar ---------------------------------------------


def is_simple_split(ratio: pd.Series) -> pd.Series:
    """True where the ratio is (within ``SPLIT_RATIO_TOLERANCE``) a simple split ratio."""
    values = ratio.to_numpy(dtype="float64")
    distance = np.abs(np.log(values[:, None] / SIMPLE_SPLIT_RATIOS))
    nearest = SIMPLE_SPLIT_RATIOS[distance.argmin(axis=1)]
    return pd.Series(np.abs(values / nearest - 1) < SPLIT_RATIO_TOLERANCE, index=ratio.index)


def share_count_ratio(daily: pd.DataFrame) -> pd.Series:
    """Change in shares per old share on each day: the Yahoo ratio for simple splits, the
    ``SHARE_COUNT_EVENTS`` value for listed events, 1.0 otherwise (spin-offs, no event)."""
    event = daily["splits"] > 0
    ratio = pd.Series(1.0, index=daily.index)
    simple = event & is_simple_split(daily["splits"].where(event, 1.0))
    ratio[simple] = daily.loc[simple, "splits"]
    events = daily.loc[event, ["ticker", "date"]]
    for idx, ticker, day in zip(events.index, events["ticker"], events["date"]):
        ratio[idx] = SHARE_COUNT_EVENTS.get((ticker, f"{day:%Y-%m-%d}"), ratio[idx])
    return ratio


def _factor_after(ratio: pd.Series, tickers: pd.Series) -> pd.Series:
    """Product of ``ratio`` over each ticker's later rows (rows sorted by ticker and date). A
    day's own ratio is excluded: prices on the ex-date are already post-event."""
    from_here_on = ratio.iloc[::-1].groupby(tickers.iloc[::-1]).cumprod().iloc[::-1]
    return from_here_on / ratio


def add_split_factor(daily: pd.DataFrame) -> pd.DataFrame:
    """Add ``price_factor``, ``split_factor``, ``close_raw`` and ``close_split_adj`` (see the
    module docstring). Expects rows sorted by ticker and date. A "Stock Splits" value of 0 or
    NaN means no event (Yahoo's convention)."""
    out = daily.copy()
    price_ratio = out["splits"].where(out["splits"] > 0, 1.0).astype("float64")
    out["price_factor"] = _factor_after(price_ratio, out["ticker"])
    out["split_factor"] = _factor_after(share_count_ratio(out), out["ticker"])
    out["close_raw"] = out["close"] * out["price_factor"]
    out["close_split_adj"] = out["close_raw"] / out["split_factor"]
    return out


def trading_calendar(daily: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DatetimeIndex:
    """Trading days = the dates the benchmark (SPY, NYSE Arca) traded."""
    dates = daily.loc[daily["ticker"] == benchmark, "date"]
    if dates.empty:
        raise ValueError(f"no {benchmark} prices to build the trading calendar from")
    return pd.DatetimeIndex(dates.drop_duplicates().sort_values(), name="date")


def save_calendar(calendar: pd.DatetimeIndex, path: Path = CALENDAR_PATH) -> None:
    calendar.to_frame(index=False).to_parquet(path, index=False)


def load_calendar(path: Path = CALENDAR_PATH) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(as_days(pd.read_parquet(path)["date"]), name="date")


def next_trading_day(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """First trading day strictly after each date (NaT past the end of the calendar)."""
    days = as_days(dates)
    pos = calendar.searchsorted(days.to_numpy(), side="right")
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    inside = pos < len(calendar)
    out[inside] = calendar[pos[inside]]
    return out


def previous_trading_day(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day strictly before each date."""
    pos = calendar.searchsorted(as_days(dates).to_numpy(), side="left") - 1
    if (pos < 0).any():
        raise ValueError("date before the start of the trading calendar")
    return calendar[pos]


# --- monthly series ------------------------------------------------------------------------


def month_end_trading_days(calendar: pd.DatetimeIndex) -> pd.Series:
    """Last trading day of each calendar month, indexed by monthly Period."""
    days = pd.Series(calendar, index=calendar.to_period("M"))
    return days.groupby(level=0).max()


def monthly_prices(daily: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (ticker, month) from the last trade in the month.

    ``t`` is the calendar's last trading day of the month (``date`` is the ticker's own last
    trade). ``ret`` is the month's total return, adj_close(t) / adj_close(t-1) - 1, and is NaN
    when the previous month has no price. ``close_raw`` is the unadjusted month-end price and
    ``close_split_adj`` the same price in today's share-count basis (for market cap).
    """
    month = daily["date"].dt.to_period("M")
    last = daily.assign(month=month).groupby(["ticker", "month"], sort=True).tail(1)
    last = last.sort_values(["ticker", "month"], ignore_index=True)
    ends = month_end_trading_days(calendar)
    last["t"] = last["month"].map(ends)
    prev_month = last.groupby("ticker")["month"].shift()
    prev_adj = last.groupby("ticker")["adj_close"].shift()
    consecutive = prev_month == (last["month"] - 1)
    last["ret"] = (last["adj_close"] / prev_adj - 1).where(consecutive)
    cols = [
        "ticker", "t", "date", "close_raw", "close_split_adj", "adj_close", "split_factor", "ret",
    ]
    return last.dropna(subset=["t"])[cols].reset_index(drop=True)


# --- data quality --------------------------------------------------------------------------


def price_coverage(daily: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Per ticker: first/last date, trading days present and missing between them, the longest
    run of missing trading days, and an ``issue`` label ('gap', 'ends early') or NA."""
    rows = []
    for ticker, g in daily.groupby("ticker", sort=True):
        pos = calendar.searchsorted(g["date"].to_numpy())
        span = calendar[(calendar >= g["date"].iloc[0]) & (calendar <= g["date"].iloc[-1])]
        runs = np.diff(pos) - 1
        rows.append(
            {
                "ticker": ticker,
                "first_date": g["date"].iloc[0],
                "last_date": g["date"].iloc[-1],
                "n_days": len(g),
                "missing_days": len(span) - g["date"].isin(span).sum(),
                "max_gap_days": int(runs.max()) if len(runs) else 0,
                "trading_days_to_end": len(calendar) - 1 - int(pos[-1]),
            }
        )
    out = pd.DataFrame(rows)
    issue = pd.Series(pd.NA, index=out.index, dtype="string")
    issue[out["trading_days_to_end"] > ENDS_EARLY_TRADING_DAYS] = "ends early"
    issue[out["max_gap_days"] > MAX_GAP_TRADING_DAYS] = "gap"
    out["issue"] = issue
    return out


def log_price_issues(coverage: pd.DataFrame, no_data: Sequence[str]) -> None:
    """Warn about tickers with no data, long gaps, or histories that stop early."""
    if no_data:
        log.warning("no price data: %s", list(no_data))
    for label, group in coverage.dropna(subset=["issue"]).groupby("issue"):
        detail = ", ".join(
            f"{r.ticker} (max gap {r.max_gap_days}d, last {r.last_date:%Y-%m-%d})"
            for r in group.itertuples()
        )
        log.warning("%d tickers with %s: %s", len(group), label, detail)


# --- market capitalisation -----------------------------------------------------------------


def _asof_join(
    left: pd.DataFrame, right: pd.DataFrame, left_on: str, direction: str,
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """merge_asof of ``left[left_on]`` onto ``right['date']`` by ticker; keeps left's index."""
    keyed = left.rename_axis("_row").reset_index().dropna(subset=[left_on, "ticker"])
    keyed[left_on] = as_days(keyed[left_on])
    joined = pd.merge_asof(
        keyed.sort_values(left_on), right.sort_values("date"), left_on=left_on,
        right_on="date", by="ticker", direction=direction, tolerance=tolerance,
    )
    return joined.set_index("_row").reindex(left.index)


def split_factor_at(dates: pd.Series, tickers: pd.Series, daily: pd.DataFrame) -> pd.Series:
    """``split_factor`` per row on the last trading day on or before ``date``; for a date before
    the ticker's first price, its first day's factor (no splits before listing)."""
    left = pd.DataFrame({"ticker": tickers, "on": dates})
    right = daily[["ticker", "date", "split_factor"]]
    before = _asof_join(left, right, "on", "backward")["split_factor"]
    after = _asof_join(left, right, "on", "forward")["split_factor"]
    return before.fillna(after)


def share_count_errors(
    cik: pd.Series, period_end: pd.Series, shares: pd.Series,
    min_shares: float = MIN_SHARES, max_jump: float = MAX_SHARE_JUMP,
    max_age: pd.Timedelta = SHARE_REFERENCE_AGE,
) -> pd.Series:
    """True for share counts treated as data errors, each decided from the firm's earlier
    rows only (point-in-time): below ``min_shares``, or more than ``max_jump`` times away from
    the last accepted count of the past ``max_age``. A rejected count that the next row
    confirms (within ``max_jump``) starts a new level, e.g. a big merger, from that next row
    on. ``shares`` must be in one split basis (``shares_out_adj``); missing counts are never
    errors."""
    frame = pd.DataFrame({"cik": cik, "period_end": period_end, "shares": shares})
    errors = pd.Series(False, index=shares.index)
    for _, firm in frame.dropna(subset=["shares"]).sort_values("period_end").groupby("cik"):
        accepted: tuple[float, pd.Timestamp] | None = None
        rejected: tuple[float, pd.Timestamp] | None = None
        for idx, x, end in zip(firm.index, firm["shares"], firm["period_end"]):
            recent = [r for r in (accepted, rejected) if r is not None and end - r[1] <= max_age]
            fresh_start = accepted is None or end - accepted[1] > max_age
            near = any(max(x / ref, ref / x) <= max_jump for ref, _ in recent)
            if x >= min_shares and (fresh_start or near):
                accepted, rejected = (x, end), None
            else:
                errors[idx] = True
                rejected = (x, end) if x >= min_shares else rejected
    return errors


def add_market_cap(
    panel: pd.DataFrame, firms: pd.DataFrame, daily: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """Add ``shares_out_adj``, ``market_cap_date`` and ``market_cap`` to the wide PIT panel.

    market_cap = close_split_adj on the first trading day after ``availability_date`` (a later
    trade within ``MCAP_TOLERANCE`` if the stock did not trade that day) * shares_out_adj, with
    shares_out_adj = shares_out * split_factor(first_filed); see the module docstring. With no
    true split since the filing this is that day's unadjusted close * shares_out. Non-positive
    share counts, counts rejected by ``share_count_errors`` (kept in
    ``shares_out_adj_reported``) and firms without prices give NaN.
    """
    out = panel.copy()
    tickers = out["cik"].map(firms.set_index("cik")["ticker"])
    shares = out["shares_out"].where(out["shares_out"] > 0)
    reported = shares * split_factor_at(out["first_filed"], tickers, daily)
    errors = share_count_errors(out["cik"], out["period_end"], reported)
    _log_share_count_errors(out[errors], tickers[errors])
    out["shares_out_adj_reported"] = reported
    out["shares_out_adj"] = reported.mask(errors)

    first_day = next_trading_day(out["availability_date"], calendar)
    left = pd.DataFrame({"ticker": tickers, "first_day": first_day})
    traded = _asof_join(
        left, daily[["ticker", "date", "close_split_adj"]], "first_day", "forward",
        tolerance=MCAP_TOLERANCE,
    )
    out["market_cap_date"] = traded["date"]
    out["market_cap"] = traded["close_split_adj"] * out["shares_out_adj"]
    return out


def _log_share_count_errors(rows: pd.DataFrame, tickers: pd.Series) -> None:
    if rows.empty:
        return
    detail = ", ".join(
        f"{t} FY{fy} ({n:.3g})"
        for t, fy, n in zip(tickers, rows["fiscal_year"], rows["shares_out"])
    )
    log.warning(
        "%d share counts rejected as data errors (market cap left NaN): %s", len(rows), detail
    )


# --- share-count sanity check ---------------------------------------------------------------


def yahoo_shares(ticker: str) -> float | None:
    """Yahoo's current share count: implied shares outstanding (all share classes, in this
    ticker's units, e.g. Class B equivalents for BRK-B), else the listed class's count."""
    import yfinance as yf

    info = yf.Ticker(ticker).info
    shares = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
    return float(shares) if shares else None


def fetch_current_shares(
    tickers: Sequence[str],
    *,
    path: Path = YAHOO_SHARES_PATH,
    fetch: SharesFetcher = yahoo_shares,
    refresh: bool = False,
    pause: float = 0.2,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Current share counts (``ticker``, ``shares``, ``fetched``), cached at ``path``; only
    tickers missing from the cache are requested. Failures are logged and not cached."""
    cached = pd.read_parquet(path) if path.exists() and not refresh else pd.DataFrame(
        {"ticker": pd.Series(dtype="str"), "shares": pd.Series(dtype="float64"),
         "fetched": pd.Series(dtype="datetime64[ns]")}
    )
    todo = [t for t in dict.fromkeys(tickers) if t not in set(cached["ticker"])]
    rows, failed = [], []
    for i, ticker in enumerate(todo):
        if i:
            sleep(pause)
        try:
            shares = fetch(ticker)
        except Exception as exc:  # yfinance raises many unrelated types on bad tickers
            log.debug("share count for %s failed: %s", ticker, exc)
            shares = None
        if shares:
            today = pd.Timestamp.today().normalize()
            rows.append({"ticker": ticker, "shares": shares, "fetched": today})
        else:
            failed.append(ticker)
    if failed:
        log.warning("no Yahoo share count for %d tickers: %s", len(failed), failed)
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        _write_atomic(cached, path)
    return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)


def check_shares(
    panel: pd.DataFrame, firms: pd.DataFrame, current: pd.DataFrame,
    tolerance: float = SHARES_TOLERANCE,
) -> pd.DataFrame:
    """Compare each firm's latest-year reported count in today's split basis
    (``shares_out_adj_reported`` from ``add_market_cap``) with Yahoo's current count;
    ``mismatch`` where they differ by more than ``tolerance``. Buybacks since the filing move
    the count a few percent a year, so a >20% gap means a units error, a share-class problem,
    a split the panel did not see, or a merger since the filing. ``rejected`` marks counts
    ``share_count_errors`` already set aside."""
    latest = panel.sort_values("period_end").groupby("cik").tail(1)
    latest = latest.assign(
        rejected=latest["shares_out_adj"].isna() & latest["shares_out_adj_reported"].notna()
    )
    cols = ["cik", "fiscal_year", "first_filed", "shares_out", "shares_out_adj_reported", "rejected"]
    yahoo = current[["ticker", "shares"]].rename(columns={"shares": "yahoo_shares"})
    out = latest[cols].merge(firms[["cik", "ticker", "name"]], on="cik", how="left")
    out = out.merge(yahoo, on="ticker", how="left")
    out["ratio"] = out["shares_out_adj_reported"] / out["yahoo_shares"]
    gap = (out["ratio"] - 1).abs() > tolerance
    out["mismatch"] = gap.astype("boolean").mask(out["ratio"].isna())
    bad = out[out["mismatch"].fillna(False)]
    if not bad.empty:
        detail = ", ".join(f"{r.ticker} x{r.ratio:.2f}" for r in bad.itertuples())
        log.warning("%d firms: latest shares_out differs from Yahoo by >%.0f%%: %s",
                    len(bad), tolerance * 100, detail)
    return out.sort_values("ratio", key=lambda r: (r - 1).abs(), ascending=False, ignore_index=True)
