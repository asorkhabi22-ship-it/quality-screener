"""Helpers for a single-firm restatement case study: when the market could react to a filing,
event-window abnormal returns, original vs restated numbers, and the scores recomputed on
restated data.

Event study convention (MacKinlay, A. C. (1997), Event Studies in Economics and Finance,
Journal of Economic Literature 35(1), 13-39), market-adjusted model:
- day 0 = first trading day the market could react: the filing day if EDGAR accepted it before
  the 16:00 New York close, else the next trading day (CLAUDE.md rule 1 in spirit: nothing is
  known before it is public);
- abnormal return = r_firm - r_SPY each day; the window's buy-and-hold abnormal return (BHAR)
  = prod(1 + r_firm) - prod(1 + r_SPY) (Barber, B. M. and Lyon, J. D. (1997), Journal of
  Financial Economics 43, 341-372).
"""

from __future__ import annotations

import pandas as pd

from quality_screener import panel, prices, scores

NEW_YORK = "America/New_York"
MARKET_CLOSE_HOUR = 16
CASE_FIELDS = (
    "revenue", "cogs", "gross_profit", "sga", "net_income", "cfo", "receivables",
    "current_assets", "total_assets", "current_liabilities", "total_liabilities", "equity",
)


def event_day(accepted: str | pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Day 0 for a filing accepted at ``accepted`` (EDGAR's ``acceptanceDateTime``, read as
    UTC as its 'Z' suffix says): the same trading day if accepted before the New York close,
    else the next trading day."""
    ts = pd.Timestamp(accepted)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    local = ts.tz_convert(NEW_YORK)
    day = pd.Timestamp(local.date())
    if day in calendar and local.hour < MARKET_CLOSE_HOUR:
        return day
    return calendar[calendar.searchsorted(day, side="right")]


def daily_returns(daily: pd.DataFrame, ticker: str) -> pd.Series:
    """Daily total returns (from adjusted close) of one ticker, indexed by date."""
    px = daily.loc[daily["ticker"] == ticker].set_index("date")["adj_close"].sort_index()
    return px.pct_change().dropna()


def event_window(
    daily: pd.DataFrame, ticker: str, day0: pd.Timestamp, before: int = 5, after: int = 10,
    benchmark: str = prices.BENCHMARK,
) -> pd.DataFrame:
    """Trading days ``-before`` .. ``+after`` around ``day0``: the firm's and the benchmark's
    returns, the daily abnormal return and the cumulative BHAR from day ``-before``."""
    firm, bench = daily_returns(daily, ticker), daily_returns(daily, benchmark)
    joined = pd.DataFrame({"ret": firm, "bench_ret": bench}).dropna()
    pos = joined.index.get_loc(day0)
    window = joined.iloc[pos - before : pos + after + 1].copy()
    window.insert(0, "rel_day", range(-before, -before + len(window)))
    window["abnormal"] = window["ret"] - window["bench_ret"]
    window["cum_bhar"] = (1 + window["ret"]).cumprod() - (1 + window["bench_ret"]).cumprod()
    return window.rename_axis("date").reset_index()


def bhar(window: pd.DataFrame, start: int, end: int) -> float:
    """Buy-and-hold abnormal return over relative days ``start`` .. ``end`` (inclusive)."""
    w = window[window["rel_day"].between(start, end)]
    return float((1 + w["ret"]).prod() - (1 + w["bench_ret"]).prod())


def compare_versions(
    versions: pd.DataFrame, cik: str, restated_accn: str,
    periods: list[pd.Timestamp], fields: tuple[str, ...] = CASE_FIELDS,
) -> pd.DataFrame:
    """As-first-reported vs the restating filing's value for each field and period, with
    the change and % change (NaN where either side is missing)."""
    v = versions[(versions["cik"] == cik) & versions["field"].isin(fields)
                 & versions["period_end"].isin(periods)]
    first = panel.first_reported(v).set_index(["field", "period_end"])
    restated = v[v["accn"] == restated_accn].set_index(["field", "period_end"])
    out = pd.DataFrame({
        "original": first["val"], "original_accn": first["accn"],
        "restated": restated["val"],
    })
    out["change"] = out["restated"] - out["original"]
    out["pct_change"] = out["change"] / out["original"].abs().where(out["original"] != 0)
    out = out.reset_index()
    rank = out["field"].map({f: i for i, f in enumerate(fields)})
    out = out.assign(_rank=rank).sort_values(["period_end", "_rank"])
    return out.drop(columns="_rank").reset_index(drop=True)


def first_reported_long(versions: pd.DataFrame, cik: str) -> pd.DataFrame:
    """The firm's point-in-time long panel, exactly as the main panel builds it: earliest
    filing per value, values reported >90 days after the row's 10-K left out."""
    kept, _ = panel.drop_late_values(panel.first_reported(versions[versions["cik"] == cik]))
    return kept


def restated_long(versions: pd.DataFrame, cik: str, restated_accn: str) -> pd.DataFrame:
    """The as-first-reported long panel with every value the restating filing reports
    replaced by the restated one (its accn, form and filing date come along)."""
    base = first_reported_long(versions, cik)
    keys = ["cik", "field", "period_end"]
    new = versions[(versions["cik"] == cik) & (versions["accn"] == restated_accn)]
    untouched = base.merge(new[keys], on=keys, how="left", indicator=True)
    untouched = untouched[untouched["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([untouched, new[base.columns]], ignore_index=True)


def scores_for(
    long: pd.DataFrame, market_cap: pd.Series | None = None, firms: pd.DataFrame | None = None,
    versions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Scores (``scores.compute_all``) for one firm's long panel. ``market_cap`` (indexed by
    period_end) is attached so both views use the same price; on a single firm the Beneish
    indices are not winsorised (fewer than ``WINSOR_MIN_OBS`` observations). With
    ``versions``, prior-year values come from each row's own filing (``panel.add_lags``);
    without, from the prior row as first reported (the rule before this case study)."""
    wide = panel.add_lags(panel.to_wide(long), versions)
    if market_cap is not None:
        wide["market_cap"] = wide["period_end"].map(market_cap)
    return scores.compute_all(wide, firms)
