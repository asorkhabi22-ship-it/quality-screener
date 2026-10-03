"""Monthly calendar-time quintile sorts on the quality signals.

Timing (CLAUDE.md rule 1). Formation date t = the last trading day of each month, from January
of ``START_YEAR + 2`` through the last full month. A scores row is usable at t only if its
``availability_date`` is on or before the trading day before t, so filing date + 1 trading day
<= t: a 10-K filed on the last trading day of March is first used at the end of April.
Portfolios are formed at the close of t and held over the following month (returns: stats.py).

At each t, per firm: the most recent fiscal period whose row is usable (an older period that
becomes available later never replaces a newer one). Rows whose period ended more than
``MAX_AGE_MONTHS`` months before t are dropped as stale: the firm stopped filing or its next
10-K is missing from the data. A firm also needs a month-end price at t (it must be tradable).

Signals, oriented so that a higher bucket is always higher quality (Q5 = best):
- ``f_group``     Piotroski (2000) F-score in groups: 0-3 = 1 (low), 4-6 = 2, 7-9 = 3 (high)
- ``f_score``     rank-based quintiles of F (tied scores share a bucket, so buckets are uneven)
- ``altman_z``    quintiles of Altman (1968) Z; high Z = good
- ``beneish_m``   quintiles of -M, Beneish (1999); low M = good
- ``accruals_cf`` quintiles of -accruals, Sloan (1996); low accruals = good
- ``composite``   quintiles of the mean cross-sectional percentile rank of the four oriented
                  signals (at least ``COMPOSITE_MIN_SIGNALS``), in the spirit of Asness,
                  Frazzini and Pedersen (2019), Quality minus junk, Review of Accounting
                  Studies 24, 34-112.
- ``accruals_bs`` quintiles of -balance-sheet accruals (Sloan's original measure), a
                  robustness check only: not part of the composite.

Piotroski's own extreme portfolios are F 0-1 and 8-9 among high book-to-market firms. With
about 360 large caps those groups are nearly empty, so the coarser 0-3 / 4-6 / 7-9 split is
used. F-scores rescaled from 7 or 8 signals are rounded half up before grouping.

Quintile = ceil(5 * percentile rank) within t, percentile rank = average rank / n.
Financials, utilities and firms with unknown SIC are left out unless ``include_fin_util``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quality_screener import config
from quality_screener.prices import as_days, month_end_trading_days, previous_trading_day

log = logging.getLogger(__name__)

HOLDINGS_PATH = config.PROCESSED_DIR / "holdings.parquet"
HOLDINGS_FIN_UTIL_PATH = config.PROCESSED_DIR / "holdings_incl_fin_util.parquet"

N_QUANTILES = 5
MAX_AGE_MONTHS = 15
COMPOSITE_MIN_SIGNALS = 3
F_LOW_MAX, F_HIGH_MIN = 3, 7

# signal name -> (scores column, orientation: +1 if high = good quality)
SIGNALS: dict[str, tuple[str, int]] = {
    "f_score": ("f_score", 1),
    "altman_z": ("z_score", 1),
    "beneish_m": ("m_score", -1),
    "accruals_cf": ("accruals_cf", -1),
}
F_GROUP, COMPOSITE = "f_group", "composite"
# sorted like the others but kept out of the composite (robustness checks)
EXTRA_SIGNALS: dict[str, tuple[str, int]] = {"accruals_bs": ("accruals_bs", -1)}
SIGNAL_ORDER = [F_GROUP, *SIGNALS, COMPOSITE, *EXTRA_SIGNALS]

ROW_COLUMNS = ["cik", "fiscal_year", "period_end", "availability_date"]
HOLDINGS_COLUMNS = [
    "t", "cik", "ticker", "signal", "bucket", "market_cap", "value", *ROW_COLUMNS[1:],
]


@dataclass(frozen=True)
class Holdings:
    holdings: pd.DataFrame  # HOLDINGS_COLUMNS, one row per (t, firm, signal)
    stale: pd.DataFrame  # (t, cik, period_end): firm-months dropped because the data was stale


# --- dates ---------------------------------------------------------------------------------


def formation_dates(
    calendar: pd.DatetimeIndex, start_year: int, asof: pd.Timestamp
) -> pd.DatetimeIndex:
    """Last trading day of each month from January ``start_year`` up to the last full month:
    months before both ``asof``'s month and the calendar's final month."""
    ends = month_end_trading_days(calendar)
    last_full = min(pd.Period(asof, "M"), pd.Period(calendar[-1], "M"))
    keep = (ends.index.year >= start_year) & (ends.index < last_full)
    return pd.DatetimeIndex(ends[keep].to_numpy(), name="t")


def months_between(earlier: pd.Series, later: pd.Series) -> pd.Series:
    """Whole calendar months from ``earlier`` to ``later`` (2019-12-31 -> 2021-03-31 = 15)."""
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


# --- which row is known at t ---------------------------------------------------------------


def select_sample(
    scores: pd.DataFrame, firms: pd.DataFrame, include_fin_util: bool
) -> pd.DataFrame:
    """Scores rows with the firm's ``ticker``; without ``include_fin_util`` only firms whose
    ``excluded_from_main`` is False (financials, utilities and unknown SIC are dropped)."""
    info = firms[["cik", "ticker", "excluded_from_main"]]
    out = scores.drop(columns=["ticker", "excluded_from_main"], errors="ignore").merge(
        info, on="cik", how="inner", validate="many_to_one"
    )
    if not include_fin_util:
        out = out[out["excluded_from_main"].eq(False).fillna(False)]
    return out.reset_index(drop=True)


def usable_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """Rows that bring a newer fiscal period than anything the firm already had available.
    A row for an older period that only becomes available later never replaces a newer one."""
    s = scores.sort_values(["cik", "availability_date", "period_end"], ignore_index=True)
    newest_before = s.groupby("cik")["period_end"].cummax().groupby(s["cik"]).shift()
    return s[newest_before.isna() | (s["period_end"] > newest_before)]


def latest_rows(
    scores: pd.DataFrame, dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """For each formation date t and firm, the most recent usable row whose
    ``availability_date`` <= the trading day before t (``cutoff``)."""
    grid = pd.DataFrame({"t": dates, "cutoff": previous_trading_day(dates, calendar)})
    grid = grid.merge(pd.DataFrame({"cik": scores["cik"].unique()}), how="cross")
    rows = usable_rows(scores).assign(_available=lambda d: as_days(d["availability_date"]))
    picked = pd.merge_asof(
        grid.sort_values("cutoff"), rows.sort_values(["_available", "period_end"]),
        left_on="cutoff", right_on="_available", by="cik", direction="backward",
    )
    picked = picked.dropna(subset=["_available"]).drop(columns="_available")
    return picked.sort_values(["t", "cik"], ignore_index=True)


def split_stale(
    snapshot: pd.DataFrame, max_age_months: int = MAX_AGE_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fresh, stale): stale where the period ended more than ``max_age_months`` before t."""
    stale = months_between(snapshot["period_end"], snapshot["t"]) > max_age_months
    return snapshot[~stale], snapshot[stale]


def tradable(snapshot: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Keep firms with a month-end price at t; market_cap = close_split_adj(t) *
    shares_out_adj, both in today's share-count basis (see ``prices``)."""
    price = monthly[["ticker", "t", "close_split_adj"]]
    out = snapshot.merge(price, on=["ticker", "t"], how="inner")
    shares = out["shares_out_adj"] if "shares_out_adj" in out else np.nan
    return out.assign(market_cap=out["close_split_adj"] * shares).drop(columns="close_split_adj")


# --- buckets -------------------------------------------------------------------------------


def percentile_rank(values: pd.Series, by: pd.Series) -> pd.Series:
    """Average rank / n within each group (ties share a rank); NaN values stay NaN."""
    return values.groupby(by).rank(method="average", pct=True)


def quantile_bucket(pct: pd.Series, n: int = N_QUANTILES) -> pd.Series:
    """1..n from a percentile rank in (0, 1]: ceil(n * pct)."""
    return pd.Series(np.ceil(pct * n), index=pct.index).clip(1, n).astype("Int8")


def f_group(f_score: pd.Series) -> pd.Series:
    """Piotroski groups: 1 = F 0-3, 2 = F 4-6, 3 = F 7-9, after rounding half up."""
    rounded = np.floor(f_score + 0.5)
    group = np.select([rounded <= F_LOW_MAX, rounded >= F_HIGH_MIN], [1, 3], 2)
    return pd.Series(group, index=f_score.index).astype("Int8").mask(f_score.isna())


def composite_score(pct_ranks: pd.DataFrame, min_signals: int = COMPOSITE_MIN_SIGNALS) -> pd.Series:
    """Mean of the available oriented percentile ranks; NaN with fewer than ``min_signals``."""
    return pct_ranks.mean(axis=1).where(pct_ranks.notna().sum(axis=1) >= min_signals)


def assign_buckets(universe: pd.DataFrame) -> pd.DataFrame:
    """Long frame: one row per (t, firm, signal) with a bucket; ``value`` is the raw signal
    (F for ``f_group``, the composite score for ``composite``)."""
    t = universe["t"]
    pct = pd.DataFrame(
        {name: percentile_rank(universe[col] * sign, t) for name, (col, sign) in SIGNALS.items()}
    )
    composite = composite_score(pct)
    buckets = {F_GROUP: (f_group(universe["f_score"]), universe["f_score"])}
    buckets |= {
        name: (quantile_bucket(pct[name]), universe[col]) for name, (col, _) in SIGNALS.items()
    }
    buckets[COMPOSITE] = (quantile_bucket(percentile_rank(composite, t)), composite)
    for name, (col, sign) in EXTRA_SIGNALS.items():
        if col in universe:
            ranks = percentile_rank(universe[col] * sign, t)
            buckets[name] = (quantile_bucket(ranks), universe[col])

    base = universe[["t", "cik", "ticker", "market_cap", *ROW_COLUMNS[1:]]]
    parts = [
        base.assign(signal=name, bucket=bucket, value=value)[bucket.notna()]
        for name, (bucket, value) in buckets.items()
    ]
    out = pd.concat(parts, ignore_index=True)
    return out.astype({"bucket": "int8"})[HOLDINGS_COLUMNS]


def build_holdings(
    scores: pd.DataFrame,
    firms: pd.DataFrame,
    monthly: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    start_year: int = config.START_YEAR + 2,
    asof: pd.Timestamp | None = None,
    include_fin_util: bool = False,
    max_age_months: int = MAX_AGE_MONTHS,
) -> Holdings:
    """Holdings for every formation date and signal (see the module docstring for the rules).

    ``monthly`` is ``prices.monthly_prices`` output (``ticker``, ``t``, ``close_split_adj``);
    ``asof`` defaults to today and bounds the last full month.
    """
    asof = pd.Timestamp.today() if asof is None else pd.Timestamp(asof)
    dates = formation_dates(calendar, start_year, asof)
    sample = select_sample(scores, firms, include_fin_util)
    fresh, stale = split_stale(latest_rows(sample, dates, calendar), max_age_months)
    holdings = assign_buckets(tradable(fresh, monthly))
    return Holdings(holdings, stale[["t", "cik", "ticker", "period_end"]].reset_index(drop=True))


# --- summaries -----------------------------------------------------------------------------


def bucket_counts(holdings: pd.DataFrame) -> pd.DataFrame:
    """Average number of firms per bucket per formation month (empty bucket = 0), by signal
    and year; buckets a signal does not have (4-5 for ``f_group``) are NaN."""
    by_signal = {}
    for signal, g in holdings.groupby("signal"):
        n = g.groupby(["t", "bucket"]).size().unstack("bucket", fill_value=0)
        by_signal[signal] = n.groupby(n.index.year.rename("year")).mean()
    return pd.concat(by_signal, names=["signal"]).reindex(SIGNAL_ORDER, level="signal")


def stale_counts(stale: pd.DataFrame) -> pd.DataFrame:
    """Per year: distinct firms dropped as stale at least once, and firm-months dropped."""
    year = stale["t"].dt.year.rename("year")
    return stale.groupby(year).agg(firms=("cik", "nunique"), firm_months=("cik", "size"))


# --- holding-period returns ----------------------------------------------------------------
#
# Returns run from the close of formation date t. A stock that stops trading is valued at its
# last price: the return up to its last trade, then 0 (the position sits at that price). Yahoo
# has no delisting returns, and Shumway (1997), The Delisting Bias in CRSP Data, Journal of
# Finance 52, 327-340, shows performance delistings average about -30%, so this is optimistic
# for failing firms. Each such case is logged.


def full_month_ends(
    calendar: pd.DatetimeIndex, asof: pd.Timestamp | None = None
) -> pd.DatetimeIndex:
    """Last trading day of every complete month in the calendar (see ``formation_dates``)."""
    asof = pd.Timestamp.today() if asof is None else pd.Timestamp(asof)
    return formation_dates(calendar, calendar[0].year, asof)


def months_ahead(month_ends: pd.DatetimeIndex, steps: int) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map each month end to the month end ``steps`` months later (absent near the end)."""
    return dict(zip(month_ends[:-steps], month_ends[steps:]))


def firm_next_month_returns(
    holdings: pd.DataFrame, monthly: pd.DataFrame, month_ends: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per held (t, ticker) whose next month is complete: ``ret`` = total return from t to the
    next month end (``month``). ``status`` is 'full', 'partial' (stopped trading during the
    month: return to the last trade) or 'none' (no trade in the month: 0)."""
    firms = holdings[["t", "ticker"]].drop_duplicates()
    firms = firms.assign(month=firms["t"].map(months_ahead(month_ends, 1)))
    firms = firms.dropna(subset=["month"])
    px = monthly[["ticker", "t", "date", "ret"]].rename(columns={"t": "month"})
    out = firms.merge(px, on=["ticker", "month"], how="left")
    out["status"] = np.select(
        [out["date"].isna(), out["date"] < out["month"]], ["none", "partial"], "full"
    )
    out["ret"] = out["ret"].mask(out["status"] == "none", 0.0)
    _log_lost_prices(out, "next-month")
    return out[["t", "ticker", "month", "ret", "status"]].reset_index(drop=True)


def firm_forward_returns(
    holdings: pd.DataFrame,
    monthly: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    horizons: Sequence[int] = (3, 6, 12),
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Per held (t, ticker) and horizon h: buy-and-hold total return from t to the month end h
    months later (``end``) minus the benchmark's over the same span (``ret``, an excess
    return; ``bh`` and ``bh_benchmark`` are the parts). A stock that stops trading is valued
    at its last price (``status`` 'lost'). Overlapping by construction: consecutive formation
    dates share h-1 months."""
    px = monthly[["ticker", "t", "adj_close"]].assign(t=lambda d: as_days(d["t"]))
    bench = px[px["ticker"] == benchmark].set_index("t")["adj_close"]
    firms = holdings[["t", "ticker"]].drop_duplicates().assign(t=lambda d: as_days(d["t"]))
    start = firms.merge(px, on=["ticker", "t"])
    later = px.rename(columns={"t": "last_t", "adj_close": "adj_end"}).sort_values("last_t")
    parts = []
    for h in horizons:
        f = start.assign(horizon=h, end=start["t"].map(months_ahead(month_ends, h)))
        f = f.dropna(subset=["end"]).assign(end=lambda d: as_days(d["end"])).sort_values("end")
        last = pd.merge_asof(
            f, later, left_on="end", right_on="last_t", by="ticker", direction="backward"
        )
        last["bh"] = last["adj_end"] / last["adj_close"] - 1
        bench_end = bench.reindex(last["end"]).to_numpy()
        last["bh_benchmark"] = bench_end / bench.reindex(last["t"]).to_numpy() - 1
        last["status"] = np.where(last["last_t"] < last["end"], "lost", "full")
        parts.append(last)
    out = pd.concat(parts, ignore_index=True)
    out["ret"] = out["bh"] - out["bh_benchmark"]
    _log_lost_prices(out.rename(columns={"end": "month"}), "buy-and-hold")
    cols = ["t", "ticker", "horizon", "end", "bh", "bh_benchmark", "ret", "status"]
    return out[cols].sort_values(["horizon", "t", "ticker"], ignore_index=True)


def _log_lost_prices(firm_returns: pd.DataFrame, label: str) -> None:
    lost = firm_returns[firm_returns["status"] != "full"]
    if lost.empty:
        return
    first = lost.drop_duplicates("ticker")
    detail = ", ".join(f"{r.ticker} {r.month:%Y-%m}" for r in first.itertuples())
    log.warning("%s returns: %d firm-periods lost price data, valued at the last price (no "
                "delisting return): %s", label, len(lost), detail)


def bucket_returns(
    holdings: pd.DataFrame, firm_returns: pd.DataFrame, keys: Sequence[str] = ("month",)
) -> pd.DataFrame:
    """Per signal, formation date t, ``keys`` and bucket: equal-weighted (``ew``) and
    value-weighted (``vw``, weights = market_cap at t) mean of the firms' ``ret``, with the
    number of firms in each. Firms without a market cap are left out of ``vw`` only."""
    merged = holdings[["signal", "t", "ticker", "bucket", "market_cap"]].merge(
        firm_returns[["t", "ticker", *keys, "ret"]], on=["t", "ticker"], how="inner"
    ).dropna(subset=["ret"])
    cap = merged["market_cap"].where(merged["market_cap"] > 0)
    merged = merged.assign(cap=cap, cap_ret=cap * merged["ret"])
    grouped = merged.groupby(["signal", "t", *keys, "bucket"], sort=True)
    out = grouped.agg(ew=("ret", "mean"), n=("ret", "size"), cap=("cap", "sum"),
                      cap_ret=("cap_ret", "sum"), n_vw=("cap", "count"))
    out["vw"] = out["cap_ret"] / out["cap"].where(out["n_vw"] > 0)
    return out.drop(columns=["cap", "cap_ret"]).reset_index()


def long_short(port: pd.DataFrame, keys: Sequence[str] = ("month",)) -> pd.DataFrame:
    """Top minus bottom bucket per signal, t and ``keys``: Q5 - Q1 for quintiles, high (3) -
    low (1) for ``f_group``; ``ew`` and ``vw``. Missing where either leg is empty."""
    top = port.groupby("signal")["bucket"].transform("max")
    index = ["signal", "t", *keys]
    high = port[port["bucket"] == top].set_index(index)[["ew", "vw"]]
    low = port[port["bucket"] == 1].set_index(index)[["ew", "vw"]]
    return high.sub(low).dropna(how="all").reset_index()

