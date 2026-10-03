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

Piotroski's own extreme portfolios are F 0-1 and 8-9 among high book-to-market firms. With
about 360 large caps those groups are nearly empty, so the coarser 0-3 / 4-6 / 7-9 split is
used. F-scores rescaled from 7 or 8 signals are rounded half up before grouping.

Quintile = ceil(5 * percentile rank) within t, percentile rank = average rank / n.
Financials, utilities and firms with unknown SIC are left out unless ``include_fin_util``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quality_screener import config
from quality_screener.prices import as_days, month_end_trading_days, previous_trading_day

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
SIGNAL_ORDER = [F_GROUP, *SIGNALS, COMPOSITE]

ROW_COLUMNS = ["cik", "fiscal_year", "period_end", "availability_date"]
HOLDINGS_COLUMNS = ["t", "cik", "ticker", "signal", "bucket", "market_cap", "value", *ROW_COLUMNS[1:]]


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


def select_sample(scores: pd.DataFrame, firms: pd.DataFrame, include_fin_util: bool) -> pd.DataFrame:
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
    buckets |= {name: (quantile_bucket(pct[name]), universe[col]) for name, (col, _) in SIGNALS.items()}
    buckets[COMPOSITE] = (quantile_bucket(percentile_rank(composite, t)), composite)

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
