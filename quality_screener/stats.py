"""Return statistics for the quintile portfolios, in plain numpy.

Formulas and sources:
- Newey-West standard errors: Newey, W. K. and West, K. D. (1987). A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  Econometrica 55(3), 703-708. Long-run covariance of the scores g_t = x_t e_t with Bartlett
  weights, S = G_0 + sum_{j=1..L} (1 - j/(L+1)) (G_j + G_j'), G_j = (1/T) sum_t g_t g_{t-j}';
  Var(b) = (X'X)^-1 (T S) (X'X)^-1. No small-sample correction. For a mean (X = 1) this is
  Var(mean) = S / T.
- Lags: 6 for monthly returns; h - 1 for overlapping h-month returns (Hansen and Hodrick
  (1980), Journal of Political Economy 88(5), 829-853: overlap induces MA(h-1) errors).
- Sharpe ratio: Sharpe, W. F. (1966), Mutual Fund Performance, Journal of Business 39,
  119-138; annualised as mean / sd * sqrt(12) of monthly excess returns. A long-short
  portfolio is self-financing, so its own return is already an excess return.
- CAPM alpha: intercept of r_LS = a + b (r_SPY - r_f) + e (Jensen, M. C. (1968), The
  Performance of Mutual Funds, Journal of Finance 23(2), 389-416).
- Factor alpha: intercept on the Fama and French (2015) five factors (A five-factor asset
  pricing model, Journal of Financial Economics 116, 1-22) plus momentum (Carhart, M. M.
  (1997), On Persistence in Mutual Fund Performance, Journal of Finance 52(1), 57-82).
- Monotonicity: Spearman (1904) rank correlation of bucket number with bucket mean return.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
NW_LAGS_MONTHLY = 6
WEIGHTINGS = ("ew", "vw")


# --- core estimators -----------------------------------------------------------------------


@dataclass(frozen=True)
class Regression:
    params: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    nobs: int


def newey_west_lrv(scores: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-weighted long-run covariance S of the rows of ``scores`` (T x k, mean zero)."""
    g = np.asarray(scores, dtype=float).reshape(len(scores), -1)
    T = len(g)
    s = g.T @ g / T
    for j in range(1, min(lags, T - 1) + 1):
        gamma = g[j:].T @ g[:-j] / T
        s += (1 - j / (lags + 1)) * (gamma + gamma.T)
    return s


def ols_newey_west(y: np.ndarray, X: np.ndarray, lags: int) -> Regression:
    """OLS of y on X (include a constant column yourself) with Newey-West standard errors.
    Rows with a missing value are dropped."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[keep], X[keep]
    T, k = X.shape
    if T <= k:
        nan = np.full(k, np.nan)
        return Regression(nan, nan, nan, T)
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    cov = xtx_inv @ (T * newey_west_lrv(X * resid[:, None], lags)) @ xtx_inv
    se = np.sqrt(np.diag(cov))
    return Regression(beta, se, beta / se, T)


def mean_newey_west(x: Sequence[float] | pd.Series, lags: int) -> Regression:
    """Sample mean with its Newey-West standard error (regression on a constant)."""
    x = np.asarray(x, dtype=float)
    return ols_newey_west(x, np.ones((len(x), 1)), lags)


def annualised_mean(monthly: pd.Series) -> float:
    return float(monthly.mean() * MONTHS_PER_YEAR)


def annualised_vol(monthly: pd.Series) -> float:
    return float(monthly.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def sharpe_ratio(monthly_excess: pd.Series) -> float:
    """Annualised Sharpe ratio of monthly excess returns: mean / sd * sqrt(12)."""
    sd = monthly_excess.std(ddof=1)
    return float(monthly_excess.mean() / sd * np.sqrt(MONTHS_PER_YEAR)) if sd > 0 else np.nan


def hit_rate(returns: pd.Series) -> float:
    """Share of periods with a positive return."""
    r = returns.dropna()
    return float((r > 0).mean()) if len(r) else np.nan


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative wealth prod(1 + r), starting from 1 (so a
    loss in the first month counts); a negative number."""
    wealth = (1 + returns.dropna()).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    return float((wealth.to_numpy() / peak - 1).min()) if len(wealth) else np.nan


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation: Pearson correlation of average ranks (ties share a rank)."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 2:
        return np.nan
    ranks = frame.rank(method="average")
    return float(np.corrcoef(ranks["x"], ranks["y"])[0, 1])


# --- portfolio summaries -------------------------------------------------------------------


def _by_month(series: pd.Series) -> pd.Series:
    """Index a series by calendar month (Period), so returns and factors line up."""
    return series.set_axis(pd.PeriodIndex(series.index, freq="M"))


def bucket_summary(port: pd.DataFrame, rf: pd.Series | None = None) -> pd.DataFrame:
    """Per signal, weighting and bucket: months, mean monthly return, annualised mean and
    volatility, and Sharpe ratio of returns in excess of ``rf`` (monthly, Period index; raw
    returns when None)."""
    rows = []
    for (signal, bucket), g in port.groupby(["signal", "bucket"]):
        for w in WEIGHTINGS:
            r = _by_month(g.set_index("month")[w]).dropna()
            excess = r - rf.reindex(r.index) if rf is not None else r
            rows.append({
                "signal": signal, "weighting": w, "bucket": bucket, "months": len(r),
                "mean": r.mean(), "ann_mean": annualised_mean(r), "ann_vol": annualised_vol(r),
                "sharpe": sharpe_ratio(excess.dropna()),
            })
    return pd.DataFrame(rows)


def monotonicity(buckets: pd.DataFrame) -> pd.DataFrame:
    """Per signal and weighting: Spearman correlation of bucket number with mean return, and
    whether the means rise strictly from the lowest to the highest bucket."""
    rows = []
    for (signal, w), g in buckets.sort_values("bucket").groupby(["signal", "weighting"]):
        rows.append({"signal": signal, "weighting": w,
                     "spearman": spearman(g["bucket"], g["mean"]),
                     "increasing": bool((g["mean"].diff().dropna() > 0).all())})
    return pd.DataFrame(rows)


def long_short_summary(
    ls: pd.DataFrame, market_excess: pd.Series, lags: int = NW_LAGS_MONTHLY
) -> pd.DataFrame:
    """Per signal and weighting of the monthly long-short: mean with Newey-West t-stat,
    annualised mean, volatility and Sharpe, hit rate, max drawdown, and CAPM alpha / beta on
    ``market_excess`` (SPY minus the risk-free rate, monthly Period index)."""
    rows = []
    for signal, g in ls.groupby("signal"):
        for w in WEIGHTINGS:
            r = _by_month(g.set_index("month")[w]).dropna()
            mean = mean_newey_west(r, lags)
            mkt = market_excess.reindex(r.index)
            capm = ols_newey_west(r.to_numpy(), np.column_stack([np.ones(len(r)), mkt]), lags)
            rows.append({
                "signal": signal, "weighting": w, "months": len(r),
                "mean": mean.params[0], "nw_t": mean.tstat[0], "nw_se": mean.se[0],
                "ann_mean": annualised_mean(r), "ann_vol": annualised_vol(r),
                "sharpe": sharpe_ratio(r), "hit_rate": hit_rate(r), "max_drawdown": max_drawdown(r),
                "capm_alpha": capm.params[0], "capm_alpha_t": capm.tstat[0],
                "capm_beta": capm.params[1],
            })
    return pd.DataFrame(rows)


def factor_alpha(
    ls: pd.DataFrame, factors: pd.DataFrame, lags: int = NW_LAGS_MONTHLY
) -> pd.DataFrame:
    """Per signal and weighting: intercept of the long-short on ``factors`` (monthly Period
    index, decimal returns) with its Newey-West t-stat; months without factors are dropped."""
    rows = []
    for signal, g in ls.groupby("signal"):
        for w in WEIGHTINGS:
            r = _by_month(g.set_index("month")[w]).dropna()
            f = factors.reindex(r.index)
            X = np.column_stack([np.ones(len(r)), f.to_numpy()])
            fit = ols_newey_west(r.to_numpy(), X, lags)
            rows.append({"signal": signal, "weighting": w, "months": fit.nobs,
                         "alpha": fit.params[0], "alpha_t": fit.tstat[0]})
    return pd.DataFrame(rows)


def forward_summary(port: pd.DataFrame, ls: pd.DataFrame) -> pd.DataFrame:
    """Per signal, weighting and horizon h of the overlapping buy-and-hold excess returns:
    bottom and top bucket means and the long-short mean with a Newey-West t-stat (h - 1 lags)."""
    top = port.groupby("signal")["bucket"].transform("max")
    rows = []
    for (signal, h), g in ls.groupby(["signal", "horizon"]):
        legs = port[(port["signal"] == signal) & (port["horizon"] == h)]
        legs_top = legs[legs["bucket"] == top[legs.index]]
        for w in WEIGHTINGS:
            r = g.sort_values("t")[w].dropna()
            fit = mean_newey_west(r, lags=int(h) - 1)
            rows.append({
                "signal": signal, "weighting": w, "horizon": int(h), "periods": fit.nobs,
                "low": legs.loc[legs["bucket"] == 1, w].mean(), "high": legs_top[w].mean(),
                "long_short": fit.params[0], "nw_t": fit.tstat[0],
            })
    return pd.DataFrame(rows)

