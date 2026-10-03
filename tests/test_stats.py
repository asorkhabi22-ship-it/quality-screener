"""Tests for stats: Newey-West against a hand-worked example, OLS, Sharpe, drawdown, hit rate,
Spearman and the portfolio summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quality_screener import stats

# Hand-worked Newey-West example, T = 10:
#   x = [2, -1, 3, 0, 1, -2, 4, 1, -1, 3], mean = 10 / 10 = 1
#   u = x - 1 = [1, -2, 2, -1, 0, -3, 3, 0, -2, 2]
#   gamma_0 = sum u_t^2 / T              = 36 / 10 = 3.6
#   gamma_1 = sum u_t u_{t-1} / T        = (-2 - 4 - 2 + 0 + 0 - 9 + 0 + 0 - 4) / 10 = -2.1
#   gamma_2 = sum u_t u_{t-2} / T        = (2 + 2 + 0 + 3 + 0 + 0 - 6 + 0) / 10 = 0.1
#   lag 2, Bartlett weights 1 - j/3: w_1 = 2/3, w_2 = 1/3
#   S = 3.6 + 2 (2/3 * -2.1 + 1/3 * 0.1) = 3.6 - 2.8 + 0.0667 = 0.866667
#   Var(mean) = S / T = 0.0866667, se = 0.294392, t = 1 / se = 3.396831
#   lag 0: S = 3.6, se = sqrt(0.36) = 0.6, t = 1.666667
HAND_X = [2, -1, 3, 0, 1, -2, 4, 1, -1, 3]


def test_newey_west_mean_matches_hand_calculation() -> None:
    fit = stats.mean_newey_west(HAND_X, lags=2)
    assert fit.params[0] == pytest.approx(1.0)
    assert fit.se[0] == pytest.approx(np.sqrt(0.866667 / 10), rel=1e-5)
    assert fit.tstat[0] == pytest.approx(3.396831, rel=1e-5)
    assert fit.nobs == 10


def test_newey_west_lag_zero_is_plain_variance_over_t() -> None:
    fit = stats.mean_newey_west(HAND_X, lags=0)
    assert fit.se[0] == pytest.approx(0.6)
    assert fit.tstat[0] == pytest.approx(1 / 0.6)


def test_long_run_variance_hand_gammas() -> None:
    u = np.array(HAND_X, dtype=float) - 1
    assert stats.newey_west_lrv(u, 1)[0, 0] == pytest.approx(3.6 + 2 * 0.5 * -2.1)
    assert stats.newey_west_lrv(u, 2)[0, 0] == pytest.approx(0.866667, rel=1e-5)


def test_ols_matches_polyfit_and_white_errors_at_lag_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = 0.5 + 2.0 * x + rng.normal(scale=0.3, size=50) * (1 + np.abs(x))
    X = np.column_stack([np.ones(50), x])
    fit = stats.ols_newey_west(y, X, lags=0)
    slope, intercept = np.polyfit(x, y, 1)
    assert fit.params == pytest.approx([intercept, slope])
    e = y - X @ fit.params
    bread = np.linalg.inv(X.T @ X)
    white = bread @ (X.T * e**2) @ X @ bread  # HC0
    assert fit.se == pytest.approx(np.sqrt(np.diag(white)))


def test_ols_drops_missing_rows_and_handles_too_few() -> None:
    y = np.array([1.0, 2.0, np.nan, 4.0])
    fit = stats.ols_newey_west(y, np.ones((4, 1)), lags=1)
    assert fit.nobs == 3 and fit.params[0] == pytest.approx(7 / 3)
    assert np.isnan(stats.mean_newey_west([1.0], lags=0).se[0])


def test_sharpe_vol_and_annualisation() -> None:
    r = pd.Series([0.01, 0.03, -0.01, 0.01])  # mean 0.01, sd (ddof 1) = sqrt(0.0008/3)
    sd = np.sqrt(0.0008 / 3)
    assert stats.annualised_mean(r) == pytest.approx(0.12)
    assert stats.annualised_vol(r) == pytest.approx(sd * np.sqrt(12))
    assert stats.sharpe_ratio(r) == pytest.approx(0.01 / sd * np.sqrt(12))
    assert np.isnan(stats.sharpe_ratio(pd.Series([0.01, 0.01])))


def test_max_drawdown_and_hit_rate() -> None:
    # wealth 1.1, 0.88, 0.924, 1.0164: peak 1.1 -> trough 0.88 = -20%
    assert stats.max_drawdown(pd.Series([0.1, -0.2, 0.05, 0.1])) == pytest.approx(-0.2)
    # a loss from the start counts against the initial wealth of 1: 0.9 * 0.9 = 0.81
    assert stats.max_drawdown(pd.Series([-0.1, -0.1])) == pytest.approx(-0.19)
    assert stats.hit_rate(pd.Series([0.1, -0.2, 0.0, 0.3, np.nan])) == pytest.approx(0.5)


def test_spearman() -> None:
    assert stats.spearman([1, 2, 3, 4, 5], [0.1, 0.2, 0.25, 0.9, 1.0]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    # y ranks with a tie: [1, 2.5, 2.5, 4] vs x ranks [1, 2, 3, 4] -> r = 0.948683
    assert stats.spearman([1, 2, 3, 4], [1.0, 2.0, 2.0, 5.0]) == pytest.approx(0.948683, rel=1e-5)


# --- summaries -------------------------------------------------------------------------------

MONTHS = pd.date_range("2020-01-31", periods=24, freq="ME")


def port_frame(means: dict[int, float], noise: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for bucket, mu in means.items():
        r = mu + rng.normal(scale=noise, size=len(MONTHS))
        rows += [{"signal": "sig", "t": m, "month": m, "bucket": bucket, "ew": v, "vw": v}
                 for m, v in zip(MONTHS, r)]
    return pd.DataFrame(rows)


def test_bucket_summary_and_monotonicity() -> None:
    port = port_frame({1: 0.00, 2: 0.005, 3: 0.01}, noise=0.0)
    rf = pd.Series(0.001, index=pd.PeriodIndex(MONTHS, freq="M"))
    b = stats.bucket_summary(port, rf)
    top = b[(b["bucket"] == 3) & (b["weighting"] == "ew")].iloc[0]
    assert top["mean"] == pytest.approx(0.01) and top["months"] == 24
    assert top["ann_mean"] == pytest.approx(0.12)
    mono = stats.monotonicity(b)
    assert mono["spearman"].tolist() == [1.0, 1.0] and mono["increasing"].all()


def test_long_short_summary_capm() -> None:
    """LS = 0.002 + 0.5 * market exactly: alpha 0.002, beta 0.5."""
    mkt = pd.Series(np.linspace(-0.05, 0.05, len(MONTHS)), index=pd.PeriodIndex(MONTHS, freq="M"))
    ls_ret = 0.002 + 0.5 * mkt.to_numpy()
    ls = pd.DataFrame({"signal": "sig", "t": MONTHS, "month": MONTHS, "ew": ls_ret, "vw": ls_ret})
    out = stats.long_short_summary(ls, mkt).set_index("weighting")
    assert out.loc["ew", "capm_alpha"] == pytest.approx(0.002)
    assert out.loc["ew", "capm_beta"] == pytest.approx(0.5)
    assert out.loc["ew", "mean"] == pytest.approx(ls_ret.mean())
    assert out.loc["ew", "months"] == 24


def test_factor_alpha_drops_months_without_factors() -> None:
    idx = pd.PeriodIndex(MONTHS, freq="M")
    f = pd.DataFrame({"F1": np.linspace(-1, 1, 24) / 100}, index=idx).iloc[:20]
    ls_ret = 0.003 + 2 * np.r_[f["F1"].to_numpy(), np.zeros(4)]
    ls = pd.DataFrame({"signal": "sig", "t": MONTHS, "month": MONTHS, "ew": ls_ret, "vw": ls_ret})
    out = stats.factor_alpha(ls, f).set_index("weighting")
    assert out.loc["ew", "months"] == 20
    assert out.loc["ew", "alpha"] == pytest.approx(0.003)


def test_forward_summary_uses_horizon_minus_one_lags() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(0.01, 0.02, size=24)
    ls = pd.DataFrame({"signal": "sig", "t": MONTHS, "horizon": 3, "ew": r, "vw": r})
    port = pd.DataFrame({"signal": "sig", "t": np.r_[MONTHS, MONTHS], "horizon": 3,
                         "bucket": [1] * 24 + [5] * 24, "ew": np.r_[-r / 2, r / 2], "vw": 0.0})
    out = stats.forward_summary(port, ls).set_index("weighting")
    assert out.loc["ew", "nw_t"] == pytest.approx(stats.mean_newey_west(r, lags=2).tstat[0])
    assert out.loc["ew", "high"] == pytest.approx(r.mean() / 2)
    assert out.loc["ew", "low"] == pytest.approx(-r.mean() / 2)
