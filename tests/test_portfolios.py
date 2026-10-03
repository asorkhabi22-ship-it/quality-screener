"""Tests for portfolios: point-in-time timing (the look-ahead tests), stale rows, the sample
filter, tradability, signal orientation, F groups and the composite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quality_screener import portfolios

CAL = pd.bdate_range("2019-01-01", "2022-01-31").astype("datetime64[ns]")  # trading days
T = pd.Timestamp


def score_row(cik: str, period_end: str, available: str, f: float = 6.0, z: float = 3.0,
              m: float = -2.5, acc: float = -0.05, shares: float = 100.0) -> dict[str, object]:
    return {
        "cik": cik, "period_end": T(period_end), "fiscal_year": T(period_end).year,
        "availability_date": T(available), "f_score": f, "z_score": z, "m_score": m,
        "accruals_cf": acc, "accruals_bs": acc, "shares_out_adj": shares,
    }


def scores_frame(*rows: dict[str, object]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    return out.astype({"period_end": "datetime64[us]", "availability_date": "datetime64[us]"})


def firms_frame(ciks: list[str], excluded: list[bool | None] | None = None) -> pd.DataFrame:
    flags = excluded if excluded is not None else [False] * len(ciks)
    return pd.DataFrame({"cik": ciks, "ticker": [f"T{c}" for c in ciks],
                         "excluded_from_main": pd.array(flags, dtype="boolean")})


def monthly_frame(ciks: list[str], close: float = 10.0) -> pd.DataFrame:
    ends = portfolios.month_end_trading_days(CAL)
    return pd.DataFrame([{"ticker": f"T{c}", "t": t, "close_split_adj": close} for c in ciks for t in ends])


def build(scores: pd.DataFrame, firms: pd.DataFrame, monthly: pd.DataFrame | None = None,
          **kwargs: object) -> portfolios.Holdings:
    monthly = monthly if monthly is not None else monthly_frame(firms["cik"].tolist())
    return portfolios.build_holdings(scores, firms, monthly, CAL, start_year=2019,
                                     asof=T("2022-01-15"), **kwargs)


def months_held(h: pd.DataFrame, cik: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(h.loc[h["cik"] == cik, "t"].unique()))


# --- look-ahead -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filed", "first_t"),
    [
        ("2020-03-15", "2020-03-31"),  # filed mid-month: usable at that month end
        ("2020-03-31", "2020-04-30"),  # filed on the last trading day: waits a month
    ],
)
def test_10k_first_appears_after_filing_never_at_period_end(filed: str, first_t: str) -> None:
    built = build(scores_frame(score_row("1", "2019-12-31", filed)), firms_frame(["1"]))
    held = months_held(built.holdings, "1")
    assert held[0] == T(first_t)
    assert T("2019-12-31") not in held  # period end is not when the market knew
    first = built.holdings[built.holdings["t"] == T(first_t)]
    assert set(first["signal"]) == set(portfolios.SIGNAL_ORDER)


def test_cutoff_is_the_trading_day_before_t() -> None:
    """t = Monday 2020-08-31: a filing on Friday 08-28 counts, one dated Saturday 08-29 waits."""
    rows = scores_frame(score_row("1", "2020-05-31", "2020-08-28"), score_row("2", "2020-05-31", "2020-08-29"))
    h = build(rows, firms_frame(["1", "2"])).holdings
    assert months_held(h, "1")[0] == T("2020-08-31")
    assert months_held(h, "2")[0] == T("2020-09-30")


def test_newest_period_is_used_and_never_replaced_by_an_older_one() -> None:
    rows = scores_frame(
        score_row("1", "2019-12-31", "2020-03-02", f=8.0),
        score_row("1", "2018-12-31", "2020-05-01", f=1.0),  # older period arriving later
        score_row("1", "2020-12-31", "2021-03-01", f=2.0),
    )
    h = build(rows, firms_frame(["1"])).holdings
    f = h[h["signal"] == "f_group"].set_index("t")["value"]
    assert f[T("2020-06-30")] == 8.0
    assert f[T("2021-02-26")] == 8.0 and f[T("2021-03-31")] == 2.0


def test_rows_older_than_15_months_are_dropped_as_stale() -> None:
    built = build(scores_frame(score_row("1", "2019-12-31", "2020-03-02")), firms_frame(["1"]))
    held = months_held(built.holdings, "1")
    assert held[-1] == T("2021-03-31")  # 15 months after period end: still used
    assert built.stale["t"].min() == T("2021-04-30")
    counts = portfolios.stale_counts(built.stale)
    assert counts.loc[2021].tolist() == [1, 9]  # one firm, April-December 2021


# --- sample and tradability -----------------------------------------------------------------


def test_financials_utilities_and_unknown_sic_excluded_unless_flag() -> None:
    rows = scores_frame(*(score_row(c, "2019-12-31", "2020-03-02") for c in ["1", "2", "3"]))
    firms = firms_frame(["1", "2", "3"], excluded=[False, True, None])
    assert set(build(rows, firms).holdings["cik"]) == {"1"}
    assert set(build(rows, firms, include_fin_util=True).holdings["cik"]) == {"1", "2", "3"}


def test_firm_needs_a_month_end_price_and_market_cap_is_at_t() -> None:
    rows = scores_frame(score_row("1", "2019-12-31", "2020-03-02", shares=50.0),
                        score_row("2", "2019-12-31", "2020-03-02"))
    monthly = monthly_frame(["1"], close=12.0)  # firm 2 has no prices
    h = build(rows, firms_frame(["1", "2"]), monthly).holdings
    assert set(h["cik"]) == {"1"}
    assert (h["market_cap"] == 600.0).all()


# --- buckets -------------------------------------------------------------------------------


def test_orientation_puts_high_quality_in_bucket_5_and_quintiles_balance() -> None:
    """Ten firms; firm i has F = Z = i and M = accruals = -i, so firm 9 is the best on every
    signal (highest F and Z, LOWEST M and accruals) and firm 0 the worst."""
    rows = scores_frame(*(
        score_row(str(i), "2019-12-31", "2020-03-02", f=float(i), z=float(i), m=-float(i), acc=-i / 100)
        for i in range(10)
    ))
    h = build(rows, firms_frame([str(i) for i in range(10)])).holdings
    at = h[h["t"] == T("2020-03-31")].pivot(index="cik", columns="signal", values="bucket")
    quintiles = ["f_score", "altman_z", "beneish_m", "accruals_cf", "composite", "accruals_bs"]
    assert at.loc["9", quintiles].tolist() == [5] * 6
    assert at.loc["0", quintiles].tolist() == [1] * 6
    for signal in quintiles:
        assert at[signal].value_counts().tolist() == [2] * 5
    assert at["f_group"].value_counts().sort_index().tolist() == [4, 3, 3]  # 0-3, 4-6, 7-9


def test_f_group_rounds_rescaled_scores_half_up() -> None:
    f = pd.Series([0.0, 3.0, 3.375, 3.857, 4.5, 6.43, 6.75, 7.0, 9.0, np.nan])
    groups = portfolios.f_group(f)
    assert groups[:9].tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert pd.isna(groups.iloc[9])


def test_ties_share_a_bucket() -> None:
    pct = portfolios.percentile_rank(pd.Series([1.0, 1.0, 1.0, 1.0, 2.0]), pd.Series(["t"] * 5))
    assert portfolios.quantile_bucket(pct).tolist() == [3, 3, 3, 3, 5]  # avg rank 2.5 / 5 = 0.5


def test_composite_is_mean_percentile_rank_and_needs_three_signals() -> None:
    pct = pd.DataFrame({"a": [0.2, 0.4], "b": [0.6, 0.8], "c": [1.0, np.nan], "d": [np.nan] * 2})
    comp = portfolios.composite_score(pct)
    assert comp[0] == pytest.approx(0.6)
    assert np.isnan(comp[1])


def test_formation_dates_are_month_end_trading_days_through_last_full_month() -> None:
    dates = portfolios.formation_dates(CAL, 2020, T("2021-06-15"))
    assert dates[0] == T("2020-01-31") and dates[-1] == T("2021-05-31")
    assert T("2020-05-29") in dates  # May 31 2020 was a Sunday
    short = CAL[CAL <= "2021-03-10"]  # prices end mid-March: March is not a full month
    assert portfolios.formation_dates(short, 2020, T("2021-06-15"))[-1] == T("2021-02-26")


def test_bucket_counts_average_per_month() -> None:
    h = pd.DataFrame({"signal": "f_group", "t": [T("2020-01-31")] * 3 + [T("2020-02-28")],
                      "bucket": [1, 1, 3, 1]})
    counts = portfolios.bucket_counts(h)
    assert counts.loc[("f_group", 2020), 1] == 1.5
    assert counts.loc[("f_group", 2020), 3] == 0.5  # empty in February counts as 0


# --- holding-period returns -----------------------------------------------------------------

ENDS = portfolios.full_month_ends(CAL, T("2022-01-15"))
MAR, APR, MAY, JUN, JUL = (T(d) for d in ["2020-03-31", "2020-04-30", "2020-05-29",
                                          "2020-06-30", "2020-07-31"])


def held(tickers: list[str], t: pd.Timestamp = MAR, caps: list[float] | None = None,
         buckets: list[int] | None = None, signal: str = "composite") -> pd.DataFrame:
    n = len(tickers)
    return pd.DataFrame({
        "t": t, "cik": tickers, "ticker": tickers, "signal": signal,
        "bucket": buckets or [5] * n, "market_cap": caps or [1.0] * n,
    })


def monthly_rows(*rows: tuple[str, pd.Timestamp, pd.Timestamp, float, float]) -> pd.DataFrame:
    """(ticker, t, date of last trade, adj_close, ret)."""
    return pd.DataFrame(rows, columns=["ticker", "t", "date", "adj_close", "ret"])


def test_next_month_return_when_a_stock_stops_trading(caplog: pytest.LogCaptureFixture) -> None:
    monthly = monthly_rows(
        ("A", MAR, MAR, 100.0, 0.0), ("A", APR, APR, 110.0, 0.10),
        ("B", MAR, MAR, 50.0, 0.0), ("B", APR, T("2020-04-15"), 40.0, -0.20),  # stops mid-April
        ("C", MAR, MAR, 20.0, 0.0),  # never trades in April
    )
    with caplog.at_level("WARNING"):
        out = portfolios.firm_next_month_returns(held(["A", "B", "C"]), monthly, ENDS).set_index("ticker")
    assert out["month"].unique().tolist() == [APR]
    assert out.loc["A", ["ret", "status"]].tolist() == [0.10, "full"]
    assert out.loc["B", ["ret", "status"]].tolist() == [-0.20, "partial"]  # to its last trade
    assert out.loc["C", ["ret", "status"]].tolist() == [0.0, "none"]  # held at its last price
    assert "lost price data" in caplog.text and "B 2020-04" in caplog.text


def test_last_full_month_has_no_next_month_return() -> None:
    monthly = monthly_rows(("A", ENDS[-1], ENDS[-1], 100.0, 0.0))
    assert portfolios.firm_next_month_returns(held(["A"], t=ENDS[-1]), monthly, ENDS).empty


def test_bucket_returns_equal_and_value_weighted_by_hand() -> None:
    holdings = held(["A", "B", "C"], caps=[100.0, 300.0, np.nan])
    firm_ret = pd.DataFrame({"t": MAR, "ticker": ["A", "B", "C"], "month": APR, "ret": [0.1, 0.0, -0.1]})
    out = portfolios.bucket_returns(holdings, firm_ret).iloc[0]
    assert out["ew"] == pytest.approx(0.0)  # (0.1 + 0 - 0.1) / 3
    assert out["vw"] == pytest.approx(0.025)  # (0.1 * 100 + 0 * 300) / 400; C has no cap
    assert (out["n"], out["n_vw"]) == (3, 2)


def test_long_short_is_top_minus_bottom_bucket() -> None:
    port = pd.DataFrame({
        "signal": ["composite"] * 3 + ["f_group"] * 3, "t": MAR, "month": APR,
        "bucket": [1, 3, 5, 1, 2, 3], "ew": [0.01, 0.02, 0.04, 0.03, 0.0, 0.01],
        "vw": [0.02, 0.0, 0.01, 0.0, 0.0, 0.05],
    })
    ls = portfolios.long_short(port).set_index("signal")
    assert ls.loc["composite", ["ew", "vw"]].tolist() == pytest.approx([0.03, -0.01])
    assert ls.loc["f_group", ["ew", "vw"]].tolist() == pytest.approx([-0.02, 0.05])  # 3 - 1


def test_forward_returns_in_excess_of_benchmark_and_lost_prices() -> None:
    monthly = monthly_rows(
        ("SPY", MAR, MAR, 100.0, 0.0), ("SPY", APR, APR, 101.0, 0.0),
        ("SPY", MAY, MAY, 102.0, 0.0), ("SPY", JUN, JUN, 104.0, 0.0),
        ("A", MAR, MAR, 50.0, 0.0), ("A", APR, APR, 55.0, 0.0),
        ("A", MAY, MAY, 60.0, 0.0), ("A", JUN, JUN, 65.0, 0.0),
        ("B", MAR, MAR, 10.0, 0.0), ("B", APR, APR, 8.0, 0.0),  # no prices after April
    )
    out = portfolios.firm_forward_returns(held(["A", "B"]), monthly, ENDS, horizons=(3,))
    out = out.set_index("ticker")
    assert out.loc["A", "end"] == JUN
    assert out.loc["A", "bh"] == pytest.approx(0.30)  # 65 / 50 - 1
    assert out.loc["A", "ret"] == pytest.approx(0.30 - 0.04)  # SPY 104 / 100 - 1
    assert out.loc["B", "bh"] == pytest.approx(-0.20) and out.loc["B", "status"] == "lost"
