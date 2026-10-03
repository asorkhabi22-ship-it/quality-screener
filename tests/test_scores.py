"""Scores against a hand-calculated example.

One firm, FY2022 (t-1) and FY2023 (t), round numbers. Total assets at FY2021 end (t-2) = 800,
because Piotroski's changes use each year's beginning-of-year assets.

                         t-1 (2022)    t (2023)
revenue                     1000         1200
gross_profit                 400          540
sga                          150          168
dep_amort                     50           60
ebit                         200          300
net_income                   100          150
income_cont_ops              100          160
cfo                          120          140
equity_issued                  0            0
total_assets                1000         1200      (t-2: 800)
current_assets               400          500
current_liabilities          200          250
total_liabilities            600          700
receivables                  100          150
ppe_net                      300          400
cash                          50           80
long_term_debt               300          300
retained_earnings            250          350
equity                       400          500
shares_out                   100          101
debt_current                  20           30
taxes_payable                 10           15

PIOTROSKI (t)
  ROA        = 150 / 1000                         = 0.15       > 0      -> 1
  CFO/TA     = 140 / 1000                         = 0.14       > 0      -> 1
  dROA       = 0.15 - 100/800 = 0.15 - 0.125      = 0.025      > 0      -> 1
  accrual    = 0.14 > 0.15 ?                                    no      -> 0
  dLEVER     = 300/((1200+1000)/2) - 300/((1000+800)/2)
             = 0.272727 - 0.333333                = -0.060606  < 0      -> 1
  dCR        = 500/250 - 400/200 = 2.0 - 2.0     = 0          > 0 ?    -> 0
  EQ_OFFER   = equity_issued 0 <= 0                                     -> 1
  dMARGIN    = 540/1200 - 400/1000 = 0.45 - 0.40 = 0.05       > 0      -> 1
  dTURN      = 1200/1000 - 1000/800 = 1.2 - 1.25 = -0.05      > 0 ?    -> 0
  F = 6 of 9 components

ALTMAN (TA = 1200, TL = 700)
  X1 = (500-250)/1200 = 0.208333   X2 = 350/1200 = 0.291667   X3 = 300/1200 = 0.25
  X4'' = 500/700 = 0.714286        X5 = 1200/1200 = 1.0
  Z'' = 6.56*0.208333 + 3.26*0.291667 + 6.72*0.25 + 1.05*0.714286
      = 1.366667 + 0.950833 + 1.68 + 0.75 = 4.7475                    -> safe (> 2.60)
  with market_cap 1400: X4 = 1400/700 = 2.0
  Z   = 1.2*0.208333 + 1.4*0.291667 + 3.3*0.25 + 0.6*2.0 + 1.0*1.0
      = 0.25 + 0.408333 + 0.825 + 1.2 + 1.0 = 3.683333                 -> safe (> 2.99)

BENEISH
  DSRI = (150/1200)/(100/1000) = 0.125/0.10              = 1.25
  GMI  = 0.40/0.45                                       = 0.888889
  AQI  = (1 - 900/1200)/(1 - 700/1000) = 0.25/0.30       = 0.833333
  SGI  = 1200/1000                                       = 1.2
  DEPI = (50/350)/(60/460) = 23000/21000                 = 1.095238
  SGAI = (168/1200)/(150/1000) = 0.14/0.15               = 0.933333
  TATA = (160 - 140)/1200                                = 0.016667
  LVGI = ((250+300)/1200)/((200+300)/1000) = 0.458333/0.5 = 0.916667
  M = -4.84 + 1.15 + 0.469333 + 0.336667 + 1.0704 + 0.125952 - 0.160533 + 0.077983 - 0.29975
    = -2.069948                                          -> m_flag False (not > -1.78)
  (one firm, so the point-in-time winsorisation has < 200 observations and clips nothing)

ACCRUALS (avg TA = (1200+1000)/2 = 1100)
  cash-flow     = (150 - 140)/1100                                       = 0.009091
  balance-sheet = [(100 - 30) - (50 - 10 - 5) - 60]/1100 = -25/1100      = -0.022727
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quality_screener import scores

PRIOR = dict(
    revenue=1000, gross_profit=400, sga=150, dep_amort=50, ebit=200, net_income=100,
    income_cont_ops=100, cfo=120, equity_issued=0, total_assets=1000, current_assets=400,
    current_liabilities=200, total_liabilities=600, receivables=100, ppe_net=300, cash=50,
    long_term_debt=300, retained_earnings=250, equity=400, shares_out=100, debt_current=20,
    taxes_payable=10,
)
CURRENT = dict(
    revenue=1200, gross_profit=540, sga=168, dep_amort=60, ebit=300, net_income=150,
    income_cont_ops=160, cfo=140, equity_issued=0, total_assets=1200, current_assets=500,
    current_liabilities=250, total_liabilities=700, receivables=150, ppe_net=400, cash=80,
    long_term_debt=300, retained_earnings=350, equity=500, shares_out=101, debt_current=30,
    taxes_payable=15,
)


def row(**overrides: float) -> pd.DataFrame:
    """The FY2023 wide-panel row (current + *_lag1 + total_assets_lag2), with overrides."""
    data: dict[str, object] = {
        "cik": "0000000001",
        "period_end": pd.Timestamp("2023-12-31"),
        "fiscal_year": 2023,
        "availability_date": pd.Timestamp("2024-02-01"),
        "period_end_lag1": pd.Timestamp("2022-12-31"),
        "total_assets_lag2": 800.0,
    }
    data.update({k: float(v) for k, v in CURRENT.items()})
    data.update({f"{k}_lag1": float(v) for k, v in PRIOR.items()})
    data.update(overrides)
    return pd.DataFrame([data])


def first(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[0]


# --- Piotroski ----------------------------------------------------------------------------


def test_piotroski_hand_example() -> None:
    f = first(scores.piotroski(row()))
    assert f["roa"] == pytest.approx(0.15, abs=1e-4)
    assert f["cfo_to_assets"] == pytest.approx(0.14, abs=1e-4)
    assert f["d_roa"] == pytest.approx(0.025, abs=1e-4)
    assert f["d_leverage"] == pytest.approx(-0.0606, abs=1e-4)
    assert f["d_current_ratio"] == pytest.approx(0.0, abs=1e-4)
    assert f["d_gross_margin"] == pytest.approx(0.05, abs=1e-4)
    assert f["d_asset_turnover"] == pytest.approx(-0.05, abs=1e-4)
    assert f[scores.F_SIGNALS].tolist() == [1, 1, 1, 0, 1, 0, 1, 1, 0]
    assert (f["f_raw"], f["f_n_components"], f["f_score"]) == (6, 9, 6)


def test_piotroski_rescales_with_eight_components() -> None:
    f = first(scores.piotroski(row(long_term_debt=np.nan)))
    assert math.isnan(f["f_dlever"])
    assert f["f_n_components"] == 8
    assert f["f_score"] == pytest.approx(5 * 9 / 8, abs=1e-4)


def test_piotroski_needs_seven_components() -> None:
    f = first(scores.piotroski(row(long_term_debt=np.nan, total_assets_lag2=np.nan)))
    assert f["f_n_components"] == 6  # dLEVER, dROA, dTURN all lost
    assert math.isnan(f["f_score"])


@pytest.mark.parametrize(
    ("issued", "shares", "expected"),
    [
        (0.0, 101.0, 1.0),     # tagged, nothing issued
        (50.0, 101.0, 0.0),    # tagged, issued
        (np.nan, 101.0, 1.0),  # untagged, shares +1%
        (np.nan, 105.0, 0.0),  # untagged, shares +5%
        (np.nan, np.nan, np.nan),
    ],
)
def test_equity_offer_signal(issued: float, shares: float, expected: float) -> None:
    got = first(scores.piotroski(row(equity_issued=issued, shares_out=shares)))["f_eq_offer"]
    assert (math.isnan(got) and math.isnan(expected)) or got == expected


# --- Altman -------------------------------------------------------------------------------


def test_altman_hand_example_book_equity_only() -> None:
    a = first(scores.altman(row()))
    expected = {"altman_x1": 0.2083, "altman_x2": 0.2917, "altman_x3": 0.25,
                "altman_x4_book": 0.7143, "altman_x5": 1.0, "z2_score": 4.7475}
    for col, val in expected.items():
        assert a[col] == pytest.approx(val, abs=1e-4), col
    assert a["z2_zone"] == "safe"
    assert math.isnan(a["z_score"]) and pd.isna(a["z_zone"])  # no market cap yet
    assert (a["z_n_components"], a["z2_n_components"]) == (4, 4)


def test_altman_with_market_cap() -> None:
    a = first(scores.altman(row(market_cap=1400.0)))
    assert a["altman_x4"] == pytest.approx(2.0, abs=1e-4)
    assert a["z_score"] == pytest.approx(3.6833, abs=1e-4)
    assert a["z_zone"] == "safe"


def test_altman_negative_equity_is_scored_not_dropped() -> None:
    a = first(scores.altman(row(equity=-200.0)))
    assert a["altman_x4_book"] == pytest.approx(-0.2857, abs=1e-4)
    assert a["z2_score"] == pytest.approx(3.6975, abs=1e-4)  # 3.9975 - 0.3


@pytest.mark.parametrize(
    ("z", "label"), [(3.0, "safe"), (2.99, "grey"), (1.81, "grey"), (1.80, "distress")]
)
def test_z_zone_boundaries(z: float, label: str) -> None:
    assert scores.zone(pd.Series([z]), scores.Z_SAFE, scores.Z_DISTRESS).iloc[0] == label


# --- Beneish ------------------------------------------------------------------------------

M_HAND = -2.069948


def test_beneish_hand_example() -> None:
    b = first(scores.beneish(row()))
    expected = {"m_dsri": 1.25, "m_gmi": 0.8889, "m_aqi": 0.8333, "m_sgi": 1.2, "m_depi": 1.0952,
                "m_sgai": 0.9333, "m_tata": 0.0167, "m_lvgi": 0.9167}
    for col, val in expected.items():
        assert b[col] == pytest.approx(val, abs=1e-4), col
        assert b[f"{col}_w"] == pytest.approx(val, abs=1e-4), f"{col}_w"
    assert b["m_score"] == pytest.approx(M_HAND, abs=1e-4)
    assert b["m_flag"] == False  # noqa: E712 (nullable boolean)
    assert b["m_n_components"] == 8
    assert pd.isna(b["m_imputed_components"])


def test_beneish_missing_sga_is_neutral_and_visible() -> None:
    b = first(scores.beneish(row(sga=np.nan)))
    assert math.isnan(b["m_sgai"])  # raw index stays missing
    assert b["m_sgai_w"] == 1.0
    assert b["m_imputed_components"] == "SGAI"
    assert b["m_n_components"] == 7
    # SGAI 0.933333 -> 1.0 changes M by -0.172 * 0.066667
    assert b["m_score"] == pytest.approx(M_HAND - 0.172 * (1 - 0.14 / 0.15), abs=1e-4)


def test_beneish_missing_depreciation_is_neutral() -> None:
    b = first(scores.beneish(row(dep_amort_lag1=np.nan)))
    assert b["m_depi_w"] == 1.0
    assert b["m_imputed_components"] == "DEPI"


def test_zero_prior_revenue_gives_nan_not_inf() -> None:
    b = first(scores.beneish(row(revenue_lag1=0.0)))
    assert math.isnan(b["m_dsri"]) and math.isnan(b["m_sgi"]) and math.isnan(b["m_gmi"])
    assert math.isnan(b["m_sgai"]) and pd.isna(b["m_imputed_components"])  # not an SGA gap
    assert math.isnan(b["m_score"]) and pd.isna(b["m_flag"])
    everything = scores.compute_all(row(revenue_lag1=0.0)).select_dtypes("number")
    assert not np.isinf(everything.to_numpy(dtype=float)).any()


def test_point_in_time_winsorisation_ignores_later_filings() -> None:
    n = 300
    dates = pd.Series(pd.date_range("2015-01-01", periods=n, freq="D"))
    values = pd.DataFrame({"x": np.arange(1.0, n + 1)})
    values.loc[n - 1, "x"] = 1e6  # extreme value filed last
    clipped = scores.winsorize_point_in_time(values, dates, min_obs=200)
    # rows before #200 have too little history to clip
    assert clipped["x"].iloc[:199].tolist() == values["x"].iloc[:199].tolist()
    # the extreme value is clipped to the 99th percentile of what was public by then
    assert clipped["x"].iloc[-1] < 1e6
    # an extreme value arriving later must not move earlier bounds
    without_extreme = scores.winsorize_point_in_time(values.iloc[:-1], dates.iloc[:-1], min_obs=200)
    pd.testing.assert_series_equal(clipped["x"].iloc[:-1], without_extreme["x"])


# --- accruals -----------------------------------------------------------------------------


def test_accruals_hand_example() -> None:
    a = first(scores.accruals(row()))
    assert a["avg_total_assets"] == 1100
    assert a["accruals_cf"] == pytest.approx(0.0091, abs=1e-4)
    assert a["accruals_bs"] == pytest.approx(-0.0227, abs=1e-4)
    assert a["accruals_bs_n_components"] == 6
    assert pd.isna(a["accruals_bs_omitted"])


def test_balance_sheet_accruals_name_omitted_terms() -> None:
    a = first(scores.accruals(row(debt_current=np.nan)))
    # [(100 - 30) - (50 - 5) - 60] / 1100 = -35/1100
    assert a["accruals_bs"] == pytest.approx(-0.0318, abs=1e-4)
    assert a["accruals_bs_omitted"] == "dSTD"


def test_balance_sheet_accruals_need_core_terms() -> None:
    a = first(scores.accruals(row(cash_lag1=np.nan)))
    assert math.isnan(a["accruals_bs"]) and pd.isna(a["accruals_bs_omitted"])
    assert a["accruals_cf"] == pytest.approx(0.0091, abs=1e-4)


# --- compute_all --------------------------------------------------------------------------


def two_row_panel() -> pd.DataFrame:
    """FY2022 row (its lag is FY2021, TA 800) and the FY2023 row, without total_assets_lag2."""
    fy22 = row().drop(columns="total_assets_lag2").assign(
        period_end=pd.Timestamp("2022-12-31"), fiscal_year=2022,
        availability_date=pd.Timestamp("2023-03-15"), period_end_lag1=pd.Timestamp("2021-12-31"),
        total_assets=1000.0, total_assets_lag1=800.0,
    )
    fy23 = row().drop(columns="total_assets_lag2")
    return pd.concat([fy22, fy23], ignore_index=True)


def test_compute_all_takes_lag2_from_prior_row_and_respects_its_availability() -> None:
    out = scores.compute_all(two_row_panel()).set_index("fiscal_year")
    assert out.loc[2023, "f_score"] == 6  # needed TA_lag2 = 800 from the FY2022 row
    # FY2022 row was available 2023-03-15, later than FY2023's own 2024-02-01? no -> unchanged
    assert out.loc[2023, "availability_date"] == pd.Timestamp("2024-02-01")
    assert math.isnan(out.loc[2022, "d_roa"])  # no FY2020 assets for the FY2022 row


def test_compute_all_availability_moves_to_later_prior_row() -> None:
    panel = two_row_panel()
    panel.loc[0, "availability_date"] = pd.Timestamp("2024-06-30")  # prior row usable later
    out = scores.compute_all(panel).set_index("fiscal_year")
    assert out.loc[2023, "availability_date"] == pd.Timestamp("2024-06-30")


def test_compute_all_attaches_firm_flags() -> None:
    firms = pd.DataFrame(
        {"cik": ["0000000001"], "ticker": ["TST"], "name": ["Test Co"], "sector": ["Manufacturing"],
         "sic": pd.array([3571], dtype="Int64"), "is_financial": pd.array([False], dtype="boolean"),
         "is_utility": pd.array([False], dtype="boolean"),
         "excluded_from_main": pd.array([False], dtype="boolean")}
    )
    out = scores.compute_all(row(), firms)
    assert out.loc[0, "ticker"] == "TST" and out.loc[0, "excluded_from_main"] == False  # noqa: E712
    assert list(out.columns[:4]) == scores.ID_COLUMNS
