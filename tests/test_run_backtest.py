"""Tests for the helpers in scripts/run_backtest.py: tables, scenarios and the generated
reading of the results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_backtest as rb

MONTHS = pd.date_range("2017-01-31", periods=48, freq="ME")


def test_formatting_helpers() -> None:
    assert rb.pct(0.0123) == "+1.23%" and rb.pct(-0.004, 1) == "-0.4%" and rb.pct(np.nan) == ""
    assert rb.num(1.234) == "1.23" and rb.num(np.nan) == ""
    table = rb.md_table(pd.DataFrame({"A": ["x", "y"], "B": ["1", "2"]}))
    assert table.splitlines() == ["| A | B |", "|---|---|", "| x | 1 |", "| y | 2 |"]


def test_market_excess_subtracts_risk_free() -> None:
    monthly = pd.DataFrame({"ticker": "SPY", "t": MONTHS[:2], "ret": [0.02, -0.01]})
    rf = pd.Series([0.001, 0.002], index=pd.PeriodIndex(MONTHS[:2], freq="M"))
    assert rb.market_excess_returns(monthly, rf).tolist() == pytest.approx([0.019, -0.012])
    assert rb.market_excess_returns(monthly, None).tolist() == pytest.approx([0.02, -0.01])


def bucket_rows(signal: str, means: list[float]) -> list[dict[str, object]]:
    return [{"signal": signal, "weighting": w, "bucket": b + 1, "mean": m}
            for w in ("ew", "vw") for b, m in enumerate(means)]


def ls_rows(
    signal: str, t_ew: float, t_vw: float, increasing: bool = False
) -> list[dict[str, object]]:
    return [{"signal": signal, "weighting": w, "mean": 0.001 * t, "nw_t": t, "nw_se": 0.001,
             "increasing": increasing} for w, t in (("ew", t_ew), ("vw", t_vw))]


def test_main_table_puts_f_groups_in_q1_q3_q5() -> None:
    buckets = pd.DataFrame(
        bucket_rows("f_group", [0.03, 0.02, 0.01]) + bucket_rows("composite", [0.1, 0.2, 0.3, 0.4, 0.5])
    )
    ls_stats = pd.DataFrame(ls_rows("f_group", -2.5, -1.0) + ls_rows("composite", 0.5, 0.2))
    empty = pd.DataFrame()
    bt = rb.Backtest(empty, empty, buckets, ls_stats, empty, pd.Series(dtype=float), empty)
    table = rb.main_table(bt, None)
    f = table.loc[("f_group", "ew")]
    assert [f["q1"], f["q3"], f["q5"]] == [0.03, 0.02, 0.01]
    assert np.isnan(f["q2"]) and np.isnan(f["q4"])
    assert table.loc[("composite", "vw"), "q4"] == 0.4
    assert table.index.get_level_values(0).tolist()[:2] == ["f_group", "f_group"]  # signal order


def test_reading_notes_follow_the_estimates() -> None:
    table = pd.DataFrame(
        ls_rows("f_score", -2.3, -2.1) + ls_rows("accruals_cf", 2.8, 1.5) + ls_rows("altman_z", 0.4, -0.9)
    ).set_index(["signal", "weighting"])
    notes = "\n".join(rb.reading_notes(table))
    assert "Piotroski F (quintiles): significant with the opposite sign" in notes
    assert "Accruals (cash flow): significant equal-weighted only" in notes
    assert "Altman Z: no detectable effect" in notes
    assert "No signal's bucket means rise steadily" in notes


def test_bottom_bucket_drivers_and_note() -> None:
    holdings = pd.DataFrame({"signal": "f_group", "t": MONTHS[0], "ticker": ["A", "B", "C"],
                             "bucket": [1, 1, 3]})
    firm_ret = pd.DataFrame({"t": MONTHS[0], "ticker": ["A", "B", "C"], "month": MONTHS[1],
                             "ret": [0.30, 0.02, 0.5]})
    d = rb.bottom_bucket_drivers(holdings, firm_ret, "f_group", n=1)
    assert d.tickers == ["A"]
    assert d.mean == pytest.approx(0.16) and d.mean_without == pytest.approx(0.02)
    table = pd.DataFrame(ls_rows("f_group", -2.5, -1.0)).set_index(["signal", "weighting"])
    notes = "\n".join(rb.reading_notes(table, {"f_group": d}))
    assert "largest contributors are A" in notes and "+16.00% to +2.00%" in notes


def test_verdict_lines_name_direction() -> None:
    table = pd.DataFrame(ls_rows("accruals_cf", 2.8, 1.5)).set_index(["signal", "weighting"])
    (line,) = rb.verdict_lines(table)
    assert "significant EW (high quality beat low quality)" in line
    assert "VW" not in line.split(":")[-1]


def test_robustness_scenarios_split_months() -> None:
    rng = np.random.default_rng(0)
    ls = pd.DataFrame({"signal": "composite", "t": MONTHS, "month": MONTHS,
                       "ew": rng.normal(0, 0.01, 48), "vw": rng.normal(0, 0.01, 48)})
    mkt = pd.Series(rng.normal(0, 0.04, 48), index=pd.PeriodIndex(MONTHS, freq="M"))
    out = rb.robustness(ls, ls.iloc[:10], mkt)
    months = out[out["weighting"] == "ew"].set_index("scenario")["months"]
    assert months["Base (ex financials & utilities)"] == 48
    assert months["2012-2018"] == 24 and months["2019-2026"] == 24
    assert months["Ex 2020"] == 36
    assert months["Incl. financials & utilities"] == 10
