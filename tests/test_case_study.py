"""Tests for case_study: event day, event-window returns, version comparison and the
restated panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quality_screener import case_study as cs

T = pd.Timestamp
CAL = pd.bdate_range("2018-11-01", "2018-12-31").astype("datetime64[ns]")


def test_event_day_respects_the_new_york_close() -> None:
    # 21:53 UTC = 16:53 New York (EST): after the close -> next trading day
    assert cs.event_day("2018-11-15T21:53:25.000Z", CAL) == T("2018-11-16")
    assert cs.event_day("2018-11-15T15:00:00.000Z", CAL) == T("2018-11-15")  # 10:00 NY
    assert cs.event_day("2018-11-17T15:00:00.000Z", CAL) == T("2018-11-19")  # Saturday


def daily_frame() -> pd.DataFrame:
    dates = CAL[:8]
    firm = [100, 101, 102, 110, 99, 99, 100, 100]
    spy = [200, 200, 202, 202, 202, 204, 204, 204]
    return pd.concat([
        pd.DataFrame({"ticker": "AAA", "date": dates, "adj_close": np.array(firm, dtype=float)}),
        pd.DataFrame({"ticker": "SPY", "date": dates, "adj_close": np.array(spy, dtype=float)}),
    ])


def test_event_window_and_bhar_by_hand() -> None:
    w = cs.event_window(daily_frame(), "AAA", CAL[3], before=2, after=2)
    assert w["rel_day"].tolist() == [-2, -1, 0, 1, 2]
    # day 0: firm 102 -> 110 (+7.843%), SPY flat
    assert w.loc[w["rel_day"] == 0, "abnormal"].item() == pytest.approx(110 / 102 - 1)
    # days 0..1: firm 102 -> 99, SPY 202 -> 202
    assert cs.bhar(w, 0, 1) == pytest.approx(99 / 102 - 1)
    # days -1..2: firm 101 -> 99, SPY 200 -> 204
    assert cs.bhar(w, -1, 2) == pytest.approx((99 / 101 - 1) - (204 / 200 - 1))


def versions() -> pd.DataFrame:
    rows = [
        ("revenue", "2016-06-30", 100.0, "orig", "2016-08-26"),
        ("receivables", "2016-06-30", 30.0, "orig", "2016-08-26"),
        ("total_assets", "2016-06-30", 200.0, "orig", "2016-08-26"),
        ("revenue", "2016-06-30", 98.0, "rest", "2019-05-17"),
        ("receivables", "2016-06-30", 18.0, "rest", "2019-05-17"),
        ("revenue", "2017-06-30", 120.0, "rest", "2019-05-17"),
    ]
    return pd.DataFrame([
        {"cik": "1", "field": f, "period_end": T(p), "val": v, "accn": a, "form": "10-K",
         "filed": T(d), "source_tag": "us-gaap:X"}
        for f, p, v, a, d in rows
    ])


def test_compare_versions_side_by_side() -> None:
    out = cs.compare_versions(versions(), "1", "rest", [T("2016-06-30")],
                              fields=("revenue", "receivables", "total_assets"))
    by = out.set_index("field")
    assert by.loc["receivables", ["original", "restated"]].tolist() == [30.0, 18.0]
    assert by.loc["receivables", "pct_change"] == pytest.approx(-0.4)
    assert np.isnan(by.loc["total_assets", "restated"])  # not in the restating filing
    assert out["field"].tolist() == ["revenue", "receivables", "total_assets"]


def test_restated_long_replaces_only_reported_values() -> None:
    long = cs.restated_long(versions(), "1", "rest").set_index(["field", "period_end"])
    assert long.loc[("revenue", T("2016-06-30")), "val"] == 98.0
    assert long.loc[("receivables", T("2016-06-30")), "accn"] == "rest"
    assert long.loc[("total_assets", T("2016-06-30")), "val"] == 200.0  # kept original
    assert long.loc[("revenue", T("2017-06-30")), "val"] == 120.0
    assert len(long) == 4
