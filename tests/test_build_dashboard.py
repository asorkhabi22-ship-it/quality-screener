"""Tests for the firm-selection rules in scripts/build_dashboard.py."""

from __future__ import annotations

import pandas as pd

from scripts.build_dashboard import most_flagged, radar_tickers, radar_year


def main_rows() -> pd.DataFrame:
    rows = []
    for year, n_m in ((2024, 4), (2025, 2)):
        for i in range(4):
            rows.append({"ticker": f"T{i}", "fiscal_year": year,
                         "period_end": pd.Timestamp(f"{year}-12-31"), "market_cap": 10.0 * (i + 1),
                         "m_score": [-1.0, -2.0, -1.5, -3.0][i] if i < n_m else None})
    return pd.DataFrame(rows)


def test_radar_year_needs_enough_m_scores() -> None:
    assert radar_year(main_rows(), min_firms=3) == 2024
    assert radar_year(main_rows(), min_firms=2) == 2025


def test_radar_tickers_largest_then_highest_m() -> None:
    # 2024 M-scores: T0 -1.0, T1 -2.0, T2 -1.5, T3 -3.0; caps T3 > T2 > T1 > T0
    assert radar_tickers(main_rows(), 2024, n_largest=1, n_highest_m=2) == ["T3", "T0", "T2"]


def test_most_flagged_breaks_ties_by_market_cap() -> None:
    # flagged (M > -1.78): T0 in 2024 and 2025, T2 in 2024 only -> T0
    assert most_flagged(main_rows()) == "T0"
    tied = main_rows().assign(m_score=lambda d: d["m_score"].where(d["ticker"] != "T0"))
    tied.loc[(tied["ticker"] == "T1") & (tied["fiscal_year"] == 2025), "m_score"] = -1.0
    assert most_flagged(tied) == "T2"  # T1 and T2 one flag each; T2 is larger
