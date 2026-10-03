"""Compute all quality scores for the universe and print a summary.

Reads data/processed/panel_pit.parquet and universe.parquet (run scripts/build_panels.py
first); writes data/processed/scores.parquet.

Usage:
    python scripts/compute_scores.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from quality_screener import config, panel, scores, universe

SCORE_COLUMNS = ["f_score", "z2_score", "m_score", "accruals_cf", "accruals_bs"]


def main_sample(scored: pd.DataFrame) -> pd.DataFrame:
    """Firm-years outside financials and utilities (unknown SIC excluded too)."""
    return scored[scored["excluded_from_main"].eq(False).fillna(False)]


def top_m_scores(scored: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The n highest M-scores with the indices that drive them."""
    cols = ["ticker", "name", "fiscal_year", "m_score", "m_sgi_w", "m_tata_w", "m_dsri_w",
            "m_aqi_w", "m_gmi_w", "m_imputed_components"]
    return scored.nlargest(n, "m_score")[cols].reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    pit = pd.read_parquet(panel.PIT_PATH)
    firms = pd.read_parquet(universe.UNIVERSE_PATH)
    scored = scores.compute_all(pit, firms)
    scored.to_parquet(scores.SCORES_PATH, index=False)
    print(f"wrote {len(scored)} firm-years to {scores.SCORES_PATH.relative_to(config.PROJECT_ROOT)}")

    main = main_sample(scored)
    in_window = main[main["fiscal_year"].between(config.START_YEAR, config.END_YEAR)]
    print("\nScored firm-years per fiscal year (main sample: ex financials and utilities):")
    print(in_window.groupby("fiscal_year")[SCORE_COLUMNS].count().T.to_string())

    print("\nSummary statistics (main sample, FY%d-FY%d):" % (config.START_YEAR, config.END_YEAR))
    print(in_window[SCORE_COLUMNS].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.round(3).to_string())

    print("\nF-score distribution (rescaled scores rounded to nearest integer):")
    f_counts = in_window["f_score"].round().value_counts().sort_index().astype(int)
    print((f_counts / f_counts.sum() * 100).round(1).rename("pct").to_frame().T.to_string())

    print(f"\nM-score flag rate (M > {scores.M_THRESHOLD}): {in_window['m_flag'].mean():.1%}")
    print("\n10 highest M-scores (main sample, winsorised indices):")
    print(top_m_scores(in_window).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
