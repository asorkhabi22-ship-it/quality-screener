"""Build the point-in-time and all-versions panels for the universe, plus a coverage report.

Reads companyfacts through EdgarClient (cache first; run scripts/fetch_all.py beforehand).

Outputs:
    data/processed/panel_pit.parquet           wide, one row per (cik, period_end), with *_lag1
    data/processed/panel_pit_long.parquet      long, first-reported value + source_tag
    data/processed/panel_all_versions.parquet  long, every filing's value (restatements)
    data/processed/panel_late_values.parquet   values left out as reported >90d after the 10-K
    data/processed/panel_failures.csv          CIKs that could not be built
    reports/coverage_by_year.csv               % non-null by field x fiscal year (main sample)
    reports/coverage_by_year_fin_util.csv      same, financials + utilities
    reports/source_tags.csv                    which tag supplied each field, how often

Usage:
    python scripts/build_panels.py
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import pandas as pd

from quality_screener import config, panel, universe
from quality_screener.edgar import EdgarClient

log = logging.getLogger("build_panels")

KEY_FIELDS = ["revenue", "net_income", "cfo", "total_assets"]
FIRST_REPORT_YEAR = 2012


def split_main_sample(pit: pd.DataFrame, firms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(main sample, financials + utilities). Firms with unknown SIC go to neither."""
    excluded = firms.set_index("cik")["excluded_from_main"]
    flag = pit["cik"].map(excluded).astype("boolean")
    return pit[flag.eq(False).fillna(False)], pit[flag.eq(True).fillna(False)]


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)

    client = EdgarClient()
    firms = universe.load_or_build_universe(client)
    built, failures = panel.build_universe_panel(firms["cik"].tolist(), client)
    panel.save_panel(built)
    failures.to_csv(config.PROCESSED_DIR / "panel_failures.csv", index=False)

    main_pit, fin_util_pit = split_main_sample(built.pit, firms)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    coverage = panel.coverage_by_year(main_pit)
    coverage.to_csv(config.REPORTS_DIR / "coverage_by_year.csv")
    panel.coverage_by_year(fin_util_pit).to_csv(config.REPORTS_DIR / "coverage_by_year_fin_util.csv")
    panel.source_tag_counts(built.pit_long).to_csv(config.REPORTS_DIR / "source_tags.csv", index=False)

    years = [y for y in coverage.columns if FIRST_REPORT_YEAR <= y <= config.END_YEAR]
    log.info(
        "%d firms, %d firm-years, %d failures; main-sample coverage %%, FY%d-FY%d:\n%s",
        built.pit["cik"].nunique(), len(built.pit), len(failures), years[0], years[-1],
        coverage.loc[KEY_FIELDS, years].to_string(),
    )
    late = built.late_values
    log.info(
        "%d late-reported values left out of the PIT panel (in %d firm-years); most common: %s",
        len(late), late.groupby(["cik", "period_end"]).ngroups,
        late["field"].value_counts().head(5).to_dict(),
    )
    network_note = "no" if client.network_calls == 0 else str(client.network_calls)
    log.info("%s network calls", network_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
