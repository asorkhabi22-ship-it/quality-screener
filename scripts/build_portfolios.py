"""Monthly quintile holdings on each quality signal, formed on filing dates.

Reads data/processed/scores.parquet (scripts/compute_scores.py), universe.parquet, and
prices_monthly.parquet + trading_calendar.parquet (scripts/fetch_prices.py).

Outputs:
    data/processed/holdings.parquet                 main sample (ex financials and utilities)
    data/processed/holdings_incl_fin_util.parquet   with --include-fin-util (robustness)

Usage:
    python scripts/build_portfolios.py [--include-fin-util]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from quality_screener import config, portfolios, prices, scores, universe


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--include-fin-util", action="store_true",
                        help="keep financials and utilities in the sorts")
    args = parser.parse_args(argv)
    pd.set_option("display.width", 200)

    built = portfolios.build_holdings(
        pd.read_parquet(scores.SCORES_PATH),
        pd.read_parquet(universe.UNIVERSE_PATH),
        pd.read_parquet(prices.MONTHLY_PATH),
        prices.load_calendar(),
        include_fin_util=args.include_fin_util,
    )
    path = portfolios.HOLDINGS_FIN_UTIL_PATH if args.include_fin_util else portfolios.HOLDINGS_PATH
    built.holdings.to_parquet(path, index=False)
    h = built.holdings
    print(f"wrote {len(h)} holdings to {path.relative_to(config.PROJECT_ROOT)}: "
          f"{h['t'].nunique()} formation dates {h['t'].min():%Y-%m-%d} to {h['t'].max():%Y-%m-%d}")

    counts = portfolios.bucket_counts(h)
    for signal in portfolios.SIGNAL_ORDER:
        table = counts.loc[signal].dropna(axis=1, how="all")
        print(f"\n{signal}: average firms per bucket per month (bucket {table.columns.max()} = "
              "high quality)")
        print(table.round(1).to_string())

    max_age = portfolios.MAX_AGE_MONTHS
    print(f"\nDropped as stale (latest row's period ended >{max_age} months before t):")
    print(portfolios.stale_counts(built.stale).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
