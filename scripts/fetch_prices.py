"""Download daily prices for the universe plus SPY, build month-end series, check share counts.

Only tickers not yet cached in data/raw/prices are requested from Yahoo Finance (yfinance).
The share check needs data/processed/panel_pit.parquet (scripts/build_panels.py).

Outputs:
    data/raw/prices/<TICKER>.parquet         daily close, adj close, dividends, splits (cache)
    data/raw/yahoo_shares.parquet            Yahoo's current share counts (cache)
    data/processed/prices_monthly.parquet    month-end prices and total returns per ticker
    data/processed/trading_calendar.parquet  trading days (SPY dates)
    reports/price_coverage.csv               per-ticker date range, gaps and issues
    reports/shares_check.csv                 latest-year shares_out vs Yahoo's current count

Usage:
    python scripts/fetch_prices.py [--refresh] [--skip-shares-check]
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import pandas as pd

from quality_screener import config, panel, prices, universe

log = logging.getLogger("fetch_prices")


def price_tickers(firms: pd.DataFrame) -> list[str]:
    """Universe tickers (one per firm) plus the benchmark, without duplicates."""
    return list(dict.fromkeys([*firms["ticker"], prices.BENCHMARK]))


def coverage_report(
    daily: pd.DataFrame, calendar: pd.DatetimeIndex, no_data: Sequence[str]
) -> pd.DataFrame:
    """``price_coverage`` plus one 'no data' row per ticker Yahoo had nothing for."""
    coverage = prices.price_coverage(daily, calendar)
    missing = pd.DataFrame({"ticker": list(no_data), "issue": "no data"})
    return pd.concat([coverage, missing], ignore_index=True).astype({"issue": "string"})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download cached tickers")
    parser.add_argument("--skip-shares-check", action="store_true", help="no Yahoo share counts")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 200)

    firms = pd.read_parquet(universe.UNIVERSE_PATH)
    tickers = price_tickers(firms)
    no_data = prices.download_prices(tickers, refresh=args.refresh)
    daily = prices.load_daily(tickers)
    calendar = prices.trading_calendar(daily)
    prices.save_calendar(calendar)

    coverage = coverage_report(daily, calendar, no_data)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(prices.COVERAGE_REPORT, index=False)
    prices.log_price_issues(coverage[coverage["issue"] != "no data"], no_data)

    monthly = prices.monthly_prices(daily, calendar)
    monthly.to_parquet(prices.MONTHLY_PATH, index=False)
    print(f"{daily['ticker'].nunique()} of {len(tickers)} tickers with prices, "
          f"{calendar[0]:%Y-%m-%d} to {calendar[-1]:%Y-%m-%d} ({len(calendar)} trading days)")
    out_path = prices.MONTHLY_PATH.relative_to(config.PROJECT_ROOT)
    print(f"wrote {len(monthly)} ticker-months to {out_path}")
    print("\nPrice issues:", coverage["issue"].value_counts().to_dict() or "none")
    late = coverage["first_date"] > calendar[0] + pd.Timedelta(days=7)
    print(f"{late.sum()} tickers start after {calendar[0]:%Y-%m-%d} (later listings / spin-offs)")

    if args.skip_shares_check:
        return 0
    pit = pd.read_parquet(panel.PIT_PATH)
    with_mcap = prices.add_market_cap(pit, firms, daily, calendar)
    current = prices.fetch_current_shares(firms["ticker"].tolist())
    check = prices.check_shares(with_mcap, firms, current)
    check.to_csv(prices.SHARES_CHECK_REPORT, index=False)
    flagged = check[check["mismatch"].fillna(False)]
    print(f"\nShares check: {check['ratio'].notna().sum()} firms compared, {len(flagged)} differ "
          f"from Yahoo by >{prices.SHARES_TOLERANCE:.0%} (reports/shares_check.csv)")
    cols = ["ticker", "fiscal_year", "shares_out", "shares_out_adj_reported", "yahoo_shares",
            "ratio", "rejected"]
    print(flagged[cols].head(25).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
