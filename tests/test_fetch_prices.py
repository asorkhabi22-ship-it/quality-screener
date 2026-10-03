"""Tests for the helpers in scripts/fetch_prices.py."""

from __future__ import annotations

import pandas as pd

from quality_screener import prices
from scripts.fetch_prices import coverage_report, price_tickers


def test_price_tickers_adds_benchmark_once() -> None:
    firms = pd.DataFrame({"ticker": ["AAA", "SPY", "BBB"]})
    assert price_tickers(firms) == ["AAA", "SPY", "BBB"]
    assert price_tickers(pd.DataFrame({"ticker": ["AAA"]})) == ["AAA", prices.BENCHMARK]


def test_coverage_report_lists_tickers_without_data() -> None:
    cal = pd.bdate_range("2020-01-01", periods=5).astype("datetime64[ns]")
    daily = prices.add_split_factor(pd.DataFrame({
        "ticker": "AAA", "date": cal, "close": 1.0, "adj_close": 1.0,
        "dividends": 0.0, "splits": 0.0, "volume": 1.0,
    }))
    report = coverage_report(daily, cal, ["ZZZ"]).set_index("ticker")
    assert pd.isna(report.loc["AAA", "issue"])
    assert report.loc["ZZZ", "issue"] == "no data"
