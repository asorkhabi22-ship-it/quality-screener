"""Download companyfacts + submissions for every firm in the universe into data/raw.

Cached files are skipped (EdgarClient serves them from disk without a network call). A firm
that fails is logged to data/raw/failures.csv and the run continues.

Usage:
    python scripts/fetch_all.py                         # whole universe (built first if missing)
    python scripts/fetch_all.py --tickers AAPL MSFT KO  # just these tickers
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from quality_screener import config, universe
from quality_screener.edgar import EdgarClient

FAILURES_CSV = config.RAW_DIR / "failures.csv"
FAILURE_COLUMNS = ["cik", "ticker", "endpoint", "error"]

log = logging.getLogger("fetch_all")


def resolve_targets(client: EdgarClient, tickers: Sequence[str] | None) -> pd.DataFrame:
    """Firms to download as columns ``cik``, ``ticker``: given tickers, or the full universe."""
    if tickers:
        requested = pd.DataFrame({"ticker": list(tickers)})
        mapped = universe.map_tickers_to_ciks(requested, client.get_ticker_map())
        return universe.collapse_share_classes(mapped)[["cik", "ticker"]]
    return universe.load_or_build_universe(client)[["cik", "ticker"]]


def fetch_all(client: EdgarClient, targets: pd.DataFrame) -> pd.DataFrame:
    """Fetch submissions and companyfacts per firm; return one row per failed request."""
    endpoints = {
        "submissions": client.get_submissions,
        "companyfacts": client.get_company_facts,
    }
    failures: list[dict[str, str]] = []
    for firm in tqdm(targets.itertuples(index=False), total=len(targets), unit="firm"):
        for endpoint, fetch in endpoints.items():
            try:
                fetch(firm.cik)
            except Exception as exc:  # batch job: record any failure and keep going
                failures.append(
                    {"cik": firm.cik, "ticker": firm.ticker, "endpoint": endpoint,
                     "error": f"{type(exc).__name__}: {exc}"}
                )
    return pd.DataFrame(failures, columns=FAILURE_COLUMNS)


def write_failures(failures: pd.DataFrame, path: Path = FAILURES_CSV) -> None:
    """Overwrite the failures log with this run's failures (header only if none)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", help="only these tickers (default: universe)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    client = EdgarClient()
    targets = resolve_targets(client, args.tickers)
    failures = fetch_all(client, targets)
    write_failures(failures)
    log.info(
        "%d firms, %d failed requests (see %s), %d network calls",
        len(targets), len(failures), FAILURES_CSV, client.network_calls,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
