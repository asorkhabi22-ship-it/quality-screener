"""scripts/fetch_all.py: a failing firm is logged, the batch keeps going."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quality_screener.edgar import EdgarError
from scripts.fetch_all import FAILURE_COLUMNS, fetch_all, write_failures


class RecordingClient:
    def __init__(self, broken: set[tuple[str, str]]) -> None:
        self.broken = broken
        self.calls: list[tuple[str, str]] = []

    def _get(self, endpoint: str, cik: str) -> dict:
        self.calls.append((endpoint, cik))
        if (endpoint, cik) in self.broken:
            raise EdgarError("GET ... -> HTTP 404 (not retryable)")
        return {}

    def get_submissions(self, cik: str) -> dict:
        return self._get("submissions", cik)

    def get_company_facts(self, cik: str) -> dict:
        return self._get("companyfacts", cik)


def test_failure_is_recorded_and_batch_continues() -> None:
    targets = pd.DataFrame({"cik": ["0000000001", "0000000002", "0000000003"], "ticker": ["A", "B", "C"]})
    client = RecordingClient(broken={("companyfacts", "0000000002")})

    failures = fetch_all(client, targets)  # type: ignore[arg-type]

    assert len(client.calls) == 6  # every endpoint for every firm was attempted
    assert failures[["ticker", "endpoint"]].values.tolist() == [["B", "companyfacts"]]
    assert "HTTP 404" in failures.loc[0, "error"]


def test_write_failures_always_writes_header(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "failures.csv"
    write_failures(pd.DataFrame(columns=FAILURE_COLUMNS), path)
    assert path.read_text().strip() == ",".join(FAILURE_COLUMNS)
