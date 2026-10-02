"""Test doubles for the network layer: a scripted requests.Session and a fake clock."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeResponse:
    status_code: int
    content: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


def json_response(payload: Any, status: int = 200) -> FakeResponse:
    return FakeResponse(status, json.dumps(payload).encode())


class FakeSession:
    """Serves scripted responses per URL, in order; records every requested URL."""

    def __init__(self, routes: dict[str, list[FakeResponse | Exception]]) -> None:
        self.routes = {url: list(queue) for url, queue in routes.items()}
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(
        self, url: str, timeout: float | None = None, headers: dict[str, str] | None = None
    ) -> FakeResponse:
        self.calls.append(url)
        queue = self.routes.get(url)
        if not queue:
            raise AssertionError(f"unexpected request: {url}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    """Monotonic clock whose sleep() advances time instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class NoLimit:
    """Limiter stand-in that never blocks; counts acquisitions."""

    def __init__(self) -> None:
        self.acquired = 0

    def acquire(self) -> None:
        self.acquired += 1
