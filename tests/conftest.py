"""Shared fixtures. Every test runs with real network access disabled."""

import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to open a real connection (sec.gov, Wikipedia, ...)."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("real network access attempted inside a test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
