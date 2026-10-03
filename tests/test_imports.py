"""Every package module imports without side-effect errors."""

import importlib

import pytest

MODULES = [
    "config",
    "edgar",
    "universe",
    "xbrl_map",
    "panel",
    "scores",
    "prices",
    "portfolios",
    "stats",
    "factors",
    "viz",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(f"quality_screener.{name}")
