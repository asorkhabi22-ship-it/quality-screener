"""Invariants on project configuration."""

from pathlib import Path

import pytest

from quality_screener import config


def test_user_agent_has_contact_email() -> None:
    assert "@" in config.USER_AGENT


def test_sample_window_is_ordered() -> None:
    assert config.START_YEAR < config.END_YEAR


def test_rate_limit_respects_sec_fair_access_cap() -> None:
    assert 0 < config.MAX_REQUESTS_PER_SEC <= 10


@pytest.mark.parametrize(
    "path",
    [
        config.DATA_DIR,
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR,
    ],
)
def test_paths_live_under_project_root_and_exist(path: Path) -> None:
    assert path.is_relative_to(config.PROJECT_ROOT)
    assert path.is_dir()
