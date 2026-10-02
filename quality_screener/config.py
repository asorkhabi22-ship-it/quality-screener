"""Project-wide constants: SEC identity, sample window, rate limit and filesystem paths.

Nothing here performs I/O. Modules import these values instead of hard-coding them.
"""

from pathlib import Path
from typing import Final

# SEC requires a descriptive User-Agent with contact info on every request to sec.gov.
# https://www.sec.gov/os/accessing-edgar-data
USER_AGENT: Final[str] = "Amir Sorkhabi aa.sorkhabi@gmail.com"

# Fiscal years in the sample (inclusive).
START_YEAR: Final[int] = 2010
END_YEAR: Final[int] = 2025

# SEC fair-access policy caps clients at 10 requests/second; stay below it.
MAX_REQUESTS_PER_SEC: Final[int] = 8

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
