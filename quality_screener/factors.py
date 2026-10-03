"""Fama-French five factors and momentum from Kenneth French's data library (optional).

Monthly Mkt-RF, SMB, HML, RMW, CMA and RF (Fama and French (2015), A five-factor asset pricing
model, Journal of Financial Economics 116, 1-22) and Mom (Carhart (1997), Journal of Finance
52(1), 57-82), as decimals indexed by calendar month. The zip files are cached in
``data/raw/french`` and never re-downloaded, so results stay reproducible after the library
updates. Everything here is optional: ``load_factors`` returns None if a download fails.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from quality_screener import config

log = logging.getLogger(__name__)

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_ZIP = "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_ZIP = "F-F_Momentum_Factor_CSV.zip"
CACHE_DIR = config.RAW_DIR / "french"
USER_AGENT = "quality-screener research script"  # sec.gov's identity is not needed here
MISSING_VALUES = (-99.99, -999.0)
MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,")


def fetch_zip(
    name: str, cache_dir: Path = CACHE_DIR, session: requests.Session | None = None
) -> bytes:
    """The zip file ``name`` from the cache, downloading it first if it is not there."""
    path = cache_dir / name
    if path.exists():
        return path.read_bytes()
    session = session if session is not None else requests.Session()
    resp = session.get(FRENCH_URL + name, headers={"User-Agent": USER_AGENT}, timeout=30)
    if resp.status_code != 200:
        raise requests.HTTPError(f"HTTP {resp.status_code} for {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return resp.content


def zip_text(blob: bytes) -> str:
    """Text of the single CSV inside a French data-library zip."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return zf.read(zf.namelist()[0]).decode("latin-1")


def parse_monthly(text: str) -> pd.DataFrame:
    """The first (monthly) table of a French CSV: header line starting with ',', then rows
    'yyyymm, v1, v2, ...' in percent, until the first non-month line (the annual table
    follows). Returns decimals indexed by monthly Period; -99.99 / -999 are missing."""
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines[:-1])
        if line.startswith(",") and MONTH_ROW.match(lines[i + 1])
    )
    columns = [c.strip() for c in lines[start].split(",")[1:]]
    rows = []
    for line in lines[start + 1 :]:
        if not MONTH_ROW.match(line):
            break
        fields = [f.strip() for f in line.split(",")]
        rows.append([fields[0], *map(float, fields[1 : len(columns) + 1])])
    frame = pd.DataFrame(rows, columns=["month", *columns])
    index = pd.PeriodIndex(pd.to_datetime(frame.pop("month"), format="%Y%m"), freq="M")
    return frame.set_axis(index).replace(list(MISSING_VALUES), float("nan")) / 100


def load_factors(
    cache_dir: Path = CACHE_DIR, session: requests.Session | None = None
) -> pd.DataFrame | None:
    """Monthly Mkt-RF, SMB, HML, RMW, CMA, RF and MOM, or None (logged) if unavailable."""
    try:
        ff5 = parse_monthly(zip_text(fetch_zip(FF5_ZIP, cache_dir, session)))
        mom = parse_monthly(zip_text(fetch_zip(MOM_ZIP, cache_dir, session)))
    except (requests.RequestException, zipfile.BadZipFile, StopIteration, ValueError,
            OSError) as exc:
        log.warning("Fama-French factors unavailable, skipping factor alphas: %s", exc)
        return None
    mom.columns = ["MOM"]
    return ff5.join(mom, how="inner")
