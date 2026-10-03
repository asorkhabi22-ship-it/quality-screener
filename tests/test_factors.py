"""Tests for factors: parsing Ken French's CSV layout, caching, and skipping cleanly."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from quality_screener import factors
from tests.fakes import FakeResponse, FakeSession

FF5_TEXT = """This file was created using the 202608 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RMW,CMA,RF
202606,  -1.07,   4.36,   3.58,  -5.07,   3.45,   0.29
202607,  -0.61,  -0.09, -99.99,  11.07,   2.26,   0.33

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
  2025,  10.00,   1.00,   2.00,   3.00,   4.00,   4.50
"""
MOM_TEXT = """This file was created by CMPT_ME_PRIOR_RETS using the 202608 CRSP database.

,Mom
202606,   4.62
202607, -12.20

 Annual Factors:
,Mom
  2025,   5.00
"""


def zipped(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.csv", text)
    return buf.getvalue()


def test_parse_monthly_reads_only_the_monthly_table() -> None:
    out = factors.parse_monthly(FF5_TEXT)
    assert out.columns.tolist() == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert out.index.tolist() == [pd.Period("2026-06", "M"), pd.Period("2026-07", "M")]
    assert out.loc[pd.Period("2026-06", "M"), "Mkt-RF"] == pytest.approx(-0.0107)
    assert np.isnan(out.loc[pd.Period("2026-07", "M"), "HML"])  # -99.99 = missing


def test_load_factors_downloads_once_and_joins_momentum(tmp_path: Path) -> None:
    session = FakeSession({
        factors.FRENCH_URL + factors.FF5_ZIP: [FakeResponse(200, zipped(FF5_TEXT))],
        factors.FRENCH_URL + factors.MOM_ZIP: [FakeResponse(200, zipped(MOM_TEXT))],
    })
    out = factors.load_factors(tmp_path, session)
    assert out is not None and out.columns.tolist()[-1] == "MOM"
    assert out.loc[pd.Period("2026-07", "M"), "MOM"] == pytest.approx(-0.122)
    again = factors.load_factors(tmp_path, FakeSession({}))  # served from the cache
    assert again is not None and again.equals(out)


def test_load_factors_returns_none_when_download_fails(tmp_path: Path) -> None:
    session = FakeSession({factors.FRENCH_URL + factors.FF5_ZIP: [requests.ConnectionError("down")]})
    assert factors.load_factors(tmp_path, session) is None
    assert not list(tmp_path.iterdir())  # nothing half-written is cached
