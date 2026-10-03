"""Tests for restatements: finding revisions, the benign-cause screens, and filing evidence."""

from __future__ import annotations

import pandas as pd
import pytest

from quality_screener import restatements as rs

T = pd.Timestamp


def version(field: str, period: str, val: float, filed: str, accn: str, form: str = "10-K",
            tag: str = "us-gaap:Revenues", cik: str = "1") -> dict[str, object]:
    return {"cik": cik, "field": field, "period_end": T(period), "val": val, "source_tag": tag,
            "accn": accn, "form": form, "filed": T(filed)}


VERSIONS = pd.DataFrame([
    version("revenue", "2014-12-31", 100.0, "2015-02-20", "a1"),
    version("revenue", "2014-12-31", 101.0, "2016-02-20", "a2"),  # +1%: below threshold
    version("revenue", "2014-12-31", 90.0, "2017-02-20", "a3"),  # -10%: first >2% revision
    version("revenue", "2014-12-31", 80.0, "2018-02-20", "a4"),  # later, not the first
    version("total_assets", "2014-12-31", 1000.0, "2015-02-20", "a1", tag="us-gaap:Assets"),
    version("net_income", "2014-12-31", 0.0, "2015-02-20", "a1", tag="us-gaap:NetIncomeLoss"),
    version("net_income", "2014-12-31", 5.0, "2016-02-20", "a2", tag="us-gaap:NetIncomeLoss"),
])


def test_revisions_take_first_filing_beyond_threshold() -> None:
    rev = rs.revisions(VERSIONS)
    assert len(rev) == 1  # zero original net income is skipped
    row = rev.iloc[0]
    assert (row["original"], row["restated"], row["restated_accn"]) == (100.0, 90.0, "a3")
    assert row["pct_change"] == pytest.approx(-0.10)
    sized = rs.add_size(rev, VERSIONS)
    assert sized["size"].iloc[0] == pytest.approx(10 / 1000)  # |change| / total assets


def filings(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["form", "filingDate", "items"]).assign(
        accessionNumber=[f"x{i}" for i in range(len(rows))]
    )


def test_filing_evidence_reads_8k_items_and_amendments() -> None:
    ev = rs.filing_evidence(filings([
        ("8-K", "2016-11-01", "2.02,4.02,9.01"), ("8-K", "2016-12-01", "2.02,9.01"),
        ("10-K/A", "2017-01-15", ""), ("8-K", "2016-06-01", "2.01,9.01"), ("10-Q", "2016-08-01", ""),
    ]))
    assert ev["kind"].tolist() == ["8-K 2.01", "8-K 4.02", "10-K/A"]


def revision(**overrides: object) -> dict[str, object]:
    row = {"cik": "1", "field": "revenue", "period_end": T("2014-12-31"), "original": 100.0,
           "restated": 90.0, "pct_change": -0.1, "original_filed": T("2015-02-20"),
           "original_accn": "a1", "original_form": "10-K", "original_tag": "us-gaap:Revenues",
           "restated_filed": T("2016-02-20"), "restated_accn": "a2", "restated_form": "10-K",
           "restated_tag": "us-gaap:Revenues"}
    return row | overrides


NO_EVIDENCE = pd.DataFrame(columns=["cik", "date", "kind"])
NO_DISC_OPS = pd.DataFrame(columns=["cik", "accn", "end"])


def classify(rows: list[dict[str, object]], evidence: pd.DataFrame = NO_EVIDENCE,
             disc_ops: pd.DataFrame = NO_DISC_OPS) -> pd.DataFrame:
    return rs.classify(pd.DataFrame(rows), evidence, disc_ops)


def test_screens_for_benign_causes() -> None:
    out = classify(
        [revision(),
         revision(restated=100_000.0),  # x1000 scale slip
         revision(restated_tag="us-gaap:SalesRevenueNet"),
         revision(restated_accn="dis"),
         revision(restated_filed=T("2019-02-20"), period_end=T("2017-12-31")),
         revision(restated=0.0)],
        disc_ops=pd.DataFrame({"cik": ["1"], "accn": ["dis"], "end": [T("2015-01-02")]}),
    )
    assert out["screen"].tolist()[1:] == [
        "tagging", "tag change", "discontinued operations", "standards adoption", "tagging"]
    assert pd.isna(out["screen"].iloc[0]) and out["kept"].tolist() == [True] + [False] * 5


def test_acquisition_between_filings_is_screened() -> None:
    deals = pd.DataFrame({"cik": ["1", "1"], "date": [T("2015-06-01"), T("2017-01-01")],
                          "kind": ["8-K 2.01", "8-K 2.01"]})
    out = classify([revision(), revision(restated_filed=T("2015-05-01"))], deals)
    assert out["screen"].tolist()[0] == "acquisition" and pd.isna(out["screen"].iloc[1])


def test_evidence_overrides_screens() -> None:
    notice = pd.DataFrame({"cik": ["1"], "date": [T("2015-11-01")], "kind": ["8-K 4.02"]})
    out = classify([revision(restated_tag="us-gaap:SalesRevenueNet"),
                    revision(restated_form="10-K/A", restated_tag="other"),
                    revision(restated_filed=T("2017-02-20"), restated_tag="other")], notice)
    assert out["nonreliance_8k"].iloc[0] == T("2015-11-01")
    assert out["kept"].tolist() == [True, True, False]  # notice > 1 year before the third
    assert out["in_10ka"].tolist() == [False, True, False]


def test_events_group_by_revising_filing() -> None:
    rows = pd.DataFrame([revision(field="revenue"), revision(field="cfo", period_end=T("2013-12-31")),
                         revision(restated_accn="b")]).assign(size=[0.05, 0.10, 0.01])
    ev = rs.events(rows)
    assert ev["restated_accn"].tolist() == ["a2", "b"]
    assert ev.loc[0, "field"] == "cfo" and ev.loc[0, "n_items"] == 2
    assert ev.loc[0, "periods"] == "2013-12, 2014-12"


def test_xbrl_only_amendment_is_not_evidence() -> None:
    """Capital One pattern: the FY2010 10-K carried no XBRL; the 10-K/A a week later is the
    first XBRL report for 2010-12-31, so its revision of 2009 is not restatement evidence."""
    versions = pd.DataFrame([
        version("cfo", "2009-12-31", 2797.0, "2010-02-26", "k09"),
        version("cfo", "2009-12-31", 2082.0, "2011-03-07", "ka10", form="10-K/A"),
        version("cfo", "2010-12-31", 8142.0, "2011-03-07", "ka10", form="10-K/A"),
        version("cfo", "2010-12-31", 8142.0, "2012-02-29", "k11"),
        version("cfo", "2011-12-31", 100.0, "2012-02-29", "k11"),
        version("cfo", "2011-12-31", 90.0, "2012-06-01", "ka11", form="10-K/A"),  # real amendment
    ])
    assert rs.xbrl_only_amendments(versions) == {"ka10"}
    rev = rs.revisions(versions)
    out = rs.classify(rev, NO_EVIDENCE, NO_DISC_OPS, rs.xbrl_only_amendments(versions))
    assert out.set_index("restated_accn")["in_10ka"].to_dict() == {"ka10": False, "ka11": True}
