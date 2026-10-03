"""Restatement candidates: values that a later filing reports differently from the
as-first-reported value, checked against 10-K/A and 8-K item 4.02 filings.

A 10-K repeats prior years as comparatives. When a comparative differs from what was first
reported, the cause is either a restatement (an error correction: a "Big R" restatement comes
with an 8-K item 4.02 non-reliance notice and usually a 10-K/A; a "little r" revision is
corrected quietly in the next 10-K) or a benign re-presentation. Screens for the benign causes:

- ``tagging``: the revised value is zero or about 1000x, 1e6x or -1x the original (an XBRL
  scale or sign slip, not a change in the books).
- ``tag change``: the two values come from different XBRL tags, so the difference can be a
  change of scope in our tag mapping rather than in the accounts.
- ``discontinued operations``: the revising filing reports a discontinued-operations amount
  for that period, so revenue, CFO and receivables of the sold business moved out of the
  continuing lines (Net income and total assets include discontinued operations, so they are
  not screened this way.)
- ``acquisition``: the firm filed an 8-K item 2.01 (completed acquisition or disposal)
  between the two filings. Reverse mergers, SPAC mergers and combinations under common
  control replace the comparatives with another entity's history.
- ``standards adoption``: revised by a filing made in 2018-2020, within three years of the
  period end: the window of ASC 606 (revenue, full retrospective option), ASC 842 (leases)
  and ASU 2016-15 / 2016-18 (cash-flow classification), all applied to comparatives.

A screen never overrides evidence: a revision is kept whenever it was made in a 10-K/A or the
firm filed an 8-K item 4.02 in the year before the revising filing. A 10-K/A that is the first
XBRL for its report (the original 10-K carried none, e.g. Capital One's FY2010 10-K/A, filed
"solely to provide Exhibit 101" under a temporary hardship exemption) only delivers the
delayed data, so it is not evidence (``xbrl_only_amendments``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from quality_screener import panel

REVISION_FIELDS = ("revenue", "net_income", "cfo", "receivables", "total_assets")
REVISION_THRESHOLD = 0.02
SCALE_SLIPS = (1e3, 1e6, -1.0, -1e3)
SCALE_TOLERANCE = 0.01
DISC_OPS_FIELDS = {"revenue", "cfo", "receivables"}
ADOPTION_YEARS = (2018, 2020)
ADOPTION_LOOKBACK = pd.Timedelta(days=3 * 366)
EVIDENCE_LOOKBACK = pd.Timedelta(days=365)
EVIDENCE_LOOKAHEAD = pd.Timedelta(days=30)
PERIOD_TOLERANCE = pd.Timedelta(days=7)  # 52/53-week years end a few days apart
NONRELIANCE_ITEM = "4.02"
ACQUISITION_ITEM = "2.01"


def revisions(
    versions: pd.DataFrame,
    fields: Iterable[str] = REVISION_FIELDS,
    threshold: float = REVISION_THRESHOLD,
) -> pd.DataFrame:
    """Per (cik, field, period_end): the first later filing whose value differs from the
    as-first-reported value by more than ``threshold`` (relative), with both values, filings
    and tags. Zero originals are skipped (no relative change)."""
    v = versions[versions["field"].isin(list(fields))]
    keys = ["cik", "field", "period_end"]
    first = panel.first_reported(v)[[*keys, "val", "filed", "accn", "form", "source_tag"]]
    first = first.rename(columns={"val": "original_val", "filed": "original_filed",
                                  "accn": "original_accn", "form": "original_form",
                                  "source_tag": "original_tag"})
    later = v.merge(first, on=keys)
    later = later[later["accn"] != later["original_accn"]]
    base = later["original_val"].where(later["original_val"] != 0)
    later = later.assign(pct_change=(later["val"] - base) / base.abs())
    revised = later[later["pct_change"].abs() > threshold].sort_values([*keys, "filed", "accn"])
    out = revised.drop_duplicates(keys).rename(columns={
        "val": "restated", "filed": "restated_filed", "accn": "restated_accn",
        "form": "restated_form", "source_tag": "restated_tag", "original_val": "original",
    })
    cols = [*keys, "original", "restated", "pct_change", "original_filed", "original_accn",
            "original_form", "original_tag", "restated_filed", "restated_accn", "restated_form",
            "restated_tag"]
    return out[cols].reset_index(drop=True)


def filing_evidence(filings: pd.DataFrame) -> pd.DataFrame:
    """A firm's 10-K/A filings and its 8-Ks (or 8-K/As) listing item 4.02 (non-reliance) or
    2.01 (completed acquisition or disposal), from the submissions data: ``date``, ``form``,
    ``accn``, ``items`` and ``kind`` ('10-K/A', '8-K 4.02' or '8-K 2.01')."""
    items = filings["items"].fillna("").astype(str).str.split(",").map(
        lambda parts: {p.strip() for p in parts}
    )
    is_8k = filings["form"].isin(["8-K", "8-K/A"])
    kind = np.select(
        [filings["form"].isin(panel.AMENDMENT_FORMS),
         is_8k & items.map(lambda i: NONRELIANCE_ITEM in i),
         is_8k & items.map(lambda i: ACQUISITION_ITEM in i)],
        ["10-K/A", "8-K 4.02", "8-K 2.01"], "",
    )
    out = filings.assign(kind=kind, date=pd.to_datetime(filings["filingDate"]))
    out = out[out["kind"] != ""].rename(columns={"accessionNumber": "accn"})
    return out[["date", "form", "accn", "items", "kind"]].sort_values("date", ignore_index=True)


def discontinued_ops_periods(companyfacts: dict[str, Any]) -> pd.DataFrame:
    """(accn, end) pairs where a filing reports a non-zero us-gaap discontinued-operations
    amount (any concept whose name contains 'DiscontinuedOperation')."""
    rows = []
    for name, concept in companyfacts.get("facts", {}).get("us-gaap", {}).items():
        if "DiscontinuedOperation" not in name:
            continue
        for records in concept.get("units", {}).values():
            rows += [(r["accn"], r["end"]) for r in records if r.get("val")]
    out = pd.DataFrame(rows, columns=["accn", "end"]).drop_duplicates()
    return out.assign(end=pd.to_datetime(out["end"]))


def xbrl_only_amendments(versions: pd.DataFrame) -> set[str]:
    """Accession numbers of 10-K/As that are the first XBRL report for their fiscal year:
    no earlier original 10-K of the firm reported values for the amendment's latest period
    end."""
    amendments = versions[versions["form"].isin(panel.AMENDMENT_FORMS)]
    report = amendments.groupby(["cik", "accn"]).agg(
        period=("period_end", "max"), filed=("filed", "min")
    )
    is_original = ~versions["form"].isin(panel.AMENDMENT_FORMS)
    originals = versions.loc[is_original, ["cik", "period_end", "filed"]]
    pairs = report.reset_index().merge(originals, left_on=["cik", "period"],
                                       right_on=["cik", "period_end"], suffixes=("", "_orig"))
    covered = set(pairs.loc[pairs["filed_orig"] <= pairs["filed"], "accn"])
    return set(report.index.get_level_values("accn")) - covered


def _dated(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """``column`` as datetimes (an empty frame's column is otherwise untyped)."""
    return frame.assign(**{column: pd.to_datetime(frame[column])})


def _near_slip(ratio: pd.Series) -> pd.Series:
    hit = pd.Series(False, index=ratio.index)
    for slip in SCALE_SLIPS:
        hit |= (ratio / slip - 1).abs() < SCALE_TOLERANCE
    return hit


def _nonreliance_date(rev: pd.DataFrame, evidence: pd.DataFrame) -> pd.Series:
    """Latest 8-K item 4.02 filed in [revising filing - 1 year, revising filing + 30 days]."""
    notices = _dated(evidence[evidence["kind"] == "8-K 4.02"][["cik", "date"]], "date")
    pairs = rev[["cik", "restated_filed"]].reset_index().merge(notices, on="cik")
    window = (pairs["date"] >= pairs["restated_filed"] - EVIDENCE_LOOKBACK) & (
        pairs["date"] <= pairs["restated_filed"] + EVIDENCE_LOOKAHEAD
    )
    return pairs[window].groupby("index")["date"].max().reindex(rev.index)


def _acquisition_between(rev: pd.DataFrame, evidence: pd.DataFrame) -> pd.Series:
    """True where an 8-K item 2.01 was filed after the original and up to the revising filing."""
    deals = _dated(evidence[evidence["kind"] == "8-K 2.01"][["cik", "date"]], "date")
    pairs = rev[["cik", "original_filed", "restated_filed"]].reset_index().merge(deals, on="cik")
    between = (pairs["date"] > pairs["original_filed"]) & (pairs["date"] <= pairs["restated_filed"])
    return pairs[between].groupby("index").size().gt(0).reindex(rev.index, fill_value=False)


def _discontinued(rev: pd.DataFrame, disc_ops: pd.DataFrame) -> pd.Series:
    """True where the revising filing reports discontinued operations for that period."""
    pairs = rev[["cik", "restated_accn", "period_end"]].reset_index().merge(
        _dated(disc_ops, "end").rename(columns={"accn": "restated_accn"}),
        on=["cik", "restated_accn"],
    )
    close = (pairs["end"] - pairs["period_end"]).abs() <= PERIOD_TOLERANCE
    hit = pairs[close].groupby("index").size() > 0
    return hit.reindex(rev.index, fill_value=False) & rev["field"].isin(DISC_OPS_FIELDS)


def classify(
    rev: pd.DataFrame, evidence: pd.DataFrame, disc_ops: pd.DataFrame,
    xbrl_only: Iterable[str] = (),
) -> pd.DataFrame:
    """Add ``nonreliance_8k`` (date), ``in_10ka``, ``evidence``, ``screen`` (first matching
    benign cause, or NA) and ``kept`` (evidence, or no screen matched).

    ``evidence`` and ``disc_ops`` are per-firm outputs of ``filing_evidence`` and
    ``discontinued_ops_periods`` stacked with a ``cik`` column; ``xbrl_only`` lists 10-K/As
    that do not count as evidence (``xbrl_only_amendments``)."""
    out = rev.copy()
    out["nonreliance_8k"] = _nonreliance_date(out, evidence)
    out["in_10ka"] = out["restated_form"].isin(panel.AMENDMENT_FORMS) & ~out[
        "restated_accn"
    ].isin(set(xbrl_only))
    out["evidence"] = out["in_10ka"] | out["nonreliance_8k"].notna()
    year = out["restated_filed"].dt.year
    in_window = year.between(*ADOPTION_YEARS) & (
        out["restated_filed"] - out["period_end"] <= ADOPTION_LOOKBACK
    )
    screens = [
        ("tagging", _near_slip(out["restated"] / out["original"]) | (out["restated"] == 0)),
        ("tag change", out["original_tag"] != out["restated_tag"]),
        ("discontinued operations", _discontinued(out, disc_ops)),
        ("acquisition", _acquisition_between(out, evidence)),
        ("standards adoption", in_window),
    ]
    screen = pd.Series(pd.NA, index=out.index, dtype="string")
    for label, hit in reversed(screens):  # the first listed wins
        screen = screen.mask(hit, label)
    out["screen"] = screen
    out["kept"] = out["evidence"] | out["screen"].isna()
    return out


def add_size(rev: pd.DataFrame, versions: pd.DataFrame) -> pd.DataFrame:
    """``size`` = |restated - original| / total assets as first reported for that period
    (materiality on a common scale; a % change of a near-zero net income is meaningless)."""
    assets = panel.first_reported(versions[versions["field"] == "total_assets"])
    assets = assets.set_index(["cik", "period_end"])["val"].rename("total_assets_original")
    out = rev.join(assets, on=["cik", "period_end"])
    out["size"] = (out["restated"] - out["original"]).abs() / out["total_assets_original"].where(
        out["total_assets_original"] > 0
    )
    return out


def events(rev: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, revising filing): the largest revision (by ``size``) with the
    number of items and the fields and periods it revised, ranked by size."""
    rev = rev.sort_values("size", ascending=False, na_position="last")
    grouped = rev.groupby(["cik", "restated_accn"], sort=False)
    top = grouped.head(1).set_index(["cik", "restated_accn"])
    summary = grouped.agg(
        n_items=("field", "size"),
        fields=("field", lambda f: ", ".join(sorted(set(f)))),
        periods=("period_end", lambda p: ", ".join(sorted({f"{d:%Y-%m}" for d in p}))),
    )
    return top.join(summary).reset_index().sort_values("size", ascending=False, ignore_index=True)
