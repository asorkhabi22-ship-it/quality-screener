"""Point-in-time annual fundamentals panel built from SEC ``companyfacts``.

Per company:
1. ``flatten_facts``: one row per reported fact (us-gaap + dei, USD and share units).
2. Keep annual report forms (10-K, 10-K405, 10-KT, 10-K/A).
3. ``map_to_fields``: tag -> field (see ``xbrl_map``). Flow facts must span 350-380 days,
   stock facts must be instants. The period is identified by ``end``, never by ``fy``:
   ``fy`` is the fiscal year of the FILING, so a FY2023 10-K tags its FY2022 and FY2021
   comparatives fy=2023.
4. ``assign_period_end``: period_end = ``end``; cover-page share counts move to their
   filing's period end; instants at non-fiscal-year-end dates are dropped.
5. ``resolve_tags_per_filing`` + ``add_derived_fields``: one value per (field, period_end,
   filing), first tag in priority order, else a fallback formula on that same filing.
   Result: ``all_versions`` (every filing's value; needed for the restatement study).
6. ``first_reported``: per (field, period_end) the value from the EARLIEST filing.
   The tag is resolved inside each filing *before* choosing the earliest filing, so a later
   comparative under a higher-priority tag (e.g. ASC 606 revenue) cannot displace the
   originally reported number. ``drop_late_values`` then anchors each row to its main 10-K
   and sets aside values first reported more than 90 days after it -> ``pit_long``
   (set-aside values -> ``late_values``).
7. ``to_wide`` + ``add_lags``: one row per (cik, period_end) with prior-year ``*_lag1``
   values matched on dates. ``availability_date`` is the latest ``filed`` date among all
   values in the row, lags included. It is a filing date: the earliest tradable date is the
   next trading day (applied when joining prices).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from quality_screener import config
from quality_screener.edgar import EdgarClient, pad_cik
from quality_screener.xbrl_map import (
    COVER_PAGE_TAGS,
    DERIVATIONS,
    FIELDS,
    OUTPUT_FIELDS,
    all_tags,
    tag_table,
    total_equity,
)

log = logging.getLogger(__name__)

ANNUAL_FORMS = frozenset({"10-K", "10-K405", "10-KT", "10-K/A"})
AMENDMENT_FORMS = frozenset({"10-K/A"})
UNITS = frozenset({"USD", "shares"})
ANNUAL_DAYS = (350, 380)
LAG_TOLERANCE = pd.Timedelta(days=20)
IDENTITY_TOLERANCE = 0.02
SCALE_JUMP = 500.0
SCALE_CHECK_FIELDS = ("revenue", "total_assets", "current_assets", "cash", "shares_out")
MIN_PERIOD_GAP_DAYS = 300
LATE_ROW_DAYS = 90
MAX_LATE_DAYS = 90

FACT_COLUMNS = ["tag", "unit", "start", "end", "val", "accn", "fy", "fp", "form", "filed", "frame"]
VERSION_COLUMNS = ["cik", "field", "period_end", "val", "source_tag", "accn", "form", "filed"]
ROW_META = ["cik", "period_end", "fiscal_year", "availability_date", "first_filed", "accn", "form"]

PIT_PATH = config.PROCESSED_DIR / "panel_pit.parquet"
PIT_LONG_PATH = config.PROCESSED_DIR / "panel_pit_long.parquet"
ALL_VERSIONS_PATH = config.PROCESSED_DIR / "panel_all_versions.parquet"
LATE_VALUES_PATH = config.PROCESSED_DIR / "panel_late_values.parquet"


@dataclass(frozen=True)
class Panel:
    all_versions: pd.DataFrame  # long: every filing's value per (cik, field, period_end)
    pit_long: pd.DataFrame  # long: as-first-reported value per (cik, field, period_end)
    pit: pd.DataFrame  # wide: one row per (cik, period_end), plus *_lag1 columns
    late_values: pd.DataFrame  # first-reported values set aside as filed too late (audit)


# --- 1-4: facts -> mapped rows ------------------------------------------------------------


def flatten_facts(companyfacts: dict[str, Any], tags: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per reported fact (``FACT_COLUMNS``), USD and share units only.

    ``tags`` (qualified, e.g. ``"us-gaap:Assets"``) limits which concepts are flattened.
    """
    wanted = None if tags is None else set(tags)
    frames = []
    for taxonomy, concepts in companyfacts.get("facts", {}).items():
        for name, concept in concepts.items():
            tag = f"{taxonomy}:{name}"
            if wanted is not None and tag not in wanted:
                continue
            for unit, records in concept.get("units", {}).items():
                if unit in UNITS and records:
                    frames.append(pd.DataFrame.from_records(records).assign(tag=tag, unit=unit))
    facts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    facts = facts.reindex(columns=FACT_COLUMNS)
    for col in ("start", "end", "filed"):
        facts[col] = pd.to_datetime(facts[col])
    facts["val"] = facts["val"].astype(float)
    return facts


def map_to_fields(facts: pd.DataFrame) -> pd.DataFrame:
    """Attach field and tag priority; keep 350-380 day flows and instant stocks."""
    mapped = facts.merge(tag_table(), on=["tag", "unit"], how="inner")
    days = (mapped["end"] - mapped["start"]).dt.days
    annual_flow = mapped["kind"].eq("flow") & days.between(*ANNUAL_DAYS)
    instant = mapped["kind"].eq("stock") & mapped["start"].isna()
    return mapped[annual_flow | instant].reset_index(drop=True)


def assign_period_end(mapped: pd.DataFrame) -> pd.DataFrame:
    """period_end = ``end``, except cover-page share counts, which take the period end of the
    filing that reported them. Instants not dated at a fiscal year end are dropped."""
    is_flow = mapped["kind"].eq("flow")
    report_end = mapped[is_flow].groupby("accn")["end"].max().rename("_filing_end")
    filing_end = mapped.join(report_end, on="accn")["_filing_end"]
    cover = mapped["tag"].isin(COVER_PAGE_TAGS)
    period_end = mapped["end"].where(~cover, filing_end)
    keep = period_end.isin(set(mapped.loc[is_flow, "end"]))
    log.debug("dropped %d facts not dated at a fiscal year end", int((~keep).sum()))
    return mapped.assign(period_end=period_end)[keep].reset_index(drop=True)


# --- 5: one value per filing, then derivations --------------------------------------------


def resolve_tags_per_filing(mapped: pd.DataFrame, cik: str) -> pd.DataFrame:
    """One value per (field, period_end, filing): the highest-priority tag that filing used."""
    gap_to_year = ((mapped["end"] - mapped["start"]).dt.days - 365).abs()
    ranked = mapped.assign(_gap=gap_to_year).sort_values(
        ["field", "period_end", "accn", "priority", "_gap"], na_position="last"
    )
    _warn_conflicting_values(ranked, cik)
    best = ranked.drop_duplicates(["field", "period_end", "accn"])
    best = best.rename(columns={"tag": "source_tag"}).assign(cik=cik)
    return best[VERSION_COLUMNS].reset_index(drop=True)


def _warn_conflicting_values(ranked: pd.DataFrame, cik: str) -> None:
    distinct = ranked.groupby(["field", "period_end", "accn", "tag"])["val"].nunique()
    conflicts = distinct[distinct > 1]
    if len(conflicts):
        log.warning(
            "CIK %s: %d (field, period, filing, tag) groups hold conflicting values; kept the "
            "one closest to a 365-day period, e.g. %s", cik, len(conflicts), conflicts.index[0],
        )


def per_filing_wide(versions: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, period_end, filing), one column per field (all of ``FIELDS``)."""
    keys = ["cik", "period_end", "accn", "form", "filed"]
    wide = versions.pivot_table(index=keys, columns="field", values="val", aggfunc="first")
    return wide.reindex(columns=list(FIELDS))


def add_derived_fields(versions: pd.DataFrame) -> pd.DataFrame:
    """Fill fields a filing did not tag, using ``DERIVATIONS`` (in order) on that same filing's
    values. Each filled value is visible to later rules (income_cont_ops -> net_income)."""
    wide = per_filing_wide(versions)
    derived = []
    for field, rules in DERIVATIONS.items():
        for rule in rules:
            value = rule.formula(wide)
            fill = value.notna() & wide[field].isna()
            if fill.any():
                rows = value[fill].rename("val").reset_index()
                derived.append(rows.assign(field=field, source_tag=rule.label))
                wide.loc[fill, field] = value[fill]
    if not derived:
        return versions
    return pd.concat([versions, *derived], ignore_index=True)[VERSION_COLUMNS]


# --- 6-7: as-first-reported, wide, lags ---------------------------------------------------


def first_reported(versions: pd.DataFrame) -> pd.DataFrame:
    """Per (cik, field, period_end) the value from the earliest filing (rule 2). On the same
    filing date an original 10-K beats an amendment."""
    amended = versions["form"].isin(AMENDMENT_FORMS)
    ranked = versions.assign(_amended=amended).sort_values(
        ["cik", "field", "period_end", "filed", "_amended", "accn"]
    )
    first = ranked.drop_duplicates(["cik", "field", "period_end"])
    return first.drop(columns="_amended").reset_index(drop=True)


def drop_late_values(
    pit_long: pd.DataFrame, max_delay_days: int = MAX_LATE_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split first-reported values into (usable, late).

    Each (cik, period_end) row is anchored to its main filing: the one that supplied the most
    output-field values (ties: earliest). A value first filed more than ``max_delay_days``
    after that filing, typically one tagged only as a comparative in the next year's 10-K,
    was not public when the rest of the row was. Keeping it would delay the whole row by about
    a year, so it is set aside instead: NaN in the PIT panel, still in ``all_versions``, and
    returned with its delay for audit. Amendments inside the window (a 10-K/A fixing XBRL)
    are kept and move the row's availability_date to their own filing date.
    """
    anchor = _primary_filing(pit_long[pit_long["field"].isin(OUTPUT_FIELDS)])["filed"]
    anchored = pit_long.join(anchor.rename("anchor_filed"), on=["cik", "period_end"])
    delay = (anchored["filed"] - anchored["anchor_filed"]).dt.days
    # helper fields never reach the wide panel, so only output fields can delay a row
    late = (delay > max_delay_days) & pit_long["field"].isin(OUTPUT_FIELDS)
    kept = pit_long[~late.to_numpy()].reset_index(drop=True)
    dropped = anchored[late].assign(delay_days=delay[late]).reset_index(drop=True)
    return kept, dropped


def _log_late_values(late: pd.DataFrame, cik: str) -> None:
    if len(late):
        fields = late["field"].value_counts().to_dict()
        log.info(
            "CIK %s: %d value(s) first reported >%d days after their row's main 10-K left out "
            "of the PIT panel: %s", cik, len(late), MAX_LATE_DAYS, fields,
        )


def fiscal_year(period_end: pd.Series) -> pd.Series:
    """Compustat FYEAR convention: a fiscal year ending January-May is labelled with the prior
    calendar year (Walmart's year ending 2024-01-31 is fiscal 2023)."""
    return (period_end.dt.year - (period_end.dt.month < 6)).astype("int64")


def to_wide(pit_long: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, period_end): output fields, ``availability_date`` (latest filed among
    the row's values), ``first_filed`` and the primary filing (the one supplying most values)."""
    pit = pit_long[pit_long["field"].isin(OUTPUT_FIELDS)]
    keys = ["cik", "period_end"]
    values = pit.pivot(index=keys, columns="field", values="val").reindex(columns=list(OUTPUT_FIELDS))
    values.columns.name = None
    dates = pit.groupby(keys)["filed"].agg(availability_date="max", first_filed="min")
    wide = values.join(dates).join(_primary_filing(pit)[["accn", "form"]]).reset_index()
    wide["fiscal_year"] = fiscal_year(wide["period_end"])
    return wide[[*ROW_META, *OUTPUT_FIELDS]]


def _primary_filing(pit: pd.DataFrame) -> pd.DataFrame:
    """Per (cik, period_end): accn, form and filed date of the filing supplying most values."""
    counts = pit.groupby(["cik", "period_end", "accn", "form"], as_index=False).agg(
        n=("val", "size"), filed=("filed", "min")
    )
    ranked = counts.sort_values(
        ["cik", "period_end", "n", "filed"], ascending=[True, True, False, True]
    )
    best = ranked.drop_duplicates(["cik", "period_end"]).set_index(["cik", "period_end"])
    return best[["accn", "form", "filed"]]


def add_lags(wide: pd.DataFrame) -> pd.DataFrame:
    """Join the prior fiscal year as ``*_lag1``: the row whose period_end is 12 months earlier
    (+/- 20 days), matched on dates rather than row order, so a missing year leaves NaN.

    ``availability_date`` becomes the later of the row's and the lag row's, so no value in the
    row is used before it was filed.
    """
    lagged = ["period_end", "availability_date", *OUTPUT_FIELDS]
    prior = wide[["cik", *lagged]].rename(columns={c: f"{c}_lag1" for c in lagged})
    left = wide.assign(_target=wide["period_end"] - pd.DateOffset(years=1))
    merged = pd.merge_asof(
        left.sort_values("_target"),
        prior.sort_values("period_end_lag1"),
        left_on="_target",
        right_on="period_end_lag1",
        by="cik",
        direction="nearest",
        tolerance=LAG_TOLERANCE,
    )
    merged["availability_date"] = merged[["availability_date", "availability_date_lag1"]].max(axis=1)
    merged = merged.drop(columns=["_target", "availability_date_lag1"])
    return merged.sort_values(["cik", "period_end"]).reset_index(drop=True)


# --- sanity checks: log, never drop -------------------------------------------------------


def check_balance_identity(versions: pd.DataFrame, cik: str) -> None:
    """Warn where, in the first filing with a balance sheet for a period, total assets differ
    from total liabilities + total equity (incl. NCI) by more than 2%."""
    wide = per_filing_wide(versions).reset_index()
    wide = wide[wide["total_assets"].notna()]
    original = wide.sort_values("filed").drop_duplicates("period_end")
    gap = (original["total_assets"] - original["total_liabilities"] - total_equity(original)).abs()
    bad = original.loc[gap > IDENTITY_TOLERANCE * original["total_assets"].abs(), "period_end"]
    _warn_periods(cik, "assets != liabilities + equity by >2%", bad)


def check_wide(wide: pd.DataFrame, cik: str) -> None:
    """Negative revenue/assets, 1000x jumps vs prior year, near-duplicate period ends."""
    for field in ("revenue", "total_assets"):
        _warn_periods(cik, f"negative {field}", wide.loc[wide[field] < 0, "period_end"])
    for field in SCALE_CHECK_FIELDS:
        ratio = (wide[field] / wide[f"{field}_lag1"]).abs()
        jump = (ratio > SCALE_JUMP) | (ratio < 1 / SCALE_JUMP)
        _warn_periods(cik, f"{field} changed >{SCALE_JUMP:g}x vs prior year", wide.loc[jump, "period_end"])
    close = wide["period_end"].diff().dt.days < MIN_PERIOD_GAP_DAYS
    _warn_periods(cik, f"period_ends <{MIN_PERIOD_GAP_DAYS} days apart", wide.loc[close, "period_end"])
    late = (wide["availability_date"] - wide["first_filed"]).dt.days > LATE_ROW_DAYS
    _warn_periods(
        cik, f"availability >{LATE_ROW_DAYS} days after first filing (value first reported later)",
        wide.loc[late, "period_end"], level=logging.INFO,
    )


def _warn_periods(cik: str, problem: str, periods: pd.Series, level: int = logging.WARNING) -> None:
    if len(periods):
        shown = ", ".join(str(p.date()) for p in periods.iloc[:5])
        log.log(level, "CIK %s: %s: %d period(s), e.g. %s", cik, problem, len(periods), shown)


# --- entry points -------------------------------------------------------------------------


def build_panel_from_facts(companyfacts: dict[str, Any], cik: int | str) -> Panel:
    """Run the full pipeline (module docstring) on one company's companyfacts JSON."""
    cik10 = pad_cik(cik)
    facts = flatten_facts(companyfacts, tags=all_tags())
    mapped = assign_period_end(map_to_fields(facts[facts["form"].isin(ANNUAL_FORMS)]))
    if mapped.empty:
        raise ValueError(f"CIK {cik10}: no annual 10-K facts for any mapped tag")
    versions = add_derived_fields(resolve_tags_per_filing(mapped, cik10))
    check_balance_identity(versions, cik10)
    pit_long, late = drop_late_values(first_reported(versions))
    _log_late_values(late, cik10)
    pit = add_lags(to_wide(pit_long))
    check_wide(pit, cik10)
    return Panel(all_versions=versions, pit_long=pit_long, pit=pit, late_values=late)


def build_panel(cik: int | str, client: EdgarClient | None = None) -> Panel:
    """Panel for one company, reading companyfacts through the (cached) EDGAR client."""
    client = client if client is not None else EdgarClient()
    return build_panel_from_facts(client.get_company_facts(cik), cik)


def build_universe_panel(
    ciks: Sequence[str], client: EdgarClient
) -> tuple[Panel, pd.DataFrame]:
    """Panels for many companies concatenated; failures are returned, not raised."""
    parts: list[Panel] = []
    failures: list[dict[str, str]] = []
    for cik in tqdm(ciks, desc="panel", unit="firm"):
        try:
            parts.append(build_panel(cik, client))
        except Exception as exc:  # batch job: record and keep going
            log.warning("CIK %s failed: %s", cik, exc)
            failures.append({"cik": cik, "error": f"{type(exc).__name__}: {exc}"})
    panel = Panel(
        *(pd.concat([getattr(p, f.name) for p in parts], ignore_index=True) for f in fields(Panel))
    )
    return panel, pd.DataFrame(failures, columns=["cik", "error"])


def save_panel(panel: Panel, out_dir: Path = config.PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.pit.to_parquet(out_dir / PIT_PATH.name, index=False)
    panel.pit_long.to_parquet(out_dir / PIT_LONG_PATH.name, index=False)
    panel.all_versions.to_parquet(out_dir / ALL_VERSIONS_PATH.name, index=False)
    panel.late_values.to_parquet(out_dir / LATE_VALUES_PATH.name, index=False)


# --- coverage report ----------------------------------------------------------------------


def coverage_by_year(pit: pd.DataFrame, fields: Sequence[str] = OUTPUT_FIELDS) -> pd.DataFrame:
    """Percent of firm-years with a non-null value: rows = field, columns = fiscal_year."""
    share = pit[list(fields)].notna().groupby(pit["fiscal_year"]).mean()
    return (100 * share).T.round(1)


def source_tag_counts(pit_long: pd.DataFrame) -> pd.DataFrame:
    """How often each source tag supplied a field's first-reported value."""
    pit = pit_long[pit_long["field"].isin(OUTPUT_FIELDS)]
    counts = pit.groupby(["field", "source_tag"]).size().rename("n").reset_index()
    counts["share_pct"] = (100 * counts["n"] / counts.groupby("field")["n"].transform("sum")).round(1)
    return counts.sort_values(["field", "n"], ascending=[True, False], ignore_index=True)
