"""Panel construction from hand-built companyfacts dicts. No network, no cache."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import pandas as pd
import pytest

from quality_screener import panel
from quality_screener.panel import Panel, build_panel_from_facts

CIK = "0000000001"
TS = pd.Timestamp


def fact(
    val: float,
    end: str,
    filed: str,
    accn: str,
    start: str | None = None,
    form: str = "10-K",
    fy: int | None = None,
) -> dict[str, Any]:
    record = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": "FY", "form": form, "filed": filed}
    if start is not None:
        record["start"] = start
    return record


def companyfacts(*entries: tuple[str, dict[str, Any]], unit: str = "USD") -> dict[str, Any]:
    """entries: (qualified tag, fact record). dei share counts use the 'shares' unit."""
    facts: dict[str, dict[str, Any]] = defaultdict(dict)
    for tag, record in entries:
        taxonomy, name = tag.split(":")
        rec_unit = "shares" if "Shares" in name else unit
        concept = facts[taxonomy].setdefault(name, {"units": {}})
        concept["units"].setdefault(rec_unit, []).append(record)
    return {"cik": 1, "entityName": "Test Co", "facts": dict(facts)}


def build(*entries: tuple[str, dict[str, Any]]) -> Panel:
    return build_panel_from_facts(companyfacts(*entries), CIK)


def pit_value(p: Panel, field: str, period_end: str) -> pd.Series:
    rows = p.pit_long[(p.pit_long["field"] == field) & (p.pit_long["period_end"] == TS(period_end))]
    assert len(rows) == 1
    return rows.iloc[0]


FY22 = dict(start="2022-01-01", end="2022-12-31")
FY23 = dict(start="2023-01-01", end="2023-12-31")
A22 = dict(filed="2023-02-01", accn="A22")
A23 = dict(filed="2024-02-01", accn="A23")


# --- the requested scenario ---------------------------------------------------------------


@pytest.fixture
def restated() -> Panel:
    return build(
        ("us-gaap:Revenues", fact(100, **FY22, **A22, fy=2022)),
        ("us-gaap:Revenues", fact(30, start="2022-10-01", end="2022-12-31", **A22, fy=2022)),
        ("us-gaap:Revenues", fact(95, **FY22, **A23, fy=2023)),  # restated comparative
        ("us-gaap:Revenues", fact(120, **FY23, **A23, fy=2023)),
    )


def test_pit_keeps_first_reported_value_and_its_filing_date(restated: Panel) -> None:
    row = restated.pit.set_index("period_end").loc[TS("2022-12-31")]
    assert row["revenue"] == 100
    assert row["availability_date"] == TS("2023-02-01")
    assert row["accn"] == "A22"


def test_all_versions_keeps_original_and_restatement(restated: Panel) -> None:
    v = restated.all_versions
    fy22 = v[(v["field"] == "revenue") & (v["period_end"] == TS("2022-12-31"))]
    assert sorted(zip(fy22["filed"], fy22["val"])) == [(TS("2023-02-01"), 100), (TS("2024-02-01"), 95)]


def test_quarterly_fact_is_excluded(restated: Panel) -> None:
    assert 30 not in restated.all_versions["val"].tolist()


def test_period_comes_from_end_date_not_fy(restated: Panel) -> None:
    # the 95 was tagged fy=2023 but its period ends 2022-12-31
    v = restated.all_versions
    assert v.loc[v["val"] == 95, "period_end"].tolist() == [TS("2022-12-31")]


def test_lag_comes_from_the_same_filing_as_the_current_year(restated: Panel) -> None:
    """FY2023's 10-K (A23) reports FY2022 revenue restated to 95: the growth ratio compares
    120 with 95, both from A23, not with the 100 first reported a year earlier. The FY2022
    row itself keeps its first-reported 100."""
    by_end = restated.pit.set_index("period_end")
    row = by_end.loc[TS("2023-12-31")]
    assert row["revenue_lag1"] == 95
    assert row["period_end_lag1"] == TS("2022-12-31")
    assert row["lag1_same_filing"] == 1
    assert by_end.loc[TS("2022-12-31"), "revenue"] == 100


def test_lag_falls_back_to_prior_row_when_filing_has_no_comparative() -> None:
    p = build(
        ("us-gaap:Revenues", fact(100, **FY22, **A22)),
        ("us-gaap:NetIncomeLoss", fact(7, **FY22, **A22)),
        ("us-gaap:Revenues", fact(95, **FY22, **A23)),  # A23 repeats revenue, not net income
        ("us-gaap:Revenues", fact(120, **FY23, **A23)),
        ("us-gaap:NetIncomeLoss", fact(9, **FY23, **A23)),
    )
    row = p.pit.set_index("period_end").loc[TS("2023-12-31")]
    assert (row["revenue_lag1"], row["net_income_lag1"]) == (95, 7)


def test_without_versions_lags_are_the_prior_rows_first_reported_values(restated: Panel) -> None:
    wide = panel.to_wide(restated.pit_long)
    row = panel.add_lags(wide).set_index("period_end").loc[TS("2023-12-31")]
    assert row["revenue_lag1"] == 100 and "lag1_same_filing" not in row


# --- tag priority vs filing order ---------------------------------------------------------


def test_earliest_filing_beats_higher_priority_tag_in_later_filing() -> None:
    """ASC 606: FY2017 first reported as SalesRevenueNet; the FY2018 10-K repeats it under a
    higher-priority tag. The original number and tag must win."""
    fy17 = dict(start="2016-10-01", end="2017-09-30")
    p = build(
        ("us-gaap:SalesRevenueNet", fact(229.0, **fy17, filed="2017-11-03", accn="A17")),
        ("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
         fact(229.2, **fy17, filed="2018-11-05", accn="A18")),
    )
    row = pit_value(p, "revenue", "2017-09-30")
    assert (row["val"], row["source_tag"], row["accn"]) == (229.0, "us-gaap:SalesRevenueNet", "A17")


def test_within_one_filing_first_tag_in_list_wins() -> None:
    p = build(
        ("us-gaap:SalesRevenueNet", fact(9, **FY22, **A22)),
        ("us-gaap:Revenues", fact(10, **FY22, **A22)),
    )
    row = pit_value(p, "revenue", "2022-12-31")
    assert (row["val"], row["source_tag"]) == (10, "us-gaap:Revenues")


def test_amendment_never_beats_original_and_10q_is_ignored() -> None:
    p = build(
        ("us-gaap:Revenues", fact(100, **FY22, filed="2023-02-01", accn="A22")),
        ("us-gaap:Revenues", fact(97, **FY22, filed="2023-02-01", accn="A22A", form="10-K/A")),
        ("us-gaap:Revenues", fact(1, **FY22, filed="2022-11-01", accn="Q3", form="10-Q")),
    )
    assert pit_value(p, "revenue", "2022-12-31")["accn"] == "A22"
    assert "Q3" not in p.all_versions["accn"].tolist()


# --- derivations --------------------------------------------------------------------------


def test_gross_profit_derived_in_original_filing_beats_later_tag() -> None:
    p = build(
        ("us-gaap:Revenues", fact(100, **FY22, **A22)),
        ("us-gaap:CostOfRevenue", fact(60, **FY22, **A22)),
        ("us-gaap:GrossProfit", fact(40, **FY22, **A23)),  # tagged only a year later
    )
    row = pit_value(p, "gross_profit", "2022-12-31")
    assert (row["val"], row["source_tag"], row["accn"]) == (40, "derived:revenue-cogs", "A22")
    assert p.pit["availability_date"].iloc[0] == TS("2023-02-01")


def test_total_liabilities_from_identity_includes_minority_interest() -> None:
    end = dict(end="2022-12-31", **A22)
    p = build(
        ("us-gaap:Revenues", fact(1, **FY22, **A22)),
        ("us-gaap:Assets", fact(1000, **end)),
        ("us-gaap:LiabilitiesAndStockholdersEquity", fact(1000, **end)),
        ("us-gaap:StockholdersEquity", fact(300, **end)),
        ("us-gaap:MinorityInterest", fact(50, **end)),
    )
    row = pit_value(p, "total_liabilities", "2022-12-31")
    assert (row["val"], row["source_tag"]) == (650, "derived:LiabilitiesAndStockholdersEquity-total_equity")


def test_long_term_debt_stays_nan_without_current_portion() -> None:
    end = dict(end="2022-12-31", **A22)
    p = build(
        ("us-gaap:Revenues", fact(1, **FY22, **A22)),
        ("us-gaap:LongTermDebt", fact(500, **end)),
    )
    assert math.isnan(p.pit["long_term_debt"].iloc[0])


def test_net_income_from_profit_loss_minus_nci() -> None:
    p = build(
        ("us-gaap:Revenues", fact(100, **FY22, **A22)),
        ("us-gaap:ProfitLoss", fact(12, **FY22, **A22)),
        ("us-gaap:NetIncomeLossAttributableToNoncontrollingInterest", fact(2, **FY22, **A22)),
    )
    ni = pit_value(p, "net_income", "2022-12-31")
    assert (ni["val"], ni["source_tag"]) == (
        10, "derived:ProfitLoss-NetIncomeLossAttributableToNoncontrollingInterest"
    )
    # income from continuing ops falls back to the (derived) net income of the same filing
    assert pit_value(p, "income_cont_ops", "2022-12-31")["val"] == 10


def test_tagged_net_income_beats_any_derivation() -> None:
    p = build(
        ("us-gaap:NetIncomeLoss", fact(9, **FY22, **A22)),
        ("us-gaap:ProfitLoss", fact(12, **FY22, **A22)),
    )
    assert pit_value(p, "net_income", "2022-12-31")["source_tag"] == "us-gaap:NetIncomeLoss"


def test_sga_from_selling_plus_general_only_when_both_tagged() -> None:
    p = build(
        ("us-gaap:Revenues", fact(100, **FY22, **A22)),
        ("us-gaap:SellingAndMarketingExpense", fact(7, **FY22, **A22)),
        ("us-gaap:GeneralAndAdministrativeExpense", fact(3, **FY22, **A22)),
        ("us-gaap:Revenues", fact(120, **FY23, **A23)),
        ("us-gaap:GeneralAndAdministrativeExpense", fact(4, **FY23, **A23)),  # no selling line
    )
    sga = p.pit.set_index("fiscal_year")["sga"]
    assert sga[2022] == 10
    assert math.isnan(sga[2023])


# --- stocks, cover page, odd dates --------------------------------------------------------


def test_cover_page_shares_attach_to_filing_period_end() -> None:
    p = build(
        ("us-gaap:Revenues", fact(100, **FY22, **A22)),
        ("dei:EntityCommonStockSharesOutstanding", fact(5_000, end="2023-01-20", **A22)),
    )
    row = pit_value(p, "shares_out", "2022-12-31")
    assert (row["val"], row["source_tag"]) == (5_000, "dei:EntityCommonStockSharesOutstanding")


def test_instants_not_at_a_fiscal_year_end_are_dropped() -> None:
    p = build(
        ("us-gaap:Revenues", fact(100, **FY22, **A22)),
        ("us-gaap:Assets", fact(900, end="2022-12-31", **A22)),
        ("us-gaap:Assets", fact(777, end="2022-06-30", **A22)),  # mid-year instant
    )
    assert p.pit["period_end"].tolist() == [TS("2022-12-31")]
    assert 777 not in p.all_versions["val"].tolist()


# --- wide panel: lags and availability ----------------------------------------------------


def annual_revenue(year: int, val: float, filed: str, accn: str) -> tuple[str, dict[str, Any]]:
    return ("us-gaap:Revenues", fact(val, start=f"{year}-01-01", end=f"{year}-12-31", filed=filed, accn=accn))


def test_lag_join_uses_dates_not_row_order() -> None:
    p = build(
        annual_revenue(2019, 10, "2020-02-01", "A19"),
        annual_revenue(2020, 20, "2021-02-01", "A20"),
        annual_revenue(2022, 40, "2023-02-01", "A22"),  # FY2021 missing entirely
    )
    lags = p.pit.set_index("fiscal_year")["revenue_lag1"]
    assert lags[2020] == 10
    assert math.isnan(lags[2022])  # not FY2020's 20


def test_lag_matches_53_week_year_within_tolerance() -> None:
    p = build(
        ("us-gaap:Revenues", fact(1, start="2021-09-26", end="2022-09-24", filed="2022-10-28", accn="A")),
        ("us-gaap:Revenues", fact(2, start="2022-09-25", end="2023-09-30", filed="2023-11-03", accn="B")),
    )
    assert p.pit.set_index("fiscal_year").loc[2023, "revenue_lag1"] == 1


def original_10k(year: int, filed: str, accn: str, revenue: float) -> list[tuple[str, dict[str, Any]]]:
    """A 10-K reporting revenue, CFO and assets for ``year``."""
    period = dict(start=f"{year}-01-01", end=f"{year}-12-31", filed=filed, accn=accn)
    return [
        ("us-gaap:Revenues", fact(revenue, **period)),
        ("us-gaap:NetCashProvidedByUsedInOperatingActivities", fact(revenue / 10, **period)),
        ("us-gaap:Assets", fact(revenue * 2, end=f"{year}-12-31", filed=filed, accn=accn)),
    ]


def test_value_first_reported_a_year_later_is_left_out_not_delaying_the_row() -> None:
    late_ni = ("us-gaap:NetIncomeLoss",
               fact(1, start="2021-01-01", end="2021-12-31", filed="2023-02-01", accn="A22"))
    p = build(*original_10k(2021, "2022-02-01", "A21", 10), late_ni,
              *original_10k(2022, "2023-02-01", "A22", 20))
    fy21 = p.pit.set_index("fiscal_year").loc[2021]
    assert fy21["availability_date"] == TS("2022-02-01")  # not 2023-02-01
    assert math.isnan(fy21["net_income"])
    # set aside for audit, and still present in all_versions
    late = p.late_values
    assert set(late["field"]) == {"net_income", "income_cont_ops"}
    assert (late["delay_days"] == 365).all()
    v = p.all_versions
    assert v.loc[(v["field"] == "net_income") & (v["accn"] == "A22"), "period_end"].tolist() == [TS("2021-12-31")]


def test_amendment_within_90_days_is_kept_and_moves_availability() -> None:
    amended_ni = ("us-gaap:NetIncomeLoss",
                  fact(1, start="2021-01-01", end="2021-12-31", filed="2022-03-15",
                       accn="A21A", form="10-K/A"))
    p = build(*original_10k(2021, "2022-02-01", "A21", 10), amended_ni)
    row = p.pit.iloc[0]
    assert row["net_income"] == 1
    assert row["availability_date"] == TS("2022-03-15")
    assert p.late_values.empty


def test_availability_includes_the_lag_row() -> None:
    """Prior row (late filer) gets a 10-K/A after the current 10-K: the current row can only
    be used once its lag values are public."""
    amended = ("us-gaap:NetIncomeLoss",
               fact(1, start="2021-01-01", end="2021-12-31", filed="2023-01-30",
                    accn="A21A", form="10-K/A"))
    p = build(*original_10k(2021, "2022-12-01", "A21", 10), amended,
              *original_10k(2022, "2023-01-15", "A22", 20))
    by_year = p.pit.set_index("fiscal_year")
    assert by_year.loc[2021, "availability_date"] == TS("2023-01-30")
    assert by_year.loc[2022, "availability_date"] == TS("2023-01-30")  # via the lag row


@pytest.mark.parametrize(
    ("end", "expected"), [("2023-09-30", 2023), ("2024-01-31", 2023), ("2023-06-30", 2023)]
)
def test_fiscal_year_convention(end: str, expected: int) -> None:
    assert panel.fiscal_year(pd.Series([TS(end)])).iloc[0] == expected


# --- sanity checks ------------------------------------------------------------------------


def test_sanity_checks_log_warnings_without_dropping(caplog: pytest.LogCaptureFixture) -> None:
    end22 = dict(end="2022-12-31", **A22)
    with caplog.at_level(logging.WARNING, logger="quality_screener.panel"):
        p = build(
            ("us-gaap:Revenues", fact(-5, **FY22, **A22)),
            ("us-gaap:Revenues", fact(-5_000, **FY23, **A23)),
            ("us-gaap:Assets", fact(1000, **end22)),
            ("us-gaap:Liabilities", fact(500, **end22)),
            ("us-gaap:StockholdersEquity", fact(300, **end22)),  # 1000 != 800
            ("us-gaap:Assets", fact(1_000_000, end="2023-12-31", **A23)),  # 1000x
            ("us-gaap:Revenues", fact(7, start="2023-03-01", end="2024-02-28", filed="2024-04-01", accn="AT")),
        )
    text = caplog.text
    assert "assets != liabilities + equity" in text
    assert "negative revenue" in text
    assert "total_assets changed >500x" in text
    assert "period_ends <300 days apart" in text
    assert len(p.pit) == 3  # nothing dropped


def test_company_without_annual_facts_raises() -> None:
    with pytest.raises(ValueError, match="no annual 10-K facts"):
        build(("us-gaap:Revenues", fact(1, start="2022-01-01", end="2022-03-31", filed="2022-05-01", accn="Q", form="10-Q")))


# --- universe + coverage ------------------------------------------------------------------


class FakeFactsClient:
    def __init__(self, facts_by_cik: dict[str, dict[str, Any]]) -> None:
        self.facts_by_cik = facts_by_cik

    def get_company_facts(self, cik: str) -> dict[str, Any]:
        return self.facts_by_cik[cik]


def test_universe_panel_collects_failures_and_reports_coverage() -> None:
    good = companyfacts(
        annual_revenue(2021, 10, "2022-02-01", "A21"),
        ("us-gaap:NetIncomeLoss", fact(1, start="2022-01-01", end="2022-12-31", **A22)),
        annual_revenue(2022, 20, "2023-02-01", "A22"),
    )
    client = FakeFactsClient({"0000000001": good, "0000000002": {"facts": {}}})
    p, failures = panel.build_universe_panel(["0000000001", "0000000002"], client)  # type: ignore[arg-type]

    assert failures["cik"].tolist() == ["0000000002"]
    cov = panel.coverage_by_year(p.pit, fields=["revenue", "net_income"])
    assert cov.loc["revenue"].tolist() == [100.0, 100.0]
    assert cov.loc["net_income"].tolist() == [0.0, 100.0]
    tags = panel.source_tag_counts(p.pit_long)
    assert tags.set_index("field").loc["revenue", "source_tag"] == "us-gaap:Revenues"
