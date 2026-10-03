"""Structure of the tag map and the fallback derivations."""

import math

import pandas as pd
import pytest

from quality_screener import xbrl_map
from quality_screener.xbrl_map import DERIVATIONS, FIELDS, OUTPUT_FIELDS, total_equity

REQUESTED_FIELDS = {
    "revenue", "cogs", "gross_profit", "sga", "dep_amort", "ebit", "net_income",
    "income_cont_ops", "cfo", "equity_issued", "total_assets", "current_assets",
    "current_liabilities", "total_liabilities", "receivables", "ppe_net", "cash",
    "long_term_debt", "retained_earnings", "equity", "shares_out",
}


def test_all_requested_fields_are_output_fields() -> None:
    assert set(OUTPUT_FIELDS) == REQUESTED_FIELDS


@pytest.mark.parametrize("name", list(FIELDS))
def test_tags_are_qualified_and_unique(name: str) -> None:
    tags = FIELDS[name].tags
    assert tags, f"{name} has no tags"
    assert len(set(tags)) == len(tags)
    assert all(tag.split(":")[0] in {"us-gaap", "dei"} for tag in tags)


def test_revenue_priority_matches_spec() -> None:
    assert FIELDS["revenue"].tags[:5] == (
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:SalesRevenueNet",
        "us-gaap:SalesRevenueGoodsNet",
    )


def test_shares_prefer_cover_page_and_use_share_units() -> None:
    spec = FIELDS["shares_out"]
    assert spec.tags[0] == "dei:EntityCommonStockSharesOutstanding"
    assert spec.unit == "shares"


def test_derivation_inputs_and_targets_are_known_fields() -> None:
    for target, rules in DERIVATIONS.items():
        assert target in FIELDS
        for rule in rules:
            assert set(rule.inputs) <= set(FIELDS)
            assert rule.label.startswith("derived:")


def test_tag_table_priorities_follow_list_order() -> None:
    table = xbrl_map.tag_table()
    revenue = table[table["field"] == "revenue"].sort_values("priority")
    assert revenue["tag"].tolist() == list(FIELDS["revenue"].tags)
    assert revenue["priority"].tolist() == list(range(len(FIELDS["revenue"].tags)))


def frame(**cols: float) -> pd.DataFrame:
    base = {"equity": math.nan, "equity_incl_nci": math.nan, "minority_interest": math.nan}
    return pd.DataFrame([{**base, **cols}])


@pytest.mark.parametrize(
    ("cols", "expected"),
    [
        ({"equity_incl_nci": 110.0, "equity": 100.0, "minority_interest": 10.0}, 110.0),
        ({"equity": 100.0, "minority_interest": 10.0}, 110.0),
        ({"equity": 100.0}, 100.0),
        ({}, math.nan),
    ],
)
def test_total_equity(cols: dict[str, float], expected: float) -> None:
    got = total_equity(frame(**cols)).iloc[0]
    assert (math.isnan(got) and math.isnan(expected)) or got == expected


def test_long_term_debt_derivation_needs_both_inputs() -> None:
    rule = DERIVATIONS["long_term_debt"][0]
    w = pd.DataFrame({"lt_debt_total": [500.0, 500.0], "lt_debt_current": [50.0, math.nan]})
    got = rule.formula(w)
    assert got.iloc[0] == 450.0
    assert math.isnan(got.iloc[1])  # untagged current portion is not treated as 0
