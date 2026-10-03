"""Standard fields -> ordered lists of XBRL tags, plus fallback derivations.

Companies tag the same concept differently, and tags changed over time (revenue moved to the
ASC 606 ``RevenueFromContractWithCustomer...`` tags around 2018). For each field the tags are
tried in order *within one filing*; the first one that filing reported wins and is recorded as
``source_tag``. If none of the tags is present, the field's ``DERIVATIONS`` are tried in order
on the other fields of the same filing; their label becomes the ``source_tag``.

Tags are qualified ``taxonomy:Name``. ``helper`` fields only feed derivations and checks and do
not appear in the wide panel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Kind = Literal["flow", "stock"]


@dataclass(frozen=True)
class FieldSpec:
    kind: Kind  # flow = annual duration fact; stock = balance-sheet instant
    tags: tuple[str, ...]  # priority order, first reported wins
    unit: str = "USD"
    helper: bool = False


@dataclass(frozen=True)
class Derivation:
    label: str  # recorded as source_tag
    inputs: tuple[str, ...]
    formula: Callable[[pd.DataFrame], pd.Series]


def _gaap(*names: str) -> tuple[str, ...]:
    return tuple(f"us-gaap:{name}" for name in names)


FIELDS: dict[str, FieldSpec] = {
    # --- flow items (annual duration) ---
    "revenue": FieldSpec("flow", _gaap(
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    )),
    "cogs": FieldSpec("flow", _gaap(
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices",
    )),
    "gross_profit": FieldSpec("flow", _gaap("GrossProfit")),
    "sga": FieldSpec("flow", _gaap("SellingGeneralAndAdministrativeExpense")),
    "dep_amort": FieldSpec("flow", _gaap(
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    )),
    "ebit": FieldSpec("flow", _gaap("OperatingIncomeLoss")),
    "net_income": FieldSpec("flow", _gaap("NetIncomeLoss")),
    "income_cont_ops": FieldSpec("flow", _gaap(
        "IncomeLossFromContinuingOperations", "NetIncomeLoss",
    )),
    "cfo": FieldSpec("flow", _gaap(
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    )),
    "equity_issued": FieldSpec("flow", _gaap("ProceedsFromIssuanceOfCommonStock")),
    # --- stock items (balance-sheet instants) ---
    "total_assets": FieldSpec("stock", _gaap("Assets")),
    "current_assets": FieldSpec("stock", _gaap("AssetsCurrent")),
    "current_liabilities": FieldSpec("stock", _gaap("LiabilitiesCurrent")),
    "total_liabilities": FieldSpec("stock", _gaap("Liabilities")),
    "receivables": FieldSpec("stock", _gaap(
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsNotesAndLoansReceivableNetCurrent",
        "AccountsAndOtherReceivablesNetCurrent",
    )),
    "ppe_net": FieldSpec("stock", _gaap(
        "PropertyPlantAndEquipmentNet",
        # ASC 842 (2019+): owned PP&E plus finance-lease right-of-use assets, net
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    )),
    "cash": FieldSpec("stock", _gaap(
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    )),
    "long_term_debt": FieldSpec("stock", _gaap(
        "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",  # both exclude current
    )),
    "retained_earnings": FieldSpec("stock", _gaap("RetainedEarningsAccumulatedDeficit")),
    "equity": FieldSpec("stock", _gaap("StockholdersEquity")),
    "shares_out": FieldSpec(
        "stock",
        ("dei:EntityCommonStockSharesOutstanding", "us-gaap:CommonStockSharesOutstanding"),
        unit="shares",
    ),
    # --- helpers: inputs to derivations and the balance-sheet identity check ---
    "liab_and_equity": FieldSpec("stock", _gaap("LiabilitiesAndStockholdersEquity"), helper=True),
    "equity_incl_nci": FieldSpec(
        "stock",
        _gaap("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        helper=True,
    ),
    "minority_interest": FieldSpec("stock", _gaap("MinorityInterest"), helper=True),
    "lt_debt_total": FieldSpec("stock", _gaap("LongTermDebt"), helper=True),
    "lt_debt_current": FieldSpec("stock", _gaap("LongTermDebtCurrent"), helper=True),
    "ltd_lease_total": FieldSpec(
        "stock", _gaap("LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"), helper=True
    ),
    "ltd_lease_current": FieldSpec(
        "stock", _gaap("LongTermDebtAndCapitalLeaseObligationsCurrent"), helper=True
    ),
    "selling_marketing": FieldSpec("flow", _gaap("SellingAndMarketingExpense"), helper=True),
    "general_admin": FieldSpec("flow", _gaap("GeneralAndAdministrativeExpense"), helper=True),
    "profit_loss": FieldSpec("flow", _gaap("ProfitLoss"), helper=True),  # incl. NCI
    "nci_income": FieldSpec(
        "flow", _gaap("NetIncomeLossAttributableToNoncontrollingInterest"), helper=True
    ),
    "ni_to_common": FieldSpec(
        "flow", _gaap("NetIncomeLossAvailableToCommonStockholdersBasic"), helper=True
    ),
}

# The cover-page share count is dated "as of" a day near the filing date, not the fiscal
# period end, so panel.py attaches it to the period end of the filing that reported it.
COVER_PAGE_TAGS = frozenset({"dei:EntityCommonStockSharesOutstanding"})


def total_equity(wide: pd.DataFrame) -> pd.Series:
    """Equity including non-controlling interest: the tagged total, else parent equity plus a
    separately tagged minority interest, else parent equity alone (no NCI tagged)."""
    with_nci = wide["equity"] + wide["minority_interest"]
    return wide["equity_incl_nci"].fillna(with_nci).fillna(wide["equity"])


# Per field, tried in order after the tags, each only where the field is still missing.
# Treating an untagged item as 0 would be a fill, so every formula needs all of its inputs,
# except the explicit "no NCI tagged" cases (NCI is a required face-statement line, ASC 810).
DERIVATIONS: dict[str, tuple[Derivation, ...]] = {
    "gross_profit": (
        Derivation("derived:revenue-cogs", ("revenue", "cogs"), lambda w: w["revenue"] - w["cogs"]),
    ),
    "sga": (
        Derivation(
            "derived:SellingAndMarketingExpense+GeneralAndAdministrativeExpense",
            ("selling_marketing", "general_admin"),
            lambda w: w["selling_marketing"] + w["general_admin"],
        ),
    ),
    "net_income": (
        Derivation(
            "derived:ProfitLoss-NetIncomeLossAttributableToNoncontrollingInterest",
            ("profit_loss", "nci_income"),
            lambda w: w["profit_loss"] - w["nci_income"],
        ),
        Derivation(
            "derived:ProfitLoss(no NCI tagged)",
            ("profit_loss", "nci_income"),
            lambda w: w["profit_loss"].where(w["nci_income"].isna()),
        ),
        Derivation(
            "derived:NetIncomeLossAvailableToCommonStockholdersBasic",
            ("ni_to_common",),
            lambda w: w["ni_to_common"],
        ),
    ),
    "income_cont_ops": (  # after net_income, so it can fall back to a derived net income
        Derivation("derived:net_income", ("net_income",), lambda w: w["net_income"]),
    ),
    "total_liabilities": (
        Derivation(
            "derived:LiabilitiesAndStockholdersEquity-total_equity",
            ("liab_and_equity", "equity", "equity_incl_nci", "minority_interest"),
            lambda w: w["liab_and_equity"] - total_equity(w),
        ),
    ),
    "long_term_debt": (
        Derivation(
            "derived:LongTermDebt-LongTermDebtCurrent",
            ("lt_debt_total", "lt_debt_current"),
            lambda w: w["lt_debt_total"] - w["lt_debt_current"],
        ),
        Derivation(
            "derived:LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities-Current",
            ("ltd_lease_total", "ltd_lease_current"),
            lambda w: w["ltd_lease_total"] - w["ltd_lease_current"],
        ),
    ),
}

OUTPUT_FIELDS: tuple[str, ...] = tuple(name for name, spec in FIELDS.items() if not spec.helper)


def tag_table() -> pd.DataFrame:
    """One row per (field, tag) with columns field, tag, priority, kind, unit."""
    rows = [
        {"field": name, "tag": tag, "priority": rank, "kind": spec.kind, "unit": spec.unit}
        for name, spec in FIELDS.items()
        for rank, tag in enumerate(spec.tags)
    ]
    return pd.DataFrame(rows)


def all_tags() -> frozenset[str]:
    return frozenset(tag for spec in FIELDS.values() for tag in spec.tags)
