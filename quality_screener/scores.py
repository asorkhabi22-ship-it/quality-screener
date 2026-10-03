"""Accounting quality scores computed from the wide point-in-time panel (``Panel.pit``).

Each score function takes the wide panel (current-year columns plus ``*_lag1``; Piotroski also
needs ``total_assets_lag2``, added by ``add_total_assets_lag2``) and returns a frame aligned to
it holding the score and every component as its own column, so red flags can be charted.
Missing inputs give NaN components, never 0, and each score records how many components it
could compute. Ratios go through ``safe_div``: a zero or missing denominator gives NaN, not inf.

Sources:
- Piotroski, J. D. (2000). Value Investing: The Use of Historical Financial Statement
  Information to Separate Winners from Losers. Journal of Accounting Research 38, 1-41.
- Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of
  Corporate Bankruptcy. Journal of Finance 23(4), 589-609.
- Altman, E. I. (1983). Corporate Financial Distress. Wiley (Z'' for non-manufacturers);
  zones as restated in Altman (2000), Predicting Financial Distress of Companies: Revisiting
  the Z-Score and ZETA Models, NYU Stern working paper.
- Beneish, M. D. (1999). The Detection of Earnings Manipulation. Financial Analysts Journal
  55(5), 24-36.
- Sloan, R. G. (1996). Do Stock Prices Fully Reflect Information in Accruals and Cash Flows
  about Future Earnings? The Accounting Review 71(3), 289-315.
- Hribar, P. and Collins, D. W. (2002). Errors in Estimating Accruals: Implications for
  Empirical Research. Journal of Accounting Research 40(1), 105-134.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quality_screener import config

SCORES_PATH = config.PROCESSED_DIR / "scores.parquet"

F_MIN_COMPONENTS = 7
SHARE_ISSUE_TOLERANCE = 0.02
Z_SAFE, Z_DISTRESS = 2.99, 1.81
Z2_SAFE, Z2_DISTRESS = 2.60, 1.10
M_THRESHOLD = -1.78
BENEISH_INTERCEPT = -4.84
BENEISH_WEIGHTS = {
    "dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892,
    "depi": 0.115, "sgai": -0.172, "tata": 4.679, "lvgi": -0.327,
}
WINSOR_LIMITS = (0.01, 0.99)
WINSOR_WINDOW = "1096D"  # trailing three years of filings
WINSOR_MIN_OBS = 200

F_SIGNALS = [
    "f_roa", "f_cfo", "f_droa", "f_accrual", "f_dlever",
    "f_dliquid", "f_eq_offer", "f_dmargin", "f_dturn",
]
ID_COLUMNS = ["cik", "period_end", "fiscal_year", "availability_date"]
FIRM_COLUMNS = ["ticker", "name", "sector", "sic", "is_financial", "is_utility", "excluded_from_main"]
MARKET_COLUMNS = ["shares_out_adj", "market_cap_date", "market_cap"]  # from prices.add_market_cap


# --- helpers ------------------------------------------------------------------------------


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN (never inf) where den is 0 or missing."""
    return num / den.where(den != 0)


def positive(x: pd.Series) -> pd.Series:
    """x where x > 0, else NaN (zero or negative assets/shares are data errors)."""
    return x.where(x > 0)


def indicator(condition: pd.Series, *inputs: pd.Series) -> pd.Series:
    """1.0 / 0.0 where every input is known, NaN otherwise."""
    defined = pd.concat(inputs, axis=1).notna().all(axis=1)
    return condition.astype(float).where(defined)


def zone(score: pd.Series, safe_above: float, distress_below: float) -> pd.Series:
    """'safe' above ``safe_above``, 'distress' below ``distress_below``, 'grey' between."""
    labels = pd.Series(pd.NA, index=score.index, dtype="string")
    labels[score < distress_below] = "distress"
    labels[(score >= distress_below) & (score <= safe_above)] = "grey"
    labels[score > safe_above] = "safe"
    return labels


def add_total_assets_lag2(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach total assets two years back (the prior-year row's ``total_assets_lag1``), needed
    for Piotroski's changes in ROA, leverage and turnover. ``availability_date`` becomes the
    later of the row's and the prior-year row's, so that value is never used before filing.
    Always derived from the panel itself; any ``total_assets_lag2`` passed in is replaced."""
    panel = panel.drop(columns="total_assets_lag2", errors="ignore")
    prior = panel[["cik", "period_end", "total_assets_lag1", "availability_date"]].rename(
        columns={
            "period_end": "period_end_lag1",
            "total_assets_lag1": "total_assets_lag2",
            "availability_date": "_prior_available",
        }
    )
    out = panel.merge(prior, on=["cik", "period_end_lag1"], how="left", validate="many_to_one")
    out["availability_date"] = out[["availability_date", "_prior_available"]].max(axis=1)
    return out.drop(columns="_prior_available")


def winsorize_point_in_time(
    values: pd.DataFrame,
    available: pd.Series,
    limits: tuple[float, float] = WINSOR_LIMITS,
    window: str = WINSOR_WINDOW,
    min_obs: int = WINSOR_MIN_OBS,
) -> pd.DataFrame:
    """Clip each column at its 1st/99th percentile across the firm-years filed in the trailing
    ``window`` up to each row's own ``available`` date.

    A plain per-fiscal-year cross-section would use filings published after some of that
    year's firms had filed (look-ahead in the clip bounds). Rows with fewer than ``min_obs``
    earlier observations are left unclipped.
    """
    order = available.sort_values(kind="stable").index
    when = pd.DatetimeIndex(available.loc[order])
    out = values.copy()
    for col in values.columns:
        rolling = pd.Series(values.loc[order, col].to_numpy(), index=when).rolling(
            window, min_periods=min_obs
        )
        lower = pd.Series(rolling.quantile(limits[0]).to_numpy(), index=order)
        upper = pd.Series(rolling.quantile(limits[1]).to_numpy(), index=order)
        out[col] = values[col].clip(lower=lower, upper=upper)
    return out


# --- Piotroski ----------------------------------------------------------------------------


def share_growth(panel: pd.DataFrame) -> pd.Series:
    return safe_div(positive(panel["shares_out"]), positive(panel["shares_out_lag1"])) - 1


def no_equity_issued(panel: pd.DataFrame) -> pd.Series:
    """Piotroski's EQ_OFFER signal: 1 if no common equity was issued in the year.

    Uses ``equity_issued <= 0`` when the cash-flow line is tagged; otherwise shares outstanding
    growing by no more than 2%. NaN if neither is known.
    """
    issued = panel["equity_issued"]
    by_cash = indicator(issued <= 0, issued)
    growth = share_growth(panel)
    by_shares = indicator(growth <= SHARE_ISSUE_TOLERANCE, growth)
    return by_cash.where(issued.notna(), by_shares)


def piotroski(panel: pd.DataFrame) -> pd.DataFrame:
    """Piotroski (2000) F-score: nine binary signals (1 = good news), summed.

    Profitability: ROA = net income / beginning total assets > 0; CFO / beginning assets > 0;
    change in ROA > 0; CFO / beginning assets > ROA (accrual). Leverage and liquidity: change
    in long-term debt / average total assets < 0; change in current ratio > 0; no common
    equity issued. Efficiency: change in gross margin > 0; change in asset turnover
    (revenue / beginning assets) > 0. Prior-year ratios use the prior year's beginning assets
    (``total_assets_lag2``). ``f_score`` needs at least 7 computable signals; with 7 or 8 the
    sum is rescaled to the 0-9 range (sum * 9 / n).
    """
    p = panel
    ta, ta1, ta2 = (positive(p[c]) for c in ("total_assets", "total_assets_lag1", "total_assets_lag2"))
    roa = safe_div(p["net_income"], ta1)
    cfo_ta = safe_div(p["cfo"], ta1)
    d_roa = roa - safe_div(p["net_income_lag1"], ta2)
    d_lever = safe_div(p["long_term_debt"], (ta + ta1) / 2) - safe_div(
        p["long_term_debt_lag1"], (ta1 + ta2) / 2
    )
    d_cr = safe_div(p["current_assets"], p["current_liabilities"]) - safe_div(
        p["current_assets_lag1"], p["current_liabilities_lag1"]
    )
    d_gm = safe_div(p["gross_profit"], p["revenue"]) - safe_div(p["gross_profit_lag1"], p["revenue_lag1"])
    d_turn = safe_div(p["revenue"], ta1) - safe_div(p["revenue_lag1"], ta2)

    out = pd.DataFrame(
        {
            "roa": roa,
            "cfo_to_assets": cfo_ta,
            "d_roa": d_roa,
            "d_leverage": d_lever,
            "d_current_ratio": d_cr,
            "share_growth": share_growth(p),
            "d_gross_margin": d_gm,
            "d_asset_turnover": d_turn,
            "f_roa": indicator(roa > 0, roa),
            "f_cfo": indicator(cfo_ta > 0, cfo_ta),
            "f_droa": indicator(d_roa > 0, d_roa),
            "f_accrual": indicator(cfo_ta > roa, cfo_ta, roa),
            "f_dlever": indicator(d_lever < 0, d_lever),
            "f_dliquid": indicator(d_cr > 0, d_cr),
            "f_eq_offer": no_equity_issued(p),
            "f_dmargin": indicator(d_gm > 0, d_gm),
            "f_dturn": indicator(d_turn > 0, d_turn),
        },
        index=p.index,
    )
    n = out[F_SIGNALS].notna().sum(axis=1)
    raw = out[F_SIGNALS].sum(axis=1, min_count=1)
    out["f_n_components"] = n
    out["f_raw"] = raw
    out["f_score"] = (raw * 9 / n.where(n > 0)).where(n >= F_MIN_COMPONENTS)
    return out


# --- Altman -------------------------------------------------------------------------------


def altman(panel: pd.DataFrame) -> pd.DataFrame:
    """Altman (1968) Z-score and Altman's (1983) Z'' for non-manufacturers.

    X1 = working capital / TA, X2 = retained earnings / TA, X3 = EBIT / TA, X5 = sales / TA
    (year-end total assets). Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5 with
    X4 = market value of equity / total liabilities, only when the panel has ``market_cap``;
    zones: > 2.99 safe, 1.81-2.99 grey, < 1.81 distress. Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3
    + 1.05 X4'' with X4'' = book equity / total liabilities; zones: > 2.60 safe, 1.10-2.60
    grey, < 1.10 distress. A score is NaN unless all of its components are known.
    """
    p = panel
    ta, tl = positive(p["total_assets"]), positive(p["total_liabilities"])
    market_cap = p["market_cap"] if "market_cap" in p else pd.Series(np.nan, index=p.index)
    x = pd.DataFrame(
        {
            "altman_x1": safe_div(p["current_assets"] - p["current_liabilities"], ta),
            "altman_x2": safe_div(p["retained_earnings"], ta),
            "altman_x3": safe_div(p["ebit"], ta),
            "altman_x4": safe_div(market_cap, tl),
            "altman_x4_book": safe_div(p["equity"], tl),
            "altman_x5": safe_div(p["revenue"], ta),
        },
        index=p.index,
    )
    z_parts = ["altman_x1", "altman_x2", "altman_x3", "altman_x4", "altman_x5"]
    z2_parts = ["altman_x1", "altman_x2", "altman_x3", "altman_x4_book"]
    x["z_score"] = x[z_parts].mul([1.2, 1.4, 3.3, 0.6, 1.0]).sum(axis=1, min_count=len(z_parts))
    x["z_n_components"] = x[z_parts].notna().sum(axis=1)
    x["z_zone"] = zone(x["z_score"], Z_SAFE, Z_DISTRESS)
    x["z2_score"] = x[z2_parts].mul([6.56, 3.26, 6.72, 1.05]).sum(axis=1, min_count=len(z2_parts))
    x["z2_n_components"] = x[z2_parts].notna().sum(axis=1)
    x["z2_zone"] = zone(x["z2_score"], Z2_SAFE, Z2_DISTRESS)
    return x


# --- Beneish ------------------------------------------------------------------------------


def beneish_indices(panel: pd.DataFrame) -> pd.DataFrame:
    """The eight raw Beneish (1999) indices (current year relative to prior year; TATA is a
    level). TATA = (income from continuing operations - CFO) / total assets."""
    p = panel

    def lag(col: str) -> pd.Series:
        return p[f"{col}_lag1"]

    def soft_assets(ca: pd.Series, ppe: pd.Series, ta: pd.Series) -> pd.Series:
        return 1 - safe_div(ca + ppe, positive(ta))

    def dep_rate(dep: pd.Series, ppe: pd.Series) -> pd.Series:
        return safe_div(dep, dep + ppe)

    def leverage(cl: pd.Series, ltd: pd.Series, ta: pd.Series) -> pd.Series:
        return safe_div(cl + ltd, positive(ta))

    rev, rev1 = p["revenue"], lag("revenue")
    return pd.DataFrame(
        {
            "m_dsri": safe_div(safe_div(p["receivables"], rev), safe_div(lag("receivables"), rev1)),
            "m_gmi": safe_div(safe_div(lag("gross_profit"), rev1), safe_div(p["gross_profit"], rev)),
            "m_aqi": safe_div(
                soft_assets(p["current_assets"], p["ppe_net"], p["total_assets"]),
                soft_assets(lag("current_assets"), lag("ppe_net"), lag("total_assets")),
            ),
            "m_sgi": safe_div(rev, rev1),
            "m_depi": safe_div(
                dep_rate(lag("dep_amort"), lag("ppe_net")), dep_rate(p["dep_amort"], p["ppe_net"])
            ),
            "m_sgai": safe_div(safe_div(p["sga"], rev), safe_div(lag("sga"), rev1)),
            "m_tata": safe_div(p["income_cont_ops"] - p["cfo"], positive(p["total_assets"])),
            "m_lvgi": safe_div(
                leverage(p["current_liabilities"], p["long_term_debt"], p["total_assets"]),
                leverage(lag("current_liabilities"), lag("long_term_debt"), lag("total_assets")),
            ),
        },
        index=p.index,
    )


def beneish(panel: pd.DataFrame) -> pd.DataFrame:
    """Beneish (1999) 8-variable M-score:
    M = -4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI - 0.172 SGAI
        + 4.679 TATA - 0.327 LVGI;  m_flag = M > -1.78 (likely manipulator).

    Raw indices (``m_*``) are winsorised point-in-time at the 1st/99th percentile
    (``m_*_w``, see ``winsorize_point_in_time``) before combining, because ratios explode
    when denominators approach zero. If SGA or depreciation is missing, SGAI / DEPI are set to
    1.0 (neutral; the practitioner convention) *after* winsorising, and named in
    ``m_imputed_components``. The raw index columns keep their NaN.
    """
    raw = beneish_indices(panel)
    used = winsorize_point_in_time(raw, panel["availability_date"]).add_suffix("_w")
    sga_missing = used["m_sgai_w"].isna() & (panel["sga"].isna() | panel["sga_lag1"].isna())
    dep_missing = used["m_depi_w"].isna() & (
        panel["dep_amort"].isna() | panel["dep_amort_lag1"].isna()
    )
    used.loc[sga_missing, "m_sgai_w"] = 1.0
    used.loc[dep_missing, "m_depi_w"] = 1.0

    m = BENEISH_INTERCEPT + sum(w * used[f"m_{k}_w"] for k, w in BENEISH_WEIGHTS.items())
    imputed = np.select(
        [sga_missing & dep_missing, sga_missing, dep_missing], ["SGAI,DEPI", "SGAI", "DEPI"], ""
    )
    out = pd.concat([raw, used], axis=1)
    out["m_n_components"] = raw.notna().sum(axis=1)
    out["m_imputed_components"] = pd.Series(imputed, index=panel.index, dtype="string").replace("", pd.NA)
    out["m_score"] = m
    out["m_flag"] = (m > M_THRESHOLD).astype("boolean").mask(m.isna())
    return out


# --- Sloan accruals -----------------------------------------------------------------------


def accruals(panel: pd.DataFrame) -> pd.DataFrame:
    """Sloan (1996) accruals scaled by average total assets.

    ``accruals_cf`` (primary) = (net income - CFO) / average TA. Hribar & Collins (2002) show
    the balance-sheet estimate is distorted by mergers, acquisitions and divestitures, which is
    why the cash-flow version is primary.

    ``accruals_bs`` (Sloan's original) = [(dCA - dCash) - (dCL - dSTD - dTaxPayable) - Dep]
    / average TA. dCA, dCash, dCL and Dep are required. dSTD (debt in current liabilities) and
    dTaxPayable are often untagged; such a term is left out of the sum and named in
    ``accruals_bs_omitted``, per the instruction to use the tags that exist.
    """
    p = panel

    def change(col: str) -> pd.Series:
        return p[col] - p[f"{col}_lag1"]

    avg_ta = (positive(p["total_assets"]) + positive(p["total_assets_lag1"])) / 2
    terms = pd.DataFrame(
        {
            "acc_d_current_assets": change("current_assets"),
            "acc_d_cash": change("cash"),
            "acc_d_current_liabilities": change("current_liabilities"),
            "acc_d_debt_current": change("debt_current"),
            "acc_d_taxes_payable": change("taxes_payable"),
            "acc_dep": p["dep_amort"],
        },
        index=p.index,
    )
    core = (
        terms["acc_d_current_assets"] - terms["acc_d_cash"]
        - terms["acc_d_current_liabilities"] - terms["acc_dep"]
    )
    adjustments = terms[["acc_d_debt_current", "acc_d_taxes_payable"]]
    measured_adjustments = adjustments.sum(axis=1, min_count=0)  # sum of the terms that exist

    out = terms.assign(avg_total_assets=avg_ta)
    out["accruals_cf"] = safe_div(p["net_income"] - p["cfo"], avg_ta)
    out["accruals_bs"] = safe_div(core + measured_adjustments, avg_ta)
    out["accruals_bs_n_components"] = terms.notna().sum(axis=1)
    std_missing, tax_missing = adjustments.isna().to_numpy().T
    omitted = np.select(
        [std_missing & tax_missing, std_missing, tax_missing],
        ["dSTD,dTaxPayable", "dSTD", "dTaxPayable"],
        "",
    )
    omitted = pd.Series(omitted, index=p.index, dtype="string").replace("", pd.NA)
    out["accruals_bs_omitted"] = omitted.where(out["accruals_bs"].notna())
    return out


# --- all scores ---------------------------------------------------------------------------


def compute_all(panel: pd.DataFrame, firms: pd.DataFrame | None = None) -> pd.DataFrame:
    """All scores and components, one row per (cik, period_end) of the wide PIT panel.

    ``firms`` (the universe) adds ticker, name, sector and the financial/utility flags.
    ``availability_date`` covers every input, including ``total_assets_lag2``. The market
    columns (``MARKET_COLUMNS``) are carried through when the panel has them, so portfolios
    can compute market cap at later dates.
    """
    p = add_total_assets_lag2(panel).reset_index(drop=True)
    market = [c for c in MARKET_COLUMNS if c in p]
    parts = [p[ID_COLUMNS + market], piotroski(p), altman(p), beneish(p), accruals(p)]
    scores = pd.concat(parts, axis=1)
    if firms is not None:
        scores = scores.merge(firms[["cik", *FIRM_COLUMNS]], on="cik", how="left", validate="many_to_one")
    return scores
