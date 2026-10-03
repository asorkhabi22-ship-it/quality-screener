"""Backtest the quality signals: monthly quintile returns, long-short statistics, robustness.

Reads (no downloads except the Fama-French factors on the very first run):
    data/processed/holdings.parquet, holdings_incl_fin_util.parquet  (scripts/build_portfolios.py)
    data/processed/prices_monthly.parquet, trading_calendar.parquet  (scripts/fetch_prices.py)
    data/raw/french/*.zip   Fama-French factors (fetched once; skipped if unavailable)

Outputs:
    data/processed/portfolio_returns.parquet   next-month bucket returns (EW, VW) per signal
    data/processed/long_short_returns.parquet  monthly top-minus-bottom per signal
    reports/results.csv              main table: one row per signal x weighting
    reports/results_buckets.csv      per-bucket mean, volatility, Sharpe
    reports/results_forward.csv      3/6/12-month buy-and-hold returns in excess of SPY
    reports/results_robustness.csv   long-short mean and t-stat across samples
    reports/results.md               the tables and a plain reading of them

Usage:
    python scripts/run_backtest.py
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quality_screener import config, factors, portfolios, prices, stats

log = logging.getLogger("run_backtest")

HORIZONS = (3, 6, 12)
T_CRITICAL = 1.96  # two-sided 5%
REPORT_MD = config.REPORTS_DIR / "results.md"
REPORT_CSV = config.REPORTS_DIR / "results.csv"
PORTFOLIO_RETURNS_PATH = config.PROCESSED_DIR / "portfolio_returns.parquet"
LONG_SHORT_PATH = config.PROCESSED_DIR / "long_short_returns.parquet"
SIGNAL_LABELS = {
    "f_group": "Piotroski F (groups)",
    "f_score": "Piotroski F (quintiles)",
    "altman_z": "Altman Z",
    "beneish_m": "Beneish M",
    "accruals_cf": "Accruals (cash flow)",
    "composite": "Composite",
    "accruals_bs": "Accruals (balance sheet)",
}
FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


@dataclass(frozen=True)
class Backtest:
    port: pd.DataFrame  # next-month bucket returns
    ls: pd.DataFrame  # monthly long-short
    buckets: pd.DataFrame  # bucket_summary
    ls_stats: pd.DataFrame  # long_short_summary + monotonicity
    forward: pd.DataFrame  # forward_summary
    universe: pd.Series  # equal-weighted mean return of every held firm, by month
    firm_ret: pd.DataFrame  # next-month return per held (t, ticker)


def market_excess_returns(monthly: pd.DataFrame, rf: pd.Series | None) -> pd.Series:
    """SPY's monthly return minus the risk-free rate (raw SPY return when rf is None), indexed
    by calendar month."""
    spy = monthly[monthly["ticker"] == prices.BENCHMARK].set_index("t")["ret"]
    spy.index = pd.PeriodIndex(spy.index, freq="M")
    return spy - rf.reindex(spy.index) if rf is not None else spy


def run(
    holdings: pd.DataFrame, monthly: pd.DataFrame, month_ends: pd.DatetimeIndex,
    market_excess: pd.Series, rf: pd.Series | None, horizons: Sequence[int] = HORIZONS,
) -> Backtest:
    """Next-month and buy-and-hold portfolio returns and their statistics for one holdings set."""
    firm_ret = portfolios.firm_next_month_returns(holdings, monthly, month_ends)
    port = portfolios.bucket_returns(holdings, firm_ret)
    ls = portfolios.long_short(port)
    buckets = stats.bucket_summary(port, rf)
    ls_stats = stats.long_short_summary(ls, market_excess).merge(
        stats.monotonicity(buckets), on=["signal", "weighting"]
    )
    fwd_firm = portfolios.firm_forward_returns(holdings, monthly, month_ends, horizons)
    fwd_port = portfolios.bucket_returns(holdings, fwd_firm, keys=("horizon",))
    forward = stats.forward_summary(fwd_port, portfolios.long_short(fwd_port, keys=("horizon",)))
    universe = firm_ret.groupby("month")["ret"].mean()
    return Backtest(port, ls, buckets, ls_stats, forward, universe, firm_ret)


def main_table(bt: Backtest, ff: pd.DataFrame | None) -> pd.DataFrame:
    """One row per signal and weighting: bucket means q1..q5 (F groups: low, mid, high in
    q1, q3, q5), long-short statistics and, when available, the FF5 + momentum alpha."""
    b = bt.buckets.copy()
    top = b.groupby("signal")["bucket"].transform("max")
    f_columns = b["bucket"].map({1: "q1", 2: "q3", 3: "q5"})
    b["column"] = np.where(top == 3, f_columns, "q" + b["bucket"].astype(str))
    means = b.pivot_table(index=["signal", "weighting"], columns="column", values="mean")
    out = means.reindex(columns=["q1", "q2", "q3", "q4", "q5"]).join(
        bt.ls_stats.set_index(["signal", "weighting"])
    )
    if ff is not None:
        alphas = ff.set_index(["signal", "weighting"])[["alpha", "alpha_t"]]
        out = out.join(alphas.add_prefix("ff6_"))
    order = pd.MultiIndex.from_product([portfolios.SIGNAL_ORDER, stats.WEIGHTINGS])
    out = out.reindex(order.intersection(out.index, sort=False))
    return out.rename_axis(["signal", "weighting"])


def robustness(
    base: pd.DataFrame, fin_util: pd.DataFrame | None, market_excess: pd.Series
) -> pd.DataFrame:
    """Long-short mean and Newey-West t-stat per signal and weighting across samples."""
    year = base["month"].dt.year
    scenarios = {
        "Base (ex financials & utilities)": base,
        "2012-2018": base[year <= 2018],
        "2019-2026": base[year >= 2019],
        "Ex 2020": base[year != 2020],
    }
    if fin_util is not None:
        scenarios["Incl. financials & utilities"] = fin_util
    cols = ["signal", "weighting", "months", "mean", "nw_t"]
    parts = [
        stats.long_short_summary(ls, market_excess)[cols].assign(scenario=name)
        for name, ls in scenarios.items()
    ]
    return pd.concat(parts, ignore_index=True)


# --- markdown ------------------------------------------------------------------------------


def pct(x: float, digits: int = 2) -> str:
    return "" if pd.isna(x) else f"{x * 100:+.{digits}f}%"


def num(x: float, digits: int = 2) -> str:
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def md_table(frame: pd.DataFrame) -> str:
    """GitHub markdown table of a frame of strings (index not included)."""
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def with_t(estimates: pd.Series, tstats: pd.Series) -> list[str]:
    """'+0.12% (1.85)' cells."""
    return [f"{pct(e)} ({num(t)})" for e, t in zip(estimates, tstats)]


def main_table_md(table: pd.DataFrame, weighting: str, has_ff: bool) -> str:
    t = table.xs(weighting, level="weighting")
    rows = pd.DataFrame({
        "Signal": [SIGNAL_LABELS.get(s, s) for s in t.index],
        **{q.upper(): t[q].map(pct) for q in ["q1", "q2", "q3", "q4", "q5"]},
        "Q5-Q1": t["mean"].map(pct),
        "NW t": t["nw_t"].map(num),
        "CAPM alpha (t)": with_t(t["capm_alpha"], t["capm_alpha_t"]),
        "Sharpe L/S": t["sharpe"].map(num),
    })
    if has_ff:
        rows["FF5+MOM alpha (t)"] = with_t(t["ff6_alpha"], t["ff6_alpha_t"])
    return md_table(rows)


def detail_md(table: pd.DataFrame) -> str:
    t = table.reset_index()
    rows = pd.DataFrame({
        "Signal": t["signal"].map(lambda s: SIGNAL_LABELS.get(s, s)),
        "Weighting": t["weighting"].str.upper(),
        "Months": t["months"],
        "Hit rate": t["hit_rate"].map(lambda x: f"{x:.0%}"),
        "Max drawdown": t["max_drawdown"].map(lambda x: f"{x:.0%}"),
        "Ann. vol": t["ann_vol"].map(lambda x: f"{x:.1%}"),
        "CAPM beta": t["capm_beta"].map(num),
        "Spearman (bucket vs mean)": t["spearman"].map(num),
        "Monotonic": t["increasing"].map({True: "yes", False: "no"}),
    })
    return md_table(rows)


def forward_md(forward: pd.DataFrame) -> str:
    f = forward.assign(cell=with_t(forward["long_short"], forward["nw_t"]))
    wide = f.pivot_table(index=["signal", "weighting"], columns="horizon", values="cell",
                         aggfunc="first")
    order = pd.MultiIndex.from_product([portfolios.SIGNAL_ORDER, stats.WEIGHTINGS])
    wide = wide.reindex(order).dropna(how="all")
    rows = pd.DataFrame({
        "Signal": [SIGNAL_LABELS.get(s, s) for s, _ in wide.index],
        "Weighting": [w.upper() for _, w in wide.index],
        **{f"{h}m L/S (t)": wide[h].to_numpy() for h in wide.columns},
    })
    return md_table(rows)


def robustness_md(robust: pd.DataFrame, weighting: str) -> str:
    r = robust[robust["weighting"] == weighting]
    r = r.assign(cell=with_t(r["mean"], r["nw_t"]))
    wide = r.pivot_table(index="signal", columns="scenario", values="cell", aggfunc="first")
    wide = wide.reindex([s for s in portfolios.SIGNAL_ORDER if s in wide.index])
    wide = wide[[c for c in r["scenario"].unique()]]
    wide.insert(0, "Signal", [SIGNAL_LABELS.get(s, s) for s in wide.index])
    return md_table(wide)


def _labels(signals: Sequence[str]) -> str:
    return "; ".join(SIGNAL_LABELS.get(s, s) for s in signals)


@dataclass(frozen=True)
class Drivers:
    tickers: list[str]  # largest contributors to the bottom bucket's equal-weighted return
    mean: float  # bottom bucket's mean monthly EW return
    mean_without: float  # the same without those tickers


def bottom_bucket_drivers(
    holdings: pd.DataFrame, firm_ret: pd.DataFrame, signal: str, n: int = 5
) -> Drivers:
    """Which firms drive the bottom bucket's equal-weighted return, and how much it changes
    without them."""
    low = holdings[(holdings["signal"] == signal) & (holdings["bucket"] == 1)]
    low = low.merge(firm_ret, on=["t", "ticker"])
    top = low.groupby("ticker")["ret"].sum().nlargest(n).index.tolist()
    without = low[~low["ticker"].isin(top)]
    return Drivers(top, low.groupby("month")["ret"].mean().mean(),
                   without.groupby("month")["ret"].mean().mean())


def reading_notes(table: pd.DataFrame, drivers: dict[str, Drivers] | None = None) -> list[str]:
    """Interpretation bullets chosen by the estimates (which signals are significant, with
    which sign, in which weighting), plus the standing caveat on long-horizon t-stats.
    ``drivers`` (by signal) adds the firms behind a wrong-signed bottom bucket."""
    signals = [s for s in portfolios.SIGNAL_ORDER if s in table.index.get_level_values(0)]
    t = table["nw_t"].unstack("weighting")
    wrong_sign = [s for s in signals if (t.loc[s] <= -T_CRITICAL).any()]
    ew_only = [s for s in signals if s not in wrong_sign
               and abs(t.loc[s, "ew"]) >= T_CRITICAL and abs(t.loc[s, "vw"]) < T_CRITICAL]
    null = [s for s in signals if (t.loc[s].abs() < T_CRITICAL).all()]
    notes = []
    if wrong_sign:
        notes.append(
            f"- **{_labels(wrong_sign)}: significant with the opposite sign to the hypothesis.** "
            "This is the direction survivorship bias predicts. A firm in today's S&P 500 that "
            "scored badly years ago is, by construction, one that survived and recovered, so "
            "its later returns are high; the low scorers that failed are not in the data. Treat "
            "this as a property of the sample, not as evidence that low quality earns a "
            "premium, until it is repeated on a point-in-time universe that includes delisted "
            "firms."
        )
        for s in wrong_sign:
            if drivers and s in drivers:
                d = drivers[s]
                notes.append(
                    f"  - {SIGNAL_LABELS.get(s, s)}, bottom bucket: the largest contributors are "
                    f"{', '.join(d.tickers)}. Without them its mean falls from {pct(d.mean)} to "
                    f"{pct(d.mean_without)} a month, so the effect is not just a few names."
                )
    if ew_only:
        notes.append(
            f"- **{_labels(ew_only)}: significant equal-weighted only.** The spread sits in the "
            "smaller members of the index; in value-weighted portfolios, dominated by the "
            "largest firms, it cannot be told apart from zero."
        )
    if null:
        notes.append(f"- **{_labels(null)}: no detectable effect** in either weighting.")
    if not table["increasing"].any():
        notes.append("- No signal's bucket means rise steadily from Q1 to Q5 in either weighting.")
    notes.append(
        "- The 3/6/12-month buy-and-hold t-stats use overlapping returns. Newey-West errors are "
        "known to over-reject in samples this short at long horizons (Ang and Bekaert 2007, "
        "Review of Financial Studies 20, 651-707), so read the 12-month t-stats as optimistic."
    )
    return notes


def verdict_lines(table: pd.DataFrame) -> list[str]:
    """One plain sentence per signal from the long-short estimates."""
    lines = []
    for signal in [s for s in portfolios.SIGNAL_ORDER if s in table.index.get_level_values(0)]:
        ew, vw = table.loc[(signal, "ew")], table.loc[(signal, "vw")]
        sig = [w.upper() for w, r in (("ew", ew), ("vw", vw)) if abs(r["nw_t"]) >= T_CRITICAL]
        if not sig:
            judgement = "not statistically significant in either weighting"
        else:
            direction = [
                "high quality beat low quality" if table.loc[(signal, w.lower()), "mean"] > 0
                else "LOW quality beat high quality"
                for w in sig
            ]
            judgement = "; ".join(f"significant {w} ({d})" for w, d in zip(sig, direction))
        lines.append(
            f"- **{SIGNAL_LABELS.get(signal, signal)}**: Q5-Q1 {pct(ew['mean'])}/month EW "
            f"(t = {num(ew['nw_t'])}), {pct(vw['mean'])} VW (t = {num(vw['nw_t'])}): {judgement}."
        )
    return lines


def build_markdown(
    table: pd.DataFrame, bt: Backtest, robust: pd.DataFrame, holdings: pd.DataFrame,
    has_ff: bool, rf_source: str, spy: pd.Series, drivers: dict[str, Drivers] | None = None,
) -> str:
    months = bt.ls["month"]
    n_tests = int(table["nw_t"].notna().sum())
    n_sig = int((table["nw_t"].abs() >= T_CRITICAL).sum())
    n_right = int((table["nw_t"] >= T_CRITICAL).sum())
    spy_mean = spy.reindex(pd.PeriodIndex(bt.universe.index, freq="M")).mean()
    detectable = T_CRITICAL * table["nw_se"].median()
    firms = holdings[holdings["signal"] == "composite"].groupby("t")["cik"].nunique()
    no_cap = holdings["market_cap"].isna().mean()
    sections = [
        "# Results: do high-quality firms outperform low-quality firms?",
        f"Generated by `scripts/run_backtest.py` on {pd.Timestamp.today():%Y-%m-%d}. "
        f"Return months {months.min():%Y-%m} to {months.max():%Y-%m} ({months.nunique()} months); "
        f"about {firms.mean():.0f} firms per formation date in the composite sort "
        f"(current S&P 500 members, ex financials and utilities). Portfolios are formed at "
        f"each month end from the latest 10-K filed at least one trading day earlier and held "
        f"for one month. Returns are total returns (Yahoo adjusted close), in % per month.",
        "## Short answer",
        f"{n_sig} of {n_tests} long-short tests (signals x EW/VW) have |t| >= {T_CRITICAL}, "
        f"{n_right} of them with the hypothesised sign (high quality beats low). "
        f"At a 5% level about {0.05 * n_tests:.1f} would clear that bar by chance alone, and "
        f"Harvey, Liu and Zhu (2016) argue for t > 3 after decades of factor mining. "
        f"With this sample, a long-short needs a mean of roughly {pct(detectable)}/month "
        f"({pct(detectable * 12, 1)} a year) to reach t = {T_CRITICAL}, so small effects "
        f"cannot be told apart from zero.",
        "\n".join(verdict_lines(table)),
        "## Reading the evidence",
        "\n".join(reading_notes(table, drivers)),
        "## Main table: equal-weighted",
        main_table_md(table, "ew", has_ff),
        "## Main table: value-weighted (market cap at formation)",
        main_table_md(table, "vw", has_ff),
        "Q1 = lowest quality, Q5 = highest (low M and low accruals count as high quality). "
        "Piotroski F groups have three buckets, shown as Q1 (F 0-3), Q3 (F 4-6) and Q5 "
        "(F 7-9); their 'Q5-Q1' is high minus low. NW t = Newey-West t-statistic, 6 lags. "
        "CAPM alpha is per month, on SPY minus the risk-free rate. Sharpe L/S is annualised.",
        "## Long-short details",
        detail_md(table),
        "## Buy-and-hold returns in excess of SPY",
        "Overlapping returns from each formation date; Newey-West t with h-1 lags.",
        forward_md(bt.forward),
        "## Robustness: long-short mean per month (NW t)",
        "### Equal-weighted",
        robustness_md(robust, "ew"),
        "### Value-weighted",
        robustness_md(robust, "vw"),
        "Composite vs individual signals and cash-flow vs balance-sheet accruals are rows of "
        "the same tables.",
        "## Why the results may be weak, and what they cannot show",
        "\n".join([
        "- **Large caps only.** Accounting anomalies are strongest in small, illiquid, "
        "less-followed stocks (Fama and French 2008, Dissecting Anomalies; Hou, Xue and Zhang "
        "2020, Replicating Anomalies). S&P 500 firms are the most heavily analysed stocks in the "
        "world, so mispricing should be small here.",
        "- **Survivorship bias.** The universe is today's S&P 500 list. Firms that failed, were "
        "acquired cheaply or dropped out of the index since 2012 are missing, and those are "
        "disproportionately the low-quality firms. Leaving them out flatters Q1, which pushes "
        "Q5-Q1 toward zero or below. Firms that joined the index because they grew are in the "
        "sample from 2012, which selects on later success in every bucket. No delisting returns are "
        f"included (Shumway 1997). The bias is visible: over the same months the equal-weighted "
        f"average of these firms returned {pct(bt.universe.mean())}/month against "
        f"{pct(spy_mean)} for SPY.",
        "- **Publication and crowding.** All four signals were published between 1968 and "
        "2000. McLean and Pontiff (2016) find anomaly returns fall by about a third to a half "
        "after publication as investors trade on them.",
        "- **Short sample.** About fourteen years of monthly data from one market regime. "
        "Several sub-period estimates rest on 80 or so months.",
        "- **Signals built for other jobs.** Altman Z and Beneish M were designed to predict "
        "bankruptcy and earnings manipulation, not returns; most S&P 500 firms are far from "
        "both thresholds, so their quintiles mostly sort ordinary firms.",
        "- **No trading costs, no shorting costs.** A monthly-rebalanced long-short pays both.",
        ]),
        "## Data notes",
        "\n".join([
        f"- Risk-free rate and factors: {rf_source}.",
        f"- {no_cap:.1%} of holdings have no market cap (mostly multi-class firms whose share "
        "count is not in companyfacts); they are equal-weighted but left out of value weights.",
        "- Stocks that stop trading are valued at their last price for the rest of the period "
        "(no delisting returns from Yahoo); see the run log for any such cases.",
        "- Parameters (15-month staleness, quintiles, 6 Newey-West lags, the F group cutoffs) "
        "were fixed before looking at returns and are not tuned.",
        ]),
    ]
    return "\n\n".join(sections) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    started = time.perf_counter()

    holdings = pd.read_parquet(portfolios.HOLDINGS_PATH)
    monthly = pd.read_parquet(prices.MONTHLY_PATH)
    month_ends = portfolios.full_month_ends(prices.load_calendar())
    ff = factors.load_factors()
    rf = ff["RF"] if ff is not None else None
    market_excess = market_excess_returns(monthly, rf)

    bt = run(holdings, monthly, month_ends, market_excess, rf)
    ff_alpha = stats.factor_alpha(bt.ls, ff[FACTOR_COLUMNS]) if ff is not None else None
    table = main_table(bt, ff_alpha)

    fin_util = None
    if portfolios.HOLDINGS_FIN_UTIL_PATH.exists():
        h_all = pd.read_parquet(portfolios.HOLDINGS_FIN_UTIL_PATH)
        firm_ret = portfolios.firm_next_month_returns(h_all, monthly, month_ends)
        fin_util = portfolios.long_short(portfolios.bucket_returns(h_all, firm_ret))
    else:
        log.warning("no %s; run build_portfolios.py --include-fin-util",
                    portfolios.HOLDINGS_FIN_UTIL_PATH.name)
    robust = robustness(bt.ls, fin_util, market_excess)

    bt.port.to_parquet(PORTFOLIO_RETURNS_PATH, index=False)
    bt.ls.to_parquet(LONG_SHORT_PATH, index=False)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_CSV)
    bt.buckets.to_csv(config.REPORTS_DIR / "results_buckets.csv", index=False)
    bt.forward.to_csv(config.REPORTS_DIR / "results_forward.csv", index=False)
    robust.to_csv(config.REPORTS_DIR / "results_robustness.csv", index=False)
    rf_source = (
        "Kenneth French data library (RF, Mkt-RF, SMB, HML, RMW, CMA, MOM)" if ff is not None
        else "unavailable: Sharpe ratios use raw returns, CAPM uses raw SPY"
    )
    spy = market_excess_returns(monthly, None)
    drivers = {s: bottom_bucket_drivers(holdings, bt.firm_ret, s) for s in ("f_group", "f_score")}
    markdown = build_markdown(table, bt, robust, holdings, ff is not None, rf_source, spy, drivers)
    REPORT_MD.write_text(markdown)

    shown = table[["q1", "q2", "q3", "q4", "q5", "mean", "nw_t", "capm_alpha", "capm_alpha_t",
                   "sharpe"]].copy()
    returns = ["q1", "q2", "q3", "q4", "q5", "mean", "capm_alpha"]
    shown[returns] *= 100
    print("Monthly returns in % (mean = Q5-Q1):")
    print(shown.round(2).to_string())
    print(f"\nwrote {REPORT_MD.relative_to(config.PROJECT_ROOT)} and "
          f"{REPORT_CSV.relative_to(config.PROJECT_ROOT)} in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
