"""Build every figure (HTML + 1600x900 PNG in reports/figures) and reports/dashboard.html.

Reads data/processed/scores.parquet, universe.parquet, portfolio_returns.parquet and
long_short_returns.parquet (scripts/run_backtest.py), reports/results.md, and the cached SEC
submissions of the timeline firm. No downloads.

Radar firms are picked by rule, not by hand: in the latest fiscal year with M-scores for at
least ``MIN_M_FIRMS`` firms, the 10 largest firms by market cap with an M-score, then the 10
highest M-scores among the rest. The timeline firm is the one flagged by Beneish (M > -1.78)
in the most fiscal years (ties: larger market cap), unless --timeline-ticker is given.

Usage:
    python scripts/build_dashboard.py [--timeline-ticker AAPL] [--no-png]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from quality_screener import config, portfolios, scores, universe, viz
from quality_screener.edgar import EdgarClient

DASHBOARD_PATH = config.REPORTS_DIR / "dashboard.html"
RESULTS_MD = config.REPORTS_DIR / "results.md"  # written by scripts/run_backtest.py
RESULT_TABLES = [  # (label, heading in results.md)
    ("Equal-weighted", "## Main table: equal-weighted"),
    ("Value-weighted", "## Main table: value-weighted (market cap at formation)"),
]
MIN_M_FIRMS = 150
N_LARGEST, N_HIGHEST_M = 10, 10


def radar_year(main: pd.DataFrame, min_firms: int = MIN_M_FIRMS) -> int:
    """Latest fiscal year with M-scores for at least ``min_firms`` firms."""
    counts = main.groupby("fiscal_year")["m_score"].count()
    return int(counts[counts >= min_firms].index.max())


def radar_tickers(
    main: pd.DataFrame, year: int, n_largest: int = N_LARGEST, n_highest_m: int = N_HIGHEST_M
) -> list[str]:
    """The ``n_largest`` firms by market cap with an M-score that year, then the
    ``n_highest_m`` highest M-scores among the rest."""
    rows = main[(main["fiscal_year"] == year) & main["m_score"].notna()]
    largest = rows.nlargest(n_largest, "market_cap")["ticker"].tolist()
    rest = rows[~rows["ticker"].isin(largest)]
    return largest + rest.nlargest(n_highest_m, "m_score")["ticker"].tolist()


def most_flagged(main: pd.DataFrame) -> str:
    """Ticker flagged by Beneish (M > -1.78) in the most fiscal years; ties go to the larger
    latest market cap."""
    flags = (main["m_score"] > scores.M_THRESHOLD).groupby(main["ticker"]).sum()
    cap = main.sort_values("period_end").groupby("ticker")["market_cap"].last()
    ranked = pd.DataFrame({"flags": flags, "cap": cap.reindex(flags.index)})
    return ranked.sort_values(["flags", "cap"], ascending=False).index[0]


def intro_html(year: int, tickers: Sequence[str], timeline: str, months: pd.Series) -> str:
    return (
        "<p>Do firms with strong accounting quality earn higher stock returns? Four classic "
        "scores (Piotroski F, Altman Z, Beneish M, Sloan accruals) are computed from SEC "
        "EDGAR filings as first reported, and stocks are sorted into quintiles at each month "
        f"end using only 10-Ks filed at least one trading day earlier ({months.min():%Y-%m} "
        f"to {months.max():%Y-%m}).</p>"
        "<p><b>Read with care.</b> The universe is today's S&amp;P 500 (ex financials and "
        "utilities) applied back to 2012, so firms that failed or left the index are missing. "
        "That survivorship bias favours the low-quality buckets, and it is the most likely "
        "reason the F-score spread comes out negative. The full write-up, with robustness "
        "checks and caveats, is in <code>reports/results.md</code>. Chart titles state what "
        "the data show; they are computed, not written by hand.</p>"
        f"<p class=\"muted\">Radar firms (FY{year}): the {N_LARGEST} largest by market cap, then "
        f"the {N_HIGHEST_M} highest M-scores: {', '.join(tickers)}. Timeline firm: {timeline}, "
        "the firm flagged by Beneish in the most fiscal years.</p>"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeline-ticker", help="firm for the score timeline")
    parser.add_argument("--no-png", action="store_true", help="skip the PNG export")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    scored = pd.read_parquet(scores.SCORES_PATH)
    firms = pd.read_parquet(universe.UNIVERSE_PATH)
    port = pd.read_parquet(portfolios.PORTFOLIO_RETURNS_PATH)
    ls = pd.read_parquet(portfolios.LONG_SHORT_PATH)
    results_md = RESULTS_MD.read_text()
    main_rows = viz.main_sample(scored)
    in_window = main_rows[main_rows["fiscal_year"].between(config.START_YEAR, config.END_YEAR)]

    year = radar_year(main_rows)
    tickers = radar_tickers(main_rows, year)
    client = EdgarClient()
    timeline = args.timeline_ticker or most_flagged(main_rows)
    cik = firms.loc[firms["ticker"] == timeline, "cik"].item()

    signals = [s for s in portfolios.SIGNAL_ORDER if s in set(ls["signal"])]
    figures: dict[str, go.Figure] = {"score_distributions": viz.score_distributions(in_window)}
    figures |= {f"quintile_returns_{s}": viz.quintile_bars(port, ls, s) for s in signals}
    figures["cumulative_long_short"] = viz.cumulative_long_short(ls)
    figures |= {
        f"red_flag_radar_{t}_FY{year}": viz.red_flag_radar(t, year, scored) for t in tickers
    }
    figures["red_flag_radar_selector"] = viz.radar_selector(scored, tickers, year)
    figures[f"score_timeline_{timeline}"] = viz.score_timeline(
        timeline, scored, client.get_submissions(cik).filings
    )
    written = viz.save_figures(figures, write_png=None if args.no_png else viz.kaleido_writer)

    tables = "".join(
        f"<h3>{label}</h3>" + viz.markdown_table_html(results_md, heading)
        for label, heading in RESULT_TABLES
    )
    shown = ["score_distributions", "cumulative_long_short",
             *[f"quintile_returns_{s}" for s in signals], "red_flag_radar_selector",
             f"score_timeline_{timeline}"]
    headings = {"score_distributions": "Score distributions",
                "cumulative_long_short": "Cumulative long-short returns",
                "red_flag_radar_selector": "Red-flag radar (pick a firm)",
                f"score_timeline_{timeline}": f"Score timeline: {timeline}"}
    sections = [("Main results", tables)]
    for i, name in enumerate(shown):
        signal = name.removeprefix("quintile_returns_")
        heading = headings.get(name, f"Quintile returns: {portfolios.SIGNAL_LABELS.get(signal, name)}")
        height = 820 if "timeline" in name or "distributions" in name else 640
        sections.append((heading, viz.figure_html(figures[name], first=(i == 0), height=height)))
    page = viz.dashboard_html("Accounting quality and stock returns: S&P 500, 2012-2026",
                              intro_html(year, tickers, timeline, ls["month"]), sections)
    DASHBOARD_PATH.write_text(page)
    print(f"{len(figures)} figures -> {len(written)} files in "
          f"{viz.FIGURES_DIR.relative_to(config.PROJECT_ROOT)}; radar FY{year}: "
          f"{', '.join(tickers)}; timeline: {timeline}")
    elapsed = time.perf_counter() - started
    print(f"wrote {DASHBOARD_PATH.relative_to(config.PROJECT_ROOT)} in {elapsed:.1f}s")
    for name, fig in figures.items():
        if not name.startswith("red_flag_radar_") or name.endswith("selector"):
            print(f"  {name}: {fig.layout.title.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
