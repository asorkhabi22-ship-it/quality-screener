"""Tests for viz: finding titles, figure contents, dropdown states, red-flag percentiles, the
filing markers and the dashboard page. No PNGs are rendered (kaleido needs Chrome)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from quality_screener import stats, viz

MONTHS = pd.date_range("2020-01-31", periods=24, freq="ME")


def test_finding_titles_state_direction_and_significance() -> None:
    assert viz.finding_title("accruals_cf", 0.056, 2.84) == (
        "Low-accrual firms beat high-accrual firms by 5.6% a year (t = 2.84)")
    assert viz.finding_title("f_score", -0.051, -2.24) == (
        "Low-F firms beat high-F firms by 5.1% a year (t = -2.24)")
    assert viz.finding_title("altman_z", 0.01, 0.5).endswith("(t = 0.50, not significant)")


def test_dollars_avoid_latex_and_log_ticks_cover_range() -> None:
    assert viz.dollars(2.164) == "&#36;2.16"
    assert viz.log_ticks(0.19, 2.2) == [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


# --- distributions -------------------------------------------------------------------------


def firm_years() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "fiscal_year": rng.integers(2010, 2026, n), "f_score": rng.integers(0, 10, n).astype(float),
        "z_score": rng.normal(3, 5, n), "m_score": np.r_[np.full(190, -2.5), np.full(10, -1.0)],
        "accruals_cf": rng.normal(-0.04, 0.08, n),
    })


def test_score_distributions_panels() -> None:
    data = firm_years()
    fig = viz.score_distributions(data)
    f_bar, z_hist, m_hist, acc_hist = fig.data
    assert list(f_bar.x) == list(range(10)) and sum(f_bar.y) == 200
    assert min(z_hist.x) >= -2 and max(z_hist.x) <= 10
    lo, hi = data["accruals_cf"].quantile([0.01, 0.99])
    assert min(acc_hist.x) == pytest.approx(lo) and max(acc_hist.x) == pytest.approx(hi)
    notes = [a.text for a in fig.layout.annotations]
    assert "5.0% above -1.78" in notes  # 10 of 200
    assert "5.0% flagged by Beneish" in fig.layout.title.text


# --- quintile bars and cumulative returns ---------------------------------------------------


def port_and_ls(signal: str = "accruals_cf", n_buckets: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(1)
    rows = []
    for b in range(1, n_buckets + 1):
        r = 0.01 + 0.001 * b + rng.normal(0, 0.02, len(MONTHS))
        rows += [{"signal": signal, "t": m, "month": m, "bucket": b, "ew": v, "vw": v / 2}
                 for m, v in zip(MONTHS, r)]
    port = pd.DataFrame(rows)
    top = port[port["bucket"] == n_buckets].set_index("month")[["ew", "vw"]]
    low = port[port["bucket"] == 1].set_index("month")[["ew", "vw"]]
    ls = (top - low).reset_index().assign(signal=signal, t=lambda d: d["month"])
    return port, ls


def test_bucket_ci_comes_from_newey_west() -> None:
    port, _ = port_and_ls()
    est = viz.bucket_estimates(port, "accruals_cf").set_index(["weighting", "bucket"])
    r = port[(port["bucket"] == 3)].sort_values("month")["ew"]
    fit = stats.mean_newey_west(r, 6)
    assert est.loc[("ew", 3), "ann_mean"] == pytest.approx(12 * fit.params[0])
    assert est.loc[("ew", 3), "ci"] == pytest.approx(12 * 1.96 * fit.se[0])


def test_quintile_bars_dropdown_swaps_traces_title_and_note() -> None:
    port, ls = port_and_ls()
    fig = viz.quintile_bars(port, ls, "accruals_cf")
    assert [t.visible for t in fig.data] == [True, False]
    assert list(fig.data[0].marker.color) == list(viz.QUINTILE_COLORS)
    buttons = fig.layout.updatemenus[0].buttons
    assert [b.label for b in buttons] == ["Equal-weighted", "Value-weighted"]
    assert buttons[1].args[0]["visible"] == [False, True]
    vw_spread = stats.mean_newey_west(ls.sort_values("month")["vw"], 6).params[0] * 12
    assert f"{vw_spread * 100:+.1f}% a year" in buttons[1].args[1]["annotations"][0]["text"]
    assert fig.layout.title.text == buttons[0].args[1]["title.text"]


def test_f_groups_use_three_colours_and_labels() -> None:
    port, ls = port_and_ls("f_group", n_buckets=3)
    fig = viz.quintile_bars(port, ls, "f_group")
    assert list(fig.data[0].x) == ["F 0-3 (low)", "F 4-6", "F 7-9 (high)"]
    assert list(fig.data[0].marker.color) == [viz.QUINTILE_COLORS[i] for i in (0, 2, 4)]


def test_cumulative_wealth_by_hand() -> None:
    ls = pd.DataFrame({"signal": "x", "t": MONTHS[:2], "month": MONTHS[1:3], "ew": [0.1, -0.5], "vw": 0.0})
    wealth = viz.cumulative_wealth(ls, "x", "ew")
    assert wealth.tolist() == pytest.approx([1.0, 1.1, 0.55])
    assert wealth.index[0] == MONTHS[0]  # starts at $1 on the first formation date


def test_cumulative_long_short_figure() -> None:
    _, a = port_and_ls("accruals_cf")
    _, f = port_and_ls("f_score")
    fig = viz.cumulative_long_short(pd.concat([a, f]))
    assert len(fig.data) == 4 and [t.visible for t in fig.data] == [True, True, False, False]
    assert fig.layout.yaxis.type == "log"
    assert "$" not in fig.layout.title.text  # escaped, or plotly renders it as LaTeX
    covid = [s for s in fig.layout.shapes if s.type == "rect"]
    assert covid and str(covid[0].x0).startswith("2020-02")


# --- red flags -----------------------------------------------------------------------------


def scored_frame() -> pd.DataFrame:
    rows = []
    for year, scale in ((2023, 1.0), (2024, 1.1)):
        for i, ticker in enumerate(["AAA", "BBB", "CCC", "FIN"]):
            row = {"cik": str(i), "ticker": ticker, "name": f"{ticker} Inc", "fiscal_year": year,
                   "period_end": pd.Timestamp(f"{year}-12-31"),
                   "availability_date": pd.Timestamp(f"{year + 1}-02-15"),
                   "excluded_from_main": ticker == "FIN", "f_score": 5.0 + i, "z_score": 3.0,
                   "m_score": -1.5 if ticker == "CCC" else -2.5, "accruals_cf": 0.01 * i,
                   "market_cap": 100.0 * (i + 1)}
            row |= {col: scale * (i + 1) for col in viz.RED_FLAGS.values() if col != "accruals_cf"}
            rows.append(row)
    return pd.DataFrame(rows)


def test_red_flag_percentiles_rank_within_year_excluding_financials() -> None:
    pct = viz.red_flag_percentiles(scored_frame(), 2024)
    assert pct.index.tolist() == ["AAA", "BBB", "CCC"]  # FIN left out
    assert pct["DSRI"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert pct["Accruals"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_red_flag_radar_title_median_and_prior_year() -> None:
    fig = viz.red_flag_radar("CCC", 2024, scored_frame())
    names = [t.name for t in fig.data]
    assert names == ["Median firm", "FY2023", "FY2024"]
    assert fig.data[0].line.dash == "dash"
    assert len(fig.data[2].r) == len(viz.RED_FLAGS) + 1  # closed polygon
    assert "M-score -1.50, above the -1.78 threshold (flagged" in fig.layout.title.text
    calm = viz.red_flag_radar("AAA", 2024, scored_frame(), prior_year=False)
    assert "below the -1.78 threshold (not flagged)" in calm.layout.title.text
    assert len(calm.data) == 2
    with pytest.raises(ValueError):
        viz.red_flag_radar("FIN", 2024, scored_frame())


def test_radar_selector_one_button_per_firm() -> None:
    fig = viz.radar_selector(scored_frame(), ["AAA", "CCC"], 2024)
    buttons = fig.layout.updatemenus[0].buttons
    assert [b.label for b in buttons] == ["AAA", "CCC"]
    assert buttons[1].args[0]["visible"] == [False] * 3 + [True] * 3
    assert "CCC FY2024" in buttons[1].args[1]["title.text"]
    assert [t.visible for t in fig.data] == [True] * 3 + [False] * 3


# --- timeline ------------------------------------------------------------------------------


FILINGS = pd.DataFrame({
    "form": ["10-K", "10-Q", "10-K/A", "10-K", "8-K", "10-KT"],
    "filingDate": pd.to_datetime(["2024-02-15", "2024-05-01", "2024-06-01", "2025-02-15",
                                  "2025-03-01", "2025-09-01"]),
})


def test_annual_report_filings_classifies_amendments() -> None:
    out = viz.annual_report_filings(FILINGS)
    assert out["kind"].tolist() == ["10-K", "10-K/A", "10-K", "10-K"]


def test_score_timeline_marks_every_filing() -> None:
    fig = viz.score_timeline("CCC", scored_frame(), FILINGS)
    lines = [s for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1]
    red = [s for s in lines if s.line.color == viz.QUINTILE_COLORS[0]]
    assert len(lines) == 4 * 4  # 4 filings x 4 panels
    assert len(red) == 4  # the 10-K/A, on every panel
    assert "Beneish flag in FY2023, FY2024" in fig.layout.title.text
    assert "3 10-Ks, 1 10-K/A" in fig.layout.title.text


# --- saving and dashboard -------------------------------------------------------------------


def test_save_figures_writes_html_and_batches_pngs(tmp_path: Path) -> None:
    calls = []
    figs = {"a": go.Figure(go.Bar(x=[1], y=[1])), "b": go.Figure(go.Bar(x=[2], y=[2]))}
    written = viz.save_figures(figs, tmp_path, write_png=lambda f, p: calls.append((f, p)))
    assert (tmp_path / "a.html").exists() and (tmp_path / "b.html").exists()
    assert len(calls) == 1 and [p.name for p in calls[0][1]] == ["a.png", "b.png"]
    assert len(written) == 4
    assert "cdn.plot.ly" in (tmp_path / "a.html").read_text()


def test_markdown_table_to_html() -> None:
    md = "# x\n\n## Main table\n\n| Signal | Q1 |\n|---|---|\n| F & co | +1.0% |\n\nafter"
    out = viz.markdown_table_html(md, "## Main table")
    assert "<th>Signal</th>" in out and "<td>F &amp; co</td>" in out and "---" not in out


def test_dashboard_page_loads_plotly_once_and_nothing_else() -> None:
    figs = [go.Figure(go.Bar(x=[1], y=[1])), go.Figure(go.Bar(x=[2], y=[2]))]
    sections = [("One", viz.figure_html(figs[0], first=True)), ("Two", viz.figure_html(figs[1], first=False))]
    page = viz.dashboard_html("Title & more", "<p>intro</p>", sections)
    assert page.count("<script src=") + page.count('<script charset="utf-8" src=') == 1
    assert page.count("cdn.plot.ly") == 1 and "Title &amp; more" in page
    assert "<h2>Two</h2>" in page
