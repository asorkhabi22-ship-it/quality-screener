"""Plotly figures for the quality screener, in one house style.

Style: white background, one font, a red-to-green sequential palette for Q1 (lowest quality)
to Q5 (highest), and titles that state the finding, computed from the data, rather than name
the variable. ``save_figures`` writes every figure as HTML (plotly.js from the CDN) and as a
1600x900 PNG (kaleido, which drives a local Chrome).

Formulas behind the annotations:
- 95% confidence interval of an annualised mean = 12 * (mean +/- 1.96 * Newey-West se), with
  6 lags (Newey and West 1987; see ``stats``).
- Cumulative long-short = growth of $1 in the monthly-rebalanced Q5 - Q1 portfolio,
  prod(1 + r_t), before costs.
- Red-flag percentiles: Beneish (1999) indices and Sloan (1996) accruals ranked against the
  same fiscal year's S&P 500 non-financials, higher = more suspicious. That follows Beneish's
  hypotheses for all eight indices, although SGAI and LVGI carry negative weights in his
  fitted M-score.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from quality_screener import config, portfolios, scores, stats

FIGURES_DIR = config.FIGURES_DIR
WIDTH, HEIGHT = 1600, 900
FONT_FAMILY = "Arial, Helvetica, sans-serif"
TEXT = "#222222"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
QUINTILE_COLORS = ("#c62828", "#ef6c00", "#f9a825", "#7cb342", "#2e7d32")  # Q1 red .. Q5 green
QUINTILE_SCALE = [[i / 4, c] for i, c in enumerate(QUINTILE_COLORS)]
SIGNAL_COLORS = {  # Okabe-Ito, colour-blind safe, for one line per signal
    "f_group": "#0072B2", "f_score": "#56B4E9", "altman_z": "#E69F00", "beneish_m": "#CC79A7",
    "accruals_cf": "#009E73", "composite": "#000000", "accruals_bs": "#D55E00",
}
Z_DISTRESS, Z_SAFE = scores.Z_DISTRESS, scores.Z_SAFE
M_THRESHOLD = scores.M_THRESHOLD
T_CRITICAL = 1.96
WEIGHTING_LABELS = {"ew": "Equal-weighted", "vw": "Value-weighted"}
# (high-quality name, low-quality name) used in finding titles
QUALITY_NAMES = {
    "f_group": ("High-F (7-9)", "low-F (0-3)"),
    "f_score": ("High-F", "low-F"),
    "altman_z": ("High-Z", "low-Z"),
    "beneish_m": ("Low-M", "high-M"),
    "accruals_cf": ("Low-accrual", "high-accrual"),
    "composite": ("High-composite", "low-composite"),
    "accruals_bs": ("Low balance-sheet-accrual", "high balance-sheet-accrual"),
}
RED_FLAGS = {  # axis label -> scores column (higher = more suspicious)
    "DSRI": "m_dsri", "GMI": "m_gmi", "AQI": "m_aqi", "SGI": "m_sgi", "DEPI": "m_depi",
    "SGAI": "m_sgai", "TATA": "m_tata", "LVGI": "m_lvgi", "Accruals": "accruals_cf",
}
ANNUAL_FORMS = {"10-K", "10-K405", "10-KT"}
AMENDMENT_FORMS = {"10-K/A", "10-K405/A", "10-KT/A"}

TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT_FAMILY, size=16, color=TEXT),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(x=0.02, xanchor="left", font=dict(size=24)),
        xaxis=dict(showgrid=False, linecolor="#9e9e9e", ticks="outside", zeroline=False),
        yaxis=dict(gridcolor=GRID, linecolor="#9e9e9e", zerolinecolor="#9e9e9e"),
        colorway=list(SIGNAL_COLORS.values()),
        margin=dict(l=90, r=50, t=130, b=80),
        legend=dict(bgcolor="rgba(255,255,255,0.85)"),
        hoverlabel=dict(font_family=FONT_FAMILY),
    )
)
pio.templates["quality"] = TEMPLATE
STYLE = "plotly_white+quality"

ImageWriter = Callable[[list[go.Figure], list[Path]], None]


# --- style and saving ----------------------------------------------------------------------


def styled(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    """Apply the house template and a finding title (plus a smaller subtitle)."""
    fig.update_layout(
        template=STYLE,
        title=dict(text=title, subtitle=dict(text=subtitle or "", font=dict(size=15, color=MUTED))),
    )
    return fig


def title_layout(title: str, subtitle: str) -> dict[str, object]:
    """Layout update for a dropdown button that swaps the title."""
    return {"title.text": title, "title.subtitle.text": subtitle}


def kaleido_writer(figures: list[go.Figure], paths: list[Path]) -> None:
    """All PNGs in one kaleido call (one Chrome session), at WIDTH x HEIGHT."""
    pio.write_images(figures, [str(p) for p in paths], width=WIDTH, height=HEIGHT)


def save_figures(
    figures: dict[str, go.Figure], out_dir: Path = FIGURES_DIR,
    write_png: ImageWriter | None = kaleido_writer,
) -> list[Path]:
    """Write each figure as ``<name>.html`` (plotly.js from the CDN) and ``<name>.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fig in figures.items():
        path = out_dir / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        written.append(path)
    if write_png is not None and figures:
        pngs = [out_dir / f"{name}.png" for name in figures]
        write_png(list(figures.values()), pngs)
        written += pngs
    return written


def dollars(amount: float) -> str:
    """'$2.16' that plotly will not read as LaTeX (it treats text between $ signs as math)."""
    return f"&#36;{amount:.2f}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _low(text: str) -> str:
    return text[:1].lower() + text[1:]


def finding_title(signal: str, spread_annual: float, tstat: float) -> str:
    """'High-F firms beat low-F firms by 5.1% a year (t = 2.30)', or the reverse when the
    spread is negative; '(not significant)' when |t| < 1.96."""
    good, bad = QUALITY_NAMES.get(signal, (f"High-{signal}", f"low-{signal}"))
    if spread_annual >= 0:
        lead = f"{_cap(good)} firms beat {_low(bad)} firms by {spread_annual * 100:.1f}% a year"
    else:
        lead = f"{_cap(bad)} firms beat {_low(good)} firms by {-spread_annual * 100:.1f}% a year"
    verdict = "" if abs(tstat) >= T_CRITICAL else ", not significant"
    return f"{lead} (t = {tstat:.2f}{verdict})"


# --- 1. score distributions ------------------------------------------------------------------


def score_distributions(firm_years: pd.DataFrame) -> go.Figure:
    """2x2 grid: F-score counts 0-9; Altman Z clipped to [-2, 10] with its zones; Beneish M with
    the -1.78 threshold and the share above it; cash-flow accruals clipped at the 1st/99th
    percentile. ``firm_years`` = the scored firm-years to describe."""
    f = np.floor(firm_years["f_score"].dropna() + 0.5).astype(int)
    f_counts = f.value_counts().reindex(range(10), fill_value=0)
    z = firm_years["z_score"].dropna()
    m = firm_years["m_score"].dropna()
    acc = firm_years["accruals_cf"].dropna()
    lo, hi = acc.quantile([0.01, 0.99])
    flagged, safe = (m > M_THRESHOLD).mean(), (z > Z_SAFE).mean()

    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.09, vertical_spacing=0.17,
        subplot_titles=[
            f"Piotroski F-score (n = {len(f):,})",
            f"Altman Z, clipped to [-2, 10] (n = {len(z):,})",
            f"Beneish M-score (n = {len(m):,})",
            f"Cash-flow accruals / avg assets, clipped 1-99% (n = {len(acc):,})",
        ],
    )
    colors = sample_colorscale(QUINTILE_SCALE, np.linspace(0, 1, 10))
    fig.add_trace(go.Bar(x=f_counts.index, y=f_counts.to_numpy(), marker_color=colors), 1, 1)
    fig.add_trace(go.Histogram(x=z.clip(-2, 10), xbins=dict(start=-2, end=10.01, size=0.25),
                               marker_color="#5f6f7f"), 1, 2)
    fig.add_trace(go.Histogram(x=m, nbinsx=60, marker_color="#5f6f7f"), 2, 1)
    fig.add_trace(go.Histogram(x=acc.clip(lo, hi), nbinsx=60, marker_color="#5f6f7f"), 2, 2)

    zones = [(-2, Z_DISTRESS, QUINTILE_COLORS[0], "distress"),
             (Z_DISTRESS, Z_SAFE, "#9e9e9e", "grey"),
             (Z_SAFE, 10, QUINTILE_COLORS[4], "safe")]
    for x0, x1, color, label in zones:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.10, line_width=0, row=1, col=2,
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=13, color=MUTED))
    for x in (Z_DISTRESS, Z_SAFE):
        fig.add_vline(x=x, line=dict(color=TEXT, dash="dash", width=1.5), row=1, col=2)
    fig.add_vline(x=M_THRESHOLD, line=dict(color=QUINTILE_COLORS[0], dash="dash", width=2),
                  row=2, col=1, annotation_text=f"{flagged:.1%} above {M_THRESHOLD}",
                  annotation_position="top right",
                  annotation_font=dict(size=14, color=QUINTILE_COLORS[0]))
    fig.add_vline(x=float(acc.median()), line=dict(color=TEXT, dash="dot", width=1.5), row=2, col=2,
                  annotation_text=f"median {acc.median():.3f}", annotation_position="top right",
                  annotation_font=dict(size=13, color=MUTED))
    fig.update_xaxes(title_text="F-score (rounded)", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="firm-years", row=1, col=1)
    fig.update_yaxes(title_text="firm-years", row=2, col=1)
    fig.update_layout(showlegend=False, bargap=0.05)
    years = firm_years["fiscal_year"]
    title = (f"Most firm-years look healthy: median F = {f.median():.0f}, {safe:.0%} in Altman's "
             f"safe zone, {flagged:.1%} flagged by Beneish")
    subtitle = (f"S&P 500 members ex financials and utilities, fiscal years {years.min()}-"
                f"{years.max()}; Z and accruals are clipped, so the edge bins hold the tails")
    return styled(fig, title, subtitle)


# --- 2. quintile return bars -----------------------------------------------------------------


def bucket_estimates(
    port: pd.DataFrame, signal: str, lags: int = stats.NW_LAGS_MONTHLY
) -> pd.DataFrame:
    """Per weighting and bucket: annualised mean return and the half-width of its 95% CI from
    the Newey-West standard error."""
    rows = []
    for (bucket, w), g in port[port["signal"] == signal].melt(
        id_vars=["bucket", "month"], value_vars=list(stats.WEIGHTINGS), var_name="w"
    ).groupby(["bucket", "w"]):
        fit = stats.mean_newey_west(g.sort_values("month")["value"].dropna(), lags)
        rows.append({"bucket": bucket, "weighting": w,
                     "ann_mean": fit.params[0] * stats.MONTHS_PER_YEAR,
                     "ci": T_CRITICAL * fit.se[0] * stats.MONTHS_PER_YEAR})
    return pd.DataFrame(rows).sort_values(["weighting", "bucket"], ignore_index=True)


def _bucket_labels(signal: str, buckets: Sequence[int]) -> tuple[list[str], list[str]]:
    if len(buckets) == 3:
        return ["F 0-3 (low)", "F 4-6", "F 7-9 (high)"], [QUINTILE_COLORS[i] for i in (0, 2, 4)]
    labels = [f"Q{b}" for b in buckets]
    labels[0], labels[-1] = f"{labels[0]} (low)", f"{labels[-1]} (high)"
    return labels, list(QUINTILE_COLORS[: len(buckets)])


def quintile_bars(port: pd.DataFrame, ls: pd.DataFrame, signal: str) -> go.Figure:
    """Annualised mean return per bucket with 95% CI whiskers, the top-minus-bottom spread and
    its Newey-West t-stat annotated, and an EW / VW dropdown."""
    est = bucket_estimates(port, signal)
    buckets = sorted(est["bucket"].unique())
    labels, colors = _bucket_labels(signal, buckets)
    months = port.loc[port["signal"] == signal, "month"]
    fig = go.Figure()
    views = []
    for i, w in enumerate(stats.WEIGHTINGS):
        e = est[est["weighting"] == w]
        fig.add_trace(go.Bar(
            x=labels, y=e["ann_mean"] * 100, marker_color=colors, name=WEIGHTING_LABELS[w],
            error_y=dict(type="data", array=e["ci"] * 100, color=TEXT, thickness=1.5, width=8),
            texttemplate="%{y:.1f}%", textposition="inside", insidetextanchor="start",
            textfont=dict(color="white", size=16), visible=(i == 0),
            hovertemplate="%{x}: %{y:.1f}% a year<extra></extra>",
        ))
        r = ls.loc[ls["signal"] == signal, ["month", w]].sort_values("month")[w].dropna()
        fit = stats.mean_newey_west(r, stats.NW_LAGS_MONTHLY)
        spread, t = fit.params[0] * stats.MONTHS_PER_YEAR, fit.tstat[0]
        top = "High - low" if len(buckets) == 3 else "Q5 - Q1"
        note = dict(text=f"<b>{top}: {spread * 100:+.1f}% a year</b>   Newey-West t = {t:.2f}",
                    xref="paper", yref="paper", x=0.0, y=1.02, xanchor="left", yanchor="bottom",
                    showarrow=False, align="left", bgcolor="white", bordercolor="#bdbdbd",
                    borderpad=6, font=dict(size=17))
        subtitle = (f"{WEIGHTING_LABELS[w]}, monthly rebalanced, {months.min():%Y-%m} to "
                    f"{months.max():%Y-%m}; whiskers = 95% CI from Newey-West (6 lags) "
                    "standard errors")
        views.append((finding_title(signal, spread, t), subtitle, note))
    title, subtitle, note = views[0]
    high = float((est["ann_mean"] + est["ci"]).max()) * 100
    low = float((est["ann_mean"] - est["ci"]).min()) * 100
    fig.update_layout(
        annotations=[note], showlegend=False, margin=dict(t=170),
        yaxis=dict(title="Annualised mean return (%)", range=[min(0.0, low) * 1.08, high * 1.06]),
        xaxis=dict(title=f"{portfolios.SIGNAL_LABELS.get(signal, signal)} bucket "
                         "(low to high quality)"),
        updatemenus=[_weighting_menu([
            ([j == i for j in range(len(views))], {**title_layout(t_, s_), "annotations": [n_]})
            for i, (t_, s_, n_) in enumerate(views)
        ])],
    )
    return styled(fig, title, subtitle)


def _weighting_menu(states: list[tuple[list[bool], dict[str, object]]]) -> dict[str, object]:
    buttons = [
        dict(label=WEIGHTING_LABELS[w], method="update", args=[{"visible": visible}, layout])
        for w, (visible, layout) in zip(stats.WEIGHTINGS, states)
    ]
    return menu(buttons)


def menu(buttons: list[dict[str, object]]) -> dict[str, object]:
    """Dropdown at the top right, above the plot area and clear of the subtitle."""
    return dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.02,
                yanchor="bottom", bgcolor="white", bordercolor="#bdbdbd", font=dict(size=15))


# --- 3. cumulative long-short ---------------------------------------------------------------


def cumulative_wealth(ls: pd.DataFrame, signal: str, weighting: str) -> pd.Series:
    """Growth of $1 in the long-short, starting at 1.0 on the first formation date."""
    g = ls[ls["signal"] == signal].sort_values("month")
    r = g.set_index("month")[weighting].dropna()
    start = pd.Series([1.0], index=[g["t"].min()])
    return pd.concat([start, (1 + r).cumprod()])


LOG_TICK_CANDIDATES = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0)


def log_ticks(low: float, high: float) -> list[float]:
    """Round tick values covering [low, high] on a log axis."""
    inside = [v for v in LOG_TICK_CANDIDATES if low / 1.15 <= v <= high * 1.15]
    return inside or [1.0]


def cumulative_long_short(ls: pd.DataFrame) -> go.Figure:
    """One line per signal: growth of $1 in its top-minus-bottom portfolio (log scale), with
    the 2020 COVID recession shaded and an EW / VW dropdown."""
    signals = [s for s in portfolios.SIGNAL_ORDER if s in set(ls["signal"])]
    fig = go.Figure()
    views = []
    all_values: list[float] = []
    for i, w in enumerate(stats.WEIGHTINGS):
        finals = {}
        for s in signals:
            wealth = cumulative_wealth(ls, s, w)
            finals[s] = wealth.iloc[-1]
            all_values += [wealth.min(), wealth.max()]
            fig.add_trace(go.Scatter(
                x=wealth.index, y=wealth.to_numpy(), mode="lines", name=portfolios.SIGNAL_LABELS[s],
                line=dict(color=SIGNAL_COLORS.get(s), width=3 if s == "composite" else 2),
                visible=(i == 0), hovertemplate="%{x|%Y-%m}: &#36;%{y:.2f}<extra>" + s + "</extra>",
            ))
        best, worst = max(finals, key=finals.get), min(finals, key=finals.get)
        title = (f"{portfolios.SIGNAL_LABELS[best]} turned {dollars(1)} into "
                 f"{dollars(finals[best])}; {portfolios.SIGNAL_LABELS[worst]} shrank it to "
                 f"{dollars(finals[worst])}")
        months = ls["month"]
        subtitle = (f"{WEIGHTING_LABELS[w]} high-minus-low quality portfolios, monthly rebalanced, "
                    f"before costs, {months.min():%Y-%m} to {months.max():%Y-%m}; log scale")
        views.append((title, subtitle))
    n = len(signals)
    fig.add_vrect(x0="2020-02-01", x1="2020-04-30", fillcolor="#9e9e9e", opacity=0.2, line_width=0,
                  annotation_text="COVID recession<br>(NBER, Feb-Apr 2020)",
                  annotation_position="top left", annotation_font=dict(size=13, color=MUTED))
    fig.add_hline(y=1, line=dict(color="#9e9e9e", width=1))
    ticks = log_ticks(min(v for v in all_values), max(v for v in all_values))
    fig.update_layout(
        yaxis=dict(type="log", title="Growth of &#36;1", tickvals=ticks,
                   ticktext=[dollars(v) for v in ticks]),
        legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
        margin=dict(r=260, t=170),
        updatemenus=[_weighting_menu([
            ([k // n == i for k in range(n * len(stats.WEIGHTINGS))], title_layout(*views[i]))
            for i in range(len(stats.WEIGHTINGS))
        ])],
    )
    return styled(fig, *views[0])


# --- 4. red-flag radar ----------------------------------------------------------------------


def main_sample(scored: pd.DataFrame) -> pd.DataFrame:
    """Scores outside financials and utilities, one row per firm and fiscal year (latest
    period end if a fiscal year has two, e.g. after a change of year end)."""
    main = scored[scored["excluded_from_main"].eq(False).fillna(False)]
    return main.sort_values("period_end").drop_duplicates(["cik", "fiscal_year"], keep="last")


def red_flag_percentiles(scored: pd.DataFrame, fiscal_year: int) -> pd.DataFrame:
    """Cross-sectional percentile (average rank / n among firms with the value) of each
    red-flag measure in ``fiscal_year``, by ticker; 0.9 = higher than about 90% of firms."""
    year = main_sample(scored)
    year = year[year["fiscal_year"] == fiscal_year].set_index("ticker")
    return pd.DataFrame({label: year[col].rank(pct=True) for label, col in RED_FLAGS.items()})


def _closed(values: pd.Series) -> tuple[list[float | None], list[str]]:
    r = [None if pd.isna(v) else float(v) for v in values]
    return r + r[:1], list(values.index) + [values.index[0]]


def red_flag_radar(
    ticker: str, fiscal_year: int, scored: pd.DataFrame | None = None, prior_year: bool = True
) -> go.Figure:
    """Radar of the 8 Beneish indices and accruals as the firm's percentile in that fiscal
    year, over the cross-sectional median (dashed); optionally the prior year too."""
    scored = pd.read_parquet(scores.SCORES_PATH) if scored is None else scored
    now = red_flag_percentiles(scored, fiscal_year)
    if ticker not in now.index:
        raise ValueError(f"{ticker} has no FY{fiscal_year} scores in the main sample "
                         "(Beneish M is not computed for financials and utilities)")
    row = main_sample(scored).query("ticker == @ticker and fiscal_year == @fiscal_year").iloc[-1]
    m = row["m_score"]
    flagged = pd.notna(m) and m > M_THRESHOLD
    fig = go.Figure()
    r, theta = _closed(now.median())
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, name="Median firm", mode="lines",
                                  line=dict(color=MUTED, dash="dash", width=2)))
    before = red_flag_percentiles(scored, fiscal_year - 1) if prior_year else None
    if before is not None and ticker in before.index:
        r, theta = _closed(before.loc[ticker])
        fig.add_trace(go.Scatterpolar(r=r, theta=theta, name=f"FY{fiscal_year - 1}",
                                      mode="lines+markers",
                                      line=dict(color="#90a4ae", width=2, dash="dot")))
    r, theta = _closed(now.loc[ticker])
    color = QUINTILE_COLORS[0] if flagged else "#1f4e79"
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, name=f"FY{fiscal_year}", mode="lines+markers",
                                  fill="toself", line=dict(color=color, width=3),
                                  fillcolor=_rgba(color, 0.18)))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0], gridcolor=GRID,
                                   angle=90, tickfont=dict(size=12, color=MUTED)),
                   angularaxis=dict(gridcolor=GRID, tickfont=dict(size=16)), bgcolor="white"),
        legend=dict(x=1.02, y=0.9), margin=dict(t=170),
    )
    return styled(fig, _radar_title(ticker, fiscal_year, m), _radar_subtitle(now.loc[ticker]))


def _radar_title(ticker: str, fiscal_year: int, m: float) -> str:
    if pd.isna(m):
        return f"{ticker} FY{fiscal_year}: M-score not available (inputs missing)"
    side = "above" if m > M_THRESHOLD else "below"
    verdict = "flagged as a likely manipulator" if m > M_THRESHOLD else "not flagged"
    return (f"{ticker} FY{fiscal_year}: M-score {m:.2f}, {side} the {M_THRESHOLD} threshold "
            f"({verdict})")


def _radar_subtitle(percentiles: pd.Series) -> str:
    missing = [k for k, v in percentiles.items() if pd.isna(v)]
    gap = f"; not available: {', '.join(missing)}" if missing else ""
    return ("Percentile vs S&P 500 non-financials that year (1 = most suspicious); "
            f"dashed = median firm{gap}")


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def radar_selector(scored: pd.DataFrame, tickers: Sequence[str], fiscal_year: int) -> go.Figure:
    """Pre-rendered radars for ``tickers`` in one figure with a ticker dropdown."""
    radars = [red_flag_radar(t, fiscal_year, scored) for t in tickers]
    fig = go.Figure()
    owner = []
    for i, radar in enumerate(radars):
        for trace in radar.data:
            trace.visible = i == 0
            fig.add_trace(trace)
            owner.append(i)
    buttons = [
        dict(label=t, method="update", args=[
            {"visible": [o == i for o in owner]},
            title_layout(radar.layout.title.text, radar.layout.title.subtitle.text),
        ])
        for i, (t, radar) in enumerate(zip(tickers, radars))
    ]
    fig.update_layout(radars[0].layout)
    fig.update_layout(updatemenus=[menu(buttons)])
    return fig


# --- 5. score timeline ----------------------------------------------------------------------


def annual_report_filings(filings: pd.DataFrame) -> pd.DataFrame:
    """10-K filings (``kind`` '10-K') and their amendments ('10-K/A') from the submissions
    data, with ``date`` = filing date."""
    forms = filings["form"]
    kind = np.select(
        [forms.isin(ANNUAL_FORMS), forms.isin(AMENDMENT_FORMS)], ["10-K", "10-K/A"], ""
    )
    out = filings.assign(kind=kind, date=pd.to_datetime(filings["filingDate"]))
    return out[out["kind"] != ""][["date", "form", "kind"]].sort_values("date", ignore_index=True)


def score_timeline(ticker: str, scored: pd.DataFrame, filings: pd.DataFrame) -> go.Figure:
    """F, Z, M and accruals on four stacked panels, each value plotted at the date it became
    usable (``availability_date``), with a line at every 10-K filing and amendment."""
    firm = scored[scored["ticker"] == ticker].sort_values("availability_date")
    if firm.empty:
        raise ValueError(f"no scores for {ticker}")
    reports = annual_report_filings(filings)
    reports = reports[reports["date"] >= firm["availability_date"].min() - pd.Timedelta(days=400)]
    panels = [
        ("f_score", "Piotroski F", []), ("z_score", "Altman Z", [Z_DISTRESS, Z_SAFE]),
        ("m_score", "Beneish M", [M_THRESHOLD]), ("accruals_cf", "Accruals (cash flow)", [0.0]),
    ]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[label for _, label, _ in panels])
    for i, (col, label, refs) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(
            x=firm["availability_date"], y=firm[col], mode="lines+markers", name=label,
            line=dict(color="#1f4e79", width=2.5), marker=dict(size=7), showlegend=False,
            customdata=firm["fiscal_year"],
            hovertemplate="FY%{customdata}: %{y:.2f}<extra></extra>",
        ), row=i, col=1)
        for y in refs:
            fig.add_hline(y=y, line=dict(color=QUINTILE_COLORS[0], dash="dash", width=1),
                          row=i, col=1)
    styles = {"10-K": dict(color="#9e9e9e", dash="dot", width=1),
              "10-K/A": dict(color=QUINTILE_COLORS[0], dash="dash", width=2)}
    for date, kind in zip(reports["date"], reports["kind"]):
        fig.add_vline(x=date, line=styles[kind], row="all", col=1)
    for kind, line in styles.items():  # legend entries for the vertical markers
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=line,
                                 name=f"{kind} filed" if kind == "10-K" else "10-K/A amendment"))
    fig.update_layout(legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom"),
                      height=HEIGHT)
    n_amend = int((reports["kind"] == "10-K/A").sum())
    flags = firm.loc[firm["m_score"] > M_THRESHOLD, "fiscal_year"].astype(int).tolist()
    flag_text = (f"Beneish flag in FY{', FY'.join(map(str, flags))}" if flags
                 else "never flagged by Beneish")
    name = firm["name"].iloc[-1] if "name" in firm else ticker
    title = (f"{name} ({ticker}): {flag_text}; {int((reports['kind'] == '10-K').sum())} 10-Ks, "
             f"{n_amend} 10-K/A")
    subtitle = ("Scores plotted when they became usable (filing date). Dotted grey = 10-K, dashed "
                "red = 10-K/A (often Part III governance items, not restatements). Horizontal "
                "references: Altman 1.81 / 2.99, Beneish -1.78, zero accruals")
    return styled(fig, title, subtitle)


# --- dashboard ------------------------------------------------------------------------------


def markdown_table_html(markdown: str, heading: str) -> str:
    """The first markdown table after ``heading`` in ``markdown`` as an HTML table."""
    lines = markdown.splitlines()
    start = lines.index(heading)
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("|"):
            rows.append([html.escape(c.strip()) for c in line.strip().strip("|").split("|")])
        elif rows:
            break
    header, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-: ")]
    head = "".join(f"<th>{c}</th>" for c in header)
    cells = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
    return (f'<div class="table-wrap"><table class="results"><thead><tr>{head}</tr></thead>'
            f"<tbody>{cells}</tbody></table></div>")


DASHBOARD_CSS = f"""
body {{ font-family: {FONT_FAMILY}; color: {TEXT}; background: white; margin: 0; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px 64px; }}
h1 {{ font-size: 30px; margin-bottom: 4px; }} h2 {{ font-size: 22px; margin-top: 40px; }}
p, li {{ font-size: 16px; line-height: 1.5; max-width: 980px; }}
.muted {{ color: {MUTED}; }}
table.results {{ border-collapse: collapse; font-size: 14px; margin: 8px 0 16px; }}
table.results th, table.results td {{ border-bottom: 1px solid {GRID}; padding: 6px 10px;
  text-align: right; }}
table.results th:first-child, table.results td:first-child {{ text-align: left; }}
.figure {{ margin: 12px 0 28px; }}
.table-wrap {{ overflow-x: auto; }}
"""


def dashboard_html(title: str, intro_html: str, sections: Sequence[tuple[str, str]]) -> str:
    """One self-contained page: ``sections`` are (heading, html) pairs; figure html comes from
    ``figure_html``. The only external resource is plotly.js from its CDN."""
    body = "".join(f"<h2>{html.escape(h)}</h2>\n{content}\n" for h, content in sections)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{DASHBOARD_CSS}</style>\n</head>\n"
        f"<body><main>\n<h1>{html.escape(title)}</h1>\n{intro_html}\n{body}</main></body>\n"
        "</html>\n"
    )


def figure_html(fig: go.Figure, first: bool, height: int = 640) -> str:
    """A figure as an embeddable div; only the first one carries the plotly.js CDN tag."""
    div = fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                      default_height=f"{height}px", default_width="100%",
                      config={"displaylogo": False, "responsive": True})
    return f'<div class="figure">{div}</div>'
