# quality-screener

Research project: pull annual financial data for ~500 US companies from SEC EDGAR, compute the
Piotroski F-score, Altman Z-score, Beneish M-score and Sloan accruals ratio, then test whether
high-quality firms outperform low-quality firms using quintile portfolios formed on SEC filing
dates. Portfolio piece for quant/fundamental finance roles: code quality, reproducibility and
honest methodology matter more than speed.

## Non-negotiable rules

These apply to all work in this repo. Do not relax them without an explicit instruction from the user.

1. **POINT-IN-TIME ONLY.** Every fundamental value is "known" from its SEC filing date (the
   `filed` field), never from its period-end date. Any join between fundamentals and prices
   must use **filed date + 1 trading day** as the earliest usable date. Look-ahead bias is the
   worst bug in this project.

2. **Prefer "as-first-reported" values.** When the same fiscal period appears in several
   filings (later 10-Ks repeat prior years as comparatives, sometimes restated), the backtest
   uses the value from the EARLIEST filing. Keep all versions in the raw panel, because the
   restatement case study needs them.
   A value first reported more than 90 days after its row's main 10-K (typically tagged only as
   a comparative in the next year's 10-K) is left out of the PIT panel, NOT allowed to delay
   the whole row; it stays in `panel_all_versions` and is listed in `panel_late_values`.

3. **sec.gov HTTP discipline.** Every HTTP request to sec.gov sends the User-Agent from
   `quality_screener/config.py`, is rate-limited (`MAX_REQUESTS_PER_SEC`), retried with
   backoff on 429/5xx, and cached to `data/raw` so re-runs never re-download.

4. **Missing data stays NaN. Never fill with 0.** A score component that cannot be computed is
   NaN, and the score records how many components were available.

5. **Every module gets pytest tests.** Scoring formulas are tested against a hand-calculated
   example.

6. **Use plotly for all charts**; save PNG and HTML to `reports/figures`.

7. **Keep functions small and typed**; docstrings cite the source paper for any formula.

## Layout

```
quality_screener/     package
  config.py           User-Agent, sample years, rate limit, paths
  edgar.py            sec.gov client (rate limit, retry, cache)
  universe.py         company universe (tickers, CIKs)
  xbrl_map.py         XBRL tag -> line-item mapping
  panel.py            firm-year panel, all filed versions + as-first-reported view
  scores.py           F / Z / M / accruals
  prices.py           prices, returns, trading calendar
  portfolios.py       quintile portfolios on filing dates
  stats.py            return statistics and tests (Newey-West, CAPM / factor alphas)
  factors.py          Fama-French 5 factors + momentum (Ken French library, cached; optional)
  viz.py              plotly figures (one house style) + dashboard page helpers
scripts/fetch_all.py  bulk download of companyfacts + submissions (failures -> data/raw/failures.csv)
scripts/build_panels.py  PIT + all-versions panels for the universe, coverage report in reports/
scripts/fetch_prices.py   Yahoo daily prices (cache: data/raw/prices), month-end returns, share check
scripts/compute_scores.py  F / Z / Z'' / M / accruals (market cap from prices) -> scores.parquet
scripts/build_portfolios.py  monthly quintile holdings per signal -> data/processed/holdings.parquet
scripts/run_backtest.py   portfolio returns, long-short stats, robustness -> reports/results.md
scripts/build_dashboard.py  every figure (HTML + PNG) -> reports/figures, reports/dashboard.html
data/sp500_constituents.csv  Wikipedia S&P 500 snapshot (fallback when the scrape fails)
data/raw/             cached API responses (gitignored)
data/processed/       parquet outputs
notebooks/            exploration and case studies
reports/figures/      PNG + HTML charts
tests/                pytest suite
```

## Commands

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/python scripts/fetch_all.py                 # whole universe
.venv/bin/python scripts/fetch_all.py --tickers AAPL  # subset
.venv/bin/python scripts/build_panels.py             # panels + coverage (cache only)
.venv/bin/python scripts/fetch_prices.py             # prices (Yahoo, cached) + share check
.venv/bin/python scripts/compute_scores.py           # scores + summary (needs cached prices)
.venv/bin/python scripts/build_portfolios.py         # holdings (--include-fin-util: robustness)
.venv/bin/python scripts/run_backtest.py             # returns + statistics -> reports/results.md
.venv/bin/python scripts/build_dashboard.py          # figures + dashboard (PNG needs Chrome)
```

Homebrew Python is PEP 668 "externally managed": always install into `.venv`, never system Python.
`requirements-lock.txt` pins the exact versions used; regenerate it with
`.venv/bin/pip freeze --exclude-editable > requirements-lock.txt` after changing dependencies.
