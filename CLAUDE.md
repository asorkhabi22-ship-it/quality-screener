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
  stats.py            return statistics and tests
  viz.py              plotly figures
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
```

Homebrew Python is PEP 668 "externally managed": always install into `.venv`, never system Python.
`requirements-lock.txt` pins the exact versions used; regenerate it with
`.venv/bin/pip freeze --exclude-editable > requirements-lock.txt` after changing dependencies.
