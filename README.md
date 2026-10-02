# quality-screener

Point-in-time accounting quality scores for ~500 US companies, built from SEC EDGAR XBRL data:

- Piotroski F-score (Piotroski, 2000)
- Altman Z-score (Altman, 1968)
- Beneish M-score (Beneish, 1999)
- Sloan accruals ratio (Sloan, 1996)

The research question: do high-quality firms outperform low-quality firms? Quintile portfolios
are formed on SEC **filing** dates (not fiscal period ends), using as-first-reported values, so
the backtest only uses information that was public at the time.

**Status:** project scaffold. No data pipeline or scoring logic yet.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Exact dependency versions are pinned in `requirements-lock.txt`.

## Methodology rules

See [CLAUDE.md](CLAUDE.md) for the rules every part of this project follows (point-in-time
data, as-first-reported values, SEC request etiquette, NaN handling, testing).
