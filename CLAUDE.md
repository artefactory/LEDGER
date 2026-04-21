# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`ardian-dataset-bench` builds a structured dataset from OCRed financial reports (~100,000 reports in an external database). The goal is to cluster companies by sector and identify competitive peer groups for benchmarking tasks. The pipeline starts with ticker extraction → Yahoo Finance enrichment → sector clustering → LLM-based competition validation.

## Commands

This project uses `uv` for dependency management (Python 3.13).

```bash
uv run python tickers_lists/scripts/extract.py       # Extract tickers from file_list.txt
uv run python tickers_lists/scripts/map_tickers.py LSE  # Enrich tickers with yfinance metadata (pass exchange name)
uv run python tickers_lists/scripts/clean_mapped.py  # Drop rows with N/A / Error / empty cells
uv run python tickers_lists/scripts/group_industries.py  # Group cleaned rows by Sector → Industry
uv add <package>                                     # Add a dependency
```

## Architecture

### Data pipeline (`tickers_lists/`)

```
tickers_lists/
├── file_list.txt   # input: one PDF filename per line, format EXCHANGE_TICKER_YEAR.pdf
├── scripts/        # extract.py, map_tickers.py, clean_mapped.py, group_industries.py
├── tickers/        # {EXCHANGE}_tickers.txt    (output of extract.py)
├── mapped/         # {EXCHANGE}_mapped.csv     (output of map_tickers.py)
├── cleaned/        # {EXCHANGE}_mapped_clean.csv (output of clean_mapped.py)
└── grouped/        # {EXCHANGE}/ and all/ subdirs, each with companies_by_industry.json + summary.md
```

Four scripts chain together. All resolve paths relative to `tickers_lists/`, so they can be run from anywhere.

1. **`extract.py`** — Parses `file_list.txt` and writes `tickers/{EXCHANGE}_tickers.txt`. Supported exchanges include NYSE, NASDAQ, AMEX, LSE, AIM, ASX, TSX, TSX-V, OTC.

2. **`map_tickers.py`** — Takes an exchange name as a CLI argument, reads `tickers/{EXCHANGE}_tickers.txt`, fetches `longName`, `sector`, `industry`, and `fullExchangeName` via `yfinance`, and appends row-by-row to `mapped/{EXCHANGE}_mapped.csv`. Designed for resumability: appends incrementally so a restart won't lose progress, but **will duplicate already-fetched rows** if re-run — delete or move the output first. 1 s sleep between requests. The `Exchange (Yahoo)` column lets you detect cases where yfinance redirects a ticker to a different exchange (e.g. LSE → NYSE).

3. **`clean_mapped.py`** — Reads every `mapped/*_mapped.csv` and writes `cleaned/*_mapped_clean.csv` with rows dropped when any cell is `N/A`, `Error`, or empty.

4. **`group_industries.py`** — Groups cleaned rows as Sector → Industry → companies. Writes one grouping *per exchange* under `grouped/{EXCHANGE}/` and a combined grouping under `grouped/all/`, each with `companies_by_industry.json` (nested) and `summary.md` (human-readable). Per-exchange groupings exist because yfinance sometimes resolves LSE tickers to their NYSE-listed equivalents, so the combined view has known duplicates that are easier to spot by comparing the per-exchange outputs. Industry peer groups are the candidates for downstream LLM-based competitor validation.

Mapped CSVs currently exist for AIM, AMEX, LSE, NASDAQ, and NYSE. `main.py` at the repo root is an unused stub.

### Planned next steps (from README/Plan.md)

- Fuse mapped ticker data with the existing OCR report database
- Cluster companies by sector
- Use an LLM to validate whether clustered companies are genuine competitors
