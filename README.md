# ardian-dataset-bench

Build a structured benchmark dataset from a corpus of OCR'd annual reports
by grouping companies into industry peer groups, extracting financial KPIs,
and linking each filing to market reactions.

## Overview

The pipeline has four stages that run in order:

```
1. tickers_lists/      — discover companies, classify by industry, select peer groups
2. doc_text_processing/ — filter 10-K filings, extract CEO / shareholder letters
3. KPI_analysis/        — fetch financial KPIs, build dataset, validate against OCR
4. KPI_analysis/        — link 10-K filings to next-day / next-week market reactions
```

All scripts use `uv` for dependency management (Python 3.13).

---

## Stage 1 — Company discovery & peer-group selection (`tickers_lists/`)

Turns a flat list of PDF filenames (`EXCHANGE_TICKER_YEAR.pdf`) into curated
industry peer groups with a common year window.

```
file_list.txt
  → extract.py              parse filenames → tickers/{EXCHANGE}_tickers.txt
  → map_tickers.py          yfinance enrichment → mapped/{EXCHANGE}_mapped.csv
  → clean_mapped.py         drop N/A rows → cleaned/*_mapped_clean.csv
  → verify_exchange.py      re-query fullExchangeName → *_verified.csv
  → filter_exchange.py      drop cross-exchange redirects → *_mapped_clean.csv
  → group_industries.py     Sector → Industry groupings → grouped/
  → list_selected_industries.py  hand-picked industries → grouped/selected/
  → copy_selected_pdfs.py   copy matched PDFs to output tree
  → year_coverage.py        find best common year window → year_coverage.{md,json}
  → prune_ocr.py            remove OCR subdirs outside the selected window
```

The `verify_exchange` + `filter_exchange` pair is important: yfinance often
silently resolves an LSE ticker to its US-listed counterpart (~72% of LSE rows
redirect), so checking `fullExchangeName` and dropping redirects keeps each
exchange's data honest.

```bash
uv run python tickers_lists/scripts/extract.py
uv run python tickers_lists/scripts/map_tickers.py LSE
uv run python tickers_lists/scripts/clean_mapped.py
uv run python tickers_lists/scripts/verify_exchange.py LSE
uv run python tickers_lists/scripts/filter_exchange.py LSE
uv run python tickers_lists/scripts/group_industries.py
uv run python tickers_lists/scripts/list_selected_industries.py
uv run python tickers_lists/scripts/copy_selected_pdfs.py
uv run python tickers_lists/scripts/year_coverage.py
uv run python tickers_lists/scripts/prune_ocr.py --industry "Auto Parts" --start 2017 --end 2022 --ocr-dir /path/to/ocr
```

Selected industries are hard-coded at the top of `list_selected_industries.py`.
See `Notes.md` for the selection rationale and `grouped/selected/year_coverage.md`
for per-industry year-window results.

---

## Stage 2 — Document text processing (`doc_text_processing/`)

Operates on the OCR'd annual-report tree where each report is a subdirectory
`{EX}_{TICKER}_{YEAR}/` containing a `.mmd` file with pages separated by
`<--- Page Split --->`.

### 10-K classifier (`10K_or_not/classify_10k.py`)

Scans the first few pages of each report for SEC Form 10-K cover-page markers
(e.g. `UNITED STATES SECURITIES AND EXCHANGE COMMISSION`, `Commission File
Number`). Reports that match are flagged as 10-Ks and excluded from the letter
extractor downstream.

Outputs:
- `10K_or_not/classification.json` — per-report match details
- `10K_or_not/is_10k.txt` — plain list of flagged report names

```bash
uv run python doc_text_processing/10K_or_not/classify_10k.py
```

### CEO / shareholder letter extractor (`CEO_word_extraction/extract_letters.py`)

For every non-10-K report, finds section headings that match phrases from
`expressions.txt` (e.g. `Dear Shareholders`, `Letter from the CEO`) and
extracts a configurable page window of text starting from that heading. Handles
OCR artifacts (line-breaks mid-heading, smart quotes), TOC false positives
(heading followed by a page number), and overlapping sections.

Outputs:
- `CEO_word_extraction/extractions.json` — structured per-report records
- `CEO_word_extraction/extractions/` — one markdown file per extracted letter

```bash
uv run python doc_text_processing/CEO_word_extraction/extract_letters.py
```

---

## Stage 3 — KPI extraction & dataset build (`KPI_analysis/`)

Fetches annual consolidated KPIs (revenue, net income, total assets, capex, …)
for all selected companies and consolidates them into analysis-ready CSVs.

Data sources by exchange:

| Exchange | Source | Notes |
|---|---|---|
| NYSE, NASDAQ, AMEX, Cboe | SEC EDGAR (XBRL) | Full history back to ~2009, free |
| LSE, AIM, ASX, TSX, … | yfinance (fallback) | ~4 most recent fiscal years |
| Any (opt-in) | Alpha Vantage (gap-fill) | Fills holes only; 25 calls/key/day |

### Scripts

- **`fetch_kpis.py`** — orchestrator; routes each ticker to the right source,
  resolves XBRL tag ambiguity, writes `output/raw/{TICKER}.json`.
- **`build_dataset.py`** — consolidates raw JSONs into `output/kpis_long.csv`
  (ticker × year × kpi × value) and `output/kpis_wide.csv` (one row per
  ticker × year).
- **`validate_ocr_kpis.py`** — checks whether KPI target values from
  `kpis_long.csv` can be found in the OCR text, using a forward pipeline
  (alias → numeric candidate → unit normalisation) and a reverse pipeline
  (scaled target literal → alias confirmation). Outputs audit CSVs and
  summary markdown under `output/ocr_validation/`.
- **`tags.py`** — XBRL tag definitions; defines the ordered candidate tag list
  per logical KPI. Tag order is load-bearing (scope conventions). Read the
  comments before modifying.
- **`edgar.py`** / **`yf_fallback.py`** / **`alpha_vantage.py`** — source
  clients used by `fetch_kpis.py`.

```bash
# Fetch KPIs for all selected companies, FY 2017-2022:
uv run python KPI_analysis/fetch_kpis.py --selected --years 2017-2022

# Add Alpha Vantage gap-fill on top:
uv run python KPI_analysis/fetch_kpis.py --selected --alphavantage

# Consolidate into CSV:
uv run python KPI_analysis/build_dataset.py

# Validate KPIs against OCR text:
uv run python KPI_analysis/validate_ocr_kpis.py
```

---

## Stage 4 — Filing-date market reactions (`KPI_analysis/fetch_filing_returns.py`)

For every US-listed `(ticker, fiscal year)`, finds the original 10-K filing on
EDGAR, reads its acceptance timestamp (UTC), classifies it as pre-market /
intraday / after-hours / non-trading-day, and computes next-day and next-week
price reactions from yfinance — both raw and SPY-relative.

Output `output/filing_returns.csv` is joinable to `kpis_long.csv` on
`(ticker, year)`.

```bash
uv run python KPI_analysis/fetch_filing_returns.py --selected --years 2017-2022
```

See `KPI_analysis/README.md` for the full event-window definition, output
schema, and error-bucket explanations.

---

## Key data files

| Path | Description |
|---|---|
| `tickers_lists/file_list.txt` | Input: one PDF filename per line |
| `tickers_lists/grouped/selected/companies.json` | Hand-picked industry peer groups |
| `tickers_lists/grouped/selected/year_coverage.md` | Best common year windows per industry |
| `KPI_analysis/output/kpis_long.csv` | (ticker, year, kpi, value) |
| `KPI_analysis/output/kpis_wide.csv` | One row per (ticker, year), KPI columns |
| `KPI_analysis/output/filing_returns.csv` | 10-K filing date + market reaction |
| `KPI_analysis/output/coverage.md` | KPI coverage % per year |

---

## Notes

- `Notes.md` — sector selection rationale, data-quality caveats, and year-window analysis.
- `KPI_analysis/README.md` — full KPI pipeline documentation: tag ambiguity, Alpha Vantage setup, filing-returns event-window convention.
- `SEC_USER_AGENT` env var overrides the default EDGAR request header
  (`"ardian-dataset-bench research (charles.moslonka@artefact.com)"`).
