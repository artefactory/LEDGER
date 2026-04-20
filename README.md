# ardian-dataset-bench

Build a structured benchmark dataset from a corpus of OCR'd annual reports by
grouping companies into industry peer groups, restricted to a common year
window.

## Pipeline

Scripts live in `tickers_lists/scripts/` and chain together through
subdirectories of `tickers_lists/`:

```
file_list.txt → extract.py → tickers/
             → map_tickers.py <EXCHANGE> → mapped/
             → clean_mapped.py → cleaned/*_mapped_clean.csv
             → verify_exchange.py <EXCHANGE> → cleaned/*_mapped_clean_verified.csv
             → filter_exchange.py <EXCHANGE> → cleaned/*_mapped_clean.csv (filtered)
             → group_industries.py → grouped/{EXCHANGE}/, grouped/all/
             → list_selected_industries.py → grouped/selected/
             → copy_selected_pdfs.py → /data/.../annual_reports_pdfs_selected/
             → year_coverage.py → grouped/selected/year_coverage.{md,json}
```

The `verify_exchange` + `filter_exchange` pair exists because yfinance often
resolves a ticker to a different exchange than the one we queried (e.g. LSE
`AAL` → NASDAQ `American Airlines`). Verifying with `fullExchangeName` and
dropping the redirects keeps each exchange's rows honest.

## Usage

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
```

The selected industries are hard-coded at the top of
`list_selected_industries.py`. See `Notes.md` for the selection rationale,
known data-quality caveats, and suggested year windows.
