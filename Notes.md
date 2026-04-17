Note for the sector selection : 

quantitative investing approach. Industries that resemble "perfect competition", companies "price takers" rather than "price makers." Lack power to dictate terms to the market, so stock performance is heavily dictated by macroeconomic tides (blocked Suez canal, interest rate hikes, or raw material shortages) and how efficiently management can execute on the margins (the KPIs).

Subject to Physical Supply Chain & Inflation Shocks:

Consumer Cyclical / Auto Parts (Sensitive to global freight, raw materials, and factory shutdowns)

Basic Materials / Specialty Chemicals (Sensitive to industrial demand, energy feedstock prices, and shipping logistics)

Consumer Defensive / Packaged Foods (Sensitive to agricultural commodity prices and domestic freight costs)

Subject to Global Commodity Prices & Capital Cycles:

Energy / Oil & Gas E&P (Pure price takers driven entirely by the global spot price of crude/natural gas)

Energy / Oil & Gas Equipment & Services (Driven entirely by the capital expenditure budgets of the E&P sector)

Subject to Interest Rates & Monetary Policy Shocks:

Financial Services / Banks - Regional (Driven by central bank rates, yield curves, and local economic health)

Real Estate / REIT - Mortgage (Driven by borrowing costs and highly sensitive to interest rate volatility)

To put this in perspective, some industries experience very low direct competition because of high differentiation or structural moats:

Utilities / Utilities - Regulated Electric: The exact opposite of narrow competition. They are usually natural monopolies granted exclusive rights to a geographic area. They face virtually zero direct competition for their clients.

Healthcare / Biotechnology: While highly competitive to get a drug to market, it is not "narrow" in terms of products. Biotech relies heavily on patents. If one company cures a specific type of leukemia and another treats rheumatoid arthritis, they aren't competing for the same patient at all.

---

## Approach summary

Goal: turn the ~6k OCR'd annual reports into a structured dataset of peer groups that actually compete, restricted to a common time window so cross-company comparisons are well-defined.

### Pipeline (in `tickers_lists/scripts/`)

1. **`extract.py`** — parses PDF filenames (`EXCHANGE_TICKER_YEAR.pdf`) into one `{EXCHANGE}_tickers.txt` per exchange. Ground truth for what we actually have on disk.
2. **`map_tickers.py`** — enriches each ticker via yfinance (`longName`, `sector`, `industry`) into `mapped/{EXCHANGE}_mapped.csv`. Incremental append so interruptions don't lose work; 1 s sleep to respect rate limits.
3. **`clean_mapped.py`** — drops rows where any of the four columns is `N/A`, `Error`, or empty. Result: 4,091 companies (from 7,038 tickers) in `cleaned/`.
4. **`group_industries.py`** — pivots rows into Sector → Industry → [companies]. Produces one grouping per exchange (`grouped/{EXCHANGE}/`) and a combined one (`grouped/all/`). The per-exchange split is there because yfinance often resolves LSE tickers to NYSE-listed counterparts, inflating the "all" view with duplicates; comparing per-exchange groupings makes the overlap visible.
5. **`list_selected_industries.py`** — hand-picked (Sector, Industry) pairs (the 7 above) are materialized into `grouped/selected/companies.{json,md}`, broken down by exchange.
6. **`copy_selected_pdfs.py`** — copies the matching PDFs out of the raw corpus into `/data/.../annual_reports_pdfs_selected/{industry-slug}/`. Idempotent: skips files already present at the destination. 7,941 PDFs (~31 GB) ready for OCR.
7. **`year_coverage.py`** — for each industry, sweeps every consecutive year window and reports, per window size k, the window of k years with the maximum number of companies having reports in *all* those years. Output: `grouped/selected/year_coverage.{md,json}`.

### Why industry, not sector

Sectors (11 of them) are too broad — "Financial Services" lumps banks, asset managers, insurers, card networks. Industries (143) are the level at which "do these companies compete?" starts to be a defensible question. They're still not perfect — e.g. Banks - Regional has 287 companies in the combined view (229 on NASDAQ alone), which is more of a universe than a peer group — so LLM-based validation will still be needed within the larger industries.

### Known issues

- **LSE → NYSE ticker collisions from yfinance.** The mapped LSE CSV contains rows whose company info is actually for the US-listed namesake. Per-exchange groupings help spot these; the combined `all/` grouping shouldn't be trusted without deduping by company name.
- **Ticker = company assumption breaks for dual listings.** Currently `(exchange, ticker)` is the company key. Same company listed on two exchanges counts twice in coverage stats.
- **`group_industries.py` output sizes are skewed by listing density,** not genuine competitor count. NASDAQ-heavy industries look larger than they really are for competitor-set purposes.

### Candidate common time windows

From `year_coverage.py`, the per-industry best-(companies × years) windows all land in roughly the 2007–2023 range, and `2017–2022` is a reasonable all-industry common window — it's the optimum for Oil & Gas Equipment & Services (the weakest coverage of the seven) and still retains 25–40% of docs and 25+ companies in every other industry. Narrower windows (e.g. 2020–2022) keep nearly all companies but fewer reports per company; wider windows (2011–2022) keep more reports but drop the smaller industries. Final choice of window should be taken after deduplication.

### Next steps

- Dedupe companies across exchanges (likely by `longName`) before finalizing peer sets.
- Trim over-populated industries (Banks - Regional especially) down to a defensible competitor set — LLM validation, market-cap filter, or geography filter.
- Freeze a common year window, drop companies that don't fully cover it, and produce the final OCR queue.
- After OCR, fuse the text with the Sector/Industry/Year metadata as the actual benchmark dataset.