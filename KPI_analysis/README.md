# KPI_analysis

Fetches annual consolidated KPIs (revenue, net income, total assets, capex, …)
for the companies we've mapped in `tickers_lists/`. Designed to produce one
point per company × year × KPI so downstream scripts can compare peers.

## Design

Hybrid pipeline — we pick the data source based on the listing exchange, with
an optional gap-fill on top:

| Exchange                                              | Source                   | Why                                                  |
| ----------------------------------------------------- | ------------------------ | ---------------------------------------------------- |
| NYSE, NYSE American (AMEX), Nasdaq (GS/GM/CM), Cboe   | **SEC EDGAR** companyfacts | Free, unlimited, full history (XBRL back to ~2009). |
| LSE, AIM, ASX, TSX, …                                 | **yfinance** (fallback)  | No SEC filings. yfinance covers ~4 recent fiscal years. |
| any (opt-in, `--alphavantage`)                        | **Alpha Vantage** (gap-fill) | Fills missing (KPI, year) cells per ticker. Rate-limited (25/key/day) so reserved for low-coverage tickers. |

EDGAR is the preferred source because it publishes structured XBRL for every
10-K filing, so we can pull 6+ years of history without hitting rate limits.
For non-US listings we fall back to yfinance (the same library we already use
in `tickers_lists/scripts/map_tickers.py`); coverage is shallower and the field
labels are less stable, but it's enough for the later years in our corpus.

Alpha Vantage is a *third*, optional layer that runs after EDGAR/yfinance and
fills holes only — it never overwrites values. Useful for US tickers whose
EDGAR coverage starts mid-window (e.g. APA, post-2021 holding-co reorg) and
some non-US tickers AV happens to cover.

## Files

```
KPI_analysis/
├── tags.py                       # logical KPI -> candidate XBRL tags (ordered by preference)
├── edgar.py                      # SEC EDGAR companyfacts client (CIK lookup, XBRL parsing)
├── edgar_filings.py              # SEC EDGAR submissions client (10-K filing dates + acceptanceDateTime)
├── yf_fallback.py                # yfinance fallback for non-US tickers
├── alpha_vantage.py              # Alpha Vantage gap-fill: keys + budget + 3-statement client
├── alpha_venture_API_keys.txt    # one AV API key per line (gitignored, *.txt)
├── fetch_kpis.py                 # orchestrator CLI; writes output/raw/{TICKER}.json
├── fetch_filing_returns.py       # 10-K filing date + market reaction (next-day / next-week / SPY-alpha)
├── build_dataset.py              # consolidates output/raw/*.json into long + wide CSVs
├── validate_ocr_kpis.py          # pilot: validate EDGAR KPI values against OCR text
├── cache/                        # ticker->CIK map, cached SEC + AV responses, AV budget (gitignored)
│   ├── ticker_cik.json
│   ├── companyfacts/CIK*.json
│   ├── submissions/CIK*.json     # EDGAR submissions JSON (+ older shards) for filing dates
│   ├── prices/{TICKER}.csv       # cached yfinance daily OHLC for filing-returns
│   ├── alphavantage/{SYMBOL}__{ENDPOINT}.json
│   └── alphavantage_budget.json  # per-key per-UTC-day call counts
└── output/
    ├── raw/                      # one JSON per ticker
    ├── kpis_long.csv             # (ticker, year, kpi, value) long form
    ├── kpis_wide.csv             # (ticker, year) rows × KPI columns
    ├── coverage.md               # coverage % per KPI per year
    └── filing_returns.csv        # one row per (ticker, year) with 10-K filing date + market reaction
```

## Setup

The only new dependency is `requests`, which is already pulled in transitively
by `yfinance`. No extra `uv add` needed.

SEC requires every request to carry a descriptive `User-Agent` header with
contact info (see https://www.sec.gov/os/accessing-edgar-data). We default to
`"ardian-dataset-bench research (charles.moslonka@artefact.com)"`. Override by
exporting:

```bash
export SEC_USER_AGENT="Your Name your@email"
```

SEC throttles at 10 req/s; we self-limit to ~2 req/s. Responses are cached in
`KPI_analysis/cache/companyfacts/CIK*.json` so re-runs are local-disk cheap.

## Usage

```bash
# All companies in the selected industries (tickers_lists/grouped/selected/):
uv run python KPI_analysis/fetch_kpis.py --selected --years 2017-2022

# A single industry:
uv run python KPI_analysis/fetch_kpis.py --industry "Consumer Cyclical / Auto Parts"

# An explicit list:
uv run python KPI_analysis/fetch_kpis.py --tickers ORLY AZO GPC --years 2017-2022

# A whole cleaned CSV:
uv run python KPI_analysis/fetch_kpis.py --csv tickers_lists/cleaned/NYSE_mapped_clean_verified.csv

# Add the Alpha Vantage gap-fill on top of any of the above:
uv run python KPI_analysis/fetch_kpis.py --selected --alphavantage

# Consolidate into CSVs:
uv run python KPI_analysis/build_dataset.py
```

## Alpha Vantage gap-fill (`--alphavantage`)

Opt-in third layer that runs after EDGAR/yfinance and fills missing
(KPI, year) cells. Each call to AV's `INCOME_STATEMENT`, `BALANCE_SHEET`,
or `CASH_FLOW` returns the *full multi-year annual history* for a ticker —
so 3 calls cover ~25 KPIs across all years in one go.

### Keys & budget

API keys live in `KPI_analysis/alpha_venture_API_keys.txt` — one key per line,
blank lines and `#` comments are skipped, duplicates are dropped. Adding or
removing keys requires no code change; the file is re-read each run. Budget
scales linearly: `daily_budget = N_keys × 25 calls/day`.

Per-key per-day usage is tracked in `cache/alphavantage_budget.json`, keyed on
UTC date (matches AV's reset). Within a run we always pick the key with the
most remaining quota; when all keys are exhausted, the run stops gracefully.

### Prioritisation

Tickers are scored by the number of *AV-fillable* (KPI × year) cells still
missing after EDGAR/yfinance. Worst-coverage tickers run first, until the
daily budget runs out or every eligible ticker has been processed. Use
`--alphavantage-min-shortfall N` to skip tickers that are already nearly
complete.

### Caching & idempotency

Responses are cached on disk under `cache/alphavantage/{SYMBOL}__{ENDPOINT}.json`.
Re-running the fallback re-reads the cache (0 quota). Pass
`--alphavantage-refresh` to bypass the cache.

### CLI flags

| Flag | Purpose |
|---|---|
| `--alphavantage` | Enable the fallback (off by default). |
| `--alphavantage-keys PATH` | Override the keys file path. |
| `--alphavantage-daily-quota N` | Per-key quota (default 25, AV free tier). |
| `--alphavantage-budget N` | Cap total live calls in this run. |
| `--alphavantage-min-shortfall N` | Skip tickers below this missing-cell count. |
| `--alphavantage-include-earnings` | Also fetch EARNINGS (gives EPS, costs +1 call/ticker). |
| `--alphavantage-refresh` | Bypass the on-disk response cache. |

### What gets recorded

For every ticker AV touched:

```json
{
  "source": "edgar + alphavantage",
  "alphavantage": {
    "symbol_used": "APA",
    "reported_currency": {"INCOME_STATEMENT": "USD", "BALANCE_SHEET": "USD", "CASH_FLOW": "USD"},
    "endpoints_called": ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"],
    "cells_added": 86
  },
  "alphavantage_tag_used": {
    "revenue": "alphavantage:INCOME_STATEMENT.totalRevenue",
    "...": "..."
  }
}
```

### Coverage caveats

- AV's coverage is strongest for US-listed equities. LSE / AIM / ASX coverage
  is uneven; symbols are translated `<TICKER>.L` → `<TICKER>.LON` etc., but
  many non-US tickers return empty `{}` (the call still costs quota).
- AV's `netIncome` is parent-attributable and `totalShareholderEquity` is
  parent-only — same scope conventions baked into `tags.py`.
- AV's `capitalExpenditures` is reported as a positive cash outflow (matches
  EDGAR sign), while yfinance's `Capital Expenditure` is negative; if a
  ticker's row mixes sources, signs may differ.
- The reported currency is preserved in `alphavantage.reported_currency` but
  values are stored as-reported (no FX conversion).

## Filing-date market reaction (`fetch_filing_returns.py`)

Links each annual report to the market's reaction at publication. For every
US-listed `(ticker, fiscal_year)` we resolve the *original* 10-K filing on
EDGAR, read its `acceptanceDateTime` (the moment EDGAR accepted the
submission, in **UTC** despite the trailing `Z` — verified empirically), and
compute next-day and next-week price reactions from yfinance, both raw and
SPY-relative.

```bash
# All US-listed selected companies, FY2017-2022:
uv run python KPI_analysis/fetch_filing_returns.py --selected --years 2017-2022

# Single industry (filters to its US listings):
uv run python KPI_analysis/fetch_filing_returns.py \
  --industry "Consumer Cyclical / Auto Parts" --years 2017-2022

# Explicit tickers (US assumed unless suffix like .L):
uv run python KPI_analysis/fetch_filing_returns.py --tickers AZO ORLY AAP

# Skip benchmark (alpha columns will be empty):
uv run python KPI_analysis/fetch_filing_returns.py --selected --no-benchmark
```

LSE/AIM and other non-US listings are filtered out by default — they have no
EDGAR equivalent, and a per-ticker manual `filing_dates_lse.csv` is the
intended next layer (deferred). Pass `--include-non-us` to include them
anyway (every row will surface `error="no CIK in EDGAR ticker map"`).

### Event-window convention

We model the market reaction as the move from the *last close before* the
news was public to the *first close after* it:

  - **t0** = last trading day with a close BEFORE the filing was public
  - **t1** = first trading day with a close AFTER the filing was public
  - **t5** = `t1 + 4` trading days (5-day window measured from t0)
  - `r_1d = Close[t1] / Close[t0] - 1`
  - `r_5d = Close[t5] / Close[t0] - 1`
  - `a_Nd = r_Nd - spy_r_Nd` (SPY-relative alpha; `--benchmark` overrides)

The filing time of day matters because most 10-Ks land after market close.
We classify each filing into a `filing_window_class`:

| Class             | Trigger (acceptance time in ET)                      | t1 anchor              |
| ----------------- | ---------------------------------------------------- | ---------------------- |
| `pre_market`      | Trading day, before 09:30                            | Same day's close       |
| `intraday`        | Trading day, 09:30 - 16:00                           | Same day's close       |
| `after_hours`     | Trading day, after 16:00                             | Next trading day       |
| `non_trading_day` | Weekend / holiday                                    | Next trading day       |

In our full --selected run, ~60% of filings are after-hours, ~29% intraday,
~12% pre-market, <1% on non-trading days.

### Output schema (`output/filing_returns.csv`)

One row per `(ticker, year)`. Columns:

| Column                  | Notes                                                              |
| ----------------------- | ------------------------------------------------------------------ |
| `ticker`, `year`        | Joinable to `kpis_long.csv` / `kpis_wide.csv`.                     |
| `accession`             | EDGAR accession number (the 10-K filing's primary key).            |
| `form`                  | `10-K` or `20-F` (foreign private issuers).                        |
| `filing_date`           | EDGAR's `filingDate` — the *official* filing day (ET).             |
| `report_date`           | Period of report (fiscal year-end date). Used for FY keying.       |
| `acceptance_dt_utc`     | EDGAR's `acceptanceDateTime` (UTC, exact to the second).           |
| `acceptance_dt_et`      | Same instant in America/New_York.                                  |
| `filing_window_class`   | See table above.                                                   |
| `has_amendment`         | `True` when a 10-K/A exists for the same fiscal year.              |
| `t0`, `t1`, `t5`        | Anchored trading days (YYYY-MM-DD).                                |
| `close_t0/1/5`          | Auto-adjusted close prices on each anchor.                         |
| `r_1d`, `r_5d`          | Raw returns (decimals: 0.0123 = +1.23%).                           |
| `spy_r_1d`, `spy_r_5d`  | SPY return over the same anchor pair.                              |
| `a_1d`, `a_5d`          | Alpha vs SPY (`r_Nd - spy_r_Nd`).                                  |
| `error`                 | Populated when the row could not be computed (see below).          |

### Why some rows have errors

| Error bucket                        | Typical cause                                                            |
| ----------------------------------- | ------------------------------------------------------------------------ |
| `no original 10-K for FY{year}`     | 52/53-week fiscal-year edge — period ends in early Jan, lands one year over. Most common. |
| `no yfinance prices`                | Delisted / merged / very thin liquidity (e.g. some MLPs after wind-down).|
| `insufficient price history around filing` | Trading halt or post-IPO with not enough pre-news data. |
| `no CIK in EDGAR ticker map`        | US-listed but not in SEC's company_tickers map (rare; ADRs).             |
| `no 10-K/20-F on EDGAR`             | Edge cases (foreign issuers that file 6-K only, etc.).                   |

### Caching

  - `cache/submissions/CIK{cik}.json` — main submissions JSON (last ~1000 filings).
  - `cache/submissions/CIK{cik}-submissions-NNN.json` — older-filings shards, fetched on demand.
  - `cache/prices/{ticker}.csv` — yfinance daily OHLC over the requested window
    plus 30d head / 15d tail, so boundary t0/t5 lookups never run off the edge.

Re-runs read entirely from cache; pass `--refresh-cache` to force re-download.

### Joining with KPIs

Both `filing_returns.csv` and `kpis_long.csv` use the same `(ticker, year)`
key, so the canonical join is a one-liner:

```python
import pandas as pd
kpis = pd.read_csv("KPI_analysis/output/kpis_long.csv")
returns = pd.read_csv("KPI_analysis/output/filing_returns.csv")
joined = kpis.merge(returns, on=["ticker", "year"], how="inner")
```

The duplication is intentional — KPIs are wide (one column per metric per
year per ticker), market reactions are narrow (one event per ticker per
year). Keeping them in separate tables avoids a giant cross-join.

### A note on amendments (10-K/A)

We deliberately use the *original* 10-K's filing instant, not the latest
amendment's. Original-10-K reactions are the cleaner publication-day signal;
amendments often fix errors months later and the reaction is much smaller
(and noisier). The `has_amendment` flag lets you exclude or treat
separately the 8% of fiscal years that were later restated.

## OCR validation pilot

`validate_ocr_kpis.py` is a first-pass validator that checks whether selected
KPI target values (from `output/kpis_long.csv`) can be found in OCR annual
reports with two complementary pipelines:

- **Forward pipeline**: alias-first extraction + unit detection from OCR context
  (thousands/millions/billions -> single dollars).
- **Reverse pipeline**: value-first search using scaled targets
  (`target / divisor`, by default 1, 1e3, 1e6, 1e9), then validating against
  the original single-dollar target.
- **Merged pipeline**: combines forward + reverse decisions into one final
  status per `(report, ticker, year, kpi)`.

Default pilot scope:

- OCR root: `sample_data/subset_auto_parts_2017_2022/`
- KPI set: `revenue`, `gross_profit`, `operating_income`, `net_income`,
  `total_assets`, `total_liabilities`, `cash_and_equivalents`,
  `operating_cash_flow`, `capex`
- Match tolerance: +/-1% relative error

Run:

```bash
# Full pilot on sample_data subset
uv run python KPI_analysis/validate_ocr_kpis.py

# Fast smoke test
uv run python KPI_analysis/validate_ocr_kpis.py --max-reports 5 --max-targets 80

# Reverse tuning example
uv run python KPI_analysis/validate_ocr_kpis.py \
  --reverse-divisors 1,1000,1000000,1000000000 \
  --reverse-literal-tolerance 0.01

# Keep only the original forward pipeline
uv run python KPI_analysis/validate_ocr_kpis.py --disable-reverse
```

Outputs are written under `KPI_analysis/output/ocr_validation/`:

- `targets_pilot.csv` (final target table used in the run)
- `audit_rows.csv` (one row per report/ticker/year/KPI target with best evidence)
- `candidates.csv` (all extracted numeric candidates near KPI aliases)
- `coverage_kpi_year.csv` (KPI-year match/ambiguous/unmatched counts)
- `diagnostics_reasons.csv` (unmatched/ambiguous reason buckets)
- `company_failures.csv` (per-ticker failure diagnostics)
- `manual_qa_sample.csv` (fixed random sample for manual review)
- `reverse_audit_rows.csv`, `reverse_candidates.csv`
- `merged_audit_rows.csv`
- `coverage_kpi_year_reverse.csv`, `coverage_kpi_year_merged.csv`
- `diagnostics_reasons_reverse.csv`, `diagnostics_reasons_merged.csv`
- `company_failures_reverse.csv`, `company_failures_merged.csv`
- `manual_qa_sample_reverse.csv`, `manual_qa_sample_merged.csv`
- `summary_forward.md`, `summary_reverse.md`, `summary.md` (merged), and
  `run_meta.json`

Per-ticker JSON looks like:

```json
{
  "ticker": "ORLY",
  "company_name": "O'Reilly Automotive, Inc.",
  "exchange": "NASDAQ",
  "source": "edgar",
  "cik": "0000898173",
  "years": [2017, 2018, 2019, 2020, 2021, 2022],
  "kpis": {
    "revenue": {"2017": 8977726000.0, "2018": 9535866000.0, "...": "..."},
    "net_income": {"...": "..."}
  },
  "tag_used": {"revenue": "Revenues", "net_income": "NetIncomeLoss"}
}
```

`tag_used` records which XBRL tag we actually pulled from — different filers
use different tags for the same line item (e.g. `Revenues` vs
`RevenueFromContractWithCustomerExcludingAssessedTax`), so keeping the audit
trail is worth the extra column.

## KPIs extracted

Defined in `tags.py`. The full list:

- **Income statement**: revenue, cost_of_revenue, gross_profit, rd_expense,
  sga_expense, operating_income, interest_expense, income_tax_expense,
  net_income, eps_basic, eps_diluted
- **Balance sheet**: total_assets, total_liabilities, stockholders_equity,
  stockholders_equity_incl_nci, cash_and_equivalents, cash_incl_restricted,
  long_term_debt_total, long_term_debt_noncurrent, long_term_debt_current,
  short_term_borrowings, inventory, accounts_receivable, accounts_payable,
  shares_outstanding
- **Cash flow**: operating_cash_flow, investing_cash_flow, financing_cash_flow,
  capex, depreciation_amortization, dividends_paid

Derived KPIs (EBITDA, margins, ROA, ROE, leverage ratios) are intentionally
*not* computed here — they're one-liners on the wide CSV and belong in the
analysis step, not the fetch step.

## Handling multi-tag ambiguity (READ THIS before changing `tags.py`)

Filers often populate more than one candidate XBRL tag for the same logical
KPI in the same year. Three distinct situations arise; the extractor handles
them differently, and one of them (Case 2) is held together by conventions
baked into the tag ordering — **changes there silently shift every number in
the dataset, so be careful**.

### Case 1 — synonyms (benign)

Tags carry the same value (within rounding). Example: post-ASC 606, most
filers set `RevenueFromContractWithCustomerExcludingAssessedTax` and
`Revenues` to the same number. Waterfall picks the first, no impact.

### Case 2 — same concept, DIFFERENT scope (⚠️ load-bearing convention)

Several "canonically named" KPIs admit more than one scope. We **always pick
the attributable-to-parent / unrestricted variant** because that's the
convention peer-benchmarking analyses expect. The tag ordering in `tags.py`
bakes this in. If you reorder tags or add new ones, re-check this list:

| KPI key | Scope we picked | Alternate scope (explicitly NOT chosen) | Typical magnitude of drift |
| --- | --- | --- | --- |
| `net_income` | `NetIncomeLoss` — attributable to parent | `ProfitLoss` — includes non-controlling interest | DAN 2022: −$242M vs −$311M (29% drift) |
| `stockholders_equity` | `StockholdersEquity` — parent only | `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` — incl. NCI | DAN 2017: $1,013M vs $1,114M (~10%) |
| `cash_and_equivalents` | `CashAndCashEquivalentsAtCarryingValue` — unrestricted | `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents` — incl. restricted | typically small, can be large at startups |
| `cost_of_revenue` | `CostOfRevenue` / `CostOfGoodsAndServicesSold` — aggregate first | `CostOfGoodsSold` alone — narrower (goods only) | material for mixed-model filers |

The incl.-NCI and incl.-restricted variants are exposed as separate KPI keys
(`stockholders_equity_incl_nci`, `cash_incl_restricted`) when you need them.
**Do not fold them back into the primary KPI's tag list** — doing so would
mean different companies in the same peer group get different scopes,
silently. Any change here MUST be re-validated against this table.

The debt tags are split the same way to avoid silent mixing:

| KPI key | Scope |
| --- | --- |
| `long_term_debt_total` | `LongTermDebt` — includes current portion |
| `long_term_debt_noncurrent` | `LongTermDebtNoncurrent` — excludes current portion |
| `long_term_debt_current` | `LongTermDebtCurrent` — current portion of LT debt |
| `short_term_borrowings` | `ShortTermBorrowings` — bank lines / commercial paper |

Pick the key whose scope matches your analysis; don't mix.

### Case 3 — aggregate missing, only components reported

Some filers tag only components (e.g. `LiabilitiesCurrent` +
`LiabilitiesNoncurrent`, no `Liabilities`; or `CostOfGoodsSold` +
`CostOfServices`, no aggregate). `KpiDef.sum_components` lists tag-sets to
sum as a last-resort fallback — fires only when the primary waterfall
leaves the year empty and **all** listed components are present.

A component tag prefixed with `-` means subtract, which lets us derive
missing aggregates via accounting identities. The main use is
`total_liabilities ← Assets − StockholdersEquityIncludingNCI`: many filers
(ORLY, GPC) skip both the `Liabilities` tag and its sub-components entirely,
and this identity recovers them. `tag_used` records the derivation as
`"sum:Assets-StockholdersEquity..."` so audit is possible.

### Audit trail

Every extracted ticker writes `ambiguous_tags` in its JSON output listing
any (kpi, year) where ≥2 candidate tags disagreed by more than 0.1%. The
chosen value is included in the entry so you can see what we picked vs
what we dropped. Summed-fallback values are tagged `"sum:A+B"` in
`tag_used` to make them auditable.

## Known limitations

- **yfinance depth**: for non-US tickers, only ~4 years of annual data are
  usually available. If we need full 2017–2022 coverage for LSE/AIM, we'll
  need a paid API (Financial Modeling Prep, Alpha Vantage Premium, or
  EODHD).
- **Tag drift**: XBRL tags are filer-specific. Our candidate-tag lists cover
  the common cases but some smaller filers will miss individual KPIs.
  `coverage.md` shows where.
- **Fiscal year ≠ calendar year**: we key facts off the period-end date, so
  a company with a March-ending fiscal year will have its FY2019 numbers
  land under year `2019` (period ending March 2019). Downstream analysis
  should consider aligning to calendar year if needed.
- **Restatements**: we keep the most recently filed value for each (ticker,
  year), so amended 10-K/A filings supersede originals.
