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
├── kpi_fetch_and_build/          # KPI fetching, ordering & dataset building
│   ├── tags.py                       # logical KPI -> candidate XBRL tags (ordered by preference)
│   ├── _fiscal.py                    # fiscal year derivation from period-end dates
│   ├── edgar.py                      # SEC EDGAR companyfacts client (CIK lookup, XBRL parsing)
│   ├── edgar_filings.py              # SEC EDGAR submissions client (10-K filing dates + acceptanceDateTime)
│   ├── yf_fallback.py                # yfinance fallback for non-US tickers
│   ├── alpha_vantage.py              # Alpha Vantage gap-fill: keys + budget + 3-statement client
│   ├── alpha_venture_API_keys.txt    # one AV API key per line (gitignored, *.txt)
│   ├── fetch_kpis.py                 # orchestrator CLI; writes output/raw/{TICKER}.json
│   ├── fetch_filing_returns.py       # 10-K filing date + market reaction (next-day / next-week / SPY-alpha)
│   ├── build_dataset.py              # consolidates output/raw/*.json into long + wide CSVs
│   └── kpi_aliases.json              # alias dict for all 30 KPIs (consumed by generate_qrels.py)
├── validate_ocr_kpis.py          # pilot: validate EDGAR KPI values against OCR text
├── retrieval_bench/              # TREC qrels generation and LLM annotation
│   ├── generate_qrels.py         # TREC qrels generator for KPI retrieval tasks
│   ├── llm_annotate_qrels.py     # LLM re-annotation of review candidates
│   └── queries/                  # 31 JSON query-template files (one per KPI)
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
    ├── filing_returns.csv        # one row per (ticker, year) with 10-K filing date + market reaction
    └── qrels/                    # output of generate_qrels.py (see below)
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
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_kpis --selected --years 2017-2022

# A single industry:
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_kpis --industry "Consumer Cyclical / Auto Parts"

# An explicit list:
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_kpis --tickers ORLY AZO GPC --years 2017-2022

# A whole cleaned CSV:
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_kpis --csv tickers_lists/cleaned/NYSE_mapped_clean_verified.csv

# Add the Alpha Vantage gap-fill on top of any of the above:
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_kpis --selected --alphavantage

# Consolidate into CSVs:
uv run python -m KPI_analysis.kpi_fetch_and_build.build_dataset
```

## Alpha Vantage gap-fill (`--alphavantage`)

Opt-in third layer that runs after EDGAR/yfinance and fills missing
(KPI, year) cells. Each call to AV's `INCOME_STATEMENT`, `BALANCE_SHEET`,
or `CASH_FLOW` returns the *full multi-year annual history* for a ticker —
so 3 calls cover ~25 KPIs across all years in one go.

### Keys & budget

API keys live in `KPI_analysis/kpi_fetch_and_build/alpha_venture_API_keys.txt` — one key per line,
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

## Filing-date market reaction (`kpi_fetch_and_build/fetch_filing_returns.py`)

Links each annual report to the market's reaction at publication. For every
US-listed `(ticker, fiscal_year)` we resolve the *original* 10-K filing on
EDGAR, read its `acceptanceDateTime` (the moment EDGAR accepted the
submission, in **UTC** despite the trailing `Z` — verified empirically), and
compute next-day and next-week price reactions from yfinance, both raw and
SPY-relative.

```bash
# All US-listed selected companies, FY2017-2022:
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_filing_returns --selected --years 2017-2022

# Single industry (filters to its US listings):
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_filing_returns \
  --industry "Consumer Cyclical / Auto Parts" --years 2017-2022

# Explicit tickers (US assumed unless suffix like .L):
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_filing_returns --tickers AZO ORLY AAP

# Skip benchmark (alpha columns will be empty):
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_filing_returns --selected --no-benchmark
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

## Qrels generator for KPI retrieval (`generate_qrels.py`)

Builds TREC-format relevance judgments (`qrels`) for evaluating retrieval
systems on the KPI question-answering task. For each `(company, year, KPI)`
triple with ground-truth in `kpis_long.csv`, the script searches OCR'd annual
reports for pages that contain the KPI value.

### Search strategy

The default search targets only the **target-year report**. Two match types are
used:

| Match type | Condition | When emitted |
|---|---|---|
| `alias+value` | KPI alias regex **and** numeric value (within tolerance) found on the same page | Always |
| `value-only` | Numeric value found, no alias | Target-year report only |

The value match uses the same unit-normalisation hierarchy as
`validate_ocr_kpis.py` (inline suffix → line → page → document → default 1×),
with a configurable relative error tolerance (default ±1%). A **literal search**
also scans for pre-formatted variants of the target value ($9.7 billion,
9,709,003, etc.) to catch OCR formatting edge cases.

Optionally, with `--search-future`, the script also searches **N+1 and N+2
reports** for comparative tables that restate the prior year's value. For
future-year reports, both alias AND value must match on the same page (to avoid
false positives from unrelated numbers).

### Key CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--industry STR` | — | Restrict to one industry (exact match on `companies.json` key). |
| `--tickers T1 T2` | — | Restrict to specific tickers. |
| `--kpis k1 k2` | — | Restrict to specific KPI keys (e.g. `revenue net_income`). |
| `--years RANGE` | `2017-2022` | Year range (e.g. `2018-2021`) or single year (`2020`). |
| `--limit N` | — | Process at most N queries (smoke test). |
| `--tolerance F` | `0.01` | Relative error tolerance for value matching (0.01 = 1%). |
| `--search-future` | `False` | Also search N+1 and N+2 reports for comparative-table restatements. |
| `--max-future-years N` | `2` | Years after target year to search (only with `--search-future`). |
| `--seed N` | `42` | Random seed for query template selection. |
| `--ocr-root PATH` | `DeepSeekOCR_Ardian_pruned_1k` | Root of OCR'd reports (.mmd files). |

### Usage

```bash
# All reports in one industry (target-year only, default)
uv run python KPI_analysis/retrieval_bench/generate_qrels.py \
    --industry "Consumer Cyclical / Auto Parts"

# Specific tickers, subset of KPIs
uv run python KPI_analysis/retrieval_bench/generate_qrels.py \
    --tickers AAP AZO --kpis revenue net_income

# Enable N+1/N+2 future-year search
uv run python KPI_analysis/retrieval_bench/generate_qrels.py \
    --industry "Consumer Cyclical / Auto Parts" --search-future

# Fast smoke test (10 queries)
uv run python KPI_analysis/retrieval_bench/generate_qrels.py \
    --industry "Consumer Cyclical / Auto Parts" --limit 10
```

### Output

Written to `KPI_analysis/output/qrels/`:

| File | Format | Description |
|---|---|---|
| `qrels.txt` | TREC | One line per `(query_id, doc_id)` with relevance `1`. Deduplicated. |
| `review_candidates.csv` | CSV | Detailed candidate info: report name, page index, match type, alias matched, raw value, normalised value, relative error, unit source, snippet. |
| `summary.md` | Markdown | Per-query and per-report statistics. |

The TREC qrels format is tab-separated: `query_id  0  doc_id  1`. Document IDs
are formatted as `{REPORT_NAME}/page_{NNNN}` (0-indexed page numbers).

### Query instantiation

For each `(ticker, year, KPI)` triple with ground-truth, a natural-language
query is generated by randomly selecting a template from
`KPI_analysis/retrieval_bench/queries/*.json` and substituting the company name (from
`companies_alt_names.json`) and year. The query set covers all 30 KPIs.

### Data-quality findings from early runs

- For Auto Parts (17 tickers), GTEC has no OCR report in the corpus — 4/5
  queries hit at least one candidate page.
- Most hits come from the target-year report's financial statements. The
  `--search-future` flag adds marginal recall (comparative tables) but also
  increases false-positive risk, hence the stricter alias+value requirement.
- Literal value matches handle cases where OCR formatting diverges from the
  numeric tolerance search (e.g. "$9.7 billion" vs. the raw number 9709003000).

## LLM annotation of qrels (`llm_annotate_qrels.py`)

Re-validates the regex-matched candidates from `review_candidates.csv` using an
LLM. The regex pipeline is designed for high recall — it finds pages where a
number close to the target exists near a KPI alias. Many of those matches are
coincidental (the right number, wrong context). The LLM annotation filters and
grades those candidates.

### Pipeline

```
generate_qrels.py          llm_annotate_qrels.py       human review
(candidates)  ──────────>  (LLM grades 0/1/2)  ────>  (flagged edge cases)
392K candidates             ~2 pages/sec, ~55h           ~few hundred rows
```

### Grading rubric (0/1/2)

| Grade | Label | Definition |
|---|---|---|
| **2** | Primary source | The page directly reports the target KPI value for the target fiscal year in a financial statement (income statement, balance sheet, cash-flow statement), a data table, or an explicit narrative sentence. The value matches after unit scaling. |
| **1** | Contextual mention | The KPI concept appears and a value is nearby, but: (a) the value is for a different fiscal year (comparative restatement), (b) the value is for a subsidiary/segment not the consolidated entity, (c) the value is approximate/rounded, (d) the KPI is mentioned in prose without a specific figure, or (e) the match is in a footnote or discussion rather than a primary financial statement. |
| **0** | Not relevant | The page does not mention the target KPI, or the numeric match is purely coincidental (the same number appears in an unrelated context). |

### System prompt guidance

The LLM prompt includes explicit rules for the most common edge cases:

- **Unit scaling**: "in thousands" header × 1,000, "in millions" × 1,000,000.
- **Multi-year tables**: most annual reports show 2–3 years side by side. Only
  the column for the **target fiscal year** is grade 2; prior-year columns are
  grade 1 (comparative restatements for a different year).
- **52/53-week fiscal years**: US retailers (AAP, COST, AZO) end their fiscal
  year in early January. "Year Ended January 1, 2022" = fiscal year 2021. The
  `report_year` in the prompt uses the filer's own label, not the calendar year.
- **Scope distinctions**: `net_income` is parent-only (excluding NCI);
  `stockholders_equity` is parent-only; `cash_and_equivalents` is unrestricted.
  If the page shows a broader scope, the LLM gives grade 1, not 2.
- **Different phrasing**: "net sales" for revenue, "capital expenditure" for
  capex — all known aliases are listed in the prompt.

### Examples from actual annotations

**Grade 2 — primary source (balance sheet line item):**

> Query: `AAP_accounts_payable_2017`
> Page 42: Consolidated Balance Sheet showing "Accounts payable" = $2,894,582
> (in thousands) for December 30, 2017.
> LLM: *"The page contains the Consolidated Balance Sheet for Advance Auto
> Parts. The column 'December 30, 2017' corresponds to fiscal year 2017. The
> accounts payable line item reports $2,894,582 in thousands = $2,894,582,000,
> matching the target."*

**Grade 1 — contextual mention (wrong year in comparative table):**

> Query: `AAP_cost_of_revenue_2017`
> Page 72: "Cost of sales" = 5,314,246 (in thousands) but on a "Condensed
> Consolidating Statement of Operations For the Year Ended January 2, 2016"
> (fiscal year 2015).
> LLM: *"The page contains the exact target value for 'Cost of sales', but it
> is located in the statement for fiscal year 2016, not the target year 2017."*

**Grade 1 — contextual mention (scope mismatch: unrestricted vs restricted):**

> Query: `AAP_cash_incl_restricted_2017`
> Page 42: Balance sheet shows "Cash and cash equivalents" = $546,937 (in
> thousands). The value matches, but the KPI is `cash_incl_restricted` (which
> includes restricted cash).
> LLM: *"The page reports 'Cash and cash equivalents' for the target year,
> which matches the target value. However, the KPI is 'cash incl restricted'
> and the page only shows unrestricted cash. This is a scope mismatch."*

**Grade 1 — contextual mention (EPS variant):**

> Query: `AAP_eps_basic_2017`
> Page 2: Financial Highlights table shows "Diluted EPS" = $6.42 and "Adjusted
> Diluted EPS" = $5.37, but no basic EPS.
> LLM: *"The page reports Diluted EPS and Adjusted Diluted EPS for the target
> year, but the KPI is basic EPS. The page does not state basic EPS."*

**Grade 0 — not relevant (coincidental number, different line item):**

> Query: `AAP_accounts_receivable_2017`
> Page 26: "$600.8 million" appears, but the text says "We generated operating
> cash flow of $600.8 million during 2017."
> LLM: *"The page mentions $600.8 million, but it explicitly attributes this
> figure to operating cash flow, not accounts receivable."*

**Grade 0 — not relevant (alias matched in wrong context):**

> Query: `AAP_depreciation_amortization_2017`
> Page 30: "amortization" appears near "$250 million", but the text says
> "In 2018, we anticipate our capital expenditures... will be up to $250
> million."
> LLM: *"The page mentions 'amortization' and the value '$250 million', but
> the $250 million refers to anticipated capital expenditures, not depreciation
> and amortization."*

### Flagging logic

When the LLM gives grade < 2 on a **high-confidence** regex match, the
candidate is flagged for human review. A match is high-confidence when all
three conditions hold:

| Condition | Threshold | Rationale |
|---|---|---|
| `match_type` | `alias+value` | Both the KPI alias and a numeric match were found on the same page. |
| `rel_error` | `< 0.005` (0.5%) | The numeric value is very close to the target. |
| `unit_source` | `page`, `line`, or `inline` | The unit scaling is from a reliable source (not `default` or `literal`). |

This is deliberately narrow. A `value-only` match with `rel_error=0.007` and
`unit_source=literal` won't be flagged even if the LLM says grade 0 — that's
expected noise from the regex pipeline. Only disagreements where the regex
evidence is strong get flagged.

Flagged candidates are written to `review_flagged.csv` with the LLM's
reasoning, the regex evidence, and a `flag_reason` column.

### Usage

```bash
# Run on all candidates (full pipeline, ~55h at 2 pages/sec)
uv run python KPI_analysis/retrieval_bench/llm_annotate_qrels.py --model Qwen/Qwen3.6-27B-FP8

# Smoke test on a small subset
uv run python KPI_analysis/retrieval_bench/llm_annotate_qrels.py --model Qwen/Qwen3.6-27B-FP8 --limit 100

# Resume after interruption (reads existing audit CSV, skips already-done)
uv run python KPI_analysis/retrieval_bench/llm_annotate_qrels.py --model Qwen/Qwen3.6-27B-FP8 --resume

# Custom endpoint
uv run python KPI_analysis/retrieval_bench/llm_annotate_qrels.py \
    --model Qwen/Qwen3.6-27B-FP8 \
    --base-url http://gpu-server:8000/v1
```

### Output

Written to `KPI_analysis/output/qrels/`:

| File | Format | Description |
|---|---|---|
| `qrels_llm.txt` | TREC | Graded relevance (0/1/2) for every candidate. All grades are written so downstream evaluation can use any threshold. |
| `annotations_audit.csv` | CSV | Per-candidate detail: regex match info, LLM grade, reasoning, latency, token counts. |
| `review_flagged.csv` | CSV | High-confidence regex matches where LLM grade < 2. For human review. |
| `annotations_summary.md` | Markdown | Grade distribution by match type and by KPI. |

### Using graded qrels for evaluation

The `qrels_llm.txt` file writes all three grades. For binary evaluation
metrics (recall@k, MAP), threshold as needed:

- **High-precision set**: grade 2 only — these are pages where the LLM is
  confident the KPI value appears in a primary financial statement.
- **High-recall set**: grades 1+2 — includes contextual mentions, useful for
  evaluating whether a retriever finds the page at all.
- **Full set**: all grades — use graded metrics (NDCG, ERR) for the most
  informative evaluation.

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
