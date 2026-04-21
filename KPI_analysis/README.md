# KPI_analysis

Fetches annual consolidated KPIs (revenue, net income, total assets, capex, …)
for the companies we've mapped in `tickers_lists/`. Designed to produce one
point per company × year × KPI so downstream scripts can compare peers.

## Design

Hybrid pipeline — we pick the data source based on the listing exchange:

| Exchange                                              | Source                   | Why                                                  |
| ----------------------------------------------------- | ------------------------ | ---------------------------------------------------- |
| NYSE, NYSE American (AMEX), Nasdaq (GS/GM/CM), Cboe   | **SEC EDGAR** companyfacts | Free, unlimited, full history (XBRL back to ~2009). |
| LSE, AIM, ASX, TSX, …                                 | **yfinance** (fallback)  | No SEC filings. yfinance covers ~4 recent fiscal years. |

EDGAR is the preferred source because it publishes structured XBRL for every
10-K filing, so we can pull 6+ years of history without hitting rate limits.
For non-US listings we fall back to yfinance (the same library we already use
in `tickers_lists/scripts/map_tickers.py`); coverage is shallower and the field
labels are less stable, but it's enough for the later years in our corpus.

## Files

```
KPI_analysis/
├── tags.py            # logical KPI -> candidate XBRL tags (ordered by preference)
├── edgar.py           # SEC EDGAR client (CIK lookup, companyfacts fetch, parsing)
├── yf_fallback.py     # yfinance fallback for non-US tickers
├── fetch_kpis.py      # orchestrator CLI; writes output/raw/{TICKER}.json
├── build_dataset.py   # consolidates output/raw/*.json into long + wide CSVs
├── cache/             # ticker->CIK map and cached SEC responses (gitignored)
└── output/
    ├── raw/           # one JSON per ticker
    ├── kpis_long.csv  # (ticker, year, kpi, value) long form
    ├── kpis_wide.csv  # (ticker, year) rows × KPI columns
    └── coverage.md    # coverage % per KPI per year
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

# Consolidate into CSVs:
uv run python KPI_analysis/build_dataset.py
```

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
  cash_and_equivalents, long_term_debt, short_term_debt, inventory,
  accounts_receivable, accounts_payable, shares_outstanding
- **Cash flow**: operating_cash_flow, investing_cash_flow, financing_cash_flow,
  capex, depreciation_amortization, dividends_paid

Derived KPIs (EBITDA, margins, ROA, ROE, leverage ratios) are intentionally
*not* computed here — they're one-liners on the wide CSV and belong in the
analysis step, not the fetch step.

## Known limitations

- **yfinance depth**: for non-US tickers, only ~4 years of annual data are
  usually available. If we need full 2017–2022 coverage for LSE/AIM, we'll
  need a paid API (Financial Modeling Prep, Alpha Vantage Premium, or
  EODHD).
- **Tag drift**: XBRL tags are filer-specific. Our candidate-tag lists cover
  the common cases but some smaller filers will miss individual KPIs.
  `coverage.md` shows where.
- **Fiscal year ≠ calendar year**: the `fy` field in EDGAR is the filer's
  fiscal year, so an April-year-end company's FY2019 covers May 2018 – April
  2019. We keep that convention; downstream analysis should consider aligning
  to calendar year if needed.
- **Restatements**: we keep the most recently filed value for each (ticker,
  fy), so amended 10-K/A filings supersede originals.
