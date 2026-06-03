# Needle-in-a-Haystack Query Generation Method

## Purpose

Generate a set of natural-language queries for evaluating LLM performance on
financial KPI extraction from full annual reports ("needle-in-a-haystack"
setting). Each query asks for a specific KPI value for a given company and
fiscal year; the LLM must locate and extract the correct value from the
corresponding OCR'd annual report.

## Inputs

| Source | Description |
|--------|-------------|
| `KPI_analysis/output/kpis_long.csv` | Ground-truth KPI values; defines the set of `(ticker, kpi, year)` triples |
| `tickers_lists/companies_alt_names.json` | Maps each ticker to a list of alternative company name strings (aliases) |
| `KPI_analysis/retrieval_bench/queries/*.json` | 31 template files, one per KPI, each containing 8–10 natural-language query variants |
| `DeepSeekOCR_Ardian_pruned_1k/` | OCR report tree; used to filter tickers to the 165 with available reports |

## Template structure

Each template file is a JSON array of strings with two placeholders.
31 template files cover all KPIs in the ground-truth dataset (30 existing +
1 for `inventory`).

- **`ABC`** — replaced by a company name alias
- **`X`** — replaced by the fiscal year (e.g. `2020`)

Example templates from `Revenue queries.json` and `Inventory queries.json`:

```
"What was the revenue of company ABC in year X?"
"Company ABC revenue year X"
"How much money did company ABC bring in during X?"
"What was the inventory for Company ABC in X?"
"Company ABC inventory year X"
```

## Substitution method

For each `(ticker, kpi, year)` triple with ground truth in `kpis_long.csv`:

1. **Pick one random template** from the KPI's template file
2. **Pick one random alias** from the ticker's alias list in `companies_alt_names.json`
3. **Substitute**: `template.replace("ABC", alias).replace("X", year)`

Both selections use a shared `random.Random(42)` instance, ensuring
deterministic output across runs.

### Randomization rationale

- **Random template** (not all): mirrors the qrels pipeline; one query per
  triple keeps the evaluation set manageable (~26K queries vs ~250K).
- **Random alias** (not canonical): simulates real user input where company
  names may be informal, abbreviated, or contain typos (e.g. "Advance auto",
  "Advanced auto parts", "Advance Dicount Auto Parts" for AAP).

## Filters

Two filtering steps reduce the full ground-truth set to the evaluation subset:

1. **OCR availability** — only tickers present in `DeepSeekOCR_Ardian_pruned_1k/`
   are included (165 tickers across 6 industries). This ensures every query has
   a corresponding annual report to retrieve from.
2. **Template coverage** — only KPIs with query templates are included. All 31
   KPIs in `kpis_long.csv` that have templates are covered (including `inventory`).

This yields 26,050 queries — one per ground-truth triple, matching the qrels
pipeline exactly.

## Test set selection (`test_set.csv`)

A 10,000-query subset is selected from `queries.csv` for practical LLM
evaluation. Three filters ensure quality and feasibility:

### Selection criteria

1. **Token budget** — source report's clean `.mmd` must have < 115,000 tokens
   (cl100k_base). This excludes very long reports that would blow up inference
   cost and latency. Result: 589 of 990 reports pass.
2. **KPI coverage** — reports are sorted by the number of distinct KPIs they
   have in `kpis_long.csv` (descending). Reports with more KPIs are selected
   first, maximizing the diversity of KPI types in the test set.
3. **Grade-2 qrels (same-year)** — each selected query about year X must have
   at least one page **in the year-X report** annotated with LLM grade = 2
   (primary source) in `qrels_llm.txt`. Pages from future-year documents
   (X+1/X+2) do not count. This guarantees the answer is definitively present
   in the target document, making evaluation unambiguous.

### Selection algorithm

```
1. Discover all 990 reports in DeepSeekOCR_Ardian_pruned_1k/
2. Count tokens in each clean .mmd with cl100k_base
3. Filter to reports with <115K tokens (589 reports)
4. Sort by KPI count descending (tie-break: ticker, year)
5. Greedy: iterate sorted reports, collect queries that have same-year grade-2
   pages, stop at 10,000
```

### Result

| Metric | Value |
|--------|-------|
| Queries selected | 10,000 |
| Unique tickers | 111 |
| Unique KPIs | 31 |
| Years covered | 2017–2022 |

### Reproducing

```bash
uv run python KPI_analysis/llm_benchmark/needle_haystack/select_test_set.py
```

CLI flags: `--max-tokens 115000`, `--target-size 10000`, `--ocr-root <path>`,
`--qrels-llm <path>`.

## Output format

**`queries.csv`** — full query set (26,050 rows):

| Column | Description |
|--------|-------------|
| `query_id` | `{ticker}_{kpi}_{year}` — matches the query ID format used in the qrels pipeline |
| `query_text` | The instantiated natural-language query |

**`test_set.csv`** — evaluation subset (10,000 rows), same schema.

## Reproducing

```bash
# Full query set
uv run python KPI_analysis/llm_benchmark/needle_haystack/generate_queries.py

# Test subset
uv run python KPI_analysis/llm_benchmark/needle_haystack/select_test_set.py
```

Use `--seed <N>` to vary alias/template selection. Default seed is `42`.
Use `--ocr-root <path>` to override the OCR directory for ticker filtering.
