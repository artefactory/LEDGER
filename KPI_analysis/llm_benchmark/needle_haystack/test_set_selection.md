# Test Set Selection

## Why `test_set.csv`?

The full `queries.csv` has 26,050 queries — too many for a practical LLM
evaluation. `test_set.csv` is a 10,000-query subset selected to be fast to run
while still representative.

## Selection criteria

### 1. Token budget (< 115K tokens)

Each source report's clean `.mmd` is tokenized with `cl100k_base`. Reports
exceeding 115K tokens are excluded. This caps inference cost and latency —
longer reports would dominate wall-clock time and make the benchmark slow to
iterate on.

Of 990 reports, 589 pass this filter (median: 104K tokens).

### 2. KPI coverage (greedy, most-first)

Reports are sorted by the number of distinct KPIs they have in the ground-truth
dataset. Reports covering more KPIs are selected first. This maximizes KPI
diversity in the test set — a report with 31 KPIs contributes 31 queries
spanning the full income statement, balance sheet, and cash flow statement.

### 3. Grade-2 qrels (answer must be in the same-year document)

Each selected query about fiscal year X must have at least one page **in the
year-X report** annotated with LLM grade = 2 in `qrels_llm.txt`. Grade 2 means
"primary source" — the KPI value appears directly on that page (not just a
passing reference). Pages from future-year documents (X+1 or X+2) do not count,
even if they restate the year-X figure in a comparative table.

**Why grade 2 and not grade 1 or 0?**

- **Grade 2** = the answer is definitively in the document. If the LLM fails to
  extract it, that's a clear extraction error. No ambiguity in scoring.
- **Grade 1** = contextual mention. The KPI is referenced but not as the primary
  value on that page (e.g., a comparative table restating last year's number).
  Scoring against these is noisy — the LLM might extract a different (also
  correct) value from a different page.
- **Grade 0** = not present. Testing "does the LLM refuse when the answer isn't
  there?" is a valid evaluation goal, but it conflates two capabilities:
  extraction and refusal. It also creates a scoring paradox — the ground truth
  exists in `kpis_long.csv` (from EDGAR/yfinance) but not in the OCR text, so
  the "correct" answer is simultaneously "value X" and "not found."

By requiring grade 2, we keep the test set focused on one question: *can the
LLM find and extract a KPI value that is known to be in the document?*

## Result

| Metric | Value |
|--------|-------|
| Queries | 10,000 |
| Reports used | ~494 |
| Tickers | 111 |
| KPIs | 31 |
| Years | 2017–2022 |

## Reproducing

```bash
uv run python KPI_analysis/llm_benchmark/needle_haystack/select_test_set.py
```

See `method.md` for the full pipeline documentation.
