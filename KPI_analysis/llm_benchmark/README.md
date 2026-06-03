# llm_benchmark

LLM KPI extraction benchmarks. Contains two sub-benchmarks:

- **multi-KPI extraction** (`multi_kpi_extraction/`) — extract all KPIs from a report in one pass
- **needle-in-a-haystack** (`needle_haystack/`) — find a single KPI value in a long document

Both share the document loader and KPI catalogue at this level.

**Scoring scope** — scoring is restricted to the same reports as the
needle-in-a-haystack benchmark, listed in
`needle_haystack/test_set_reports.txt` (494 reports → 13,265 ground-truth
KPIs). Only ground-truth cells and predictions for those `(ticker, year)`
pairs count; the denominator is the fixed 13,265 test-set KPIs rather than all
of `kpis_long.csv`. The previous whole-`kpis_long.csv` scorer is preserved as
`old_scoring_benchmark.py`.

## Files

```
llm_benchmark/
├── document.py          # SHARED — OCR discovery + page-marker rendering + tail-truncation
├── kpi_catalogue.py     # SHARED — canonical KPI list embedded in the system prompt
├── multi_kpi_extraction/
│   ├── schema.py            # ReportExtraction Pydantic model = xgrammar schema
│   ├── client.py            # OpenAI-compatible chat call + retry + JSON repair
│   ├── prompts.py           # SYSTEM_PROMPT (rules + catalogue) + optional few-shot
│   ├── run_benchmark.py     # orchestrator CLI; writes output/<model-slug>/raw/*.json
│   ├── score_benchmark.py   # joins predictions vs ground truth (test-set scope), emits metrics
│   ├── old_scoring_benchmark.py  # previous scorer (whole kpis_long.csv, no test-set restriction)
│   └── output/<model-slug>/ # raw/, predictions_long.csv, per_*_metrics.csv, summary.md, run_meta.json
├── needle_haystack/     # needle-in-a-haystack benchmark (see its own README)
```

## Commands

```bash
# Smoke test (8 reports)
uv run python KPI_analysis/llm_benchmark/multi_kpi_extraction/run_benchmark.py \
    --model Qwen/Qwen3.6-27B-FP8 --limit 8

# Full run on the auto-parts subset
uv run python KPI_analysis/llm_benchmark/multi_kpi_extraction/run_benchmark.py \
    --model Qwen/Qwen3.6-27B-FP8

# Score
uv run python KPI_analysis/llm_benchmark/multi_kpi_extraction/score_benchmark.py \
    --model Qwen/Qwen3.6-27B-FP8
```

`--enable-thinking` / `--no-thinking` toggle Qwen3 / Nemotron Nano 3 thinking
mode; `--reasoning-effort {low,medium,high}` sets gpt-oss Harmony effort.

## Known scoring caveats

The score is a join on `(ticker, year, kpi)` between the LLM's emitted
`fiscal_year` and `kpis_long.csv`'s `year`. Both sides are *supposed* to use
the filer's labelled FY (the year on the 10-K cover, also what the OCR PDF
directory name encodes), but several conventions need to line up exactly.

The list below tracks the discrepancies we've found between LLM output and
ground truth that are *not* genuine LLM extraction errors. New runs should
re-check these before attributing low recall to the model.

### 1. FY mapping for March-ending US filers (fixed 2026-05-06)

**Symptom:** every (ticker, year, KPI) for MNRO, MOD, MPAA looked "wrong",
with `pred_value[year] == gt_value[year-1]` to within rounding. Affected ~296
of 671 wrong rows on the first Qwen3.6-27B-FP8 run (~44% of the wrong
bucket); recall was understated by ~7 points.

**Root cause:** `KPI_analysis/kpi_fetch_and_build/_fiscal.filer_fy_from_period_end` used to fire
year-1 for any period-end in months 1–3. That's right for early-Jan 52/53-week
US retailers like AAP/COST/AZO (whose "fiscal 2021" period ends Jan 1, 2022),
but wrong for US filers ending in late March, who state explicitly in their
10-Ks that "references to fiscal YYYY are to the fiscal year ENDED March DD,
YYYY" (verbatim from MOD, MNRO, MPAA). The LLM was picking up the cover
label correctly; ground truth was being keyed one year low.

**Fix:** rule tightened to `month == 1 → year - 1; else year` so only the
early-Jan 52/53-week case carries the offset. The system prompt in
`prompts.py` was rewritten in the same change so the LLM is no longer told
to subtract a year for March period-ends.

**To verify after a re-run:** MNRO/MOD/MPAA per-ticker `wrong` counts should
drop from ~120 each to roughly the same level as other US filers (~10–20).

### 2. Sign convention on yfinance-sourced rows

For non-US tickers (LSE/AIM/ASX/...) we use yfinance, which signs cash-flow
fields differently from the SYSTEM_PROMPT we give the LLM:

- `capex` and `dividends_paid` — yfinance returns these *negative* (cash
  outflow). The prompt asks the LLM for *positive* (matches EDGAR's sign).
- `interest_expense` — yfinance occasionally returns negative for filers
  reporting net interest income; prompt asks for positive.

These show up as `exact_negation` (ratio = −1) wrongs against yfinance gt
rows. Whose convention is "right" is debatable; for now the LLM follows the
prompt and yfinance rows show as wrong. Mostly affects ABDP.L. Decide once
and pick a side — either flip yfinance signs at ingestion in `yf_fallback.py`
or relax the score to treat exact-negation against `yfinance:*` tags as
matched.

### 3. `CommonStockSharesOutstanding` units (FOXF 2018)

EDGAR's companyfacts payload reports `CommonStockSharesOutstanding` in
`units: "shares"`, but for a few filers (FOXF observed; possibly others)
the numeric value is already in thousands. `edgar.py` doesn't apply a
1000× scale, so `kpis_long.csv` ends up with `shares_outstanding=37991`
where the LLM correctly emits `37991000`. Bug is in our ingestion, not
the LLM. Has not been fixed yet — flagged for later.

### 4. Scope / tag disagreements (genuinely ambiguous)

Real disagreements where neither side is clearly wrong, but they show up
as scoring "wrong" because we picked one specific XBRL tag and the LLM
picked a different (also reasonable) line:

- `cost_of_revenue` — EDGAR uses `CostOfGoodsAndServicesSold` (product
  COGS only); LLMs often pick the larger "Total cost of sales" line that
  also includes distribution / occupancy / services costs. Differs by
  20–30% on retailers like AAP.
- `depreciation_amortization` — EDGAR uses
  `DepreciationDepletionAndAmortization` (cash-flow line); LLMs sometimes
  emit a smaller "depreciation" subset or a larger figure that includes
  amortization of right-of-use lease assets.
- `capex` — `PaymentsToAcquirePropertyPlantAndEquipment` vs broader
  capex definitions that include intangibles or capitalized software.
- `long_term_debt_*` scope splits — `total` vs `current` vs `noncurrent`
  vs `short_term_borrowings` are distinct KPI keys; LLMs sometimes
  conflate them. SRI 2017–2019 noncurrent debt looked dramatically wrong
  but turned out to be the LLM picking the total-debt line.

These contribute roughly 190 of 671 wrongs (`small_factor` bucket, ratio
in [0.5, 2]). Treat them as *signal about prompt clarity* — if too many
LLMs pick the same "wrong" line, the canonical KPI definition probably
needs to be sharpened in `prompts.py` / `kpi_catalogue.py`.

### 5. Restatements (small_diff bucket, ~98 wrongs)

`AAP 2021 accounts_payable`: gt $3.97B, pred $3.92B — ~1.1% apart. The
ground-truth picker is `_best_*_entry_for_year(... key=lambda e: e.get("filed"))`
in `edgar.py`, which takes the LATEST filed value within an FY bucket
(restated in a subsequent 10-K). The LLM, reading the original 10-K,
sees the as-originally-filed number. Documented but not fixed — switching
the picker to `min(filed)` would align it with the LLM but change every
EDGAR-restated value across the dataset, so it's a deliberate choice.

## Score interpretation

Read `summary.md` for the top-line numbers. When recall looks low, *first*
slice `predictions_long.csv` by ticker and look for systematic year shifts:

```python
import csv
from collections import Counter

gt = {}
with open("KPI_analysis/output/kpis_long.csv") as f:
    for r in csv.DictReader(f):
        try:
            gt[(r["ticker"], int(r["year"]), r["kpi"])] = float(r["value"])
        except ValueError:
            continue

shifted = Counter()
with open("KPI_analysis/llm_benchmark/multi_kpi_extraction/output/<model>/predictions_long.csv") as f:
    for r in csv.DictReader(f):
        if r["status"] != "wrong":
            continue
        try:
            y = int(r["year"]); pv = float(r["pred_value"])
        except ValueError:
            continue
        gtm = gt.get((r["ticker"], y - 1, r["kpi"]))
        if gtm and abs(pv - gtm) / max(abs(gtm), 1e-6) < 0.01:
            shifted[r["ticker"]] += 1

for t, n in shifted.most_common():
    print(f"{t:10s} {n}")
```

If a ticker shows up with a high shift count, the wrong is almost certainly
not the LLM's fault — it's a ground-truth keying issue (currently: bug 1
above; in future runs, possibly a new filer convention we haven't seen).

Note with needle-in-a-haystack

### The two benchmarks test different things

| | Needle-in-a-haystack | Multi-KPI extraction |
|---|---|---|
| Task | "Find this one value" | "Extract everything you can" |
| Core skill tested | Retrieval + precision in a long context | Completeness + knowing when to abstain |
| Natural prompt | "What is revenue for FY2019?" | "Extract all 31 KPIs from this report" |

In the multi-KPI setting, the model **doesn't know in advance** which KPIs are in the document — that's part of the task. Restricting to the 10,000 grade-2 subset removes the "abstain when absent" dimension entirely, which is arguably the harder real-world skill (hallucinating a plausible number is the common failure mode).
