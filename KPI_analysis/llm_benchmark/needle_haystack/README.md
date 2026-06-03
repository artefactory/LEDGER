# Needle-in-a-haystack KPI benchmark

Measures whether a long-context LLM can **find one specific KPI value inside a
large, noisy OCR'd annual report and transcribe it precisely**. Each query names
one company, one fiscal year, and one metric; the model must locate that single
figure in a ~100k-token markdown document and return it as a short JSON object,
in raw single units. Predictions are scored against the ground-truth values in
`KPI_analysis/output/kpis_long.csv`.

This is the single-needle counterpart to the multi-KPI extraction benchmark in
`KPI_analysis/llm_benchmark/` (extract everything in one pass). It reuses that
package's document loader, KPI catalogue, and KPI definitions.

## Why prefix caching is the whole point

Each of the 494 test reports is targeted by ~20 queries (10000 queries / 494
reports = 20.2 avg). The expensive work in every call is prefilling the
~100k-token document; the question that follows is ~120 tokens. So we structure
every prompt as

```
role=system : SYSTEM_PROMPT                       <- identical for every query
role=user   : <DOCUMENT, [Page N]-marked>         <- identical for a given report
              ===== END OF ANNUAL REPORT =====     (constant separator)
              <QUESTION block>                     <- the ONLY part that varies
```

With vLLM launched `--enable-prefix-caching`, the `SYSTEM + DOCUMENT` token
prefix is prefilled **once per report** and its KV blocks are reused for all of
that report's queries. `run_needle.py` enforces this by processing one report at
a time: it sends a single **warm-up query** (blocking) to populate the cache,
then fires the rest concurrently against the cached prefix.

Measured effect on the full test set (`--dry-run`, cl100k estimate):

| | prefill tokens |
|---|---|
| **with** prefix caching (1 prefill / report) | ~12.7 M |
| **without** (1 prefill / query) | ~267.9 M |
| | **≈ 21× less prefill** |

Observed on a real prototype run (Qwen3.6-27B-FP8, prefix caching on): a
report's first (cold) query takes ~27 s to prefill its ~97k-token document; the
remaining ~20 queries return in ~2.5–6 s each — direct confirmation the prefix
is reused. Note: some vLLM builds do **not** populate
`usage.prompt_tokens_details.cached_tokens` (it stays null here even though
caching is active), so the latency drop, not the `cached_tokens` field, is the
reliable signal. And remember `--enable-prefix-caching` is **not** on by default
on this server — without it every query re-prefills the whole document.

## Serving the model (operator)

The documents reach ~115k cl100k tokens, so the server needs a large context and
prefix caching **on** (the project's usual `--max-model-len 16384/32768` serves
are far too small for this benchmark):

```bash
vllm serve <MODEL> \
    --enable-prefix-caching \
    --max-model-len 131072 \
    --port 8000
# (--guided-decoding-backend xgrammar is auto/default on recent vLLM and may be
#  rejected as an unknown flag — drop it; response_format json_schema still works.)
# Large models: --tensor-parallel-size 2 and/or an FP8 build.
```

Per-model serving + client flags:

| model family | server flags | client flags |
|---|---|---|
| Qwen3 (e.g. `Qwen/Qwen3.6-27B-FP8`) | `--reasoning-parser qwen3` | `--no-thinking` (emit bare JSON under guided decoding) |
| gpt-oss (e.g. `openai/gpt-oss-20b`) | none extra (Harmony auto-handled; no `--reasoning-parser qwen3`, no `--language-model-only`) | `--reasoning-effort low --max-tokens 2048` (leave room for the reasoning channel before the JSON) |
| Mistral (e.g. `Ministral-3-…`) | `--tokenizer_mode mistral` etc. | *(no thinking flag — its tokenizer rejects `chat_template_kwargs`)* |

**Run two models at once** — give each server its own port, then point the
client at it with `--base-url` (default `http://localhost:8000/v1`). Results go
to `output/<model-slug>/`, so the two never collide:

```bash
# server A (already running): Qwen on :8000
# server B: vllm serve openai/gpt-oss-20b --enable-prefix-caching --max-model-len 131072 \
#           --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --port 8008
uv run python $NH/run_needle.py --model openai/gpt-oss-20b \
    --base-url http://localhost:8008/v1 --reasoning-effort low --max-tokens 2048 --prototype
```

The benchmark client itself only needs `openai` (already in the env) — it is a
pure HTTP client against the server(s) above.

## Commands

```bash
NH=KPI_analysis/llm_benchmark/needle_haystack

# 0. Plan offline — prints the prefix-cache plan + token savings, no server needed
uv run python $NH/run_needle.py --model <MODEL> --dry-run
uv run python $NH/run_needle.py --model <MODEL> --prototype --dry-run

# 1. Smoke test: prototype set (3 reports, 75 queries)
uv run python $NH/run_needle.py --model <MODEL> --prototype

# 2. Full run (10000 queries, 494 reports)
uv run python $NH/run_needle.py --model <MODEL>
uv run python $NH/run_needle.py --model <MODEL> --resume   # restart-safe

# 3. Score
uv run python $NH/score_needle.py --model <MODEL>
```

Outputs land in `output/<model-slug>/` (slug = model id with `/`→`__`).

## Files

```
needle_haystack/
├── generate_queries.py    # (existing) builds queries.csv from kpis_long + templates
├── select_test_set.py     # (existing) picks the 10000-query test_set.csv
├── test_set.csv           # (existing) 10000 queries: query_id, query_text
├── prototype_3_reports.csv# (existing) 75-query smoke set (FOXF/HRL/SLB 2017)
│   --- benchmark runtime (new) ---
├── needle_schema.py       # NeedleAnswer pydantic model = xgrammar schema
├── needle_data.py         # query_id -> ground truth + report + KPI unit class/definition
├── needle_prompt.py       # constant SYSTEM_PROMPT + per-query QUESTION suffix
├── needle_client.py       # deterministic OpenAI-compatible call (seed, temp 0)
├── run_needle.py          # prefix-cache-aware orchestrator -> responses.jsonl
├── score_needle.py        # join + classify + metrics -> scored.csv, summary.md
└── output/<model-slug>/
    ├── responses.jsonl    # one line per query: the full record incl. raw model text + usage
    ├── run_meta.json      # model, decoding params, seed, prompt_version, counts, timing
    ├── scored.csv         # one row per query with outcome + diagnostics
    ├── per_kpi.csv / per_year.csv / per_source.csv / per_unit_class.csv
    └── summary.md         # headline metrics + wrong-answer diagnostics + slices
```

## The answer schema (short, structured)

`NeedleAnswer` (xgrammar-enforced): `found` (bool), `value` (number|null, **raw
single units**), `value_verbatim` (string|null, the figure exactly as printed),
`unit_scale` (`units`/`thousands`/`millions`/`billions`/`per_share`/`unknown`),
`page` (int|null). Only `value` is scored; the others are audit aids — e.g. you
can check `value ≈ parse(value_verbatim) × scale`, or spot-check `page` against
the qrels.

## Prompt rules (rigor)

The constant system prompt instructs the model to: read the **primary
consolidated statements** (not highlights/MD&A/pro-forma); return the **most
precise representation** (the statement figure, not "≈$3.4 billion"); emit
**raw single units** by applying the statement's in-thousands/millions/billions
scale (EPS and per-share figures are **not** scaled); convert accounting
parentheses to a minus sign; report **capex / dividends as positive outflows**
and the three cash-flow subtotals with their reported sign; pick the **exact
fiscal-year column**; honour **scope** (net income attributable to parent;
unrestricted cash; the specific debt scope); and answer `found=false` rather
than guess when the figure is absent.

### Query modes (`--query-mode`)

- **`defined`** (default): the question carries the **canonical, scope-precise
  definition** of the requested KPI (reused from `kpi_catalogue.DESCRIPTIONS`)
  plus the unit hint. This isolates the skill under test — *locate + transcribe
  + scale the right figure* — from *guessing which scope we meant*. Ground truth
  commits to one scope per key (e.g. `cost_of_revenue` = COGS only, not total
  cost of sales), so naming it is what makes the score measure extraction.
- **`plain`**: only the informal natural-language question. Harder and more
  realistic; scope ambiguity counts against the model. Use for ablation.

The active mode and a `prompt_version` hash are recorded in `run_meta.json`.

## Scoring

Join is exact: ground truth is indexed by reconstructed `query_id`
(`f"{ticker}_{kpi}_{year}"`, the same construction as `generate_queries.py`),
which matches 10000/10000 test queries.

**Outcomes** per query: `matched` (|pred−gt|/|gt| ≤ tolerance, default ±1%),
`wrong`, `not_found` (model abstained: `found=false`), `no_response` (call
failed), `skipped` (report over `--max-doc-tokens`, never run).

**Headline metrics** over the eval set (matched + wrong + not_found):

- `accuracy` = matched / eval
- `accuracy_strict` = matched within ±0.05% / eval (rewards exact transcription)
- `attempt_rate` = (matched + wrong) / eval (1 − abstention)
- `precision_when_found` = matched / (matched + wrong)

sliced per KPI / year / ground-truth source / unit class.

**Wrong-answer diagnostics** — every `wrong` is bucketed into a systematic
failure mode, so a low score is *readable*:

- `year_shift(±k)` — pred equals this metric's value for an adjacent fiscal year
  (read the wrong column of a multi-year statement).
- `sign_error` — pred ≈ −gt.
- `scale_error(x1e±k)` — pred ≈ gt × 10^±3/6/9 (mis-applied the unit scaling).
- `scope_factor` — |pred/gt| ∈ [0.5, 2] (likely a related-but-different line).
- `other`.

## Reproducibility

Greedy decoding (`temperature=0`, `top_p=1`) with a fixed `seed` (1234) and no
temperature bumping → identical outputs across runs of the same model+prompt.
`max_tokens` is small (256) because the answer is tiny. `run_meta.json` pins the
model, every decoding parameter, the seed, the `prompt_version`, and the
`--max-doc-tokens` guard. Reports are processed in a deterministic order; the
`.mmd` is loaded with full page content (**never tail-truncated** — the
financial statements are usually in the back half, so truncation would remove
the needle; over-budget reports are recorded as `skipped`, not cut).

## Known caveats (inherited from the multi-KPI benchmark)

These are ground-truth / convention issues, not model errors — check them before
attributing a low score to the model:

- **`shares_outstanding` units** — EDGAR reports `CommonStockSharesOutstanding`
  in "shares" but a few filers (e.g. FOXF 2018) encode it already-in-thousands,
  so `kpis_long.csv` can be 1000× low. A correct model answer then shows as
  `scale_error(x1e+3)`. Bug is in ingestion, not the LLM.
- **yfinance sign conventions** — for non-US tickers, yfinance signs `capex` /
  `dividends_paid` negative and occasionally `interest_expense`; a
  prompt-following model shows `sign_error` against those rows. Mostly LSE/AIM.
- **scope / tag disagreement** — `cost_of_revenue` (COGS vs total cost of
  sales), `depreciation_amortization`, `capex`, and the `long_term_debt_*`
  scopes are where `defined` mode helps most; residual `scope_factor` wrongs are
  signal that a KPI definition needs sharpening.
- **restatements** — `kpis_long.csv` takes the latest-filed value within an FY,
  while the report shows the as-originally-filed number; differences are usually
  ~1% (`scope_factor` or just outside tolerance).
