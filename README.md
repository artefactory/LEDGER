# LEDGER

**Long-context Evaluation of Documents for Grounded Extraction and Retrieval**

A benchmark of 6,100 OCR'd corporate annual reports with ~30 consolidated financial KPIs per company-year, natural-language questions with page-level relevance judgments, and market-reaction linkage. Built to evaluate retrieval and extraction systems on genuinely long, visually dense financial documents (median 115 pages, ~104k tokens per report).

The resource induces three evaluation tasks of increasing difficulty over the same documents and ground truth:

| Task | Description | Scale |
|---|---|---|
| **Page-level KPI retrieval** | Given a natural-language KPI question, find the relevant page(s) in the corresponding report. TREC-style graded qrels (0/1/2). | ~64,000 questions, 392,484 candidate (query, page) pairs |
| **Needle-in-a-haystack** | Feed an entire OCR'd report (~100k tokens) and extract a single specified KPI as structured JSON. | 10,000 questions over 494 reports |
| **Multi-KPI extraction** | Extract all ~30 KPIs from a report in a single pass under a constrained-decoding JSON schema. | 494 reports, 13,455 ground-truth cells |

A fourth **case study** links CEO/Chairman letter rhetoric to earnings surprise and post-publication returns, demonstrating cross-modal research utility beyond model benchmarking.

---

## Repository structure

```
ardian-dataset-bench/
├── tickers_lists/               # Stage 1: company discovery & industry peer-group selection
│   ├── scripts/                 #   pipeline scripts (extract → map → clean → group → select → copy → coverage → prune)
│   └── grouped/selected/        #   curated peer groups + year-coverage analysis
│
├── doc_text_processing/         # Stage 2: document text processing
│   ├── 10K_or_not/              #   SEC Form 10-K classifier (regex markers on first pages)
│   └── CEO_word_extraction/     #   CEO / shareholder letter extractor (heading-based, page-windowed)
│
├── KPI_analysis/                # Stages 3–4: KPI fetching, validation, retrieval qrels, filing returns
│   ├── kpi_fetch_and_build/     #   KPI fetch orchestrator (EDGAR → yfinance → Alpha Vantage), dataset builder
│   │   ├── tags.py              #     XBRL tag definitions (ordered candidate waterfall per KPI)
│   │   ├── edgar.py             #     SEC EDGAR companyfacts client
│   │   ├── yf_fallback.py       #     yfinance fallback for non-US tickers
│   │   ├── alpha_vantage.py     #     Alpha Vantage gap-fill (opt-in, 25 calls/key/day)
│   │   ├── fetch_kpis.py        #     CLI orchestrator → output/raw/{TICKER}.json
│   │   ├── build_dataset.py     #     raw JSONs → kpis_long.csv + kpis_wide.csv
│   │   ├── fetch_filing_returns.py  # 10-K filing date + market reaction (US-only)
│   ├── validate_ocr_kpis.py     #   forward + reverse OCR validation of KPI targets
│   ├── generate_qrels.py        #   TREC qrels generator for page-level retrieval
│   ├── llm_benchmark/           #   LLM extraction benchmarks (needle + multi-KPI)
│   │   ├── document.py          #     shared OCR document loader
│   │   ├── kpi_catalogue.py     #     shared KPI definitions for prompts
│   │   ├── needle_haystack/     #     needle-in-a-haystack single-KPI benchmark
│   │   └── multi_kpi_extraction/#     multi-KPI constrained-decoding benchmark
│   └── retrieval_bench/         #   LLM-assisted qrels annotation
│
├── retrieval/                   # Retrieval evaluation runner (BM25, SPLADE, ColBERT)
├── annotation_OCR/              # Browser interface for human OCR table-quality annotation
├── annotation_qrels/            # Browser interface for qrels relevance annotation
│
├── Notes.md                     # Research notes, sector-selection rationale, data-quality caveats
└── scripts/                     # Misc utility scripts
```

---

## The three benchmarks

### 1. Page-level KPI retrieval

For every (company, year, KPI) triple, a natural-language question is generated (company names sourced from DBPedia for semantic variability, question templates via Gemma). Given a query, the task is to retrieve the relevant page(s) from the corresponding OCR'd report. Relevance is graded on a 0/1/2 scale (not relevant / contextual mention / primary source) using unit-normalized value matching, with an LLM judge for grading.

Baselines compare lexical BM25, learned-sparse SPLADE, and the dense late-interaction retriever ColBERT. ColBERT consistently outperforms but MRR tops out at 0.449 — dense numerical pages remain exceptionally hard for off-the-shelf retrievers.

```bash
# Index OCR'd pages with BM25
uv run retrieval/retrieval.py index --method bm25 --root /path/to/mmd_tree

# Query within a single report
uv run retrieval/retrieval.py query --method bm25 --report NYSE_SLB_2018 --query "total revenue net sales"

# Batch query from a TSV file (qid<TAB>text)
uv run retrieval/retrieval.py query --method bm25 --queries_file queries.tsv --top_k 10

# SPLADE (GPU recommended)
uv run retrieval/retrieval.py index --method splade --root /path/to/mmd_tree
uv run retrieval/retrieval.py query --method splade --queries_file queries.tsv --top_k 10
```

Outputs (TREC run file + human-readable JSONL) land in `retrieval/output/<method>/`. Score against the qrels with `trec_eval` or any standard IR evaluation tool.

```bash
# Generate the TREC qrels (relevance judgments) for a given industry
uv run python KPI_analysis/generate_qrels.py --industry "Auto Parts"

# Also search N+1/N+2 year reports
uv run python KPI_analysis/generate_qrels.py --industry "Auto Parts" --search-future
```

### 2. Needle-in-a-haystack single-KPI extraction

A model receives an entire OCR'd report (~100k tokens) and must locate and transcribe a single specified KPI as a structured JSON object (`found`, `value`, `unit_scale`, `page`). Matched within ±1% of ground truth (±0.05% for strict match). Prefix caching cuts prefill by ~21x, making full-corpus evaluation tractable on a single GPU server.

The strongest baseline (Qwen3.6-27B) reaches 93.6% recall at 95.8% precision. A model with systematic unit-scaling errors (Nemotron) collapses to 15.8%.

#### Serving the model

Documents reach ~115k tokens, so the server needs a large context window and prefix caching enabled:

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
    --enable-prefix-caching \
    --max-model-len 131072 \
    --port 8000
```

Per-model flags:

| Model family | Server flags | Client flags |
|---|---|---|
| Qwen3 (e.g. `Qwen/Qwen3.6-27B-FP8`) | `--reasoning-parser qwen3` | `--no-thinking` |
| gpt-oss (e.g. `openai/gpt-oss-20b`) | none extra | `--reasoning-effort low --max-tokens 2048` |
| Mistral (e.g. `Ministral-3-14B-Instruct-2512`) | `--tokenizer_mode mistral` | *(no thinking flag)* |

#### Running

```bash
NH=KPI_analysis/llm_benchmark/needle_haystack

# Smoke test (3 reports, 75 queries)
uv run python $NH/run_needle.py --model Qwen/Qwen3.6-27B-FP8 --prototype

# Full run (10,000 queries, 494 reports)
uv run python $NH/run_needle.py --model Qwen/Qwen3.6-27B-FP8

# Resume an interrupted run (skips already-done query_ids)
uv run python $NH/run_needle.py --model Qwen/Qwen3.6-27B-FP8 --resume

# Dry run: print prefix-cache plan and token savings, no server needed
uv run python $NH/run_needle.py --model Qwen/Qwen3.6-27B-FP8 --dry-run
```

#### Scoring

```bash
uv run python $NH/score_needle.py --model Qwen/Qwen3.6-27B-FP8
```

Writes `output/<model-slug>/summary.md` with headline metrics (accuracy, precision, attempt rate), wrong-answer diagnostics (year shifts, sign/scale errors), and per-KPI / per-year / per-source slices.

### 3. Multi-KPI extraction

The hardest task: extract all ~30 KPIs from a report in a single pass under constrained decoding, scored against 13,455 ground-truth labels. Single-value skill does not transfer — Ministral (second-best at needle, 90.7%) collapses to 42.6% recall under structured extraction, while Nemotron recovers to 67.6% once schema constraints suppress its scaling error. No model exceeds 80% recall, establishing the task as an open challenge.

Uses the same vLLM server as the needle benchmark (see above for serving instructions).

#### Running

```bash
MK=KPI_analysis/llm_benchmark/multi_kpi_extraction

# Smoke test (8 reports)
uv run python $MK/run_benchmark.py --model Qwen/Qwen3.6-27B-FP8 --limit 8

# Full run
uv run python $MK/run_benchmark.py --model Qwen/Qwen3.6-27B-FP8

# Resume (skips reports whose output JSON already has status=ok)
uv run python $MK/run_benchmark.py --model Qwen/Qwen3.6-27B-FP8 --resume
```

#### Scoring

```bash
uv run python $MK/score_benchmark.py --model Qwen/Qwen3.6-27B-FP8
```

Writes `output/<model-slug>/summary.md` with recall/precision, plus per-KPI, per-year, and per-source metric CSVs. Scoring is restricted to the same 494 test-set reports as the needle benchmark.

### Case study: CEO-letter rhetoric → market reaction

Extracts 542 CEO/Chairman letters from non-10-K reports, trains L2-regularized linear probes on frozen encoder embeddings to predict EPS surprise and 90-day post-filing returns. Several encoder/target combinations land well above the random baseline (PR-AUC up to 0.47 vs 0.10 random), indicating a genuine textual signal in corporate rhetoric.

---

## Data pipeline

The benchmark is built through a four-stage pipeline:

### Stage 1 — Company discovery & peer-group selection (`tickers_lists/`)

Turns a flat list of PDF filenames into curated industry peer groups with a common year window. Six "price-taker" industries (specialty chemicals, auto parts, packaged foods, oil & gas E&P, oil & gas equipment & services, mortgage REITs) are selected because macroeconomic tides dominate results, concentrating the company-specific signal in financial KPIs. An exchange-verification step catches that ~72% of nominal LSE tickers are silently redirected by yfinance to U.S. listings.

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

### Stage 2 — Document text processing (`doc_text_processing/`)

- **10-K classifier** — scans first pages for SEC Form 10-K cover-page markers; flagged reports are excluded from the letter extractor.
- **CEO/Shareholder letter extractor** — finds section headings matching curated phrases (`Dear Shareholders`, `Letter from the CEO`, etc.) and extracts a configurable page window. Handles OCR artifacts, TOC false positives, and overlapping sections.

```bash
uv run python doc_text_processing/10K_or_not/classify_10k.py
uv run python doc_text_processing/CEO_word_extraction/extract_letters.py
```

### Stage 3 — KPI extraction & dataset build (`KPI_analysis/`)

31 consolidated KPIs across the three financial statements (income, balance sheet, cash flow) are fetched via a three-tier source waterfall: SEC EDGAR XBRL for U.S. listings, yfinance for non-U.S., and Alpha Vantage as opt-in gap-fill. Fiscal-year keying handles 52/53-week retailers. The result is 37,282 audited facts with per-KPI yearly coverage above 85%.

```bash
# Fetch KPIs for all selected companies
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_kpis --selected --years 2017-2022

# Optional: Alpha Vantage gap-fill
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_kpis --selected --alphavantage

# Build consolidated CSVs
uv run python -m KPI_analysis.kpi_fetch_and_build.build_dataset

# Validate KPIs against OCR text
uv run python KPI_analysis/validate_ocr_kpis.py
```

### Stage 4 — Filing-date market reactions

For each U.S. (ticker, fiscal year), finds the original 10-K on EDGAR, reads its acceptance timestamp, classifies pre-market / intraday / after-hours, and computes raw and SPY-relative returns from yfinance daily prices.

```bash
uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_filing_returns --selected --years 2017-2022
```

---

## Tools and infrastructure

- **OCR**: DeepSeek-OCR-2 digitizes PDFs into page-aligned Markdown with tables (HTML/LaTeX) and per-page raster images.
- **KPI sources**: SEC EDGAR companyfacts (XBRL), yfinance, Alpha Vantage (opt-in gap-fill, 25 calls/key/day with key rotation).
- **Retrieval**: BM25 (PyTerrier), SPLADE, ColBERT (`lightonai/GTE-ModernColBERT-v1`).
- **Annotation**: Custom browser interfaces for OCR table-quality review (`annotation_OCR/`) and qrels relevance grading (`annotation_qrels/`), with inter-annotator agreement computation (Fleiss' kappa).
- **Dependency management**: `uv` (Python 3.13).

---

## Documentation

| File | Contents |
|---|---|
| `Notes.md` | Sector selection rationale, data-quality findings, year-window analysis |
| `KPI_analysis/README.md` | KPI pipeline: tag ambiguity, Alpha Vantage setup, filing-returns event-window convention |
| `KPI_analysis/llm_benchmark/README.md` | LLM extraction benchmarks: multi-KPI structure, scoring caveats, score interpretation |
| `KPI_analysis/llm_benchmark/needle_haystack/README.md` | Needle benchmark: prefix caching, serving, prompt rules, scoring, reproducibility |
| `annotation_OCR/README.md` | OCR annotation interface: setup, queue modes, keyboard shortcuts, outputs |

---

## License

- **Code**: MIT
- **Data**: CC-BY-4.0

