"""
Prepare the Ardian needle-in-a-haystack benchmark dataset for HuggingFace upload.

Output layout (under hf_output/needle/):
    hf_output/needle/
    ├── README.md                  # dataset card (generated)
    ├── eval/                      # "eval" config
    │   ├── data.parquet           # one row per query: query_id, query_text, value, kpi, ticker, year, exchange, company_name, industry, mmd_text
    │   └── mmd/                   # raw .mmd files (one per report)
    │       └── {EX}_{TICK}_{YEAR}.mmd
    └── no_eval/                   # "no_eval" config (placeholder for future)
        ├── data.parquet
        └── mmd/

Each parquet row = one query targeting one KPI value in one annual report.
The mmd_text column contains the full OCR text of the corresponding report.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── default paths (relative to repo root) ─────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
NEEDLE_DIR = REPO_ROOT / "KPI_analysis" / "llm_benchmark" / "needle_haystack"

DEFAULTS = {
    "eval": {
        "test_set": NEEDLE_DIR / "test_set.csv",
        "kpi_csv": REPO_ROOT
        / "KPI_analysis/find_more_queries/full_6k/kpi_long_eval.csv",
        "reports": REPO_ROOT / "DeepSeekOCR_Ardian_evaluation_set_reports",
        "qrels": REPO_ROOT / "KPI_analysis" / "output" / "qrels" / "qrels_llm.txt",
    },
    "no_eval": {
        "test_set": None,  # not yet generated
        "kpi_csv": REPO_ROOT
        / "KPI_analysis/find_more_queries/full_6k/kpi_long_no_eval.csv",
        "reports": REPO_ROOT / "DeepSeekOCR_Ardian_with_kpis_no_eval",
        "qrels": None,  # not yet available
    },
}


def load_test_set(test_set_path: Path) -> list[dict[str, str]]:
    """Load query_id, query_text from the test_set CSV."""
    rows = []
    with test_set_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"query_id": row["query_id"], "query_text": row["query_text"]})
    return rows


def load_kpi_ground_truth(kpi_csv_path: Path) -> dict[str, dict]:
    """Index kpi_long CSV by query_id = f"{ticker}_{kpi}_{year}".

    Returns {query_id: {ticker, exchange_ocr, company_name, industry, year, kpi, value, source, tag}}.
    """
    index: dict[str, dict] = {}
    with kpi_csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                year = int(row["year"])
                value = float(row["value"])
            except (ValueError, KeyError):
                continue
            ticker = row["ticker"].strip()
            kpi = row["kpi"].strip()
            qid = f"{ticker}_{kpi}_{year}"
            # Keep only the first occurrence (matches generate_queries.py behavior)
            if qid not in index:
                index[qid] = {
                    "ticker": ticker,
                    "exchange": row.get(
                        "exchange_ocr", row.get("exchange", "")
                    ).strip(),
                    "company_name": row.get("company_name", "").strip(),
                    "industry": row.get("industry", "").strip(),
                    "year": year,
                    "kpi": kpi,
                    "value": value,
                    "source": row.get("source", "").strip(),
                    "tag": row.get("tag", "").strip(),
                }
    return index


def load_qrels(qrels_path: Path | None) -> dict[str, list[dict[str, str | int]]]:
    """Load TREC-format qrels and group by query_id.

    Returns {query_id: [{"doc_id": "...", "relevance": int}, ...]}.
    If qrels_path is None or doesn't exist, returns empty dict.
    """
    if qrels_path is None or not qrels_path.exists():
        return {}
    index: dict[str, list[dict[str, str | int]]] = {}
    with qrels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            query_id, _iter, doc_id, relevance = parts
            if query_id not in index:
                index[query_id] = []
            index[query_id].append({"doc_id": doc_id, "relevance": int(relevance)})
    return index


def find_mmd_path(
    reports_dir: Path, exchange: str, ticker: str, year: int
) -> Path | None:
    """Find the .mmd file (not _det.mmd) for a given report."""
    report_id = f"{exchange}_{ticker}_{year}"
    report_dir = reports_dir / report_id
    if not report_dir.is_dir():
        return None
    mmd_path = report_dir / f"{report_id}.mmd"
    if mmd_path.exists():
        return mmd_path
    return None


def build_needle_dataframe(
    queries: list[dict[str, str]],
    gt_index: dict[str, dict],
    reports_dir: Path,
    qrels_index: dict[str, list[dict[str, str | int]]] | None = None,
) -> pd.DataFrame:
    """Join queries with ground truth, mmd text, and qrels.

    Returns a DataFrame with one row per query.
    """
    rows = []
    mmd_cache: dict[str, str | None] = {}  # report_id -> mmd text (cached)
    missing_gt = 0
    missing_mmd = 0

    for q in tqdm(queries, desc="Building rows", unit="query"):
        qid = q["query_id"]
        gt = gt_index.get(qid)
        if gt is None:
            missing_gt += 1
            continue

        report_id = f"{gt['exchange']}_{gt['ticker']}_{gt['year']}"

        # Cache mmd reads (many queries point to the same report)
        if report_id not in mmd_cache:
            mmd_path = find_mmd_path(
                reports_dir, gt["exchange"], gt["ticker"], gt["year"]
            )
            if mmd_path is not None:
                mmd_cache[report_id] = mmd_path.read_text(encoding="utf-8")
            else:
                mmd_cache[report_id] = None

        mmd_text = mmd_cache[report_id]
        if mmd_text is None:
            missing_mmd += 1
            continue

        # Look up qrels for this query (empty list if none available)
        qrels = qrels_index.get(qid, []) if qrels_index else []

        rows.append(
            {
                "query_id": qid,
                "query_text": q["query_text"],
                "ticker": gt["ticker"],
                "exchange": gt["exchange"],
                "company_name": gt["company_name"],
                "industry": gt["industry"],
                "year": gt["year"],
                "kpi": gt["kpi"],
                "value": gt["value"],
                "source": gt["source"],
                "tag": gt["tag"],
                "qrels": qrels,
                "mmd_text": mmd_text,
            }
        )

    if missing_gt:
        print(
            f"  WARNING: {missing_gt} queries had no ground-truth match",
            file=sys.stderr,
        )
    if missing_mmd:
        print(f"  WARNING: {missing_mmd} queries had no .mmd file", file=sys.stderr)

    return pd.DataFrame(rows)


def copy_mmd_files(df: pd.DataFrame, reports_dir: Path, dest_dir: Path):
    """Copy unique .mmd files referenced by the dataframe."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Get unique reports
    unique_reports = df[["exchange", "ticker", "year"]].drop_duplicates()
    copied = 0
    for _, row in tqdm(
        unique_reports.iterrows(),
        total=len(unique_reports),
        desc=f"Copying .mmd → {dest_dir.name}",
        unit="file",
    ):
        report_id = f"{row['exchange']}_{row['ticker']}_{int(row['year'])}"
        src = reports_dir / report_id / f"{report_id}.mmd"
        if src.exists():
            shutil.copy2(src, dest_dir / src.name)
            copied += 1
    print(f"  Copied {copied} .mmd files")


def build_readme(
    output_dir: Path,
    eval_nrows: int,
    eval_nreports: int,
    no_eval_nrows: int,
    no_eval_nreports: int,
):
    """Write a HuggingFace dataset card README.md."""
    readme = f"""---
configs:
- config_name: eval
  data_files:
  - split: test
    path: eval/data-*.parquet
- config_name: no_eval
  data_files:
  - split: train
    path: no_eval/data-*.parquet
task_categories:
- question-answering
language:
- en
tags:
- financial-reports
- ocr
- kpi-extraction
- annual-reports
- needle-in-a-haystack
- long-context
---

# Ardian Needle-in-a-Haystack KPI Benchmark

A long-context benchmark measuring whether an LLM can **find one specific KPI value
inside a large, noisy OCR'd annual report and transcribe it precisely**.

## Dataset Description

Each row is a single query that names one company, one fiscal year, and one KPI.
The model must locate that figure in a ~100k-token OCR'd Markdown document and
return it as a numeric value in raw single units. Predictions are scored against
ground-truth values from SEC EDGAR / yfinance / Alpha Vantage.

### Configs

| Config | Queries | Reports | Purpose |
|--------|---------|---------|---------|
| `eval` | {eval_nrows:,} | {eval_nreports:,} | Benchmark evaluation |
| `no_eval` | {no_eval_nrows:,} | {no_eval_nreports:,} | Training / development |

### Schema

Each row in the parquet files contains:

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | string | Unique query identifier (`{{ticker}}_{{kpi}}_{{year}}`) |
| `query_text` | string | Natural-language question |
| `ticker` | string | Stock ticker symbol |
| `exchange` | string | Stock exchange (NYSE, NASDAQ, LSE, AMEX, ASX, OTC) |
| `company_name` | string | Company long name |
| `industry` | string | Industry classification |
| `year` | int | Fiscal year |
| `kpi` | string | KPI key (e.g. `revenue`, `net_income`, `total_assets`) |
| `value` | float64 | Ground-truth KPI value (raw single units) |
| `source` | string | Data source (`edgar`, `yfinance`, `alphavantage`) |
| `tag` | string | XBRL tag or derivation method used |
| `qrels` | list[{{doc_id: str, relevance: int}}] | Page-level relevance judgments (TREC grades 0/1/2) |
| `mmd_text` | string | Full OCR text of the annual report (Markdown with page splits) |

### Additional Files

- `eval/mmd/` and `no_eval/mmd/`: Raw `.mmd` files (same text as the `mmd_text` column).

### OCR Format

The `.mmd` files use Markdown with page boundaries marked by `<--- Page Split --->`.

### KPI Value Conventions

- **Monetary values** (revenue, net_income, total_assets, etc.): in raw single units
  of the reporting currency. E.g. $1.5 billion revenue = `1500000000.0`.
- **Per-share values** (eps_basic, eps_diluted): as reported, not scaled.
- **Share counts** (shares_outstanding): in single shares.
- **Capex / dividends_paid**: positive outflows.
- **Cash flow subtotals**: with their reported sign (negative = outflow).

## Usage

```python
from datasets import load_dataset

# Load eval set
ds = load_dataset("ardian/ardian-needle-haystack", "eval")

# Each row is one query
row = ds["test"][0]
print(row["query_text"])   # natural-language question
print(row["value"])        # ground-truth answer
print(len(row["mmd_text"]))  # ~500k chars of OCR text

# Page-level relevance judgments (for retrieval evaluation)
relevant_pages = [q for q in row["qrels"] if q["relevance"] == 2]
print(relevant_pages)  # [{{\"doc_id\": \"NYSE_AAP_2017/page_0042\", \"relevance\": 2}}, ...]
```

## Scoring

A prediction is **matched** if `|pred − gt| / |gt| ≤ 1%` (default tolerance).
Stricter scoring at ±0.05% is also reported. See the benchmark README for
full scoring methodology including wrong-answer diagnostics (year_shift,
sign_error, scale_error, scope_factor).

## Data Sources

- **OCR text**: DeepSeek OCR applied to annual report PDFs.
- **KPI values**: SEC EDGAR (XBRL) for US listings; yfinance for non-US; Alpha Vantage gap-fill.
- **Queries**: Randomly sampled from curated templates with company name aliases.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"  Wrote {output_dir / 'README.md'}")


def prepare_config(
    config_name: str,
    test_set_path: Path,
    kpi_csv_path: Path,
    reports_dir: Path,
    output_dir: Path,
    qrels_path: Path | None = None,
    skip_copy: bool = False,
) -> tuple[int, int]:
    """Prepare one config. Returns (n_queries, n_unique_reports)."""
    print(f"\n{'=' * 60}")
    print(f"Preparing config: {config_name}")
    print(f"  Test set: {test_set_path}")
    print(f"  KPI CSV:  {kpi_csv_path}")
    print(f"  Reports:  {reports_dir}")
    print(f"  Qrels:    {qrels_path or '(none)'}")
    print(f"  Output:   {output_dir}")
    print(f"{'=' * 60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading queries and ground truth...")
    queries = load_test_set(test_set_path)
    print(f"  Queries loaded: {len(queries):,}")
    gt_index = load_kpi_ground_truth(kpi_csv_path)
    print(f"  Ground-truth entries: {len(gt_index):,}")

    # Load qrels
    print("\n[2/4] Loading qrels...")
    qrels_index = load_qrels(qrels_path)
    if qrels_index:
        print(
            f"  Qrels loaded: {sum(len(v) for v in qrels_index.values()):,} judgments for {len(qrels_index):,} queries"
        )
    else:
        print("  No qrels available (column will be empty lists)")

    # Build dataframe
    print("\n[3/4] Joining queries with ground truth, mmd text, and qrels...")
    df = build_needle_dataframe(queries, gt_index, reports_dir, qrels_index)
    print(f"  Final rows: {len(df):,}")
    n_reports = df[["exchange", "ticker", "year"]].drop_duplicates().shape[0]
    print(f"  Unique reports: {n_reports:,}")
    if qrels_index:
        n_with_qrels = df["qrels"].apply(len).gt(0).sum()
        print(f"  Rows with qrels: {n_with_qrels:,}/{len(df):,}")

    # Write sharded parquet (HF viewer has a 300MB scan limit)
    ROWS_PER_SHARD = 1000
    n_shards = (len(df) + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD
    print(f"\n[4/4] Writing {n_shards} parquet shards ({ROWS_PER_SHARD} rows each)...")
    total_bytes = 0
    for i in range(n_shards):
        shard = df.iloc[i * ROWS_PER_SHARD : (i + 1) * ROWS_PER_SHARD]
        shard_path = output_dir / f"data-{i:05d}-of-{n_shards:05d}.parquet"
        shard.to_parquet(shard_path, index=False, engine="pyarrow")
        total_bytes += shard_path.stat().st_size
    total_mb = total_bytes / (1024 * 1024)
    print(
        f"  Written: {total_mb:.1f} MB total, {n_shards} shards, {len(df):,} rows × {len(df.columns)} columns"
    )

    # Copy .mmd files
    if not skip_copy:
        mmd_dest = output_dir / "mmd"
        copy_mmd_files(df, reports_dir, mmd_dest)

    return len(df), n_reports


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the Ardian needle-in-a-haystack benchmark for HuggingFace upload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "hf_output" / "needle",
        help="Output directory for the HF dataset repo (default: hf_output/needle/)",
    )
    parser.add_argument(
        "--eval-test-set",
        type=Path,
        default=DEFAULTS["eval"]["test_set"],
        help="Path to eval test_set.csv (query_id, query_text)",
    )
    parser.add_argument(
        "--eval-kpi-csv",
        type=Path,
        default=DEFAULTS["eval"]["kpi_csv"],
        help="Path to eval kpi_long CSV",
    )
    parser.add_argument(
        "--eval-reports",
        type=Path,
        default=DEFAULTS["eval"]["reports"],
        help="Path to eval reports directory",
    )
    parser.add_argument(
        "--eval-qrels",
        type=Path,
        default=DEFAULTS["eval"]["qrels"],
        help="Path to eval qrels file (TREC format)",
    )
    parser.add_argument(
        "--no-eval-test-set",
        type=Path,
        default=None,
        help="Path to no_eval test_set.csv (not yet available)",
    )
    parser.add_argument(
        "--no-eval-kpi-csv",
        type=Path,
        default=DEFAULTS["no_eval"]["kpi_csv"],
        help="Path to no_eval kpi_long CSV",
    )
    parser.add_argument(
        "--no-eval-reports",
        type=Path,
        default=DEFAULTS["no_eval"]["reports"],
        help="Path to no_eval reports directory",
    )
    parser.add_argument(
        "--no-eval-qrels",
        type=Path,
        default=DEFAULTS["no_eval"]["qrels"],
        help="Path to no_eval qrels file (TREC format, not yet available)",
    )
    parser.add_argument(
        "--skip-no-eval",
        action="store_true",
        default=True,
        help="Skip no_eval config (default: skipped since queries not yet generated)",
    )
    parser.add_argument(
        "--include-no-eval",
        action="store_true",
        help="Include no_eval config (requires --no-eval-test-set)",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip copying .mmd files (parquet only)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_nrows = 0
    eval_nreports = 0
    no_eval_nrows = 0
    no_eval_nreports = 0

    # Always run eval
    eval_nrows, eval_nreports = prepare_config(
        config_name="eval",
        test_set_path=args.eval_test_set,
        kpi_csv_path=args.eval_kpi_csv,
        reports_dir=args.eval_reports,
        output_dir=args.output_dir / "eval",
        qrels_path=args.eval_qrels,
        skip_copy=args.skip_copy,
    )

    # Optionally run no_eval
    if args.include_no_eval:
        if args.no_eval_test_set is None:
            print(
                "\nERROR: --no-eval-test-set required when --include-no-eval is set",
                file=sys.stderr,
            )
            sys.exit(1)
        no_eval_nrows, no_eval_nreports = prepare_config(
            config_name="no_eval",
            test_set_path=args.no_eval_test_set,
            kpi_csv_path=args.no_eval_kpi_csv,
            reports_dir=args.no_eval_reports,
            output_dir=args.output_dir / "no_eval",
            qrels_path=args.no_eval_qrels,
            skip_copy=args.skip_copy,
        )

    # Generate README
    print(f"\n{'=' * 60}")
    print("Writing README.md...")
    build_readme(
        args.output_dir, eval_nrows, eval_nreports, no_eval_nrows, no_eval_nreports
    )

    print(f"\n{'=' * 60}")
    print("Done! Output ready at:", args.output_dir)
    print("\nNext steps:")
    print(f"  1. Review {args.output_dir / 'README.md'}")
    print(
        "  2. Generate no_eval queries, then re-run with --include-no-eval --no-eval-test-set <path>"
    )
    print(
        f"  3. Upload: huggingface-cli upload --repo-type dataset <org>/<name> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
