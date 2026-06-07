"""
Prepare the Ardian annual-reports + KPI dataset for HuggingFace upload.

Output layout (one HF Dataset repository):
    hf_output/
    ├── README.md                  # dataset card (generated)
    ├── no_eval/                   # "no_eval" config
    │   ├── data.parquet           # wide KPIs + mmd_text per report
    │   └── mmd/                   # raw .mmd files
    │       └── {EX}_{TICK}_{YEAR}.mmd
    └── eval/                      # "eval" config
        ├── data.parquet
        ├── mmd/
        └── images/                # page-level JPEGs (eval only)
            └── {EX}_{TICK}_{YEAR}/

Each parquet row = one annual report (ticker, exchange, company_name,
industry, year, 31 KPI value columns, mmd_text).
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── 31 tracked KPIs (sorted alphabetically) ──────────────────────────
KPI_NAMES = sorted([
    "accounts_payable",
    "accounts_receivable",
    "capex",
    "cash_and_equivalents",
    "cash_incl_restricted",
    "cost_of_revenue",
    "depreciation_amortization",
    "dividends_paid",
    "eps_basic",
    "eps_diluted",
    "financing_cash_flow",
    "gross_profit",
    "income_tax_expense",
    "interest_expense",
    "inventory",
    "investing_cash_flow",
    "long_term_debt_current",
    "long_term_debt_noncurrent",
    "long_term_debt_total",
    "net_income",
    "operating_cash_flow",
    "operating_income",
    "rd_expense",
    "revenue",
    "sga_expense",
    "shares_outstanding",
    "short_term_borrowings",
    "stockholders_equity",
    "stockholders_equity_incl_nci",
    "total_assets",
    "total_liabilities",
])

# ── default paths (relative to repo root) ─────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "no_eval": {
        "csv": REPO_ROOT / "KPI_analysis/find_more_queries/full_6k/kpi_long_no_eval.csv",
        "reports": REPO_ROOT / "DeepSeekOCR_Ardian_with_kpis_no_eval",
    },
    "eval": {
        "csv": REPO_ROOT / "KPI_analysis/find_more_queries/full_6k/kpi_long_eval.csv",
        "reports": REPO_ROOT / "DeepSeekOCR_Ardian_evaluation_set_reports",
    },
}


def pivot_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format KPI rows into wide format (one row per report)."""
    # Columns to drop from the CSV
    drop_cols = ["source", "verified", "name_match_score", "tag"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # exchange_ocr matches directory names; rename to "exchange" and
    # drop the Yahoo-derived "exchange" column
    if "exchange_ocr" in df.columns:
        df = df.drop(columns=["exchange"]).rename(columns={"exchange_ocr": "exchange"})

    # Fill NaN in metadata columns so pivot doesn't drop rows
    for col in ["company_name", "industry"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Pivot: one row per (ticker, exchange, company_name, industry, year)
    index_cols = ["ticker", "exchange", "company_name", "industry", "year"]
    wide = df.pivot_table(
        index=index_cols,
        columns="kpi",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Flatten column names (pivot_table produces MultiIndex)
    wide.columns.name = None
    wide.columns = [str(c) for c in wide.columns]

    # Deduplicate: one row per (ticker, exchange, year)
    # A few reports may have varying company_name/industry across KPI rows;
    # keep the first occurrence to ensure strict 1-to-1 with report directories.
    wide = wide.drop_duplicates(subset=["ticker", "exchange", "year"], keep="first")

    # Ensure all 31 KPI columns exist (fill missing with NaN)
    for kpi in KPI_NAMES:
        if kpi not in wide.columns:
            wide[kpi] = pd.NA

    # Reorder: metadata first, then KPIs alphabetically
    meta_cols = ["ticker", "exchange", "company_name", "industry", "year"]
    wide = wide[meta_cols + KPI_NAMES]

    return wide


def attach_mmd_text(df: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    """Add an mmd_text column by reading the .mmd file for each row."""
    texts = []
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading .mmd files", unit="file"):
        report_id = f"{row['exchange']}_{row['ticker']}_{row['year']}"
        mmd_path = reports_dir / report_id / f"{report_id}.mmd"
        if mmd_path.exists():
            texts.append(mmd_path.read_text(encoding="utf-8"))
        else:
            texts.append(None)
            missing += 1

    df["mmd_text"] = texts
    if missing:
        print(f"  WARNING: {missing}/{len(df)} .mmd files not found", file=sys.stderr)
    return df


def copy_mmd_files(df: pd.DataFrame, reports_dir: Path, dest_dir: Path):
    """Copy raw .mmd files to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying .mmd → {dest_dir.name}/mmd", unit="file"):
        report_id = f"{row['exchange']}_{row['ticker']}_{row['year']}"
        src = reports_dir / report_id / f"{report_id}.mmd"
        if src.exists():
            shutil.copy2(src, dest_dir / src.name)
            copied += 1
    print(f"  Copied {copied} .mmd files")


def copy_images(df: pd.DataFrame, reports_dir: Path, dest_dir: Path):
    """Copy images/ subdirectories to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying images → {dest_dir.name}/images", unit="dir"):
        report_id = f"{row['exchange']}_{row['ticker']}_{row['year']}"
        src = reports_dir / report_id / "images"
        if src.is_dir():
            shutil.copytree(src, dest_dir / report_id / "images")
            copied += 1
    print(f"  Copied {copied} image directories")


def build_readme(output_dir: Path, no_eval_nrows: int, eval_nrows: int):
    """Write a HuggingFace dataset card README.md."""
    readme = f"""---
configs:
- config_name: no_eval
  data_files:
  - split: train
    path: no_eval/data.parquet
- config_name: eval
  data_files:
  - split: test
    path: eval/data.parquet
task_categories:
- question-answering
- table-question-answering
language:
- en
tags:
- financial-reports
- ocr
- kpi-extraction
- annual-reports
---

# Ardian Annual Reports + KPIs

OCR'd annual reports with ground-truth KPI values for financial information extraction benchmarking.

## Dataset Description

This dataset pairs OCR-extracted annual report text (from DeepSeek OCR) with structured KPI ground-truth values. It is designed for evaluating LLM-based financial information extraction, retrieval, and needle-in-a-haystack tasks.

### Configs

| Config | Reports | Companies | KPI rows | Years | Purpose |
|--------|---------|-----------|----------|-------|---------|
| `no_eval` | {no_eval_nrows:,} | 725 | 104,529 | 2009–2024 | Training / development |
| `eval` | {eval_nrows:,} | 111 | 13,519 | 2017–2022 | Benchmark evaluation |

### Schema

Each row in the parquet files contains:

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Stock ticker symbol |
| `exchange` | string | Stock exchange (NYSE, NASDAQ, LSE, AMEX, ASX, OTC) |
| `company_name` | string | Company long name |
| `industry` | string | Industry classification |
| `year` | int | Fiscal year |
| `revenue` | float64 | Total revenue |
| `net_income` | float64 | Net income |
| `total_assets` | float64 | Total assets |
| `total_liabilities` | float64 | Total liabilities |
| ... | float64 | 31 KPI columns total (see below) |
| `mmd_text` | string | Full OCR text of the annual report (Markdown with page splits) |

**KPI columns (31):** `accounts_payable`, `accounts_receivable`, `capex`, `cash_and_equivalents`, `cash_incl_restricted`, `cost_of_revenue`, `depreciation_amortization`, `dividends_paid`, `eps_basic`, `eps_diluted`, `financing_cash_flow`, `gross_profit`, `income_tax_expense`, `interest_expense`, `inventory`, `investing_cash_flow`, `long_term_debt_current`, `long_term_debt_noncurrent`, `long_term_debt_total`, `net_income`, `operating_cash_flow`, `operating_income`, `rd_expense`, `revenue`, `sga_expense`, `shares_outstanding`, `short_term_borrowings`, `stockholders_equity`, `stockholders_equity_incl_nci`, `total_assets`, `total_liabilities`.

KPI values are in millions (as-reported, no FX conversion). NaN means the KPI was not available for that report/year.

### Additional Files

- `no_eval/mmd/` and `eval/mmd/`: Raw `.mmd` files (same text as the `mmd_text` column, for direct file access).
- `eval/images/`: Page-level JPEG images for eval reports (not included for no_eval to save space).

### OCR Format

The `.mmd` files use Markdown with page boundaries marked by `<--- Page Split --->`. Images are referenced as `![](images/{{page}}_{{idx}}.jpg)`.

## Usage

```python
from datasets import load_dataset

# Load training set
ds = load_dataset("ardian/ardian-annual-reports", "no_eval")

# Load eval set
ds_eval = load_dataset("ardian/ardian-annual-reports", "eval")

# Example: filter to reports with revenue data
ds_filtered = ds["train"].filter(lambda x: x["revenue"] is not None)
```

## Data Sources

- **OCR text**: DeepSeek OCR applied to annual report PDFs from SEC EDGAR, LSE, ASX, and other exchanges.
- **KPI values**: SEC EDGAR (XBRL companyfacts) for US listings; yfinance for non-US; Alpha Vantage for gap-fill.

## Citation

If you use this dataset, please cite the Ardian dataset bench project.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"  Wrote {output_dir / 'README.md'}")


def prepare_config(
    config_name: str,
    csv_path: Path,
    reports_dir: Path,
    output_dir: Path,
    copy_img: bool,
) -> int:
    """Prepare one config (no_eval or eval). Returns the number of parquet rows."""
    print(f"\n{'='*60}")
    print(f"Preparing config: {config_name}")
    print(f"  CSV:      {csv_path}")
    print(f"  Reports:  {reports_dir}")
    print(f"  Output:   {output_dir}")
    print(f"  Images:   {'yes' if copy_img else 'no'}")
    print(f"{'='*60}")

    # Create output dirs
    mmd_dest = output_dir / "mmd"
    img_dest = output_dir / "images" if copy_img else None
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read and pivot
    print("\n[1/4] Reading CSV and pivoting to wide format...")
    df = pd.read_csv(csv_path)
    print(f"  Raw rows: {len(df):,}")
    wide = pivot_kpis(df)
    print(f"  Pivoted rows (unique reports): {len(wide):,}")
    print(f"  Columns: {list(wide.columns)}")

    # Attach mmd text
    print("\n[2/4] Attaching .mmd text...")
    wide = attach_mmd_text(wide, reports_dir)

    # Write parquet
    parquet_path = output_dir / "data.parquet"
    print(f"\n[3/4] Writing parquet → {parquet_path}...")
    wide.to_parquet(parquet_path, index=False, engine="pyarrow")
    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"  Written: {size_mb:.1f} MB, {len(wide):,} rows × {len(wide.columns)} columns")

    # Copy raw files
    print(f"\n[4/4] Copying raw files...")
    copy_mmd_files(wide, reports_dir, mmd_dest)
    if copy_img and img_dest is not None:
        copy_images(wide, reports_dir, img_dest)

    return len(wide)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the Ardian dataset for HuggingFace upload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "hf_output",
        help="Output directory for the HF dataset repo (default: hf_output/)",
    )
    parser.add_argument(
        "--no-eval-csv",
        type=Path,
        default=DEFAULTS["no_eval"]["csv"],
        help="Path to no_eval KPI CSV",
    )
    parser.add_argument(
        "--no-eval-reports",
        type=Path,
        default=DEFAULTS["no_eval"]["reports"],
        help="Path to no_eval reports directory",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=DEFAULTS["eval"]["csv"],
        help="Path to eval KPI CSV",
    )
    parser.add_argument(
        "--eval-reports",
        type=Path,
        default=DEFAULTS["eval"]["reports"],
        help="Path to eval reports directory",
    )
    parser.add_argument(
        "--skip-no-eval",
        action="store_true",
        help="Skip no_eval config (useful for testing eval only)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval config",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip copying eval images",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip copying .mmd and image files (parquet only)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    no_eval_nrows = 0
    eval_nrows = 0

    if not args.skip_no_eval:
        no_eval_nrows = prepare_config(
            config_name="no_eval",
            csv_path=args.no_eval_csv,
            reports_dir=args.no_eval_reports,
            output_dir=args.output_dir / "no_eval",
            copy_img=False,  # never copy images for no_eval
        )

    if not args.skip_eval:
        eval_nrows = prepare_config(
            config_name="eval",
            csv_path=args.eval_csv,
            reports_dir=args.eval_reports,
            output_dir=args.output_dir / "eval",
            copy_img=not args.skip_images,
        )

    # Generate README
    print(f"\n{'='*60}")
    print("Writing README.md...")
    build_readme(args.output_dir, no_eval_nrows, eval_nrows)

    print(f"\n{'='*60}")
    print("Done! Output ready at:", args.output_dir)
    print("\nNext steps:")
    print(f"  1. Review {args.output_dir / 'README.md'}")
    print(f"  2. Create a HuggingFace dataset repo")
    print(f"  3. Upload: huggingface-cli upload --repo-type dataset <org>/<name> {args.output_dir}")


if __name__ == "__main__":
    main()
