"""Read all per-ticker JSON files produced by scan_and_fetch.py and write:

  find_more_queries/companies.csv    — ticker catalogue (same format as cleaned/)
  find_more_queries/summary.md       — per-industry and per-KPI coverage report

Usage
-----
    uv run python KPI_analysis/find_more_queries/build_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KPI_DIR = HERE.parent
REPO_ROOT = KPI_DIR.parent
sys.path.insert(0, str(KPI_DIR))

from tags import KPI_DEFS

OUTPUT_DIR = HERE / "output" / "raw"
MANIFEST_PATH = HERE / "ocr_companies.json"
COMPANIES_CSV = HERE / "companies.csv"
SUMMARY_MD = HERE / "summary.md"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_records() -> list[dict]:
    records = []
    for json_file in sorted(OUTPUT_DIR.glob("*.json")):
        try:
            records.append(json.loads(json_file.read_text()))
        except Exception as exc:
            print(f"  [WARN] could not read {json_file.name}: {exc}", file=sys.stderr)
    return records


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_KPI_KEYS = [kpi.key for kpi in KPI_DEFS]

# Core KPIs used for the qrels pipeline (matches validate_ocr_kpis.py defaults)
CORE_KPI_KEYS = [
    "revenue", "gross_profit", "operating_income", "net_income",
    "total_assets", "total_liabilities", "cash_and_equivalents",
    "operating_cash_flow", "capex",
]


def years_with_data(record: dict, kpi: str) -> set[int]:
    per_year = record.get("kpis", {}).get(kpi, {})
    return {int(y) for y in per_year}


def total_kpi_years(record: dict) -> tuple[int, int]:
    """(filled cells, total possible cells) across all KPIs × years_fetch."""
    years = set(record.get("years_fetch") or record.get("years_in_ocr", []))
    filled = 0
    total = 0
    for kpi in ALL_KPI_KEYS:
        for y in years:
            total += 1
            per_year = record.get("kpis", {}).get(kpi, {})
            if str(y) in per_year or y in per_year:
                filled += 1
    return filled, total


def core_kpi_years(record: dict) -> tuple[int, int]:
    years = set(record.get("years_fetch") or record.get("years_in_ocr", []))
    filled = 0
    total = 0
    for kpi in CORE_KPI_KEYS:
        for y in years:
            total += 1
            per_year = record.get("kpis", {}).get(kpi, {})
            if str(y) in per_year or y in per_year:
                filled += 1
    return filled, total


def potential_queries(record: dict) -> int:
    """Unique (ticker, year, kpi) triples where both OCR and KPI data exist (all 31 KPIs)."""
    ocr_years = set(record.get("years_in_ocr", []))
    fetch_years = set(record.get("years_fetch") or [])
    count = 0
    for kpi in ALL_KPI_KEYS:
        per_year = record.get("kpis", {}).get(kpi, {})
        for y_str, _ in per_year.items():
            y = int(y_str)
            if y in ocr_years and y in fetch_years:
                count += 1
    return count


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_companies_csv(records: list[dict]) -> None:
    # Same columns as cleaned/ CSVs plus extras for traceability
    fieldnames = [
        "Ticker", "Company Name", "Sector", "Industry",
        "Exchange (Yahoo)", "Industry Dir", "Years in OCR",
        "Years Fetched", "KPIs Found", "Core KPI Coverage %",
        "Potential Queries", "Source", "Error",
    ]
    with COMPANIES_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(records, key=lambda x: (x.get("industry_dir", ""), x.get("ticker", ""))):
            filled_core, total_core = core_kpi_years(r)
            core_pct = round(100 * filled_core / total_core, 1) if total_core else 0.0
            years_ocr = r.get("years_in_ocr", [])
            years_fetch = r.get("years_fetch", [])
            kpis_found = len([k for k in ALL_KPI_KEYS if r.get("kpis", {}).get(k)])
            writer.writerow({
                "Ticker": r.get("ticker", ""),
                "Company Name": r.get("company_name") or r.get("entity_name", ""),
                "Sector": r.get("sector", ""),
                "Industry": r.get("industry") or r.get("industry_dir", ""),
                "Exchange (Yahoo)": r.get("exchange", ""),
                "Industry Dir": r.get("industry_dir", ""),
                "Years in OCR": f"{min(years_ocr)}-{max(years_ocr)}" if years_ocr else "",
                "Years Fetched": f"{min(years_fetch)}-{max(years_fetch)}" if years_fetch else "",
                "KPIs Found": kpis_found,
                "Core KPI Coverage %": core_pct,
                "Potential Queries": potential_queries(r),
                "Source": r.get("source", ""),
                "Error": r.get("error", ""),
            })
    print(f"Wrote {len(records)} rows to {COMPANIES_CSV}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def write_summary_md(records: list[dict], manifest: list[dict]) -> None:
    lines: list[str] = []

    lines += [
        "# Extended KPI Coverage Summary",
        "",
        f"Generated from `scan_and_fetch.py` output.  "
        f"Source OCR tree: `{OCR_ROOT_LABEL}`",
        "",
    ]

    # --- Overall stats ---
    total = len(records)
    with_error = sum(1 for r in records if r.get("error"))
    with_any_kpi = sum(1 for r in records if r.get("kpis"))
    total_potential = sum(potential_queries(r) for r in records)
    total_filled, total_cells = 0, 0
    for r in records:
        f, t = total_kpi_years(r)
        total_filled += f
        total_cells += t

    lines += [
        "## Overall",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Companies in OCR corpus | {total} |",
        f"| Successfully fetched (no error) | {total - with_error} |",
        f"| Companies with at least one KPI | {with_any_kpi} |",
        f"| Total potential (ticker, year, KPI) triples | {total_potential:,} |",
        f"| Overall KPI×year fill rate (all 31 KPIs) | {100*total_filled/total_cells:.1f}% ({total_filled:,}/{total_cells:,}) |",
        "",
    ]

    # --- By industry ---
    lines += ["## By Industry", ""]
    lines += [
        "| Industry | Companies | No Error | Potential Queries | Core Coverage % |",
        "|----------|-----------|----------|-------------------|-----------------|",
    ]

    by_industry: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = r.get("industry_dir") or r.get("industry", "unknown")
        by_industry[key].append(r)

    for ind in sorted(by_industry):
        group = by_industry[ind]
        ok = sum(1 for r in group if not r.get("error"))
        pq = sum(potential_queries(r) for r in group)
        filled_c, total_c = 0, 0
        for r in group:
            f, t = core_kpi_years(r)
            filled_c += f
            total_c += t
        core_pct = f"{100*filled_c/total_c:.1f}%" if total_c else "n/a"
        lines.append(f"| {ind} | {len(group)} | {ok} | {pq:,} | {core_pct} |")
    lines.append("")

    # --- By exchange ---
    lines += ["## By Exchange (OCR label)", ""]
    lines += [
        "| Exchange | Companies | No Error | Potential Queries |",
        "|----------|-----------|----------|-------------------|",
    ]
    by_exchange: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_exchange[r.get("exchange_ocr", "?")].append(r)
    for ex in sorted(by_exchange):
        group = by_exchange[ex]
        ok = sum(1 for r in group if not r.get("error"))
        pq = sum(potential_queries(r) for r in group)
        lines.append(f"| {ex} | {len(group)} | {ok} | {pq:,} |")
    lines.append("")

    # --- By KPI (core set) ---
    lines += ["## Per-KPI Coverage (core set, years present in OCR)", ""]
    lines += [
        "| KPI | Companies with Data | Total (ticker,year) Pairs | Fill Rate |",
        "|-----|---------------------|---------------------------|-----------|",
    ]
    for kpi in CORE_KPI_KEYS:
        companies_with = 0
        pairs_filled = 0
        pairs_total = 0
        for r in records:
            years_fetch = set(r.get("years_fetch") or [])
            per_year = r.get("kpis", {}).get(kpi, {})
            has_any = bool(per_year)
            if has_any:
                companies_with += 1
            for y in years_fetch:
                pairs_total += 1
                if str(y) in per_year or y in per_year:
                    pairs_filled += 1
        rate = f"{100*pairs_filled/pairs_total:.1f}%" if pairs_total else "n/a"
        lines.append(f"| {kpi} | {companies_with}/{total} | {pairs_filled:,}/{pairs_total:,} | {rate} |")
    lines.append("")

    # --- All KPIs ---
    lines += ["## Per-KPI Coverage (all 31 KPIs, years present in OCR)", ""]
    lines += [
        "| KPI | Companies with Data | Fill Rate |",
        "|-----|---------------------|-----------|",
    ]
    for kpi in ALL_KPI_KEYS:
        companies_with = 0
        pairs_filled = 0
        pairs_total = 0
        for r in records:
            years_fetch = set(r.get("years_fetch") or [])
            per_year = r.get("kpis", {}).get(kpi, {})
            has_any = bool(per_year)
            if has_any:
                companies_with += 1
            for y in years_fetch:
                pairs_total += 1
                if str(y) in per_year or y in per_year:
                    pairs_filled += 1
        rate = f"{100*pairs_filled/pairs_total:.1f}%" if pairs_total else "n/a"
        lines.append(f"| {kpi} | {companies_with}/{total} | {rate} |")
    lines.append("")

    # --- Matched vs unmatched in cleaned CSVs ---
    if manifest:
        matched = sum(1 for e in manifest if e.get("in_cleaned_csv"))
        lines += [
            "## Metadata Matching (OCR vs Cleaned CSVs)",
            "",
            f"| Status | Count |",
            f"|--------|-------|",
            f"| Matched in cleaned/ CSVs | {matched} |",
            f"| Not found in cleaned/ CSVs | {len(manifest) - matched} |",
            "",
            "### Companies not in cleaned/ CSVs",
            "",
            "These companies exist in the OCR corpus but were not found in "
            "`tickers_lists/cleaned/` (likely filtered out during cleaning or "
            "added later).  KPIs were still fetched directly using the OCR ticker.",
            "",
            "| Ticker (OCR) | Exchange | Industry Dir |",
            "|-------------|---------|--------------|",
        ]
        for e in manifest:
            if not e.get("in_cleaned_csv"):
                lines.append(f"| {e['ticker_ocr']} | {e['exchange_ocr']} | {e['industry_dir']} |")
        lines.append("")

    # --- Year range summary ---
    lines += ["## Year Range in OCR Corpus", ""]
    year_counts: dict[int, int] = defaultdict(int)
    for r in records:
        for y in r.get("years_in_ocr", []):
            year_counts[y] += 1
    lines += [
        "| Year | Reports in OCR |",
        "|------|----------------|",
    ]
    for y in sorted(year_counts):
        lines.append(f"| {y} | {year_counts[y]} |")
    lines.append("")

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"Wrote summary to {SUMMARY_MD}", file=sys.stderr)


OCR_ROOT_LABEL = "/data/workspace/charles/pdf_ocr_deepseek/DeepSeekOCR_Ardian_raw_3kdocs"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    records = load_records()
    if not records:
        print(f"No JSON files found in {OUTPUT_DIR}", file=sys.stderr)
        return 1
    print(f"Loaded {len(records)} records from {OUTPUT_DIR}", file=sys.stderr)
    manifest = load_manifest()
    write_companies_csv(records)
    write_summary_md(records, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())


