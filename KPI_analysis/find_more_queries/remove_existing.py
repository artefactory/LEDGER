"""Remove from find_more_queries/output/raw/ all (ticker, year) pairs that are
already covered by the pruned 1k OCR corpus, to avoid duplicate work downstream.

The pruned 1k lives at DeepSeekOCR_Ardian_pruned_1k/ and uses the directory
naming convention {EXCHANGE}_{TICKER}_{YEAR}/ — the same convention as the
JSON filenames in find_more_queries/output/raw/ ({EXCHANGE}_{TICKER}.json).

What this script does
---------------------
1. Scans DeepSeekOCR_Ardian_pruned_1k for every (exchange, ticker, year) triple
   that has already been analysed.
2. For each JSON in find_more_queries/output/raw/, strips out the KPI data for
   years already in the pruned corpus, and updates years_in_ocr / years_fetch
   accordingly.  Adds a `years_removed` list for audit.
3. Rewrites the modified JSONs in place.
4. Prints a before/after summary.

After running this script, re-run build_report.py to refresh companies.csv and
summary.md.

Usage
-----
    uv run python KPI_analysis/find_more_queries/remove_existing.py
    uv run python KPI_analysis/find_more_queries/remove_existing.py --dry-run
    uv run python KPI_analysis/find_more_queries/remove_existing.py \\
        --pruned-root /other/path
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output" / "raw"
DEFAULT_PRUNED_ROOT = Path("/home/cmoslonka/ardian_dataset_bench/DeepSeekOCR_Ardian_pruned_1k")


# ---------------------------------------------------------------------------
# Step 1 – scan pruned corpus
# ---------------------------------------------------------------------------

def scan_pruned_corpus(pruned_root: Path) -> dict[tuple[str, str], set[int]]:
    """Return {(exchange, ticker): {years already analysed}} from the pruned dir."""
    covered: dict[tuple[str, str], set[int]] = defaultdict(set)
    for industry_dir in pruned_root.iterdir():
        if not industry_dir.is_dir():
            continue
        for report_dir in industry_dir.iterdir():
            if not report_dir.is_dir():
                continue
            name = report_dir.name
            parts = name.split("_")
            if len(parts) < 3:
                continue
            exchange = parts[0]
            year_str = parts[-1]
            ticker = "_".join(parts[1:-1])
            if not year_str.isdigit():
                continue
            covered[(exchange, ticker)].add(int(year_str))
    return dict(covered)


# ---------------------------------------------------------------------------
# Step 2 – filter JSONs
# ---------------------------------------------------------------------------

def filter_record(record: dict, years_to_remove: set[int]) -> tuple[dict, int]:
    """Strip KPI data for years_to_remove from record.  Returns (updated_record, cells_removed)."""
    removed_cells = 0

    for kpi_data in record.get("kpis", {}).values():
        for y in list(kpi_data.keys()):
            if int(y) in years_to_remove:
                del kpi_data[y]
                removed_cells += 1

    # Update year lists
    prev_ocr = record.get("years_in_ocr", [])
    prev_fetch = record.get("years_fetch", [])
    new_ocr = [y for y in prev_ocr if y not in years_to_remove]
    new_fetch = [y for y in prev_fetch if y not in years_to_remove]

    record["years_in_ocr"] = new_ocr
    record["years_fetch"] = new_fetch
    record.setdefault("years_removed", [])
    record["years_removed"] = sorted(record["years_removed"] + sorted(years_to_remove & set(prev_ocr)))

    # Remove KPI keys that are now fully empty
    for kpi in list(record.get("kpis", {}).keys()):
        if not record["kpis"][kpi]:
            del record["kpis"][kpi]

    return record, removed_cells


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pruned-root",
        type=Path,
        default=DEFAULT_PRUNED_ROOT,
        help=f"Root of the pruned 1k OCR corpus (default: {DEFAULT_PRUNED_ROOT}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without modifying any files.",
    )
    args = p.parse_args(argv)

    print("Scanning pruned corpus...", file=sys.stderr)
    covered = scan_pruned_corpus(args.pruned_root)
    total_pruned_pairs = sum(len(v) for v in covered.values())
    print(
        f"  Found {len(covered)} unique (exchange, ticker) pairs, "
        f"{total_pruned_pairs} (exchange, ticker, year) triples already analysed.",
        file=sys.stderr,
    )

    json_files = sorted(OUTPUT_DIR.glob("*.json"))
    print(f"  Processing {len(json_files)} JSON files in {OUTPUT_DIR}...\n", file=sys.stderr)

    stats = {
        "files_touched": 0,
        "files_fully_removed": 0,
        "years_removed": 0,
        "cells_removed": 0,
    }

    for json_file in json_files:
        record = json.loads(json_file.read_text())
        exchange_ocr = record.get("exchange_ocr", "")
        ticker = record.get("ticker", "")

        covered_years = covered.get((exchange_ocr, ticker), set())
        if not covered_years:
            continue

        # Determine which years present in this record's OCR list are covered
        ocr_years = set(record.get("years_in_ocr", []))
        years_to_remove = covered_years & ocr_years
        if not years_to_remove:
            continue

        all_removed = (years_to_remove >= ocr_years)

        if args.dry_run:
            print(
                f"  {'[FULLY REMOVED] ' if all_removed else ''}"
                f"{exchange_ocr}_{ticker}  remove years: {sorted(years_to_remove)}  "
                f"keep years: {sorted(ocr_years - years_to_remove)}"
            )
            stats["files_touched"] += 1
            stats["years_removed"] += len(years_to_remove)
            if all_removed:
                stats["files_fully_removed"] += 1
            continue

        record, cells_removed = filter_record(record, years_to_remove)
        json_file.write_text(json.dumps(record, indent=2, default=str))

        stats["files_touched"] += 1
        stats["years_removed"] += len(years_to_remove)
        stats["cells_removed"] += cells_removed
        if all_removed:
            stats["files_fully_removed"] += 1

        print(
            f"  {'[ALL YEARS GONE] ' if all_removed else ''}"
            f"{exchange_ocr}_{ticker:<14}  "
            f"removed {len(years_to_remove)} yrs ({sorted(years_to_remove)})  "
            f"kept {len(ocr_years - years_to_remove)} yrs  "
            f"cells_removed={cells_removed}"
        )

    print()
    if args.dry_run:
        print(f"Dry-run complete.")
    else:
        print(f"Done.")
    print(
        f"  Files touched:          {stats['files_touched']}\n"
        f"  Fully emptied:          {stats['files_fully_removed']}\n"
        f"  (ticker,year) removed:  {stats['years_removed']}\n"
        f"  KPI cells removed:      {stats['cells_removed']}"
    )
    if not args.dry_run:
        print("\nRun build_report.py to refresh companies.csv and summary.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
