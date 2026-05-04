"""Prune an OCR output directory to match the final selected company list.

The OCR root is expected to contain one subdirectory per report, named
`{EXCHANGE}_{TICKER}_{YEAR}` (matching the corresponding PDF filename
without the `.pdf` extension).

A subdirectory is KEPT iff all three hold:

1. `(EXCHANGE, TICKER)` is in `grouped/selected/companies.json` for the
   requested industry — i.e. the company survived the `filter_exchange`
   pass.
2. The company has raw PDFs for every year in the requested window (so
   the eventual dataset is balanced across years).
3. `YEAR` is within the requested window.

Every other recognized subdirectory is removed. Unrecognized entries
(e.g. `logs/`, loose files) are left alone and flagged as warnings.

Dry-run by default. Pass `--execute` to actually delete.

Usage:
    uv run python tickers_lists/scripts/prune_ocr.py \\
        --industry "Consumer Cyclical / Auto Parts" \\
        --start 2017 --end 2022 \\
        --ocr-dir /data/workspace/charles/subset_auto_parts_2017_2022
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SELECTION = os.path.join(ROOT, "grouped", "selected", "companies.json")
DEFAULT_PDFS_ROOT = (
    "/data/raw_data/argimi_corpuses/annual_reports_pdfs_selected_checked"
)

DIRNAME_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})(?:_[a-f0-9]{8,})?$")
PDF_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})\.pdf$")


def slugify(text: str) -> str:
    text = text.lower().replace(" / ", "_")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def company_years_in_pdfs(pdfs_dir: str) -> dict[tuple[str, str], set[int]]:
    years: dict[tuple[str, str], set[int]] = defaultdict(set)
    for fname in os.listdir(pdfs_dir):
        m = PDF_RE.match(fname)
        if m:
            years[(m.group(1), m.group(2))].add(int(m.group(3)))
    return years


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--industry",
        required=True,
        help='Industry key, e.g. "Consumer Cyclical / Auto Parts"',
    )
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--ocr-dir", required=True)
    p.add_argument("--selection", default=DEFAULT_SELECTION)
    p.add_argument("--pdfs-root", default=DEFAULT_PDFS_ROOT)
    p.add_argument(
        "--execute", action="store_true", help="Actually delete. Default is dry-run."
    )
    args = p.parse_args()

    if not os.path.isdir(args.ocr_dir):
        print(f"OCR dir not found: {args.ocr_dir}", file=sys.stderr)
        return 1

    industry_slug = slugify(args.industry)
    pdfs_dir = os.path.join(args.pdfs_root, industry_slug)
    if not os.path.isdir(pdfs_dir):
        print(f"PDFs dir not found: {pdfs_dir}", file=sys.stderr)
        return 1

    window = set(range(args.start, args.end + 1))

    with open(args.selection) as f:
        selected = json.load(f)
    by_exchange = selected.get(args.industry)
    if not by_exchange:
        print(f"Industry not present in selection: {args.industry}", file=sys.stderr)
        print(f"Available: {sorted(selected)}", file=sys.stderr)
        return 1
    post_filter: set[tuple[str, str]] = {
        (c["exchange"], c["ticker"]) for comps in by_exchange.values() for c in comps
    }

    pdf_years = company_years_in_pdfs(pdfs_dir)
    full_coverage = {k for k, ys in pdf_years.items() if window.issubset(ys)}

    kept_companies = post_filter & full_coverage

    entries = sorted(os.listdir(args.ocr_dir))
    to_keep: list[str] = []
    to_remove: list[str] = []
    unrecognized: list[str] = []

    for name in entries:
        full = os.path.join(args.ocr_dir, name)
        if not os.path.isdir(full):
            unrecognized.append(name)
            continue
        m = DIRNAME_RE.match(name)
        if not m:
            unrecognized.append(name)
            continue
        exch, tkr, year = m.group(1), m.group(2), int(m.group(3))
        if (exch, tkr) in kept_companies and year in window:
            to_keep.append(name)
        else:
            to_remove.append(name)

    # Break down why things are being removed, for diagnostics.
    reason_counts = {"not-in-selection": 0, "partial-coverage": 0, "out-of-window": 0}
    for name in to_remove:
        m = DIRNAME_RE.match(name)
        exch, tkr, year = m.group(1), m.group(2), int(m.group(3))
        key = (exch, tkr)
        if key not in post_filter:
            reason_counts["not-in-selection"] += 1
        elif key not in full_coverage:
            reason_counts["partial-coverage"] += 1
        elif year not in window:
            reason_counts["out-of-window"] += 1

    print(f"Industry:     {args.industry}")
    print(f"Window:       {args.start}-{args.end}")
    print(f"OCR dir:      {args.ocr_dir}")
    print(f"Selection:    {args.selection}")
    print(f"PDFs dir:     {pdfs_dir}")
    print()
    print(f"Post-filter companies in industry:   {len(post_filter)}")
    print(f"With full {args.start}-{args.end} PDF coverage: {len(full_coverage)}")
    print(f"Kept (intersection):                 {len(kept_companies)}")
    print()
    print(
        f"OCR entries: total={len(entries)} "
        f"keep={len(to_keep)} remove={len(to_remove)} "
        f"unrecognized={len(unrecognized)}"
    )
    print(f"  reasons for removal: {reason_counts}")
    print()
    print("Kept companies:")
    for exch, tkr in sorted(kept_companies):
        name_lookup = {
            (c["exchange"], c["ticker"]): c["name"]
            for comps in by_exchange.values()
            for c in comps
        }
        print(f"  {exch:<8} {tkr:<10} {name_lookup.get((exch, tkr), '')}")
    if unrecognized:
        print()
        print(
            f"Unrecognized (left alone): {unrecognized[:10]}"
            + (" ..." if len(unrecognized) > 10 else "")
        )
    print()
    print(f"Sample to remove (first 15 of {len(to_remove)}):")
    for d in to_remove[:15]:
        print(f"  - {d}")

    if not args.execute:
        print("\n(dry run — re-run with --execute to delete)")
        return 0

    for d in to_remove:
        shutil.rmtree(os.path.join(args.ocr_dir, d))
    print(f"\nRemoved {len(to_remove)} directories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
