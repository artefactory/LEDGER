"""Copy annual-report PDFs for companies in the selected industries.

Reads `grouped/selected/companies.json` (produced by
`list_selected_industries.py`) and copies matching PDFs from the raw-data
tree into a destination directory organized by industry:

    dest/
    ├── basic-materials_specialty-chemicals/
    │   ├── NYSE_APD_2020.pdf
    │   └── ...
    └── consumer-cyclical_auto-parts/
        └── ...

The source layout is expected to be `{source}/{EXCHANGE}/{EXCHANGE}_{TICKER}_{YEAR}.pdf`.

Copies are skipped if the destination file already exists, so re-runs are
idempotent and incremental.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SELECTION = os.path.join(ROOT, "grouped", "selected", "companies.json")
DEFAULT_SOURCE = "/data/raw_data/argimi_corpuses/annual_reports_pdfs_deduplicated"
DEFAULT_DEST = "/data/raw_data/argimi_corpuses/annual_reports_pdfs_selected"


def slugify(text: str) -> str:
    text = text.lower()
    text = text.replace(" / ", "_")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def find_pdfs(source: str, exchange: str, ticker: str) -> list[str]:
    # Ticker can contain dots (e.g. `ABDP.L`) which glob treats literally; fine.
    # Escape `[` since glob treats it as a character class.
    safe_ticker = glob.escape(ticker)
    pattern = os.path.join(source, exchange, f"{exchange}_{safe_ticker}_*.pdf")
    return sorted(glob.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Root of the per-exchange PDF tree.")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help="Destination directory (industry subdirs created).")
    parser.add_argument("--selection", default=DEFAULT_SELECTION,
                        help="Path to selection JSON.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only; do not copy any files.")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"Source directory not found: {args.source}", file=sys.stderr)
        return 1

    with open(args.selection) as f:
        selection = json.load(f)

    # Collect work items first so we can show a plan before copying.
    work: list[tuple[str, str, str]] = []  # (src_path, industry_slug, dest_path)
    missing: list[tuple[str, str, str]] = []  # (industry, exchange, ticker)
    per_industry: dict[str, list[str]] = {}  # slug -> [src paths]

    for industry_key, by_exchange in selection.items():
        slug = slugify(industry_key)
        per_industry.setdefault(slug, [])
        for exchange, companies in by_exchange.items():
            for c in companies:
                pdfs = find_pdfs(args.source, exchange, c["ticker"])
                if not pdfs:
                    missing.append((industry_key, exchange, c["ticker"]))
                    continue
                for src in pdfs:
                    dst = os.path.join(args.dest, slug, os.path.basename(src))
                    work.append((src, slug, dst))
                    per_industry[slug].append(src)

    # Plan summary.
    total_bytes = 0
    for src, _, _ in work:
        try:
            total_bytes += os.path.getsize(src)
        except OSError:
            pass

    print(f"Source:      {args.source}")
    print(f"Destination: {args.dest}")
    print(f"Selection:   {args.selection}")
    print(f"Mode:        {'DRY RUN' if args.dry_run else 'COPY'}")
    print()
    print(f"{'Industry':<50} {'PDFs':>6}")
    print("-" * 58)
    for slug in sorted(per_industry):
        print(f"{slug:<50} {len(per_industry[slug]):>6}")
    print("-" * 58)
    print(f"{'TOTAL':<50} {len(work):>6}   ({human_size(total_bytes)})")
    print()
    if missing:
        print(f"Tickers with no matching PDFs: {len(missing)}")
        for ind, ex, tk in missing[:10]:
            print(f"  {ex}/{tk}  ({ind})")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print()

    if args.dry_run or not work:
        return 0

    copied = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    for src, slug, dst in work:
        if os.path.exists(dst):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except OSError as e:
            errors.append((src, str(e)))
        if (copied + skipped) % 200 == 0:
            print(f"  ...progress: {copied} copied, {skipped} skipped")

    print()
    print(f"Copied:  {copied}")
    print(f"Skipped: {skipped} (already present)")
    if errors:
        print(f"Errors:  {len(errors)}")
        for src, err in errors[:5]:
            print(f"  {src}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
