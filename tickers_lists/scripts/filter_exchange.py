"""Drop rows whose Yahoo-reported exchange doesn't match the expected one.

Reads `cleaned/{EXCHANGE}_mapped_clean_verified.csv` (produced by
`verify_exchange.py`) and writes `cleaned/{EXCHANGE}_mapped_clean.csv`,
keeping only rows whose `Exchange (Yahoo)` value is in the expected set
for that exchange.

The original cleaned CSV is overwritten. The verified CSV is left intact,
so the filter can be re-run with different rules.

Usage:
    uv run python tickers_lists/scripts/filter_exchange.py LSE
    uv run python tickers_lists/scripts/filter_exchange.py LSE --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(ROOT, "cleaned")

# Our exchange label -> set of Yahoo fullExchangeName / exchange strings we
# consider valid. Derived from the LSE verification pass; extend as needed.
EXPECTED: dict[str, set[str]] = {
    "NYSE": {"NYSE", "NYQ"},
    "NASDAQ": {"NasdaqGS", "NasdaqGM", "NasdaqCM", "NMS", "NGM", "NCM"},
    "AMEX": {"NYSE American", "NYSE AMEX", "ASE", "AMEX"},
    "LSE": {"LSE"},
    "AIM": {"AIM", "LSE"},  # Yahoo often lumps AIM into LSE
    "ASX": {"ASX"},
    "TSX": {"Toronto", "TSX"},
    "TSX-V": {"Toronto Venture", "TSXV"},
    "OTC": {
        "OTC Markets OTCPK",
        "OTC Markets OTCQB",
        "OTC Markets OTCQX",
        "PNK",
        "OTCBB",
    },
}


def filter_exchange(exchange: str, dry_run: bool) -> int:
    verified_path = os.path.join(
        CLEANED_DIR, f"{exchange}_mapped_clean_verified.csv"
    )
    output_path = os.path.join(CLEANED_DIR, f"{exchange}_mapped_clean.csv")

    if not os.path.exists(verified_path):
        print(
            f"No verified file: {verified_path}\n"
            f"Run verify_exchange.py {exchange} first.",
            file=sys.stderr,
        )
        return 1

    expected = EXPECTED.get(exchange)
    if expected is None:
        print(
            f"No expected Yahoo labels configured for '{exchange}'. "
            f"Configured: {sorted(EXPECTED)}",
            file=sys.stderr,
        )
        return 1

    with open(verified_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows or "Exchange (Yahoo)" not in rows[0]:
        print(
            "Verified file is missing the 'Exchange (Yahoo)' column.",
            file=sys.stderr,
        )
        return 1

    kept: list[dict] = []
    dropped: dict[str, int] = {}
    for row in rows:
        yahoo = row["Exchange (Yahoo)"]
        if yahoo in expected:
            kept.append(row)
        else:
            dropped[yahoo] = dropped.get(yahoo, 0) + 1

    print(f"{exchange}: expected Yahoo labels = {sorted(expected)}")
    print(f"  Total:   {len(rows)}")
    print(f"  Kept:    {len(kept)}")
    print(f"  Dropped: {sum(dropped.values())}")
    if dropped:
        for label, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>4}  {label}")

    if dry_run:
        print("(dry run, no file written)")
        return 0

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept:
            writer.writerow(row)

    print(f"  Wrote {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exchange")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show kept/dropped counts without writing the output file.",
    )
    args = parser.parse_args()
    sys.exit(filter_exchange(args.exchange, args.dry_run))
