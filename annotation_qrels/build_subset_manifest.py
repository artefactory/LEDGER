"""Build a random subset manifest for qrels annotation.

Samples N items from review_candidates.csv, builds the full manifest
using the same logic as qrels_index.py, and writes it as a standalone
JSON file that can be loaded via ``--manifest-path``.

Usage::

    uv run python annotation_qrels/build_subset_manifest.py \
        --n 300 --seed 42 \
        --output annotation_qrels/manifests/subset_300.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from qrels_index import (
    DEFAULT_CANDIDATES_CSV,
    DEFAULT_KPIS_CSV,
    DEFAULT_OCR_ROOT,
    DEFAULT_RAW_ROOT,
    build_queue,
)

HERE = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a random subset manifest for qrels annotation."
    )
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--kpis-csv", type=Path, default=DEFAULT_KPIS_CSV)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--n", type=int, default=300, help="Number of items to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "manifests" / "subset_300.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--stratify",
        choices=["none", "match_type", "kpi"],
        default="match_type",
        help="Stratify the sample by match_type or kpi. Default: match_type.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Read all candidate rows
    with args.candidates_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Total candidates in CSV: {len(all_rows)}")

    # Sample
    rng = random.Random(args.seed)

    if args.stratify == "none":
        sampled = rng.sample(all_rows, min(args.n, len(all_rows)))
    else:
        group_key = args.stratify  # "match_type" or "kpi" (derived from query_id)
        groups: dict[str, list[dict]] = {}
        for row in all_rows:
            if group_key == "match_type":
                key = row["match_type"]
            else:
                # Derive kpi from query_id: TICKER_kpi_name_year
                qid = row["query_id"]
                parts = qid.rsplit("_", 1)
                first_us = qid.index("_")
                key = qid[first_us + 1 : qid.rindex("_")]
            groups.setdefault(key, []).append(row)

        # Proportional allocation
        sampled = []
        remaining = args.n
        sorted_keys = sorted(groups.keys())
        for i, key in enumerate(sorted_keys):
            group = groups[key]
            if i == len(sorted_keys) - 1:
                alloc = remaining
            else:
                alloc = max(1, round(args.n * len(group) / len(all_rows)))
                alloc = min(alloc, len(group), remaining)
            sampled.extend(rng.sample(group, alloc))
            remaining -= alloc

        # If we undershoot due to rounding, fill from largest groups
        if remaining > 0:
            already = {id(r) for r in sampled}
            pool = [r for r in all_rows if id(r) not in already]
            extra = rng.sample(pool, min(remaining, len(pool)))
            sampled.extend(extra)

        # If we overshoot, trim
        if len(sampled) > args.n:
            sampled = rng.sample(sampled, args.n)

    print(f"Sampled {len(sampled)} items (stratify={args.stratify}, seed={args.seed})")

    # Write sampled rows to a temporary CSV so we can reuse build_queue
    tmp_csv = args.output.parent / f".tmp_subset_{args.seed}.csv"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(all_rows[0].keys())
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    # Build the full manifest using the existing queue builder
    items, summary = build_queue(
        candidates_csv=tmp_csv,
        kpis_csv=args.kpis_csv,
        ocr_root=args.ocr_root,
        raw_root=args.raw_root,
    )
    tmp_csv.unlink(missing_ok=True)

    # Add subset metadata
    summary["subset_seed"] = args.seed
    summary["subset_n"] = args.n
    summary["subset_stratify"] = args.stratify
    summary["subset_sampled_from_csv"] = str(args.candidates_csv)

    payload = {"summary": summary, "items": items}
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nManifest written to {args.output}")
    print(f"  Items in manifest: {len(items)}")
    print(f"  Missing .mmd: {summary.get('missing_mmd', 0)}")
    print(f"  Missing raw PNG: {summary.get('missing_raw_png', 0)}")

    # Print stratification stats
    from collections import Counter

    match_counts = Counter(item["match_type"] for item in items)
    print(f"\n  Match type distribution:")
    for mt, cnt in match_counts.most_common():
        print(f"    {mt}: {cnt}")

    kpi_counts = Counter(item["kpi"] for item in items)
    print(f"\n  KPI distribution (top 10):")
    for kpi, cnt in kpi_counts.most_common(10):
        print(f"    {kpi}: {cnt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
