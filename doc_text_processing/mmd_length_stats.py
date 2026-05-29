"""Compute token-length and page-count statistics for *.mmd files in the pruned 1k OCR tree.

Uses the cl100k_base tokenizer (GPT-4 / ChatGPT family).
Pages are counted via the ``<--- Page Split --->`` marker in each .mmd file.

Usage:
    uv run python doc_text_processing/mmd_length_stats.py [--root PATH]
"""

import argparse
import json
import statistics
from pathlib import Path

import tiktoken

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "DeepSeekOCR_Ardian_pruned_1k"
PAGE_SPLIT = "<--- Page Split --->"


def collect_mmd_files(root: Path) -> list[Path]:
    """Return sorted list of *.mmd (excluding *_det.mmd) under *root*."""
    return sorted(
        p for p in root.rglob("*.mmd") if not p.stem.endswith("_det")
    )


def summarise(values: list[int], label: str) -> dict:
    n = len(values)
    if n == 0:
        return {}
    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if n > 1 else 0.0
    mn, mx = min(values), max(values)
    quantiles = statistics.quantiles(values, n=10) if n >= 10 else []
    return {
        f"{label}_mean": round(mean, 1),
        f"{label}_median": median,
        f"{label}_stdev": round(stdev, 1),
        f"{label}_min": mn,
        f"{label}_max": mx,
        f"{label}_p10": quantiles[0] if quantiles else mn,
        f"{label}_p90": quantiles[8] if quantiles else mx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root of the pruned OCR tree (default: DeepSeekOCR_Ardian_pruned_1k/)",
    )
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    files = collect_mmd_files(args.root)
    if not files:
        print(f"No *.mmd files found under {args.root}")
        return

    per_file: list[dict] = []
    token_counts: list[int] = []
    page_counts: list[int] = []

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        n_tokens = len(enc.encode(text))
        n_pages = text.count(PAGE_SPLIT) + 1
        token_counts.append(n_tokens)
        page_counts.append(n_pages)
        per_file.append({
            "file": str(f.relative_to(args.root)),
            "tokens": n_tokens,
            "pages": n_pages,
        })

    n = len(files)
    stats = {
        "root": str(args.root),
        "file_count": n,
        "total_tokens": sum(token_counts),
        "total_pages": sum(page_counts),
        **summarise(token_counts, "tokens"),
        **summarise(page_counts, "pages"),
    }

    print(json.dumps(stats, indent=2))

    out_path = Path(__file__).with_name("mmd_length_stats.json")
    out_path.write_text(
        json.dumps({"summary": stats, "files": per_file}, indent=2),
        encoding="utf-8",
    )
    print(f"\nPer-file details written to {out_path}")


if __name__ == "__main__":
    main()
