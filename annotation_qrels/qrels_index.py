"""Build annotation queues from qrels review_candidates.csv."""

from __future__ import annotations

import csv
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DEFAULT_CANDIDATES_CSV = PROJECT_ROOT / "KPI_analysis" / "output" / "qrels" / "review_candidates.csv"
DEFAULT_KPIS_CSV = PROJECT_ROOT / "KPI_analysis" / "output" / "kpis_long.csv"
DEFAULT_OCR_ROOT = PROJECT_ROOT / "DeepSeekOCR_Ardian_pruned_1k"
DEFAULT_RAW_ROOT = Path("/data/workspace/charles/pdf_ocr_deepseek/DeepSeekOCR_Ardian_raw_3kdocs")

PAGE_SPLIT_RE = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)
REPORT_DIR_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})(?:_[0-9a-fA-F]+)?$")


def format_target_value(value: float) -> str:
    """Format a dollar value for display: $1.2B, $456.7M, $12.3K, or raw."""
    if value == 0:
        return "$0"
    sign = ""
    if value < 0:
        sign = "-"
        value = abs(value)
    if value >= 1_000_000_000:
        formatted = f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        formatted = f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        formatted = f"{value / 1_000:.1f}K"
    else:
        formatted = f"{value:.0f}"
    return f"{sign}${formatted}"


def load_kpis_long(kpis_csv: Path) -> dict[str, float]:
    """Load kpis_long.csv and return {(ticker, kpi, year): value}."""
    mapping: dict[str, float] = {}
    with kpis_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['ticker']}_{row['kpi']}_{row['year']}"
            mapping[key] = float(row["value"])
    return mapping


def resolve_report_dir(report_name: str, ocr_root: Path) -> Path | None:
    """Find the report directory under ocr_root, handling hash suffixes."""
    # Try exact match first under any subdirectory
    for mmd in ocr_root.rglob(f"{report_name}/{report_name}.mmd"):
        return mmd.parent
    # Try stripping hash suffix from found dirs
    for mmd in ocr_root.rglob("*.mmd"):
        parent = mmd.parent
        if parent.name.startswith(report_name) and REPORT_DIR_RE.match(parent.name):
            return parent
    return None


def find_raw_dir(report_name: str, raw_root: Path) -> Path | None:
    """Find the report directory under raw_root for page PNGs."""
    # Try exact match
    for candidate in raw_root.rglob(f"{report_name}"):
        if candidate.is_dir():
            return candidate
    # Try without hash suffix
    for pages_dir in raw_root.rglob("pages"):
        parent = pages_dir.parent
        if parent.name.startswith(report_name) and REPORT_DIR_RE.match(parent.name):
            return parent
    return None


@lru_cache(maxsize=2048)
def load_pages(mmd_path: str) -> tuple[str, ...]:
    """Split an .mmd file into per-page text segments."""
    text = Path(mmd_path).read_text(encoding="utf-8")
    return tuple(PAGE_SPLIT_RE.split(text))


def build_queue(
    candidates_csv: Path,
    kpis_csv: Path,
    ocr_root: Path,
    raw_root: Path,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read review_candidates.csv and build a manifest queue.

    Returns (items, summary).
    """
    kpi_values = load_kpis_long(kpis_csv)
    items: list[dict[str, Any]] = []
    missing_mmd = 0
    missing_raw = 0
    report_dir_cache: dict[str, Path | None] = {}
    raw_dir_cache: dict[str, Path | None] = {}

    with candidates_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit is not None:
        rows = rows[:limit]

    for row in rows:
        query_id = row["query_id"]
        doc_id = row["doc_id"]
        report_name = row["report_name"]
        page_idx = int(row["page_idx"])

        # Resolve report directory
        if report_name not in report_dir_cache:
            report_dir_cache[report_name] = resolve_report_dir(report_name, ocr_root)
        report_dir = report_dir_cache[report_name]

        if report_dir is None:
            missing_mmd += 1
            continue

        mmd_candidates = list(report_dir.glob("*.mmd"))
        mmd_path = None
        for p in mmd_candidates:
            if p.stem == report_name or p.stem == report_name + "_det":
                continue
            if "_det" not in p.stem:
                mmd_path = p
                break
        if mmd_path is None and mmd_candidates:
            # Fallback: pick any non-det .mmd
            for p in mmd_candidates:
                if "_det" not in p.stem:
                    mmd_path = p
                    break
        if mmd_path is None:
            missing_mmd += 1
            continue

        # Resolve raw page PNG
        if report_name not in raw_dir_cache:
            raw_dir_cache[report_name] = find_raw_dir(report_name, raw_root)
        raw_dir = raw_dir_cache[report_name]

        raw_png_path: str | None = None
        if raw_dir is not None:
            pages_dir = raw_dir / "pages"
            if pages_dir.is_dir():
                # Try exact filename first
                exact = pages_dir / f"page_{page_idx:04d}.png"
                if exact.is_file():
                    raw_png_path = str(exact)
                else:
                    # Fallback: positional indexing
                    pngs = sorted(pages_dir.glob("page_*.png"))
                    if 0 <= page_idx < len(pngs):
                        raw_png_path = str(pngs[page_idx])
        if raw_png_path is None:
            missing_raw += 1

        # Load page text for sha256
        pages = load_pages(str(mmd_path))
        page_text = pages[page_idx] if 0 <= page_idx < len(pages) else ""
        page_text_sha256 = hashlib.sha256(page_text.encode("utf-8")).hexdigest()

        # Parse query_id → ticker, kpi, year
        # query_id format: TICKER_kpi_name_year (e.g. AAP_net_income_2019)
        parts = query_id.rsplit("_", 1)
        year = int(parts[1])

        # Parse exchange and ticker from report_name (e.g. NYSE_AAP_2019)
        report_parts = REPORT_DIR_RE.match(report_name)
        exchange = report_parts.group(1) if report_parts else ""
        ticker_from_report = report_parts.group(2) if report_parts else ""

        # Look up ground-truth target value
        target_value = kpi_values.get(query_id)

        # Derive kpi name: remove ticker prefix and year suffix from query_id
        ticker_clean = ticker_from_report
        if ticker_clean and query_id.startswith(ticker_clean + "_") and query_id.endswith(f"_{year}"):
            kpi_name = query_id[len(ticker_clean) + 1 : -len(f"_{year}")]
        else:
            first_us = query_id.index("_")
            last_us = query_id.rindex("_")
            ticker_clean = query_id[:first_us]
            kpi_name = query_id[first_us + 1 : last_us]

        # Derive industry slug from report_dir parent
        industry_slug = report_dir.parent.name if report_dir.parent != ocr_root else ""

        item_id = f"{query_id}__{doc_id.replace('/', '_')}"

        items.append(
            {
                "item_id": item_id,
                "query_id": query_id,
                "doc_id": doc_id,
                "report_name": report_name,
                "report_year": int(row["report_year"]),
                "page_idx": page_idx,
                "kpi": kpi_name,
                "ticker": ticker_clean,
                "year": year,
                "exchange": exchange,
                "target_value": target_value,
                "target_value_display": format_target_value(target_value) if target_value is not None else "N/A",
                "match_type": row["match_type"],
                "alias_matched": row.get("alias_matched", ""),
                "raw_value": row.get("raw_value", ""),
                "normalized_value": float(row["normalized_value"]) if row.get("normalized_value") else None,
                "rel_error": float(row["rel_error"]) if row.get("rel_error") else None,
                "unit_source": row.get("unit_source", ""),
                "snippet": row.get("snippet", ""),
                "mmd_path": str(mmd_path),
                "raw_png_path": raw_png_path,
                "raw_root": str(raw_root),
                "page_text_chars": len(page_text),
                "page_text_sha256": page_text_sha256,
                "industry_slug": industry_slug,
            }
        )

    summary = {
        "candidates_csv": str(candidates_csv),
        "kpis_csv": str(kpis_csv),
        "ocr_root": str(ocr_root),
        "raw_root": str(raw_root),
        "total_rows_in_csv": len(rows),
        "items_built": len(items),
        "missing_mmd": missing_mmd,
        "missing_raw_png": missing_raw,
    }
    return items, summary


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build qrels annotation queue (dry run).")
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--kpis-csv", type=Path, default=DEFAULT_KPIS_CSV)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Write manifest JSON to file.")
    args = parser.parse_args()

    items, summary = build_queue(
        candidates_csv=args.candidates_csv,
        kpis_csv=args.kpis_csv,
        ocr_root=args.ocr_root,
        raw_root=args.raw_root,
        limit=args.limit,
    )

    print(f"Built {len(items)} items from {summary['total_rows_in_csv']} CSV rows")
    print(f"  Missing .mmd: {summary['missing_mmd']}")
    print(f"  Missing raw PNG: {summary['missing_raw_png']}")

    if items:
        print("\nSample item:")
        sample = items[0]
        for k, v in sample.items():
            print(f"  {k}: {v}")

    if args.output:
        payload = {"summary": summary, "items": items}
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nManifest written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
