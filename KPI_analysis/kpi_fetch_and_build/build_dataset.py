"""Consolidate per-ticker JSONs into long/wide CSVs + a coverage report.

Reads KPI_analysis/output/raw/*.json (produced by fetch_kpis.py) and writes:
  - output/kpis_long.csv   : one row per (ticker, year, kpi)
  - output/kpis_wide.csv   : one row per (ticker, year), columns = KPI keys
  - output/coverage.md     : per-KPI coverage % per year, plus ticker-level summary

Usage:
  uv run python -m KPI_analysis.kpi_fetch_and_build.build_dataset
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from .tags import KPI_DEFS
except ImportError:
    from tags import KPI_DEFS

HERE = Path(__file__).resolve().parent
KPI_ROOT = HERE.parent
RAW_DIR = KPI_ROOT / "output" / "raw"
OUT_DIR = KPI_ROOT / "output"
LONG_CSV = OUT_DIR / "kpis_long.csv"
WIDE_CSV = OUT_DIR / "kpis_wide.csv"
COVERAGE_MD = OUT_DIR / "coverage.md"


def load_records() -> list[dict]:
    if not RAW_DIR.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(RAW_DIR.glob("*.json"))]


def write_long(records: list[dict]) -> int:
    rows = []
    for rec in records:
        for kpi, by_year in rec.get("kpis", {}).items():
            tag = rec.get("tag_used", {}).get(kpi, "")
            for year, val in by_year.items():
                rows.append(
                    {
                        "ticker": rec["ticker"],
                        "company_name": rec.get("company_name", ""),
                        "exchange": rec.get("exchange", ""),
                        "industry": rec.get("industry", ""),
                        "source": rec.get("source", ""),
                        "year": int(year),
                        "kpi": kpi,
                        "value": val,
                        "tag": tag,
                    }
                )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LONG_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ticker",
                "company_name",
                "exchange",
                "industry",
                "source",
                "year",
                "kpi",
                "value",
                "tag",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_wide(records: list[dict]) -> int:
    kpi_keys = [k.key for k in KPI_DEFS]
    # (ticker, year) -> {kpi: value}
    cells: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    meta: dict[str, dict[str, str]] = {}
    for rec in records:
        meta[rec["ticker"]] = {
            "company_name": rec.get("company_name", ""),
            "exchange": rec.get("exchange", ""),
            "industry": rec.get("industry", ""),
            "source": rec.get("source", ""),
        }
        for kpi, by_year in rec.get("kpis", {}).items():
            for year, val in by_year.items():
                cells[(rec["ticker"], int(year))][kpi] = val
    fieldnames = [
        "ticker",
        "year",
        "company_name",
        "exchange",
        "industry",
        "source",
        *kpi_keys,
    ]
    with WIDE_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (ticker, year), kpis in sorted(cells.items()):
            row = {
                "ticker": ticker,
                "year": year,
                **meta.get(ticker, {}),
                **{k: kpis.get(k, "") for k in kpi_keys},
            }
            w.writerow(row)
    return len(cells)


def write_coverage(records: list[dict]) -> None:
    years_seen: set[int] = set()
    per_kpi_year: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    per_ticker: dict[str, int] = {}
    errors: list[tuple[str, str]] = []
    source_counts: dict[str, int] = defaultdict(int)

    for rec in records:
        source_counts[rec.get("source", "unknown")] += 1
        if rec.get("error"):
            errors.append((rec["ticker"], rec["error"]))
        total = 0
        for kpi, by_year in rec.get("kpis", {}).items():
            for year in by_year:
                y = int(year)
                years_seen.add(y)
                per_kpi_year[kpi][y] += 1
                total += 1
        per_ticker[rec["ticker"]] = total

    years_sorted = sorted(years_seen)
    n_tickers = len(records)
    lines: list[str] = []
    lines.append("# KPI coverage\n")
    lines.append(f"- Tickers processed: **{n_tickers}**")
    for src, n in sorted(source_counts.items()):
        lines.append(f"  - {src}: {n}")
    lines.append(f"- Errors: **{len(errors)}**\n")

    lines.append("## Per-KPI coverage (tickers with data for each year)\n")
    header = "| KPI | " + " | ".join(str(y) for y in years_sorted) + " |"
    sep = "| --- |" + "|".join(["---"] * len(years_sorted)) + "|"
    lines.append(header)
    lines.append(sep)
    for kpi in KPI_DEFS:
        counts = per_kpi_year.get(kpi.key, {})
        cells = " | ".join(
            f"{counts.get(y, 0)}/{n_tickers}" for y in years_sorted
        )
        lines.append(f"| `{kpi.key}` | {cells} |")

    if errors:
        lines.append("\n## Errors\n")
        for t, e in errors[:50]:
            lines.append(f"- `{t}`: {e}")
        if len(errors) > 50:
            lines.append(f"- ... and {len(errors) - 50} more")

    COVERAGE_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    records = load_records()
    if not records:
        print(f"No records found in {RAW_DIR}. Run fetch_kpis.py first.")
        return 1
    n_long = write_long(records)
    n_wide = write_wide(records)
    write_coverage(records)
    print(
        f"Wrote {LONG_CSV.name} ({n_long} rows), {WIDE_CSV.name} ({n_wide} rows), "
        f"{COVERAGE_MD.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
