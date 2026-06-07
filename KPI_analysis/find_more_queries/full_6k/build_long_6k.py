"""Consolidate per-ticker KPI JSONs from all three fetch runs into:

  full_6k/kpis_long.csv           one row per (ticker, year, kpi), with verification cols
  full_6k/verification_report.csv one row per ticker (reference/fetched name + verdict)
  full_6k/summary.md              coverage + verification summary

The three source directories (in priority order — first write wins):

  1. KPI_analysis/output/raw/                    (244 tickers, original selected industries)
  2. KPI_analysis/find_more_queries/output/raw/  (292 tickers, expanded set)
  3. KPI_analysis/find_more_queries/full_6k/output/raw/  (671 tickers, full 6k corpus)

Overlapping (ticker, year, kpi) keys are resolved by source priority: the first
source to write a key wins.  This preserves the dense per-ticker coverage from
the original pipeline while adding breadth from the full-6k fetch.

Usage
-----
    uv run python KPI_analysis/find_more_queries/full_6k/build_long_6k.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FMQ_DIR = HERE.parent
KPI_DIR = FMQ_DIR.parent
sys.path.insert(0, str(KPI_DIR / "kpi_fetch_and_build"))

from tags import KPI_DEFS

sys.path.insert(0, str(HERE))
from scan_and_fetch_6k import verify_identity

# Three raw-source directories, in priority order (first wins on dedup).
RAW_DIRS: list[Path] = [
    KPI_DIR / "output" / "raw",
    FMQ_DIR / "output" / "raw",
    HERE / "output" / "raw",
]
LONG_CSV = HERE / "kpis_long.csv"
FULL_CSV = HERE / "kpi_long_full.csv"
VERIFY_CSV = HERE / "verification_report.csv"
SUMMARY_MD = HERE / "summary.md"

ALL_KPI_KEYS = [k.key for k in KPI_DEFS]
CORE_KPI_KEYS = [
    "revenue", "gross_profit", "operating_income", "net_income",
    "total_assets", "total_liabilities", "cash_and_equivalents",
    "operating_cash_flow", "capex",
]


def _load_one(raw_dir: Path, source_label: str) -> list[dict]:
    """Load all JSON records from *raw_dir*, tagging each with *source_label*."""
    if not raw_dir.exists():
        return []
    records = []
    for p in sorted(raw_dir.glob("*.json")):
        r = json.loads(p.read_text())
        r["_source_dir"] = source_label
        # Normalise exchange_ocr: fall back to exchange for the original pipeline
        # records which predate the exchange_ocr field.
        if not r.get("exchange_ocr"):
            r["exchange_ocr"] = r.get("exchange", "")
        # Recompute the identity verdict from stored names.  For the original
        # pipeline records (no reference_name / fetched_name) this will yield
        # ("no_reference", None) which is correct — they were never verified
        # against an external name.
        verdict, score = verify_identity(r.get("reference_name") or None,
                                         r.get("fetched_name") or None)
        r["verified"], r["name_match_score"] = verdict, score
        records.append(r)
    return records


def load_records() -> tuple[list[dict], list[dict]]:
    """Load and merge records from all three source directories.

    Returns (merged_records, flat_rows) where flat_rows is the deduplicated
    list of (ticker, year, kpi) dicts ready for CSV writing.

    Priority: the first directory to provide a (ticker, year, kpi) key wins.
    """
    seen_keys: set[tuple[str, int, str]] = set()  # (ticker, year, kpi)
    merged: list[dict] = []
    flat_rows: list[dict] = []

    for raw_dir, label in zip(RAW_DIRS, ["original", "find_more_queries", "full_6k"]):
        records = _load_one(raw_dir, label)
        source_rows = 0
        source_dupes = 0
        for r in records:
            tags = r.get("tag_used", {})
            added = 0
            for kpi, by_year in r.get("kpis", {}).items():
                for year, val in by_year.items():
                    key = (r.get("ticker", ""), int(year), kpi)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        added += 1
                        flat_rows.append({
                            "ticker": r.get("ticker", ""),
                            "exchange": r.get("exchange", ""),
                            "exchange_ocr": r.get("exchange_ocr", ""),
                            "company_name": r.get("company_name", ""),
                            "industry": r.get("industry", ""),
                            "source": r.get("source", ""),
                            "verified": r.get("verified", ""),
                            "name_match_score": r.get("name_match_score"),
                            "year": int(year),
                            "kpi": kpi,
                            "value": val,
                            "tag": tags.get(kpi, ""),
                        })
            if added > 0:
                merged.append(r)
                source_rows += added
            else:
                source_dupes += 1
        print(f"  {label}: {len(records)} tickers, "
              f"{source_rows:,} new KPI rows, {source_dupes} fully-duplicate tickers",
              file=sys.stderr)

    return merged, flat_rows


# ---------------------------------------------------------------------------
# kpis_long.csv
# ---------------------------------------------------------------------------

LONG_FIELDS = [
    "ticker", "exchange", "exchange_ocr", "company_name", "industry", "source",
    "verified", "name_match_score", "year", "kpi", "value", "tag",
]


def _flatten_records(records: list[dict]) -> list[dict]:
    """Flatten per-ticker records into one row per (ticker, year, kpi)."""
    rows = []
    for r in records:
        tags = r.get("tag_used", {})
        for kpi, by_year in r.get("kpis", {}).items():
            for year, val in by_year.items():
                rows.append({
                    "ticker": r.get("ticker", ""),
                    "exchange": r.get("exchange", ""),
                    "exchange_ocr": r.get("exchange_ocr", ""),
                    "company_name": r.get("company_name", ""),
                    "industry": r.get("industry", ""),
                    "source": r.get("source", ""),
                    "verified": r.get("verified", ""),
                    "name_match_score": r.get("name_match_score"),
                    "year": int(year),
                    "kpi": kpi,
                    "value": val,
                    "tag": tags.get(kpi, ""),
                })
    return rows


def write_long(records: list[dict], flat_rows: list[dict]) -> tuple[int, int]:
    """Write both the 6k-only kpis_long.csv and the merged kpi_long_full.csv.

    Returns (n_6k, n_full).  Unverified rows (delisted tickers with no API
    metadata) are excluded from both outputs.
    """
    # --- 6k-only (backward compat) ---
    records_6k = [r for r in records if r.get("_source_dir") == "full_6k"]
    rows_6k = [r for r in _flatten_records(records_6k) if r.get("verified") != "unverified"]
    with LONG_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_FIELDS)
        w.writeheader()
        w.writerows(rows_6k)

    # --- merged (all sources, already deduped) ---
    rows_full = [r for r in flat_rows if r.get("verified") != "unverified"]
    with FULL_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_FIELDS)
        w.writeheader()
        w.writerows(rows_full)

    return len(rows_6k), len(rows_full)


# ---------------------------------------------------------------------------
# verification_report.csv
# ---------------------------------------------------------------------------

VERIFY_FIELDS = [
    "ticker", "exchange_ocr", "source", "_source_dir",
    "verified", "name_match_score",
    "reference_source", "reference_name", "fetched_name",
    "kpis_found", "years_fetched", "error",
]


def write_verification(records: list[dict]) -> None:
    with VERIFY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VERIFY_FIELDS)
        w.writeheader()
        for r in sorted(records, key=lambda x: (x.get("verified", ""), x.get("ticker", ""))):
            years_fetch = r.get("years_fetch") or []
            w.writerow({
                "ticker": r.get("ticker", ""),
                "exchange_ocr": r.get("exchange_ocr", ""),
                "source": r.get("source", ""),
                "_source_dir": r.get("_source_dir", ""),
                "verified": r.get("verified", ""),
                "name_match_score": r.get("name_match_score"),
                "reference_source": r.get("reference_source", ""),
                "reference_name": r.get("reference_name", ""),
                "fetched_name": r.get("fetched_name", ""),
                "kpis_found": len([k for k in ALL_KPI_KEYS if r.get("kpis", {}).get(k)]),
                "years_fetched": f"{min(years_fetch)}-{max(years_fetch)}" if years_fetch else "",
                "error": r.get("error", ""),
            })


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fill_counts(records: list[dict], kpi_keys: list[str]) -> tuple[int, int]:
    filled = total = 0
    for r in records:
        years = set(r.get("years_fetch") or r.get("years") or [])
        kpis = r.get("kpis", {})
        for kpi in kpi_keys:
            per_year = kpis.get(kpi, {})
            for y in years:
                total += 1
                if str(y) in per_year or y in per_year:
                    filled += 1
    return filled, total


def potential_queries(records: list[dict]) -> int:
    """(ticker, year, kpi) triples that have both a fetched value and an OCR report."""
    count = 0
    for r in records:
        ocr_years = set(r.get("years_in_ocr", []))
        for per_year in r.get("kpis", {}).values():
            for y in per_year:
                if int(y) in ocr_years:
                    count += 1
    return count


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------


def write_summary(records: list[dict]) -> None:
    total = len(records)
    verdicts = Counter(r.get("verified", "?") for r in records)
    verified_ok = [r for r in records if r.get("verified") == "match"]
    filled_all, total_all = fill_counts(records, ALL_KPI_KEYS)
    filled_ok, total_ok = fill_counts(verified_ok, ALL_KPI_KEYS)

    by_source_dir = Counter(r.get("_source_dir", "?") for r in records)

    L: list[str] = [
        "# Merged KPI Coverage & Identity Verification",
        "",
        "Generated by `build_long_6k.py` from three source directories:",
        "",
        f"- `KPI_analysis/output/raw/` — {by_source_dir.get('original', 0)} tickers (original selected industries)",
        f"- `KPI_analysis/find_more_queries/output/raw/` — {by_source_dir.get('find_more_queries', 0)} tickers (expanded set)",
        f"- `KPI_analysis/find_more_queries/full_6k/output/raw/` — {by_source_dir.get('full_6k', 0)} tickers (full 6k corpus)",
        "",
        "Overlapping (ticker, year, kpi) keys resolved by source priority (original > find_more > full_6k).",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Companies (unique ticker) | {total} |",
        f"| With at least one KPI | {sum(1 for r in records if r.get('kpis'))} |",
        f"| With a fetch error | {sum(1 for r in records if r.get('error'))} |",
        f"| Potential (ticker, year, KPI) triples | {potential_queries(records):,} |",
        f"| KPI×year fill rate (all 31, all rows) | "
        f"{100*filled_all/total_all:.1f}% ({filled_all:,}/{total_all:,}) |"
        if total_all else "| KPI×year fill rate | n/a |",
        f"| KPI×year fill rate (all 31, verified=match only) | "
        f"{100*filled_ok/total_ok:.1f}% ({filled_ok:,}/{total_ok:,}) |"
        if total_ok else "| KPI×year fill rate (verified) | n/a |",
        "",
        "## Identity Verification",
        "",
        "| Verdict | Companies |",
        "|---------|-----------|",
    ]
    for v in ("match", "mismatch", "no_reference", "unverified"):
        L.append(f"| {v} | {verdicts.get(v, 0)} |")
    L += [
        "",
        "- **match** — fetched company name agrees with the reference (cleaned-CSV "
        "name, else yfinance longName).",
        "- **mismatch** — names disagree; likely a cross-exchange ticker collision. "
        "Inspect before using these rows.",
        "- **no_reference / unverified** — no reference name available or the source "
        "returned no name to compare.",
        "",
        "_Note: for cleaned-CSV-referenced tickers this is a document-tied cross-check; "
        "for US tickers verified only against yfinance it is a self-consistency check "
        "(same US ticker on both sides), so it cannot catch a foreign-company collision._",
        "",
    ]

    # mismatches list
    mism = [r for r in records if r.get("verified") == "mismatch"]
    if mism:
        L += ["### Mismatches (review before use)", "",
              "| Ticker | Exch | Score | Reference | Fetched |",
              "|--------|------|-------|-----------|---------|"]
        for r in sorted(mism, key=lambda x: x.get("name_match_score") or 0):
            L.append(f"| {r.get('ticker','')} | {r.get('exchange_ocr','')} | "
                     f"{r.get('name_match_score')} | {r.get('reference_name','')} | "
                     f"{r.get('fetched_name','')} |")
        L.append("")

    # by exchange
    L += ["## By Exchange (OCR label)", "",
          "| Exchange | Companies | No Error | match | Potential Queries |",
          "|----------|-----------|----------|-------|-------------------|"]
    by_ex: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_ex[r.get("exchange_ocr", "?")].append(r)
    for ex in sorted(by_ex):
        g = by_ex[ex]
        ok = sum(1 for r in g if not r.get("error"))
        mt = sum(1 for r in g if r.get("verified") == "match")
        L.append(f"| {ex} | {len(g)} | {ok} | {mt} | {potential_queries(g):,} |")
    L.append("")

    # by source
    L += ["## By Source", "",
          "| Source | Companies |", "|--------|-----------|"]
    for src, n in sorted(Counter(r.get("source", "?") for r in records).items()):
        L.append(f"| {src} | {n} |")
    L.append("")

    # per-KPI (all 31), verified=match rows only
    L += ["## Per-KPI Coverage (all 31, verified=match rows, years in fetch range)", "",
          "| KPI | Companies with Data | Fill Rate |",
          "|-----|---------------------|-----------|"]
    for kpi in ALL_KPI_KEYS:
        cw = sum(1 for r in verified_ok if r.get("kpis", {}).get(kpi))
        f1, t1 = fill_counts(verified_ok, [kpi])
        rate = f"{100*f1/t1:.1f}%" if t1 else "n/a"
        L.append(f"| {kpi} | {cw}/{len(verified_ok)} | {rate} |")
    L.append("")

    SUMMARY_MD.write_text("\n".join(L))


def main() -> int:
    records, flat_rows = load_records()
    if not records:
        print(f"No JSON files in any source directory", file=sys.stderr)
        return 1
    n_6k, n_full = write_long(records, flat_rows)
    write_verification(records)
    write_summary(records)
    print(f"\nLoaded {len(records)} ticker records (merged from 3 sources).",
          file=sys.stderr)
    print(f"  {n_6k:,} rows -> {LONG_CSV}  (full_6k only)", file=sys.stderr)
    print(f"  {n_full:,} rows -> {FULL_CSV}  (merged)", file=sys.stderr)
    print(f"  verification report -> {VERIFY_CSV}", file=sys.stderr)
    print(f"  summary -> {SUMMARY_MD}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
