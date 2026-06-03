"""Select a 10,000-query test subset for the needle-in-a-haystack benchmark.

Filters queries by three criteria:
1. Source report's clean .mmd has < 115,000 tokens (cl100k_base)
2. Source report has high KPI coverage (greedy selection)
3. Each query has at least one page with LLM-graded qrels = 2

Usage:
    uv run python KPI_analysis/llm_benchmark/needle_haystack/select_test_set.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

NEEDLE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = NEEDLE_DIR.parent
KPI_DIR = BENCHMARK_DIR.parent
REPO_ROOT = KPI_DIR.parent

DEFAULT_QUERIES = NEEDLE_DIR / "queries.csv"
DEFAULT_KPIS_LONG = KPI_DIR / "output" / "kpis_long.csv"
DEFAULT_QRELS_LLM = KPI_DIR / "output" / "qrels" / "qrels_llm.txt"
DEFAULT_OCR_ROOT = REPO_ROOT / "DeepSeekOCR_Ardian_pruned_1k"
DEFAULT_OUTPUT_DIR = NEEDLE_DIR

REPORT_NAME_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})(?:_[0-9a-fA-F]+)?$")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_queries(path: Path) -> dict[str, str]:
    """Return {query_id: query_text} from queries.csv."""
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["query_id"]] = row["query_text"]
    return out


def load_kpi_coverage(kpis_long_path: Path) -> dict[tuple[str, int], set[str]]:
    """Return {(ticker, year): set of kpi names} from kpis_long.csv."""
    coverage: dict[tuple[str, int], set[str]] = defaultdict(set)
    with kpis_long_path.open(newline="") as f:
        for row in csv.DictReader(f):
            coverage[(row["ticker"], int(row["year"]))].add(row["kpi"])
    return dict(coverage)


def _query_year(query_id: str) -> str | None:
    """Extract the trailing year from a query_id like 'AAP_accounts_payable_2017'."""
    last_us = query_id.rfind("_")
    if last_us == -1:
        return None
    return query_id[last_us + 1 :]


def _doc_year(doc_id: str) -> str | None:
    """Extract the report year from a doc_id like 'NYSE_AAP_2017/page_0042'."""
    slash = doc_id.find("/")
    if slash == -1:
        return None
    prefix = doc_id[:slash]  # e.g. NYSE_AAP_2017
    last_us = prefix.rfind("_")
    if last_us == -1:
        return None
    return prefix[last_us + 1 :]


def load_grade2_query_ids(qrels_path: Path) -> set[str]:
    """Return set of query_ids that have at least one same-year page with grade 2.

    A query about year X only qualifies if a grade-2 page exists in the year-X
    document (not X+1 or X+2).
    """
    ids: set[str] = set()
    with qrels_path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == "2":
                qid = parts[0]
                doc = parts[2]
                qy = _query_year(qid)
                dy = _doc_year(doc)
                if qy and dy and qy == dy:
                    ids.add(qid)
    return ids


# ---------------------------------------------------------------------------
# Report discovery + token counting
# ---------------------------------------------------------------------------


def parse_report_name(name: str) -> tuple[str, str, int] | None:
    m = REPORT_NAME_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def find_clean_mmd(report_dir: Path) -> Path | None:
    """Pick the clean .mmd (not _det.mmd) from a report directory."""
    preferred = report_dir / f"{report_dir.name}.mmd"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        p for p in report_dir.glob("*.mmd") if not p.name.endswith("_det.mmd")
    )
    if candidates:
        return candidates[0]
    return None


def discover_reports(ocr_root: Path) -> list[tuple[str, str, int, Path]]:
    """Return [(exchange, ticker, year, clean_mmd_path), ...] for all reports."""
    out: list[tuple[str, str, int, Path]] = []
    seen: set[Path] = set()
    for mmd in ocr_root.rglob("*.mmd"):
        d = mmd.parent
        if d in seen:
            continue
        seen.add(d)
    for d in sorted(seen):
        parsed = parse_report_name(d.name)
        if parsed is None:
            continue
        clean = find_clean_mmd(d)
        if clean is None:
            continue
        exchange, ticker, year = parsed
        out.append((exchange, ticker, year, clean))
    return out


def count_tokens(mmd_path: Path, enc: tiktoken.Encoding) -> int:
    """Count tokens in a .mmd file using the given tokenizer."""
    text = mmd_path.read_text(encoding="utf-8", errors="replace")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_test_set(
    queries: dict[str, str],
    kpi_coverage: dict[tuple[str, int], set[str]],
    grade2_ids: set[str],
    reports: list[tuple[str, str, int, Path]],
    max_tokens: int,
    target_size: int,
) -> list[str]:
    """Return list of query_ids for the test set."""
    enc = tiktoken.get_encoding("cl100k_base")

    # Step 1: count tokens and filter
    filtered: list[tuple[str, int, int, Path]] = []  # (ticker, year, n_tokens, path)
    token_counts: list[int] = []
    for exchange, ticker, year, mmd_path in reports:
        n_tokens = count_tokens(mmd_path, enc)
        token_counts.append(n_tokens)
        if n_tokens < max_tokens:
            filtered.append((ticker, year, n_tokens, mmd_path))

    # Print token stats
    token_counts.sort()
    n_total = len(token_counts)
    n_pass = len(filtered)
    print(
        f"Token stats: min={token_counts[0]:,}  max={token_counts[-1]:,}  "
        f"median={token_counts[n_total // 2]:,}  pass(<{max_tokens:,})={n_pass}/{n_total}"
    )

    # Step 2: compute KPI count per surviving report
    report_kpis: list[tuple[str, int, int]] = []  # (ticker, year, kpi_count)
    for ticker, year, n_tokens, _ in filtered:
        kpis = kpi_coverage.get((ticker, year), set())
        report_kpis.append((ticker, year, len(kpis)))

    # Step 3: sort by KPI count descending, tie-break by ticker+year
    report_kpis.sort(key=lambda x: (-x[2], x[0], x[1]))

    # Step 4: group queries by (ticker, year) for fast lookup
    queries_by_ticker_year: dict[tuple[str, int], list[str]] = defaultdict(list)
    for qid in queries:
        # query_id format: {ticker}_{kpi}_{year}
        # ticker may contain dots (e.g., ABDP.L) but not underscores
        # kpi contains underscores, year is 4 digits at the end
        last_underscore = qid.rfind("_")
        if last_underscore == -1:
            continue
        ticker_part = qid[: qid.index("_")]
        year_part = qid[last_underscore + 1 :]
        kpi_part = qid[qid.index("_") + 1 : last_underscore]
        try:
            year = int(year_part)
        except ValueError:
            continue
        queries_by_ticker_year[(ticker_part, year)].append(qid)

    # Step 5: greedy selection
    selected: list[str] = []
    for ticker, year, kpi_count in report_kpis:
        if len(selected) >= target_size:
            break
        # Get all queries for this (ticker, year) that have grade-2 pages
        all_qids = queries_by_ticker_year.get((ticker, year), [])
        eligible = [q for q in all_qids if q in grade2_ids]
        if not eligible:
            continue
        # Sort by KPI name for determinism
        eligible.sort()
        remaining = target_size - len(selected)
        take = eligible[:remaining]
        selected.extend(take)

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--kpis-long", type=Path, default=DEFAULT_KPIS_LONG)
    parser.add_argument("--qrels-llm", type=Path, default=DEFAULT_QRELS_LLM)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--max-tokens", type=int, default=115_000)
    parser.add_argument("--target-size", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print("Loading data...")
    queries = load_queries(args.queries)
    print(f"  queries.csv: {len(queries):,} queries")

    kpi_coverage = load_kpi_coverage(args.kpis_long)
    print(f"  kpis_long.csv: {len(kpi_coverage):,} (ticker, year) pairs")

    grade2_ids = load_grade2_query_ids(args.qrels_llm)
    print(f"  grade-2 query_ids: {len(grade2_ids):,}")

    reports = discover_reports(args.ocr_root)
    print(f"  reports discovered: {len(reports)}")

    # Overlap check
    grade2_in_queries = grade2_ids & set(queries.keys())
    print(f"  grade-2 queries in queries.csv: {len(grade2_in_queries):,}")

    print("\nSelecting test set...")
    selected = select_test_set(
        queries,
        kpi_coverage,
        grade2_in_queries,
        reports,
        args.max_tokens,
        args.target_size,
    )

    # Write output
    out_path = args.output_dir / "test_set.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_text"])
        for qid in selected:
            writer.writerow([qid, queries[qid]])

    print(f"\nWrote {len(selected):,} queries to {out_path}")

    # Summary stats
    tickers_seen = set()
    kpis_seen = set()
    years_seen = set()
    for qid in selected:
        parts = qid.split("_")
        ticker = parts[0]
        year = parts[-1]
        kpi = "_".join(parts[1:-1])
        tickers_seen.add(ticker)
        kpis_seen.add(kpi)
        years_seen.add(year)
    print(
        f"  tickers: {len(tickers_seen)}, KPIs: {len(kpis_seen)}, years: {len(years_seen)}"
    )


if __name__ == "__main__":
    main()
