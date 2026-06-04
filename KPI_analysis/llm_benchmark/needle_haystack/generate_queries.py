"""Generate query list for needle-in-a-haystack LLM benchmark.

For each (ticker, year, KPI) triple with ground-truth in ``kpis_long.csv``,
picks one random query template and one random company alias, substitutes
placeholders, and writes a CSV of (query_id, query_text) pairs.

Usage:
    uv run python KPI_analysis/llm_benchmark/needle_haystack/generate_queries.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

NEEDLE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = NEEDLE_DIR.parent
KPI_DIR = BENCHMARK_DIR.parent
REPO_ROOT = KPI_DIR.parent

DEFAULT_KPIS_LONG = KPI_DIR / "output" / "kpis_long.csv"
DEFAULT_ALT_NAMES = REPO_ROOT / "tickers_lists" / "companies_alt_names.json"
DEFAULT_QUERIES_DIR = KPI_DIR / "retrieval_bench" / "queries"
DEFAULT_OCR_ROOT = REPO_ROOT / "DeepSeekOCR_Ardian_pruned_1k"
DEFAULT_OUTPUT_DIR = NEEDLE_DIR

REPORT_NAME_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})(?:_[0-9a-fA-F]+)?$")

QUERY_FILE_TO_KPI: dict[str, str] = {
    "Accounts payable queries.json": "accounts_payable",
    "Accounts receivable queries.json": "accounts_receivable",
    "Capital expenditure queries.json": "capex",
    "Cash & equivalents (unrestricted) queries.json": "cash_and_equivalents",
    "Cash, equivalents & restricted cash queries.json": "cash_incl_restricted",
    "Cost of revenue queries.json": "cost_of_revenue",
    "Current portion of long-term debt queries.json": "long_term_debt_current",
    "Depreciation & amortization queries.json": "depreciation_amortization",
    "Dividends paid queries.json": "dividends_paid",
    "EPS (basic) queries.json": "eps_basic",
    "EPS (diluted) queries.json": "eps_diluted",
    "Financing cash flow queries.json": "financing_cash_flow",
    "Gross profit queries.json": "gross_profit",
    "Income tax expense queries.json": "income_tax_expense",
    "Interest expense queries.json": "interest_expense",
    "Inventory queries.json": "inventory",
    "Investing cash flow queries.json": "investing_cash_flow",
    "Long-term debt (incl. current portion) queries.json": "long_term_debt_total",
    "Long-term debt (noncurrent portion only) queries.json": "long_term_debt_noncurrent",
    "Net income (attributable to parent) queries.json": "net_income",
    "Operating cash flow queries.json": "operating_cash_flow",
    "Operating income queries.json": "operating_income",
    "R&D expense queries.json": "rd_expense",
    "Revenue queries.json": "revenue",
    "SG&A expense queries.json": "sga_expense",
    "Shares outstanding queries.json": "shares_outstanding",
    "Short-term borrowings queries.json": "short_term_borrowings",
    "Stockholders' equity (attributable to parent) queries.json": "stockholders_equity",
    "Stockholders' equity (incl. non-controlling interest) queries.json": (
        "stockholders_equity_incl_nci"
    ),
    "Total assets queries.json": "total_assets",
    "Total liabilities queries.json": "total_liabilities",
}


def load_query_templates(queries_dir: Path) -> dict[str, list[str]]:
    templates: dict[str, list[str]] = {}
    for fname, kpi_key in QUERY_FILE_TO_KPI.items():
        path = queries_dir / fname
        if not path.is_file():
            continue
        templates[kpi_key] = json.loads(path.read_text())
    return templates


def load_alt_names(json_path: Path) -> dict[str, list[str]]:
    return json.loads(json_path.read_text())


def load_ground_truth(kpis_long_path: Path) -> dict[tuple[str, str, int], str]:
    """Read kpis_long.csv and return {(ticker, kpi, year): company_name}."""
    gt: dict[tuple[str, str, int], str] = {}
    with kpis_long_path.open(newline="") as f:
        for row in csv.DictReader(f):
            gt[(row["ticker"], row["kpi"], int(row["year"]))] = row["company_name"]
    return gt


def discover_tickers(ocr_root: Path) -> set[str]:
    """Scan OCR root for report dirs and return the set of tickers present."""
    tickers: set[str] = set()
    for mmd in ocr_root.rglob("*.mmd"):
        m = REPORT_NAME_RE.match(mmd.parent.name)
        if m:
            tickers.add(m.group(2))
    return tickers


def generate_queries(
    ground_truth: dict[tuple[str, str, int], str],
    alt_names: dict[str, list[str]],
    templates: dict[str, list[str]],
    seed: int = 42,
) -> tuple[list[tuple[str, str]], set[str]]:
    rng = random.Random(seed)
    queries: list[tuple[str, str]] = []
    covered_kpis: set[str] = set()
    for (ticker, kpi, year), company_name in sorted(ground_truth.items()):
        kpi_templates = templates.get(kpi)
        if not kpi_templates:
            continue
        covered_kpis.add(kpi)
        names = alt_names.get(ticker)
        if names:
            alias = rng.choice(names)
        else:
            alias = company_name if company_name else ticker
        tpl = rng.choice(kpi_templates)
        qtext = tpl.replace("ABC", alias).replace("X", str(year))
        qid = f"{ticker}_{kpi}_{year}"
        queries.append((qid, qtext))
    return queries, covered_kpis


def write_queries(queries: list[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_text"])
        for qid, qtext in queries:
            writer.writerow([qid, qtext])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kpis-long",
        type=Path,
        default=DEFAULT_KPIS_LONG,
        help="Path to kpis_long.csv",
    )
    parser.add_argument(
        "--alt-names",
        type=Path,
        default=DEFAULT_ALT_NAMES,
        help="Path to companies_alt_names.json",
    )
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=DEFAULT_QUERIES_DIR,
        help="Directory containing query template JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for queries.csv",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--ocr-root",
        type=Path,
        default=DEFAULT_OCR_ROOT,
        help="OCR root directory; if set, filters to tickers present in this tree",
    )
    args = parser.parse_args()

    templates = load_query_templates(args.queries_dir)
    alt_names = load_alt_names(args.alt_names)
    ground_truth = load_ground_truth(args.kpis_long)

    if args.ocr_root and args.ocr_root.is_dir():
        ocr_tickers = discover_tickers(args.ocr_root)
        ground_truth = {
            k: v for k, v in ground_truth.items() if k[0] in ocr_tickers
        }
        print(f"Filtered to {len(ocr_tickers)} tickers from {args.ocr_root}")

    print(f"Loaded {len(templates)} template sets ({sum(len(v) for v in templates.values())} total templates)")
    print(f"Loaded {len(alt_names)} ticker aliases")
    print(f"Loaded {len(ground_truth)} ground-truth triples")

    queries, covered_kpis = generate_queries(ground_truth, alt_names, templates, seed=args.seed)
    out_path = args.output_dir / "queries.csv"
    write_queries(queries, out_path)
    print(f"Wrote {len(queries)} queries to {out_path}")

    all_kpis = {kpi for _, kpi, _ in ground_truth}
    missing = all_kpis - covered_kpis
    if missing:
        print(f"KPIs without templates (skipped): {sorted(missing)}")


if __name__ == "__main__":
    main()
