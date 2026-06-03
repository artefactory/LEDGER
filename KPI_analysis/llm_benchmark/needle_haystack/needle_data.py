"""Data layer for the needle-in-a-haystack benchmark.

Joins the natural-language queries in ``test_set.csv`` to:

1. the ground-truth value in ``KPI_analysis/output/kpis_long.csv``, and
2. the OCR'd annual report the value lives in
   (``DeepSeekOCR_Ardian_pruned_1k/{EX}_{TICKER}_{YEAR}/``).

The join key is the ``query_id``. ``generate_queries.py`` builds it as
``f"{ticker}_{kpi}_{year}"`` (see ``generate_queries.py:125``), so we
reconstruct the *same* string from each ``kpis_long.csv`` row and join on it
exactly — no fragile ``query_id`` parsing required. This was verified to match
3000/3000 test-set queries.

Both ``run_needle.py`` (needs the report path + query text) and
``score_needle.py`` (needs the ground-truth value + metadata) import from here
so the join logic lives in exactly one place.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

NEEDLE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = NEEDLE_DIR.parent
KPI_DIR = BENCHMARK_DIR.parent
REPO_ROOT = KPI_DIR.parent

# Reuse the parent package's report discovery / mmd selection so this benchmark
# loads documents identically to the multi-KPI one.
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(KPI_DIR / "kpi_fetch_and_build"))

from document import ReportInfo, discover_reports  # noqa: E402
from tags import KPI_DEFS  # noqa: E402

try:
    # Canonical, scope-precise one-line KPI definitions, reused verbatim from
    # the multi-KPI benchmark so the two benchmarks agree on what each key means.
    from kpi_catalogue import DESCRIPTIONS as KPI_DESCRIPTIONS  # noqa: E402
except Exception:  # pragma: no cover - kpi_catalogue should always import
    KPI_DESCRIPTIONS = {}


DEFAULT_TEST_SET = NEEDLE_DIR / "test_set.csv"
DEFAULT_PROTOTYPE = NEEDLE_DIR / "prototype_3_reports.csv"
DEFAULT_KPIS_LONG = KPI_DIR / "output" / "kpis_long.csv"
DEFAULT_OCR_ROOT = REPO_ROOT / "DeepSeekOCR_Ardian_pruned_1k"


# ---------------------------------------------------------------------------
# KPI metadata (unit class + human label + canonical definition)
# ---------------------------------------------------------------------------

# Maps each KPI key to how its value is expressed, which drives the scaling
# instruction in the prompt and the unit shown to the model:
#   "monetary"  -> single units of reporting currency; apply in-thousands/
#                  millions/billions scaling.
#   "per_share" -> EPS; report as printed, do NOT scale.
#   "shares"    -> share count; scale to single shares (often printed in
#                  thousands / millions).
_KIND_BY_UNIT = {"USD": "monetary", "USD/shares": "per_share", "shares": "shares"}
KPI_UNIT_CLASS: dict[str, str] = {d.key: _KIND_BY_UNIT.get(d.unit, "monetary") for d in KPI_DEFS}
KPI_LABEL: dict[str, str] = {d.key: d.label for d in KPI_DEFS}

UNIT_PHRASE: dict[str, str] = {
    "monetary": (
        "single units of the reporting currency (e.g. dollars). Apply any "
        "'in thousands' / 'in millions' / 'in billions' scaling printed on the "
        "statement before reporting."
    ),
    "per_share": (
        "a per-share figure in the reporting currency. Report it exactly as "
        "printed — do NOT apply any thousands/millions scaling."
    ),
    "shares": (
        "a number of shares (not currency). Apply any 'in thousands' / "
        "'in millions' scaling so the value is in single shares."
    ),
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruth:
    ticker: str
    kpi: str
    year: int
    value: float
    source: str
    company_name: str
    exchange: str
    industry: str
    tag: str


@dataclass(frozen=True)
class QueryRecord:
    """One benchmark query, fully resolved to its report and ground truth."""

    query_id: str
    query_text: str
    ticker: str
    kpi: str
    year: int
    unit_class: str          # "monetary" | "per_share" | "shares"
    kpi_label: str           # human label, e.g. "Net income"
    kpi_definition: str      # canonical scope-precise one-liner
    gt: GroundTruth | None   # None if no ground-truth row (should not happen)
    report: ReportInfo | None  # None if no OCR report for (ticker, year)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_queries(test_set_path: Path) -> list[tuple[str, str]]:
    """Return [(query_id, query_text), ...] preserving file order."""
    out: list[tuple[str, str]] = []
    with test_set_path.open(newline="") as f:
        for row in csv.DictReader(f):
            out.append((row["query_id"], row["query_text"]))
    return out


def load_ground_truth(kpis_long_path: Path) -> dict[str, GroundTruth]:
    """Index ``kpis_long.csv`` by reconstructed ``query_id``.

    Key construction mirrors ``generate_queries.py:125`` exactly:
    ``f"{ticker}_{kpi}_{year}"``.
    """
    out: dict[str, GroundTruth] = {}
    with kpis_long_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                year = int(row["year"])
                value = float(row["value"])
            except (ValueError, KeyError):
                continue
            ticker = row["ticker"].strip()
            kpi = row["kpi"].strip()
            qid = f"{ticker}_{kpi}_{year}"
            out[qid] = GroundTruth(
                ticker=ticker,
                kpi=kpi,
                year=year,
                value=value,
                source=row.get("source", "").strip(),
                company_name=row.get("company_name", "").strip(),
                exchange=row.get("exchange", "").strip(),
                industry=row.get("industry", "").strip(),
                tag=row.get("tag", "").strip(),
            )
    return out


def build_report_index(ocr_root: Path) -> dict[tuple[str, int], ReportInfo]:
    """Map ``(ticker, year) -> ReportInfo`` for every discoverable report.

    If two report dirs share a (ticker, year) (e.g. a re-OCR'd duplicate with a
    hex suffix), the lexicographically-first directory name wins, deterministically.
    """
    index: dict[tuple[str, int], ReportInfo] = {}
    for r in discover_reports(ocr_root):
        key = (r.ticker, r.year)
        cur = index.get(key)
        if cur is None or r.name < cur.name:
            index[key] = r
    return index


def build_query_records(
    *,
    test_set_path: Path = DEFAULT_TEST_SET,
    kpis_long_path: Path = DEFAULT_KPIS_LONG,
    ocr_root: Path = DEFAULT_OCR_ROOT,
    require_report: bool = True,
) -> tuple[list[QueryRecord], dict[str, int]]:
    """Resolve every query in ``test_set_path`` to its ground truth + report.

    Returns ``(records, stats)`` where ``stats`` counts dropped queries by
    reason. When ``require_report`` is True, queries with no OCR report are
    dropped (they cannot be answered); otherwise they are kept with
    ``report=None`` for diagnostics.
    """
    queries = load_queries(test_set_path)
    gt_index = load_ground_truth(kpis_long_path)
    report_index = build_report_index(ocr_root)

    records: list[QueryRecord] = []
    stats = {"total": len(queries), "no_ground_truth": 0, "no_report": 0, "kept": 0}

    for qid, qtext in queries:
        gt = gt_index.get(qid)
        if gt is None:
            stats["no_ground_truth"] += 1
            continue
        report = report_index.get((gt.ticker, gt.year))
        if report is None:
            stats["no_report"] += 1
            if require_report:
                continue
        unit_class = KPI_UNIT_CLASS.get(gt.kpi, "monetary")
        records.append(
            QueryRecord(
                query_id=qid,
                query_text=qtext,
                ticker=gt.ticker,
                kpi=gt.kpi,
                year=gt.year,
                unit_class=unit_class,
                kpi_label=KPI_LABEL.get(gt.kpi, gt.kpi),
                kpi_definition=KPI_DESCRIPTIONS.get(gt.kpi, ""),
                gt=gt,
                report=report,
            )
        )
        stats["kept"] += 1
    return records, stats


def group_by_report(records: list[QueryRecord]) -> list[tuple[ReportInfo, list[QueryRecord]]]:
    """Group resolved queries by their source report, sorted deterministically.

    Reports are ordered by directory name; queries within a report are ordered
    by query_id. This contiguity is what lets vLLM's prefix cache serve a
    report's whole batch of queries from a single document prefill.
    """
    groups: dict[str, list[QueryRecord]] = {}
    report_by_name: dict[str, ReportInfo] = {}
    for rec in records:
        if rec.report is None:
            continue
        report_by_name[rec.report.name] = rec.report
        groups.setdefault(rec.report.name, []).append(rec)
    out: list[tuple[ReportInfo, list[QueryRecord]]] = []
    for name in sorted(groups):
        qs = sorted(groups[name], key=lambda r: r.query_id)
        out.append((report_by_name[name], qs))
    return out
