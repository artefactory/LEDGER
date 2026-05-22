"""Build page-level OCR annotation queues.

The annotation UI compares one raw page image with the corresponding Markdown
page extracted by DeepSeekOCR. Page positions are preserved exactly: page index
``i`` in an ``.mmd`` split maps to ``pages/page_XXXX.png`` with the same
zero-based index when the raw image exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

DEFAULT_OCR_ROOT = REPO_ROOT / "DeepSeekOCR_Ardian_pruned_1k"
DEFAULT_RAW_ROOT = Path(
    "/data/workspace/charles/pdf_ocr_deepseek/DeepSeekOCR_Ardian_raw_3kdocs"
)

PAGE_SPLIT_RE = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)
REPORT_NAME_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})(?:_[0-9a-fA-F]{8,})?$")
HASH_SUFFIX_RE = re.compile(r"_[0-9a-fA-F]{8,}$")

CORE_KPI_ALIASES = {
    "revenue": [
        "net sales",
        "total net sales",
        "sales revenue",
        "revenues",
        "revenue",
        "net revenue",
    ],
    "gross_profit": ["gross profit", "gross margin"],
    "operating_income": [
        "operating income",
        "income from operations",
        "operating profit",
    ],
    "net_income": [
        "net income",
        "net earnings",
        "net loss",
        "net income attributable",
    ],
    "total_assets": ["total assets"],
    "total_liabilities": ["total liabilities", "liabilities"],
    "cash_and_equivalents": [
        "cash and cash equivalents",
        "cash equivalents",
        "cash, cash equivalents",
    ],
    "operating_cash_flow": [
        "net cash provided by operating activities",
        "cash flow from operating activities",
        "operating cash flow",
    ],
    "capex": [
        "capital expenditures",
        "capital expenditure",
        "additions to property, plant and equipment",
        "purchase of property and equipment",
        "additions of long-lived assets",
    ],
}

FINANCIAL_TABLE_HEADINGS = [
    "consolidated statement of operations",
    "consolidated statements of operations",
    "consolidated income statement",
    "consolidated statements of income",
    "consolidated balance sheet",
    "consolidated balance sheets",
    "consolidated cash flow statement",
    "consolidated statements of cash flows",
    "consolidated statement of cash flows",
    "statements of comprehensive income",
    "statement of financial position",
    "notes to the consolidated financial statements",
    "selected financial data",
    "five year record",
]

NUMERIC_ROW_RE = re.compile(
    r"(?<![A-Za-z])\(?\$?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?\)?|(?<![A-Za-z])\$?\d+\.\d+"
)
MARKDOWN_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


@dataclass(frozen=True)
class ReportInfo:
    industry_slug: str
    name: str
    exchange: str
    ticker: str
    year: int
    report_dir: Path
    mmd_path: Path


@dataclass
class PageItem:
    item_id: str
    industry_slug: str
    report_name: str
    exchange: str
    ticker: str
    year: int
    page_index: int
    page_number: int
    ocr_root: str
    raw_root: str
    report_dir: str
    raw_dir: str | None
    mmd_path: str
    raw_png_path: str | None
    mmd_page_count: int
    png_page_count: int
    mapping_status: str
    mapping_warnings: list[str]
    candidate_reasons: list[str]
    page_text_sha256: str
    page_text_chars: int
    page_text_preview: str
    page_text: str

    def to_manifest_record(self, *, include_text: bool = False) -> dict[str, Any]:
        record = asdict(self)
        if not include_text:
            record.pop("page_text", None)
        return record


def parse_report_name(name: str) -> tuple[str, str, int] | None:
    match = REPORT_NAME_RE.match(name)
    if not match:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def strip_hash_suffix(name: str) -> str:
    return HASH_SUFFIX_RE.sub("", name)


def report_base_name(name: str) -> str:
    parsed = parse_report_name(name)
    if parsed is None:
        return strip_hash_suffix(name)
    exchange, ticker, year = parsed
    return f"{exchange}_{ticker}_{year}"


def find_mmd(report_dir: Path) -> Path | None:
    preferred = report_dir / f"{report_dir.name}.mmd"
    if preferred.is_file():
        return preferred

    base_preferred = report_dir / f"{report_base_name(report_dir.name)}.mmd"
    if base_preferred.is_file():
        return base_preferred

    candidates = sorted(
        path for path in report_dir.glob("*.mmd") if not path.name.endswith("_det.mmd")
    )
    if candidates:
        return candidates[0]

    fallback = sorted(report_dir.glob("*.mmd"))
    return fallback[0] if fallback else None


def discover_reports(root: Path) -> list[ReportInfo]:
    reports: list[ReportInfo] = []
    seen_dirs = sorted({mmd.parent for mmd in root.rglob("*.mmd")})
    for report_dir in seen_dirs:
        parsed = parse_report_name(report_dir.name)
        if parsed is None:
            continue
        mmd_path = find_mmd(report_dir)
        if mmd_path is None:
            continue
        exchange, ticker, year = parsed
        industry_slug = report_dir.parent.name
        reports.append(
            ReportInfo(
                industry_slug=industry_slug,
                name=report_dir.name,
                exchange=exchange,
                ticker=ticker,
                year=year,
                report_dir=report_dir,
                mmd_path=mmd_path,
            )
        )
    return reports


def split_pages(raw: str) -> list[str]:
    pages = [page.strip() for page in PAGE_SPLIT_RE.split(raw)]
    if pages and not pages[-1]:
        pages.pop()
    return pages


def load_pages(mmd_path: Path) -> list[str]:
    raw = mmd_path.read_text(encoding="utf-8", errors="replace")
    return split_pages(raw)


def resolve_raw_dir(report: ReportInfo, raw_root: Path) -> tuple[Path | None, str]:
    industry_root = raw_root / report.industry_slug
    if not industry_root.is_dir():
        return None, "raw-industry-missing"

    exact = industry_root / report.name
    if exact.is_dir():
        return exact, "ok-exact"

    base_name = report_base_name(report.name)
    stripped = industry_root / base_name
    if stripped.is_dir():
        return stripped, "ok-hash-stripped"

    matches = sorted(
        path for path in industry_root.glob(f"{base_name}*") if path.is_dir()
    )
    if len(matches) == 1:
        return matches[0], "ok-glob"
    if len(matches) > 1:
        return None, "raw-dir-ambiguous"
    return None, "raw-dir-missing"


def list_page_pngs(raw_dir: Path | None) -> list[Path]:
    if raw_dir is None:
        return []
    pages_dir = raw_dir / "pages"
    if not pages_dir.is_dir():
        return []
    return sorted(p for p in pages_dir.glob("page_*.png") if p.is_file())


def page_png_for(page_pngs: list[Path], page_index: int) -> Path | None:
    expected_name = f"page_{page_index:04d}.png"
    for path in page_pngs:
        if path.name == expected_name:
            return path
    if 0 <= page_index < len(page_pngs):
        return page_pngs[page_index]
    return None


def has_markdown_table(lines: list[str]) -> bool:
    if any(MARKDOWN_TABLE_SEPARATOR_RE.match(line) for line in lines):
        return True
    pipe_rows = sum(1 for line in lines if line.count("|") >= 2)
    return pipe_rows >= 2


def dense_numeric_row_count(lines: list[str]) -> int:
    return sum(1 for line in lines if len(NUMERIC_ROW_RE.findall(line)) >= 3)


def detect_candidate_reasons(text: str) -> list[str]:
    lowered = text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    reasons: list[str] = []

    if has_markdown_table(lines):
        reasons.append("markdown-table")
    if "<table" in lowered or "</td>" in lowered or "</tr>" in lowered:
        reasons.append("html-table")

    numeric_rows = dense_numeric_row_count(lines)
    if numeric_rows >= 3:
        reasons.append("dense-numeric-rows")

    if any(heading in lowered for heading in FINANCIAL_TABLE_HEADINGS):
        reasons.append("financial-heading")

    aliases = sorted({alias for vals in CORE_KPI_ALIASES.values() for alias in vals})
    alias_hits = [alias for alias in aliases if alias in lowered]
    if len(alias_hits) >= 2:
        reasons.append("kpi-aliases")

    return reasons


def text_preview(text: str, max_chars: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def page_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def make_mapping_warnings(
    *, raw_dir: Path | None, page_pngs: list[Path], page_index: int, mmd_page_count: int
) -> list[str]:
    warnings: list[str] = []
    if raw_dir is None:
        warnings.append("raw-directory-missing")
    elif not (raw_dir / "pages").is_dir():
        warnings.append("raw-pages-directory-missing")
    if len(page_pngs) != mmd_page_count:
        warnings.append("page-count-mismatch")
    if page_png_for(page_pngs, page_index) is None:
        warnings.append("raw-page-image-missing")
    return warnings


def build_all_items(
    *,
    ocr_root: Path,
    raw_root: Path,
    limit_reports: int | None = None,
) -> list[PageItem]:
    return list(
        iter_page_items(
            ocr_root=ocr_root,
            raw_root=raw_root,
            limit_reports=limit_reports,
        )
    )


def iter_page_items(
    *,
    ocr_root: Path,
    raw_root: Path,
    limit_reports: int | None = None,
):
    reports = discover_reports(ocr_root)
    if limit_reports is not None:
        reports = reports[:limit_reports]

    for report in reports:
        pages = load_pages(report.mmd_path)
        raw_dir, raw_status = resolve_raw_dir(report, raw_root)
        page_pngs = list_page_pngs(raw_dir)
        mmd_page_count = len(pages)
        png_page_count = len(page_pngs)

        for page_index, page_text in enumerate(pages):
            raw_png = page_png_for(page_pngs, page_index)
            warnings = make_mapping_warnings(
                raw_dir=raw_dir,
                page_pngs=page_pngs,
                page_index=page_index,
                mmd_page_count=mmd_page_count,
            )
            reasons = detect_candidate_reasons(page_text)
            item_id = f"{report.industry_slug}/{report.name}/page_{page_index:04d}"
            yield PageItem(
                item_id=item_id,
                industry_slug=report.industry_slug,
                report_name=report.name,
                exchange=report.exchange,
                ticker=report.ticker,
                year=report.year,
                page_index=page_index,
                page_number=page_index + 1,
                ocr_root=str(ocr_root),
                raw_root=str(raw_root),
                report_dir=str(report.report_dir),
                raw_dir=str(raw_dir) if raw_dir else None,
                mmd_path=str(report.mmd_path),
                raw_png_path=str(raw_png) if raw_png else None,
                mmd_page_count=mmd_page_count,
                png_page_count=png_page_count,
                mapping_status=raw_status,
                mapping_warnings=warnings,
                candidate_reasons=reasons,
                page_text_sha256=page_text_hash(page_text),
                page_text_chars=len(page_text),
                page_text_preview=text_preview(page_text),
                page_text="",
            )


def new_summary_state() -> dict[str, Any]:
    return {
        "report_names": set(),
        "pages_total": 0,
        "mapping_status_counts": {},
        "mapping_warning_counts": {},
        "candidate_reason_counts": {},
    }


def update_summary_state(state: dict[str, Any], item: PageItem) -> None:
    state["report_names"].add(item.report_name)
    state["pages_total"] += 1
    statuses = state["mapping_status_counts"]
    statuses[item.mapping_status] = statuses.get(item.mapping_status, 0) + 1
    warnings = state["mapping_warning_counts"]
    for warning in item.mapping_warnings:
        warnings[warning] = warnings.get(warning, 0) + 1
    reasons = state["candidate_reason_counts"]
    for reason in item.candidate_reasons:
        reasons[reason] = reasons.get(reason, 0) + 1


def finish_summary_state(
    state: dict[str, Any], queue: list[PageItem]
) -> dict[str, Any]:
    return {
        "reports_total": len(state["report_names"]),
        "pages_total": state["pages_total"],
        "queue_reports": len({item.report_name for item in queue}),
        "queue_pages": len(queue),
        "mapping_status_counts": state["mapping_status_counts"],
        "mapping_warning_counts": state["mapping_warning_counts"],
        "candidate_reason_counts": state["candidate_reason_counts"],
    }


def select_queue(
    items: list[PageItem],
    *,
    queue_mode: str,
    sample_size: int | None = None,
    seed: int = 17,
    limit: int | None = None,
) -> list[PageItem]:
    if queue_mode == "all":
        selected = list(items)
    elif queue_mode == "table-candidates":
        selected = [item for item in items if item.candidate_reasons]
    elif queue_mode == "sample":
        size = sample_size if sample_size is not None else 100
        rng = random.Random(seed)
        selected = rng.sample(items, min(size, len(items)))
        selected.sort(
            key=lambda item: (item.industry_slug, item.report_name, item.page_index)
        )
    else:
        raise ValueError(f"unknown queue mode: {queue_mode}")

    if limit is not None:
        selected = selected[:limit]
    return selected


def build_queue(
    *,
    ocr_root: Path,
    raw_root: Path,
    queue_mode: str = "table-candidates",
    sample_size: int | None = None,
    seed: int = 17,
    limit: int | None = None,
    limit_reports: int | None = None,
) -> tuple[list[PageItem], dict[str, Any]]:
    if queue_mode not in {"all", "table-candidates", "sample"}:
        raise ValueError(f"unknown queue mode: {queue_mode}")

    queue: list[PageItem] = []
    summary_state = new_summary_state()
    rng = random.Random(seed)
    sample_seen = 0
    sample_target = sample_size if sample_size is not None else 100
    scan_stopped_by_limit = False

    for item in iter_page_items(
        ocr_root=ocr_root,
        raw_root=raw_root,
        limit_reports=limit_reports,
    ):
        update_summary_state(summary_state, item)
        if queue_mode == "sample":
            sample_seen += 1
            if len(queue) < sample_target:
                queue.append(item)
            else:
                replace_at = rng.randint(0, sample_seen - 1)
                if replace_at < sample_target:
                    queue[replace_at] = item
            continue

        include_item = queue_mode == "all" or bool(item.candidate_reasons)
        if not include_item:
            continue
        queue.append(item)
        if limit is not None and len(queue) >= limit:
            scan_stopped_by_limit = True
            break

    if queue_mode == "sample":
        queue.sort(
            key=lambda item: (item.industry_slug, item.report_name, item.page_index)
        )
        if limit is not None:
            queue = queue[:limit]

    summary = finish_summary_state(summary_state, queue)
    summary.update(
        {
            "queue_mode": queue_mode,
            "sample_size": sample_size,
            "seed": seed,
            "limit": limit,
            "limit_reports": limit_reports,
            "scan_stopped_by_limit": scan_stopped_by_limit,
            "ocr_root": str(ocr_root),
            "raw_root": str(raw_root),
        }
    )
    return queue, summary


def summarize_items(all_items: list[PageItem], queue: list[PageItem]) -> dict[str, Any]:
    report_names = {item.report_name for item in all_items}
    queue_reports = {item.report_name for item in queue}
    warnings: dict[str, int] = {}
    statuses: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for item in all_items:
        statuses[item.mapping_status] = statuses.get(item.mapping_status, 0) + 1
        for warning in item.mapping_warnings:
            warnings[warning] = warnings.get(warning, 0) + 1
        for reason in item.candidate_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "reports_total": len(report_names),
        "pages_total": len(all_items),
        "queue_reports": len(queue_reports),
        "queue_pages": len(queue),
        "mapping_status_counts": statuses,
        "mapping_warning_counts": warnings,
        "candidate_reason_counts": reason_counts,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an OCR page annotation queue.")
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--queue-mode",
        choices=["all", "table-candidates", "sample"],
        default="table-candidates",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None, help="Maximum queued pages.")
    parser.add_argument(
        "--limit-reports",
        type=int,
        default=None,
        help="Read only the first N reports before queue selection.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional manifest JSON path."
    )
    parser.add_argument("--check", action="store_true", help="Print summary and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    queue, summary = build_queue(
        ocr_root=args.ocr_root,
        raw_root=args.raw_root,
        queue_mode=args.queue_mode,
        sample_size=args.sample_size,
        seed=args.seed,
        limit=args.limit,
        limit_reports=args.limit_reports,
    )

    payload = {
        "summary": summary,
        "items": [item.to_manifest_record() for item in queue],
    }
    if args.output:
        write_json(args.output, payload)
    if args.check or not args.output:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
