"""Pilot OCR-to-KPI validator with explicit unit scaling and audit outputs.

This script validates whether KPI targets from `output/kpis_long.csv` appear in
OCR annual-report text (`*.mmd`) after unit normalization.

Design goals:
- Build a target table for core KPIs, preferring EDGAR rows when available.
- Read OCR reports with page-level traceability.
- Detect unit context at inline, line, page, and document scope.
- Extract numeric candidates near KPI aliases from table and narrative text.
- Normalize candidates to single-dollar units and match with strict tolerance.
- Write auditable outputs for matched, unmatched, and ambiguous rows.

Default pilot corpus: sample_data/subset_auto_parts_2017_2022/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

DEFAULT_OCR_ROOT = REPO_ROOT / "sample_data" / "subset_auto_parts_2017_2022"
DEFAULT_KPIS_LONG = HERE / "output" / "kpis_long.csv"
DEFAULT_OUT_DIR = HERE / "output" / "ocr_validation"

PAGE_SPLIT = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)
REPORT_NAME_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})(?:_[0-9a-fA-F]+)?$")

SMART_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00b4": "'",
    }
)

# Core KPI set for the pilot.
CORE_KPI_ALIASES: dict[str, list[str]] = {
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

UNIT_HINT_PATTERNS: list[re.Pattern[str]] = [
    # Covers: (in millions), ($ in millions), (in millions, except per share data)
    re.compile(
        r"\((?:[^)]*?)\b(?:\$|usd|us\$|u\.s\.\$)?\s*(?:in\s+)?"
        r"(?P<unit>thousands?|millions?|billions?)\b[^)]*\)",
        re.IGNORECASE,
    ),
    # Covers: dollars in thousands
    re.compile(
        r"\bdollars?\s+in\s+(?P<unit>thousands?|millions?|billions?)\b",
        re.IGNORECASE,
    ),
    # Covers: all dollar amounts are stated in millions
    re.compile(
        r"\ball\s+dollar\s+amounts?\s+(?:are\s+)?(?:stated|presented|shown)\s+"
        r"in\s+(?P<unit>thousands?|millions?|billions?)\b",
        re.IGNORECASE,
    ),
    # Covers: in thousands, except per share data
    re.compile(
        r"\bin\s+(?P<unit>thousands?|millions?|billions?)\b"
        r"(?:\s*,?\s*except\s+per\s+share\s+data)?",
        re.IGNORECASE,
    ),
]

NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_/.-])
    (?P<raw>
      (?P<open>\()?\s*
      (?P<sign>-)?\s*
      \$?\s*
      (?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
      \s*(?P<suffix>bn|billion|billions|mm|million|millions|thousand|thousands|k|m|b)?
      \s*(?(open)\))
    )
    (?![A-Za-z0-9_/.-])
    """,
    re.IGNORECASE | re.VERBOSE,
)

NON_USD_CONTEXT_RE = re.compile(
    r"\b(eur|euro|gbp|sterling|pounds?|rmb|yuan|cny|cad|aud|jpy|yen|chf)\b",
    re.IGNORECASE,
)
USD_CONTEXT_RE = re.compile(r"\b(usd|u\.s\.\$|us\$|dollars?)\b|\$", re.IGNORECASE)

UNIT_SOURCE_RANK = {
    "inline": 0,
    "line": 1,
    "page": 2,
    "document": 3,
    "default": 4,
}
LOCATION_RANK = {
    "table": 0,
    "narrative": 1,
}


@dataclass(frozen=True)
class TargetRow:
    ticker: str
    year: int
    kpi: str
    value: float
    source: str
    tag: str
    company_name: str
    exchange: str
    industry: str


@dataclass(frozen=True)
class ReportInfo:
    name: str
    exchange: str
    ticker: str
    year: int
    mmd_path: Path


@dataclass(frozen=True)
class NumberToken:
    raw: str
    value: float
    start: int
    end: int
    inline_multiplier: float | None


@dataclass
class Candidate:
    report_name: str
    ticker: str
    year: int
    kpi: str
    target_value: float
    alias: str
    page: int
    line_index: int
    alias_line_offset: int
    location_kind: str
    raw_number: str
    parsed_value: float
    inline_multiplier: float | None
    chosen_multiplier: float
    unit_source: str
    unit_conflict: bool
    currency_context: str
    normalized_value: float
    snippet: str
    rel_error: float | None = None
    rejected_reason: str = ""


def normalize_text(text: str) -> str:
    text = text.translate(SMART_MAP)
    text = text.replace("\\(", "(").replace("\\)", ")").replace("\\$", "$")
    text = re.sub(r"[\u2013\u2014\u2212]", "-", text)
    return text


def normalize_line(line: str) -> str:
    line = normalize_text(line)
    # OCR tables sometimes emit stray close parens before positive values.
    line = re.sub(r"(?<!\()\)\s+(?=\d)", "", line)
    return line.strip()


def parse_report_name(name: str) -> tuple[str, str, int] | None:
    m = REPORT_NAME_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def find_mmd(report_dir: Path) -> Path | None:
    preferred = report_dir / f"{report_dir.name}.mmd"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        p for p in report_dir.glob("*.mmd") if not p.name.endswith("_det.mmd")
    )
    if candidates:
        return candidates[0]
    fallback = sorted(report_dir.glob("*.mmd"))
    return fallback[0] if fallback else None


def discover_reports(root: Path) -> list[ReportInfo]:
    out: list[ReportInfo] = []
    for report_dir in sorted({mmd.parent for mmd in root.rglob("*.mmd")}):
        parsed = parse_report_name(report_dir.name)
        if parsed is None:
            continue
        mmd = find_mmd(report_dir)
        if mmd is None:
            continue
        exchange, ticker, year = parsed
        out.append(
            ReportInfo(
                name=report_dir.name,
                exchange=exchange,
                ticker=ticker,
                year=year,
                mmd_path=mmd,
            )
        )
    return out


def unit_word_to_multiplier(word: str) -> float | None:
    w = word.lower().rstrip(".")
    if w in {"thousand", "thousands", "k"}:
        return 1e3
    if w in {"million", "millions", "mm", "m"}:
        return 1e6
    if w in {"billion", "billions", "bn", "b"}:
        return 1e9
    return None


def detect_unit_hints(text: str) -> list[float]:
    hints: list[float] = []
    if not text:
        return hints
    for pat in UNIT_HINT_PATTERNS:
        for m in pat.finditer(text):
            mult = unit_word_to_multiplier(m.group("unit"))
            if mult is not None:
                hints.append(mult)
    return hints


def resolve_multiplier(
    inline_multiplier: float | None,
    line_hints: list[float],
    page_hints: list[float],
    doc_hints: list[float],
) -> tuple[float, str, bool]:
    if inline_multiplier is not None:
        return inline_multiplier, "inline", False
    if line_hints:
        line_unique = sorted(set(line_hints))
        return line_unique[0], "line", len(line_unique) > 1
    if page_hints:
        page_unique = sorted(set(page_hints))
        return page_unique[0], "page", len(page_unique) > 1
    if doc_hints:
        doc_unique = sorted(set(doc_hints))
        return doc_unique[0], "document", len(doc_unique) > 1
    return 1.0, "default", False


def classify_currency_context(line: str, start: int, end: int) -> str:
    lo = max(0, start - 32)
    hi = min(len(line), end + 32)
    window = line[lo:hi]
    if NON_USD_CONTEXT_RE.search(window):
        return "non_usd"
    if USD_CONTEXT_RE.search(window):
        return "usd"
    return "unknown"


def parse_numbers_from_line(line: str) -> list[NumberToken]:
    nums: list[NumberToken] = []
    seen_spans: set[tuple[int, int]] = set()
    for m in NUMBER_RE.finditer(line):
        span = (m.start("raw"), m.end("raw"))
        if span in seen_spans:
            continue
        seen_spans.add(span)

        raw = m.group("raw").strip()
        num_text = m.group("num")
        suffix = m.group("suffix")

        # Skip percentages.
        after = line[m.end("raw") : m.end("raw") + 1]
        if after == "%":
            continue

        try:
            value = float(num_text.replace(",", ""))
        except ValueError:
            continue

        # Drop likely years unless an explicit unit suffix exists.
        if suffix is None and num_text.isdigit() and len(num_text) == 4:
            y = int(num_text)
            if 1900 <= y <= 2100:
                continue

        neg = (m.group("open") is not None and ")" in raw) or (m.group("sign") == "-")
        if neg:
            value = -value

        inline_multiplier = unit_word_to_multiplier(suffix) if suffix else None
        nums.append(
            NumberToken(
                raw=raw,
                value=value,
                start=m.start("raw"),
                end=m.end("raw"),
                inline_multiplier=inline_multiplier,
            )
        )
    return nums


def rel_error(candidate: float, target: float) -> float:
    return abs(candidate - target) / max(abs(target), 1.0)


def compact_snippet(line: str, width: int = 220) -> str:
    text = re.sub(r"\s+", " ", line).strip()
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def compile_alias_patterns(
    core_aliases: dict[str, list[str]],
    selected_kpis: list[str],
) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    out: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for kpi in selected_kpis:
        aliases = core_aliases.get(kpi, [])
        pats: list[tuple[str, re.Pattern[str]]] = []
        for alias in aliases:
            esc = re.escape(alias)
            esc = esc.replace(r"\ ", r"\s+")
            esc = esc.replace(r"\-", r"[-\s]?")
            # Keep boundaries permissive enough for OCR punctuation noise.
            pat = re.compile(rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])", re.IGNORECASE)
            pats.append((alias, pat))
        out[kpi] = pats
    return out


def load_targets(
    kpis_long_csv: Path,
    selected_kpis: list[str],
    *,
    edgar_only: bool,
) -> list[TargetRow]:
    grouped: dict[tuple[str, int, str], list[TargetRow]] = defaultdict(list)
    with kpis_long_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kpi = row.get("kpi", "")
            if kpi not in selected_kpis:
                continue
            try:
                year = int(row["year"])
                value = float(row["value"])
            except (KeyError, ValueError):
                continue
            target = TargetRow(
                ticker=row.get("ticker", "").strip(),
                year=year,
                kpi=kpi,
                value=value,
                source=row.get("source", "").strip(),
                tag=row.get("tag", "").strip(),
                company_name=row.get("company_name", "").strip(),
                exchange=row.get("exchange", "").strip(),
                industry=row.get("industry", "").strip(),
            )
            grouped[(target.ticker, target.year, target.kpi)].append(target)

    out: list[TargetRow] = []
    for key, rows in sorted(grouped.items()):
        edgar_rows = [r for r in rows if r.source == "edgar"]
        if edgar_rows:
            out.append(edgar_rows[0])
            continue
        if edgar_only:
            continue
        out.append(rows[0])
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def candidate_priority(c: Candidate) -> tuple[float, float, float, float]:
    return (
        0 if c.alias_line_offset == 0 else 1,
        LOCATION_RANK.get(c.location_kind, 99),
        UNIT_SOURCE_RANK.get(c.unit_source, 99),
        c.rel_error if c.rel_error is not None else float("inf"),
    )


def evaluate_target(
    target: TargetRow,
    alias_found: bool,
    candidates: list[Candidate],
    tolerance: float,
) -> tuple[dict[str, object], Candidate | None]:
    valid = [c for c in candidates if not c.rejected_reason]
    for c in valid:
        c.rel_error = rel_error(c.normalized_value, target.value)

    hits = [c for c in valid if c.rel_error is not None and c.rel_error <= tolerance]

    status = "unmatched"
    reason = "alias-not-found"
    best: Candidate | None = None

    if not alias_found:
        reason = "alias-not-found"
    elif not candidates:
        reason = "alias-found-no-numeric"
    elif not valid:
        reason = "non-usd-context"
    elif len(hits) > 1:
        status = "ambiguous"
        reason = "multiple-valid-candidates"
        best = min(hits, key=candidate_priority)
    elif len(hits) == 1:
        status = "matched"
        reason = "within-threshold"
        best = hits[0]
    else:
        best = min(
            valid,
            key=lambda c: c.rel_error if c.rel_error is not None else float("inf"),
        )
        if any(c.unit_conflict for c in valid):
            reason = "multiple-unit-scopes"
        elif all(c.unit_source == "default" for c in valid):
            reason = "unit-not-found"
        else:
            reason = "numeric-found-but-outside-threshold"

    row: dict[str, object] = {
        "report_name": "",
        "ticker": target.ticker,
        "year": target.year,
        "kpi": target.kpi,
        "company_name": target.company_name,
        "target_source": target.source,
        "target_tag": target.tag,
        "target_value": target.value,
        "status": status,
        "reason": reason,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "hit_count": len(hits),
        "best_rel_error": "",
        "best_normalized_value": "",
        "best_raw_number": "",
        "best_page": "",
        "best_line_index": "",
        "best_alias": "",
        "best_unit_source": "",
        "best_multiplier": "",
        "best_currency_context": "",
        "best_location_kind": "",
        "best_snippet": "",
    }

    if best is not None:
        row.update(
            {
                "report_name": best.report_name,
                "best_rel_error": best.rel_error,
                "best_normalized_value": best.normalized_value,
                "best_raw_number": best.raw_number,
                "best_page": best.page,
                "best_line_index": best.line_index,
                "best_alias": best.alias,
                "best_unit_source": best.unit_source,
                "best_multiplier": best.chosen_multiplier,
                "best_currency_context": best.currency_context,
                "best_location_kind": best.location_kind,
                "best_snippet": best.snippet,
            }
        )

    return row, best


def build_summary_tables(
    audit_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    by_kpi_year: dict[tuple[str, int], Counter] = defaultdict(Counter)
    by_reason: Counter = Counter()
    by_ticker: dict[str, Counter] = defaultdict(Counter)

    for row in audit_rows:
        kpi = str(row["kpi"])
        year = int(row["year"])
        status = str(row["status"])
        reason = str(row["reason"])
        ticker = str(row["ticker"])

        by_kpi_year[(kpi, year)]["total"] += 1
        by_kpi_year[(kpi, year)][status] += 1
        if status != "matched":
            by_reason[reason] += 1

        by_ticker[ticker]["total"] += 1
        by_ticker[ticker][status] += 1
        if status != "matched":
            by_ticker[ticker][f"reason::{reason}"] += 1

    kpi_year_rows: list[dict[str, object]] = []
    for (kpi, year), c in sorted(by_kpi_year.items()):
        total = c["total"]
        matched = c["matched"]
        ambiguous = c["ambiguous"]
        unmatched = total - matched - ambiguous
        kpi_year_rows.append(
            {
                "kpi": kpi,
                "year": year,
                "total": total,
                "matched": matched,
                "ambiguous": ambiguous,
                "unmatched": unmatched,
                "match_rate": (matched / total) if total else 0.0,
            }
        )

    reason_rows = [
        {"reason": reason, "count": count} for reason, count in by_reason.most_common()
    ]

    ticker_rows: list[dict[str, object]] = []
    for ticker, c in sorted(by_ticker.items()):
        total = c["total"]
        matched = c["matched"]
        ambiguous = c["ambiguous"]
        unmatched = total - matched - ambiguous
        reason_counts = [
            (k.split("::", 1)[1], v) for k, v in c.items() if k.startswith("reason::")
        ]
        reason_counts.sort(key=lambda kv: -kv[1])
        top_reason = reason_counts[0][0] if reason_counts else ""
        ticker_rows.append(
            {
                "ticker": ticker,
                "total": total,
                "matched": matched,
                "ambiguous": ambiguous,
                "unmatched": unmatched,
                "match_rate": (matched / total) if total else 0.0,
                "top_failure_reason": top_reason,
            }
        )

    return kpi_year_rows, reason_rows, ticker_rows


def build_manual_qa_sample(
    audit_rows: list[dict[str, object]],
    sample_per_bucket: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    matched = [r for r in audit_rows if r["status"] == "matched"]
    not_matched = [r for r in audit_rows if r["status"] != "matched"]

    sample: list[dict[str, object]] = []
    for label, pool in (("matched", matched), ("not_matched", not_matched)):
        if not pool:
            continue
        n = min(sample_per_bucket, len(pool))
        for row in rng.sample(pool, n):
            sample.append({"qa_bucket": label, **row})
    return sample


def parse_divisors(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        token = part.strip().replace("_", "")
        if not token:
            continue
        try:
            v = float(token)
        except ValueError as exc:
            raise ValueError(f"invalid divisor: {part!r}") from exc
        if v <= 0:
            raise ValueError(f"divisor must be > 0: {part!r}")
        out.append(v)
    # Keep user order but dedupe.
    dedup: list[float] = []
    seen: set[float] = set()
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def int_or_zero(v: object) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def float_or_none(v: object) -> float | None:
    try:
        if v == "" or v is None:
            return None
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def find_alias_near_line(
    line_idx: int,
    lines: list[str],
    patterns: list[tuple[str, re.Pattern[str]]],
    max_distance: int,
) -> tuple[str, int]:
    for dist in range(0, max_distance + 1):
        probes = [line_idx] if dist == 0 else [line_idx - dist, line_idx + dist]
        for p in probes:
            if p < 0 or p >= len(lines):
                continue
            line = lines[p]
            if not line:
                continue
            for alias, pat in patterns:
                if pat.search(line):
                    return alias, dist
    return "", 99


def evaluate_reverse_target(
    target: TargetRow,
    candidates: list[Candidate],
    tolerance: float,
) -> tuple[dict[str, object], Candidate | None]:
    valid = [c for c in candidates if not c.rejected_reason]
    for c in valid:
        c.rel_error = rel_error(c.normalized_value, target.value)
    hits = [c for c in valid if c.rel_error is not None and c.rel_error <= tolerance]

    status = "unmatched"
    reason = "value-not-found"
    best: Candidate | None = None

    if not candidates:
        reason = "value-not-found"
    elif not valid:
        reason = "non-usd-context"
    elif len(hits) > 1:
        status = "ambiguous"
        reason = "multiple-valid-candidates"
        best = min(hits, key=candidate_priority)
    elif len(hits) == 1:
        status = "matched"
        reason = "within-threshold"
        best = hits[0]
    else:
        best = min(
            valid,
            key=lambda c: c.rel_error if c.rel_error is not None else float("inf"),
        )
        reason = "numeric-found-but-outside-threshold"

    row: dict[str, object] = {
        "report_name": "",
        "ticker": target.ticker,
        "year": target.year,
        "kpi": target.kpi,
        "company_name": target.company_name,
        "target_source": target.source,
        "target_tag": target.tag,
        "target_value": target.value,
        "status": status,
        "reason": reason,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "hit_count": len(hits),
        "best_rel_error": "",
        "best_normalized_value": "",
        "best_raw_number": "",
        "best_page": "",
        "best_line_index": "",
        "best_alias": "",
        "best_unit_source": "",
        "best_multiplier": "",
        "best_currency_context": "",
        "best_location_kind": "",
        "best_snippet": "",
    }

    if best is not None:
        row.update(
            {
                "report_name": best.report_name,
                "best_rel_error": best.rel_error,
                "best_normalized_value": best.normalized_value,
                "best_raw_number": best.raw_number,
                "best_page": best.page,
                "best_line_index": best.line_index,
                "best_alias": best.alias,
                "best_unit_source": best.unit_source,
                "best_multiplier": best.chosen_multiplier,
                "best_currency_context": best.currency_context,
                "best_location_kind": best.location_kind,
                "best_snippet": best.snippet,
            }
        )

    return row, best


def merge_audit_rows(
    forward_row: dict[str, object],
    reverse_row: dict[str, object],
) -> dict[str, object]:
    f_status = str(forward_row["status"])
    r_status = str(reverse_row["status"])
    f_reason = str(forward_row["reason"])
    r_reason = str(reverse_row["reason"])

    merged_status = "unmatched"
    merged_reason = "forward+reverse-unmatched"
    winner = ""
    winner_row: dict[str, object] | None = None

    f_err = float_or_none(forward_row.get("best_rel_error"))
    r_err = float_or_none(reverse_row.get("best_rel_error"))

    if f_status == "matched" and r_status == "matched":
        merged_status = "matched"
        merged_reason = "both-matched"
        if r_err is None or (f_err is not None and f_err <= r_err):
            winner = "forward"
            winner_row = forward_row
        else:
            winner = "reverse"
            winner_row = reverse_row
    elif f_status == "matched":
        merged_status = "matched"
        merged_reason = "forward-only-match"
        winner = "forward"
        winner_row = forward_row
    elif r_status == "matched":
        merged_status = "matched"
        merged_reason = "reverse-only-match"
        winner = "reverse"
        winner_row = reverse_row
    elif f_status == "ambiguous" and r_status == "ambiguous":
        merged_status = "ambiguous"
        merged_reason = "both-ambiguous"
        if r_err is None or (f_err is not None and f_err <= r_err):
            winner = "forward"
            winner_row = forward_row
        else:
            winner = "reverse"
            winner_row = reverse_row
    elif f_status == "ambiguous":
        merged_status = "ambiguous"
        merged_reason = "forward-ambiguous"
        winner = "forward"
        winner_row = forward_row
    elif r_status == "ambiguous":
        merged_status = "ambiguous"
        merged_reason = "reverse-ambiguous"
        winner = "reverse"
        winner_row = reverse_row
    else:
        merged_status = "unmatched"
        merged_reason = f"forward:{f_reason}|reverse:{r_reason}"
        if r_err is None or (f_err is not None and f_err <= r_err):
            winner = "forward"
            winner_row = forward_row
        else:
            winner = "reverse"
            winner_row = reverse_row

    out: dict[str, object] = {
        "report_name": str(
            forward_row.get("report_name") or reverse_row.get("report_name") or ""
        ),
        "ticker": forward_row["ticker"],
        "year": forward_row["year"],
        "kpi": forward_row["kpi"],
        "company_name": forward_row.get("company_name", ""),
        "target_source": forward_row.get("target_source", ""),
        "target_tag": forward_row.get("target_tag", ""),
        "target_value": forward_row.get("target_value", ""),
        "status": merged_status,
        "reason": merged_reason,
        "winner_pipeline": winner,
        "forward_status": f_status,
        "forward_reason": f_reason,
        "reverse_status": r_status,
        "reverse_reason": r_reason,
        "forward_candidate_count": int_or_zero(forward_row.get("candidate_count")),
        "reverse_candidate_count": int_or_zero(reverse_row.get("candidate_count")),
        "candidate_count": int_or_zero(forward_row.get("candidate_count"))
        + int_or_zero(reverse_row.get("candidate_count")),
        "forward_hit_count": int_or_zero(forward_row.get("hit_count")),
        "reverse_hit_count": int_or_zero(reverse_row.get("hit_count")),
        "hit_count": int_or_zero(forward_row.get("hit_count"))
        + int_or_zero(reverse_row.get("hit_count")),
        "valid_candidate_count": int_or_zero(forward_row.get("valid_candidate_count"))
        + int_or_zero(reverse_row.get("valid_candidate_count")),
        "best_rel_error": "",
        "best_normalized_value": "",
        "best_raw_number": "",
        "best_page": "",
        "best_line_index": "",
        "best_alias": "",
        "best_unit_source": "",
        "best_multiplier": "",
        "best_currency_context": "",
        "best_location_kind": "",
        "best_snippet": "",
    }

    if winner_row is not None:
        out.update(
            {
                "best_rel_error": winner_row.get("best_rel_error", ""),
                "best_normalized_value": winner_row.get("best_normalized_value", ""),
                "best_raw_number": winner_row.get("best_raw_number", ""),
                "best_page": winner_row.get("best_page", ""),
                "best_line_index": winner_row.get("best_line_index", ""),
                "best_alias": winner_row.get("best_alias", ""),
                "best_unit_source": winner_row.get("best_unit_source", ""),
                "best_multiplier": winner_row.get("best_multiplier", ""),
                "best_currency_context": winner_row.get("best_currency_context", ""),
                "best_location_kind": winner_row.get("best_location_kind", ""),
                "best_snippet": winner_row.get("best_snippet", ""),
            }
        )

    return out


def render_summary_md(
    run_meta: dict[str, object],
    kpi_year_rows: list[dict[str, object]],
    reason_rows: list[dict[str, object]],
) -> str:
    lines: list[str] = []
    lines.append("# OCR KPI validation pilot summary")
    lines.append("")
    lines.append(f"- OCR root: `{run_meta['ocr_root']}`")
    lines.append(f"- KPI source: `{run_meta['kpis_long_csv']}`")
    lines.append(f"- Reports scanned: **{run_meta['reports_scanned']}**")
    lines.append(f"- Target rows: **{run_meta['targets_processed']}**")
    lines.append(f"- Tolerance: **{run_meta['tolerance']}**")
    lines.append("")

    status_counts = run_meta["status_counts"]
    lines.append("## Overall status")
    lines.append("")
    lines.append(f"- matched: **{status_counts.get('matched', 0)}**")
    lines.append(f"- ambiguous: **{status_counts.get('ambiguous', 0)}**")
    lines.append(f"- unmatched: **{status_counts.get('unmatched', 0)}**")
    lines.append("")

    lines.append("## KPI/year coverage")
    lines.append("")
    lines.append(
        "| KPI | Year | Total | Matched | Ambiguous | Unmatched | Match rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in kpi_year_rows:
        lines.append(
            "| {kpi} | {year} | {total} | {matched} | {ambiguous} | {unmatched} | {match_rate:.1%} |".format(
                **row
            )
        )
    lines.append("")

    lines.append("## Unmatched/Ambiguous reasons")
    lines.append("")
    if reason_rows:
        for row in reason_rows:
            lines.append(f"- {row['reason']}: {row['count']}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--kpis-long", type=Path, default=DEFAULT_KPIS_LONG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--core-kpis",
        default=",".join(CORE_KPI_ALIASES.keys()),
        help="Comma-separated KPI keys for the pilot target table.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative error tolerance for a match (default: 0.01 for +/-1%).",
    )
    parser.add_argument(
        "--edgar-only",
        action="store_true",
        help="Drop target rows that do not have an EDGAR source.",
    )
    parser.add_argument(
        "--max-reports",
        type=int,
        default=0,
        help="Optional cap on reports scanned (0 means no cap).",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=0,
        help="Optional cap on target rows after filtering (0 means no cap).",
    )
    parser.add_argument(
        "--qa-sample-per-bucket",
        type=int,
        default=10,
        help="Rows to sample per QA bucket (matched / not_matched).",
    )
    parser.add_argument("--qa-seed", type=int, default=42)
    parser.add_argument(
        "--reverse-divisors",
        default="1,1000,1000000,1000000000",
        help=(
            "Comma-separated divisors used by reverse matching. For each target, "
            "the script searches OCR numbers close to target/divisor."
        ),
    )
    parser.add_argument(
        "--reverse-literal-tolerance",
        type=float,
        default=0.01,
        help=(
            "Relative tolerance used when comparing OCR literal values against "
            "target/divisor in reverse matching."
        ),
    )
    parser.add_argument(
        "--reverse-alias-window",
        type=int,
        default=2,
        help=(
            "Max line distance used to attach alias evidence to reverse matches "
            "(same page)."
        ),
    )
    parser.add_argument(
        "--reverse-require-alias",
        action="store_true",
        help="Reject reverse candidates that do not have nearby alias evidence.",
    )
    parser.add_argument(
        "--disable-reverse",
        action="store_true",
        help="Run only the original forward matcher (no reverse, no merged outputs).",
    )
    args = parser.parse_args(argv)

    args.root = args.root.expanduser().resolve()
    args.kpis_long = args.kpis_long.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()

    try:
        reverse_divisors = parse_divisors(args.reverse_divisors)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.root.is_dir():
        print(f"error: OCR root not found: {args.root}", file=sys.stderr)
        return 2
    if not args.kpis_long.is_file():
        print(f"error: KPI long CSV not found: {args.kpis_long}", file=sys.stderr)
        return 2

    selected_kpis = [k.strip() for k in args.core_kpis.split(",") if k.strip()]
    unknown = [k for k in selected_kpis if k not in CORE_KPI_ALIASES]
    if unknown:
        print(
            f"error: unknown KPI aliases for: {unknown}. "
            f"Known keys: {sorted(CORE_KPI_ALIASES)}",
            file=sys.stderr,
        )
        return 2

    targets = load_targets(args.kpis_long, selected_kpis, edgar_only=args.edgar_only)
    if not targets:
        print("No targets after filtering.", file=sys.stderr)
        return 1
    if args.max_targets > 0:
        targets = targets[: args.max_targets]

    targets_by_pair: dict[tuple[str, int], list[TargetRow]] = defaultdict(list)
    for t in targets:
        targets_by_pair[(t.ticker, t.year)].append(t)

    reports = discover_reports(args.root)
    reports = [r for r in reports if (r.ticker, r.year) in targets_by_pair]
    if args.max_reports > 0:
        reports = reports[: args.max_reports]

    if not reports:
        print(
            "No reports matched target (ticker, year) pairs in the selected OCR root.",
            file=sys.stderr,
        )
        return 1

    alias_patterns = compile_alias_patterns(CORE_KPI_ALIASES, selected_kpis)

    target_rows_for_output = [
        {
            "ticker": t.ticker,
            "year": t.year,
            "kpi": t.kpi,
            "target_value": t.value,
            "target_source": t.source,
            "target_tag": t.tag,
            "company_name": t.company_name,
            "exchange": t.exchange,
            "industry": t.industry,
        }
        for t in targets
    ]

    audit_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    reverse_audit_rows: list[dict[str, object]] = []
    reverse_candidate_rows: list[dict[str, object]] = []
    merged_audit_rows: list[dict[str, object]] = []

    for report in reports:
        report_targets = targets_by_pair[(report.ticker, report.year)]
        by_kpi = {t.kpi: t for t in report_targets}
        text = normalize_text(
            report.mmd_path.read_text(encoding="utf-8", errors="replace")
        )
        pages = PAGE_SPLIT.split(text)
        lines_by_page = [
            [normalize_line(ln) for ln in page.splitlines()] for page in pages
        ]

        # Document-level defaults are taken from the opening pages first.
        doc_hints = detect_unit_hints("\n".join(pages[:4]))
        if not doc_hints:
            doc_hints = detect_unit_hints(text)

        alias_seen: dict[str, bool] = {kpi: False for kpi in by_kpi}
        candidates_by_kpi: dict[str, list[Candidate]] = {kpi: [] for kpi in by_kpi}

        seen_candidate_keys: set[tuple] = set()

        for page_idx, page in enumerate(pages):
            page_hints = detect_unit_hints(page)
            lines = lines_by_page[page_idx]
            for line_idx, line in enumerate(lines):
                if not line:
                    continue

                for kpi, target in by_kpi.items():
                    alias_hit: str | None = None
                    for alias, pat in alias_patterns[kpi]:
                        if pat.search(line):
                            alias_hit = alias
                            break
                    if alias_hit is None:
                        continue

                    alias_seen[kpi] = True

                    # Try same line, then neighboring lines for table splits.
                    for offset in (0, 1, -1):
                        probe_idx = line_idx + offset
                        if probe_idx < 0 or probe_idx >= len(lines):
                            continue
                        probe_line = lines[probe_idx]
                        if not probe_line:
                            continue

                        nums = parse_numbers_from_line(probe_line)
                        if not nums:
                            continue

                        window_lo = max(0, probe_idx - 1)
                        window_hi = min(len(lines), probe_idx + 2)
                        line_hints = detect_unit_hints(
                            " ".join(lines[window_lo:window_hi])
                        )

                        for num in nums:
                            key = (
                                report.name,
                                kpi,
                                page_idx,
                                probe_idx,
                                num.start,
                                num.end,
                                num.raw,
                            )
                            if key in seen_candidate_keys:
                                continue
                            seen_candidate_keys.add(key)

                            chosen_mult, unit_source, unit_conflict = (
                                resolve_multiplier(
                                    num.inline_multiplier,
                                    line_hints,
                                    page_hints,
                                    doc_hints,
                                )
                            )
                            currency = classify_currency_context(
                                probe_line, num.start, num.end
                            )
                            normalized = num.value * chosen_mult
                            location_kind = (
                                "table"
                                if "<td" in probe_line.lower()
                                or "</tr>" in probe_line.lower()
                                else "narrative"
                            )

                            candidate = Candidate(
                                report_name=report.name,
                                ticker=report.ticker,
                                year=report.year,
                                kpi=kpi,
                                target_value=target.value,
                                alias=alias_hit,
                                page=page_idx,
                                line_index=probe_idx,
                                alias_line_offset=offset,
                                location_kind=location_kind,
                                raw_number=num.raw,
                                parsed_value=num.value,
                                inline_multiplier=num.inline_multiplier,
                                chosen_multiplier=chosen_mult,
                                unit_source=unit_source,
                                unit_conflict=unit_conflict,
                                currency_context=currency,
                                normalized_value=normalized,
                                snippet=compact_snippet(probe_line),
                            )
                            if currency == "non_usd":
                                candidate.rejected_reason = "non-usd-context"

                            candidates_by_kpi[kpi].append(candidate)

        for kpi, target in by_kpi.items():
            target_candidates = candidates_by_kpi[kpi]
            audit_row, _ = evaluate_target(
                target,
                alias_seen[kpi],
                target_candidates,
                args.tolerance,
            )
            if not audit_row.get("report_name"):
                audit_row["report_name"] = report.name
            audit_rows.append(audit_row)

            reverse_candidates: list[Candidate] = []
            reverse_seen_keys: set[tuple] = set()
            if not args.disable_reverse:
                patterns = alias_patterns[kpi]
                for page_idx, lines in enumerate(lines_by_page):
                    for line_idx, line in enumerate(lines):
                        if not line:
                            continue
                        nums = parse_numbers_from_line(line)
                        if not nums:
                            continue

                        alias_near, alias_dist = find_alias_near_line(
                            line_idx,
                            lines,
                            patterns,
                            max_distance=max(args.reverse_alias_window, 0),
                        )
                        if args.reverse_require_alias and not alias_near:
                            continue

                        location_kind = (
                            "table"
                            if "<td" in line.lower() or "</tr>" in line.lower()
                            else "narrative"
                        )
                        for num in nums:
                            currency = classify_currency_context(
                                line, num.start, num.end
                            )
                            for divisor in reverse_divisors:
                                literal_target = target.value / divisor
                                literal_err = rel_error(num.value, literal_target)
                                if literal_err > args.reverse_literal_tolerance:
                                    continue

                                key = (
                                    page_idx,
                                    line_idx,
                                    num.start,
                                    num.end,
                                    num.raw,
                                    divisor,
                                )
                                if key in reverse_seen_keys:
                                    continue
                                reverse_seen_keys.add(key)

                                divisor_label = (
                                    str(int(divisor))
                                    if float(divisor).is_integer()
                                    else str(divisor)
                                )
                                candidate = Candidate(
                                    report_name=report.name,
                                    ticker=report.ticker,
                                    year=report.year,
                                    kpi=kpi,
                                    target_value=target.value,
                                    alias=alias_near,
                                    page=page_idx,
                                    line_index=line_idx,
                                    alias_line_offset=alias_dist,
                                    location_kind=location_kind,
                                    raw_number=num.raw,
                                    parsed_value=num.value,
                                    inline_multiplier=num.inline_multiplier,
                                    chosen_multiplier=divisor,
                                    unit_source=f"reverse/{divisor_label}",
                                    unit_conflict=False,
                                    currency_context=currency,
                                    normalized_value=num.value * divisor,
                                    snippet=compact_snippet(line),
                                )
                                if currency == "non_usd":
                                    candidate.rejected_reason = "non-usd-context"

                                reverse_candidates.append(candidate)

            if args.disable_reverse:
                reverse_row: dict[str, object] = {
                    "report_name": report.name,
                    "ticker": target.ticker,
                    "year": target.year,
                    "kpi": target.kpi,
                    "company_name": target.company_name,
                    "target_source": target.source,
                    "target_tag": target.tag,
                    "target_value": target.value,
                    "status": "unmatched",
                    "reason": "reverse-disabled",
                    "candidate_count": 0,
                    "valid_candidate_count": 0,
                    "hit_count": 0,
                    "best_rel_error": "",
                    "best_normalized_value": "",
                    "best_raw_number": "",
                    "best_page": "",
                    "best_line_index": "",
                    "best_alias": "",
                    "best_unit_source": "",
                    "best_multiplier": "",
                    "best_currency_context": "",
                    "best_location_kind": "",
                    "best_snippet": "",
                }
            else:
                reverse_row, _ = evaluate_reverse_target(
                    target,
                    reverse_candidates,
                    args.tolerance,
                )
                if not reverse_row.get("report_name"):
                    reverse_row["report_name"] = report.name

            reverse_audit_rows.append(reverse_row)
            merged_audit_rows.append(merge_audit_rows(audit_row, reverse_row))

            for cand in target_candidates:
                candidate_rows.append(
                    {
                        "report_name": cand.report_name,
                        "ticker": cand.ticker,
                        "year": cand.year,
                        "kpi": cand.kpi,
                        "target_value": cand.target_value,
                        "alias": cand.alias,
                        "page": cand.page,
                        "line_index": cand.line_index,
                        "alias_line_offset": cand.alias_line_offset,
                        "location_kind": cand.location_kind,
                        "raw_number": cand.raw_number,
                        "parsed_value": cand.parsed_value,
                        "inline_multiplier": cand.inline_multiplier,
                        "chosen_multiplier": cand.chosen_multiplier,
                        "unit_source": cand.unit_source,
                        "unit_conflict": cand.unit_conflict,
                        "currency_context": cand.currency_context,
                        "normalized_value": cand.normalized_value,
                        "rel_error": cand.rel_error,
                        "rejected_reason": cand.rejected_reason,
                        "snippet": cand.snippet,
                    }
                )

            for cand in reverse_candidates:
                reverse_candidate_rows.append(
                    {
                        "report_name": cand.report_name,
                        "ticker": cand.ticker,
                        "year": cand.year,
                        "kpi": cand.kpi,
                        "target_value": cand.target_value,
                        "alias": cand.alias,
                        "page": cand.page,
                        "line_index": cand.line_index,
                        "alias_line_offset": cand.alias_line_offset,
                        "location_kind": cand.location_kind,
                        "raw_number": cand.raw_number,
                        "parsed_value": cand.parsed_value,
                        "inline_multiplier": cand.inline_multiplier,
                        "chosen_multiplier": cand.chosen_multiplier,
                        "unit_source": cand.unit_source,
                        "unit_conflict": cand.unit_conflict,
                        "currency_context": cand.currency_context,
                        "normalized_value": cand.normalized_value,
                        "rel_error": cand.rel_error,
                        "rejected_reason": cand.rejected_reason,
                        "snippet": cand.snippet,
                    }
                )

    kpi_year_rows, reason_rows, ticker_rows = build_summary_tables(audit_rows)
    reverse_kpi_year_rows, reverse_reason_rows, reverse_ticker_rows = (
        build_summary_tables(reverse_audit_rows)
    )
    merged_kpi_year_rows, merged_reason_rows, merged_ticker_rows = build_summary_tables(
        merged_audit_rows
    )
    qa_rows = build_manual_qa_sample(
        audit_rows, args.qa_sample_per_bucket, args.qa_seed
    )
    reverse_qa_rows = build_manual_qa_sample(
        reverse_audit_rows, args.qa_sample_per_bucket, args.qa_seed
    )
    merged_qa_rows = build_manual_qa_sample(
        merged_audit_rows, args.qa_sample_per_bucket, args.qa_seed
    )

    status_counts = Counter(str(r["status"]) for r in audit_rows)
    reverse_status_counts = Counter(str(r["status"]) for r in reverse_audit_rows)
    merged_status_counts = Counter(str(r["status"]) for r in merged_audit_rows)
    run_meta = {
        "ocr_root": str(args.root),
        "kpis_long_csv": str(args.kpis_long),
        "out_dir": str(args.out_dir),
        "selected_kpis": selected_kpis,
        "tolerance": args.tolerance,
        "reverse_divisors": reverse_divisors,
        "reverse_literal_tolerance": args.reverse_literal_tolerance,
        "reverse_alias_window": args.reverse_alias_window,
        "reverse_require_alias": args.reverse_require_alias,
        "reverse_disabled": args.disable_reverse,
        "edgar_only": args.edgar_only,
        "reports_scanned": len(reports),
        "targets_processed": len(audit_rows),
        "status_counts_forward": dict(status_counts),
        "status_counts_reverse": dict(reverse_status_counts),
        "status_counts_merged": dict(merged_status_counts),
        "status_counts": dict(merged_status_counts),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        args.out_dir / "targets_pilot.csv",
        [
            "ticker",
            "year",
            "kpi",
            "target_value",
            "target_source",
            "target_tag",
            "company_name",
            "exchange",
            "industry",
        ],
        target_rows_for_output,
    )
    write_csv(
        args.out_dir / "audit_rows.csv",
        [
            "report_name",
            "ticker",
            "year",
            "kpi",
            "company_name",
            "target_source",
            "target_tag",
            "target_value",
            "status",
            "reason",
            "candidate_count",
            "valid_candidate_count",
            "hit_count",
            "best_rel_error",
            "best_normalized_value",
            "best_raw_number",
            "best_page",
            "best_line_index",
            "best_alias",
            "best_unit_source",
            "best_multiplier",
            "best_currency_context",
            "best_location_kind",
            "best_snippet",
        ],
        audit_rows,
    )
    write_csv(
        args.out_dir / "candidates.csv",
        [
            "report_name",
            "ticker",
            "year",
            "kpi",
            "target_value",
            "alias",
            "page",
            "line_index",
            "alias_line_offset",
            "location_kind",
            "raw_number",
            "parsed_value",
            "inline_multiplier",
            "chosen_multiplier",
            "unit_source",
            "unit_conflict",
            "currency_context",
            "normalized_value",
            "rel_error",
            "rejected_reason",
            "snippet",
        ],
        candidate_rows,
    )
    write_csv(
        args.out_dir / "coverage_kpi_year.csv",
        ["kpi", "year", "total", "matched", "ambiguous", "unmatched", "match_rate"],
        kpi_year_rows,
    )
    write_csv(
        args.out_dir / "diagnostics_reasons.csv",
        ["reason", "count"],
        reason_rows,
    )
    write_csv(
        args.out_dir / "company_failures.csv",
        [
            "ticker",
            "total",
            "matched",
            "ambiguous",
            "unmatched",
            "match_rate",
            "top_failure_reason",
        ],
        ticker_rows,
    )
    write_csv(
        args.out_dir / "manual_qa_sample.csv",
        [
            "qa_bucket",
            "report_name",
            "ticker",
            "year",
            "kpi",
            "company_name",
            "target_source",
            "target_tag",
            "target_value",
            "status",
            "reason",
            "candidate_count",
            "valid_candidate_count",
            "hit_count",
            "best_rel_error",
            "best_normalized_value",
            "best_raw_number",
            "best_page",
            "best_line_index",
            "best_alias",
            "best_unit_source",
            "best_multiplier",
            "best_currency_context",
            "best_location_kind",
            "best_snippet",
        ],
        qa_rows,
    )
    write_csv(
        args.out_dir / "reverse_audit_rows.csv",
        [
            "report_name",
            "ticker",
            "year",
            "kpi",
            "company_name",
            "target_source",
            "target_tag",
            "target_value",
            "status",
            "reason",
            "candidate_count",
            "valid_candidate_count",
            "hit_count",
            "best_rel_error",
            "best_normalized_value",
            "best_raw_number",
            "best_page",
            "best_line_index",
            "best_alias",
            "best_unit_source",
            "best_multiplier",
            "best_currency_context",
            "best_location_kind",
            "best_snippet",
        ],
        reverse_audit_rows,
    )
    write_csv(
        args.out_dir / "reverse_candidates.csv",
        [
            "report_name",
            "ticker",
            "year",
            "kpi",
            "target_value",
            "alias",
            "page",
            "line_index",
            "alias_line_offset",
            "location_kind",
            "raw_number",
            "parsed_value",
            "inline_multiplier",
            "chosen_multiplier",
            "unit_source",
            "unit_conflict",
            "currency_context",
            "normalized_value",
            "rel_error",
            "rejected_reason",
            "snippet",
        ],
        reverse_candidate_rows,
    )
    write_csv(
        args.out_dir / "coverage_kpi_year_reverse.csv",
        ["kpi", "year", "total", "matched", "ambiguous", "unmatched", "match_rate"],
        reverse_kpi_year_rows,
    )
    write_csv(
        args.out_dir / "diagnostics_reasons_reverse.csv",
        ["reason", "count"],
        reverse_reason_rows,
    )
    write_csv(
        args.out_dir / "company_failures_reverse.csv",
        [
            "ticker",
            "total",
            "matched",
            "ambiguous",
            "unmatched",
            "match_rate",
            "top_failure_reason",
        ],
        reverse_ticker_rows,
    )
    write_csv(
        args.out_dir / "manual_qa_sample_reverse.csv",
        [
            "qa_bucket",
            "report_name",
            "ticker",
            "year",
            "kpi",
            "company_name",
            "target_source",
            "target_tag",
            "target_value",
            "status",
            "reason",
            "candidate_count",
            "valid_candidate_count",
            "hit_count",
            "best_rel_error",
            "best_normalized_value",
            "best_raw_number",
            "best_page",
            "best_line_index",
            "best_alias",
            "best_unit_source",
            "best_multiplier",
            "best_currency_context",
            "best_location_kind",
            "best_snippet",
        ],
        reverse_qa_rows,
    )
    write_csv(
        args.out_dir / "merged_audit_rows.csv",
        [
            "report_name",
            "ticker",
            "year",
            "kpi",
            "company_name",
            "target_source",
            "target_tag",
            "target_value",
            "status",
            "reason",
            "winner_pipeline",
            "forward_status",
            "forward_reason",
            "reverse_status",
            "reverse_reason",
            "candidate_count",
            "valid_candidate_count",
            "hit_count",
            "forward_candidate_count",
            "reverse_candidate_count",
            "forward_hit_count",
            "reverse_hit_count",
            "best_rel_error",
            "best_normalized_value",
            "best_raw_number",
            "best_page",
            "best_line_index",
            "best_alias",
            "best_unit_source",
            "best_multiplier",
            "best_currency_context",
            "best_location_kind",
            "best_snippet",
        ],
        merged_audit_rows,
    )
    write_csv(
        args.out_dir / "coverage_kpi_year_merged.csv",
        ["kpi", "year", "total", "matched", "ambiguous", "unmatched", "match_rate"],
        merged_kpi_year_rows,
    )
    write_csv(
        args.out_dir / "diagnostics_reasons_merged.csv",
        ["reason", "count"],
        merged_reason_rows,
    )
    write_csv(
        args.out_dir / "company_failures_merged.csv",
        [
            "ticker",
            "total",
            "matched",
            "ambiguous",
            "unmatched",
            "match_rate",
            "top_failure_reason",
        ],
        merged_ticker_rows,
    )
    write_csv(
        args.out_dir / "manual_qa_sample_merged.csv",
        [
            "qa_bucket",
            "report_name",
            "ticker",
            "year",
            "kpi",
            "company_name",
            "target_source",
            "target_tag",
            "target_value",
            "status",
            "reason",
            "winner_pipeline",
            "forward_status",
            "forward_reason",
            "reverse_status",
            "reverse_reason",
            "candidate_count",
            "valid_candidate_count",
            "hit_count",
            "forward_candidate_count",
            "reverse_candidate_count",
            "forward_hit_count",
            "reverse_hit_count",
            "best_rel_error",
            "best_normalized_value",
            "best_raw_number",
            "best_page",
            "best_line_index",
            "best_alias",
            "best_unit_source",
            "best_multiplier",
            "best_currency_context",
            "best_location_kind",
            "best_snippet",
        ],
        merged_qa_rows,
    )

    (args.out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2),
        encoding="utf-8",
    )
    forward_meta = {**run_meta, "status_counts": dict(status_counts)}
    reverse_meta = {**run_meta, "status_counts": dict(reverse_status_counts)}
    merged_meta = {**run_meta, "status_counts": dict(merged_status_counts)}
    (args.out_dir / "summary_forward.md").write_text(
        render_summary_md(forward_meta, kpi_year_rows, reason_rows),
        encoding="utf-8",
    )
    (args.out_dir / "summary_reverse.md").write_text(
        render_summary_md(reverse_meta, reverse_kpi_year_rows, reverse_reason_rows),
        encoding="utf-8",
    )
    (args.out_dir / "summary.md").write_text(
        render_summary_md(merged_meta, merged_kpi_year_rows, merged_reason_rows),
        encoding="utf-8",
    )

    print(
        f"Processed {len(reports)} reports and {len(audit_rows)} target rows.\n"
        f"Forward: matched={status_counts.get('matched', 0)} "
        f"ambiguous={status_counts.get('ambiguous', 0)} "
        f"unmatched={status_counts.get('unmatched', 0)}\n"
        f"Reverse: matched={reverse_status_counts.get('matched', 0)} "
        f"ambiguous={reverse_status_counts.get('ambiguous', 0)} "
        f"unmatched={reverse_status_counts.get('unmatched', 0)}\n"
        f"Merged: matched={merged_status_counts.get('matched', 0)} "
        f"ambiguous={merged_status_counts.get('ambiguous', 0)} "
        f"unmatched={merged_status_counts.get('unmatched', 0)}"
    )
    print(f"Wrote outputs to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
