"""LLM-as-a-annotator for TREC qrels relevance judgment.

Re-validates regex-matched candidate pages from ``review_candidates.csv``
using an LLM served via an OpenAI-compatible endpoint (e.g. vLLM).  For each
candidate the LLM receives the KPI name, its ground-truth value, and the
single candidate page, then returns a graded relevance judgment (0/1/2) with
a short justification.

Output:
- ``qrels_llm.txt``          — TREC-format qrels with graded relevance (0/1/2)
- ``annotations_audit.csv``  — per-candidate detail with LLM decision
- ``review_flagged.csv``     — high-confidence regex matches where LLM grade < 2
- ``annotations_summary.md`` — agreement stats

Usage:

    uv run python KPI_analysis/llm_annotate_qrels.py \\
        --model Qwen/Qwen3.6-27B-FP8
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

try:
    from openai import APIError, OpenAI
except ImportError:
    sys.stderr.write("openai SDK not installed. Run: uv add openai\n")
    raise

try:
    from pydantic import ValidationError
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(HERE / "llm_benchmark"))

from document import find_mmd, parse_report_name, split_pages  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_OCR_ROOT = REPO_ROOT / "DeepSeekOCR_Ardian_pruned_1k"
DEFAULT_REVIEW_CANDIDATES = HERE / "output" / "qrels" / "review_candidates.csv"
DEFAULT_OUTPUT_DIR = HERE / "output" / "qrels"
DEFAULT_ALIASES = HERE / "kpi_aliases.json"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class PageRelevance(BaseModel):
    """Graded relevance judgment for a single candidate page."""

    relevance: int = Field(
        description=(
            "0 = not relevant: the page does not state the target KPI value "
            "for the target fiscal year, or the match is coincidental. "
            "1 = contextual mention: the KPI concept appears and a value is "
            "nearby, but it may be for a different year, a different entity, "
            "a comparative restatement, or a rounded/approximate figure. "
            "2 = primary source: the page directly states or derives the "
            "exact target KPI value for the target fiscal year in a "
            "financial statement, table, or explicit narrative sentence."
        ),
        ge=0,
        le=2,
    )
    reasoning: str = Field(
        description="1-2 sentence justification for the decision.",
        max_length=500,
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a financial-document relevance judge.

You will receive:
- A KPI name (e.g. "revenue") and its known aliases
- A ground-truth value for a specific company and fiscal year
- A single page from that company's annual report (OCR text)

Your task: assign a relevance grade (0, 1, or 2) indicating whether this page \
states or derives the target KPI value for the target fiscal year.

## Grading scale

- **2 — Primary source.** The page directly reports the target KPI value for \
the target fiscal year in a financial statement (income statement, balance \
sheet, cash-flow statement), a data table, or an explicit narrative sentence. \
The value matches the target (possibly after unit scaling).
- **1 — Contextual mention.** The KPI concept appears and a value is nearby, \
but one or more of: (a) the value is for a DIFFERENT fiscal year (e.g. a \
comparative restatement of a prior year), (b) the value is for a subsidiary \
or segment rather than the consolidated entity, (c) the value is approximate \
or rounded, (d) the KPI is mentioned in prose without a specific figure, or \
(e) the match is in a footnote, risk factor, or discussion rather than a \
primary financial statement.
- **0 — Not relevant.** The page does not mention the target KPI, or the \
numeric match is purely coincidental (the same number appears in an unrelated \
context).

## Rules

1. **Unit scaling.** Financial statements often report "in thousands" or "in \
millions". A value of "9,709" on a page headed "(in thousands)" means \
$9,709,000. That IS the target value $9,709,003,000 (within 0.1%). Always \
check the unit header before comparing.

2. **Multi-year / comparative tables.** Most annual reports show 2–3 fiscal \
years side by side for comparison. ONLY the column for the TARGET fiscal year \
is grade 2. Values in prior-year columns are grade 1 at best (they restate \
the KPI for a different year). Identify the target year from the "report \
year" field in the prompt.

3. **Fiscal year convention.** Some companies (especially US retailers like \
Advance Auto Parts, Costco, AutoZone) use 52/53-week fiscal years ending in \
early January. For these filers, a period ending "January 1, 2022" is fiscal \
year 2021 — the filer's own label, not the calendar year. The report_year in \
the prompt uses the filer's fiscal year label. When the page shows "Year \
Ended January 1, 2022" and the target is report_year=2021, that IS the target \
year.

4. **Scope distinctions.** Several KPIs have parent-only vs consolidated \
variants:
   - "Net income" in the KPI data means attributable to parent (excluding \
non-controlling interest). If the page shows consolidated net income \
including NCI, that is a different value — grade 1, not 2.
   - "Stockholders' equity" is parent-only; equity including NCI is a \
separate KPI.
   - "Cash and equivalents" is unrestricted only; cash including restricted \
is a separate KPI.

5. **Different phrasing.** The page may use different wording for the KPI \
(e.g. "net sales" for revenue, "capital expenditure" for capex). All known \
aliases are listed. Matching an alias is expected.

Respond with a JSON object: {"relevance": 0/1/2, "reasoning": "brief \
justification"}."""


def _format_target(value: float) -> str:
    """Human-readable scaling of a raw-dollar value."""
    av = abs(value)
    sign = "-" if value < 0 else ""
    if av >= 1e9:
        return f"{sign}${av / 1e9:,.1f} billion"
    if av >= 1e6:
        return f"{sign}${av / 1e6:,.1f} million"
    if av >= 1e3:
        return f"{sign}${av / 1e3:,.1f} thousand"
    return f"{sign}${av:,.0f}"


def build_user_message(
    *,
    ticker: str,
    report_name: str,
    report_year: int,
    kpi: str,
    target_value: float,
    aliases: list[str],
    match_type: str,
    alias_matched: str,
    raw_value: str,
    page_idx: int,
    page_text: str,
) -> str:
    target_fmt = _format_target(target_value)
    aliases_csv = ", ".join(aliases) if aliases else "(none)"
    return (
        f"Ticker: {ticker}\n"
        f"Report: {report_name} (year {report_year})\n"
        f"KPI: {kpi}\n"
        f"Target value: {target_fmt} (single dollars: {target_value:,.0f})\n"
        f"Known aliases: {aliases_csv}\n"
        f"Regex match info: match_type={match_type}, "
        f"alias={alias_matched}, raw_value={raw_value}\n\n"
        f"--- Page {page_idx + 1} (from {report_name}) ---\n"
        f"{page_text}\n"
        f"--- End of page ---"
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw).strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    s = _strip_fences(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in response: {raw!r}")
    return json.loads(s[start : end + 1])


@dataclass
class RelevanceResult:
    relevance: PageRelevance | None
    raw_response: str
    attempts: int
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


def call_relevance(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    retries: int = 3,
) -> RelevanceResult:
    schema_dict = PageRelevance.model_json_schema()
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "PageRelevance",
            "schema": schema_dict,
            "strict": True,
        },
    }
    extra_body: dict[str, Any] = {
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error: str | None = None
    last_raw = ""
    last_prompt_tokens: int | None = None
    last_completion_tokens: int | None = None
    started = time.monotonic()

    for attempt in range(1, retries + 1):
        attempt_temp = temperature + 0.2 * (attempt - 1)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=attempt_temp,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format=response_format,
                extra_body=extra_body,
            )
        except APIError as e:
            last_error = f"api_error: {e}"
            time.sleep(min(2**attempt, 10))
            continue

        last_raw = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if usage is not None:
            last_prompt_tokens = getattr(usage, "prompt_tokens", None)
            last_completion_tokens = getattr(usage, "completion_tokens", None)

        try:
            obj = _parse_json_object(last_raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = f"json_parse_error: {e}"
            continue

        try:
            relevance = PageRelevance.model_validate(obj)
        except ValidationError as e:
            last_error = f"schema_validation_error: {e}"
            continue

        return RelevanceResult(
            relevance=relevance,
            raw_response=last_raw,
            attempts=attempt,
            latency_s=time.monotonic() - started,
            prompt_tokens=last_prompt_tokens,
            completion_tokens=last_completion_tokens,
        )

    return RelevanceResult(
        relevance=None,
        raw_response=last_raw,
        attempts=retries,
        latency_s=time.monotonic() - started,
        prompt_tokens=last_prompt_tokens,
        completion_tokens=last_completion_tokens,
        error=last_error or "unknown_error",
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class CandidateRow:
    query_id: str
    doc_id: str
    report_name: str
    report_year: int
    page_idx: int  # 0-indexed
    match_type: str
    alias_matched: str
    raw_value: str
    normalized_value: float
    rel_error: float
    unit_source: str
    snippet: str


def load_review_candidates(csv_path: Path) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                page_idx = int(r["page_idx"])
                report_year = int(r["report_year"])
                normalized_value = float(r["normalized_value"])
                rel_error_str = r.get("rel_error", "").strip()
                rel_error = float(rel_error_str) if rel_error_str else 0.0
            except (ValueError, KeyError):
                continue
            rows.append(
                CandidateRow(
                    query_id=r["query_id"],
                    doc_id=r["doc_id"],
                    report_name=r["report_name"],
                    report_year=report_year,
                    page_idx=page_idx,
                    match_type=r.get("match_type", ""),
                    alias_matched=r.get("alias_matched", ""),
                    raw_value=r.get("raw_value", ""),
                    normalized_value=normalized_value,
                    rel_error=rel_error,
                    unit_source=r.get("unit_source", ""),
                    snippet=r.get("snippet", ""),
                )
            )
    return rows


def load_kpi_aliases(json_path: Path) -> dict[str, list[str]]:
    return json.loads(json_path.read_text())


# ---------------------------------------------------------------------------
# Report discovery (mirrors llm_benchmark/document.py)
# ---------------------------------------------------------------------------


REPORT_NAME_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})(?:_[0-9a-fA-F]+)?$")


def discover_report_dirs(root: Path) -> dict[str, Path]:
    """Map report_name -> directory path for all .mmd-containing dirs."""
    out: dict[str, Path] = {}
    for mmd in root.rglob("*.mmd"):
        d = mmd.parent
        if d.name in out:
            continue
        m = REPORT_NAME_RE.match(d.name)
        if m is None:
            continue
        out[d.name] = d
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

QRELS_FIELDS = [
    "query_id",
    "doc_id",
    "report_name",
    "report_year",
    "page_idx",
    "kpi",
    "target_value",
    "regex_match_type",
    "regex_alias_matched",
    "regex_raw_value",
    "regex_rel_error",
    "llm_grade",
    "llm_reasoning",
    "latency_s",
    "prompt_tokens",
    "completion_tokens",
]


def _extract_kpi_from_query_id(query_id: str) -> str:
    """Parse KPI key from query_id like 'AAP_revenue_2019' -> 'revenue'."""
    parts = query_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:-1])
    return ""


def _extract_year_from_query_id(query_id: str) -> int:
    """Parse year from query_id like 'AAP_revenue_2019' -> 2019."""
    parts = query_id.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def _extract_ticker_from_query_id(query_id: str) -> str:
    """Parse ticker from query_id like 'AAP_revenue_2019' -> 'AAP'."""
    return query_id.split("_")[0] if "_" in query_id else query_id


def write_audit_row(
    writer: csv.DictWriter,
    *,
    candidate: CandidateRow,
    kpi: str,
    target_value: float,
    result: RelevanceResult,
) -> None:
    row = {
        "query_id": candidate.query_id,
        "doc_id": candidate.doc_id,
        "report_name": candidate.report_name,
        "report_year": candidate.report_year,
        "page_idx": candidate.page_idx,
        "kpi": kpi,
        "target_value": target_value,
        "regex_match_type": candidate.match_type,
        "regex_alias_matched": candidate.alias_matched,
        "regex_raw_value": candidate.raw_value,
        "regex_rel_error": f"{candidate.rel_error:.6f}" if candidate.rel_error else "",
        "llm_grade": result.relevance.relevance if result.relevance else None,
        "llm_reasoning": (
            result.relevance.reasoning if result.relevance else (result.error or "")
        ),
        "latency_s": f"{result.latency_s:.2f}",
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
    writer.writerow(row)


def write_qrels_llm(
    annotations: list[tuple[str, str, int]], path: Path
) -> int:
    """Write TREC-format qrels with graded relevance (0/1/2). Returns line count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for query_id, doc_id, grade in annotations:
        key = (query_id, doc_id)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{query_id}\t0\t{doc_id}\t{grade}\n")
    with path.open("w") as f:
        f.writelines(sorted(lines))
    return len(lines)


def write_review_flagged(
    flagged: list[dict[str, Any]], path: Path
) -> int:
    """Write the manual-review queue. Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = QRELS_FIELDS + ["flag_reason"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in flagged:
            w.writerow(row)
    return len(flagged)


def write_summary(
    *,
    total: int,
    n_relevant: int,
    n_flagged: int,
    by_match_type: dict[str, dict[str, int]],
    by_kpi: dict[str, dict[str, int]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# LLM annotation summary\n")
    lines.append(f"- Total candidates annotated: {total}")
    lines.append(f"- Grade 2 (primary source): {n_relevant}")
    lines.append(f"- Grade 1 (contextual mention): {sum(d.get('grade_1', 0) for d in by_match_type.values())}")
    lines.append(f"- Grade 0 (not relevant): {sum(d.get('grade_0', 0) for d in by_match_type.values())}")
    lines.append(f"- Flagged for manual review: {n_flagged}")
    lines.append("")

    lines.append("## Agreement: regex match_type × LLM grade\n")
    lines.append("| match_type | grade_0 | grade_1 | grade_2 | total |")
    lines.append("| --- | --- | --- | --- | --- |")
    for mt in sorted(by_match_type):
        d = by_match_type[mt]
        g0 = d.get("grade_0", 0)
        g1 = d.get("grade_1", 0)
        g2 = d.get("grade_2", 0)
        lines.append(f"| {mt} | {g0} | {g1} | {g2} | {g0 + g1 + g2} |")
    lines.append("")

    lines.append("## Per-KPI breakdown\n")
    lines.append("| kpi | n_candidates | grade_0 | grade_1 | grade_2 | n_flagged |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for kpi in sorted(by_kpi):
        d = by_kpi[kpi]
        lines.append(
            f"| {kpi} | {d.get('total', 0)} | "
            f"{d.get('grade_0', 0)} | {d.get('grade_1', 0)} | {d.get('grade_2', 0)} | "
            f"{d.get('flagged', 0)} |"
        )
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="vLLM model name.")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    p.add_argument("--review-candidates", type=Path, default=DEFAULT_REVIEW_CANDIDATES)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    p.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help=(
            "Parallel in-flight LLM requests. Default 16 — each relevance "
            "call is ~4k tokens (prompt+page+output), far below vLLM's "
            "max_model_len, so high parallelism is safe and fast."
        ),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "annotations_audit.csv"
    qrels_path = args.output_dir / "qrels_llm.txt"
    flagged_path = args.output_dir / "review_flagged.csv"
    summary_path = args.output_dir / "annotations_summary.md"

    # --- Load data ---
    sys.stderr.write("[load] Loading review candidates...\n")
    candidates = load_review_candidates(args.review_candidates)
    sys.stderr.write(f"[load] {len(candidates)} candidates\n")

    sys.stderr.write("[load] Loading KPI aliases...\n")
    kpi_aliases = load_kpi_aliases(args.aliases)

    # --- Resume: load already-annotated ---
    already_done: set[tuple[str, str]] = set()
    if args.resume and audit_path.is_file():
        with audit_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row.get("query_id", "")
                did = row.get("doc_id", "")
                if qid and did:
                    already_done.add((qid, did))
        sys.stderr.write(
            f"[resume] {len(already_done)} candidates already annotated\n"
        )

    # --- Filter ---
    if already_done:
        candidates = [c for c in candidates if (c.query_id, c.doc_id) not in already_done]
        sys.stderr.write(f"[filter] {len(candidates)} candidates remaining\n")

    if args.limit:
        candidates = candidates[: args.limit]
        sys.stderr.write(f"[limit] Processing at most {len(candidates)} candidates\n")

    if not candidates:
        sys.stderr.write("[done] Nothing to do\n")
        return

    # --- Discover reports ---
    sys.stderr.write("[discover] Scanning OCR reports...\n")
    report_dirs = discover_report_dirs(args.ocr_root)
    sys.stderr.write(f"[discover] {len(report_dirs)} report directories found\n")

    # --- Group candidates by report ---
    by_report: dict[str, list[CandidateRow]] = defaultdict(list)
    for c in candidates:
        by_report[c.report_name].append(c)
    sys.stderr.write(
        f"[group] {len(by_report)} unique reports to load\n"
    )

    # --- LLM setup ---
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    sys.stderr.write(
        f"[llm] model={args.model}, concurrency={args.concurrency}, "
        f"temperature={args.temperature}, max_tokens={args.max_tokens}\n"
    )

    # --- Open audit CSV (append mode for resume) ---
    audit_file = audit_path.open("a", newline="")
    audit_writer = csv.DictWriter(audit_file, fieldnames=QRELS_FIELDS)
    if not already_done or not audit_path.is_file():
        audit_writer.writeheader()

    # --- Process ---
    all_annotations: list[tuple[str, str, int]] = []
    flagged_rows: list[dict[str, Any]] = []
    by_match_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_kpi: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_ok = 0
    n_fail = 0
    n_error = 0
    total_to_process = sum(len(v) for v in by_report.values())

    # --- Pre-load all report pages ---
    sys.stderr.write(f"[preload] Loading pages for {len(by_report)} reports...\n")
    report_pages: dict[str, list[str]] = {}
    for report_name in sorted(by_report):
        if report_name not in report_dirs:
            continue
        report_dir = report_dirs[report_name]
        mmd_path = find_mmd(report_dir)
        if mmd_path is None:
            continue
        raw = mmd_path.read_text(encoding="utf-8", errors="replace")
        report_pages[report_name] = split_pages(raw)
    sys.stderr.write(f"[preload] {len(report_pages)} reports loaded\n")

    # --- Prepare all tasks (messages built upfront) ---
    @dataclass
    class AnnotTask:
        candidate: CandidateRow
        kpi: str
        target_value: float
        messages: list[dict[str, str]]

    tasks: list[AnnotTask] = []
    for report_name, report_candidates in sorted(by_report.items()):
        pages = report_pages.get(report_name)
        if pages is None:
            n_error += len(report_candidates)
            continue

        for cand in report_candidates:
            kpi = _extract_kpi_from_query_id(cand.query_id)
            target_value = cand.normalized_value
            aliases = kpi_aliases.get(kpi, [])

            if cand.page_idx >= len(pages):
                n_error += 1
                continue

            page_text = pages[cand.page_idx]
            user_msg = build_user_message(
                ticker=_extract_ticker_from_query_id(cand.query_id),
                report_name=report_name,
                report_year=cand.report_year,
                kpi=kpi,
                target_value=target_value,
                aliases=aliases,
                match_type=cand.match_type,
                alias_matched=cand.alias_matched,
                raw_value=cand.raw_value,
                page_idx=cand.page_idx,
                page_text=page_text,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            tasks.append(AnnotTask(
                candidate=cand, kpi=kpi,
                target_value=target_value, messages=messages,
            ))

    sys.stderr.write(
        f"[run] {len(tasks)} tasks, concurrency={args.concurrency}\n"
    )

    # --- Parallel LLM calls ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_task(task: AnnotTask) -> tuple[AnnotTask, RelevanceResult]:
        result = call_relevance(
            client,
            model=args.model,
            messages=task.messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            retries=args.retries,
        )
        return task, result

    pbar = tqdm(total=len(tasks), desc="annotating", unit="page")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(_run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            task, result = fut.result()
            cand = task.candidate

            # Write audit row immediately (atomic append)
            write_audit_row(
                audit_writer,
                candidate=cand,
                kpi=task.kpi,
                target_value=task.target_value,
                result=result,
            )
            audit_file.flush()

            grade = 0
            if result.relevance is not None:
                grade = result.relevance.relevance
                n_ok += 1
            else:
                n_error += 1
                pbar.write(
                    f"ERR {cand.doc_id}: {result.error} "
                    f"(after {result.attempts} attempts)"
                )

            all_annotations.append((cand.query_id, cand.doc_id, grade))

            grade_str = f"grade_{grade}"
            by_match_type[cand.match_type][grade_str] += 1
            by_kpi[task.kpi]["total"] += 1
            by_kpi[task.kpi][grade_str] = by_kpi[task.kpi].get(grade_str, 0) + 1

            # Flagging: LLM gave grade 0 or 1 but regex was high-confidence
            # alias+value with tight numeric tolerance and reliable unit source.
            # Only flag when the regex evidence is strong enough that the LLM
            # rejection is likely an error worth human review.
            high_confidence_regex = (
                cand.match_type == "alias+value"
                and cand.rel_error < 0.005
                and cand.unit_source in ("page", "line", "inline")
            )
            if grade < 2 and high_confidence_regex:
                flag_row = {
                    "query_id": cand.query_id,
                    "doc_id": cand.doc_id,
                    "report_name": cand.report_name,
                    "report_year": cand.report_year,
                    "page_idx": cand.page_idx,
                    "kpi": task.kpi,
                    "target_value": task.target_value,
                    "regex_match_type": cand.match_type,
                    "regex_alias_matched": cand.alias_matched,
                    "regex_raw_value": cand.raw_value,
                    "regex_rel_error": f"{cand.rel_error:.6f}",
                    "llm_grade": grade,
                    "llm_reasoning": result.relevance.reasoning
                    if result.relevance
                    else "",
                    "latency_s": f"{result.latency_s:.2f}",
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "flag_reason": (
                        f"LLM gave grade {grade} on high-confidence regex match "
                        f"(alias='{cand.alias_matched}', "
                        f"raw='{cand.raw_value}', "
                        f"rel_error={cand.rel_error:.6f}, "
                        f"unit_source='{cand.unit_source}')"
                    ),
                }
                flagged_rows.append(flag_row)
                by_kpi[task.kpi]["flagged"] = by_kpi[task.kpi].get("flagged", 0) + 1

            pbar.update(1)

    pbar.close()
    audit_file.close()

    # --- Write outputs ---
    sys.stderr.write("\n[write] Writing qrels_llm.txt...\n")
    n_qrels = write_qrels_llm(all_annotations, qrels_path)

    sys.stderr.write("[write] Writing review_flagged.csv...\n")
    n_flagged = write_review_flagged(flagged_rows, flagged_path)

    sys.stderr.write("[write] Writing annotations_summary.md...\n")
    write_summary(
        total=len(all_annotations),
        n_relevant=sum(1 for _, _, g in all_annotations if g == 2),
        n_flagged=n_flagged,
        by_match_type=dict(by_match_type),
        by_kpi=dict(by_kpi),
        path=summary_path,
    )

    sys.stderr.write(f"\n[done] Output in {args.output_dir}/\n")
    sys.stderr.write(f"  qrels_llm.txt:          {n_qrels} lines\n")
    sys.stderr.write(f"  annotations_audit.csv:   {len(all_annotations)} rows\n")
    sys.stderr.write(f"  review_flagged.csv:      {n_flagged} rows\n")
    sys.stderr.write(f"  annotations_summary.md\n")

    n_relevant = sum(1 for _, _, g in all_annotations if g == 2)
    n_contextual = sum(1 for _, _, g in all_annotations if g == 1)
    n_not_relevant = sum(1 for _, _, g in all_annotations if g == 0)
    sys.stderr.write(
        f"\n[stats] Grading distribution: "
        f"grade_2={n_relevant}, grade_1={n_contextual}, grade_0={n_not_relevant} "
        f"(of {len(all_annotations)} annotated)\n"
    )
    sys.stderr.write(
        f"[stats] {n_flagged} high-confidence regex matches with grade < 2 "
        f"(flagged for manual review)\n"
    )


if __name__ == "__main__":
    main()
