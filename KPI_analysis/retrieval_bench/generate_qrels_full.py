"""Generate TREC-format qrels for KPI retrieval tasks — full 6K dataset variant.

Adapted from ``generate_qrels.py`` for the full dataset:
- Reads ground truth from ``kpi_long_full.csv`` (which includes ``company_name``)
- Scans ``DeepSeekOCR_Ardian_full_dataset/`` for OCR reports
- No dependency on ``companies_alt_names.json`` or ``companies.json``
- Industry filtering uses the CSV's ``industry`` column directly

All regex matching, alias scanning, number extraction, and unit normalisation
logic is identical to the original script.

Output:
- ``qrels.txt`` — TREC-format relevance judgments
- ``review_candidates.csv`` — detailed candidate info for annotation review
- ``summary.md`` — per-query statistics

Usage examples:

    # All reports, all KPIs (full run)
    uv run python KPI_analysis/retrieval_bench/generate_qrels_full.py

    # Specific tickers, subset of KPIs
    uv run python KPI_analysis/retrieval_bench/generate_qrels_full.py \\
        --tickers AAP AZO --kpis revenue net_income --years 2017-2022

    # Filter by industry (uses CSV industry column)
    uv run python KPI_analysis/retrieval_bench/generate_qrels_full.py \\
        --industry "Consumer Cyclical / Auto Parts"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
KPI_ANALYSIS = HERE.parent
REPO_ROOT = KPI_ANALYSIS.parent

DEFAULT_OCR_ROOT = REPO_ROOT / "DeepSeekOCR_Ardian_full_dataset"
DEFAULT_KPIS_LONG = KPI_ANALYSIS / "find_more_queries" / "full_6k" / "kpi_long_full.csv"
DEFAULT_ALIASES = KPI_ANALYSIS / "kpi_fetch_and_build" / "kpi_aliases.json"
DEFAULT_QUERIES_DIR = HERE / "queries"
DEFAULT_OUTPUT_DIR = KPI_ANALYSIS / "output" / "qrels_full"

PAGE_SPLIT_RE = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)
REPORT_NAME_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})(?:_[0-9a-fA-F]+)?$")

SMART_MAP = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00b4": "'"}
)

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

UNIT_HINT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\((?:[^)]*?)\b(?:\$|usd|us\$|u\.s\.\$)?\s*(?:in\s+)?"
        r"(?P<unit>thousands?|millions?|billions?)\b[^)]*\)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdollars?\s+in\s+(?P<unit>thousands?|millions?|billions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\ball\s+dollar\s+amounts?\s+(?:are\s+)?(?:stated|presented|shown)\s+"
        r"in\s+(?P<unit>thousands?|millions?|billions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bin\s+(?P<unit>thousands?|millions?|billions?)\b"
        r"(?:\s*,?\s*except\s+per\s+share\s+data)?",
        re.IGNORECASE,
    ),
]

UNIT_SOURCE_RANK = {
    "inline": 0,
    "line": 1,
    "page": 2,
    "document": 3,
    "default": 4,
}

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

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberToken:
    raw: str
    value: float
    start: int
    end: int
    inline_multiplier: float | None


@dataclass
class PageCandidate:
    """A candidate page where the KPI value may appear."""

    report_name: str
    ticker: str
    report_year: int
    page_idx: int  # 0-indexed
    match_type: str  # "alias+value", "value-only", "alias-only"
    alias_matched: str
    raw_value: str
    normalized_value: float
    rel_error: float
    unit_source: str
    snippet: str


@dataclass
class Query:
    query_id: str
    ticker: str
    year: int
    kpi: str
    target_value: float
    company_name: str
    query_text: str


@dataclass
class ParsedLine:
    text: str
    numbers: list[NumberToken]
    unit_hints: list[float]  # hints from the 3-line window around this line
    alias_kpis: set[str] = field(
        default_factory=set
    )  # KPIs with alias hits on this line


@dataclass
class ParsedPage:
    raw_text: str
    lines: list[ParsedLine]
    page_hints: list[float]


@dataclass
class ReportData:
    name: str
    exchange: str
    ticker: str
    year: int
    mmd_path: Path
    pages: list[str] = field(default_factory=list)
    lines_by_page: list[list[str]] = field(default_factory=list)
    doc_hints: list[float] = field(default_factory=list)
    parsed_pages: list[ParsedPage] = field(default_factory=list)
    industry_slug: str = ""


# ---------------------------------------------------------------------------
# Text normalization (reused from validate_ocr_kpis.py)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = text.translate(SMART_MAP)
    text = text.replace("\\(", "(").replace("\\)", ")").replace("\\$", "$")
    text = re.sub(r"[\u2013\u2014\u2212]", "-", text)
    return text


def normalize_line(line: str) -> str:
    line = normalize_text(line)
    line = re.sub(r"(?<!\()\)\s+(?=\d)", "", line)
    return line.strip()


def compact_snippet(line: str, width: int = 220) -> str:
    text = re.sub(r"\s+", " ", line).strip()
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


# ---------------------------------------------------------------------------
# Unit handling (reused from validate_ocr_kpis.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Number extraction (reused from validate_ocr_kpis.py)
# ---------------------------------------------------------------------------


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

        after = line[m.end("raw") : m.end("raw") + 1]
        if after == "%":
            continue

        try:
            value = float(num_text.replace(",", ""))
        except ValueError:
            continue

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


# ---------------------------------------------------------------------------
# Alias compilation (reused from validate_ocr_kpis.py)
# ---------------------------------------------------------------------------


def compile_alias_patterns(
    aliases: dict[str, list[str]],
    selected_kpis: list[str],
) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    out: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for kpi in selected_kpis:
        kpi_aliases = aliases.get(kpi, [])
        pats: list[tuple[str, re.Pattern[str]]] = []
        for alias in kpi_aliases:
            esc = re.escape(alias)
            esc = esc.replace(r"\ ", r"\s+")
            esc = esc.replace(r"\-", r"[-\s]?")
            pat = re.compile(rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])", re.IGNORECASE)
            pats.append((alias, pat))
        out[kpi] = pats
    return out


def build_alias_index(
    alias_patterns: dict[str, list[tuple[str, re.Pattern[str]]]],
) -> dict:
    """Build substring pre-filter set for fast alias scanning."""
    substrings: set[str] = set()
    min_len = 999
    for kpi, pats in alias_patterns.items():
        for alias_text, _pat in pats:
            canonical = alias_text.lower().replace("-", " ")
            canonical = re.sub(r"\s+", " ", canonical).strip()
            substrings.add(canonical)
            if len(canonical) < min_len:
                min_len = len(canonical)
    return {
        "substrings": substrings,
        "min_alias_len": min_len,
    }


def find_alias_kpis(
    line: str,
    alias_index: dict,
    alias_patterns: dict[str, list[tuple[str, re.Pattern[str]]]],
) -> set[str]:
    """Return set of KPI keys that have at least one alias match on the line."""
    if len(line) < 3:
        return set()
    line_canonical = line.lower().replace("-", " ")
    line_canonical = re.sub(r"\s+", " ", line_canonical)
    subs: set[str] = alias_index["substrings"]
    if not any(s in line_canonical for s in subs):
        return set()
    kpis: set[str] = set()
    for kpi, pats in alias_patterns.items():
        for _alias_text, pat in pats:
            if pat.search(line):
                kpis.add(kpi)
                break
    return kpis


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


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


def discover_reports(root: Path) -> list[ReportData]:
    out: list[ReportData] = []
    for report_dir in sorted({mmd.parent for mmd in root.rglob("*.mmd")}):
        parsed = parse_report_name(report_dir.name)
        if parsed is None:
            continue
        mmd = find_mmd(report_dir)
        if mmd is None:
            continue
        exchange, ticker, year = parsed
        out.append(
            ReportData(
                name=report_dir.name,
                exchange=exchange,
                ticker=ticker,
                year=year,
                mmd_path=mmd,
            )
        )
    return out


def load_ground_truth(
    csv_path: Path,
) -> tuple[dict[tuple[str, int, str], dict], dict[str, str], dict[str, str]]:
    """Load ground-truth CSV and also extract ticker->company_name and ticker->industry mappings."""
    gt: dict[tuple[str, int, str], dict] = {}
    ticker_to_name: dict[str, str] = {}
    ticker_to_industry: dict[str, str] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip()
            if not ticker:
                continue
            try:
                year = int(row["year"])
                value = float(row["value"])
            except (ValueError, KeyError):
                continue
            kpi = row.get("kpi", "").strip()
            company_name = row.get("company_name", "").strip()
            industry = row.get("industry", "").strip()

            gt[(ticker, year, kpi)] = {
                "value": value,
                "source": row.get("source", "").strip(),
                "tag": row.get("tag", "").strip(),
                "company_name": company_name,
                "exchange": row.get("exchange", "").strip(),
                "exchange_ocr": row.get("exchange_ocr", "").strip(),
                "industry": industry,
            }
            if company_name and ticker not in ticker_to_name:
                ticker_to_name[ticker] = company_name
            if industry and ticker not in ticker_to_industry:
                ticker_to_industry[ticker] = industry
    return gt, ticker_to_name, ticker_to_industry


def load_kpi_aliases(json_path: Path) -> dict[str, list[str]]:
    return json.loads(json_path.read_text())


def load_query_templates(queries_dir: Path) -> dict[str, list[str]]:
    templates: dict[str, list[str]] = {}
    for fname, kpi_key in QUERY_FILE_TO_KPI.items():
        path = queries_dir / fname
        if not path.is_file():
            continue
        templates[kpi_key] = json.loads(path.read_text())
    return templates


def prepare_report(
    report: ReportData,
    alias_index: dict,
    alias_patterns: dict[str, list[tuple[str, re.Pattern[str]]]],
) -> None:
    """Load and parse the .mmd file in-place, pre-caching numbers, unit hints, and alias hits."""
    raw = report.mmd_path.read_text(encoding="utf-8", errors="replace")
    raw = normalize_text(raw)
    pages = [p.strip() for p in PAGE_SPLIT_RE.split(raw) if p.strip()]
    report.pages = pages
    report.lines_by_page = [
        [normalize_line(ln) for ln in page.splitlines()] for page in pages
    ]
    doc_text = "\n".join(pages[:4])
    report.doc_hints = detect_unit_hints(doc_text)
    if not report.doc_hints:
        report.doc_hints = detect_unit_hints(raw)

    parsed_pages: list[ParsedPage] = []
    for page_idx, page_text in enumerate(pages):
        lines = report.lines_by_page[page_idx]
        page_hints = detect_unit_hints(page_text)
        parsed_lines: list[ParsedLine] = []
        for line_idx, line in enumerate(lines):
            if not line:
                parsed_lines.append(ParsedLine(text=line, numbers=[], unit_hints=[]))
                continue
            nums = (
                parse_numbers_from_line(line) if any(c.isdigit() for c in line) else []
            )
            window_lo = max(0, line_idx - 1)
            window_hi = min(len(lines), line_idx + 2)
            line_hints = (
                detect_unit_hints(" ".join(lines[window_lo:window_hi])) if nums else []
            )
            alias_kpis = find_alias_kpis(line, alias_index, alias_patterns)
            parsed_lines.append(
                ParsedLine(
                    text=line,
                    numbers=nums,
                    unit_hints=line_hints,
                    alias_kpis=alias_kpis,
                )
            )
        parsed_pages.append(
            ParsedPage(raw_text=page_text, lines=parsed_lines, page_hints=page_hints)
        )
    report.parsed_pages = parsed_pages


def build_report_index(
    reports: list[ReportData],
) -> dict[tuple[str, int], ReportData]:
    return {(r.ticker, r.year): r for r in reports}


# ---------------------------------------------------------------------------
# Literal value formatting
# ---------------------------------------------------------------------------


def format_literal_variants(value: float) -> list[str]:
    """Generate pre-formatted string variants of a target value for literal search."""
    variants: list[str] = []
    abs_val = abs(value)

    # Raw with commas (no decimal)
    if abs_val >= 1:
        raw_int = f"{abs_val:,.0f}"
        variants.append(raw_int)
        variants.append(f"${raw_int}")

    # Millions scale
    if abs_val >= 1e6:
        mill = abs_val / 1e6
        if mill >= 100:
            variants.append(f"{mill:,.0f}")
            variants.append(f"${mill:,.0f}")
        else:
            variants.append(f"{mill:,.1f}")
            variants.append(f"${mill:,.1f}")
        variants.append(f"{mill:,.0f} million")
        variants.append(f"{mill:,.1f} million")

    # Billions scale
    if abs_val >= 1e9:
        bill = abs_val / 1e9
        variants.append(f"{bill:,.1f}")
        variants.append(f"${bill:,.1f}")
        variants.append(f"{bill:,.2f}")
        variants.append(f"${bill:,.2f}")
        variants.append(f"{bill:,.1f} billion")
        variants.append(f"{bill:,.2f} billion")

    # Thousands scale
    if 1e3 <= abs_val < 1e6:
        thou = abs_val / 1e3
        variants.append(f"{thou:,.0f}")
        variants.append(f"${thou:,.0f}")
        variants.append(f"{thou:,.0f} thousand")

    return variants


def search_literal_in_page(page_text: str, value: float) -> str | None:
    """Search for pre-formatted variants of value in page text. Returns the matched variant or None."""
    variants = format_literal_variants(value)
    page_lower = page_text.lower()
    for v in variants:
        if v.lower() in page_lower:
            return v
    return None


# ---------------------------------------------------------------------------
# Core search logic
# ---------------------------------------------------------------------------


def search_report_for_kpi(
    report: ReportData,
    kpi: str,
    target_value: float,
    alias_patterns: dict[str, list[tuple[str, re.Pattern[str]]]],
    tolerance: float,
    *,
    is_target_year: bool,
) -> list[PageCandidate]:
    """Search a single report for pages containing the KPI value.

    Uses pre-parsed numbers, unit hints, and alias hits from ``report.parsed_pages``.
    """
    candidates: list[PageCandidate] = []
    patterns = alias_patterns.get(kpi, [])

    for page_idx, parsed_page in enumerate(report.parsed_pages):
        page_hints = parsed_page.page_hints
        parsed_lines = parsed_page.lines

        # --- Alias scan (use pre-computed alias_kpis) ---
        alias_hits: list[tuple[int, str]] = []
        for line_idx, pl in enumerate(parsed_lines):
            if kpi not in pl.alias_kpis:
                continue
            for alias_text, pat in patterns:
                if pat.search(pl.text):
                    alias_hits.append((line_idx, alias_text))
                    break

        has_alias = len(alias_hits) > 0

        # --- Value scan (numeric tolerance) using pre-parsed numbers ---
        value_hits: list[tuple[int, NumberToken, float, str]] = []
        for line_idx, pl in enumerate(parsed_lines):
            if not pl.numbers:
                continue
            for num in pl.numbers:
                chosen_mult, unit_source, _ = resolve_multiplier(
                    num.inline_multiplier, pl.unit_hints, page_hints, report.doc_hints
                )
                normalized = num.value * chosen_mult
                if abs(target_value) > 0:
                    rel_err = abs(normalized - target_value) / abs(target_value)
                else:
                    rel_err = abs(normalized)
                if rel_err <= tolerance:
                    value_hits.append((line_idx, num, rel_err, unit_source))

        has_value_numeric = len(value_hits) > 0

        # --- Value scan (literal string) ---
        literal_match = search_literal_in_page(parsed_page.raw_text, target_value)
        has_literal = literal_match is not None

        has_value = has_value_numeric or has_literal

        # --- Determine match type and whether to emit candidate ---
        if is_target_year:
            if has_alias and has_value:
                match_type = "alias+value"
            elif has_value:
                match_type = "value-only"
            else:
                continue
        else:
            if has_alias and has_value:
                match_type = "alias+value"
            else:
                continue

        # Build the best candidate for this page
        best_alias = alias_hits[0][1] if alias_hits else ""
        best_rel_error = min((h[2] for h in value_hits), default=0.0)

        if value_hits:
            best_raw = value_hits[0][1].raw
            best_num = value_hits[0][1]
            best_line_idx = value_hits[0][0]
            best_mult, _, _ = resolve_multiplier(
                best_num.inline_multiplier,
                parsed_lines[best_line_idx].unit_hints,
                page_hints,
                report.doc_hints,
            )
            best_normalized = best_num.value * best_mult
            best_unit = value_hits[0][3]
        else:
            best_raw = literal_match or ""
            best_normalized = target_value
            best_unit = "literal"

        # Use the alias line for the snippet, or the first value line
        snippet_line = ""
        if alias_hits:
            snippet_line = parsed_lines[alias_hits[0][0]].text
        elif value_hits:
            snippet_line = parsed_lines[value_hits[0][0]].text

        candidates.append(
            PageCandidate(
                report_name=report.name,
                ticker=report.ticker,
                report_year=report.year,
                page_idx=page_idx,
                match_type=match_type,
                alias_matched=best_alias,
                raw_value=best_raw,
                normalized_value=best_normalized,
                rel_error=best_rel_error,
                unit_source=best_unit,
                snippet=compact_snippet(snippet_line),
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Query instantiation
# ---------------------------------------------------------------------------


def instantiate_queries(
    tickers: list[str],
    years: list[int],
    kpis: list[str],
    ground_truth: dict[tuple[str, int, str], dict],
    ticker_to_name: dict[str, str],
    templates: dict[str, list[str]],
    seed: int = 42,
) -> list[Query]:
    rng = random.Random(seed)
    queries: list[Query] = []
    for ticker in tickers:
        display_name = ticker_to_name.get(ticker, ticker)
        for year in years:
            for kpi in kpis:
                gt = ground_truth.get((ticker, year, kpi))
                if gt is None:
                    continue
                kpi_templates = templates.get(kpi, [])
                if kpi_templates:
                    tpl = rng.choice(kpi_templates)
                    qtext = tpl.replace("ABC", display_name).replace("X", str(year))
                else:
                    qtext = f"What is the {kpi} of {display_name} in {year}?"
                qid = f"{ticker}_{kpi}_{year}"
                queries.append(
                    Query(
                        query_id=qid,
                        ticker=ticker,
                        year=year,
                        kpi=kpi,
                        target_value=gt["value"],
                        company_name=display_name,
                        query_text=qtext,
                    )
                )
    return queries


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_qrels(candidates: list[tuple[str, PageCandidate]], path: Path) -> int:
    """Write TREC-format qrels. Returns number of lines written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for query_id, c in candidates:
        doc_id = f"{c.report_name}/page_{c.page_idx:04d}"
        key = (query_id, doc_id)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{query_id}\t0\t{doc_id}\t1\n")
    with path.open("w") as f:
        f.writelines(sorted(lines))
    return len(lines)


def write_review_csv(candidates: list[tuple[str, PageCandidate]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id",
        "doc_id",
        "report_name",
        "report_year",
        "page_idx",
        "match_type",
        "alias_matched",
        "raw_value",
        "normalized_value",
        "rel_error",
        "unit_source",
        "snippet",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for query_id, c in candidates:
            doc_id = f"{c.report_name}/page_{c.page_idx:04d}"
            w.writerow(
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "report_name": c.report_name,
                    "report_year": c.report_year,
                    "page_idx": c.page_idx,
                    "match_type": c.match_type,
                    "alias_matched": c.alias_matched,
                    "raw_value": c.raw_value,
                    "normalized_value": c.normalized_value,
                    "rel_error": f"{c.rel_error:.6f}" if c.rel_error else "",
                    "unit_source": c.unit_source,
                    "snippet": c.snippet,
                }
            )


def write_summary(
    queries: list[Query],
    candidates: list[tuple[str, PageCandidate]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_query: dict[str, list[PageCandidate]] = defaultdict(list)
    for qid, c in candidates:
        by_query[qid].append(c)

    lines: list[str] = []
    lines.append("# Qrels generation summary\n")
    lines.append(f"- Total queries: {len(queries)}")
    lines.append(f"- Total candidate pages: {len(candidates)}")
    lines.append(
        f"- Queries with at least 1 candidate: {sum(1 for q in queries if by_query.get(q.query_id))}"
    )
    lines.append("")

    lines.append("## Per-query breakdown\n")
    lines.append(
        "| query_id | ticker | year | kpi | target_value | n_candidates | match_types |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for q in queries:
        qc = by_query.get(q.query_id, [])
        n = len(qc)
        types = ", ".join(sorted(set(c.match_type for c in qc))) if qc else "—"
        tv = (
            f"{q.target_value:,.0f}"
            if abs(q.target_value) >= 1
            else f"{q.target_value:.2f}"
        )
        lines.append(
            f"| {q.query_id} | {q.ticker} | {q.year} | {q.kpi} | {tv} | {n} | {types} |"
        )
    lines.append("")

    # Per-report breakdown
    by_report: dict[str, int] = defaultdict(int)
    for _, c in candidates:
        by_report[c.report_name] += 1
    lines.append("## Per-report candidate counts\n")
    lines.append("| report_name | n_candidates |")
    lines.append("| --- | --- |")
    for name, count in sorted(by_report.items(), key=lambda x: -x[1]):
        lines.append(f"| {name} | {count} |")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ocr-root",
        type=Path,
        default=DEFAULT_OCR_ROOT,
        help="Root directory of OCR'd reports.",
    )
    p.add_argument(
        "--kpis-long",
        type=Path,
        default=DEFAULT_KPIS_LONG,
        help="Path to kpis_long_full.csv.",
    )
    p.add_argument(
        "--aliases",
        type=Path,
        default=DEFAULT_ALIASES,
        help="Path to kpi_aliases.json.",
    )
    p.add_argument(
        "--queries-dir",
        type=Path,
        default=DEFAULT_QUERIES_DIR,
        help="Directory containing query template JSON files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for qrels, review CSV, and summary.",
    )

    sel = p.add_argument_group("selection filters")
    sel.add_argument(
        "--industry",
        type=str,
        default=None,
        help="Restrict to one industry (matches the 'industry' column in the CSV).",
    )
    sel.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Restrict to specific tickers.",
    )
    sel.add_argument(
        "--kpis",
        nargs="+",
        default=None,
        help="Restrict to specific KPI keys (e.g. revenue net_income).",
    )
    sel.add_argument(
        "--years",
        type=str,
        default=None,
        help="Year range, e.g. 2018-2021 or a single year 2020.",
    )
    sel.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N queries.",
    )

    tuning = p.add_argument_group("tuning")
    tuning.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative error tolerance for value matching. Default 0.01 (1%%).",
    )
    tuning.add_argument(
        "--search-future",
        action="store_true",
        default=False,
        help="Also search N+1 and N+2 reports for comparative-table restatements.",
    )
    tuning.add_argument(
        "--max-future-years",
        type=int,
        default=2,
        help="Years after the target year to search (only with --search-future). Default 2.",
    )
    tuning.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for template selection. Default 42.",
    )
    args = p.parse_args()

    if not args.search_future:
        args.max_future_years = 0

    # --- Load data ---
    sys.stderr.write("[load] Loading ground truth...\n")
    ground_truth, ticker_to_name, ticker_to_industry = load_ground_truth(args.kpis_long)
    sys.stderr.write(
        f"[load] {len(ground_truth)} (ticker, year, kpi) cells, "
        f"{len(ticker_to_name)} tickers\n"
    )

    sys.stderr.write("[load] Loading aliases...\n")
    kpi_aliases = load_kpi_aliases(args.aliases)

    sys.stderr.write("[load] Loading query templates...\n")
    templates = load_query_templates(args.queries_dir)
    sys.stderr.write(f"[load] {len(templates)} KPIs with templates\n")

    # --- Discover reports ---
    sys.stderr.write("[discover] Scanning OCR reports...\n")
    all_reports = discover_reports(args.ocr_root)
    sys.stderr.write(f"[discover] {len(all_reports)} reports found\n")

    report_index = build_report_index(all_reports)

    # --- Determine selection ---
    if args.tickers:
        selected_tickers = args.tickers
    elif args.industry:
        # Filter by industry column from the CSV
        selected_tickers = [
            t
            for t, ind in ticker_to_industry.items()
            if args.industry.lower() in ind.lower()
            or ind.lower() in args.industry.lower()
        ]
        sys.stderr.write(
            f"[select] {len(selected_tickers)} tickers in industry '{args.industry}'\n"
        )
    else:
        # Use all tickers that have both ground truth AND an OCR report
        tickers_with_reports = {r.ticker for r in all_reports}
        selected_tickers = sorted(ticker_to_name.keys() & tickers_with_reports)
        sys.stderr.write(
            f"[select] {len(selected_tickers)} tickers with both GT and OCR reports\n"
        )

    selected_kpis = args.kpis if args.kpis else sorted(kpi_aliases.keys())

    if args.years:
        if "-" in args.years:
            y_start, y_end = args.years.split("-", 1)
            selected_years = list(range(int(y_start), int(y_end) + 1))
        else:
            selected_years = [int(args.years)]
    else:
        # No default year restriction — use all years present in data
        all_years = sorted({y for (_, y, _) in ground_truth.keys()})
        selected_years = all_years

    sys.stderr.write(
        f"[select] {len(selected_tickers)} tickers, "
        f"{len(selected_kpis)} KPIs, "
        f"{len(selected_years)} years ({min(selected_years)}-{max(selected_years)})\n"
    )

    # --- Instantiate queries ---
    queries = instantiate_queries(
        tickers=selected_tickers,
        years=selected_years,
        kpis=selected_kpis,
        ground_truth=ground_truth,
        ticker_to_name=ticker_to_name,
        templates=templates,
        seed=args.seed,
    )
    if args.limit:
        queries = queries[: args.limit]
    sys.stderr.write(f"[queries] {len(queries)} queries instantiated\n")

    if not queries:
        sys.stderr.write(
            "[queries] Nothing to do — no matching (ticker, year, kpi) triples.\n"
        )
        return

    # --- Compile alias patterns ---
    alias_patterns = compile_alias_patterns(kpi_aliases, selected_kpis)
    alias_index = build_alias_index(alias_patterns)

    # --- Group queries by (ticker, year) for efficient report loading ---
    queries_by_pair: dict[tuple[str, int], list[Query]] = defaultdict(list)
    for q in queries:
        queries_by_pair[(q.ticker, q.year)].append(q)

    # --- Process reports ---
    all_candidates: list[tuple[str, PageCandidate]] = []
    reports_processed = 0
    reports_loaded: dict[tuple[str, int], ReportData] = {}

    pairs = sorted(queries_by_pair.keys())
    sys.stderr.write(
        f"[search] Processing {len(pairs)} unique (ticker, year) pairs...\n"
    )

    for ticker, year in pairs:
        pair_queries = queries_by_pair[(ticker, year)]
        kpis_for_pair = {q.kpi: q for q in pair_queries}

        # Search in target year + future years
        years_to_search = [year] + [
            year + dy for dy in range(1, args.max_future_years + 1)
        ]

        for search_year in years_to_search:
            report = report_index.get((ticker, search_year))
            if report is None:
                continue

            # Load report lazily
            if (ticker, search_year) not in reports_loaded:
                prepare_report(report, alias_index, alias_patterns)
                reports_loaded[(ticker, search_year)] = report
                reports_processed += 1
                if reports_processed % 100 == 0:
                    sys.stderr.write(
                        f"[search] Loaded {reports_processed} reports...\n"
                    )

            is_target = search_year == year

            for kpi, q in kpis_for_pair.items():
                page_candidates = search_report_for_kpi(
                    report=report,
                    kpi=kpi,
                    target_value=q.target_value,
                    alias_patterns=alias_patterns,
                    tolerance=args.tolerance,
                    is_target_year=is_target,
                )
                for c in page_candidates:
                    all_candidates.append((q.query_id, c))

    sys.stderr.write(
        f"[search] Done: {reports_processed} reports loaded, "
        f"{len(all_candidates)} candidate pages found\n"
    )

    # --- Write outputs ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_qrels = write_qrels(all_candidates, args.output_dir / "qrels.txt")
    write_review_csv(all_candidates, args.output_dir / "review_candidates.csv")
    write_summary(queries, all_candidates, args.output_dir / "summary.md")

    sys.stderr.write(f"\n[done] Wrote to {args.output_dir}/\n")
    sys.stderr.write(f"  qrels.txt:             {n_qrels} lines\n")
    sys.stderr.write(f"  review_candidates.csv: {len(all_candidates)} rows\n")
    sys.stderr.write("  summary.md\n")

    # Quick stats
    queries_with_candidates = len({qid for qid, _ in all_candidates})
    sys.stderr.write(
        f"\n[stats] {queries_with_candidates}/{len(queries)} queries "
        f"have at least 1 candidate page\n"
    )


if __name__ == "__main__":
    main()
