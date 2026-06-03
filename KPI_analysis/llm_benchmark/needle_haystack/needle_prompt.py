"""Prompt construction for the needle-in-a-haystack KPI benchmark.

Design goal: **prefix caching**. The expensive part of every call is prefilling
the ~100k-token annual report. We want vLLM (launched with
``--enable-prefix-caching``) to prefill each report exactly once and reuse those
KV blocks for all ~20 queries that target it. That requires the per-report token
prefix to be byte-identical across those queries, with only a tiny suffix
varying:

    role=system : SYSTEM_PROMPT                      <- constant for ALL queries
    role=user   : <DOCUMENT with [Page N] markers>   <- constant for ALL queries
                  <DOC_QUESTION_SEPARATOR>             on the SAME report
                  <QUESTION block>                    <- the ONLY varying part

So a report's whole batch shares the prefix ``SYSTEM + DOCUMENT + SEPARATOR``;
only the final QUESTION block (company / year / metric / informal phrasing,
~80 tokens) changes. ``run_needle.py`` issues one warm-up query per report to
populate the cache, then fires the rest against it.

Two query modes (``--query-mode``):

- ``defined`` (default): the QUESTION block includes the canonical, scope-precise
  definition of the requested KPI (reused from ``kpi_catalogue.DESCRIPTIONS``).
  This isolates the skill under test — *locate + transcribe + scale the right
  figure* — from the orthogonal guessing game of *which scope did they mean*
  (e.g. "cost of revenue" = COGS-only vs total cost of sales; net income parent-
  only vs incl. NCI). Ground truth in ``kpis_long.csv`` commits to one scope per
  key, so telling the model that scope is the only way the score measures
  extraction rather than scope-guessing.
- ``plain``: only the informal natural-language question is given (no definition,
  no unit hint). Harder and more realistic; scope ambiguity counts against the
  model. Useful as an ablation.
"""

from __future__ import annotations

import hashlib

from needle_data import UNIT_PHRASE


# ---------------------------------------------------------------------------
# System prompt (constant for every query -> fully prefix-cached)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a meticulous financial-statement analyst. You will be given the full \
OCR'd text of ONE company's annual report (a markdown document; page \
boundaries are marked with `[Page N]` headers, and tables may appear as HTML, \
markdown, or whitespace-aligned text — read all of them).

You will then be asked for the value of ONE specific financial metric for ONE \
specific fiscal year. Your job is to locate that single figure in the report \
and return it as a small JSON object. There is exactly one correct figure; \
find it precisely.

Follow these rules exactly.

1. SOURCE. Take the figure from the company's PRIMARY CONSOLIDATED financial \
statements — the consolidated income statement, consolidated balance sheet, \
and consolidated statement of cash flows (and their accompanying notes). \
Prefer these audited statements over the financial-highlights page, the \
letter to shareholders, MD&A narrative, segment notes, or any pro-forma / \
"adjusted" / non-GAAP table.

2. MOST PRECISE REPRESENTATION. The same number may appear in several places \
at different precision. Always return the most precise one — the exact figure \
printed in the financial statements — never a rounded narrative figure. If the \
balance sheet (headed "in thousands") shows `3,400,300` and the letter says \
"approximately $3.4 billion", the answer derives from `3,400,300`, not "3.4 \
billion". Put the exact printed digits in `value_verbatim`.

3. RAW SINGLE UNITS. `value` must be in single units, with the statement's \
scale already applied:
   - "in thousands"  -> multiply the printed number by 1,000
   - "in millions"   -> multiply by 1,000,000
   - "in billions"   -> multiply by 1,000,000,000
   The scale is usually stated once in the statement header (e.g. "(in \
millions, except per-share amounts)") or a column heading; apply it. Record \
which scale you used in `unit_scale`. Worked example: a statement headed "in \
millions" showing `Total assets ... 1,234.5` -> `value_verbatim` = "1,234.5", \
`unit_scale` = "millions", `value` = 1234500000.0 (never 1234.5).
   EXCEPTION — per-share figures: earnings per share (EPS) and other \
per-share amounts are NOT scaled. Report them exactly as printed (e.g. 1.08) \
with `unit_scale` = "per_share".

4. SIGN. Use a leading minus sign for negative values; convert accounting \
parentheses to a minus sign (`(1,505)` -> -1505...). Capital expenditure and \
dividends paid are reported as POSITIVE cash outflows here: if the cash-flow \
statement shows them in parentheses, return the absolute (positive) value. The \
three cash-flow subtotals (operating / investing / financing activities) keep \
their reported sign (investing is usually negative).

5. EXACT FISCAL YEAR. Annual reports show two or three years side by side \
(current year plus comparatives). Return the value for the EXACT fiscal year \
requested — match the column whose period-end / fiscal-year label is that year. \
Do not return an adjacent year's value from a neighbouring column.

6. EXACT METRIC / SCOPE. The request names the precise metric and its scope. \
Honour scope distinctions strictly: net income ATTRIBUTABLE TO THE PARENT \
excludes non-controlling (minority) interest; "unrestricted" cash excludes \
restricted cash; the specific debt scope requested (total vs current portion \
vs non-current vs short-term borrowings) is not interchangeable. If both a \
parent-only and a consolidated line are shown, pick the one the request asks \
for.

7. NOT PRESENT. If the requested metric for the requested fiscal year is \
genuinely not stated anywhere in the report, set `found` = false and `value` = \
null. Do NOT estimate, infer, or compute it from other lines. A truthful "not \
found" is correct and is better than a fabricated number.

OUTPUT. Return a SINGLE JSON object matching the provided schema and nothing \
else — no explanation, no markdown, no code fences. The fields are: `found` \
(bool), `value` (number or null), `value_verbatim` (string or null, the exact \
printed figure), `unit_scale` (one of "units", "thousands", "millions", \
"billions", "per_share", "unknown"), and `page` (the [Page N] number you read \
it from, or null)."""


# Marks the end of the (cached) document and the start of the (varying)
# question. Constant string -> stays inside the cached prefix.
DOC_QUESTION_SEPARATOR = "\n\n===== END OF ANNUAL REPORT =====\n\n"


def build_question_block(
    *,
    company_name: str,
    ticker: str,
    year: int,
    kpi_label: str,
    kpi_definition: str,
    unit_class: str,
    query_text: str,
    query_mode: str = "defined",
) -> str:
    """Build the per-query suffix (the only part that varies within a report)."""
    if query_mode == "plain":
        # Realistic, ambiguous: just the informal question. No scope/unit aid.
        return (
            "QUESTION\n"
            f'Answer this question about the annual report above: "{query_text}"\n\n'
            "Return ONLY the JSON object for this single figure."
        )

    unit_phrase = UNIT_PHRASE.get(unit_class, UNIT_PHRASE["monetary"])
    definition = f" — {kpi_definition}" if kpi_definition else ""
    return (
        "QUESTION\n"
        f"Company: {company_name} (ticker {ticker})\n"
        f"Fiscal year requested: {year}\n"
        f"Metric requested: {kpi_label}{definition}\n"
        f"Unit for `value`: {unit_phrase}\n\n"
        f'A user phrased this request informally as: "{query_text}"\n\n'
        f"Find the {kpi_label} for fiscal year {year} in the report above and "
        "return ONLY the JSON object for this single figure."
    )


def build_user_message(document_text: str, question_block: str) -> str:
    """Assemble the user turn: cached document prefix + varying question suffix."""
    return f"{document_text}{DOC_QUESTION_SEPARATOR}{question_block}"


def build_messages(
    document_text: str,
    *,
    company_name: str,
    ticker: str,
    year: int,
    kpi_label: str,
    kpi_definition: str,
    unit_class: str,
    query_text: str,
    query_mode: str = "defined",
) -> list[dict[str, str]]:
    """Assemble the full chat messages for one needle query."""
    question_block = build_question_block(
        company_name=company_name,
        ticker=ticker,
        year=year,
        kpi_label=kpi_label,
        kpi_definition=kpi_definition,
        unit_class=unit_class,
        query_text=query_text,
        query_mode=query_mode,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(document_text, question_block)},
    ]


def prompt_version(query_mode: str = "defined") -> str:
    """Stable short hash of the prompt template, recorded in run_meta for repro.

    Changes whenever SYSTEM_PROMPT, the separator, or the question-block
    template changes — so a results directory can be tied to an exact prompt.
    """
    probe = build_question_block(
        company_name="ACME Corp.",
        ticker="ACME",
        year=2020,
        kpi_label="Net income",
        kpi_definition="Net income attributable to parent.",
        unit_class="monetary",
        query_text="What was ACME's net income in 2020?",
        query_mode=query_mode,
    )
    h = hashlib.sha256()
    h.update(SYSTEM_PROMPT.encode())
    h.update(DOC_QUESTION_SEPARATOR.encode())
    h.update(probe.encode())
    return h.hexdigest()[:12]
