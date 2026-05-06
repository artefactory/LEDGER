"""System prompt and message builder for the KPI extraction benchmark.

Default mode is zero-shot: the system prompt carries the full KPI catalogue
plus the load-bearing rules (raw units, parent-only scope, sign
conventions). The xgrammar JSON schema does the rest.

A single optional snippet-level few-shot pair is available behind
``few_shot=True`` for ablation; it shows how to read a tiny synthetic income
statement / balance sheet headed "(in millions)" and emit raw-dollar values.
Full-document few-shots are deliberately not included — they would 2-3× the
prompt size for marginal benefit on a strong long-context model.
"""

from __future__ import annotations

import json

from kpi_catalogue import CATALOGUE_MARKDOWN


SYSTEM_PROMPT = f"""You are a precise financial-statement extraction assistant.

You will receive the OCR'd text of a single annual report (a markdown file).
Page boundaries are marked with `[Page N]` headers. Tables may be rendered as
markdown / HTML or as whitespace-aligned text — read both.

Your task: extract values for the canonical KPIs listed below for **every
fiscal year visible in the report's primary financial statements** (typically
the current fiscal year plus 1-2 prior years shown for comparison). Return a
single JSON object matching the provided schema.

## Critical rules

1. **Raw units, not scaled.** If the financial statements are headed "in
   thousands" multiply each number by 1,000 before emitting. If "in millions"
   multiply by 1,000,000. If "in billions" multiply by 1,000,000,000. The
   `value` field must be in single units of the reporting currency (or
   single shares for `shares_outstanding`). Example: a balance sheet headed
   "in millions" showing "Total assets ... 1,234.5" must be emitted as
   `1234500000.0`, never `1234.5`. Record the scale you saw in `units_note`.

2. **Reporting currency.** Identify the report's reporting currency once
   (USD, GBP, EUR, JPY, CAD, AUD, ...) and put the ISO code in
   `reporting_currency`. Do NOT convert to USD. If a US-listed filer reports
   in USD and provides a non-USD reconciliation, use USD.

3. **Sign conventions.**
   - `capex` and `dividends_paid` are **positive cash outflows**. If the
     cash-flow statement shows "(123)" for capex, emit `123.0`, not
     `-123.0`.
   - `operating_cash_flow`, `investing_cash_flow`, `financing_cash_flow` are
     **signed**. Investing is usually negative (net investment); operating
     is usually positive. Preserve the sign as reported.
   - `net_income`, `operating_income`, `gross_profit` can be negative —
     preserve the sign. Use a leading minus sign, never accounting
     parentheses.
   - `interest_expense`, `income_tax_expense`, `depreciation_amortization`,
     `rd_expense`, `sga_expense`, `cost_of_revenue` are positive expenses
     (negative tax = tax benefit, fine to emit negative).

4. **Scope distinctions — these are LOAD-BEARING.** Several KPIs come in
   parent-only vs consolidated variants kept as DISTINCT keys:
   - `net_income` is **attributable to parent only** — exclude
     non-controlling / minority interest. If the report has a "Net income
     attributable to parent" line and a separate consolidated total, use
     the parent-only line for `net_income`.
   - `stockholders_equity` is **parent-only**. Equity including
     non-controlling interest goes under `stockholders_equity_incl_nci`
     (separate KPI). If the balance sheet breaks out both, emit both.
   - `cash_and_equivalents` is **unrestricted only**.
     `cash_incl_restricted` is a separate KPI for the combined balance.
   - The four debt-scope KPIs (`long_term_debt_total`,
     `long_term_debt_noncurrent`, `long_term_debt_current`,
     `short_term_borrowings`) mean different things — never substitute one
     for another to fill a gap.

5. **Filer's labelled fiscal year.** `fiscal_year` is the FY as the filer
   itself labels it on the 10-K cover and in the column headers of the
   primary financial statements (e.g. "Fiscal 2021", "Year ended January
   1, 2022", "FY2021" — all the same thing). Practical rule that mirrors
   how the rest of the pipeline keys data:
   - Fiscal year ending April–December → use the calendar year of the
     period-end (e.g. period ending December 31, 2022 → 2022; September 24,
     2022 → 2022).
   - Fiscal year ending January–March → use the calendar year **before**
     the period-end (e.g. period ending January 1, 2022 → 2021 — this is
     the 52/53-week-filer case for US retailers like Advance Auto Parts,
     Costco, AutoZone; period ending March 31, 2019 → 2018 — typical for
     UK filers).
   When the report shows three columns for FY2022 / FY2021 / FY2020 with
   period-end dates "January 1, 2022", "January 2, 2021", "December 28,
   2019", emit `fiscal_year=2021`, `2020`, `2019` respectively — match the
   filer's own labels, not the calendar year of the period-end.

6. **Skip if absent.** If a KPI is not stated in the report, omit that
   `(kpi, fiscal_year)` row entirely. Do NOT fabricate values, do NOT
   estimate, do NOT derive from other lines unless the report itself shows
   the derivation. Partial extractions are fine and expected.

7. **Use only the primary consolidated financial statements** (consolidated
   income statement, consolidated balance sheet, consolidated cash-flow
   statement). Ignore segment notes, MD&A roll-forwards, and pro-forma
   tables — those carry adjusted or non-consolidated figures.

## Canonical KPIs

{CATALOGUE_MARKDOWN}

## Output

Return a single JSON object matching the schema. The schema is enforced —
unknown KPI keys will be rejected. Do not include any commentary, code
fences, or text outside the JSON object.
"""


# A single optional snippet-level few-shot. Off by default. Shows raw-units
# scaling and parent-only net-income scope on a tiny synthetic snippet — NOT
# a full report.
FEW_SHOT_USER = """[Page 42]

## Consolidated statements of operations
(in millions, except per-share amounts)

|                                  | 2022    | 2021    |
| -------------------------------- | ------- | ------- |
| Net sales                        | 4,210.3 | 3,875.1 |
| Cost of sales                    | 2,820.7 | 2,610.4 |
| Gross profit                     | 1,389.6 | 1,264.7 |
| Operating income                 | 612.4   | 540.8   |
| Net income                       | 421.5   | 380.2   |
| Less: net income attributable to non-controlling interest | 18.3 | 14.7 |
| Net income attributable to ACME Corp. | 403.2 | 365.5 |
"""

FEW_SHOT_ASSISTANT_OBJ: dict = {
    "ticker": "ACME",
    "reporting_currency": "USD",
    "units_note": "Income statement reported in millions of USD",
    "kpis": [
        {"kpi": "revenue", "fiscal_year": 2022, "value": 4210300000.0},
        {"kpi": "revenue", "fiscal_year": 2021, "value": 3875100000.0},
        {"kpi": "cost_of_revenue", "fiscal_year": 2022, "value": 2820700000.0},
        {"kpi": "cost_of_revenue", "fiscal_year": 2021, "value": 2610400000.0},
        {"kpi": "gross_profit", "fiscal_year": 2022, "value": 1389600000.0},
        {"kpi": "gross_profit", "fiscal_year": 2021, "value": 1264700000.0},
        {"kpi": "operating_income", "fiscal_year": 2022, "value": 612400000.0},
        {"kpi": "operating_income", "fiscal_year": 2021, "value": 540800000.0},
        # net_income uses the PARENT-ONLY line (403.2 / 365.5), NOT the
        # consolidated 421.5 / 380.2.
        {"kpi": "net_income", "fiscal_year": 2022, "value": 403200000.0},
        {"kpi": "net_income", "fiscal_year": 2021, "value": 365500000.0},
    ],
}


def build_messages(
    report_text: str,
    *,
    ticker: str,
    few_shot: bool = False,
) -> list[dict[str, str]]:
    """Assemble the chat messages for one extraction call.

    The user message embeds the ticker so the LLM can copy it into the
    output (the schema requires a non-null ``ticker`` field) without us
    needing a separate tool call.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if few_shot:
        messages.append({"role": "user", "content": _wrap_user(FEW_SHOT_USER, ticker="ACME")})
        messages.append(
            {"role": "assistant", "content": json.dumps(FEW_SHOT_ASSISTANT_OBJ)}
        )
    messages.append({"role": "user", "content": _wrap_user(report_text, ticker=ticker)})
    return messages


def _wrap_user(report_text: str, *, ticker: str) -> str:
    return (
        f"Ticker: {ticker}\n"
        f"Annual report (OCR text follows; page boundaries marked as [Page N]):\n\n"
        f"{report_text}"
    )
