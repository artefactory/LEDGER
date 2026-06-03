"""KPI reference catalogue rendered into the system prompt.

Each canonical KPI key from ``tags.KPI_DEFS`` gets a one-line description
emphasising scope, sign convention, and unit. Wording is chosen to prevent
the LLM from silently conflating scope variants (parent-only vs incl. NCI;
unrestricted vs incl. restricted; debt-scope variants).

The catalogue text is generated once at import time and exposed as
``CATALOGUE_MARKDOWN``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kpi_fetch_and_build"))
from tags import KPI_DEFS  # noqa: E402


# Hand-written descriptions, keyed by canonical KPI key. KpiDef.label is too
# terse to capture scope (e.g. parent-only vs incl. NCI), so we override it
# here. Order is informational only — the LLM sees the catalogue grouped by
# financial statement section.
DESCRIPTIONS: dict[str, str] = {
    # --- Income statement ---
    "revenue": "Total operating revenue / net sales (top line). Flow. Reporting currency.",
    "cost_of_revenue": "Cost of goods and services sold. Flow. Reporting currency.",
    "gross_profit": "Revenue minus cost of revenue. Flow. Reporting currency.",
    "rd_expense": "Research and development expense (R&D only — exclude SG&A). Flow. Reporting currency.",
    "sga_expense": "Selling, general and administrative expense. Flow. Reporting currency.",
    "operating_income": "Operating income / operating profit (EBIT-level). Flow. Reporting currency. Sign: usually positive but can be negative (use minus sign).",
    "interest_expense": "Interest expense on debt. Flow. Reporting currency. Sign: positive (cost).",
    "income_tax_expense": "Income tax expense / (benefit). Flow. Reporting currency. Sign: positive expense, negative benefit.",
    "net_income": (
        "Net income **attributable to parent / common shareholders only**. "
        "Flow. Reporting currency. **Do NOT use the consolidated 'profit/loss' line that "
        "INCLUDES non-controlling interest** — that's a different KPI. Sign: minus for losses."
    ),
    "eps_basic": "Basic earnings per share. Per-share. Reporting currency / share.",
    "eps_diluted": "Diluted earnings per share. Per-share. Reporting currency / share.",
    # --- Balance sheet ---
    "total_assets": "Total assets at period end. Stock. Reporting currency.",
    "total_liabilities": "Total liabilities at period end. Stock. Reporting currency.",
    "stockholders_equity": (
        "Total stockholders' / shareholders' equity **attributable to parent only** "
        "(excluding any non-controlling / minority interest). Stock. Reporting currency."
    ),
    "stockholders_equity_incl_nci": (
        "Total equity **including non-controlling / minority interest** — the "
        "consolidated 'total equity' line. Stock. Reporting currency. **This is a "
        "SEPARATE KPI from `stockholders_equity`. If the report breaks both out, "
        "emit both.**"
    ),
    "cash_and_equivalents": (
        "Cash and cash equivalents (**unrestricted only** — exclude restricted cash). "
        "Stock. Reporting currency."
    ),
    "cash_incl_restricted": (
        "Cash, cash equivalents and **restricted** cash (the combined balance-sheet "
        "line introduced by ASU 2016-18). Stock. Reporting currency. **Separate KPI** "
        "from `cash_and_equivalents`."
    ),
    # --- Debt scopes (deliberately split — DO NOT conflate) ---
    "long_term_debt_total": (
        "Long-term debt **including the current portion** (i.e. all debt with "
        "original maturity > 1 year, regardless of how it's split on the balance "
        "sheet). Stock. Reporting currency."
    ),
    "long_term_debt_noncurrent": (
        "Long-term debt **excluding the current portion** (the noncurrent line "
        "on the balance sheet only). Stock. Reporting currency."
    ),
    "long_term_debt_current": (
        "Current portion of long-term debt only (the current-liabilities line "
        "for the next 12 months of long-term debt). Stock. Reporting currency."
    ),
    "short_term_borrowings": (
        "Short-term borrowings — bank lines, commercial paper, notes payable "
        "with original maturity ≤ 1 year. Stock. Reporting currency."
    ),
    # --- Working capital ---
    "inventory": "Inventory, net. Stock. Reporting currency.",
    "accounts_receivable": "Accounts receivable, current and net of allowance. Stock. Reporting currency.",
    "accounts_payable": "Accounts payable, current. Stock. Reporting currency.",
    "shares_outstanding": (
        "Common shares outstanding at period end. Stock. **Unit: shares (not "
        "currency).** Emit the share count as a raw integer-valued float."
    ),
    # --- Cash flow statement ---
    "operating_cash_flow": (
        "Net cash provided by / (used in) operating activities. Flow. "
        "Reporting currency. Sign: usually positive; can be negative."
    ),
    "investing_cash_flow": (
        "Net cash provided by / (used in) investing activities. Flow. "
        "Reporting currency. Sign: usually negative (net investment outflow)."
    ),
    "financing_cash_flow": (
        "Net cash provided by / (used in) financing activities. Flow. "
        "Reporting currency. Sign: signed (can be either)."
    ),
    "capex": (
        "Capital expenditure — payments to acquire property, plant and equipment. "
        "Flow. Reporting currency. **Sign: positive cash outflow** "
        "(if the report shows it in parentheses on the cash-flow statement, "
        "emit the absolute value as a positive number)."
    ),
    "depreciation_amortization": (
        "Depreciation and amortization (the addback line on the cash-flow "
        "statement). Flow. Reporting currency. Sign: positive."
    ),
    "dividends_paid": (
        "Cash dividends paid to common shareholders during the period. Flow. "
        "Reporting currency. **Sign: positive cash outflow** (emit absolute "
        "value, not a negative number)."
    ),
}


# Final guard: every key in KPI_DEFS must have a description, and we must not
# describe a key that isn't in KPI_DEFS.
_KEYS_IN_DEFS = {d.key for d in KPI_DEFS}
_KEYS_IN_DESCRIPTIONS = set(DESCRIPTIONS)
if _KEYS_IN_DEFS != _KEYS_IN_DESCRIPTIONS:
    diff = _KEYS_IN_DEFS.symmetric_difference(_KEYS_IN_DESCRIPTIONS)
    raise AssertionError(
        f"kpi_catalogue.DESCRIPTIONS out of sync with tags.KPI_DEFS: {diff}"
    )


_SECTION_TITLES: list[tuple[str, list[str]]] = [
    (
        "Income statement (flow metrics)",
        [
            "revenue",
            "cost_of_revenue",
            "gross_profit",
            "rd_expense",
            "sga_expense",
            "operating_income",
            "interest_expense",
            "income_tax_expense",
            "net_income",
            "eps_basic",
            "eps_diluted",
        ],
    ),
    (
        "Balance sheet (stock metrics, point-in-time)",
        [
            "total_assets",
            "total_liabilities",
            "stockholders_equity",
            "stockholders_equity_incl_nci",
            "cash_and_equivalents",
            "cash_incl_restricted",
            "long_term_debt_total",
            "long_term_debt_noncurrent",
            "long_term_debt_current",
            "short_term_borrowings",
            "inventory",
            "accounts_receivable",
            "accounts_payable",
            "shares_outstanding",
        ],
    ),
    (
        "Cash flow statement (flow metrics)",
        [
            "operating_cash_flow",
            "investing_cash_flow",
            "financing_cash_flow",
            "capex",
            "depreciation_amortization",
            "dividends_paid",
        ],
    ),
]


def _render_catalogue() -> str:
    out: list[str] = []
    for section_title, keys in _SECTION_TITLES:
        out.append(f"### {section_title}")
        for k in keys:
            out.append(f"- `{k}` — {DESCRIPTIONS[k]}")
        out.append("")  # blank line between sections
    return "\n".join(out).rstrip() + "\n"


CATALOGUE_MARKDOWN: str = _render_catalogue()
