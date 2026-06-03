"""Pydantic schema for the LLM KPI extraction benchmark.

The schema is the contract between the LLM and the rest of the pipeline.
It is also passed verbatim to vLLM via ``response_format={"type":
"json_schema", ...}`` so the xgrammar guided-decoding backend constrains the
LLM's output to exactly this shape.

The ``KpiKey`` Literal is the closed set of 31 canonical KPI keys defined in
``KPI_analysis/kpi_fetch_and_build/tags.py:KPI_DEFS``. xgrammar enforces it at decode time, so
the LLM cannot invent a KPI name that fails the downstream join with
``kpis_long.csv``. The runtime assertion at the bottom of this module fails
fast if ``KPI_DEFS`` and ``KpiKey`` ever drift apart.
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kpi_fetch_and_build"))
from tags import KPI_DEFS  # noqa: E402


KpiKey = Literal[
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
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    "capex",
    "depreciation_amortization",
    "dividends_paid",
]


class ExtractedKPI(BaseModel):
    """A single KPI value extracted from the report for one fiscal year."""

    kpi: KpiKey = Field(
        description="Canonical KPI key. Must be one of the 31 enumerated values."
    )
    fiscal_year: int = Field(
        ge=1990,
        le=2100,
        description=(
            "Calendar year of the period-end date (e.g. an FY ending March 2019 "
            "is year 2019)."
        ),
    )
    value: float | None = Field(
        default=None,
        description=(
            "Value in raw units. For monetary KPIs this is single dollars (or the "
            "report's reporting currency) — apply any 'in thousands' / 'in "
            "millions' / 'in billions' scaling before emitting. For "
            "shares_outstanding the unit is shares. Null if the KPI is not "
            "present in the report."
        ),
    )


class ReportExtraction(BaseModel):
    """Top-level extraction result for one annual report."""

    ticker: str = Field(
        description="Ticker symbol of the company, copied from the prompt."
    )
    reporting_currency: str | None = Field(
        default=None,
        description=(
            "ISO currency code of the report's reporting currency (USD, GBP, "
            "EUR, ...). Null if not stated."
        ),
    )
    units_note: str | None = Field(
        default=None,
        description=(
            "Diagnostic note on the scale you observed in the financial "
            "statements, e.g. 'values reported in millions'. Free text."
        ),
    )
    kpis: list[ExtractedKPI] = Field(
        default_factory=list,
        description=(
            "All KPI values extracted from the report, one row per (kpi, "
            "fiscal_year). Annual reports typically show 2-3 years per "
            "statement; emit each year as a separate item. Omit KPIs that are "
            "not present in the report — do not fabricate."
        ),
    )


# Sanity check: KpiKey must enumerate the same 31 keys as KPI_DEFS, in the
# same order. If this fires, update one or both — they are designed to stay
# in lockstep so the join with kpis_long.csv is a no-op.
_KPI_KEY_ARGS: tuple[str, ...] = typing.get_args(KpiKey)
_TAGS_KEYS: tuple[str, ...] = tuple(d.key for d in KPI_DEFS)
if _KPI_KEY_ARGS != _TAGS_KEYS:
    diff = set(_KPI_KEY_ARGS).symmetric_difference(_TAGS_KEYS)
    raise AssertionError(
        f"schema.KpiKey out of sync with tags.KPI_DEFS — symmetric diff: {diff}"
    )
