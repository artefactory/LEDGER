"""Logical KPIs -> candidate IFRS XBRL tags (ordered by preference).

This is the IFRS counterpart of tags.py (which targets US-GAAP / SEC filings).
The French ESEF filings downloaded from filings.xbrl.org use the IFRS taxonomy
(`ifrs-full` namespace), so NONE of the us-gaap tags apply. The candidate tags
below were verified against 1166 downloaded JSON filings — coverage counts are
noted inline so the waterfall ordering is grounded in real data.

`kind`:
  - "flow"  : income statement / cash flow item, reported over a duration
              (period like "2025-04-01/2026-04-01").
  - "stock" : balance sheet item, reported at a point in time
              (period like "2025-12-31").

`unit` is the value of the `dimensions.unit` key in the filings.xbrl.org JSON
fact objects. Unlike SEC XBRL-JSON (which nests facts under
`facts.<taxonomy>.<tag>.units.<unit>`), the filings.xbrl.org format stores the
unit as a flat dimension. We use wildcards here:
  - "iso4217:*"             -> any reporting currency (EUR dominates, but some
                               French filers report in USD/other). The extractor
                               should match any key starting with "iso4217:".
  - "iso4217:*/xbrli:shares" -> per-share (monetary / shares) measures.
  - "xbrli:shares"           -> pure share counts.

## How the waterfall handles multiple tags per KPI

Same three situations as the US-GAAP version:

  1. Synonyms. E.g. `Revenue` and `RevenueFromContractsWithCustomers` are
     both "total revenue" for IFRS-15 reporters. The waterfall picks the first.

  2. Same concept, DIFFERENT scope. E.g. `EquityAttributableToOwnersOfParent`
     (parent only) vs `Equity` (incl. non-controlling interest). The ordering
     is chosen so the *first* tag matches the conventional benchmarking
     definition. **This ordering is load-bearing — see the scope notes on
     net_income, stockholders_equity, and total_liabilities.**

  3. Aggregate vs component. E.g. many filers omit the aggregate `Liabilities`
     tag and only report `CurrentLiabilities` + `NoncurrentLiabilities`.
     `sum_components` tuples are summed as a last-resort fallback (only fires
     when the primary `tags` waterfall misses a year and ALL tags in the tuple
     are present). A "-" prefix means subtract (signed sum), e.g.
     `Liabilities = EquityAndLiabilities - Equity`.

## IFRS-specific caveats (vs US-GAAP)

  - Expense classification. IFRS allows "by function" (CostOfSales, GrossProfit)
     OR "by nature" (RawMaterials, EmployeeBenefits, Depreciation, Other). French
     filers frequently use BY NATURE, so `cost_of_revenue` (~28%), `gross_profit`
     (~19%) and `rd_expense` (~21%) have <30% coverage and were removed. Same
     for debt KPIs `long_term_debt_total` (~5.6%), `long_term_debt_current`
     (~4.3%) and `short_term_borrowings` (~14.7%) — no clean single-tag fallback
     without scope mixing.
  - Currency. The reporting currency is almost always EUR, but a few large
    issuers (e.g. TotalEnergies) report in USD. Use the "iso4217:*" wildcard so
    the extractor does not silently drop non-EUR filers.
  - EPS. The per-share unit is "iso4217:EUR/xbrli:shares" (or other currency).
    A combined `BasicAndDilutedEarningsLossPerShare` tag exists for a minority
    of filers and is used as a last-resort fallback for both basic and diluted.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KpiDef:
    key: str
    label: str  # human readable name of KPI
    kind: str  # "flow" | "stock"
    unit: str
    tags: tuple[str, ...]
    # Optional: list of tag-sets to sum as a last-resort fallback when the
    # `tags` waterfall misses a year. Each inner tuple must have ALL its tags
    # present for the year for the sum to be used. Tried in order.
    # A tag name prefixed with "-" means subtract (signed sum).
    sum_components: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


# All tags below are namespaced `ifrs-full:` in the filings. Only the bare
# concept name is stored here; the extractor should resolve to
# f"ifrs-full:{tag}".
KPI_DEFS: tuple[KpiDef, ...] = (
    # --- Income statement ---
    KpiDef(
        "revenue",
        "Revenue",
        "flow",
        "iso4217:*",
        # Rule: first-hit synonyms (Revenue 480/1166 preferred, fallback
        # RevenueFromContractsWithCustomers 488/1166); banks/insurers may
        # report neither -> missing.
        (
            "Revenue",
            "RevenueFromContractsWithCustomers",
        ),
    ),
    KpiDef(
        "sga_expense",
        "SG&A expense",
        "flow",
        "iso4217:*",
        # Rule: first-hit waterfall, no sum. If filer reports both
        # GeneralAndAdministrativeExpense and DistributionCosts, only first
        # present is taken -> understates total SG&A (no aggregation).
        (
            "SellingGeneralAndAdministrativeExpense",  # 94/1166 (combined)
            "GeneralAndAdministrativeExpense",  # 265/1166
            "AdministrativeExpense",  # 98/1166
            "DistributionCosts",  # 83/1166
        ),
    ),
    KpiDef(
        "operating_income",
        "Operating income",
        "flow",
        "iso4217:*",
        ("ProfitLossFromOperatingActivities",),  # 1071/1166
    ),
    KpiDef(
        "interest_expense",
        "Interest expense",
        "flow",
        "iso4217:*",
        # Rule: InterestExpense preferred (precise, 218/1166); fallback
        # FinanceCosts (broader incl. FX/other, 273/1166) only if missing ->
        # may overstate interest.
        ("InterestExpense", "FinanceCosts"),
    ),
    KpiDef(
        "income_tax_expense",
        "Income tax expense",
        "flow",
        "iso4217:*",
        ("IncomeTaxExpenseContinuingOperations",),  # 1150/1166
    ),
    KpiDef(
        # Rule: attributable to parent preferred (ProfitLossAttributable...,
        # 1084/1166); fallback ProfitLoss incl. NCI (1166/1166) only if
        # missing -> small NCI bias on fallback (~7% filers).
        "net_income",
        "Net income (attributable to parent)",
        "flow",
        "iso4217:*",
        (
            "ProfitLossAttributableToOwnersOfParent",  # 1084/1166
            "ProfitLoss",  # 1166/1166 (incl. NCI; scope differs)
        ),
    ),
    KpiDef(
        "eps_basic",
        "EPS (basic)",
        "flow",
        "iso4217:*/xbrli:shares",
        # BasicEarningsLossPerShare (1048/1166). A minority report a combined
        # basic+diluted tag (12/1166) used as last resort for both.
        ("BasicEarningsLossPerShare", "BasicAndDilutedEarningsLossPerShare"),
    ),
    KpiDef(
        "eps_diluted",
        "EPS (diluted)",
        "flow",
        "iso4217:*/xbrli:shares",
        ("DilutedEarningsLossPerShare", "BasicAndDilutedEarningsLossPerShare"),
    ),
    # --- Balance sheet ---
    KpiDef(
        "total_assets",
        "Total assets",
        "stock",
        "iso4217:*",
        ("Assets",),  # 1166/1166
    ),
    KpiDef(
        "total_liabilities",
        "Total liabilities",
        "stock",
        "iso4217:*",
        # Rule: Liabilities (212/1166) preferred; fallback sum
        # CurrentLiabilities+NoncurrentLiabilities (866+1041), last resort
        # EquityAndLiabilities - Equity (1138-1163) identity.
        ("Liabilities",),
        sum_components=(
            ("CurrentLiabilities", "NoncurrentLiabilities"),  # 866 + 1041
            ("EquityAndLiabilities", "-Equity"),  # 1138 - 1163
        ),
    ),
    KpiDef(
        # Scope: attributable to parent only. Equity (incl. NCI) is a separate
        # KPI (stockholders_equity_incl_nci) below.
        "stockholders_equity",
        "Stockholders' equity (attributable to parent)",
        "stock",
        "iso4217:*",
        (
            "EquityAttributableToOwnersOfParent",  # 1086/1166
            "Equity",  # 1163/1166 (incl. NCI; scope differs, last resort)
        ),
    ),
    KpiDef(
        "stockholders_equity_incl_nci",
        "Stockholders' equity (incl. non-controlling interest)",
        "stock",
        "iso4217:*",
        ("Equity",),  # 1163/1166 — this IS equity incl. NCI in IFRS
    ),
    KpiDef(
        "cash_and_equivalents",
        "Cash & equivalents",
        "stock",
        "iso4217:*",
        # CashAndCashEquivalents (1101/1166) is the standard. A few filers
        # report Cash (53) and/or CashEquivalents (45) separately.
        ("CashAndCashEquivalents", "Cash", "CashEquivalents"),
        sum_components=(
            ("Cash", "CashEquivalents"),  # filers that split the two
        ),
    ),
    # --- Debt (remaining KPIs >=30% coverage) ---
    # IFRS borrowings taxonomy: LongtermBorrowings = noncurrent borrowings (619/1166).
    # Other debt KPIs (long_term_debt_total, long_term_debt_current,
    # short_term_borrowings) removed — coverage <30% (5.6%/4.3%/14.7% on 1153 filings).
    KpiDef(
        "long_term_debt_noncurrent",
        "Long-term debt (noncurrent portion only)",
        "stock",
        "iso4217:*",
        # Rule: interest-bearing borrowings noncurrent only, excl. lease
        # liabilities (IFRS 16 separate). Single tag, no sum/fallback.
        ("LongtermBorrowings",),  # 619/1166
    ),
    # --- Working capital ---
    KpiDef(
        "inventory",
        "Inventory",
        "stock",
        "iso4217:*",
        ("Inventories",),  # 725/1166
    ),
    KpiDef(
        "accounts_receivable",
        "Accounts receivable (trade)",
        "stock",
        "iso4217:*",
        # Rule: trade-only preferred (CurrentTradeReceivables 581/1166);
        # fallback TradeAndOther* broader (incl. non-trade) if missing.
        (
            "CurrentTradeReceivables",  # 581/1166
            "TradeAndOtherCurrentReceivables",  # 296/1166
            "TradeReceivables",  # 33/1166
            "TradeAndOtherReceivables",  # 43/1166
        ),
    ),
    KpiDef(
        "accounts_payable",
        "Accounts payable (trade)",
        "stock",
        "iso4217:*",
        # Rule: trade-only preferred (535/1166); fallback broader
        # TradeAndOther* (incl. non-trade) if missing.
        (
            "TradeAndOtherCurrentPayablesToTradeSuppliers",  # 535/1166
            "TradeAndOtherCurrentPayables",  # 352/1166
            "TradeAndOtherPayablesToTradeSuppliers",  # 71/1166
            "TradeAndOtherPayables",  # 43/1166
        ),
    ),
    KpiDef(
        "shares_outstanding",
        "Shares outstanding",
        "stock",
        "xbrli:shares",
        # NumberOfSharesOutstanding (246) is precise. NumberOfSharesIssued
        # (189) differs when treasury shares exist, but is a reasonable
        # last-resort fallback.
        ("NumberOfSharesOutstanding", "NumberOfSharesIssued"),
    ),
    # --- Cash flow statement ---
    KpiDef(
        "operating_cash_flow",
        "Operating cash flow",
        "flow",
        "iso4217:*",
        (
            "CashFlowsFromUsedInOperatingActivities",  # 1105/1166
            "CashFlowsFromUsedInOperatingActivitiesContinuingOperations",  # 102
        ),
    ),
    KpiDef(
        "investing_cash_flow",
        "Investing cash flow",
        "flow",
        "iso4217:*",
        ("CashFlowsFromUsedInInvestingActivities",),  # 1119/1166
    ),
    KpiDef(
        "financing_cash_flow",
        "Financing cash flow",
        "flow",
        "iso4217:*",
        ("CashFlowsFromUsedInFinancingActivities",),  # 1127/1166
    ),
    KpiDef(
        "capex",
        "Capital expenditure (PPE)",
        "flow",
        "iso4217:*",
        # Rule: PPE-only first (287/1166); fallback to combined
        # PPE+intangibles+investment property (221/1166) only if missing ->
        # scope shift (broader) on fallback.
        (
            "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
            "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets",
        ),
    ),
    KpiDef(
        "depreciation_amortization",
        "Depreciation & amortization",
        "flow",
        "iso4217:*",
        # Rule: waterfall P&L DepreciationAndAmortisationExpense (282) ->
        # CFS AdjustmentsForDepreciationAndAmortisationExpense (333) ->
        # impairment-inclusive (88) last resort; sum fallback only if both
        # DepreciationExpense (16) and AmortisationExpense (18) present.
        (
            "DepreciationAndAmortisationExpense",  # 282/1166
            "AdjustmentsForDepreciationAndAmortisationExpense",  # 333/1166
            "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss",  # 88
        ),
        sum_components=(
            ("DepreciationExpense", "AmortisationExpense"),  # 16 + 18
        ),
    ),
    KpiDef(
        "dividends_paid",
        "Dividends paid",
        "flow",
        "iso4217:*",
        # DividendsPaid (832) is the total. DividendsPaidClassifiedAsFinancingActivities
        # (341) is the same concept reported on the cash-flow statement.
        ("DividendsPaid", "DividendsPaidClassifiedAsFinancingActivities"),
    ),
)


KPI_BY_KEY: dict[str, KpiDef] = {k.key: k for k in KPI_DEFS}
