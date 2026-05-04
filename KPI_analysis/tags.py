"""Logical KPIs -> candidate XBRL tags (ordered by preference).

Different filers use different us-gaap tags for the same line item. For each
logical KPI we list the tags we will try, in order. The first tag that yields
data for a given fiscal year wins.

`kind`:
  - "flow"  : income statement / cash flow item, reported as a period total.
  - "stock" : balance sheet item, reported as a point-in-time value.

`unit` is the XBRL unit key (under `facts.us-gaap.<tag>.units`). Most dollar
amounts live under "USD"; per-share metrics under "USD/shares"; share counts
under "shares".

## How the waterfall handles multiple tags per KPI

Three distinct situations arise when a filer populates more than one of the
candidate tags for the same KPI in the same year:

  1. Synonyms (same scope, same value). E.g. for a post-ASC 606 filer,
     `RevenueFromContractWithCustomerExcludingAssessedTax` and `Revenues` often
     carry identical values. The waterfall picks the first one — fine.

  2. Same concept, DIFFERENT scope. E.g. `NetIncomeLoss` (attributable to
     parent) vs `ProfitLoss` (including non-controlling interest) — both are
     real "net income" numbers but they differ. The ordering below is chosen
     so the *first* tag matches the conventional benchmarking definition.
     **This ordering is load-bearing — do not reorder without updating the
     README "Case 2" section.** See README for the full list of scope
     choices we've baked in.

  3. Aggregate vs component. E.g. `CostOfRevenue` (aggregate) vs
     `CostOfGoodsSold` + `CostOfServices` (two components). The waterfall
     alone would pick one component and silently drop the other. To handle
     this, some KpiDefs declare `sum_components`: tuples of XBRL tags that
     must ALL be present for a given year and are then summed. This only
     fires if the primary `tags` waterfall produced no hit for that year.

     A tag name prefixed with "-" means subtract (signed sum). That lets us
     derive a missing aggregate via the balance-sheet identity, e.g.
     `Liabilities = Assets - StockholdersEquityIncludingNCI` for filers that
     omit the `Liabilities` tag.
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
    sum_components: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


KPI_DEFS: tuple[KpiDef, ...] = (
    # --- Income statement ---
    KpiDef(
        "revenue",
        "Revenue",
        "flow",
        "USD",
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606 preferred tag !
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ),
    ),
    KpiDef(
        "cost_of_revenue",
        "Cost of revenue",
        "flow",
        "USD",
        (
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "CostOfGoodsSold",  # Target manufacturing companies
            "CostOfServices",  # specific for service companies like consulting
        ),
        sum_components=(
            # For mixed goods/services filers that tag components separately
            # (e.g. IBM, GE) and omit the aggregate.
            ("CostOfGoodsSold", "CostOfServices"),
        ),
    ),
    KpiDef(
        "gross_profit",
        "Gross profit",
        "flow",
        "USD",
        ("GrossProfit",),
    ),
    KpiDef(
        "rd_expense",
        "R&D expense",
        "flow",
        "USD",
        ("ResearchAndDevelopmentExpense",),
    ),
    KpiDef(
        "sga_expense",
        "SG&A expense",
        "flow",
        "USD",
        (
            "SellingGeneralAndAdministrativeExpense",
            "GeneralAndAdministrativeExpense",
        ),
    ),
    KpiDef(
        "operating_income",
        "Operating income",
        "flow",
        "USD",
        ("OperatingIncomeLoss",),
    ),
    KpiDef(
        "interest_expense",
        "Interest expense",
        "flow",
        "USD",
        ("InterestExpense", "InterestExpenseDebt"),
    ),
    KpiDef(
        "income_tax_expense",
        "Income tax expense",
        "flow",
        "USD",
        ("IncomeTaxExpenseBenefit",),
    ),
    KpiDef(
        # Case 2: attributable to parent. ProfitLoss (incl. NCI) intentionally
        # NOT in the fallback chain — see README "Case 2" section.
        "net_income",
        "Net income (attributable to parent)",
        "flow",
        "USD",
        (
            "NetIncomeLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ),
    ),
    KpiDef(
        "eps_basic",
        "EPS (basic)",
        "flow",
        "USD/shares",
        ("EarningsPerShareBasic",),
    ),
    KpiDef(
        "eps_diluted",
        "EPS (diluted)",
        "flow",
        "USD/shares",
        ("EarningsPerShareDiluted",),
    ),
    # --- Balance sheet ---
    KpiDef(
        "total_assets",
        "Total assets",
        "stock",
        "USD",
        ("Assets",),
    ),
    KpiDef(
        "total_liabilities",
        "Total liabilities",
        "stock",
        "USD",
        ("Liabilities",),
        sum_components=(
            # Many filers skip the aggregate Liabilities tag and only report
            # the two components. Summing closes the gap.
            ("LiabilitiesCurrent", "LiabilitiesNoncurrent"),
            # Fallback to the accounting identity. Preferred denominator is
            # equity incl. NCI so we land on the same scope (total Liabilities).
            (
                "Assets",
                "-StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ),
            # Last resort: use parent-only equity. Off by the NCI amount — fine
            # for filers with no minority interest, a known small bias otherwise.
            ("Assets", "-StockholdersEquity"),
        ),
    ),
    KpiDef(
        # Case 2: attributable to parent only. Incl-NCI tag is a separate KPI.
        "stockholders_equity",
        "Stockholders' equity (attributable to parent)",
        "stock",
        "USD",
        ("StockholdersEquity",),
    ),
    KpiDef(
        "stockholders_equity_incl_nci",
        "Stockholders' equity (incl. non-controlling interest)",
        "stock",
        "USD",
        ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",),
    ),
    KpiDef(
        # Unrestricted cash only. Restricted-cash-inclusive tag is a separate KPI.
        "cash_and_equivalents",
        "Cash & equivalents (unrestricted)",
        "stock",
        "USD",
        ("CashAndCashEquivalentsAtCarryingValue",),
    ),
    KpiDef(
        "cash_incl_restricted",
        "Cash, equivalents & restricted cash",
        "stock",
        "USD",
        ("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",),
    ),
    # --- Debt (split so each KPI has unambiguous scope) ---
    # NOTE: these three mean DIFFERENT things. `LongTermDebt` includes the
    # current portion; `LongTermDebtNoncurrent` excludes it. Keep them as
    # separate KPIs rather than waterfall-ing across.
    KpiDef(
        "long_term_debt_total",
        "Long-term debt (incl. current portion)",
        "stock",
        "USD",
        ("LongTermDebt",),
    ),
    KpiDef(
        "long_term_debt_noncurrent",
        "Long-term debt (noncurrent portion only)",
        "stock",
        "USD",
        ("LongTermDebtNoncurrent",),
    ),
    KpiDef(
        "long_term_debt_current",
        "Current portion of long-term debt",
        "stock",
        "USD",
        ("LongTermDebtCurrent",),
    ),
    KpiDef(
        "short_term_borrowings",
        "Short-term borrowings (bank lines, commercial paper)",
        "stock",
        "USD",
        ("ShortTermBorrowings",),
    ),
    # --- Working capital ---
    KpiDef("inventory", "Inventory", "stock", "USD", ("InventoryNet",)),
    KpiDef(
        "accounts_receivable",
        "Accounts receivable",
        "stock",
        "USD",
        ("AccountsReceivableNetCurrent",),
    ),
    KpiDef(
        "accounts_payable",
        "Accounts payable",
        "stock",
        "USD",
        ("AccountsPayableCurrent",),
    ),
    KpiDef(
        "shares_outstanding",
        "Shares outstanding",
        "stock",
        "shares",
        (
            "CommonStockSharesOutstanding",
            "EntityCommonStockSharesOutstanding",
        ),
    ),
    # --- Cash flow statement ---
    KpiDef(
        "operating_cash_flow",
        "Operating cash flow",
        "flow",
        "USD",
        (
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
    KpiDef(
        "investing_cash_flow",
        "Investing cash flow",
        "flow",
        "USD",
        ("NetCashProvidedByUsedInInvestingActivities",),
    ),
    KpiDef(
        "financing_cash_flow",
        "Financing cash flow",
        "flow",
        "USD",
        ("NetCashProvidedByUsedInFinancingActivities",),
    ),
    KpiDef(
        "capex",
        "Capital expenditure",
        "flow",
        "USD",
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ),
    ),
    KpiDef(
        "depreciation_amortization",
        "Depreciation & amortization",
        "flow",
        "USD",
        (
            "DepreciationDepletionAndAmortization",
            "DepreciationAndAmortization",
            "Depreciation",
        ),
    ),
    KpiDef(
        "dividends_paid",
        "Dividends paid",
        "flow",
        "USD",
        ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    ),
)


KPI_BY_KEY: dict[str, KpiDef] = {k.key: k for k in KPI_DEFS}
