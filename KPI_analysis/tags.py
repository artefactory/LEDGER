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

One problem with waterfall approach here:
If multiple tags are non-empty (say CostOfGoodsSold and CostOfServices), we pick the first one we encounter.
maybe we should check if multiple ?
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KpiDef:
    key: str
    label: str  # human readable name of KPI
    kind: str  # "flow" | "stock"
    unit: str
    tags: tuple[str, ...]


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
        "net_income",
        "Net income",
        "flow",
        "USD",
        (
            "NetIncomeLoss",
            "ProfitLoss",
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
    ),
    KpiDef(
        "stockholders_equity",
        "Stockholders' equity",
        "stock",
        "USD",
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ),
    KpiDef(
        "cash_and_equivalents",
        "Cash & equivalents",
        "stock",
        "USD",
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
    ),
    KpiDef(
        "long_term_debt",
        "Long-term debt",
        "stock",
        "USD",
        ("LongTermDebt", "LongTermDebtNoncurrent"),
    ),
    KpiDef(
        "short_term_debt",
        "Short-term debt",
        "stock",
        "USD",
        (
            "ShortTermBorrowings",
            "LongTermDebtCurrent",
            "DebtCurrent",
        ),
    ),
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
