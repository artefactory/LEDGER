"""yfinance fallback for tickers not on SEC EDGAR (e.g. LSE, AIM, ASX).

yfinance typically exposes only ~4 fiscal years of annual financials, which is
thinner than EDGAR but covers our later years. Each call returns a pandas
DataFrame indexed by row labels with period-end dates as columns.

We map a small set of yfinance row labels to our KPI keys. Labels are not 100%
stable; see _LABEL_MAP for the current mapping. Unknown labels are silently
ignored.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import yfinance as yf

try:
    from ._fiscal import filer_fy_from_period_end
except ImportError:
    from _fiscal import filer_fy_from_period_end


# Map yfinance label -> our KPI key. yfinance has slightly different labels for
# different statement types; keys reflect what we've observed for recent filings.
# yfinance labels are scope-ambiguous in the same way as XBRL tags; the mapping
# below commits to specific scopes — see edgar.py / tags.py for the conventions.
_LABEL_MAP: dict[str, str] = {
    # Income statement
    "Total Revenue": "revenue",
    "Cost Of Revenue": "cost_of_revenue",
    "Gross Profit": "gross_profit",
    "Research And Development": "rd_expense",
    "Selling General And Administration": "sga_expense",
    "Operating Income": "operating_income",
    "Interest Expense": "interest_expense",
    "Tax Provision": "income_tax_expense",
    "Net Income": "net_income",  # yfinance reports parent-attributable by default
    "Basic EPS": "eps_basic",
    "Diluted EPS": "eps_diluted",
    # Balance sheet
    "Total Assets": "total_assets",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Stockholders Equity": "stockholders_equity",
    "Cash And Cash Equivalents": "cash_and_equivalents",
    # yfinance "Long Term Debt" = noncurrent portion; "Current Debt" = current
    # portion of long-term debt. Matches the split we defined in tags.py.
    "Long Term Debt": "long_term_debt_noncurrent",
    "Current Debt": "long_term_debt_current",
    "Inventory": "inventory",
    "Accounts Receivable": "accounts_receivable",
    "Accounts Payable": "accounts_payable",
    "Share Issued": "shares_outstanding",
    # Cash flow statement
    "Operating Cash Flow": "operating_cash_flow",
    "Investing Cash Flow": "investing_cash_flow",
    "Financing Cash Flow": "financing_cash_flow",
    "Capital Expenditure": "capex",
    "Depreciation And Amortization": "depreciation_amortization",
    "Cash Dividends Paid": "dividends_paid",
}


def _extract_from_frame(
    df: pd.DataFrame, years: set[int], out: dict[str, dict[int, float]]
) -> None:
    if df is None or df.empty:
        return
    for label in df.index:
        key = _LABEL_MAP.get(str(label))
        if key is None:
            continue
        row = df.loc[label]
        bucket = out.setdefault(key, {})
        for col, val in row.items():
            # `col` is typically a pandas Timestamp at the period-end date.
            # Convert to filer-labelled FY (matches edgar.py / kpis_long.csv).
            try:
                period_end = col.date() if hasattr(col, "date") else None
            except Exception:
                period_end = None
            if period_end is None:
                continue
            year = filer_fy_from_period_end(period_end)
            if year not in years:
                continue
            if pd.isna(val):
                continue
            # If already populated (e.g. capex appearing in both cashflow and summary),
            # keep the first non-null value seen.
            bucket.setdefault(int(year), float(val))


def fetch_kpis_for_ticker(
    ticker: str, years: Iterable[int]
) -> dict[str, Any] | None:
    """Return {'kpis': {kpi_key: {year: val}}, 'tag_used': {kpi_key: 'yfinance:<label>'}}."""
    years_set = set(years)
    tk = yf.Ticker(ticker)
    values: dict[str, dict[int, float]] = {}
    try:
        _extract_from_frame(tk.income_stmt, years_set, values)
        _extract_from_frame(tk.balance_sheet, years_set, values)
        _extract_from_frame(tk.cashflow, years_set, values)
    except Exception as exc:  # yfinance raises various errors for delisted tickers
        return {"error": f"yfinance error: {exc!s}", "kpis": {}, "tag_used": {}}
    if not values:
        return None
    # Reverse-map: every KPI that has data came from the label keyed to it in _LABEL_MAP.
    inv: dict[str, str] = {}
    for label, key in _LABEL_MAP.items():
        inv.setdefault(key, f"yfinance:{label}")
    tag_used = {k: inv[k] for k in values if k in inv}
    return {"kpis": values, "tag_used": tag_used}
