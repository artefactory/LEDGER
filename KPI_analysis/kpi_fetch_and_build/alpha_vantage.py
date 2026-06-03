"""Alpha Vantage fallback for tickers with low KPI coverage from EDGAR / yfinance.

Each AV statement endpoint (INCOME_STATEMENT / BALANCE_SHEET / CASH_FLOW) returns
the full multi-year annual history per call, so 3 calls cover ~25 KPIs per ticker.
With per-key quota of 25 requests/day and N keys we get 25*N calls/day; in practice
that's ~25*N/3 tickers/day (3 endpoints by default; EARNINGS is opt-in for EPS).

Multiple keys are loaded from a txt file (one per line, blanks and `#` comments
skipped). Keys can be added or removed without code changes — `load_keys` re-reads
the file each call.

Per-key per-day usage is tracked in `cache/alphavantage_budget.json`, keyed on UTC
date (matches AV's daily reset). Within a run we always pick the key with the
most remaining quota. Exhausting all keys raises `BudgetExhausted`.

Responses are cached on disk under `cache/alphavantage/{symbol}__{endpoint}.json`,
so re-runs are free; pass `refresh=True` to force a fresh call.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    from ._fiscal import filer_fy_from_string as _filer_fy_from_string
except ImportError:
    from _fiscal import filer_fy_from_string as _filer_fy_from_string

HERE = Path(__file__).resolve().parent
KPI_ROOT = HERE.parent
CACHE_ROOT = KPI_ROOT / "cache"
CACHE_DIR = CACHE_ROOT / "alphavantage"
BUDGET_PATH = CACHE_ROOT / "alphavantage_budget.json"
DEFAULT_KEYS_PATH = HERE / "alpha_venture_API_keys.txt"

ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"

# 3-statement minimum; EARNINGS is opt-in (one extra call per ticker).
DEFAULT_ENDPOINTS: tuple[str, ...] = ("INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW")
EARNINGS_ENDPOINT = "EARNINGS"

# AV free tier resets daily; intra-second rate is loose. Add a small floor to be polite.
DEFAULT_DAILY_QUOTA = 25
MIN_REQUEST_INTERVAL = 1.0  # seconds between live calls

# Yahoo-style suffix -> AV-style suffix. Pass-through if absent.
# AV's coverage of non-US markets is uneven; these are the suffixes most likely
# to resolve. Untranslated suffixes still get tried — empty payload is fine.
YF_TO_AV_SUFFIX: dict[str, str] = {
    "L": "LON",
    "AX": "AX",
    "TO": "TRT",
    "V": "TSXV",
    "PA": "PAR",
    "DE": "DEX",
    "MI": "MIL",
    "AS": "AMS",
    "BR": "BRU",
    "MC": "MCE",
    "ST": "STO",
    "HE": "HEL",
    "CO": "CPH",
    "OL": "OSL",
    "IR": "IRE",
}


def yf_to_av_symbol(ticker: str) -> str:
    if "." not in ticker:
        return ticker
    base, suffix = ticker.rsplit(".", 1)
    av_suffix = YF_TO_AV_SUFFIX.get(suffix.upper())
    if av_suffix is None:
        return ticker
    return f"{base}.{av_suffix}"


# ---------------------------------------------------------------------------
# Keys & per-key per-day budget
# ---------------------------------------------------------------------------


def load_keys(path: Path | None = None) -> list[str]:
    """One key per line; blank / `#` comment lines skipped; order preserved, deduped."""
    p = path or DEFAULT_KEYS_PATH
    if not p.exists():
        return []
    seen: set[str] = set()
    keys: list[str] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s in seen:
            continue
        seen.add(s)
        keys.append(s)
    return keys


def _key_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class BudgetTracker:
    daily_quota: int = DEFAULT_DAILY_QUOTA
    path: Path = BUDGET_PATH

    def _load(self) -> dict[str, dict[str, int]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict[str, dict[str, int]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def remaining(self, key: str, *, today: str | None = None) -> int:
        date = today or _today_utc()
        used = self._load().get(date, {}).get(_key_id(key), 0)
        return max(self.daily_quota - used, 0)

    def consume(self, key: str, *, today: str | None = None) -> None:
        date = today or _today_utc()
        data = self._load()
        bucket = data.setdefault(date, {})
        kid = _key_id(key)
        bucket[kid] = bucket.get(kid, 0) + 1
        self._save(data)

    def total_remaining(self, keys: Iterable[str]) -> int:
        return sum(self.remaining(k) for k in keys)

    def pick_key(self, keys: Iterable[str]) -> str:
        scored = sorted(((self.remaining(k), k) for k in keys), reverse=True)
        if not scored or scored[0][0] <= 0:
            raise BudgetExhausted("All Alpha Vantage keys are out of daily quota.")
        return scored[0][1]


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class _RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


_limiter = _RateLimiter(MIN_REQUEST_INTERVAL)


def _cache_path(symbol: str, endpoint: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}__{endpoint}.json"


def _looks_like_quota_message(payload: dict[str, Any]) -> bool:
    return any(k in payload for k in ("Note", "Information", "Error Message"))


def _is_quota_text(payload: dict[str, Any]) -> bool:
    msg = " ".join(str(v) for v in payload.values()).lower()
    return any(
        marker in msg
        for marker in ("rate limit", "premium", "exceeded", "thank you for using")
    )


def fetch_endpoint(
    symbol: str,
    endpoint: str,
    *,
    keys: list[str],
    budget: BudgetTracker,
    refresh: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Fetch one AV endpoint with disk cache and key rotation.

    Counts the call against the chosen key whether the response carries data or
    just a quota/error message — AV deducts quota in both cases. Empty-but-valid
    payloads are cached (means "no statements available", stable over time);
    quota-error payloads are NOT cached.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(symbol, endpoint)
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())
    if not keys:
        raise BudgetExhausted("No Alpha Vantage API keys loaded.")

    key = budget.pick_key(keys)
    _limiter.wait()
    params = {"function": endpoint, "symbol": symbol, "apikey": key}
    r = requests.get(ALPHAVANTAGE_URL, params=params, timeout=30)
    r.raise_for_status()
    try:
        payload = r.json()
    except json.JSONDecodeError:
        payload = {"Error Message": f"non-JSON response: {r.text[:200]}"}

    budget.consume(key)

    if _looks_like_quota_message(payload):
        if not quiet:
            print(
                f"[alphavantage] {symbol} {endpoint} -> quota/error: "
                f"{next(iter(payload.values()))[:200]}",
                file=sys.stderr,
            )
        if _is_quota_text(payload):
            raise BudgetExhausted(f"Alpha Vantage rate-limit response on {endpoint}")
        # Non-quota error (e.g. invalid symbol). Don't cache; let caller move on.
        return payload

    cache_file.write_text(json.dumps(payload))
    return payload


# ---------------------------------------------------------------------------
# Field map: KPI key -> ordered list of (endpoint, AV field name)
# ---------------------------------------------------------------------------

# Mirrors tags.py's waterfall idea: first numeric value wins. Scope choices match
# the conventions baked into tags.py / yf_fallback.py:
#   - net_income: AV's `netIncome` is parent-attributable (matches "Case 2")
#   - stockholders_equity: AV's `totalShareholderEquity` is parent-attributable
#   - capex: AV reports POSITIVE cash outflow (matches EDGAR sign convention).
#     yfinance reports negative — sign reconciliation is the caller's problem.
AV_KPI_MAP: dict[str, tuple[tuple[str, str], ...]] = {
    "revenue": (("INCOME_STATEMENT", "totalRevenue"),),
    "cost_of_revenue": (
        ("INCOME_STATEMENT", "costOfRevenue"),
        ("INCOME_STATEMENT", "costofGoodsAndServicesSold"),
    ),
    "gross_profit": (("INCOME_STATEMENT", "grossProfit"),),
    "rd_expense": (("INCOME_STATEMENT", "researchAndDevelopment"),),
    "sga_expense": (("INCOME_STATEMENT", "sellingGeneralAndAdministrative"),),
    "operating_income": (("INCOME_STATEMENT", "operatingIncome"),),
    "interest_expense": (
        ("INCOME_STATEMENT", "interestExpense"),
        ("INCOME_STATEMENT", "interestAndDebtExpense"),
    ),
    "income_tax_expense": (("INCOME_STATEMENT", "incomeTaxExpense"),),
    "net_income": (("INCOME_STATEMENT", "netIncome"),),
    "eps_basic": (("EARNINGS", "reportedEPS"),),
    "total_assets": (("BALANCE_SHEET", "totalAssets"),),
    "total_liabilities": (("BALANCE_SHEET", "totalLiabilities"),),
    "stockholders_equity": (("BALANCE_SHEET", "totalShareholderEquity"),),
    "cash_and_equivalents": (
        ("BALANCE_SHEET", "cashAndCashEquivalentsAtCarryingValue"),
    ),
    # AV's `longTermDebt` is filer-dependent (sometimes incl-current, sometimes not).
    # Best-effort total only; prefer the explicit `_noncurrent` / `_current` keys.
    "long_term_debt_total": (("BALANCE_SHEET", "longTermDebt"),),
    "long_term_debt_noncurrent": (("BALANCE_SHEET", "longTermDebtNoncurrent"),),
    "long_term_debt_current": (("BALANCE_SHEET", "currentLongTermDebt"),),
    "short_term_borrowings": (("BALANCE_SHEET", "shortTermDebt"),),
    "inventory": (("BALANCE_SHEET", "inventory"),),
    "accounts_receivable": (("BALANCE_SHEET", "currentNetReceivables"),),
    "accounts_payable": (("BALANCE_SHEET", "currentAccountsPayable"),),
    "shares_outstanding": (("BALANCE_SHEET", "commonStockSharesOutstanding"),),
    "operating_cash_flow": (("CASH_FLOW", "operatingCashflow"),),
    "investing_cash_flow": (("CASH_FLOW", "cashflowFromInvestment"),),
    "financing_cash_flow": (("CASH_FLOW", "cashflowFromFinancing"),),
    "capex": (("CASH_FLOW", "capitalExpenditures"),),
    "depreciation_amortization": (
        ("CASH_FLOW", "depreciationDepletionAndAmortization"),
        ("INCOME_STATEMENT", "depreciationAndAmortization"),
        ("INCOME_STATEMENT", "depreciation"),
    ),
    "dividends_paid": (
        ("CASH_FLOW", "dividendPayout"),
        ("CASH_FLOW", "dividendPayoutCommonStock"),
    ),
}


def required_endpoints(kpi_keys: Iterable[str] | None = None) -> set[str]:
    """The minimum set of AV endpoints needed to (potentially) cover `kpi_keys`."""
    if kpi_keys is None:
        kpi_keys = AV_KPI_MAP.keys()
    needed: set[str] = set()
    for k in kpi_keys:
        for ep, _ in AV_KPI_MAP.get(k, ()):
            needed.add(ep)
    return needed


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_value(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
        return None if v != v else v  # NaN check
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s.lower() in {"none", "-", "—"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _year_from_period_end(period: str | None) -> int | None:
    """Map AV's ``fiscalDateEnding`` to the filer's labelled FY.

    Delegates to ``_fiscal.filer_fy_from_string`` so we stay consistent with
    the EDGAR / yfinance ingestion paths (period-end Jan 2 2021 → FY2020 etc.).
    """
    return _filer_fy_from_string(period)


def extract_kpis_for_years(
    payloads: dict[str, dict[str, Any]],
    years: Iterable[int],
) -> tuple[dict[str, dict[int, float]], dict[str, str], dict[str, str]]:
    """Convert {endpoint: payload} into our standard KPI shape.

    Returns:
      values[kpi_key][year] = float
      tag_used[kpi_key]     = "alphavantage:<ENDPOINT>.<field>"
      currency[endpoint]    = reportedCurrency observed (per-endpoint)
    """
    years_set = {int(y) for y in years}
    by_ep_year: dict[str, dict[int, dict[str, float]]] = {}
    currency: dict[str, str] = {}

    for ep, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        reports_key = "annualEarnings" if ep == EARNINGS_ENDPOINT else "annualReports"
        reports = payload.get(reports_key) or []
        if reports and isinstance(reports[0], dict):
            cur = reports[0].get("reportedCurrency")
            if cur:
                currency[ep] = cur
        for report in reports:
            if not isinstance(report, dict):
                continue
            year = _year_from_period_end(report.get("fiscalDateEnding"))
            if year is None or year not in years_set:
                continue
            row: dict[str, float] = {}
            for fname, fval in report.items():
                v = _parse_value(fval)
                if v is None:
                    continue
                row[fname] = v
            by_ep_year.setdefault(ep, {})[year] = row

    values: dict[str, dict[int, float]] = {}
    tag_used: dict[str, str] = {}
    for kpi_key, sources in AV_KPI_MAP.items():
        per_year: dict[int, float] = {}
        first_tag: str | None = None
        for ep, field_name in sources:
            ep_data = by_ep_year.get(ep)
            if not ep_data:
                continue
            for year, row in ep_data.items():
                if year in per_year:
                    continue
                if field_name in row:
                    per_year[year] = row[field_name]
                    if first_tag is None:
                        first_tag = f"alphavantage:{ep}.{field_name}"
        if per_year:
            values[kpi_key] = per_year
            if first_tag:
                tag_used[kpi_key] = first_tag

    return values, tag_used, currency


# ---------------------------------------------------------------------------
# Top-level: fetch all configured endpoints for one ticker
# ---------------------------------------------------------------------------


@dataclass
class TickerResult:
    symbol_used: str
    kpis: dict[str, dict[int, float]] = field(default_factory=dict)
    tag_used: dict[str, str] = field(default_factory=dict)
    reported_currency: dict[str, str] = field(default_factory=dict)
    endpoints_called: list[str] = field(default_factory=list)
    endpoints_cached: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_used": self.symbol_used,
            "kpis": self.kpis,
            "tag_used": self.tag_used,
            "reported_currency": self.reported_currency,
            "endpoints_called": self.endpoints_called,
            "endpoints_cached": self.endpoints_cached,
            "error": self.error,
        }


def fetch_kpis_for_ticker(
    ticker: str,
    years: Iterable[int],
    *,
    keys: list[str],
    budget: BudgetTracker,
    endpoints: Iterable[str] = DEFAULT_ENDPOINTS,
    refresh: bool = False,
    quiet: bool = False,
) -> TickerResult:
    """Fetch every endpoint in `endpoints` for `ticker`, returning parsed KPIs.

    Stops early on `BudgetExhausted` or network errors but still returns the
    KPIs derivable from already-fetched endpoints.
    """
    av_symbol = yf_to_av_symbol(ticker)
    result = TickerResult(symbol_used=av_symbol)
    payloads: dict[str, dict[str, Any]] = {}
    for ep in endpoints:
        cache_file = _cache_path(av_symbol, ep)
        cached = cache_file.exists() and not refresh
        try:
            payloads[ep] = fetch_endpoint(
                av_symbol,
                ep,
                keys=keys,
                budget=budget,
                refresh=refresh,
                quiet=quiet,
            )
        except BudgetExhausted as exc:
            result.error = f"BudgetExhausted: {exc}"
            break
        except requests.RequestException as exc:
            result.error = f"network error on {ep}: {exc}"
            break
        if cached:
            result.endpoints_cached.append(ep)
        else:
            result.endpoints_called.append(ep)

    values, tag_used, currency = extract_kpis_for_years(payloads, years)
    result.kpis = values
    result.tag_used = tag_used
    result.reported_currency = currency
    return result
