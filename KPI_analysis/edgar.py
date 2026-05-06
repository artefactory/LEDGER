"""SEC EDGAR client for XBRL company facts.

EDGAR exposes structured financial data for all US filers:
  - https://www.sec.gov/files/company_tickers.json        (ticker -> CIK)
  - https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json

SEC requires a descriptive User-Agent header with contact info (see
https://www.sec.gov/os/accessing-edgar-data) and enforces a 10 req/s limit.

This module:
  - caches ticker->CIK and companyfacts JSON on disk (cheap re-runs)
  - rate-limits to ~3 req/s
  - extracts a configurable set of KPIs for a requested year range
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import requests

from _fiscal import filer_fy_from_string as _filer_fy_from_string
from tags import KPI_DEFS, KpiDef

CACHE_DIR = Path(__file__).resolve().parent / "cache"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_USER_AGENT = (
    "Artefact Research Center (dataset constitution) (charles.moslonka@artefact.com)"
)
MIN_REQUEST_INTERVAL = 0.4  # seconds; SEC allows 10 req/s but let's be nice


def _user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)


def _headers() -> dict[str, str]:
    return {"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"}


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


def _get_json(url: str) -> dict[str, Any]:
    _limiter.wait()
    r = requests.get(url, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


# --- Ticker -> CIK ----------------------------------------------------------


def load_ticker_cik_map(*, refresh: bool = False) -> dict[str, str]:
    """Return {ticker_upper: zero-padded-10-digit-CIK}. Cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "ticker_cik.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    raw = _get_json(TICKER_MAP_URL)
    mapping: dict[str, str] = {}
    # Raw format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
    for entry in raw.values():
        ticker = str(entry["ticker"]).upper()
        cik = f"{int(entry['cik_str']):010d}"
        mapping[ticker] = cik
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return mapping


def ticker_to_cik(ticker: str, mapping: dict[str, str] | None = None) -> str | None:
    m = mapping if mapping is not None else load_ticker_cik_map()
    return m.get(ticker.upper())


# --- Company facts ----------------------------------------------------------


def fetch_companyfacts(cik: str, *, refresh: bool = False) -> dict[str, Any] | None:
    """Fetch and cache companyfacts JSON. Returns None on 404."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    facts_dir = CACHE_DIR / "companyfacts"
    facts_dir.mkdir(exist_ok=True)
    path = facts_dir / f"CIK{cik}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    url = COMPANYFACTS_URL.format(cik=cik)
    _limiter.wait()
    r = requests.get(url, headers=_headers(), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data))
    return data


# --- KPI extraction ---------------------------------------------------------


@dataclass
class KpiFact:
    value: float
    fy: int
    form: str
    filed: str
    tag: str
    start: str | None
    end: str | None


def _iter_unit_entries(
    facts: dict[str, Any], tag: str, unit: str
) -> list[dict[str, Any]]:
    ns = facts.get("facts", {}).get("us-gaap", {})
    tag_data = ns.get(tag)
    if not tag_data:
        return []
    units = tag_data.get("units", {})
    return units.get(unit, []) or []


def _is_annual_filing(entry: dict[str, Any]) -> bool:
    form = entry.get("form", "")
    fp = entry.get("fp", "")
    # Annual reports: 10-K (or amendment 10-K/A); foreign private issuers file 20-F.
    return fp == "FY" and (form.startswith("10-K") or form.startswith("20-F"))


def _span_days(entry: dict[str, Any]) -> int | None:
    start, end = entry.get("start"), entry.get("end")
    if not start or not end:
        return None
    try:
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (d1 - d0).days


def _filer_fy(entry: dict[str, Any]) -> int | None:
    """Return the filer's labelled FY for a fact, derived from its period-end.

    See ``KPI_analysis/_fiscal.py`` for the rule. We deliberately do NOT use
    EDGAR's ``fy`` field: that's the fiscal year of the FILING that reported
    the fact, so a comparative prior-year line in a 10-K inherits the
    filing's ``fy`` rather than its own period's FY label. Deriving from
    ``end`` via the shared helper sidesteps that ambiguity.
    """
    return _filer_fy_from_string(entry.get("end"))


def _best_flow_entry_for_year(
    entries: list[dict[str, Any]], year: int
) -> dict[str, Any] | None:
    """Full-year flow entry whose period belongs to filer-FY ``year``. Latest filed wins."""
    candidates = [
        e
        for e in entries
        if _is_annual_filing(e)
        and _filer_fy(e) == year
        and 340 <= (_span_days(e) or 0) <= 400
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.get("filed", ""))


def _best_stock_entry_for_year(
    entries: list[dict[str, Any]], year: int
) -> dict[str, Any] | None:
    """Balance-sheet entry as of an annual-report date in filer-FY ``year``. Latest filed wins."""
    candidates = [e for e in entries if _is_annual_filing(e) and _filer_fy(e) == year]
    if not candidates:
        return None
    return max(candidates, key=lambda e: (e.get("end", ""), e.get("filed", "")))


# Relative difference threshold above which two candidate-tag values are
# considered "scope-mismatched" (not just rounding/synonym noise).
AMBIGUITY_REL_THRESHOLD = 0.001  # 0.1%


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b))
    if denom == 0:
        return 0.0
    return abs(a - b) / denom


def extract_kpis_for_years(
    companyfacts: dict[str, Any],
    years: Iterable[int],
    kpi_defs: Iterable[KpiDef] = KPI_DEFS,
) -> tuple[
    dict[str, dict[int, float]],
    dict[str, str],
    dict[str, dict[int, dict[str, float]]],
]:
    """Return (values, tag_used, ambiguous).

    values[kpi_key][year] = chosen numeric value.
    tag_used[kpi_key] = the XBRL tag (or synthetic "sum:<a>+<b>") the value came from.
    ambiguous[kpi_key][year] = {tag: val} when >=2 candidate tags gave materially
        different values for the same period — for audit only.
    """
    years_list = list(years)
    years_set = set(years_list)
    values: dict[str, dict[int, float]] = {}
    tag_used: dict[str, str] = {}
    ambiguous: dict[str, dict[int, dict[str, float]]] = {}
    picker = {
        "flow": _best_flow_entry_for_year,
        "stock": _best_stock_entry_for_year,
    }

    for kpi in kpi_defs:
        pick = picker[kpi.kind]
        per_year: dict[int, float] = {}
        per_year_tag: dict[int, str] = {}
        # Survey: for each candidate tag, collect year -> value even if we
        # ultimately pick a different tag. Used both for value selection
        # (waterfall) and ambiguity detection.
        survey: dict[str, dict[int, float]] = {}
        for tag in kpi.tags:
            entries = _iter_unit_entries(companyfacts, tag, kpi.unit)
            if not entries:
                continue
            for year in years_list:
                hit = pick(entries, year)
                if hit is not None:
                    survey.setdefault(tag, {})[year] = float(hit["val"])

        # Waterfall selection: first tag in `tags` that has data for a given
        # year wins that year.
        for tag in kpi.tags:
            vals = survey.get(tag)
            if not vals:
                continue
            for year, v in vals.items():
                if year not in per_year:
                    per_year[year] = v
                    per_year_tag[year] = tag

        # Summation fallback for years still missing. Each tag in a component
        # set may be prefixed with "-" to indicate subtraction, allowing
        # balance-sheet identities like `Assets - Equity = Liabilities`.
        for component_set in kpi.sum_components:
            missing = [y for y in years_list if y not in per_year]
            if not missing:
                break
            # Parse signs.
            signed: list[tuple[int, str]] = [
                ((-1, c[1:]) if c.startswith("-") else (1, c)) for c in component_set
            ]
            # Build survey for each underlying tag so we can check "all present".
            comp_survey: dict[str, dict[int, float]] = {}
            for _sign, c_tag in signed:
                entries = _iter_unit_entries(companyfacts, c_tag, kpi.unit)
                if not entries:
                    continue
                for year in missing:
                    hit = pick(entries, year)
                    if hit is not None:
                        comp_survey.setdefault(c_tag, {})[year] = float(hit["val"])
            for year in missing:
                if all(
                    c_tag in comp_survey and year in comp_survey[c_tag]
                    for _sign, c_tag in signed
                ):
                    total = sum(
                        sign * comp_survey[c_tag][year] for sign, c_tag in signed
                    )
                    per_year[year] = total
                    expr = "".join(
                        ("-" if sign < 0 else ("+" if i else "")) + c_tag
                        for i, (sign, c_tag) in enumerate(signed)
                    )
                    per_year_tag[year] = "sum:" + expr

        # Ambiguity detection: for each year where >=2 candidate tags gave
        # a value, flag if they disagree beyond the threshold.
        for year in years_set:
            hits = {tag: vals[year] for tag, vals in survey.items() if year in vals}
            if len(hits) < 2:
                continue
            chosen = per_year.get(year)
            if chosen is None:
                continue
            # Normalize against the chosen value.
            disagree = {
                tag: val
                for tag, val in hits.items()
                if _rel_diff(val, chosen) > AMBIGUITY_REL_THRESHOLD
            }
            if disagree:
                # Include the chosen tag+value alongside the disagreeing ones.
                entry = {per_year_tag[year]: chosen, **disagree}
                ambiguous.setdefault(kpi.key, {})[year] = entry

        if per_year:
            values[kpi.key] = per_year
            # `tag_used` records the tag of the first year populated (stable
            # picked definition). `per_year_tag` in the survey has the exact
            # per-year tag for fine-grained audit if ever needed.
            first_year = min(per_year_tag)
            tag_used[kpi.key] = per_year_tag[first_year]

    return values, tag_used, ambiguous


def fetch_kpis_for_ticker(
    ticker: str,
    years: Iterable[int],
    *,
    mapping: dict[str, str] | None = None,
    refresh: bool = False,
) -> dict[str, Any] | None:
    """High-level: ticker -> dict with cik/kpis/tag_used/ambiguous_tags, or None."""
    cik = ticker_to_cik(ticker, mapping=mapping)
    if cik is None:
        return None
    facts = fetch_companyfacts(cik, refresh=refresh)
    if facts is None:
        return None
    values, tags, ambiguous = extract_kpis_for_years(facts, years)
    return {
        "cik": cik,
        "entity_name": facts.get("entityName"),
        "kpis": values,
        "tag_used": tags,
        "ambiguous_tags": ambiguous,
    }
