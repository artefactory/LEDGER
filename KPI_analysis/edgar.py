"""SEC EDGAR client for XBRL company facts.

EDGAR exposes structured financial data for all US filers:
  - https://www.sec.gov/files/company_tickers.json        (ticker -> CIK)
  - https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json

SEC requires a descriptive User-Agent header with contact info (see
https://www.sec.gov/os/accessing-edgar-data) and enforces a 10 req/s limit.

This module:
  - caches ticker->CIK and companyfacts JSON on disk (cheap re-runs)
  - rate-limits to ~9 req/s
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


def _end_year(entry: dict[str, Any]) -> int | None:
    end = entry.get("end")
    if not end:
        return None
    try:
        return datetime.strptime(end, "%Y-%m-%d").year
    except ValueError:
        return None


# IMPORTANT: EDGAR's `fy` field is the fiscal year of the FILING that reported
# a given fact, not the fiscal year covered by the fact itself. When a 10-K
# includes prior-year comparatives, those comparatives inherit the filing's
# `fy`. We must therefore key on `end`-date year (the period the fact actually
# covers), not `fy`.


def _best_flow_entry_for_year(
    entries: list[dict[str, Any]], year: int
) -> dict[str, Any] | None:
    """Full-year flow entry whose period ends in `year`. Latest filed wins."""
    candidates = [
        e
        for e in entries
        if _is_annual_filing(e)
        and _end_year(e) == year
        and 340 <= (_span_days(e) or 0) <= 400
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.get("filed", ""))


def _best_stock_entry_for_year(
    entries: list[dict[str, Any]], year: int
) -> dict[str, Any] | None:
    """Balance-sheet entry as of an annual-report date in `year`. Latest filed wins."""
    candidates = [e for e in entries if _is_annual_filing(e) and _end_year(e) == year]
    if not candidates:
        return None
    return max(candidates, key=lambda e: (e.get("end", ""), e.get("filed", "")))


def extract_kpis_for_years(
    companyfacts: dict[str, Any],
    years: Iterable[int],
    kpi_defs: Iterable[KpiDef] = KPI_DEFS,
) -> tuple[dict[str, dict[int, float]], dict[str, str]]:
    """Return (values_by_kpi_year, tag_used_by_kpi).

    values[kpi_key][year] = numeric value.
    tag_used[kpi_key] = first XBRL tag that yielded any data (for audit).
    """
    years_list = list(years)
    values: dict[str, dict[int, float]] = {}
    tag_used: dict[str, str] = {}
    picker = {
        "flow": _best_flow_entry_for_year,
        "stock": _best_stock_entry_for_year,
    }
    for kpi in kpi_defs:
        pick = picker[kpi.kind]
        per_year: dict[int, float] = {}
        chosen_tag: str | None = None
        for tag in kpi.tags:
            entries = _iter_unit_entries(companyfacts, tag, kpi.unit)
            if not entries:
                continue
            for year in years_list:
                if year in per_year:
                    continue
                hit = pick(entries, year)
                if hit is not None:
                    per_year[year] = float(hit["val"])
                    if chosen_tag is None:
                        chosen_tag = tag
            # If we've covered every requested year via this tag, stop early.
            if len(per_year) == len(years_list):
                break
        if per_year:
            values[kpi.key] = per_year
            tag_used[kpi.key] = chosen_tag or kpi.tags[0]
    return values, tag_used


def fetch_kpis_for_ticker(
    ticker: str,
    years: Iterable[int],
    *,
    mapping: dict[str, str] | None = None,
    refresh: bool = False,
) -> dict[str, Any] | None:
    """High-level: ticker -> {'cik', 'kpis', 'tag_used'} or None if not on EDGAR."""
    cik = ticker_to_cik(ticker, mapping=mapping)
    if cik is None:
        return None
    facts = fetch_companyfacts(cik, refresh=refresh)
    if facts is None:
        return None
    values, tags = extract_kpis_for_years(facts, years)
    return {
        "cik": cik,
        "entity_name": facts.get("entityName"),
        "kpis": values,
        "tag_used": tags,
    }
