"""SEC EDGAR submissions client for filing-date metadata.

EDGAR exposes per-CIK filings history at
  https://data.sec.gov/submissions/CIK{CIK}.json

The JSON has `filings.recent.{accessionNumber, filingDate, reportDate,
acceptanceDateTime, form, primaryDocument, isXBRL, ...}` parallel arrays.
Older filings beyond the most recent ~1000 are sharded into
`filings.files[]`, each referencing a sibling JSON like
`CIK{cik}-submissions-001.json`.

This module fetches and caches the main + shard JSONs, parses the parallel
arrays into `Filing` records, and exposes a helper to find the *original*
10-K (not 10-K/A) for a given fiscal year.

A note on `acceptanceDateTime` timezone
---------------------------------------
The string carries a trailing "Z" (e.g. `2025-10-27T20:37:35.000Z`) but
is **actually in UTC** (the Z is honest). Verified against AAPL's history:
their FY2016 10-K shows `2016-10-26T20:42:16.000Z` with
`filingDate=2016-10-26`; if 20:42 were ET, it would be past SEC's 5:30 PM ET
cutoff and EDGAR would roll the filingDate to Oct 27 — which it doesn't.
As UTC, 20:42 → 4:42 PM ET, before the cutoff, matching filingDate=Oct 26.
We therefore store acceptance as a tz-aware UTC datetime and convert to ET
only when comparing against market hours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from edgar import CACHE_DIR, _headers, _limiter

ET = ZoneInfo("America/New_York")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SHARD_URL = "https://data.sec.gov/submissions/{name}"

ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A"}
ORIGINAL_ANNUAL_FORMS = {"10-K", "20-F"}


@dataclass
class Filing:
    accession: str
    form: str
    filing_date: str  # YYYY-MM-DD (per SEC's filing-date convention, ET)
    report_date: str  # YYYY-MM-DD; period of report (fiscal year-end date)
    acceptance_dt_utc: datetime  # tz-aware UTC
    primary_document: str | None
    is_xbrl: bool


def _parse_acceptance_dt(s: str) -> datetime:
    """Parse acceptanceDateTime as UTC.

    Format examples: '2025-10-27T20:37:35.000Z', '2018-11-05T13:01:40.000Z'.
    """
    if s.endswith("Z"):
        s = s[:-1]
    if "." in s:
        s = s.split(".", 1)[0]
    naive = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    return naive.replace(tzinfo=timezone.utc)


def _parse_filings_block(block: dict[str, Any]) -> list[Filing]:
    """Parse a `filings.recent` or shard block into `Filing` objects.

    Skips rows missing core fields; tolerates partial-length parallel arrays.
    """
    out: list[Filing] = []
    n = len(block.get("accessionNumber", []))
    forms = block.get("form", [])
    accepts = block.get("acceptanceDateTime", [])
    filing_dates = block.get("filingDate", [])
    report_dates = block.get("reportDate", [])
    primary_docs = block.get("primaryDocument", [])
    is_xbrls = block.get("isXBRL", [])
    for i in range(n):
        try:
            accept = _parse_acceptance_dt(accepts[i])
        except (ValueError, IndexError, TypeError):
            continue
        try:
            out.append(
                Filing(
                    accession=block["accessionNumber"][i],
                    form=forms[i] if i < len(forms) else "",
                    filing_date=filing_dates[i] if i < len(filing_dates) else "",
                    report_date=report_dates[i] if i < len(report_dates) else "",
                    acceptance_dt_utc=accept,
                    primary_document=primary_docs[i] if i < len(primary_docs) else None,
                    is_xbrl=bool(is_xbrls[i]) if i < len(is_xbrls) else False,
                )
            )
        except (IndexError, KeyError):
            continue
    return out


def _submissions_dir() -> Path:
    p = CACHE_DIR / "submissions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_submissions(cik: str, *, refresh: bool = False) -> dict[str, Any] | None:
    """Fetch & cache the main submissions JSON for a CIK. None on 404."""
    path = _submissions_dir() / f"CIK{cik}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    url = SUBMISSIONS_URL.format(cik=cik)
    _limiter.wait()
    r = requests.get(url, headers=_headers(), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data))
    return data


def fetch_submission_shard(name: str, *, refresh: bool = False) -> dict[str, Any] | None:
    """Fetch & cache an older-filings shard JSON. None on 404."""
    path = _submissions_dir() / name
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    url = SHARD_URL.format(name=name)
    _limiter.wait()
    r = requests.get(url, headers=_headers(), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data))
    return data


def all_annual_filings(cik: str, *, refresh: bool = False) -> list[Filing]:
    """All 10-K-family filings (10-K, 10-K/A, 20-F, 20-F/A) for a CIK,
    across recent + sharded blocks, sorted oldest-first by acceptance time.
    """
    base = fetch_submissions(cik, refresh=refresh)
    if base is None:
        return []
    filings = _parse_filings_block(base.get("filings", {}).get("recent", {}))
    for shard_meta in (base.get("filings", {}).get("files") or []):
        shard = fetch_submission_shard(shard_meta["name"], refresh=refresh)
        if shard:
            # Shards have the same parallel-array shape as `recent`, just at
            # the top level rather than under a `recent` key.
            filings.extend(_parse_filings_block(shard))
    annual = [f for f in filings if f.form in ANNUAL_FORMS]
    annual.sort(key=lambda f: f.acceptance_dt_utc)
    return annual


def find_original_10k(
    filings: list[Filing], fiscal_year: int
) -> tuple[Filing | None, bool]:
    """Pick the original 10-K (or 20-F) for `fiscal_year`, where year is
    keyed on the calendar year of the period-of-report (matching
    `fetch_kpis.py`'s convention).

    Returns (filing, has_amendment).

    Selection rule (must stay consistent with `_best_*_entry_for_year` in
    `edgar.py`, otherwise filing_returns.csv and kpis_long.csv get
    mis-joined for 52/53-week filers):
      1. Filter originals to those whose `report_date` year matches.
      2. Pick the *latest* `report_date` — for filers where two fiscal
         years end in the same calendar year (e.g. AAP's fiscal-2021
         ending 2022-01-01 and fiscal-2022 ending 2022-12-31), the KPI
         picker uses the later period-end, so we mirror that choice.
      3. Within the same `report_date` (i.e. true re-filings of one
         period), pick the *earliest* acceptance — that is the genuine
         first publication.
    """
    matching: list[Filing] = []
    has_amendment = False
    for f in filings:
        if not f.report_date or len(f.report_date) < 4 or not f.report_date[:4].isdigit():
            continue
        if int(f.report_date[:4]) != fiscal_year:
            continue
        if f.form in ORIGINAL_ANNUAL_FORMS:
            matching.append(f)
        elif f.form in ANNUAL_FORMS:
            has_amendment = True
    if not matching:
        return None, has_amendment
    latest_report_date = max(f.report_date for f in matching)
    candidates = [f for f in matching if f.report_date == latest_report_date]
    candidates.sort(key=lambda f: f.acceptance_dt_utc)
    return candidates[0], has_amendment


def acceptance_in_et(f: Filing) -> datetime:
    """Convenience: `f.acceptance_dt_utc` in Eastern time (DST-aware)."""
    return f.acceptance_dt_utc.astimezone(ET)
