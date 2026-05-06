"""Single source of truth: map a fiscal-period-end date to the filer's
labelled fiscal year.

The whole pipeline keys KPI values, filings, and OCR-directory joins by the
filer's labelled fiscal year — i.e. what the company itself prints on the
10-K cover (and what the OCR PDF source uses for directory naming
``EXCHANGE_TICKER_YEAR/``). For filers whose fiscal year cleanly aligns with
a calendar year (e.g. December year-end) this equals the year of the
period-end date. For 52/53-week filers (AAP, COST, AZO, ...) whose fiscal
years end in the first days of the next calendar year, the filer's labelled
FY is one less than the period-end year.

Rule
----
``filer_fy = end.year - 1 if end.month <= 3 else end.year``

The threshold is chosen so:
- Period ending Dec 30 2019 (AAP FY2019)         → 2019  ✓
- Period ending Jan  2 2021 (AAP FY2020, 53-wk)  → 2020  ✓
- Period ending Jan  1 2022 (AAP FY2021, 52-wk)  → 2021  ✓
- Period ending Dec 31 2022 (AAP FY2022)         → 2022  ✓
- Period ending Aug 28 2021 (AZO FY2021)         → 2021  ✓
- Period ending Sep 24 2022 (AAPL FY2022)        → 2022  ✓
- Period ending Mar 31 2019 (UK March-ending)    → 2018  ← year the FY started

The Mar-ending case (year-1) matches the convention "FY label is the calendar
year in which most of the fiscal year fell" — for a March year-end, the
fiscal year ran April→March of the following year, so 9 of its 12 months
fall in the earlier calendar year.

Why not use EDGAR's `fy` field directly?
EDGAR's ``fy`` is the fiscal year of the *filing*, not the period covered by
the fact: a comparative prior-year line in the FY2020 10-K carries
``fy=2020`` even though it describes the FY2019 period. Deriving FY from
period-end with a deterministic rule sidesteps that ambiguity entirely.
"""

from __future__ import annotations

from datetime import date, datetime


def filer_fy_from_period_end(end: date | datetime | str) -> int:
    """Return the filer's labelled fiscal year for a given period-end date.

    Accepts a ``date``/``datetime``, or a string in ``YYYY-MM-DD`` form.
    Raises ``ValueError`` for unparseable strings.
    """
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()
    elif isinstance(end, datetime):
        end = end.date()
    if end.month <= 3:
        return end.year - 1
    return end.year


def filer_fy_from_string(end: str | None) -> int | None:
    """As above, but tolerant of None / unparseable strings (returns None)."""
    if not end:
        return None
    try:
        return filer_fy_from_period_end(end)
    except ValueError:
        return None
