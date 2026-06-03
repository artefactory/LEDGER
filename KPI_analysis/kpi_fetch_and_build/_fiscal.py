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
``filer_fy = end.year - 1 if end.month == 1 else end.year``

The threshold is chosen so:
- Period ending Dec 30 2019 (AAP FY2019)         → 2019  ✓
- Period ending Jan  2 2021 (AAP FY2020, 53-wk)  → 2020  ✓
- Period ending Jan  1 2022 (AAP FY2021, 52-wk)  → 2021  ✓
- Period ending Dec 31 2022 (AAP FY2022)         → 2022  ✓
- Period ending Aug 28 2021 (AZO FY2021)         → 2021  ✓
- Period ending Sep 24 2022 (AAPL FY2022)        → 2022  ✓
- Period ending Mar 28 2020 (MNRO "fiscal 2020") → 2020  ✓ (US convention: label = year of period end)
- Period ending Mar 31 2019 (MOD "fiscal 2019")  → 2019  ✓
- Period ending Mar 31 2020 (MPAA "fiscal 2020") → 2020  ✓

US filers whose fiscal year ends in late March (MNRO, MOD, MPAA, ...) state
explicitly in their 10-Ks that "references to a particular year mean the
fiscal year ended March 31 of that year". So for any period-end in February
or later, the label is the period-end's calendar year. Only the early-Jan
52/53-week case (AAP/COST/AZO-style) carries the year-1 offset, and the rule
keys off ``month == 1`` to capture exactly those filers.

UK filers with March year-ends are not covered by this rule — they vary in
labelling convention (some by start year, some by end year). EDGAR has no
data for them anyway; yfinance fallback receives whatever date yfinance
returns and is keyed via this same helper.

Why not use EDGAR's `fy` field directly?
EDGAR's ``fy`` is the fiscal year of the *filing*, not the period covered by
the fact: a comparative prior-year line in the FY2020 10-K carries
``fy=2020`` even though it describes the FY2019 period. Deriving FY from
period-end with a deterministic rule sidesteps that ambiguity entirely.
(``min(fy)`` per period_end across all entries works as a check on the
rule but isn't used as the primary derivation.)
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
    if end.month == 1:
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
