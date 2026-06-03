"""Compute market reactions to annual-report (10-K) publications.

For each (US-listed ticker, fiscal year) we:
  1. Find the *original* 10-K filing on EDGAR (10-K, not 10-K/A) whose
     period-of-report falls in the target fiscal year, and read its
     `acceptanceDateTime` (the moment EDGAR accepted the submission, in UTC).
  2. Fetch yfinance daily prices once per ticker (cached on disk).
  3. Classify the filing as pre_market / intraday / after_hours / non_trading_day
     based on the ET-local time of acceptance vs market hours (9:30 - 16:00 ET).
  4. Anchor the event window:
       t0 = last trading day with a close BEFORE the filing was public,
       t1 = first trading day with a close AFTER the filing was public,
       t5 = t1 + 4 trading days (so a 5-trading-day window from t0).
  5. Compute raw returns r_1d = Close[t1]/Close[t0] - 1, r_5d analogously.
  6. Compute SPY-relative alpha a_Nd = r_Nd - spy_r_Nd over the same days.

Outputs:
  output/filing_returns.csv

Caches:
  cache/submissions/CIK{...}.json (+ shards)
  cache/prices/{ticker}.csv

Non-US listings (LSE, AIM, ASX, ...) have no equivalent EDGAR filing-date
record, so they're skipped by default. Pass --include-non-us to attempt them
anyway (they will all surface "no CIK"; useful as a no-op smoke test).

Usage:
  uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_filing_returns --selected --years 2017-2022
  uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_filing_returns --tickers AZO ORLY AAP
  uv run python -m KPI_analysis.kpi_fetch_and_build.fetch_filing_returns --industry "Consumer Cyclical / Auto Parts"
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

try:
    from . import edgar
    from . import edgar_filings as ef
    from .fetch_kpis import (
        US_EXCHANGES,
        parse_year_range,
        tickers_from_args,
        tickers_from_csv,
        tickers_from_selected,
    )
except ImportError:
    import edgar
    import edgar_filings as ef
    from fetch_kpis import (
        US_EXCHANGES,
        parse_year_range,
        tickers_from_args,
        tickers_from_csv,
        tickers_from_selected,
    )

ET = ZoneInfo("America/New_York")

HERE = Path(__file__).resolve().parent
KPI_ROOT = HERE.parent
OUTPUT_DIR = KPI_ROOT / "output"
CACHE_DIR = KPI_ROOT / "cache"
PRICES_CACHE = CACHE_DIR / "prices"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_OUT_CSV = OUTPUT_DIR / "filing_returns.csv"

MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)


# --- Price fetching with caching --------------------------------------------


def _price_cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace(":", "_")
    return PRICES_CACHE / f"{safe}.csv"


def fetch_prices(
    ticker: str,
    start: date,
    end: date,
    *,
    refresh: bool = False,
) -> pd.DataFrame | None:
    """Daily auto-adjusted OHLC for `ticker`, disk-cached as CSV.

    The cache covers a wider span than requested (start - 30d, end + 15d) so
    boundary t0 / t5 lookups never run off the edge of the cached window.
    Cache hits are returned slice-free; the caller should filter by date.
    """
    PRICES_CACHE.mkdir(parents=True, exist_ok=True)
    path = _price_cache_path(ticker)
    fetch_start = start - timedelta(days=30)
    fetch_end = end + timedelta(days=15)
    if path.exists() and not refresh:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if (
                not df.empty
                and df.index.min().date() <= fetch_start
                and df.index.max().date() >= fetch_end - timedelta(days=15)
            ):
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception:
            pass
    try:
        df = yf.Ticker(ticker).history(
            start=fetch_start.isoformat(),
            end=(fetch_end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            actions=False,
        )
    except Exception as e:
        print(f"  yfinance error for {ticker}: {e}", file=sys.stderr)
        return None
    if df is None or df.empty:
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.to_csv(path)
    return df


# --- Event-window calculation -----------------------------------------------


@dataclass
class EventWindow:
    t0: pd.Timestamp  # last trading-day close BEFORE news public
    t1: pd.Timestamp  # first trading-day close AFTER news public
    t5: pd.Timestamp | None  # t1 + 4 trading days (None if not enough history)
    filing_window_class: str  # 'pre_market' / 'intraday' / 'after_hours' / 'non_trading_day'


def classify_filing_window(
    accept_utc: datetime, prices_idx: pd.DatetimeIndex
) -> EventWindow | None:
    """Locate t0 / t1 / t5 around `accept_utc` against a sorted DatetimeIndex
    of trading days for the ticker.

    Convention:
      - pre-market or intraday filing on trading day D     -> t1 = D's close
      - after-hours filing on trading day D                -> t1 = next trading day
      - filing on weekend/holiday                          -> t1 = next trading day
      - t0 always the trading day immediately before t1
      - t5 = t1 + 4 trading days
    """
    accept_et = accept_utc.astimezone(ET)
    accept_date = accept_et.date()
    accept_t = accept_et.time()

    is_trading_day = pd.Timestamp(accept_date) in prices_idx

    if not is_trading_day:
        cls = "non_trading_day"
        t1_target = accept_date + timedelta(days=1)
    elif accept_t < MARKET_OPEN_ET:
        cls = "pre_market"
        t1_target = accept_date
    elif accept_t <= MARKET_CLOSE_ET:
        cls = "intraday"
        t1_target = accept_date
    else:
        cls = "after_hours"
        t1_target = accept_date + timedelta(days=1)

    # First trading day at-or-after t1_target.
    t1_pos = prices_idx.searchsorted(pd.Timestamp(t1_target), side="left")
    if t1_pos >= len(prices_idx):
        return None
    t1 = prices_idx[t1_pos]

    # Trading day immediately before t1.
    t0_pos = t1_pos - 1
    if t0_pos < 0:
        return None
    t0 = prices_idx[t0_pos]

    # Four trading days after t1.
    t5_pos = t1_pos + 4
    t5 = prices_idx[t5_pos] if t5_pos < len(prices_idx) else None

    return EventWindow(t0=t0, t1=t1, t5=t5, filing_window_class=cls)


# --- Per-(ticker, year) computation ----------------------------------------


@dataclass
class FilingReturnRow:
    ticker: str
    year: int
    accession: str | None
    form: str | None
    filing_date: str | None
    report_date: str | None
    acceptance_dt_utc: str | None
    acceptance_dt_et: str | None
    filing_window_class: str | None
    has_amendment: bool
    t0: str | None
    t1: str | None
    t5: str | None
    close_t0: float | None
    close_t1: float | None
    close_t5: float | None
    r_1d: float | None
    r_5d: float | None
    spy_r_1d: float | None
    spy_r_5d: float | None
    a_1d: float | None
    a_5d: float | None
    error: str | None


def compute_filing_return(
    ticker: str,
    year: int,
    filings: list[ef.Filing],
    prices: pd.DataFrame,
    spy_prices: pd.DataFrame | None,
) -> FilingReturnRow:
    filing, has_amendment = ef.find_original_10k(filings, year)
    if filing is None:
        return FilingReturnRow(
            ticker=ticker, year=year, accession=None, form=None,
            filing_date=None, report_date=None,
            acceptance_dt_utc=None, acceptance_dt_et=None,
            filing_window_class=None, has_amendment=has_amendment,
            t0=None, t1=None, t5=None,
            close_t0=None, close_t1=None, close_t5=None,
            r_1d=None, r_5d=None,
            spy_r_1d=None, spy_r_5d=None,
            a_1d=None, a_5d=None,
            error=f"no original 10-K for FY{year}",
        )

    window = classify_filing_window(filing.acceptance_dt_utc, prices.index)
    if window is None:
        return FilingReturnRow(
            ticker=ticker, year=year, accession=filing.accession, form=filing.form,
            filing_date=filing.filing_date, report_date=filing.report_date,
            acceptance_dt_utc=filing.acceptance_dt_utc.isoformat(),
            acceptance_dt_et=ef.acceptance_in_et(filing).isoformat(),
            filing_window_class=None, has_amendment=has_amendment,
            t0=None, t1=None, t5=None,
            close_t0=None, close_t1=None, close_t5=None,
            r_1d=None, r_5d=None,
            spy_r_1d=None, spy_r_5d=None,
            a_1d=None, a_5d=None,
            error="insufficient price history around filing",
        )

    def _close_at(idx: pd.DatetimeIndex, frame: pd.DataFrame, ts: pd.Timestamp) -> float | None:
        if ts is None or ts not in idx:
            return None
        try:
            return float(frame.loc[ts, "Close"])
        except (KeyError, ValueError, TypeError):
            return None

    close_t0 = _close_at(prices.index, prices, window.t0)
    close_t1 = _close_at(prices.index, prices, window.t1)
    close_t5 = _close_at(prices.index, prices, window.t5) if window.t5 is not None else None

    r_1d = (close_t1 / close_t0 - 1) if close_t0 and close_t1 else None
    r_5d = (close_t5 / close_t0 - 1) if close_t0 and close_t5 else None

    spy_r_1d = spy_r_5d = a_1d = a_5d = None
    if spy_prices is not None:
        spy_t0 = _close_at(spy_prices.index, spy_prices, window.t0)
        spy_t1 = _close_at(spy_prices.index, spy_prices, window.t1)
        if spy_t0 and spy_t1:
            spy_r_1d = spy_t1 / spy_t0 - 1
            if r_1d is not None:
                a_1d = r_1d - spy_r_1d
        if window.t5 is not None:
            spy_t5 = _close_at(spy_prices.index, spy_prices, window.t5)
            if spy_t0 and spy_t5:
                spy_r_5d = spy_t5 / spy_t0 - 1
                if r_5d is not None:
                    a_5d = r_5d - spy_r_5d

    return FilingReturnRow(
        ticker=ticker, year=year,
        accession=filing.accession, form=filing.form,
        filing_date=filing.filing_date, report_date=filing.report_date,
        acceptance_dt_utc=filing.acceptance_dt_utc.isoformat(),
        acceptance_dt_et=ef.acceptance_in_et(filing).isoformat(),
        filing_window_class=window.filing_window_class,
        has_amendment=has_amendment,
        t0=window.t0.date().isoformat(),
        t1=window.t1.date().isoformat(),
        t5=window.t5.date().isoformat() if window.t5 is not None else None,
        close_t0=close_t0, close_t1=close_t1, close_t5=close_t5,
        r_1d=r_1d, r_5d=r_5d,
        spy_r_1d=spy_r_1d, spy_r_5d=spy_r_5d,
        a_1d=a_1d, a_5d=a_5d,
        error=None,
    )


# --- CLI --------------------------------------------------------------------


FIELDNAMES = [
    "ticker", "year", "accession", "form",
    "filing_date", "report_date", "acceptance_dt_utc", "acceptance_dt_et",
    "filing_window_class", "has_amendment",
    "t0", "t1", "t5",
    "close_t0", "close_t1", "close_t5",
    "r_1d", "r_5d", "spy_r_1d", "spy_r_5d", "a_1d", "a_5d",
    "error",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tickers", nargs="+", help="Explicit ticker list.")
    src.add_argument("--industry", help="Industry key from selected/companies.json.")
    src.add_argument(
        "--selected", action="store_true",
        help="All companies in selected/companies.json.",
    )
    src.add_argument("--csv", type=Path, help="Path to a *_mapped_clean*.csv file.")
    p.add_argument("--years", default="2017-2022")
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                   help=f"Benchmark ticker for alpha (default: {DEFAULT_BENCHMARK}).")
    p.add_argument("--no-benchmark", action="store_true",
                   help="Skip benchmark fetch; alpha columns will be empty.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--refresh-cache", action="store_true",
                   help="Re-download submissions JSON, prices, and ticker->CIK map.")
    p.add_argument("--include-non-us", action="store_true",
                   help="Don't pre-filter to US listings (non-US will surface 'no CIK').")
    args = p.parse_args(argv)

    years = parse_year_range(args.years)

    if args.tickers:
        entries = tickers_from_args(args.tickers)
    elif args.industry:
        entries = tickers_from_selected(industry=args.industry)
        if not entries:
            print(f"No companies found for industry {args.industry!r}", file=sys.stderr)
            return 2
    elif args.selected:
        entries = tickers_from_selected()
    elif args.csv:
        entries = tickers_from_csv(args.csv)
    else:
        p.error("Must pass --tickers / --industry / --selected / --csv")
        return 2

    if not args.include_non_us:
        before = len(entries)
        entries = [e for e in entries if e["exchange"] in US_EXCHANGES]
        print(
            f"Restricted to US listings: {len(entries)}/{before}",
            file=sys.stderr,
        )

    # Dedupe by ticker (selected/companies.json can list the same company in
    # multiple industries, or duplicate it within one industry). Returns are
    # company-level so duplicates would just emit identical rows.
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for e in entries:
        if e["ticker"] in seen:
            continue
        seen.add(e["ticker"])
        deduped.append(e)
    if len(deduped) < len(entries):
        print(
            f"Deduped on ticker: {len(deduped)}/{len(entries)}",
            file=sys.stderr,
        )
    entries = deduped

    cik_map = edgar.load_ticker_cik_map(refresh=args.refresh_cache)

    bench_start = date(min(years), 1, 1)
    bench_end = date(max(years) + 1, 12, 31)
    spy_prices: pd.DataFrame | None = None
    if not args.no_benchmark:
        print(f"Fetching benchmark {args.benchmark}...", file=sys.stderr)
        spy_prices = fetch_prices(
            args.benchmark, bench_start, bench_end, refresh=args.refresh_cache
        )
        if spy_prices is None:
            print(
                f"  benchmark fetch failed; alpha columns will be empty",
                file=sys.stderr,
            )

    rows: list[FilingReturnRow] = []
    for i, entry in enumerate(entries, 1):
        ticker = entry["ticker"]
        cik = edgar.ticker_to_cik(ticker, mapping=cik_map)
        if cik is None:
            print(f"[{i:>4}/{len(entries)}] {ticker:<8} - no CIK; skipping all years")
            for year in years:
                rows.append(_skip_row(ticker, year, "no CIK in EDGAR ticker map"))
            continue

        filings = ef.all_annual_filings(cik, refresh=args.refresh_cache)
        if not filings:
            print(f"[{i:>4}/{len(entries)}] {ticker:<8} - no annual filings on EDGAR")
            for year in years:
                rows.append(_skip_row(ticker, year, "no 10-K/20-F on EDGAR"))
            continue

        prices = fetch_prices(ticker, bench_start, bench_end, refresh=args.refresh_cache)
        if prices is None or prices.empty:
            print(f"[{i:>4}/{len(entries)}] {ticker:<8} - no prices on yfinance")
            for year in years:
                rows.append(_skip_row(ticker, year, "no yfinance prices"))
            continue

        per_year_summary: list[str] = []
        for year in years:
            row = compute_filing_return(ticker, year, filings, prices, spy_prices)
            rows.append(row)
            if row.error:
                per_year_summary.append(f"{year}=ERR")
            elif row.r_1d is None:
                per_year_summary.append(f"{year}=na")
            else:
                per_year_summary.append(f"{year}={row.r_1d:+.2%}")
        print(
            f"[{i:>4}/{len(entries)}] {ticker:<8} {' '.join(per_year_summary)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            d = asdict(r) if not isinstance(r, dict) else r
            w.writerow(d)

    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.error is None and r.r_1d is not None)
    n_err = sum(1 for r in rows if r.error is not None)
    print(
        f"\nWrote {n_total} rows ({n_ok} with returns, {n_err} errors) to {args.out}",
        file=sys.stderr,
    )
    return 0


def _skip_row(ticker: str, year: int, error: str) -> FilingReturnRow:
    return FilingReturnRow(
        ticker=ticker, year=year, accession=None, form=None,
        filing_date=None, report_date=None,
        acceptance_dt_utc=None, acceptance_dt_et=None,
        filing_window_class=None, has_amendment=False,
        t0=None, t1=None, t5=None,
        close_t0=None, close_t1=None, close_t5=None,
        r_1d=None, r_5d=None,
        spy_r_1d=None, spy_r_5d=None,
        a_1d=None, a_5d=None,
        error=error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
