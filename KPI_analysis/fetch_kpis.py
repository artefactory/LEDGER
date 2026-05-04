"""Fetch consolidated annual KPIs for a set of tickers.

Routing:
  - US-listed exchanges (NYSE, Nasdaq*, NYSE American/AMEX, Cboe BZX)  ->  SEC EDGAR
  - everything else (LSE, AIM, ASX, TSX, ...)                          ->  yfinance
  - optional: Alpha Vantage as a *gap-filler* for low-coverage tickers (--alphavantage)

One JSON file is written per ticker under output/raw/{TICKER}.json. Re-running
overwrites — the heavy EDGAR / Alpha Vantage responses are cached separately
under cache/.

Usage:
  uv run python KPI_analysis/fetch_kpis.py --selected --years 2017-2022
  uv run python KPI_analysis/fetch_kpis.py --tickers ORLY AZO GPC --years 2017-2022
  uv run python KPI_analysis/fetch_kpis.py --industry "Consumer Cyclical / Auto Parts"
  uv run python KPI_analysis/fetch_kpis.py --csv tickers_lists/cleaned/NYSE_mapped_clean_verified.csv
  uv run python KPI_analysis/fetch_kpis.py --selected --alphavantage   # add AV fallback

SEC requires a descriptive User-Agent with contact info; override the default
via the SEC_USER_AGENT environment variable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import alpha_vantage as av
import edgar
import yf_fallback

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUTPUT_DIR = HERE / "output" / "raw"
SELECTED_JSON = REPO_ROOT / "tickers_lists" / "grouped" / "selected" / "companies.json"
CLEANED_DIR = REPO_ROOT / "tickers_lists" / "cleaned"

# Exchange labels (from the "Exchange (Yahoo)" CSV column / selected/companies.json)
# that SEC EDGAR covers. Anything else routes to yfinance.
US_EXCHANGES = {
    "NYSE",
    "NYSEArca",
    "NYSE American",
    "NYSEAMERICAN",
    "AMEX",
    "NasdaqGS",
    "NasdaqGM",
    "NasdaqCM",
    "NASDAQ",
    "BATS",
    "CboeBZX",
}


def parse_year_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        lo, hi = int(a), int(b)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(y.strip()) for y in s.split(",") if y.strip()]


def route_source(exchange: str) -> str:
    return "edgar" if exchange in US_EXCHANGES else "yfinance"


# --- Ticker loaders ---------------------------------------------------------


def tickers_from_selected(
    industry: str | None = None,
) -> list[dict[str, str]]:
    data = json.loads(SELECTED_JSON.read_text())
    out: list[dict[str, str]] = []
    for ind, by_exchange in data.items():
        if industry and ind != industry:
            continue
        for exchange, companies in by_exchange.items():
            for c in companies:
                out.append(
                    {
                        "ticker": c["ticker"],
                        "name": c["name"],
                        "exchange": exchange,
                        "industry": ind,
                    }
                )
    return out


def tickers_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [
            {
                "ticker": row["Ticker"],
                "name": row["Company Name"],
                "exchange": row.get("Exchange (Yahoo)", ""),
                "industry": f"{row.get('Sector', '')} / {row.get('Industry', '')}",
            }
            for row in reader
        ]


def tickers_from_args(
    tickers: list[str], default_exchange: str = "NYSE"
) -> list[dict[str, str]]:
    # When the user passes raw tickers we don't always know the exchange. Assume US
    # unless the ticker has a suffix like ".L" (LSE) — yfinance convention.
    out = []
    for t in tickers:
        exch = default_exchange
        if "." in t:
            # .L -> LSE, .AX -> ASX, .TO -> TSX, .V -> TSXV
            suffix = t.rsplit(".", 1)[-1].upper()
            exch = {"L": "LSE", "AX": "ASX", "TO": "TSX", "V": "TSXV"}.get(
                suffix, "UNKNOWN"
            )
        out.append({"ticker": t, "name": "", "exchange": exch, "industry": ""})
    return out


# --- Fetching ---------------------------------------------------------------


def fetch_one(
    entry: dict[str, str],
    years: list[int],
    cik_map: dict[str, str],
    *,
    force_source: str | None = None,
) -> dict:
    source = force_source or route_source(entry["exchange"])
    record: dict = {
        "ticker": entry["ticker"],
        "company_name": entry["name"],
        "exchange": entry["exchange"],
        "industry": entry["industry"],
        "years": years,
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kpis": {},
        "tag_used": {},
        "ambiguous_tags": {},
    }
    try:
        if source == "edgar":
            result = edgar.fetch_kpis_for_ticker(
                entry["ticker"], years, mapping=cik_map
            )
            if result is None:
                # Fallback: try yfinance anyway (covers CIKs missing from the
                # ticker->CIK map, e.g. dual-listings whose primary is foreign).
                record["source"] = "yfinance (edgar miss)"
                fb = yf_fallback.fetch_kpis_for_ticker(entry["ticker"], years)
                if fb:
                    record["kpis"] = fb.get("kpis", {})
                    record["tag_used"] = fb.get("tag_used", {})
                else:
                    record["error"] = "Not on EDGAR and yfinance returned nothing"
            else:
                record["cik"] = result["cik"]
                record["entity_name"] = result.get("entity_name")
                record["kpis"] = result["kpis"]
                record["tag_used"] = result["tag_used"]
                record["ambiguous_tags"] = result.get("ambiguous_tags", {})
        else:
            fb = yf_fallback.fetch_kpis_for_ticker(entry["ticker"], years)
            if fb is None:
                record["error"] = "yfinance returned nothing"
            elif fb.get("error"):
                record["error"] = fb["error"]
            else:
                record["kpis"] = fb.get("kpis", {})
                record["tag_used"] = fb.get("tag_used", {})
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def write_record(record: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = record["ticker"].replace("/", "_").replace(":", "_")
    path = OUTPUT_DIR / f"{safe}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


# --- Alpha Vantage gap-fill -------------------------------------------------

# KPIs Alpha Vantage can plausibly fill. Used both for prioritising tickers and
# for keeping coverage scoring focused (ranking by total shortfall would
# penalise tickers whose gaps are AV-incompatible anyway).
_AV_FILLABLE_KPIS = tuple(av.AV_KPI_MAP.keys())


def _kpi_has_year(kpi_dict: dict, year: int) -> bool:
    """Tolerate JSON's string-keyed dicts where year may have been serialised."""
    return year in kpi_dict or str(year) in kpi_dict


def coverage_shortfall(
    record: dict, years: Iterable[int], kpi_keys: Iterable[str] = _AV_FILLABLE_KPIS
) -> int:
    """Number of (KPI x year) cells missing among the AV-fillable KPIs."""
    kpis = record.get("kpis") or {}
    missing = 0
    for kpi in kpi_keys:
        per_year = kpis.get(kpi) or {}
        for y in years:
            if not _kpi_has_year(per_year, int(y)):
                missing += 1
    return missing


def merge_av_into_record(
    record: dict, av_result: av.TickerResult, years: Iterable[int]
) -> int:
    """Fill missing (kpi, year) cells from Alpha Vantage. Returns cells added.

    EDGAR / yfinance values are never overwritten — AV strictly fills holes.
    Provenance is recorded under `record["alphavantage"]` plus per-KPI tags
    in `record["alphavantage_tag_used"]`.
    """
    years_set = {int(y) for y in years}
    added = 0
    kpis_out = record.setdefault("kpis", {})
    for kpi, per_year in av_result.kpis.items():
        bucket = kpis_out.setdefault(kpi, {})
        for y, v in per_year.items():
            year = int(y)
            if year not in years_set:
                continue
            if _kpi_has_year(bucket, year):
                continue
            bucket[str(year)] = float(v)
            added += 1

    if added or av_result.endpoints_called or av_result.endpoints_cached:
        av_tags = record.setdefault("alphavantage_tag_used", {})
        for kpi, tag in av_result.tag_used.items():
            if kpi in av_result.kpis:
                av_tags[kpi] = tag

    av_meta = record.setdefault("alphavantage", {})
    av_meta["symbol_used"] = av_result.symbol_used
    if av_result.reported_currency:
        av_meta["reported_currency"] = av_result.reported_currency
    if av_result.endpoints_called:
        av_meta["endpoints_called"] = av_result.endpoints_called
    if av_result.endpoints_cached:
        av_meta["endpoints_cached"] = av_result.endpoints_cached
    av_meta["cells_added"] = av_meta.get("cells_added", 0) + added
    if av_result.error:
        av_meta["error"] = av_result.error

    if added > 0:
        src = record.get("source") or ""
        if "alphavantage" not in src:
            record["source"] = f"{src} + alphavantage" if src else "alphavantage"

    return added


def run_alphavantage_fallback(
    records: list[dict],
    years: list[int],
    *,
    keys: list[str],
    budget: av.BudgetTracker,
    endpoints: tuple[str, ...],
    min_shortfall: int,
    max_calls: int | None,
    refresh: bool,
) -> dict[str, int]:
    """Rank records by AV-fillable shortfall, fill until budget is exhausted.

    Mutates each record in `records` in place and re-writes its JSON file.
    Returns a small summary dict for logging.
    """
    candidates = sorted(
        (
            (coverage_shortfall(r, years), i, r)
            for i, r in enumerate(records)
        ),
        key=lambda t: -t[0],
    )
    eligible = [(s, i, r) for s, i, r in candidates if s >= min_shortfall]

    summary = {
        "candidates": len(eligible),
        "tickers_processed": 0,
        "cells_added": 0,
        "live_calls": 0,
        "cached_calls": 0,
        "errors": 0,
    }

    remaining_budget = budget.total_remaining(keys)
    if remaining_budget <= 0:
        print(
            "[alphavantage] all keys are out of daily quota — nothing to do.",
            file=sys.stderr,
        )
        return summary

    call_cap = remaining_budget if max_calls is None else min(max_calls, remaining_budget)
    print(
        f"[alphavantage] {len(eligible)} tickers below shortfall threshold; "
        f"daily budget remaining={remaining_budget} (cap this run={call_cap}); "
        f"endpoints/ticker={len(endpoints)}.",
        file=sys.stderr,
    )

    used_calls = 0
    for rank, (shortfall, idx, record) in enumerate(eligible, 1):
        if used_calls >= call_cap:
            break
        # Stop once we can't afford even one more endpoint for this ticker
        # (cached responses don't count, but we don't know that until we look).
        if budget.total_remaining(keys) <= 0:
            break
        ticker = record["ticker"]
        before_remaining = budget.total_remaining(keys)
        result = av.fetch_kpis_for_ticker(
            ticker,
            years,
            keys=keys,
            budget=budget,
            endpoints=endpoints,
            refresh=refresh,
        )
        after_remaining = budget.total_remaining(keys)
        live_calls = max(before_remaining - after_remaining, 0)
        used_calls += live_calls
        added = merge_av_into_record(record, result, years)
        write_record(record)

        summary["tickers_processed"] += 1
        summary["cells_added"] += added
        summary["live_calls"] += live_calls
        summary["cached_calls"] += len(result.endpoints_cached)
        if result.error:
            summary["errors"] += 1

        print(
            f"[alphavantage {rank:>3}/{len(eligible)}] {ticker:<10} "
            f"shortfall={shortfall:>3}  +cells={added:>3}  "
            f"live={live_calls} cached={len(result.endpoints_cached)} "
            f"sym={result.symbol_used}"
            + (f"  err={result.error}" if result.error else "")
        )
        if "BudgetExhausted" in (result.error or ""):
            break

    return summary


def summarize(record: dict, years: list[int]) -> str:
    n_kpis = len(record.get("kpis", {}))
    covered = set()
    for v in record.get("kpis", {}).values():
        covered.update(int(y) for y in v.keys())
    coverage = f"{len(covered & set(years))}/{len(years)} yrs"
    err = f"  [ERR: {record['error']}]" if record.get("error") else ""
    return (
        f"{record['ticker']:<8} {record['exchange']:<10} "
        f"{record['source']:<22} kpis={n_kpis:>2}  yrs={coverage}{err}"
    )


# --- CLI --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tickers", nargs="+", help="Explicit ticker list.")
    src.add_argument("--industry", help="Industry key from selected/companies.json.")
    src.add_argument(
        "--selected", action="store_true", help="All companies in selected/companies.json."
    )
    src.add_argument("--csv", type=Path, help="Path to a *_mapped_clean*.csv file.")
    p.add_argument(
        "--years",
        default="2017-2022",
        help="Year range (e.g. 2017-2022) or comma list (e.g. 2018,2020,2022).",
    )
    p.add_argument(
        "--force-source",
        choices=("edgar", "yfinance"),
        help="Override the exchange-based routing.",
    )
    p.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download SEC ticker->CIK map and any requested companyfacts.",
    )
    # --- Alpha Vantage gap-fill (opt-in) ---
    p.add_argument(
        "--alphavantage",
        action="store_true",
        help=(
            "After EDGAR/yfinance, run Alpha Vantage on the worst-coverage tickers "
            "to fill missing (KPI, year) cells. Disk-cached; never overwrites EDGAR/yfinance."
        ),
    )
    p.add_argument(
        "--alphavantage-keys",
        type=Path,
        default=av.DEFAULT_KEYS_PATH,
        help=f"Path to API keys file (default: {av.DEFAULT_KEYS_PATH}).",
    )
    p.add_argument(
        "--alphavantage-daily-quota",
        type=int,
        default=av.DEFAULT_DAILY_QUOTA,
        help="Per-key per-day call quota (default: 25, AV's free tier).",
    )
    p.add_argument(
        "--alphavantage-budget",
        type=int,
        default=None,
        help="Cap total live AV calls in this run (default: all remaining quota).",
    )
    p.add_argument(
        "--alphavantage-min-shortfall",
        type=int,
        default=1,
        help="Skip tickers with fewer than this many missing AV-fillable cells.",
    )
    p.add_argument(
        "--alphavantage-include-earnings",
        action="store_true",
        help="Also fetch the EARNINGS endpoint (adds ~1 call per ticker; gives EPS).",
    )
    p.add_argument(
        "--alphavantage-refresh",
        action="store_true",
        help="Bypass on-disk AV response cache and re-fetch.",
    )
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
        p.error("Must pass one of --tickers / --industry / --selected / --csv")
        return 2

    print(
        f"Fetching {len(entries)} tickers for years {years[0]}-{years[-1]} "
        f"({len(years)} years)\n",
        file=sys.stderr,
    )

    cik_map = edgar.load_ticker_cik_map(refresh=args.refresh_cache)

    ok, errs = 0, 0
    records: list[dict] = []
    for i, entry in enumerate(entries, 1):
        record = fetch_one(
            entry,
            years,
            cik_map,
            force_source=args.force_source,
        )
        write_record(record)
        records.append(record)
        line = summarize(record, years)
        print(f"[{i:>4}/{len(entries)}] {line}")
        if record.get("error"):
            errs += 1
        else:
            ok += 1

    print(
        f"\nDone (primary). ok={ok} errors={errs} total={len(entries)}. "
        f"Per-ticker JSON in {OUTPUT_DIR}",
        file=sys.stderr,
    )

    if args.alphavantage:
        keys = av.load_keys(args.alphavantage_keys)
        if not keys:
            print(
                f"\n[alphavantage] no keys found at {args.alphavantage_keys} — skipping fallback.",
                file=sys.stderr,
            )
        else:
            endpoints = list(av.DEFAULT_ENDPOINTS)
            if args.alphavantage_include_earnings:
                endpoints.append(av.EARNINGS_ENDPOINT)
            budget = av.BudgetTracker(daily_quota=args.alphavantage_daily_quota)
            print(
                f"\n[alphavantage] starting fallback: {len(keys)} key(s), "
                f"daily_quota={args.alphavantage_daily_quota}, "
                f"endpoints={endpoints}, "
                f"min_shortfall={args.alphavantage_min_shortfall}.",
                file=sys.stderr,
            )
            summary = run_alphavantage_fallback(
                records,
                years,
                keys=keys,
                budget=budget,
                endpoints=tuple(endpoints),
                min_shortfall=args.alphavantage_min_shortfall,
                max_calls=args.alphavantage_budget,
                refresh=args.alphavantage_refresh,
            )
            print(
                f"\n[alphavantage] done. tickers={summary['tickers_processed']}/"
                f"{summary['candidates']}  cells_added={summary['cells_added']}  "
                f"live_calls={summary['live_calls']}  "
                f"cached_calls={summary['cached_calls']}  errors={summary['errors']}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
