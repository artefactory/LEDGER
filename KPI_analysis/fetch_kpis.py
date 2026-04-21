"""Fetch consolidated annual KPIs for a set of tickers.

Routing:
  - US-listed exchanges (NYSE, Nasdaq*, NYSE American/AMEX, Cboe BZX)  ->  SEC EDGAR
  - everything else (LSE, AIM, ASX, TSX, ...)                          ->  yfinance

One JSON file is written per ticker under output/raw/{TICKER}.json. Re-running
overwrites — the heavy EDGAR responses are cached separately under cache/.

Usage:
  uv run python KPI_analysis/fetch_kpis.py --selected --years 2017-2022
  uv run python KPI_analysis/fetch_kpis.py --tickers ORLY AZO GPC --years 2017-2022
  uv run python KPI_analysis/fetch_kpis.py --industry "Consumer Cyclical / Auto Parts"
  uv run python KPI_analysis/fetch_kpis.py --csv tickers_lists/cleaned/NYSE_mapped_clean_verified.csv

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
    for i, entry in enumerate(entries, 1):
        record = fetch_one(
            entry,
            years,
            cik_map,
            force_source=args.force_source,
        )
        write_record(record)
        line = summarize(record, years)
        print(f"[{i:>4}/{len(entries)}] {line}")
        if record.get("error"):
            errs += 1
        else:
            ok += 1

    print(
        f"\nDone. ok={ok} errors={errs} total={len(entries)}. "
        f"Per-ticker JSON in {OUTPUT_DIR}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
