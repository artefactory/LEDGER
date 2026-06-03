"""Scan the DeepSeekOCR directory for all companies, resolve metadata from
cleaned ticker CSVs, then fetch KPIs from EDGAR / yfinance for every year
present in the OCR corpus.

Output
------
- find_more_queries/output/raw/{TICKER}.json   one record per company
- find_more_queries/ocr_companies.json         full inventory manifest

Usage
-----
    uv run python KPI_analysis/find_more_queries/scan_and_fetch.py
    uv run python KPI_analysis/find_more_queries/scan_and_fetch.py --dry-run
    uv run python KPI_analysis/find_more_queries/scan_and_fetch.py --refresh-cache
    uv run python KPI_analysis/find_more_queries/scan_and_fetch.py --min-year 2009
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KPI_DIR = HERE.parent
REPO_ROOT = KPI_DIR.parent
sys.path.insert(0, str(KPI_DIR))

import edgar
import yf_fallback

OCR_ROOT = Path(
    "/data/workspace/charles/pdf_ocr_deepseek/DeepSeekOCR_Ardian_raw_3kdocs"
)
CLEANED_DIR = REPO_ROOT / "tickers_lists" / "cleaned"
OUTPUT_DIR = HERE / "output" / "raw"
MANIFEST_PATH = HERE / "ocr_companies.json"

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

# OCR directory uses exchange labels without the Yahoo suffix
OCR_EXCHANGE_MAP = {
    "NYSE": "NYSE",
    "NASDAQ": "NasdaqGS",  # we don't distinguish GS/GM/CM at this level
    "AMEX": "AMEX",
    "LSE": "LSE",
    "AIM": "AIM",
    "ASX": "ASX",
    "TSX": "TSX",
    "TSXV": "TSXV",
    "OTC": "OTC",
}


# ---------------------------------------------------------------------------
# Step 1 – inventory OCR directory
# ---------------------------------------------------------------------------


def scan_ocr_dir(ocr_root: Path) -> dict[tuple[str, str], dict]:
    """Return {(exchange, ticker): {industry, years, industry_dir}} from OCR tree."""
    companies: dict[tuple[str, str], dict] = {}
    for industry_dir in sorted(ocr_root.iterdir()):
        if not industry_dir.is_dir():
            continue
        for report_dir in sorted(industry_dir.iterdir()):
            if not report_dir.is_dir():
                continue
            name = report_dir.name
            parts = name.split("_")
            if len(parts) < 3:
                continue
            exchange = parts[0]
            year_str = parts[-1]
            ticker = "_".join(parts[1:-1])
            if not year_str.isdigit():
                continue
            key = (exchange, ticker)
            if key not in companies:
                companies[key] = {
                    "exchange_ocr": exchange,
                    "ticker_ocr": ticker,
                    "industry_dir": industry_dir.name,
                    "years": [],
                }
            companies[key]["years"].append(int(year_str))

    # Deduplicate and sort years
    for v in companies.values():
        v["years"] = sorted(set(v["years"]))
    return companies


# ---------------------------------------------------------------------------
# Step 2 – match to cleaned CSVs
# ---------------------------------------------------------------------------


def load_cleaned_lookup() -> dict[tuple[str, str], dict]:
    """Return {(exchange_label, ticker_with_suffix): cleaned_row}."""
    lookup: dict[tuple[str, str], dict] = {}
    for csv_file in sorted(CLEANED_DIR.glob("*_mapped_clean.csv")):
        if "verified" in csv_file.name:
            continue
        exchange_label = csv_file.name.split("_")[0]
        with csv_file.open() as f:
            for row in csv.DictReader(f):
                key = (exchange_label, row["Ticker"])
                lookup[key] = {**row, "_exchange_label": exchange_label}
    return lookup


def resolve_metadata(
    exchange_ocr: str,
    ticker_ocr: str,
    cleaned_lookup: dict,
) -> tuple[str, str | None, str | None, str | None]:
    """Return (canonical_ticker, company_name, sector, industry_from_csv).

    For LSE tickers the OCR dir uses bare tickers (e.g. ABDP) but the cleaned
    CSV uses Yahoo-style suffixes (ABDP.L).  We try both.
    """
    # Direct match
    row = cleaned_lookup.get((exchange_ocr, ticker_ocr))
    if row:
        return ticker_ocr, row["Company Name"], row.get("Sector"), row.get("Industry")

    # LSE: try appending .L
    if exchange_ocr == "LSE":
        row = cleaned_lookup.get(("LSE", ticker_ocr + ".L"))
        if row:
            return (
                ticker_ocr + ".L",
                row["Company Name"],
                row.get("Sector"),
                row.get("Industry"),
            )

    # AIM: similarly
    if exchange_ocr == "AIM":
        row = cleaned_lookup.get(("AIM", ticker_ocr + ".L"))
        if row:
            return (
                ticker_ocr + ".L",
                row["Company Name"],
                row.get("Sector"),
                row.get("Industry"),
            )

    # Not found – use bare OCR ticker, no metadata from CSV
    return ticker_ocr, None, None, None


# ---------------------------------------------------------------------------
# Step 3 – routing and fetching
# ---------------------------------------------------------------------------


def route_source(exchange_ocr: str) -> str:
    mapped = OCR_EXCHANGE_MAP.get(exchange_ocr, exchange_ocr)
    return "edgar" if mapped in US_EXCHANGES else "yfinance"


def fetch_one(
    ticker: str,
    exchange_ocr: str,
    company_name: str | None,
    sector: str | None,
    industry_csv: str | None,
    industry_dir: str,
    years: list[int],
    cik_map: dict[str, str],
) -> dict:
    source = route_source(exchange_ocr)
    record: dict = {
        "ticker": ticker,
        "company_name": company_name or "",
        "exchange": OCR_EXCHANGE_MAP.get(exchange_ocr, exchange_ocr),
        "exchange_ocr": exchange_ocr,
        "sector": sector or "",
        "industry": industry_csv or "",
        "industry_dir": industry_dir,
        "years_in_ocr": years,
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kpis": {},
        "tag_used": {},
        "ambiguous_tags": {},
    }

    try:
        if source == "edgar":
            result = edgar.fetch_kpis_for_ticker(ticker, years, mapping=cik_map)
            if result is None:
                # Fallback to yfinance
                record["source"] = "yfinance (edgar miss)"
                fb = yf_fallback.fetch_kpis_for_ticker(ticker, years)
                if fb:
                    record["kpis"] = fb.get("kpis", {})
                    record["tag_used"] = fb.get("tag_used", {})
                    if not company_name and fb.get("company_name"):
                        record["company_name"] = fb["company_name"]
                else:
                    record["error"] = "Not on EDGAR and yfinance returned nothing"
            else:
                record["cik"] = result["cik"]
                record["entity_name"] = result.get("entity_name", "")
                record["kpis"] = result["kpis"]
                record["tag_used"] = result["tag_used"]
                record["ambiguous_tags"] = result.get("ambiguous_tags", {})
                if not company_name:
                    record["company_name"] = result.get("entity_name", "")
        else:
            fb = yf_fallback.fetch_kpis_for_ticker(ticker, years)
            if fb is None:
                record["error"] = "yfinance returned nothing"
            elif fb.get("error"):
                record["error"] = fb["error"]
            else:
                record["kpis"] = fb.get("kpis", {})
                record["tag_used"] = fb.get("tag_used", {})
                if not company_name and fb.get("company_name"):
                    record["company_name"] = fb["company_name"]
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def record_filename(exchange_ocr: str, ticker: str) -> str:
    """Unique filename key that avoids collisions when the same ticker is listed
    on multiple exchanges (e.g. CRC on both LSE and NYSE are different companies)."""
    safe = f"{exchange_ocr}_{ticker}".replace("/", "_").replace(":", "_")
    return f"{safe}.json"


def write_record(record: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = record_filename(record["exchange_ocr"], record["ticker"])
    path = output_dir / fname
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def kpi_year_count(record: dict, years: list[int]) -> tuple[int, int]:
    """Return (filled_kpi_year_cells, total_kpi_year_cells) for the given years."""
    kpis = record.get("kpis") or {}
    total = len(kpis) * len(years)
    filled = sum(
        1
        for kpi_data in kpis.values()
        for y in years
        if str(y) in kpi_data or y in kpi_data
    )
    return filled, total


def summarize_line(record: dict) -> str:
    years = record.get("years_in_ocr", [])
    n_kpis = len(record.get("kpis", {}))
    covered_years = set()
    for v in record.get("kpis", {}).values():
        covered_years.update(int(y) for y in v.keys())
    yr_coverage = f"{len(covered_years & set(years))}/{len(years)} yrs"
    err = f"  [ERR: {record['error']}]" if record.get("error") else ""
    return (
        f"{record['ticker']:<12} {record['exchange']:<10} "
        f"{record['source']:<22} kpis={n_kpis:>2}  {yr_coverage}{err}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ocr-root",
        type=Path,
        default=OCR_ROOT,
        help="Root of the DeepSeekOCR directory tree.",
    )
    p.add_argument(
        "--min-year",
        type=int,
        default=2009,
        help="Ignore OCR years before this value (EDGAR XBRL starts ~2009). Default: 2009.",
    )
    p.add_argument(
        "--max-year",
        type=int,
        default=2024,
        help="Ignore OCR years after this value. Default: 2024.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the inventory without fetching KPIs.",
    )
    p.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download the SEC ticker->CIK map.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tickers whose JSON already exists in output/raw/.",
    )
    args = p.parse_args(argv)

    # --- inventory ---
    print("Scanning OCR directory...", file=sys.stderr)
    companies = scan_ocr_dir(args.ocr_root)
    print(f"  Found {len(companies)} unique (exchange, ticker) pairs", file=sys.stderr)

    cleaned_lookup = load_cleaned_lookup()
    print(f"  Cleaned CSV entries: {len(cleaned_lookup)}", file=sys.stderr)

    # Build full manifest
    manifest = []
    for (exchange_ocr, ticker_ocr), data in sorted(companies.items()):
        years_all = data["years"]
        years_filtered = [y for y in years_all if args.min_year <= y <= args.max_year]

        canonical_ticker, company_name, sector, industry_csv = resolve_metadata(
            exchange_ocr, ticker_ocr, cleaned_lookup
        )
        entry = {
            "ticker": canonical_ticker,
            "ticker_ocr": ticker_ocr,
            "exchange_ocr": exchange_ocr,
            "exchange": OCR_EXCHANGE_MAP.get(exchange_ocr, exchange_ocr),
            "company_name": company_name or "",
            "sector": sector or "",
            "industry_csv": industry_csv or "",
            "industry_dir": data["industry_dir"],
            "years_all": years_all,
            "years_fetch": years_filtered,
            "in_cleaned_csv": company_name is not None,
            "source": route_source(exchange_ocr),
        }
        manifest.append(entry)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest written to {MANIFEST_PATH}", file=sys.stderr)

    matched = sum(1 for e in manifest if e["in_cleaned_csv"])
    print(
        f"  Matched to cleaned CSV: {matched}/{len(manifest)} "
        f"({len(manifest) - matched} unmatched)",
        file=sys.stderr,
    )

    if args.dry_run:
        print("\nDry-run — inventory only, no KPI fetch.\n")
        print(
            f"{'Ticker':<14} {'Exchange':<8} {'In CSV':<7} {'Years (OCR)':<20} {'Company Name'}"
        )
        print("-" * 80)
        for e in manifest:
            yrs = (
                f"{min(e['years_all'])}-{max(e['years_all'])}"
                if e["years_all"]
                else "-"
            )
            tick = e["ticker"]
            print(
                f"{tick:<14} {e['exchange_ocr']:<8} {'yes' if e['in_cleaned_csv'] else 'NO':<7} "
                f"{yrs:<20} {e['company_name'] or '(unknown)'}"
            )
        return 0

    # --- fetch ---
    print(f"\nFetching KPIs for {len(manifest)} tickers...", file=sys.stderr)
    cik_map = edgar.load_ticker_cik_map(refresh=args.refresh_cache)

    ok, errs, skipped = 0, 0, 0
    for i, entry in enumerate(manifest, 1):
        ticker = entry["ticker"]
        years_fetch = entry["years_fetch"]
        out_path = OUTPUT_DIR / record_filename(entry["exchange_ocr"], ticker)

        if args.skip_existing and out_path.exists():
            skipped += 1
            print(f"[{i:>4}/{len(manifest)}] SKIP {ticker}")
            continue

        if not years_fetch:
            record: dict = {
                "ticker": ticker,
                "company_name": entry["company_name"],
                "exchange": entry["exchange"],
                "exchange_ocr": entry["exchange_ocr"],
                "sector": entry["sector"],
                "industry": entry["industry_csv"],
                "industry_dir": entry["industry_dir"],
                "years_in_ocr": entry["years_all"],
                "years_fetch": [],
                "source": entry["source"],
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kpis": {},
                "tag_used": {},
                "ambiguous_tags": {},
                "error": f"No years in [{args.min_year}, {args.max_year}] range",
            }
            write_record(record, OUTPUT_DIR)
            errs += 1
            print(f"[{i:>4}/{len(manifest)}] {ticker:<12} no years in fetch range")
            continue

        record = fetch_one(
            ticker=ticker,
            exchange_ocr=entry["exchange_ocr"],
            company_name=entry["company_name"] or None,
            sector=entry["sector"] or None,
            industry_csv=entry["industry_csv"] or None,
            industry_dir=entry["industry_dir"],
            years=years_fetch,
            cik_map=cik_map,
        )
        record["years_fetch"] = years_fetch
        write_record(record, OUTPUT_DIR)

        line = summarize_line(record)
        print(f"[{i:>4}/{len(manifest)}] {line}")
        if record.get("error"):
            errs += 1
        else:
            ok += 1

    print(
        f"\nDone. ok={ok}  errors={errs}  skipped={skipped}  total={len(manifest)}\n"
        f"JSON files in {OUTPUT_DIR}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
