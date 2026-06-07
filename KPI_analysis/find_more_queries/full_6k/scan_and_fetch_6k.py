"""Scan the *full 6k* DeepSeekOCR corpus (flat layout), verify each ticker's
identity, then fetch all 31 KPIs from EDGAR / yfinance for every fiscal year
present in the OCR corpus (restricted to the EDGAR-XBRL era by default).

Differences from the sibling ``find_more_queries/scan_and_fetch.py``:

- The full-6k tree is **flat**: report subdirs ``{EXCHANGE}_{TICKER}_{YEAR}/``
  sit directly under the root (no per-industry parent dir).
- **Company-identity verification**: before trusting the fetched financials we
  resolve a reference company name (cleaned-CSV name where available, else the
  yfinance ``longName``) and fuzzy-match it against the name the source
  actually returned (EDGAR ``entityName`` / yfinance ``longName``). The verdict
  is stored as ``verified`` so downstream consumers can drop mismatches — these
  flag cross-exchange ticker collisions (e.g. LSE ``AAL`` vs NASDAQ ``AAL`` =
  American Airlines) that would otherwise inject the wrong company's numbers.

Output (all under ``find_more_queries/full_6k/``)
-------------------------------------------------
- ``output/raw/{EXCHANGE}_{TICKER}.json``   one record per company
- ``ocr_companies.json``                    full inventory manifest

Usage
-----
    uv run python KPI_analysis/find_more_queries/full_6k/scan_and_fetch_6k.py --dry-run
    uv run python KPI_analysis/find_more_queries/full_6k/scan_and_fetch_6k.py
    uv run python KPI_analysis/find_more_queries/full_6k/scan_and_fetch_6k.py --skip-existing
    uv run python KPI_analysis/find_more_queries/full_6k/scan_and_fetch_6k.py --limit 20
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FMQ_DIR = HERE.parent
KPI_DIR = FMQ_DIR.parent
REPO_ROOT = KPI_DIR.parent
sys.path.insert(0, str(KPI_DIR / "kpi_fetch_and_build"))

import edgar
import yf_fallback
import yfinance as yf

OCR_ROOT = REPO_ROOT / "DeepSeekOCR_Ardian_full_6k"
CLEANED_DIR = REPO_ROOT / "tickers_lists" / "cleaned"
OUTPUT_DIR = HERE / "output" / "raw"
MANIFEST_PATH = HERE / "ocr_companies.json"

# Report subdir naming convention: EXCHANGE_TICKER_YEAR, with an optional
# trailing _<hexhash> suffix for duplicate OCR runs (see prune_ocr.py). The
# ticker may itself contain '_'.
REPORT_RE = re.compile(r"^([A-Za-z0-9]+)_(.+)_(\d{4})(?:_[0-9a-fA-F]+)?$")

US_EXCHANGES = {
    "NYSE", "NYSEArca", "NYSE American", "NYSEAMERICAN", "AMEX",
    "NasdaqGS", "NasdaqGM", "NasdaqCM", "NASDAQ", "BATS", "CboeBZX",
}

# OCR directory uses exchange labels without the Yahoo suffix.
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

# yfinance ticker suffix per OCR exchange (US listings keep the bare ticker).
YF_SUFFIX = {
    "LSE": ".L", "AIM": ".L", "ASX": ".AX", "TSX": ".TO", "TSXV": ".V",
}

# Name-match threshold: normalised SequenceMatcher ratio at/above this is a match.
NAME_MATCH_THRESHOLD = 0.6

# Throttle between live yfinance .info calls (reference-name lookups).
YF_INFO_SLEEP = 1.0


# ---------------------------------------------------------------------------
# Step 1 – inventory the (flat) OCR directory
# ---------------------------------------------------------------------------


def scan_ocr_dir(ocr_root: Path) -> tuple[dict[tuple[str, str], dict], list[str]]:
    """Return ({(exchange, ticker): {years}}, [unparsed dir names]) from the flat tree."""
    companies: dict[tuple[str, str], dict] = {}
    unparsed: list[str] = []
    for report_dir in sorted(ocr_root.iterdir()):
        if not report_dir.is_dir():
            continue
        m = REPORT_RE.match(report_dir.name)
        if not m:
            unparsed.append(report_dir.name)
            continue
        exchange, ticker, year_str = m.group(1), m.group(2), m.group(3)
        key = (exchange, ticker)
        if key not in companies:
            companies[key] = {"exchange_ocr": exchange, "ticker_ocr": ticker, "years": []}
        companies[key]["years"].append(int(year_str))
    for v in companies.values():
        v["years"] = sorted(set(v["years"]))
    return companies, sorted(unparsed)


# ---------------------------------------------------------------------------
# Step 2 – reference name resolution (cleaned CSV, else yfinance longName)
# ---------------------------------------------------------------------------


def load_cleaned_lookup() -> dict[tuple[str, str], dict]:
    """Return {(exchange_label, ticker): cleaned_row}."""
    lookup: dict[tuple[str, str], dict] = {}
    for csv_file in sorted(CLEANED_DIR.glob("*_mapped_clean.csv")):
        if "verified" in csv_file.name:
            continue
        exchange_label = csv_file.name.split("_")[0]
        with csv_file.open() as f:
            for row in csv.DictReader(f):
                lookup[(exchange_label, row["Ticker"])] = row
    return lookup


def yf_symbol(exchange_ocr: str, ticker_ocr: str) -> str:
    """Yahoo-style symbol for a bare OCR ticker (adds exchange suffix for non-US)."""
    return ticker_ocr + YF_SUFFIX.get(exchange_ocr, "")


def get_yf_longname(symbol: str) -> str | None:
    """Best-effort yfinance longName/shortName lookup (throttled, never raises)."""
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return None
    finally:
        time.sleep(YF_INFO_SLEEP)
    if not isinstance(info, dict):
        return None
    return info.get("longName") or info.get("shortName") or None


def resolve_reference(
    exchange_ocr: str, ticker_ocr: str, cleaned_lookup: dict
) -> tuple[str, str | None, str | None, str | None]:
    """Return (canonical_ticker, reference_name, reference_source, csv_metadata_industry).

    Reference name priority: cleaned CSV (document-tied) > yfinance longName.
    The yfinance lookup is deferred to fetch time (see resolve_yf_reference).
    """
    row = cleaned_lookup.get((exchange_ocr, ticker_ocr))
    if row:
        return ticker_ocr, row["Company Name"], "cleaned_csv", row.get("Industry")
    if exchange_ocr in ("LSE", "AIM"):
        row = cleaned_lookup.get((exchange_ocr, ticker_ocr + ".L"))
        if row:
            return ticker_ocr + ".L", row["Company Name"], "cleaned_csv", row.get("Industry")
    return ticker_ocr, None, None, None


# ---------------------------------------------------------------------------
# Step 3 – identity verification
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "plc", "ltd", "limited", "llc", "lp", "llp", "holdings", "holding", "group",
    "the", "sa", "ag", "nv", "se", "spa", "ab", "as", "asa", "oyj", "kgaa",
    "class", "ordinary", "shares", "and", "&",
}


def _normalize_tokens(name: str) -> list[str]:
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return [t for t in name.split() if t and t not in _LEGAL_SUFFIXES]


def verify_identity(reference_name: str | None, fetched_name: str | None) -> tuple[str, float | None]:
    """Return (verdict, score). verdict ∈ {match, mismatch, no_reference, unverified}.

    Compares normalised names with both an order-sensitive and an order-insensitive
    (token-sorted) ratio, taking the max — so word reorderings like
    "T. Rowe Price Group" vs "PRICE T ROWE GROUP" still match.
    """
    if not reference_name:
        return "no_reference", None
    if not fetched_name:
        return "unverified", None
    ta, tb = _normalize_tokens(reference_name), _normalize_tokens(fetched_name)
    a, b = " ".join(ta), " ".join(tb)
    if not a or not b:
        return "unverified", None
    if a == b or a in b or b in a:
        return "match", 1.0
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    sorted_seq = difflib.SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    score = max(seq, sorted_seq)
    return ("match" if score >= NAME_MATCH_THRESHOLD else "mismatch"), round(score, 3)


# ---------------------------------------------------------------------------
# Step 4 – routing and fetching
# ---------------------------------------------------------------------------


def route_source(exchange_ocr: str) -> str:
    mapped = OCR_EXCHANGE_MAP.get(exchange_ocr, exchange_ocr)
    return "edgar" if mapped in US_EXCHANGES else "yfinance"


def fetch_one(entry: dict, cik_map: dict[str, str]) -> dict:
    exchange_ocr = entry["exchange_ocr"]
    ticker = entry["ticker"]
    years = entry["years_fetch"]
    source = route_source(exchange_ocr)

    record: dict = {
        "ticker": ticker,
        "ticker_ocr": entry["ticker_ocr"],
        "company_name": entry["reference_name"] or "",
        "exchange": OCR_EXCHANGE_MAP.get(exchange_ocr, exchange_ocr),
        "exchange_ocr": exchange_ocr,
        "industry": entry.get("industry_csv") or "",
        "years_in_ocr": entry["years_all"],
        "years_fetch": years,
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kpis": {},
        "tag_used": {},
        "ambiguous_tags": {},
        # verification fields (populated below)
        "reference_name": entry["reference_name"] or "",
        "reference_source": entry["reference_source"] or "",
        "fetched_name": "",
        "name_match_score": None,
        "verified": "no_reference",
    }

    yf_sym = yf_symbol(exchange_ocr, entry["ticker_ocr"])
    fetched_name: str | None = None

    try:
        if source == "edgar":
            result = edgar.fetch_kpis_for_ticker(ticker, years, mapping=cik_map)
            if result is None:
                # EDGAR miss → yfinance fallback
                record["source"] = "yfinance (edgar miss)"
                fb = yf_fallback.fetch_kpis_for_ticker(yf_sym, years)
                if fb and not fb.get("error"):
                    record["kpis"] = fb.get("kpis", {})
                    record["tag_used"] = fb.get("tag_used", {})
                    fetched_name = get_yf_longname(yf_sym)
                else:
                    record["error"] = (
                        fb["error"] if fb and fb.get("error")
                        else "Not on EDGAR and yfinance returned nothing"
                    )
                    fetched_name = get_yf_longname(yf_sym)
            else:
                record["cik"] = result["cik"]
                record["entity_name"] = result.get("entity_name", "")
                record["kpis"] = result["kpis"]
                record["tag_used"] = result["tag_used"]
                record["ambiguous_tags"] = result.get("ambiguous_tags", {})
                fetched_name = result.get("entity_name")
        else:
            fb = yf_fallback.fetch_kpis_for_ticker(yf_sym, years)
            if fb is None:
                record["error"] = "yfinance returned nothing"
            elif fb.get("error"):
                record["error"] = fb["error"]
            else:
                record["kpis"] = fb.get("kpis", {})
                record["tag_used"] = fb.get("tag_used", {})
            fetched_name = get_yf_longname(yf_sym)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    # If no reference name from CSV, fall back to the yfinance longName as the
    # reference too (self-consistency check for unmatched US tickers).
    reference_name = entry["reference_name"]
    if not reference_name and fetched_name and record["reference_source"] == "":
        reference_name = fetched_name
        record["reference_name"] = fetched_name
        record["reference_source"] = "yfinance"
        if not record["company_name"]:
            record["company_name"] = fetched_name

    record["fetched_name"] = fetched_name or ""
    verdict, score = verify_identity(reference_name, fetched_name)
    record["verified"] = verdict
    record["name_match_score"] = score
    return record


def record_filename(exchange_ocr: str, ticker: str) -> str:
    safe = f"{exchange_ocr}_{ticker}".replace("/", "_").replace(":", "_")
    return f"{safe}.json"


def write_record(record: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / record_filename(record["exchange_ocr"], record["ticker"])
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def summarize_line(record: dict) -> str:
    n_kpis = len(record.get("kpis", {}))
    years = record.get("years_fetch", [])
    covered = set()
    for v in record.get("kpis", {}).values():
        covered.update(int(y) for y in v)
    yr = f"{len(covered & set(years))}/{len(years)}yr"
    err = f"  [ERR: {record['error']}]" if record.get("error") else ""
    return (
        f"{record['ticker']:<14} {record['exchange']:<10} {record['source']:<22} "
        f"kpis={n_kpis:>2} {yr:<7} verify={record['verified']}{err}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ocr-root", type=Path, default=OCR_ROOT)
    p.add_argument("--min-year", type=int, default=2009,
                   help="Ignore OCR years before this value (EDGAR XBRL starts ~2009).")
    p.add_argument("--max-year", type=int, default=2024)
    p.add_argument("--dry-run", action="store_true",
                   help="Build + write the manifest only; no KPI/name fetching.")
    p.add_argument("--refresh-cache", action="store_true",
                   help="Re-download the SEC ticker->CIK map.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip tickers whose JSON already exists in output/raw/.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N tickers (smoke test).")
    args = p.parse_args(argv)

    # --- inventory ---
    print(f"Scanning {args.ocr_root} ...", file=sys.stderr)
    companies, unparsed = scan_ocr_dir(args.ocr_root)
    print(f"  {len(companies)} unique (exchange, ticker) pairs; "
          f"{len(unparsed)} unparsed dirs", file=sys.stderr)

    cleaned_lookup = load_cleaned_lookup()
    print(f"  Cleaned CSV entries: {len(cleaned_lookup)}", file=sys.stderr)

    manifest: list[dict] = []
    for (exchange_ocr, ticker_ocr), data in sorted(companies.items()):
        years_all = data["years"]
        years_fetch = [y for y in years_all if args.min_year <= y <= args.max_year]
        canonical, ref_name, ref_source, industry_csv = resolve_reference(
            exchange_ocr, ticker_ocr, cleaned_lookup
        )
        manifest.append({
            "ticker": canonical,
            "ticker_ocr": ticker_ocr,
            "exchange_ocr": exchange_ocr,
            "exchange": OCR_EXCHANGE_MAP.get(exchange_ocr, exchange_ocr),
            "reference_name": ref_name,
            "reference_source": ref_source,
            "industry_csv": industry_csv,
            "years_all": years_all,
            "years_fetch": years_fetch,
            "source": route_source(exchange_ocr),
        })

    # Merge entries that collapse to the same output file — e.g. the OCR tree
    # spells Imperial Brands as both LSE_IMB and LSE_IMB.L. Union their OCR years
    # so neither dir's coverage is silently dropped by a filename overwrite.
    merged: dict[str, dict] = {}
    for e in manifest:
        fname = record_filename(e["exchange_ocr"], e["ticker"])
        if fname in merged:
            prev = merged[fname]
            prev["years_all"] = sorted(set(prev["years_all"]) | set(e["years_all"]))
            prev["years_fetch"] = sorted(set(prev["years_fetch"]) | set(e["years_fetch"]))
            prev.setdefault("ticker_ocr_aliases", []).append(e["ticker_ocr"])
            if not prev.get("reference_name") and e.get("reference_name"):
                prev["reference_name"] = e["reference_name"]
                prev["reference_source"] = e["reference_source"]
        else:
            merged[fname] = e
    manifest = list(merged.values())

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(
        {"unparsed_dirs": unparsed, "companies": manifest}, indent=2
    ))
    matched = sum(1 for e in manifest if e["reference_source"] == "cleaned_csv")
    print(f"  Reference name from cleaned CSV: {matched}/{len(manifest)} "
          f"({len(manifest) - matched} need yfinance lookup)", file=sys.stderr)
    print(f"  Manifest written to {MANIFEST_PATH}", file=sys.stderr)

    if args.dry_run:
        print("\nDry-run — manifest only, no fetch.")
        from collections import Counter
        by_ex = Counter(e["exchange_ocr"] for e in manifest)
        by_src = Counter(e["source"] for e in manifest)
        print(f"  By exchange: {dict(by_ex)}")
        print(f"  By source:   {dict(by_src)}")
        return 0

    # --- fetch ---
    cik_map = edgar.load_ticker_cik_map(refresh=args.refresh_cache)
    todo = manifest[: args.limit] if args.limit else manifest
    print(f"\nFetching KPIs + verifying for {len(todo)} tickers...", file=sys.stderr)

    ok = errs = skipped = mismatches = 0
    for i, entry in enumerate(todo, 1):
        out_path = OUTPUT_DIR / record_filename(entry["exchange_ocr"], entry["ticker"])
        if args.skip_existing and out_path.exists():
            skipped += 1
            print(f"[{i:>4}/{len(todo)}] SKIP {entry['ticker']}")
            continue

        if not entry["years_fetch"]:
            record = {
                **{k: entry.get(k, "") for k in ("ticker", "ticker_ocr", "exchange_ocr")},
                "company_name": entry["reference_name"] or "",
                "exchange": entry["exchange"],
                "industry": entry.get("industry_csv") or "",
                "years_in_ocr": entry["years_all"],
                "years_fetch": [],
                "source": entry["source"],
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kpis": {}, "tag_used": {}, "ambiguous_tags": {},
                "reference_name": entry["reference_name"] or "",
                "reference_source": entry["reference_source"] or "",
                "fetched_name": "", "name_match_score": None,
                "verified": "no_reference" if not entry["reference_name"] else "unverified",
                "error": f"No years in [{args.min_year}, {args.max_year}] range",
            }
            write_record(record)
            errs += 1
            print(f"[{i:>4}/{len(todo)}] {entry['ticker']:<14} no years in fetch range")
            continue

        record = fetch_one(entry, cik_map)
        write_record(record)
        print(f"[{i:>4}/{len(todo)}] {summarize_line(record)}")
        if record.get("error"):
            errs += 1
        else:
            ok += 1
        if record["verified"] == "mismatch":
            mismatches += 1

    print(f"\nDone. ok={ok} errors={errs} skipped={skipped} "
          f"mismatches={mismatches} total={len(todo)}\n"
          f"JSON files in {OUTPUT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
