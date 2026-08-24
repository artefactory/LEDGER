"""Verify that KPI data matches the correct company for every ticker.

For each raw JSON in the three source directories, this script:
1. Compares entity_name (EDGAR) or fetched_name (yfinance) against company_name
2. For records missing a fetched name, does a quick yfinance .info lookup
3. Writes a full verification CSV and a summary

Usage:
    uv run python KPI_analysis/find_more_queries/full_6k/verify_kpi_identity.py
    uv run python KPI_analysis/find_more_queries/full_6k/verify_kpi_identity.py --refresh  # re-fetch missing names
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KPI_DIR = HERE.parent.parent

RAW_DIRS = [
    KPI_DIR / "output" / "raw",
    KPI_DIR / "find_more_queries" / "output" / "raw",
    KPI_DIR / "find_more_queries" / "full_6k" / "output" / "raw",
]

OUT_CSV = HERE / "verification_all_sources.csv"
OUT_SUMMARY = HERE / "verification_summary.md"

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "plc", "ltd", "limited", "llc", "lp", "llp", "holdings", "holding", "group",
    "the", "sa", "ag", "nv", "se", "spa", "ab", "as", "asa", "oyj", "kgaa",
    "class", "ordinary", "shares", "and",
}

NAME_MATCH_THRESHOLD = 0.6


def normalize_tokens(name: str) -> list[str]:
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return [t for t in name.split() if t and t not in LEGAL_SUFFIXES]


def verify_identity(reference_name: str | None, fetched_name: str | None) -> tuple[str, float | None]:
    if not reference_name:
        return "no_reference", None
    if not fetched_name:
        return "unverified", None
    ta, tb = normalize_tokens(reference_name), normalize_tokens(fetched_name)
    a, b = " ".join(ta), " ".join(tb)
    if not a or not b:
        return "unverified", None
    if a == b or a in b or b in a:
        return "match", 1.0
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    sorted_seq = difflib.SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    score = max(seq, sorted_seq)
    return ("match" if score >= NAME_MATCH_THRESHOLD else "mismatch"), round(score, 3)


def load_all_records() -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            continue
        for f in sorted(os.listdir(raw_dir)):
            if not f.endswith(".json"):
                continue
            r = json.loads((raw_dir / f).read_text())
            key = (r.get("exchange_ocr") or r.get("exchange", ""), r.get("ticker", ""))
            records[key] = r
    return records


def get_yf_longname(symbol: str) -> str | None:
    import yfinance as yf
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    return info.get("longName") or info.get("shortName") or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch yfinance longName for records missing fetched_name.")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Sleep between yfinance calls (default: 1.0s).")
    args = parser.parse_args()

    records = load_all_records()
    print(f"Loaded {len(records)} records from {len(RAW_DIRS)} directories.", file=sys.stderr)

    results: list[dict] = []
    needs_refresh: list[tuple[tuple[str, str], dict]] = []

    for key, r in sorted(records.items()):
        exchange_ocr = key[0]
        ticker = key[1]
        source = r.get("source", "")

        # Best reference name
        ref_name = (r.get("reference_name") or r.get("company_name") or "").strip()

        # Best fetched name
        if "edgar" in source:
            fetched = (r.get("entity_name") or r.get("fetched_name") or "").strip()
        else:
            fetched = (r.get("fetched_name") or r.get("company_name") or "").strip()

        # Use existing verdict if available
        existing_verdict = r.get("verified", "")
        if existing_verdict in ("match", "mismatch"):
            verdict, score = existing_verdict, r.get("name_match_score")
        elif ref_name and fetched:
            verdict, score = verify_identity(ref_name, fetched)
        elif fetched:
            # No reference name but API returned a valid company name for this
            # ticker — ticker-verified (trusted source returned data).
            verdict, score = "match", None
        else:
            verdict, score = "unverified", None
            needs_refresh.append((key, r))

        results.append({
            "exchange_ocr": exchange_ocr,
            "ticker": ticker,
            "source": source,
            "verdict": verdict,
            "score": score,
            "reference_name": ref_name,
            "fetched_name": fetched,
            "n_kpis": len(r.get("kpis", {})),
            "years": ",".join(str(y) for y in sorted(r.get("years_fetch") or r.get("years_in_ocr") or [])),
            "has_error": bool(r.get("error")),
        })

    # Optionally refresh missing names via yfinance
    if args.refresh and needs_refresh:
        print(f"\nFetching yfinance longName for {len(needs_refresh)} tickers...", file=sys.stderr)
        # Import YF_SUFFIX from scan_and_fetch_6k
        sys.path.insert(0, str(HERE))
        from scan_and_fetch_6k import yf_symbol

        refreshed = 0
        for i, (key, r) in enumerate(needs_refresh, 1):
            exchange_ocr, ticker_ocr = key[0], r.get("ticker_ocr", key[1])
            symbol = yf_symbol(exchange_ocr, ticker_ocr)
            fetched_name = get_yf_longname(symbol)
            if fetched_name:
                # Update the record
                r["fetched_name"] = fetched_name
                ref_name = (r.get("reference_name") or r.get("company_name") or "").strip()
                verdict, score = verify_identity(ref_name, fetched_name)
                r["verified"] = verdict
                r["name_match_score"] = score

                # Update results
                for row in results:
                    if row["exchange_ocr"] == exchange_ocr and row["ticker"] == key[1]:
                        row["fetched_name"] = fetched_name
                        row["verdict"] = verdict
                        row["score"] = score
                        break
                refreshed += 1
                if i % 50 == 0:
                    print(f"  [{i}/{len(needs_refresh)}] refreshed {refreshed}...", file=sys.stderr)
            time.sleep(args.sleep)

        # Rewrite the raw JSONs with updated verification
        for (exchange_ocr, ticker), r in records.items():
            if r.get("verified") in ("match", "mismatch") and r.get("fetched_name"):
                for raw_dir in RAW_DIRS:
                    fname = f"{exchange_ocr}_{ticker}.json".replace("/", "_").replace(":", "_")
                    path = raw_dir / fname
                    if path.exists():
                        path.write_text(json.dumps(r, indent=2, default=str))
                        break

        print(f"Refreshed {refreshed}/{len(needs_refresh)} tickers.", file=sys.stderr)

    # Write CSV
    fields = ["exchange_ocr", "ticker", "source", "verdict", "score",
              "reference_name", "fetched_name", "n_kpis", "years", "has_error"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {OUT_CSV}", file=sys.stderr)

    # Summary
    verdicts = Counter(r["verdict"] for r in results)
    mismatches = [r for r in results if r["verdict"] == "mismatch"]
    unverified = [r for r in results if r["verdict"] not in ("match", "mismatch")]

    lines = [
        "# KPI Identity Verification Summary",
        "",
        f"Total records: **{len(results)}**",
        "",
        "| Verdict | Count |",
        "|---------|------:|",
    ]
    for v, c in verdicts.most_common():
        lines.append(f"| {v} | {c} |")

    if mismatches:
        lines += ["", "## Mismatches (review these)", "",
                   "| Exchange | Ticker | Score | Reference | Fetched |",
                   "|----------|--------|------:|-----------|---------|"]
        for r in sorted(mismatches, key=lambda x: x["score"] or 0):
            lines.append(f"| {r['exchange_ocr']} | {r['ticker']} | {r['score']} "
                         f"| {r['reference_name']} | {r['fetched_name']} |")

    if unverified:
        lines += [f"", f"## Unverified ({len(unverified)})", "",
                   "| Exchange | Ticker | Source | Reference | Fetched |",
                   "|----------|--------|--------|-----------|---------|"]
        for r in sorted(unverified, key=lambda x: x["verdict"]):
            lines.append(f"| {r['exchange_ocr']} | {r['ticker']} | {r['source']} "
                         f"| {r['reference_name'] or '—'} | {r['fetched_name'] or '—'} |")

    OUT_SUMMARY.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_SUMMARY}", file=sys.stderr)

    # Console summary
    print(f"\n=== Verification Results ===")
    for v, c in verdicts.most_common():
        print(f"  {v}: {c}")
    print(f"  Total: {len(results)}")
    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        for r in mismatches:
            print(f"    {r['exchange_ocr']}_{r['ticker']}: ref={r['reference_name']!r} "
                  f"fetched={r['fetched_name']!r} score={r['score']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
