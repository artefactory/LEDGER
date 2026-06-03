"""One-off migration: re-key ``output/raw/*.json`` from period-end-calendar-year
to filer-labelled FY (per ``_fiscal.filer_fy_from_period_end``).

This script uses ONLY data that is already on disk:
  - ``cache/companyfacts/CIK*.json`` for EDGAR-sourced KPIs
  - ``cache/alphavantage/{symbol}__{ENDPOINT}.json`` for Alpha Vantage gap-fills

It does NOT call the network. Records that source data exclusively from
yfinance are passed through unchanged with a ``_fy_migration`` flag set to
``yfinance_unchanged`` so they can be re-fetched later if desired.

Why a dedicated migration rather than re-running ``fetch_kpis.py``? The user
explicitly asked to avoid re-fetching where possible — yfinance has no
on-disk cache so a full ``fetch_kpis.py`` re-run would re-hit live yfinance.
This script keeps yfinance records as-is (acceptable when the affected
fiscal years end in months 4-12, which they do for typical European /
Asian filers in the auto-parts subset).

Usage::

    uv run python -m KPI_analysis.kpi_fetch_and_build.migrate_fy_keying --years 2017-2022

The script overwrites ``output/raw/{TICKER}.json`` files in place after
performing migration. Pass ``--dry-run`` to preview without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
KPI_ROOT = HERE.parent

try:
    from . import alpha_vantage as av  # noqa: E402
    from . import edgar  # noqa: E402
except ImportError:
    import alpha_vantage as av  # noqa: E402
    import edgar  # noqa: E402

DEFAULT_RAW_DIR = KPI_ROOT / "output" / "raw"
COMPANYFACTS_CACHE = KPI_ROOT / "cache" / "companyfacts"
AV_CACHE = KPI_ROOT / "cache" / "alphavantage"


def parse_year_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        lo, hi = int(a), int(b)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(y.strip()) for y in s.split(",") if y.strip()]


def _stringify_year_keys(d: dict[int, float]) -> dict[str, float]:
    return {str(int(y)): float(v) for y, v in d.items()}


def _stringify_year_keys_nested(
    d: dict[int, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {str(int(y)): {k: float(v) for k, v in inner.items()} for y, inner in d.items()}


def rebuild_edgar_block(record: dict, years: list[int]) -> bool:
    """Re-extract EDGAR KPIs from cache. Returns True if any data was loaded."""
    cik = record.get("cik")
    if not cik:
        return False
    cik_padded = str(cik).zfill(10)
    facts_path = COMPANYFACTS_CACHE / f"CIK{cik_padded}.json"
    if not facts_path.exists():
        return False
    facts = json.loads(facts_path.read_text())
    values, tag_used, ambiguous = edgar.extract_kpis_for_years(facts, years)
    if not values:
        return False
    record["kpis"] = {k: _stringify_year_keys(by_year) for k, by_year in values.items()}
    record["tag_used"] = dict(tag_used)
    record["ambiguous_tags"] = (
        _build_ambiguous(ambiguous) if ambiguous else {}
    )
    return True


def _build_ambiguous(amb: dict) -> dict:
    """Mirror fetch_kpis.py's serialised shape (string-keyed years)."""
    return {
        kpi: _stringify_year_keys_nested(per_year) for kpi, per_year in amb.items()
    }


def load_av_payloads(symbol_used: str) -> dict[str, dict] | None:
    """Load every cached AV endpoint payload for ``symbol_used``.

    Returns ``{endpoint: payload}`` or None if no cache files exist.
    """
    if not symbol_used:
        return None
    found: dict[str, dict] = {}
    for ep in (*av.DEFAULT_ENDPOINTS, av.EARNINGS_ENDPOINT):
        path = AV_CACHE / f"{symbol_used}__{ep}.json"
        if path.exists():
            try:
                found[ep] = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
    return found or None


def merge_av_block(record: dict, years: list[int]) -> int:
    """Re-merge AV gap-fills into ``record["kpis"]`` from cached payloads.

    Returns the number of (kpi, year) cells filled by AV.
    """
    av_meta = record.get("alphavantage")
    if not isinstance(av_meta, dict):
        return 0
    symbol = av_meta.get("symbol_used")
    payloads = load_av_payloads(symbol)
    if not payloads:
        return 0

    av_values, av_tag_used, av_currency = av.extract_kpis_for_years(payloads, years)
    if av_currency:
        # Preserve the previously recorded currency map verbatim where present;
        # the cache replay should match it but we don't enforce.
        av_meta.setdefault("reported_currency", {}).update(av_currency)

    added = 0
    kpis_out = record.setdefault("kpis", {})
    av_tags_out = record.setdefault("alphavantage_tag_used", {})
    for kpi, by_year in av_values.items():
        bucket = kpis_out.setdefault(kpi, {})
        for y, v in by_year.items():
            ys = str(int(y))
            if ys in bucket:
                continue
            bucket[ys] = float(v)
            added += 1
            if kpi in av_tag_used:
                av_tags_out[kpi] = av_tag_used[kpi]
    av_meta["cells_added_after_migration"] = added
    return added


def migrate_record(record: dict, years: list[int]) -> dict:
    """Apply the appropriate migration depending on ``record['source']``.

    Modifies ``record`` in-place and adds a ``_fy_migration`` key explaining
    what happened.
    """
    source = (record.get("source") or "").lower()
    flags: list[str] = []

    edgar_used = "edgar" in source and "edgar miss" not in source
    av_used = "alphavantage" in source
    yf_only_or_fallback = (not edgar_used) and ("yfinance" in source)

    if edgar_used:
        ok = rebuild_edgar_block(record, years)
        flags.append("edgar_rekeyed" if ok else "edgar_no_cache")
    if av_used:
        added = merge_av_block(record, years)
        flags.append(f"av_rekeyed(+{added})")
    if yf_only_or_fallback:
        flags.append("yfinance_unchanged")
    if not flags:
        flags.append("noop")

    record["_fy_migration"] = "; ".join(flags)
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--years", default="2017-2022")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would change without writing.")
    p.add_argument("--ticker", default=None,
                   help="Only migrate one ticker (for debugging).")
    args = p.parse_args(argv)

    years = parse_year_range(args.years)
    raw_dir = args.raw_dir
    files = sorted(raw_dir.glob("*.json"))
    if args.ticker:
        files = [f for f in files if f.stem == args.ticker]

    if not files:
        print(f"[migrate] no files in {raw_dir}", file=sys.stderr)
        return 1

    print(f"[migrate] {len(files)} record(s); years={years[0]}-{years[-1]}; "
          f"dry_run={args.dry_run}", file=sys.stderr)

    bucket_counts: dict[str, int] = {}
    for path in files:
        try:
            rec = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  {path.name}: ERROR loading: {e}", file=sys.stderr)
            continue

        old_keys = {
            kpi: sorted(v.keys()) for kpi, v in (rec.get("kpis") or {}).items()
        }

        migrate_record(rec, years)

        new_keys = {
            kpi: sorted(v.keys()) for kpi, v in (rec.get("kpis") or {}).items()
        }
        moved = sum(
            1 for kpi in set(old_keys) | set(new_keys)
            if old_keys.get(kpi) != new_keys.get(kpi)
        )

        flag = rec["_fy_migration"]
        bucket_counts[flag] = bucket_counts.get(flag, 0) + 1
        print(
            f"  {path.stem:<10} src={rec.get('source','?'):<28} "
            f"flags={flag:<35} kpis_with_changed_keys={moved}"
        )

        if not args.dry_run:
            path.write_text(json.dumps(rec, indent=2, default=str))

    print(f"\n[migrate] summary by flag:", file=sys.stderr)
    for flag, n in sorted(bucket_counts.items()):
        print(f"  {flag:<35} {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
