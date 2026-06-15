"""Post-filter for all_companies_alt_names.json.

The alias list is built by ``tickers_lists/scripts/get_all_company_names.py``
from Wikipedia redirects + DBPedia. For most companies it returns a clean
1-5 entry list. For ~5% of mega-cap issuers (AAPL, MSFT, MCD, CVX, INTC, F,
T, ...) it floods with three kinds of noise:

  1. Wikipedia meta-pages -- "List of X", "History of X", "Operations of X",
     "Campaign contributions ...", "Political contributions ...",
     "Environmental record of X", "Criticism of X", ...
  2. Domain-name redirects -- "Apple.jo", "Microsoft.de", "Apple.com.cn", ...
  3. PUA / non-printable glyphs -- " U+F8FF Inc" (the Apple-logo artifact
     from DBPedia reflecting " ][ Inc" titles).

We also drop subsidiary-style names that almost always indicate a regional
or operating subsidiary rather than a parent alias -- but conservatively,
and only when at least one clean parent alias is also present. The
qualifier set is intentionally small (``LLC``, ``GmbH``) because broader
patterns like "International" or "Ltd." have too many false positives
(e.g. "International Flavors & Fragrances", "Tesco Ltd" are real parents).

Usage
-----
  uv run python filter_alt_names.py
  uv run python filter_alt_names.py --output /path/to/filtered.json
  uv run python filter_alt_names.py --keep-empty
"""

import argparse
import json
import re
from pathlib import Path


META_PREFIXES = (
    "list of", "lists of",
    "history of", "histories of",
    "operations of",
    "campaign contributions", "political contributions",
    "environmental record of",
    "criticism of", "controversy of", "controversies of",
    "impact of", "global impact of",
)

DOMAIN_RE = re.compile(r"\.[a-z]{2,6}$", re.IGNORECASE)
PUA_RE = re.compile(r"[\uE000-\uF8FF]")

# Conservative subsidiary-indicator set. Matched as case-insensitive whole
# words with trailing optional period. Keep small: broadening to "Ltd.",
# "International", "AG" etc. produces too many false positives (e.g. "Tesco
# Ltd", "International Flavors & Fragrances", "Volkswagen AG" are parents,
# not subsidiaries). The word boundary is important: substring matching
# would flag e.g. "Wellcome" because it contains the letters "llc".
SUBSIDIARY_QUALIFIERS = ("llc", "gmbh")
SUBSIDIARY_RE = re.compile(
    r"\b(" + "|".join(re.escape(q) for q in SUBSIDIARY_QUALIFIERS) + r")\b\.?",
    re.IGNORECASE,
)


def is_meta_page(name: str) -> bool:
    lower = name.strip().lower()
    return any(lower.startswith(p) for p in META_PREFIXES)


def is_domain_noise(name: str) -> bool:
    return bool(DOMAIN_RE.search(name.strip()))


def has_pua_or_control(name: str) -> bool:
    return bool(PUA_RE.search(name))


def is_subsidiary_like(name: str) -> bool:
    return bool(SUBSIDIARY_RE.search(name))


def filter_aliases(names):
    """Drop noise, then drop subsidiary-like entries when a clean parent exists."""
    cleaned = [
        n for n in names
        if n.strip()
        and not is_meta_page(n)
        and not is_domain_noise(n)
        and not has_pua_or_control(n)
    ]
    clean_parents = [n for n in cleaned if not is_subsidiary_like(n)]
    if clean_parents:
        return clean_parents
    return cleaned


def filter_json(d):
    out = {}
    dropped_empty = 0
    for key, names in d.items():
        kept = filter_aliases(names)
        if not kept:
            dropped_empty += 1
            continue
        out[key] = kept
    return out, dropped_empty


def stats(d):
    sizes = sorted(len(v) for v in d.values())
    n = len(sizes)
    if n == 0:
        return {"keys": 0, "total_aliases": 0, "median": 0, "p90": 0, "max": 0,
                "keys_with_<=3": 0, "keys_with_4_to_20": 0, "keys_with_>20": 0,
                "top_5_largest": []}
    return {
        "keys": n,
        "total_aliases": sum(sizes),
        "median": sizes[n // 2],
        "p90": sizes[9 * n // 10],
        "max": sizes[-1],
        "keys_with_<=3": sum(1 for s in sizes if s <= 3),
        "keys_with_4_to_20": sum(1 for s in sizes if 4 <= s <= 20),
        "keys_with_>20": sum(1 for s in sizes if s > 20),
        "top_5_largest": sorted(d.items(), key=lambda kv: -len(kv[1]))[:5],
    }


DEFAULT_INPUT = (
    "/home/cmoslonka/ardian_dataset_bench/KPI_analysis/find_more_queries/full_6k/"
    "all_companies_alt_names.json"
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Path to input JSON.")
    ap.add_argument("--output", default=None,
                    help="Path to output JSON. Default: overwrite input.")
    ap.add_argument("--keep-empty", action="store_true",
                    help="Keep keys whose alias list becomes empty after filtering.")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src

    with src.open() as f:
        data = json.load(f)

    before = stats(data)
    filtered, dropped_empty = filter_json(data)
    after = stats(filtered)

    with dst.open("w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Input : {src}")
    print(f"Output: {dst}")
    print(f"Empty keys dropped: {dropped_empty}  "
          f"({'kept' if args.keep_empty else 'removed'})")
    print()
    print(f"{'metric':30s}  {'before':>10s}  {'after':>10s}  {'delta':>10s}")
    rows = [
        ("keys", before["keys"], after["keys"]),
        ("total aliases", before["total_aliases"], after["total_aliases"]),
        ("median aliases / key", before["median"], after["median"]),
        ("p90 aliases / key", before["p90"], after["p90"]),
        ("max aliases / key", before["max"], after["max"]),
        ("keys with <=3 aliases", before["keys_with_<=3"], after["keys_with_<=3"]),
        ("keys with 4-20 aliases", before["keys_with_4_to_20"], after["keys_with_4_to_20"]),
        ("keys with >20 aliases", before["keys_with_>20"], after["keys_with_>20"]),
    ]
    for label, b, a in rows:
        delta = a - b
        print(f"{label:30s}  {b:>10d}  {a:>10d}  {delta:>+10d}")
    print()
    print("Top-5 longest alias lists AFTER filtering:")
    for k, v in after["top_5_largest"]:
        print(f"  {k}  ({len(v)} names)")
        for n in v[:8]:
            print(f"      {n}")


if __name__ == "__main__":
    main()
