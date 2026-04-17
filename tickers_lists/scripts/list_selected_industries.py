"""List companies in a hand-picked set of (Sector, Industry) pairs, broken
down by stock exchange.

Reads per-exchange groupings from `grouped/<EXCHANGE>/companies_by_industry.json`
(produced by `group_industries.py`) and writes:

- `grouped/selected/companies.md`: human-readable breakdown
- `grouped/selected/companies.json`: Industry -> Exchange -> [companies]

Edit SELECTED below to change which industries to extract.
"""

from __future__ import annotations

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPED_DIR = os.path.join(ROOT, "grouped")
OUTPUT_DIR = os.path.join(GROUPED_DIR, "selected")

# (Sector, Industry) pairs to extract. Must match the yfinance labels exactly.
SELECTED: list[tuple[str, str]] = [
    ("Consumer Cyclical", "Auto Parts"),
    ("Basic Materials", "Specialty Chemicals"),
    ("Consumer Defensive", "Packaged Foods"),
    ("Energy", "Oil & Gas E&P"),
    ("Energy", "Oil & Gas Equipment & Services"),
    ("Financial Services", "Banks - Regional"),
    ("Real Estate", "REIT - Mortgage"),
]


def discover_exchanges() -> list[str]:
    exchanges: list[str] = []
    for path in sorted(glob.glob(os.path.join(GROUPED_DIR, "*", "companies_by_industry.json"))):
        name = os.path.basename(os.path.dirname(path))
        if name != "all" and name != "selected":
            exchanges.append(name)
    return exchanges


def load_exchange(exchange: str) -> dict[str, dict[str, list[dict[str, str]]]]:
    path = os.path.join(GROUPED_DIR, exchange, "companies_by_industry.json")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    exchanges = discover_exchanges()
    if not exchanges:
        print(f"No per-exchange groupings found under {GROUPED_DIR}")
        print("Run group_industries.py first.")
        return

    data = {exchange: load_exchange(exchange) for exchange in exchanges}

    # industry_key -> exchange -> [companies]
    result: dict[str, dict[str, list[dict[str, str]]]] = {}
    for sector, industry in SELECTED:
        key = f"{sector} / {industry}"
        result[key] = {}
        for exchange in exchanges:
            companies = data[exchange].get(sector, {}).get(industry, [])
            if companies:
                result[key][exchange] = companies

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "companies.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(OUTPUT_DIR, "companies.md")
    lines: list[str] = ["# Selected industries", ""]
    for industry_key, by_exchange in result.items():
        total = sum(len(c) for c in by_exchange.values())
        lines.append(f"## {industry_key} ({total} companies)")
        lines.append("")
        if not by_exchange:
            lines.append("_no matches_")
            lines.append("")
            continue
        for exchange in sorted(by_exchange):
            companies = by_exchange[exchange]
            lines.append(f"### {exchange} ({len(companies)} companies)")
            lines.append("")
            for c in companies:
                lines.append(f"- `{c['ticker']}` {c['name']}")
            lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    # Console summary.
    print(f"{'Industry':<55} " + " ".join(f"{e:>7}" for e in exchanges) + f" {'Total':>7}")
    print("-" * (56 + 8 * (len(exchanges) + 1)))
    for industry_key, by_exchange in result.items():
        counts = [len(by_exchange.get(e, [])) for e in exchanges]
        total = sum(counts)
        row = " ".join(f"{n:>7}" for n in counts)
        print(f"{industry_key:<55} {row} {total:>7}")

    print()
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
