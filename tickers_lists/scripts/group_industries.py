"""Group cleaned company rows by Sector → Industry.

Produces one grouping per exchange (under `grouped/<EXCHANGE>/`) and a
combined grouping across all exchanges (under `grouped/all/`). Doing per-
exchange groupings first is useful because yfinance sometimes resolves LSE
tickers to their NYSE-listed counterparts, so the combined view has known
duplicates — the per-exchange views help identify them.

Each output directory contains:

- `companies_by_industry.json`: nested Sector -> Industry -> [companies]
- `summary.md`: human-readable counts per sector/industry (sorted by size)
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(ROOT, "cleaned")
OUTPUT_DIR = os.path.join(ROOT, "grouped")

Company = dict[str, str]
Grouped = dict[str, dict[str, list[Company]]]


def exchange_from_path(path: str) -> str:
    # cleaned/{EXCHANGE}_mapped_clean.csv
    return os.path.basename(path).split("_mapped_clean.csv")[0]


def load_rows_by_exchange() -> dict[str, list[Company]]:
    per_exchange: dict[str, list[Company]] = {}
    for path in sorted(glob.glob(os.path.join(CLEANED_DIR, "*_mapped_clean.csv"))):
        exchange = exchange_from_path(path)
        with open(path, newline="") as f:
            rows = [{**row, "Exchange": exchange} for row in csv.DictReader(f)]
        if rows:
            per_exchange[exchange] = rows
    return per_exchange


def group_rows(rows: list[Company]) -> Grouped:
    grouped: dict[str, dict[str, list[Company]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["Sector"]][row["Industry"]].append(
            {
                "ticker": row["Ticker"],
                "name": row["Company Name"],
                "exchange": row["Exchange"],
            }
        )

    # Sort deterministically: sectors alphabetically, industries by size desc
    # then alphabetically, companies by ticker.
    sorted_out: Grouped = {}
    for sector in sorted(grouped):
        industries = grouped[sector]
        sorted_industries = sorted(
            industries.items(), key=lambda kv: (-len(kv[1]), kv[0])
        )
        sorted_out[sector] = {
            industry: sorted(companies, key=lambda c: c["ticker"])
            for industry, companies in sorted_industries
        }
    return sorted_out


def write_json(grouped: Grouped, path: str) -> None:
    with open(path, "w") as f:
        json.dump(grouped, f, indent=2, ensure_ascii=False)


def write_summary(grouped: Grouped, path: str, title: str) -> None:
    lines: list[str] = [f"# {title}", ""]
    for sector, industries in grouped.items():
        sector_total = sum(len(c) for c in industries.values())
        lines.append(f"## {sector} ({sector_total} companies)")
        lines.append("")
        for industry, companies in industries.items():
            lines.append(f"- **{industry}** — {len(companies)} companies")
            for c in companies:
                lines.append(f"  - `{c['ticker']}` ({c['exchange']}) {c['name']}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def stats(grouped: Grouped) -> tuple[int, int, int, int, int, int]:
    sectors = len(grouped)
    sizes = [
        len(companies)
        for industries_ in grouped.values()
        for companies in industries_.values()
    ]
    industries = len(sizes)
    ge_2 = sum(1 for n in sizes if n >= 2)
    ge_5 = sum(1 for n in sizes if n >= 5)
    ge_10 = sum(1 for n in sizes if n >= 10)
    companies = sum(sizes)
    return sectors, industries, ge_2, ge_5, ge_10, companies


def write_group(grouped: Grouped, label: str) -> None:
    out_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    write_json(grouped, os.path.join(out_dir, "companies_by_industry.json"))
    write_summary(
        grouped,
        os.path.join(out_dir, "summary.md"),
        title=f"Industry peer groups — {label}",
    )


def main() -> None:
    per_exchange = load_rows_by_exchange()
    if not per_exchange:
        print(f"No cleaned CSVs found in {CLEANED_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Per-exchange groupings.
    header = (
        f"{'Exchange':<10} {'Sectors':>8} {'Industries':>11} "
        f"{'(≥2)':>6} {'(≥5)':>6} {'(≥10)':>6} {'Companies':>10}"
    )
    print(header)
    print("-" * len(header))

    def row(label: str, grouped: Grouped) -> None:
        s, i, m2, m5, m10, c = stats(grouped)
        print(
            f"{label:<10} {s:>8} {i:>11} {m2:>6} {m5:>6} {m10:>6} {c:>10}"
        )

    for exchange in sorted(per_exchange):
        grouped = group_rows(per_exchange[exchange])
        write_group(grouped, exchange)
        row(exchange, grouped)

    # Combined grouping across all exchanges (duplicates kept as-is).
    all_rows = [r for rows in per_exchange.values() for r in rows]
    combined = group_rows(all_rows)
    write_group(combined, "all")
    row("all", combined)

    print()
    print("Top 10 industry peer groups (combined):")
    flat = [
        (sector, industry, len(companies))
        for sector, industries in combined.items()
        for industry, companies in industries.items()
    ]
    flat.sort(key=lambda t: -t[2])
    for sector, industry, n in flat[:10]:
        print(f"  {n:>4}  {sector} / {industry}")

    print()
    print(f"Wrote groupings under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
