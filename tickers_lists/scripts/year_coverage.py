"""Find the best common-year windows per industry.

For each industry directory under `--pdfs` (default: the selected-PDFs tree),
parses filenames of the form `EXCHANGE_TICKER_YEAR.pdf` and computes, for
every consecutive year window, how many companies have reports in ALL
years of the window.

For each industry and each window size k, the script reports:

- the window of k consecutive years that maximizes the number of covering
  companies,
- the implied total document count (k × covering companies),
- the retention (kept docs / total docs for that industry).

Written to stdout as a table plus `grouped/selected/year_coverage.json` +
`grouped/selected/year_coverage.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PDFS = "/data/raw_data/argimi_corpuses/annual_reports_pdfs_selected"
OUT_DIR = os.path.join(ROOT, "grouped", "selected")
JSON_PATH = os.path.join(OUT_DIR, "year_coverage.json")
MD_PATH = os.path.join(OUT_DIR, "year_coverage.md")

FILENAME_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})\.pdf$")


def parse_filename(name: str) -> tuple[str, str, int] | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def scan_industry(industry_dir: str) -> dict[tuple[str, str], set[int]]:
    company_years: dict[tuple[str, str], set[int]] = defaultdict(set)
    for f in os.listdir(industry_dir):
        parsed = parse_filename(f)
        if parsed is None:
            continue
        exchange, ticker, year = parsed
        company_years[(exchange, ticker)].add(year)
    return company_years


def year_histogram(company_years: dict[tuple[str, str], set[int]]) -> dict[int, int]:
    hist: dict[int, int] = defaultdict(int)
    for years in company_years.values():
        for y in years:
            hist[y] += 1
    return dict(sorted(hist.items()))


def best_window_per_k(
    company_years: dict[tuple[str, str], set[int]],
) -> list[dict]:
    all_years = sorted({y for years in company_years.values() for y in years})
    if not all_years:
        return []
    y_min, y_max = all_years[0], all_years[-1]
    span = y_max - y_min + 1

    results: list[dict] = []
    for k in range(1, span + 1):
        best_window: tuple[int, int] | None = None
        best_count = 0
        # Prefer more-recent windows when ties occur (iterate low-to-high,
        # update on strict improvement then on equality at higher `a`).
        for a in range(y_min, y_max - k + 2):
            b = a + k - 1
            window = set(range(a, b + 1))
            count = sum(
                1 for years in company_years.values() if window.issubset(years)
            )
            if count > best_count or (
                count == best_count and best_window is not None and a > best_window[0]
            ):
                best_window = (a, b)
                best_count = count
        if best_window is None or best_count == 0:
            continue
        a, b = best_window
        results.append(
            {
                "k": k,
                "window": [a, b],
                "companies": best_count,
                "docs": k * best_count,
            }
        )
    return results


def render_histogram(hist: dict[int, int], width: int = 40) -> list[str]:
    if not hist:
        return []
    peak = max(hist.values())
    lines: list[str] = []
    for year, n in hist.items():
        bar = "#" * max(1, int(round(width * n / peak))) if n else ""
        lines.append(f"  {year}  {n:>4}  {bar}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", default=DEFAULT_PDFS,
                        help="Root containing one subdirectory per industry.")
    args = parser.parse_args()

    if not os.path.isdir(args.pdfs):
        raise SystemExit(f"PDF root not found: {args.pdfs}")

    industries = sorted(
        d for d in os.listdir(args.pdfs)
        if os.path.isdir(os.path.join(args.pdfs, d))
    )

    report: dict[str, dict] = {}
    md_lines: list[str] = ["# Year coverage per industry", ""]

    for industry in industries:
        path = os.path.join(args.pdfs, industry)
        company_years = scan_industry(path)
        if not company_years:
            continue

        total_docs = sum(len(y) for y in company_years.values())
        total_companies = len(company_years)
        hist = year_histogram(company_years)
        windows = best_window_per_k(company_years)

        report[industry] = {
            "companies": total_companies,
            "docs": total_docs,
            "year_histogram": hist,
            "best_windows": windows,
        }

        # Pick a suggested window: highest docs, break ties by larger k.
        suggestion = max(windows, key=lambda w: (w["docs"], w["k"])) if windows else None

        # Console block per industry.
        print(f"\n=== {industry} ===")
        print(
            f"{total_companies} companies, {total_docs} docs, "
            f"years {min(hist)}–{max(hist)}"
        )
        if suggestion:
            a, b = suggestion["window"]
            pct = 100 * suggestion["docs"] / total_docs
            print(
                f"Best (k×C) window: {a}-{b}  "
                f"({suggestion['companies']} companies × {suggestion['k']} years "
                f"= {suggestion['docs']} docs, {pct:.0f}% retention)"
            )
        print()
        print(f"  {'k':>3}  {'window':<11}  {'companies':>9}  {'docs':>5}  retention")
        for w in windows:
            a, b = w["window"]
            pct = 100 * w["docs"] / total_docs
            marker = "  ←" if w is suggestion else ""
            print(
                f"  {w['k']:>3}  {a}-{b:<6}  {w['companies']:>9}  "
                f"{w['docs']:>5}  {pct:>5.0f}%{marker}"
            )

        # Markdown block.
        md_lines.append(f"## {industry}")
        md_lines.append("")
        md_lines.append(
            f"- Companies: **{total_companies}**  "
            f"Total docs: **{total_docs}**  "
            f"Year range: **{min(hist)}–{max(hist)}**"
        )
        if suggestion:
            a, b = suggestion["window"]
            pct = 100 * suggestion["docs"] / total_docs
            md_lines.append(
                f"- Best window by total docs: **{a}–{b}** "
                f"({suggestion['companies']} companies × {suggestion['k']} years "
                f"= {suggestion['docs']} docs, {pct:.0f}% retention)"
            )
        md_lines.append("")
        md_lines.append("### Year histogram")
        md_lines.append("")
        md_lines.append("```")
        md_lines.extend(render_histogram(hist))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("### Best window per size")
        md_lines.append("")
        md_lines.append("| k | window | companies | docs | retention |")
        md_lines.append("|---:|:---|---:|---:|---:|")
        for w in windows:
            a, b = w["window"]
            pct = 100 * w["docs"] / total_docs
            md_lines.append(
                f"| {w['k']} | {a}–{b} | {w['companies']} | {w['docs']} | {pct:.0f}% |"
            )
        md_lines.append("")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)
    with open(MD_PATH, "w") as f:
        f.write("\n".join(md_lines))

    print()
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
