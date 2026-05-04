"""Flag OCR'd annual-report subdirs whose opening pages look like a US SEC Form 10-K.

Expected input layout (default `sample_data/subset_auto_parts_2017_2022/`):

    <root>/
        EX_TICKER_YEAR/
            EX_TICKER_YEAR.mmd   # pages separated by "<--- Page Split --->"
            ...

The root may also contain grouping subdirectories (e.g. industries) where each
group contains one subdirectory per OCR report:

    <root>/
        group_a/
            EX_TICKER_YEAR/
                EX_TICKER_YEAR.mmd
        group_b/
            EX_TICKER_YEAR/
                EX_TICKER_YEAR.mmd

For each report, read the first --pages pages and test them against a list of
10-K marker regexes. Write the result as JSON (one record per report) and, for
convenience, a plain-text list of the names flagged as 10-Ks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

PAGE_SPLIT = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)

# Each marker is (label, compiled regex, strong). Patterns are case-insensitive
# and tolerant of whitespace / punctuation noise introduced by OCR.
#
# "strong" markers only appear on a genuine 10-K cover page; "weak" markers
# (e.g. a passing mention of Form 10-K, or "Securities Exchange Act of 1934"
# inside a forward-looking-statements disclaimer) can bleed into glossy
# shareholder reports, so we do not count them by themselves.
MARKERS: list[tuple[str, re.Pattern[str], bool]] = [
    (
        "sec_header",
        re.compile(
            r"UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
            re.IGNORECASE,
        ),
        True,
    ),
    (
        "washington_dc_20549",
        re.compile(r"WASHINGTON,?\s*D\.?\s*C\.?\s*20549", re.IGNORECASE),
        True,
    ),
    (
        "commission_file_number",
        re.compile(r"COMMISSION\s+FILE\s+(?:NUMBER|NO\.?)", re.IGNORECASE),
        True,
    ),
    (
        "annual_report_section_13_15d",
        re.compile(
            r"ANNUAL\s+REPORT\s+(?:PURSUANT\s+TO|UNDER)\s+SECTION\s*13\s*OR\s*15\s*\(?\s*d\s*\)?",
            re.IGNORECASE,
        ),
        True,
    ),
    ("form_10k", re.compile(r"\bFORM\s*10[-\s]?K\b", re.IGNORECASE), False),
    (
        "securities_exchange_act_1934",
        re.compile(r"SECURITIES\s+EXCHANGE\s+ACT\s+OF\s+1934", re.IGNORECASE),
        False,
    ),
]
STRONG_LABELS = {label for label, _, strong in MARKERS if strong}


@dataclass
class Result:
    name: str
    mmd_path: str
    is_10k: bool
    matched_markers: list[str]
    first_match_page: int | None
    pages_scanned: int


def first_pages(text: str, n: int) -> list[str]:
    parts = PAGE_SPLIT.split(text)
    return parts[:n]


def classify(text: str, pages: int) -> tuple[list[str], int | None, int]:
    head_pages = first_pages(text, pages)
    matched: list[str] = []
    first_page: int | None = None
    for idx, page in enumerate(head_pages):
        for label, pattern, _ in MARKERS:
            if pattern.search(page):
                if label not in matched:
                    matched.append(label)
                if first_page is None:
                    first_page = idx
    return matched, first_page, len(head_pages)


def find_mmd(report_dir: Path) -> Path | None:
    # Prefer the canonical `<name>.mmd` over the `_det.mmd` variant.
    preferred = report_dir / f"{report_dir.name}.mmd"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        p for p in report_dir.glob("*.mmd") if not p.name.endswith("_det.mmd")
    )
    if candidates:
        return candidates[0]
    fallback = sorted(report_dir.glob("*.mmd"))
    return fallback[0] if fallback else None


def iter_reports(root: Path) -> list[Path]:
    # Support both flat roots and grouped roots by finding report dirs that
    # contain at least one .mmd file anywhere under `root`.
    return sorted({mmd.parent for mmd in root.rglob("*.mmd")})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "sample_data"
        / "subset_auto_parts_2017_2022",
        help=(
            "Directory containing OCR reports, either directly as one subdir "
            "per report or nested under grouping subdirectories."
        ),
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=4,
        help="Number of leading pages to scan for 10-K markers (default: 4).",
    )
    parser.add_argument(
        "--min-strong-markers",
        type=int,
        default=1,
        help=(
            "Minimum distinct strong marker labels required to flag as 10-K "
            "(default: 1). Weak markers (bare 'Form 10-K' / 'Securities Exchange "
            "Act of 1934') are reported but never trigger a positive on their own."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parent / "classification.json",
    )
    parser.add_argument(
        "--out-list",
        type=Path,
        default=Path(__file__).resolve().parent / "is_10k.txt",
        help="Plain-text file listing report names flagged as 10-K, one per line.",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: root not found: {args.root}", file=sys.stderr)
        return 2

    reports = iter_reports(args.root)
    results: list[Result] = []
    missing: list[str] = []

    for report_dir in reports:
        mmd = find_mmd(report_dir)
        if mmd is None:
            missing.append(report_dir.name)
            continue
        text = mmd.read_text(encoding="utf-8", errors="replace")
        matched, first_page, scanned = classify(text, args.pages)
        strong_hits = [m for m in matched if m in STRONG_LABELS]
        is_10k = len(strong_hits) >= args.min_strong_markers
        results.append(
            Result(
                name=report_dir.name,
                mmd_path=str(mmd),
                is_10k=is_10k,
                matched_markers=matched,
                first_match_page=first_page,
                pages_scanned=scanned,
            )
        )

    flagged = [r for r in results if r.is_10k]

    payload = {
        "root": str(args.root),
        "pages_scanned": args.pages,
        "min_strong_markers": args.min_strong_markers,
        "strong_markers": sorted(STRONG_LABELS),
        "total_reports": len(results),
        "total_10k": len(flagged),
        "missing_mmd": missing,
        "results": [asdict(r) for r in results],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.out_list.write_text(
        "\n".join(r.name for r in flagged) + ("\n" if flagged else ""),
        encoding="utf-8",
    )

    print(
        f"scanned {len(results)} reports → {len(flagged)} classified as 10-K "
        f"(min_strong_markers={args.min_strong_markers}, pages={args.pages})"
    )
    if missing:
        print(f"warning: {len(missing)} subdir(s) had no .mmd file", file=sys.stderr)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
