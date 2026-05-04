"""Extract CEO / Shareholder letters from non-10-K OCR reports.

For each report under --root that is NOT listed in --is-10k (the plain-text
output of the 10-K classifier), read the OCR `.mmd` file, split it by
``<--- Page Split --->``, and scan for a section heading that matches any
expression in ``expressions.txt``. When a heading is found, capture the text
from the heading position through the end of a --window (default 4) page
span starting at that page.

The root may be either flat (`<root>/<EX_TICKER_YEAR>/...`) or grouped
(`<root>/<group>/<EX_TICKER_YEAR>/...`), e.g. DeepSeekOCR_Ardian_pruned_1k.

Outputs:
    extractions.json               machine-readable records
    extractions/{NAME}__NN_{slug}.md  one markdown file per extracted letter
"""

from __future__ import annotations  # Python 3.13 used here. 3.14 may break things idk.

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

PAGE_SPLIT = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)

# Unify smart quotes → straight so both expressions.txt and OCR text line up.
# Single-char → single-char so byte offsets are preserved.
# We use str.translate here, with the map below.
SMART_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "´": "'",
    }
)

# A line that, after the matched title, contains only padding + a short number
# (e.g. "Chairman's statement .... 2"). Capped at 3 digits so a real heading
# ending in a 4-digit year ("Chairman's Statement 2017") is not filtered.
TOC_TAIL = re.compile(r"[\s.\-:·…]*\d{1,3}\s*$")


def normalize(s: str) -> str:
    return s.translate(SMART_MAP)


def load_expressions(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        phrase = line.split("\t", 1)[1] if "\t" in line else line
        phrase = normalize(phrase).strip().rstrip(",:;.")  # sentence cleaning
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    # Longest first so regex alternation prefers the most specific phrase
    # (e.g. "Dear Fellow Shareholders" beats "Dear Shareholders").
    out.sort(key=lambda s: (-len(s), s.lower()))
    return out


def build_pattern(expressions: list[str]) -> re.Pattern[str]:
    placeholder_token = re.escape("<company_name>")
    alternatives: list[str] = []
    for phrase in expressions:
        tokens = [t for t in re.split(r"\s+", phrase) if t]
        parts: list[str] = []
        for tok in tokens:
            esc = re.escape(tok)
            if esc == placeholder_token:
                parts.append(r"[\w&'.\-]+(?:\s+[\w&'.\-]+){0,8}")
            else:
                parts.append(esc)
        alternatives.append(r"\s+".join(parts))
    joined = "|".join(f"(?:{alt})" for alt in alternatives)
    prefix = (
        r"(?:^|\n)"
        r"[ \t]*"
        r"(?:#{1,6}[ \t]*)?"
        r"(?:\*{1,3}[ \t]*)?"
        r"(?:_{1,3}[ \t]*)?"
    )
    # Require the title to end on a word boundary (not inside a larger word).
    suffix = r"(?=\W|$)"
    return re.compile(prefix + "(" + joined + ")" + suffix, re.IGNORECASE)


def is_toc_line(page: str, match_end: int) -> bool:
    """Heading match is probably a table-of-contents entry."""
    newline = page.find("\n", match_end)
    if newline == -1:
        newline = len(page)
    rest = page[match_end:newline]
    return bool(TOC_TAIL.fullmatch(rest))


@dataclass
class Extraction:
    title: str
    start_page: int
    end_page: int
    text: str


@dataclass
class ReportResult:
    name: str
    mmd_path: str
    total_pages: int
    num_extractions: int
    extractions: list[Extraction] = field(default_factory=list)


# mmd files are given by the DeepSeek OCR pipeline.
# this will not work on the chandra pipeline (also because it does not have a <Splitpage> logic)
def find_mmd(report_dir: Path) -> Path | None:
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


def display_source_path(path: Path) -> str:
    """Prefer repo-relative source paths in markdown headers when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def extract_letters(
    text: str, pattern: re.Pattern[str], window: int
) -> tuple[list[Extraction], int]:
    pages = PAGE_SPLIT.split(text)
    total = len(pages)
    candidates: list[tuple[int, int, str]] = []
    for i, page in enumerate(pages):
        for m in pattern.finditer(page):
            if is_toc_line(page, m.end(1)):
                continue
            candidates.append((i, m.start(1), m.group(1).strip()))
    candidates.sort(key=lambda t: (t[0], t[1]))

    out: list[Extraction] = []
    i = 0
    while i < len(candidates):
        page_idx, off, title = candidates[i]
        end_page = min(page_idx + window - 1, total - 1)
        chunks = [pages[page_idx][off:]]
        for j in range(page_idx + 1, end_page + 1):
            chunks.append(pages[j])
        body = "\n\n<--- Page Split --->\n\n".join(c.strip() for c in chunks)
        out.append(
            Extraction(
                title=title,
                start_page=page_idx,
                end_page=end_page,
                text=body,
            )
        )
        i += 1
        # Skip candidates that fall inside the window we just emitted.
        while i < len(candidates) and candidates[i][0] <= end_page:
            i += 1
    return out, total


def slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s[:maxlen] or "letter"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "sample_data" / "subset_auto_parts_2017_2022",
        help=(
            "Directory containing OCR reports, either directly as one subdir "
            "per report or nested under grouping subdirectories."
        ),
    )
    parser.add_argument("--expressions", type=Path, default=HERE / "expressions.txt")
    parser.add_argument(
        "--is-10k",
        type=Path,
        default=REPO_ROOT / "doc_text_processing" / "10K_or_not" / "is_10k.txt",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help="Maximum page span of an extracted letter (default: 4).",
    )
    parser.add_argument("--out-json", type=Path, default=HERE / "extractions.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "extractions")
    args = parser.parse_args(argv)
    args.root = args.root.expanduser().resolve()

    if not args.root.is_dir():
        print(f"error: root not found: {args.root}", file=sys.stderr)
        return 2

    expressions = load_expressions(args.expressions)
    if not expressions:
        print(f"error: no expressions loaded from {args.expressions}", file=sys.stderr)
        return 2
    pattern = build_pattern(expressions)

    exclude: set[str] = set()
    if args.is_10k.is_file():
        exclude = {
            line.strip()
            for line in args.is_10k.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale files from a previous run so the output matches the JSON.
    for p in args.out_dir.glob("*.md"):
        p.unlink()

    results: list[ReportResult] = []
    no_match: list[str] = []
    missing: list[str] = []

    for sub in iter_reports(args.root):
        if sub.name in exclude:
            continue
        mmd = find_mmd(sub)
        if mmd is None:
            missing.append(sub.name)
            continue
        text = normalize(mmd.read_text(encoding="utf-8", errors="replace"))
        letters, total_pages = extract_letters(text, pattern, args.window)
        rr = ReportResult(
            name=sub.name,
            mmd_path=str(mmd),
            total_pages=total_pages,
            num_extractions=len(letters),
            extractions=letters,
        )
        results.append(rr)
        if not letters:
            no_match.append(sub.name)
        for idx, ext in enumerate(letters, start=1):
            slug = slugify(ext.title)
            fn = args.out_dir / f"{sub.name}__{idx:02d}_{slug}.md"
            source = display_source_path(mmd)
            header = (
                f"# {ext.title}\n\n"
                f"_Source: `{source}` · "
                f"pages {ext.start_page}-{ext.end_page}_\n\n---\n\n"
            )
            fn.write_text(header + ext.text.strip() + "\n", encoding="utf-8")

    total_ext = sum(r.num_extractions for r in results)
    payload = {
        "root": str(args.root),
        "expressions": expressions,
        "num_expressions": len(expressions),
        "window": args.window,
        "excluded_10k": len(exclude),
        "total_non_10k_scanned": len(results),
        "reports_with_match": sum(1 for r in results if r.extractions),
        "total_extractions": total_ext,
        "reports_without_match": no_match,
        "missing_mmd": missing,
        "results": [asdict(r) for r in results],
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"scanned {len(results)} non-10K reports → "
        f"{payload['reports_with_match']} matched "
        f"({total_ext} total extractions, window={args.window} pages)"
    )
    if no_match:
        print(f"no match: {len(no_match)} report(s)", file=sys.stderr)
    if missing:
        print(f"missing mmd: {len(missing)}", file=sys.stderr)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_dir}/ ({total_ext} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
