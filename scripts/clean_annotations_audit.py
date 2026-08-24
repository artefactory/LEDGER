"""Clean ``annotations_audit.csv`` produced by ``llm_annotate_qrels.py``.

The audit file is written in append mode and may contain rows where the LLM
call failed (API connection errors, JSON parse errors, schema validation
errors, ...).  In those rows ``llm_grade`` is empty and ``llm_reasoning``
holds the error string.  Resume runs can also append a duplicate header row.

This script:

1. Drops every row whose ``llm_grade`` is not an integer in ``{0, 1, 2}``
   (empty grade, error strings, duplicate header, out-of-range values).
2. Normalises the grade notation to a consistent **unquoted integer**
   (``0`` / ``1`` / ``2``), regardless of whether the source file stored it
   as ``0`` or ``"0"``.  All other columns are preserved verbatim.
3. Prints a removal breakdown to stderr.

The file is streamed (csv module, no pandas) so it handles the ~300 MB /
600 k-row audit files comfortably.

Usage::

    uv run python scripts/clean_annotations_audit.py \
        KPI_analysis/output/qrels_full/annotations_audit.csv

    # overwrite the input file
    uv run python scripts/clean_annotations_audit.py path/to/annotations_audit.csv --in-place

    # custom output path
    uv run python scripts/clean_annotations_audit.py input.csv -o output.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

VALID_GRADES = {0, 1, 2}


def parse_grade(raw: str) -> int | None:
    """Return the grade as an int in {0, 1, 2}, or None if not a valid grade.

    Tolerates surrounding whitespace and stray quote characters so that both
    ``0`` and ``"0"`` (or ``'0'``) normalise to the integer ``0``.
    """
    s = (raw or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    if not s:
        return None
    try:
        g = int(s)
    except ValueError:
        return None
    return g if g in VALID_GRADES else None


def clean(in_path: Path, out_path: Path) -> tuple[int, Counter, Counter]:
    """Stream ``in_path`` -> ``out_path``. Returns (n_kept, kept_by_grade, dropped_by_reason)."""
    dropped_by_reason: Counter = Counter()
    kept_by_grade: Counter = Counter()
    n_kept = 0

    with in_path.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if fieldnames is None or "llm_grade" not in fieldnames:
            sys.stderr.write(f"[error] no 'llm_grade' column in {in_path}\n")
            sys.exit(1)

        # Stream into the final file directly (no temp rename) — the output
        # path is already distinct from the input unless --in-place, in which
        # case we stage through a temp file and atomically replace.
        if in_path.resolve() == out_path.resolve():
            tmp_dir = out_path.parent
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=tmp_dir,
                prefix=f".{out_path.stem}_clean_",
                suffix=out_path.suffix,
                delete=False,
            )
            tmp_path = Path(tmp.name)
            out_file = tmp
            needs_replace = True
        else:
            out_file = out_path.open("w", newline="", encoding="utf-8")
            tmp_path = out_path
            needs_replace = False

        try:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in tqdm(reader, desc="cleaning", unit="rows"):
                grade = parse_grade(row["llm_grade"])
                if grade is None:
                    raw = (row["llm_grade"] or "").strip()
                    if raw == "":
                        reason = "empty_grade"
                    elif raw.lower() == "llm_grade":
                        reason = "duplicate_header"
                    else:
                        reason = f"non_numeric:{raw[:40]}"
                    dropped_by_reason[reason] += 1
                    continue
                row["llm_grade"] = grade
                writer.writerow(row)
                kept_by_grade[grade] += 1
                n_kept += 1
        finally:
            out_file.close()

        if needs_replace:
            tmp_path.replace(out_path)

    return n_kept, kept_by_grade, dropped_by_reason


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        type=Path,
        help="Path to annotations_audit.csv (output of llm_annotate_qrels.py).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. If omitted, writes next to the input with a "
            "'_clean' suffix. Ignored when --in-place is set."
        ),
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file (atomic temp-file rename).",
    )
    args = p.parse_args()

    in_path: Path = args.input
    if not in_path.is_file():
        sys.stderr.write(f"[error] input not found: {in_path}\n")
        sys.exit(1)

    if args.in_place:
        out_path = in_path
    elif args.output is not None:
        out_path = args.output
    else:
        out_path = in_path.with_name(in_path.stem + "_clean" + in_path.suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stderr.write(f"[clean] {in_path} -> {out_path}\n")
    n_kept, kept_by_grade, dropped_by_reason = clean(in_path, out_path)

    n_dropped = sum(dropped_by_reason.values())
    total = n_kept + n_dropped
    sys.stderr.write("\n[done] summary\n")
    sys.stderr.write(f"  input rows:  {total}\n")
    sys.stderr.write(f"  kept:        {n_kept}\n")
    for g in sorted(kept_by_grade):
        sys.stderr.write(f"    grade {g}:   {kept_by_grade[g]}\n")
    sys.stderr.write(f"  dropped:     {n_dropped}\n")
    for reason, count in dropped_by_reason.most_common():
        sys.stderr.write(f"    {reason}: {count}\n")
    sys.stderr.write(f"  output:      {out_path}\n")


if __name__ == "__main__":
    main()
