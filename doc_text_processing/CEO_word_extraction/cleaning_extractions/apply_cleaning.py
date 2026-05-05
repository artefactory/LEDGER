"""Apply the letter-end decisions from ``detect_letter_end.py``.

Reads ``end_markers.json`` and, for each ``status=="ok"`` decision, copies
the source ``.md`` file into ``cleaned/`` truncated at the detected
boundary:

  - if ``next_offset`` is set, keep ``text[: next_offset]`` (drops the
    trailing irrelevant content while preserving any blank lines / page
    splits between the signature and the cut point);
  - else if only ``end_offset`` is set, keep ``text[: end_offset]``;
  - failed decisions are skipped (and listed at the end).

Trailing whitespace, dangling page-split markers and trailing horizontal
rules are stripped from the kept slice for tidiness.

Use ``--dry-run`` to preview what would be cut.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE.parent / "extractions"
DEFAULT_MARKERS = HERE / "end_markers.json"
DEFAULT_OUTPUT_DIR = HERE / "cleaned"

PAGE_SPLIT_RE = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)
TRAILING_NOISE_RE = re.compile(
    r"(?:\s*(?:<---\s*Page Split\s*--->|---+|\*{3,}|_{3,}))*\s*\Z",
    re.IGNORECASE,
)


def tidy(slice_: str) -> str:
    """Strip trailing page-splits / horizontal rules / blank lines."""
    return TRAILING_NOISE_RE.sub("", slice_).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument(
        "--cut-at",
        choices=("next", "end"),
        default="next",
        help=(
            "where to truncate when both anchors are present: 'next' keeps "
            "everything before the next_quote (default — preserves trailing "
            "whitespace inside the signature block); 'end' truncates exactly "
            "at the end of end_quote"
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.markers.exists():
        sys.exit(f"markers file not found: {args.markers}")
    payload = json.loads(args.markers.read_text())
    decisions = payload.get("decisions", [])

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skipped = n_failed = 0
    saved_chars = 0
    failed: list[tuple[str, str]] = []

    for rec in decisions:
        name = rec["name"]
        src = args.input_dir / f"{name}.md"
        if not src.exists():
            print(f"SKIP {name}: source not found ({src})", file=sys.stderr)
            n_skipped += 1
            continue
        if rec.get("status") != "ok":
            failed.append((name, rec.get("error") or "unknown"))
            n_failed += 1
            continue

        text = src.read_text(encoding="utf-8")
        end_off = rec.get("end_offset")
        next_off = rec.get("next_offset")

        if args.cut_at == "next" and next_off is not None:
            cut = next_off
        elif end_off is not None:
            cut = end_off
        else:
            failed.append((name, "no offsets in decision"))
            n_failed += 1
            continue

        kept = tidy(text[:cut])
        dropped = len(text) - len(kept)
        saved_chars += max(dropped, 0)

        if args.dry_run:
            print(
                f"DRY {name}: keep {len(kept)} / {len(text)} chars "
                f"(drop {dropped})"
            )
        else:
            (args.output_dir / f"{name}.md").write_text(kept, encoding="utf-8")
        n_ok += 1

    print(
        f"\nSummary: ok={n_ok} failed={n_failed} skipped={n_skipped} "
        f"chars_saved≈{saved_chars}",
        file=sys.stderr,
    )
    if failed:
        print(f"\nFailed ({len(failed)}):", file=sys.stderr)
        for name, reason in failed[:20]:
            print(f"  {name}: {reason}", file=sys.stderr)
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more", file=sys.stderr)


if __name__ == "__main__":
    main()
