#!/usr/bin/env python3
r"""Replace broken dollar markers in .mmd files using heuristic-based selection.

Heuristic A (pair-based):
- Adjacent marker pair "\\(" then "\\)" with no curly braces between them.

Heuristic B (money-context):
- Marker appears to precede an amount-like token or nearby money phrasing.
- Excludes obvious math-like markup such as "\\( _{2}" and "\\( ^{TM}".

Selection strategies:
- money: use only money-context markers (higher recall; default).
- overlap: use intersection of pair-based and money-context markers (higher precision).

Always-on exact rule:
- Replace exact table cell markers "<td>\(</td>" and "<td>\)</td>".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

MARKER_RE = re.compile(r"\\\(|\\\)")
MONEY_NUM_RE = re.compile(r"^\s*[\(\-]?\d(?:[\d,]*\.?\d*)")
MONEY_WORD_RE = re.compile(r"^.{0,24}\b(?:million|billion|thousand)\b", re.IGNORECASE)
MONEY_PHRASE_RE = re.compile(
    r"^.{0,30}\b(?:per\s+share|per\s+ton|per\s+gallon|per\s+bushel|market\s+value)\b",
    re.IGNORECASE,
)
MATHISH_RE = re.compile(r"^\s*[_\^]?\s*\{")
EXACT_TD_RE = re.compile(r"<td>(\\\(|\\\))</td>")


def iter_mmd_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.mmd"):
        if path.is_file():
            yield path


def get_markers(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(0)) for m in MARKER_RE.finditer(text)]


def select_user_markers(text: str, markers: list[tuple[int, str]]) -> set[int]:
    selected: set[int] = set()
    for i in range(len(markers) - 1):
        pos_a, tok_a = markers[i]
        pos_b, tok_b = markers[i + 1]
        if tok_a != r"\(" or tok_b != r"\)":
            continue
        between = text[pos_a + 2 : pos_b]
        if "{" in between or "}" in between:
            continue
        selected.add(pos_a)
        selected.add(pos_b)
    return selected


def select_money_context_markers(text: str, markers: list[tuple[int, str]]) -> set[int]:
    selected: set[int] = set()
    for pos, _tok in markers:
        after = text[pos + 2 : pos + 66]

        # Exclude obvious math-like constructions: \( _{...}, \( ^{...}, \({ ...
        if MATHISH_RE.match(after):
            continue

        is_money = bool(
            MONEY_NUM_RE.match(after)
            or MONEY_WORD_RE.match(after)
            or MONEY_PHRASE_RE.match(after)
        )
        if is_money:
            selected.add(pos)
    return selected


def select_exact_td_markers(text: str) -> set[int]:
    # Capture the marker token position inside exact HTML cells like <td>\(</td>.
    return {m.start(1) for m in EXACT_TD_RE.finditer(text)}


def apply_replacements(
    text: str, markers: list[tuple[int, str]], positions: set[int]
) -> tuple[str, int]:
    if not positions:
        return text, 0

    out: list[str] = []
    cursor = 0
    replaced = 0

    for pos, _tok in markers:
        if pos in positions:
            out.append(text[cursor:pos])
            out.append("$")
            cursor = pos + 2
            replaced += 1

    out.append(text[cursor:])
    return "".join(out), replaced


def process_file(path: Path, dry_run: bool, strategy: str) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    markers = get_markers(text)

    user_positions = select_user_markers(text, markers)
    money_positions = select_money_context_markers(text, markers)
    overlap = user_positions & money_positions
    td_exact_positions = select_exact_td_markers(text)

    if strategy == "money":
        selected_positions = money_positions | td_exact_positions
    elif strategy == "overlap":
        selected_positions = overlap | td_exact_positions
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    updated_text, replaced = apply_replacements(text, markers, selected_positions)

    changed = int(replaced > 0)
    if replaced > 0 and not dry_run:
        path.write_text(updated_text, encoding="utf-8")

    return {
        "markers": len(markers),
        "user": len(user_positions),
        "money": len(money_positions),
        "overlap": len(overlap),
        "td_exact": len(td_exact_positions),
        "replaced": replaced,
        "changed": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replace broken dollar markers in .mmd files using heuristic-based "
            "selection."
        )
    )
    parser.add_argument(
        "directory", type=Path, help="Root directory to scan recursively"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report changes without writing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file replacement counts",
    )
    parser.add_argument(
        "--strategy",
        choices=("money", "overlap"),
        default="money",
        help=(
            "Replacement selection strategy: 'money' (higher recall, default) "
            "or 'overlap' (higher precision)."
        ),
    )
    args = parser.parse_args()

    root = args.directory
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found or not a directory: {root}")

    totals = {
        "files": 0,
        "markers": 0,
        "user": 0,
        "money": 0,
        "overlap": 0,
        "td_exact": 0,
        "replaced": 0,
        "changed": 0,
    }

    for path in iter_mmd_files(root):
        stats = process_file(path, dry_run=args.dry_run, strategy=args.strategy)
        totals["files"] += 1
        totals["markers"] += stats["markers"]
        totals["user"] += stats["user"]
        totals["money"] += stats["money"]
        totals["overlap"] += stats["overlap"]
        totals["td_exact"] += stats["td_exact"]
        totals["replaced"] += stats["replaced"]
        totals["changed"] += stats["changed"]

        if args.verbose and stats["replaced"] > 0:
            print(f"{path}: replacements={stats['replaced']}")

    mode = "DRY RUN" if args.dry_run else "APPLY"
    print(f"MODE={mode}")
    print(f"STRATEGY={args.strategy}")
    print(f"FILES_SCANNED={totals['files']}")
    print(f"TOTAL_MARKER_TOKENS={totals['markers']}")
    print(f"USER_HEURISTIC_TOTAL={totals['user']}")
    print(f"MONEY_HEURISTIC_TOTAL={totals['money']}")
    print(f"OVERLAP_TOTAL={totals['overlap']}")
    print(f"EXACT_TD_TOTAL={totals['td_exact']}")
    print(f"REPLACEMENTS={totals['replaced']}")
    print(f"FILES_CHANGED={totals['changed']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
