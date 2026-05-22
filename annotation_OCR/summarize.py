"""Regenerate OCR annotation session summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from store import list_sessions, write_all_sessions_summary, write_summary_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate OCR annotation summaries.")
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate summaries for every session under annotation_OCR/sessions.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="Optional path for the combined all-sessions CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    session_ids = list(args.session_id)
    if args.all:
        session_ids.extend(metadata["session_id"] for metadata in list_sessions())

    seen = set()
    regenerated = []
    for session_id in session_ids:
        if session_id in seen:
            continue
        seen.add(session_id)
        regenerated.append(
            {"session_id": session_id, **write_summary_files(session_id)}
        )

    combined = None
    if args.all or args.combined_output:
        combined = str(write_all_sessions_summary(args.combined_output))

    print(
        json.dumps(
            {"regenerated": regenerated, "combined_summary_csv": combined}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
