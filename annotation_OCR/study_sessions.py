"""Build balanced table-study session bundles from a base table manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MANIFEST = HERE / "manifests" / "tables_5000.json"
DEFAULT_OUTPUT_DIR = HERE / "manifests"

DEFAULT_TOTAL_TABLES = 1200
DEFAULT_MIN_SESSION_ITEMS = 100
DEFAULT_MAX_SESSION_ITEMS = 140
DEFAULT_REQUIRED_VOTES = 3
DEFAULT_OVERLAP_BY_ANNOTATORS = {
    13: 250,
    14: 300,
    15: 300,
    16: 300,
    17: 300,
}


def load_manifest_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"invalid manifest items in {path}")
    return items


def balanced_counts(total: int, buckets: int) -> list[int]:
    base, remainder = divmod(total, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def pick_study_tables(
    items: list[dict[str, Any]], *, total_tables: int, seed: int
) -> list[dict[str, Any]]:
    if total_tables > len(items):
        raise ValueError(
            f"requested {total_tables} tables from manifest with only {len(items)} items"
        )
    rng = random.Random(seed)
    selected = rng.sample(items, total_tables)
    rng.shuffle(selected)
    return selected


def choose_overlap_sessions(
    *, overlap_items: list[dict[str, Any]], overlap_counts: list[int], seed: int
) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    remaining = overlap_counts[:]
    assignments: list[list[dict[str, Any]]] = [[] for _ in overlap_counts]

    for item in overlap_items:
        eligible = [index for index, count in enumerate(remaining) if count > 0]
        if len(eligible) < DEFAULT_REQUIRED_VOTES:
            raise ValueError(
                "not enough session capacity left for agreement assignment"
            )
        rng.shuffle(eligible)
        eligible.sort(key=lambda index: remaining[index], reverse=True)
        chosen = eligible[:DEFAULT_REQUIRED_VOTES]
        for session_index in chosen:
            assignments[session_index].append(item)
            remaining[session_index] -= 1

    if any(value != 0 for value in remaining):
        raise ValueError("failed to exhaust overlap assignment capacities")

    return assignments


def build_session_items(
    *,
    selected_items: list[dict[str, Any]],
    annotator_count: int,
    overlap_tables: int,
    seed: int,
    min_session_items: int,
    max_session_items: int,
) -> dict[str, Any]:
    total_tables = len(selected_items)
    if overlap_tables > total_tables:
        raise ValueError("overlap table count cannot exceed selected tables")

    total_annotations = total_tables + (DEFAULT_REQUIRED_VOTES - 1) * overlap_tables
    session_sizes = balanced_counts(total_annotations, annotator_count)
    if any(
        size < min_session_items or size > max_session_items for size in session_sizes
    ):
        raise ValueError(
            f"cannot distribute {total_annotations} annotations across {annotator_count} sessions "
            f"inside [{min_session_items}, {max_session_items}]"
        )

    overlap_items = selected_items[:overlap_tables]
    unique_items = selected_items[overlap_tables:]
    overlap_counts = balanced_counts(
        overlap_tables * DEFAULT_REQUIRED_VOTES, annotator_count
    )
    overlap_assignments = choose_overlap_sessions(
        overlap_items=overlap_items,
        overlap_counts=overlap_counts,
        seed=seed + annotator_count,
    )

    unique_counts = [
        session_sizes[index] - len(overlap_assignments[index])
        for index in range(annotator_count)
    ]
    if sum(unique_counts) != len(unique_items):
        raise ValueError("unique assignment counts do not match remaining tables")

    rng = random.Random(seed + 1000 + annotator_count)
    unique_pool = list(unique_items)
    rng.shuffle(unique_pool)

    sessions: list[dict[str, Any]] = []
    cursor = 0
    for session_index in range(annotator_count):
        agreement_records = [
            {
                **dict(item),
                "study_assignment": "agreement",
                "study_expected_votes": DEFAULT_REQUIRED_VOTES,
                "study_session_slot": session_index + 1,
            }
            for item in overlap_assignments[session_index]
        ]
        unique_records = []
        for _ in range(unique_counts[session_index]):
            item = unique_pool[cursor]
            cursor += 1
            unique_records.append(
                {
                    **dict(item),
                    "study_assignment": "single",
                    "study_expected_votes": 1,
                    "study_session_slot": session_index + 1,
                }
            )

        manifest_items = agreement_records + unique_records
        rng.shuffle(manifest_items)
        sessions.append(
            {
                "slot": session_index + 1,
                "target_items": len(manifest_items),
                "agreement_items": len(agreement_records),
                "single_items": len(unique_records),
                "items": manifest_items,
            }
        )

    return {
        "annotator_count": annotator_count,
        "session_item_counts": [session["target_items"] for session in sessions],
        "overlap_tables": overlap_tables,
        "unique_tables": total_tables,
        "total_annotations": total_annotations,
        "sessions": sessions,
    }


def build_study_bundle(
    *,
    source_manifest_path: Path,
    annotator_count: int,
    overlap_tables: int,
    total_tables: int,
    seed: int,
    min_session_items: int,
    max_session_items: int,
) -> dict[str, Any]:
    items = load_manifest_items(source_manifest_path)
    selected = pick_study_tables(items, total_tables=total_tables, seed=seed)
    session_payload = build_session_items(
        selected_items=selected,
        annotator_count=annotator_count,
        overlap_tables=overlap_tables,
        seed=seed,
        min_session_items=min_session_items,
        max_session_items=max_session_items,
    )
    return {
        "bundle_type": "ocr_table_study_bundle",
        "source_manifest_path": str(source_manifest_path),
        "seed": seed,
        "annotator_count": annotator_count,
        "required_votes": DEFAULT_REQUIRED_VOTES,
        "min_session_items": min_session_items,
        "max_session_items": max_session_items,
        "summary": {
            "annotator_count": annotator_count,
            "unique_tables": session_payload["unique_tables"],
            "agreement_tables": session_payload["overlap_tables"],
            "total_annotations": session_payload["total_annotations"],
            "session_item_counts": session_payload["session_item_counts"],
        },
        "sessions": session_payload["sessions"],
    }


def write_bundle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build OCR table-study session bundles."
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-tables", type=int, default=DEFAULT_TOTAL_TABLES)
    parser.add_argument(
        "--min-session-items", type=int, default=DEFAULT_MIN_SESSION_ITEMS
    )
    parser.add_argument(
        "--max-session-items", type=int, default=DEFAULT_MAX_SESSION_ITEMS
    )
    parser.add_argument(
        "--annotators",
        type=int,
        nargs="+",
        default=sorted(DEFAULT_OVERLAP_BY_ANNOTATORS),
        help="Annotator counts to build bundles for, e.g. --annotators 14 15 16",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    for annotator_count in args.annotators:
        if annotator_count not in DEFAULT_OVERLAP_BY_ANNOTATORS:
            raise ValueError(
                f"no default overlap setting for annotator count {annotator_count}"
            )
        overlap_tables = DEFAULT_OVERLAP_BY_ANNOTATORS[annotator_count]
        bundle = build_study_bundle(
            source_manifest_path=args.source_manifest,
            annotator_count=annotator_count,
            overlap_tables=overlap_tables,
            total_tables=args.total_tables,
            seed=args.seed,
            min_session_items=args.min_session_items,
            max_session_items=args.max_session_items,
        )
        output_path = args.output_dir / f"study_sessions_{annotator_count}.json"
        write_bundle(output_path, bundle)
        print(
            json.dumps(
                {
                    "annotator_count": annotator_count,
                    "overlap_tables": overlap_tables,
                    "output": str(output_path),
                    **bundle["summary"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
