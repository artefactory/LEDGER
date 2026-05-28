"""Compute agreement and accept/reject ratios for bundle-backed table studies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_SESSIONS_DIR = HERE / "sessions"
DEFAULT_ANALYSIS_ROOT = DEFAULT_SESSIONS_DIR / "study_analysis"
REVIEWED_STATUSES = ("ok", "not_ok", "uncertain")
VALID_STATUSES = set(REVIEWED_STATUSES).union({"unreviewed"})


@dataclass(slots=True)
class SessionPayload:
    session_id: str
    session_name: str
    annotator: str
    slot: int
    item_count: int
    completed_count: int
    updated_at_utc: str
    metadata: dict[str, Any]
    manifest_items: list[dict[str, Any]]
    current_annotations: dict[str, dict[str, Any]]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute agreement metrics and accept/reject ratios for OCR table study sessions."
    )
    parser.add_argument(
        "--study-bundle",
        type=Path,
        required=True,
        help="Path to the study_sessions_*.json bundle used for the annotation round.",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=DEFAULT_SESSIONS_DIR,
        help="Directory containing annotation_OCR session folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for summary artifacts. Defaults to sessions/study_analysis/<bundle-stem>/.",
    )
    parser.add_argument(
        "--session-id",
        dest="session_ids",
        nargs="+",
        default=None,
        help="Optional explicit session ids to analyze. If omitted, all sessions linked to the study bundle are used.",
    )
    parser.add_argument(
        "--strict-manifest",
        action="store_true",
        help="Fail if a selected session manifest does not match its bundle slot.",
    )
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path, *, default: Any | None = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def parse_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def normalize_status(value: Any) -> str:
    if isinstance(value, str) and value in VALID_STATUSES:
        return value
    return "unreviewed"


def load_study_bundle(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if payload.get("bundle_type") != "ocr_table_study_bundle" or not isinstance(
        sessions, list
    ):
        raise ValueError(f"invalid study bundle in {path}")
    return payload


def build_bundle_index(
    bundle: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    slot_index: dict[int, dict[str, Any]] = {}
    item_index: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for session in bundle["sessions"]:
        slot = parse_int(session.get("slot"))
        items = session.get("items")
        if slot is None or not isinstance(items, list):
            raise ValueError("invalid study session entry in bundle")
        slot_index[slot] = session
        for item in items:
            item_id = str(item.get("item_id") or "")
            if not item_id:
                raise ValueError(f"bundle slot {slot} contains an item without item_id")
            expected_votes = parse_int(item.get("study_expected_votes")) or 1
            study_assignment = str(item.get("study_assignment") or "single")
            record = item_index.setdefault(
                item_id,
                {
                    "item_id": item_id,
                    "industry_slug": item.get("industry_slug"),
                    "report_name": item.get("report_name"),
                    "exchange": item.get("exchange"),
                    "ticker": item.get("ticker"),
                    "year": item.get("year"),
                    "page_index": item.get("page_index"),
                    "page_number": item.get("page_number"),
                    "table_index": item.get("table_index"),
                    "table_row_count": item.get("table_row_count"),
                    "table_col_count": item.get("table_col_count"),
                    "study_assignment": study_assignment,
                    "expected_votes": expected_votes,
                    "assigned_slots": [],
                },
            )
            record["assigned_slots"].append(slot)
            record["expected_votes"] = max(record["expected_votes"], expected_votes)
            if study_assignment == "agreement":
                record["study_assignment"] = "agreement"

    for item_id, record in item_index.items():
        assigned_slots = sorted(record["assigned_slots"])
        record["assigned_slots"] = assigned_slots
        occurrence_count = len(assigned_slots)
        if occurrence_count > 1:
            record["study_assignment"] = "agreement"
        if occurrence_count != record["expected_votes"]:
            warnings.append(
                f"bundle item {item_id} appears in {occurrence_count} slots but declares study_expected_votes={record['expected_votes']}"
            )

    return slot_index, item_index, warnings


def load_session_payload(sessions_dir: Path, session_id: str) -> SessionPayload:
    directory = sessions_dir / session_id
    metadata = load_json(directory / "metadata.json")
    if not isinstance(metadata, dict):
        raise FileNotFoundError(f"missing metadata for session {session_id}")
    manifest = load_json(directory / "manifest.json", default={}) or {}
    manifest_items = manifest.get("items") if isinstance(manifest, dict) else None
    if not isinstance(manifest_items, list):
        raise ValueError(f"invalid manifest for session {session_id}")
    current_annotations = (
        load_json(directory / "current_annotations.json", default={}) or {}
    )
    if not isinstance(current_annotations, dict):
        raise ValueError(f"invalid current_annotations for session {session_id}")

    config = metadata.get("config") or {}
    slot = parse_int(config.get("study_slot"))
    if slot is None:
        raise ValueError(f"session {session_id} has no usable study_slot")

    return SessionPayload(
        session_id=session_id,
        session_name=str(metadata.get("session_name") or session_id),
        annotator=str(metadata.get("annotator") or ""),
        slot=slot,
        item_count=parse_int(metadata.get("item_count")) or len(manifest_items),
        completed_count=parse_int(metadata.get("completed_count")) or 0,
        updated_at_utc=str(metadata.get("updated_at_utc") or ""),
        metadata=metadata,
        manifest_items=manifest_items,
        current_annotations=current_annotations,
    )


def discover_sessions(
    *,
    sessions_dir: Path,
    bundle_path: Path,
    session_ids: list[str] | None,
) -> tuple[dict[int, SessionPayload], list[str]]:
    bundle_resolved = str(bundle_path.resolve())
    warnings: list[str] = []
    discovered: dict[int, SessionPayload] = {}

    if session_ids is None:
        candidate_ids = [
            path.name for path in sorted(sessions_dir.iterdir()) if path.is_dir()
        ]
    else:
        candidate_ids = session_ids

    for session_id in candidate_ids:
        metadata = load_json(sessions_dir / session_id / "metadata.json")
        if not isinstance(metadata, dict):
            if session_ids is None:
                continue
            raise FileNotFoundError(f"missing metadata for session {session_id}")

        config = metadata.get("config") or {}
        session_bundle = config.get("study_bundle_path")
        session_slot = parse_int(config.get("study_slot"))
        if session_ids is None:
            if session_bundle != bundle_resolved or session_slot is None:
                continue

        payload = load_session_payload(sessions_dir, session_id)
        if session_ids is not None and session_bundle not in {None, bundle_resolved}:
            warnings.append(
                f"session {session_id} references a different study bundle: {session_bundle}"
            )

        existing = discovered.get(payload.slot)
        if existing is None:
            discovered[payload.slot] = payload
            continue

        keep = payload
        drop = existing
        if (existing.updated_at_utc, existing.session_id) > (
            payload.updated_at_utc,
            payload.session_id,
        ):
            keep = existing
            drop = payload
        discovered[payload.slot] = keep
        warnings.append(
            f"multiple sessions claim study slot {payload.slot}; keeping {keep.session_id} and ignoring {drop.session_id}"
        )

    return discovered, warnings


def validate_session_manifest(
    *,
    payload: SessionPayload,
    expected_session: dict[str, Any],
    strict: bool,
) -> list[str]:
    warnings: list[str] = []
    actual_ids = [str(item.get("item_id") or "") for item in payload.manifest_items]
    expected_ids = [
        str(item.get("item_id") or "") for item in expected_session["items"]
    ]
    if Counter(actual_ids) != Counter(expected_ids):
        message = f"session {payload.session_id} manifest does not match bundle slot {payload.slot}"
        if strict:
            raise ValueError(message)
        warnings.append(message)
    return warnings


def status_ratio_block(counts: Counter[str]) -> dict[str, Any]:
    reviewed = sum(counts.get(status, 0) for status in REVIEWED_STATUSES)
    decided = counts.get("ok", 0) + counts.get("not_ok", 0)
    return {
        "reviewed": reviewed,
        "decided": decided,
        "ok": counts.get("ok", 0),
        "not_ok": counts.get("not_ok", 0),
        "uncertain": counts.get("uncertain", 0),
        "ok_rate_all": safe_div(counts.get("ok", 0), reviewed),
        "not_ok_rate_all": safe_div(counts.get("not_ok", 0), reviewed),
        "uncertain_rate_all": safe_div(counts.get("uncertain", 0), reviewed),
        "accept_ratio_decided": safe_div(counts.get("ok", 0), decided),
        "reject_ratio_decided": safe_div(counts.get("not_ok", 0), decided),
    }


def majority_status(counts: Counter[str], vote_count: int) -> str | None:
    if vote_count == 0:
        return None
    top_count = max(counts.values(), default=0)
    if top_count * 2 <= vote_count:
        return None
    winners = [status for status, count in counts.items() if count == top_count]
    if len(winners) != 1:
        return None
    return winners[0]


def compute_pairwise_agreement(item_rows: list[dict[str, Any]]) -> dict[str, Any]:
    items_considered = 0
    matching_pairs = 0
    total_pairs = 0
    for row in item_rows:
        vote_count = int(row["vote_count"])
        if vote_count < 2:
            continue
        items_considered += 1
        total_pairs += comb(vote_count, 2)
        matching_pairs += sum(
            comb(int(row[f"{status}_votes"]), 2) for status in REVIEWED_STATUSES
        )
    return {
        "items_considered": items_considered,
        "pairs_total": total_pairs,
        "pairs_matching": matching_pairs,
        "agreement_rate": safe_div(matching_pairs, total_pairs),
    }


def compute_fleiss_kappa(item_rows: list[dict[str, Any]]) -> float | None:
    if not item_rows:
        return None
    n = int(item_rows[0]["expected_votes"])
    if n < 2:
        return None
    if any(int(row["expected_votes"]) != n for row in item_rows):
        return None

    total_items = len(item_rows)
    p_i_values: list[float] = []
    category_totals = Counter[str]()
    for row in item_rows:
        row_total = 0
        squared_sum = 0
        for status in REVIEWED_STATUSES:
            count = int(row[f"{status}_votes"])
            category_totals[status] += count
            row_total += count
            squared_sum += count * count
        if row_total != n:
            return None
        p_i_values.append((squared_sum - n) / (n * (n - 1)))

    p_bar = sum(p_i_values) / total_items
    p_e = 0.0
    for status in REVIEWED_STATUSES:
        p_j = category_totals[status] / (total_items * n)
        p_e += p_j * p_j
    if p_e == 1.0:
        return None
    return (p_bar - p_e) / (1.0 - p_e)


def build_analysis(
    *,
    bundle_path: Path,
    sessions_dir: Path,
    session_ids: list[str] | None,
    strict_manifest: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = load_study_bundle(bundle_path)
    slot_index, item_index, warnings = build_bundle_index(bundle)
    selected_sessions, session_warnings = discover_sessions(
        sessions_dir=sessions_dir,
        bundle_path=bundle_path,
        session_ids=session_ids,
    )
    warnings.extend(session_warnings)

    expected_slots = sorted(slot_index)
    missing_slots = [slot for slot in expected_slots if slot not in selected_sessions]

    session_rows: list[dict[str, Any]] = []
    for slot in expected_slots:
        payload = selected_sessions.get(slot)
        if payload is None:
            continue
        warnings.extend(
            validate_session_manifest(
                payload=payload,
                expected_session=slot_index[slot],
                strict=strict_manifest,
            )
        )
        slot_status_counts = Counter[str]()
        for item in slot_index[slot]["items"]:
            item_id = str(item["item_id"])
            annotation = payload.current_annotations.get(item_id) or {}
            slot_status_counts[normalize_status(annotation.get("overall_status"))] += 1
        reviewed_count = (
            len(slot_index[slot]["items"]) - slot_status_counts["unreviewed"]
        )
        if reviewed_count != payload.completed_count:
            warnings.append(
                f"session {payload.session_id} metadata says completed_count={payload.completed_count} but current_annotations implies {reviewed_count}"
            )
        status_block = status_ratio_block(slot_status_counts)
        session_rows.append(
            {
                "slot": slot,
                "session_id": payload.session_id,
                "session_name": payload.session_name,
                "annotator": payload.annotator,
                "item_count": len(slot_index[slot]["items"]),
                "reviewed_count": reviewed_count,
                "unreviewed_count": slot_status_counts["unreviewed"],
                "ok": slot_status_counts["ok"],
                "not_ok": slot_status_counts["not_ok"],
                "uncertain": slot_status_counts["uncertain"],
                "accept_ratio_decided": status_block["accept_ratio_decided"],
                "reject_ratio_decided": status_block["reject_ratio_decided"],
                "uncertain_rate_all": status_block["uncertain_rate_all"],
                "updated_at_utc": payload.updated_at_utc,
            }
        )

    vote_level_counts_all = Counter[str]()
    vote_level_counts_single = Counter[str]()
    vote_level_counts_agreement = Counter[str]()
    item_rows: list[dict[str, Any]] = []

    for item_id, record in sorted(item_index.items()):
        votes: list[dict[str, Any]] = []
        missing_session_slots_for_item: list[int] = []
        unreviewed_slots: list[int] = []
        available_slots: list[int] = []

        for slot in record["assigned_slots"]:
            payload = selected_sessions.get(slot)
            if payload is None:
                missing_session_slots_for_item.append(slot)
                continue
            available_slots.append(slot)
            annotation = payload.current_annotations.get(item_id) or {}
            status = normalize_status(annotation.get("overall_status"))
            if status == "unreviewed":
                unreviewed_slots.append(slot)
                continue
            vote = {
                "slot": slot,
                "session_id": payload.session_id,
                "annotator": payload.annotator,
                "overall_status": status,
                "updated_at_utc": annotation.get("updated_at_utc", ""),
            }
            votes.append(vote)
            vote_level_counts_all[status] += 1
            if record["study_assignment"] == "agreement":
                vote_level_counts_agreement[status] += 1
            else:
                vote_level_counts_single[status] += 1

        vote_counts = Counter(vote["overall_status"] for vote in votes)
        vote_count = len(votes)
        majority = majority_status(vote_counts, vote_count)
        is_complete = vote_count == int(record["expected_votes"])
        is_unanimous = is_complete and len(vote_counts) == 1
        final_status = None
        if record["study_assignment"] == "single" and vote_count == 1:
            final_status = votes[0]["overall_status"]
        elif record["study_assignment"] == "agreement" and is_complete and majority:
            final_status = majority

        item_rows.append(
            {
                "item_id": item_id,
                "study_assignment": record["study_assignment"],
                "expected_votes": int(record["expected_votes"]),
                "assigned_slots": json.dumps(record["assigned_slots"]),
                "available_slots": json.dumps(available_slots),
                "missing_session_slots": json.dumps(missing_session_slots_for_item),
                "unreviewed_slots": json.dumps(unreviewed_slots),
                "vote_count": vote_count,
                "ok_votes": vote_counts["ok"],
                "not_ok_votes": vote_counts["not_ok"],
                "uncertain_votes": vote_counts["uncertain"],
                "is_complete": is_complete,
                "is_unanimous": is_unanimous,
                "has_majority": majority is not None,
                "majority_status": majority or "",
                "final_status": final_status or "",
                "votes_json": json.dumps(votes, ensure_ascii=False),
                "industry_slug": record.get("industry_slug"),
                "report_name": record.get("report_name"),
                "exchange": record.get("exchange"),
                "ticker": record.get("ticker"),
                "year": record.get("year"),
                "page_index": record.get("page_index"),
                "page_number": record.get("page_number"),
                "table_index": record.get("table_index"),
                "table_row_count": record.get("table_row_count"),
                "table_col_count": record.get("table_col_count"),
            }
        )

    agreement_rows = [
        row for row in item_rows if row["study_assignment"] == "agreement"
    ]
    complete_agreement_rows = [row for row in agreement_rows if row["is_complete"]]
    agreement_rows_with_2plus_votes = [
        row for row in agreement_rows if int(row["vote_count"]) >= 2
    ]
    unanimous_rows = [row for row in complete_agreement_rows if row["is_unanimous"]]
    majority_rows = [
        row
        for row in complete_agreement_rows
        if row["has_majority"] and not row["is_unanimous"]
    ]
    no_majority_rows = [
        row for row in complete_agreement_rows if not row["has_majority"]
    ]

    final_status_counts = Counter(
        row["final_status"] for row in item_rows if row["final_status"]
    )
    agreement_final_counts = Counter(
        row["majority_status"]
        for row in complete_agreement_rows
        if row["majority_status"]
    )

    summary = {
        "analysis_completed_at_utc": utc_now(),
        "study_bundle_path": str(bundle_path.resolve()),
        "sessions_dir": str(sessions_dir.resolve()),
        "bundle": {
            "annotator_count": bundle.get("annotator_count"),
            "required_votes": bundle.get("required_votes"),
            "summary": bundle.get("summary") or {},
        },
        "session_coverage": {
            "expected_slots": expected_slots,
            "sessions_found": len(selected_sessions),
            "missing_slots": missing_slots,
        },
        "annotation_votes": {
            "all": status_ratio_block(vote_level_counts_all),
            "single": status_ratio_block(vote_level_counts_single),
            "agreement": status_ratio_block(vote_level_counts_agreement),
        },
        "agreement": {
            "tables_total": len(agreement_rows),
            "tables_with_any_vote": sum(
                1 for row in agreement_rows if row["vote_count"] > 0
            ),
            "tables_with_2plus_votes": len(agreement_rows_with_2plus_votes),
            "tables_complete": len(complete_agreement_rows),
            "unanimous_tables": len(unanimous_rows),
            "mixed_majority_tables": len(majority_rows),
            "no_majority_tables": len(no_majority_rows),
            "exact_agreement_rate": safe_div(
                len(unanimous_rows), len(complete_agreement_rows)
            ),
            "complete_pairwise": compute_pairwise_agreement(complete_agreement_rows),
            "partial_pairwise": compute_pairwise_agreement(
                agreement_rows_with_2plus_votes
            ),
            "fleiss_kappa": compute_fleiss_kappa(complete_agreement_rows),
            "majority_status_counts": dict(agreement_final_counts),
            "majority_status_ratios": status_ratio_block(agreement_final_counts),
        },
        "final_table_decisions": {
            "tables_with_final_status": sum(final_status_counts.values()),
            "status_counts": dict(final_status_counts),
            "status_ratios": status_ratio_block(final_status_counts),
        },
        "warnings": warnings,
    }
    return summary, session_rows, item_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def render_summary_markdown(
    *, summary: dict[str, Any], session_rows: list[dict[str, Any]]
) -> str:
    session_coverage = summary["session_coverage"]
    all_votes = summary["annotation_votes"]["all"]
    agreement = summary["agreement"]
    final_tables = summary["final_table_decisions"]
    warnings = summary.get("warnings") or []

    lines = [
        "# OCR Table Study Agreement Summary",
        "",
        f"- Generated: {summary['analysis_completed_at_utc']}",
        f"- Study bundle: {summary['study_bundle_path']}",
        f"- Sessions directory: {summary['sessions_dir']}",
        f"- Sessions found: {session_coverage['sessions_found']} / {len(session_coverage['expected_slots'])}",
        f"- Missing slots: {', '.join(str(slot) for slot in session_coverage['missing_slots']) or 'none'}",
        "",
        "## Vote-Level Ratios",
        "",
        f"- Reviewed votes: {all_votes['reviewed']}",
        f"- Accept rate among all reviewed votes: {format_ratio(all_votes['ok_rate_all'])}",
        f"- Reject rate among all reviewed votes: {format_ratio(all_votes['not_ok_rate_all'])}",
        f"- Uncertain rate among all reviewed votes: {format_ratio(all_votes['uncertain_rate_all'])}",
        f"- Accept ratio among decided votes: {format_ratio(all_votes['accept_ratio_decided'])}",
        f"- Reject ratio among decided votes: {format_ratio(all_votes['reject_ratio_decided'])}",
        "",
        "## Agreement Subset",
        "",
        f"- Agreement tables total: {agreement['tables_total']}",
        f"- Agreement tables with 2+ votes: {agreement['tables_with_2plus_votes']}",
        f"- Agreement tables complete: {agreement['tables_complete']}",
        f"- Exact agreement rate: {format_ratio(agreement['exact_agreement_rate'])}",
        f"- Complete pairwise agreement: {format_ratio(agreement['complete_pairwise']['agreement_rate'])}",
        f"- Partial pairwise agreement: {format_ratio(agreement['partial_pairwise']['agreement_rate'])}",
        f"- Fleiss' kappa: {agreement['fleiss_kappa']:.4f}"
        if agreement["fleiss_kappa"] is not None
        else "- Fleiss' kappa: n/a",
        f"- Unanimous tables: {agreement['unanimous_tables']}",
        f"- Mixed-majority tables: {agreement['mixed_majority_tables']}",
        f"- No-majority tables: {agreement['no_majority_tables']}",
        "",
        "## Final Table Decisions",
        "",
        f"- Tables with a final status: {final_tables['tables_with_final_status']}",
        f"- Accept rate at table level: {format_ratio(final_tables['status_ratios']['ok_rate_all'])}",
        f"- Reject rate at table level: {format_ratio(final_tables['status_ratios']['not_ok_rate_all'])}",
        f"- Uncertain rate at table level: {format_ratio(final_tables['status_ratios']['uncertain_rate_all'])}",
        f"- Accept ratio among decided tables: {format_ratio(final_tables['status_ratios']['accept_ratio_decided'])}",
        f"- Reject ratio among decided tables: {format_ratio(final_tables['status_ratios']['reject_ratio_decided'])}",
        "",
        "## Session Breakdown",
        "",
        "| Slot | Session ID | Annotator | Reviewed | OK | Not OK | Uncertain | Accept Ratio | Reject Ratio |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in sorted(session_rows, key=lambda item: item["slot"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["slot"]),
                    row["session_id"],
                    row["annotator"],
                    str(row["reviewed_count"]),
                    str(row["ok"]),
                    str(row["not_ok"]),
                    str(row["uncertain"]),
                    format_ratio(row["accept_ratio_decided"]),
                    format_ratio(row["reject_ratio_decided"]),
                ]
            )
            + " |"
        )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir or (DEFAULT_ANALYSIS_ROOT / args.study_bundle.stem)

    summary, session_rows, item_rows = build_analysis(
        bundle_path=args.study_bundle,
        sessions_dir=args.sessions_dir,
        session_ids=args.session_ids,
        strict_manifest=args.strict_manifest,
    )

    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_text(
        output_dir / "summary.md",
        render_summary_markdown(summary=summary, session_rows=session_rows),
    )
    write_csv(
        output_dir / "session_metrics.csv",
        session_rows,
        fieldnames=[
            "slot",
            "session_id",
            "session_name",
            "annotator",
            "item_count",
            "reviewed_count",
            "unreviewed_count",
            "ok",
            "not_ok",
            "uncertain",
            "accept_ratio_decided",
            "reject_ratio_decided",
            "uncertain_rate_all",
            "updated_at_utc",
        ],
    )
    write_csv(
        output_dir / "item_metrics.csv",
        item_rows,
        fieldnames=[
            "item_id",
            "study_assignment",
            "expected_votes",
            "assigned_slots",
            "available_slots",
            "missing_session_slots",
            "unreviewed_slots",
            "vote_count",
            "ok_votes",
            "not_ok_votes",
            "uncertain_votes",
            "is_complete",
            "is_unanimous",
            "has_majority",
            "majority_status",
            "final_status",
            "votes_json",
            "industry_slug",
            "report_name",
            "exchange",
            "ticker",
            "year",
            "page_index",
            "page_number",
            "table_index",
            "table_row_count",
            "table_col_count",
        ],
    )

    print(f"Wrote study analysis to {output_dir}")
    print(
        json.dumps(
            {
                "sessions_found": summary["session_coverage"]["sessions_found"],
                "agreement_tables_complete": summary["agreement"]["tables_complete"],
                "exact_agreement_rate": summary["agreement"]["exact_agreement_rate"],
                "final_table_accept_ratio": summary["final_table_decisions"][
                    "status_ratios"
                ]["accept_ratio_decided"],
                "final_table_reject_ratio": summary["final_table_decisions"][
                    "status_ratios"
                ]["reject_ratio_decided"],
                "warnings": len(summary.get("warnings") or []),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
