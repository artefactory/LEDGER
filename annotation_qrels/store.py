"""File-backed session storage for qrels annotation runs."""

from __future__ import annotations

import csv
import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SESSIONS_DIR = HERE / "sessions"
SCHEMA_VERSION = "1.0"

VALID_OVERALL_STATUS = {"ok", "not_ok", "uncertain", "unreviewed"}

SUMMARY_FIELDS = [
    "session_id",
    "session_name",
    "annotator",
    "item_id",
    "query_id",
    "doc_id",
    "report_name",
    "report_year",
    "page_idx",
    "exchange",
    "ticker",
    "year",
    "kpi",
    "target_value",
    "target_value_display",
    "match_type",
    "alias_matched",
    "raw_value",
    "normalized_value",
    "rel_error",
    "unit_source",
    "snippet",
    "overall_status",
    "notes",
    "updated_at_utc",
    "annotation_source",
    "review_duration_ms",
    "industry_slug",
    "page_text_chars",
    "page_text_sha256",
    "raw_png_path",
    "mmd_path",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def session_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return slug[:48] or "session"


def new_session_id(session_name: str | None = None) -> str:
    prefix = session_slug(session_name or "session")[:24]
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def metadata_path(session_id: str) -> Path:
    return session_dir(session_id) / "metadata.json"


def manifest_path(session_id: str) -> Path:
    return session_dir(session_id) / "manifest.json"


def current_annotations_path(session_id: str) -> Path:
    return session_dir(session_id) / "current_annotations.json"


def annotations_log_path(session_id: str) -> Path:
    return session_dir(session_id) / "annotations.jsonl"


def create_session(
    *,
    session_name: str,
    annotator: str,
    manifest_items: list[dict[str, Any]],
    index_summary: dict[str, Any],
    config: dict[str, Any],
    session_id: str | None = None,
) -> dict[str, Any]:
    sid = session_id or new_session_id(session_name)
    directory = session_dir(sid)
    if directory.exists():
        raise FileExistsError(f"session already exists: {sid}")
    directory.mkdir(parents=True, exist_ok=False)

    now = utc_now()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "session_id": sid,
        "session_name": session_name,
        "annotator": annotator,
        "created_at_utc": now,
        "updated_at_utc": now,
        "status": "active",
        "item_count": len(manifest_items),
        "completed_count": 0,
        "index_summary": index_summary,
        "config": config,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "session_id": sid,
        "created_at_utc": now,
        "item_count": len(manifest_items),
        "items": manifest_items,
    }

    atomic_write_json(metadata_path(sid), metadata)
    atomic_write_json(manifest_path(sid), manifest)
    atomic_write_json(current_annotations_path(sid), {})
    annotations_log_path(sid).touch()
    write_summary_files(sid)
    return metadata


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(session_id: str) -> dict[str, Any]:
    metadata = load_json(metadata_path(session_id))
    if metadata is None:
        raise FileNotFoundError(f"unknown session: {session_id}")
    return metadata


def load_manifest(session_id: str) -> list[dict[str, Any]]:
    manifest = load_json(manifest_path(session_id))
    if manifest is None:
        raise FileNotFoundError(f"unknown session manifest: {session_id}")
    return manifest.get("items", [])


def load_current_annotations(session_id: str) -> dict[str, dict[str, Any]]:
    return load_json(current_annotations_path(session_id), default={}) or {}


def list_sessions() -> list[dict[str, Any]]:
    if not SESSIONS_DIR.is_dir():
        return []
    sessions: list[dict[str, Any]] = []
    for path in sorted(SESSIONS_DIR.iterdir()):
        if not path.is_dir():
            continue
        metadata = load_json(path / "metadata.json")
        if isinstance(metadata, dict):
            sessions.append(metadata)
    sessions.sort(key=lambda rec: rec.get("updated_at_utc", ""), reverse=True)
    return sessions


def manifest_index(session_id: str) -> dict[str, dict[str, Any]]:
    return {item["item_id"]: item for item in load_manifest(session_id)}


def sanitize_status(value: Any, valid: set[str], default: str) -> str:
    if isinstance(value, str) and value in valid:
        return value
    return default


def normalize_annotation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_status": sanitize_status(
            payload.get("overall_status"), VALID_OVERALL_STATUS, "unreviewed"
        ),
        "notes": str(payload.get("notes") or "").strip(),
        "annotation_source": str(payload.get("annotation_source") or "manual"),
        "review_duration_ms": payload.get("review_duration_ms"),
        "client_started_at_utc": payload.get("client_started_at_utc"),
        "client_updated_at_utc": payload.get("client_updated_at_utc"),
    }


def next_log_sequence(path: Path) -> int:
    if not path.is_file():
        return 1
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip()) + 1


def save_annotation(
    *,
    session_id: str,
    item_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    metadata = load_metadata(session_id)
    items = manifest_index(session_id)
    item = items.get(item_id)
    if item is None:
        raise KeyError(f"item not in session manifest: {item_id}")

    normalized = normalize_annotation_payload(payload)
    now = utc_now()
    log_path = annotations_log_path(session_id)
    record = {
        "schema_version": SCHEMA_VERSION,
        "sequence": next_log_sequence(log_path),
        "session_id": session_id,
        "session_name": metadata.get("session_name"),
        "annotator": metadata.get("annotator"),
        "created_at_utc": now,
        "updated_at_utc": now,
        "item_id": item_id,
        "query_id": item.get("query_id"),
        "doc_id": item.get("doc_id"),
        "report_name": item.get("report_name"),
        "report_year": item.get("report_year"),
        "page_idx": item.get("page_idx"),
        "exchange": item.get("exchange"),
        "ticker": item.get("ticker"),
        "year": item.get("year"),
        "kpi": item.get("kpi"),
        "target_value": item.get("target_value"),
        "target_value_display": item.get("target_value_display"),
        "match_type": item.get("match_type"),
        "alias_matched": item.get("alias_matched"),
        "raw_value": item.get("raw_value"),
        "normalized_value": item.get("normalized_value"),
        "rel_error": item.get("rel_error"),
        "unit_source": item.get("unit_source"),
        "snippet": item.get("snippet"),
        "industry_slug": item.get("industry_slug"),
        "page_text_chars": item.get("page_text_chars"),
        "page_text_sha256": item.get("page_text_sha256"),
        "raw_png_path": item.get("raw_png_path"),
        "mmd_path": item.get("mmd_path"),
        **normalized,
    }

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    current = load_current_annotations(session_id)
    current[item_id] = record
    atomic_write_json(current_annotations_path(session_id), current)

    completed_count = sum(
        1 for rec in current.values() if rec.get("overall_status") != "unreviewed"
    )
    metadata["updated_at_utc"] = now
    metadata["completed_count"] = completed_count
    metadata["item_count"] = len(items)
    atomic_write_json(metadata_path(session_id), metadata)
    write_summary_files(session_id)
    return record


def summary_rows(session_id: str) -> list[dict[str, Any]]:
    metadata = load_metadata(session_id)
    current = load_current_annotations(session_id)
    rows: list[dict[str, Any]] = []
    for item in load_manifest(session_id):
        annotation = current.get(item["item_id"], {})
        rows.append(
            {
                "session_id": session_id,
                "session_name": metadata.get("session_name", ""),
                "annotator": metadata.get("annotator", ""),
                "item_id": item.get("item_id"),
                "query_id": item.get("query_id"),
                "doc_id": item.get("doc_id"),
                "report_name": item.get("report_name"),
                "report_year": item.get("report_year"),
                "page_idx": item.get("page_idx"),
                "exchange": item.get("exchange"),
                "ticker": item.get("ticker"),
                "year": item.get("year"),
                "kpi": item.get("kpi"),
                "target_value": item.get("target_value"),
                "target_value_display": item.get("target_value_display"),
                "match_type": item.get("match_type"),
                "alias_matched": item.get("alias_matched"),
                "raw_value": item.get("raw_value"),
                "normalized_value": item.get("normalized_value"),
                "rel_error": item.get("rel_error"),
                "unit_source": item.get("unit_source"),
                "snippet": item.get("snippet"),
                "overall_status": annotation.get("overall_status", "unreviewed"),
                "notes": annotation.get("notes", ""),
                "updated_at_utc": annotation.get("updated_at_utc", ""),
                "annotation_source": annotation.get("annotation_source", ""),
                "review_duration_ms": annotation.get("review_duration_ms", ""),
                "industry_slug": item.get("industry_slug"),
                "page_text_chars": item.get("page_text_chars"),
                "page_text_sha256": item.get("page_text_sha256"),
                "raw_png_path": item.get("raw_png_path"),
                "mmd_path": item.get("mmd_path"),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    metadata = load_metadata(path.parent.name)
    status_counts = Counter(row["overall_status"] for row in rows)

    reviewed = len(rows) - status_counts.get("unreviewed", 0)
    lines = [
        f"# Qrels Annotation Summary: {metadata.get('session_name', path.parent.name)}",
        "",
        f"- Session ID: `{path.parent.name}`",
        f"- Annotator: `{metadata.get('annotator', '')}`",
        f"- Items: {len(rows)}",
        f"- Reviewed: {reviewed}",
        f"- Updated: {metadata.get('updated_at_utc', '')}",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    atomic_write_text(path, "\n".join(lines) + "\n")


def write_summary_files(session_id: str) -> dict[str, str]:
    rows = summary_rows(session_id)
    directory = session_dir(session_id)
    csv_path = directory / "summary.csv"
    md_path = directory / "summary.md"
    write_summary_csv(csv_path, rows)
    write_summary_md(md_path, rows)
    return {"summary_csv": str(csv_path), "summary_md": str(md_path)}
