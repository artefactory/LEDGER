"""Browser-based OCR annotation server."""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import bleach
import markdown as markdown_lib
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file

from ocr_index import DEFAULT_OCR_ROOT, DEFAULT_RAW_ROOT, build_queue, load_pages
from store import (
    create_session,
    list_sessions,
    load_current_annotations,
    load_manifest,
    load_metadata,
    save_annotation,
    session_dir,
    write_summary_files,
)


HERE = Path(__file__).resolve().parent
DEFAULT_TABLE_MANIFEST = HERE / "manifests" / "tables_5000.json"
IMAGE_REF_RE = re.compile(r"(!\[[^\]]*\]\()((?:\./)?images/[^)\s]+)(\))")

ALLOWED_TAGS = set(bleach.sanitizer.ALLOWED_TAGS).union(
    {
        "p",
        "br",
        "pre",
        "code",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "img",
        "blockquote",
        "del",
    }
)
ALLOWED_ATTRIBUTES = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    "a": ["href", "title", "rel", "target"],
    "img": ["src", "alt", "title"],
    "th": ["align", "colspan", "rowspan"],
    "td": ["align", "colspan", "rowspan"],
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OCR annotation web UI.")
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--session-id", default=None, help="Resume an existing session."
    )
    parser.add_argument("--session-name", default="OCR annotation session")
    parser.add_argument("--annotator", default="anonymous")
    parser.add_argument(
        "--study-bundle",
        type=Path,
        default=None,
        help="Optional per-session study bundle. When set, each new session gets the next precomputed session queue.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_TABLE_MANIFEST if DEFAULT_TABLE_MANIFEST.is_file() else None,
        help="Optional precomputed queue manifest to reuse instead of rescanning OCR files.",
    )
    parser.add_argument(
        "--queue-mode",
        choices=["all", "table-candidates", "sample", "tables"],
        default="tables",
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None, help="Maximum queued items.")
    parser.add_argument(
        "--limit-reports",
        type=int,
        default=None,
        help="Read only the first N reports before queue selection.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    return parser


def prepare_session(args: argparse.Namespace) -> str:
    if args.session_id:
        metadata = load_metadata(args.session_id)
        return metadata["session_id"]

    manifest_items, index_summary, study_config = resolve_session_source(
        study_bundle_path=args.study_bundle,
        manifest_path=args.manifest_path,
        ocr_root=args.ocr_root,
        raw_root=args.raw_root,
        queue_mode=args.queue_mode,
        sample_size=args.sample_size,
        seed=args.seed,
        limit=args.limit,
        limit_reports=args.limit_reports,
    )
    config = {
        "ocr_root": str(args.ocr_root),
        "raw_root": str(args.raw_root),
        "study_bundle_path": str(args.study_bundle.resolve())
        if args.study_bundle
        else None,
        "manifest_path": str(args.manifest_path) if args.manifest_path else None,
        "queue_mode": args.queue_mode,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "limit": args.limit,
        "limit_reports": args.limit_reports,
        **study_config,
    }
    metadata = create_session(
        session_name=args.session_name,
        annotator=args.annotator,
        manifest_items=manifest_items,
        index_summary=index_summary,
        config=config,
    )
    return metadata["session_id"]


@lru_cache(maxsize=64)
def cached_pages(mmd_path: str) -> tuple[str, ...]:
    return tuple(load_pages(Path(mmd_path)))


@lru_cache(maxsize=16)
def cached_manifest(session_id: str) -> tuple[dict[str, Any], ...]:
    return tuple(load_manifest(session_id))


def load_precomputed_manifest(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"invalid manifest items in {manifest_path}")
    summary = payload.get("summary") or {}
    if not isinstance(summary, dict):
        raise ValueError(f"invalid manifest summary in {manifest_path}")
    summary = {**summary, "manifest_path": str(manifest_path)}
    return items, summary


def load_study_bundle(bundle_path: Path) -> dict[str, Any]:
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    sessions = payload.get("sessions")
    if payload.get("bundle_type") != "ocr_table_study_bundle" or not isinstance(
        sessions, list
    ):
        raise ValueError(f"invalid study bundle in {bundle_path}")
    return payload


def claimed_study_slots(bundle_path: Path) -> set[int]:
    resolved = str(bundle_path.resolve())
    claimed: set[int] = set()
    for metadata in list_sessions():
        config = metadata.get("config") or {}
        if config.get("study_bundle_path") != resolved:
            continue
        slot = config.get("study_slot")
        if isinstance(slot, int):
            claimed.add(slot)
        elif isinstance(slot, str) and slot.isdigit():
            claimed.add(int(slot))
    return claimed


def allocate_study_session(
    bundle_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    bundle = load_study_bundle(bundle_path)
    claimed = claimed_study_slots(bundle_path)
    sessions = bundle["sessions"]
    next_session = None
    for entry in sessions:
        slot = entry.get("slot")
        if isinstance(slot, int) and slot not in claimed:
            next_session = entry
            break
    if next_session is None:
        raise ValueError(f"all study sessions already assigned for {bundle_path}")

    items = next_session.get("items")
    if not isinstance(items, list):
        raise ValueError(f"invalid study session items in {bundle_path}")
    summary = bundle.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    slot = int(next_session["slot"])
    summary = {
        **summary,
        "study_bundle_path": str(bundle_path.resolve()),
        "study_slot": slot,
        "study_target_items": next_session.get("target_items"),
        "study_agreement_items": next_session.get("agreement_items"),
        "study_single_items": next_session.get("single_items"),
    }
    config = {
        "study_slot": slot,
        "study_target_items": next_session.get("target_items"),
        "study_agreement_items": next_session.get("agreement_items"),
        "study_single_items": next_session.get("single_items"),
    }
    return items, summary, config


def resolve_session_source(
    *,
    study_bundle_path: Path | None,
    manifest_path: Path | None,
    ocr_root: Path,
    raw_root: Path,
    queue_mode: str,
    sample_size: int | None,
    seed: int,
    limit: int | None,
    limit_reports: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if study_bundle_path is not None:
        items, summary, config = allocate_study_session(study_bundle_path)
        if limit is not None:
            items = items[:limit]
            summary = {**summary, "limit": limit}
            config = {**config, "limit": limit}
        return items, summary, config

    if manifest_path is not None:
        items, summary = load_precomputed_manifest(manifest_path)
        if limit is not None:
            items = items[:limit]
            summary = {**summary, "limit": limit}
        return items, summary, {}

    queue, index_summary = build_queue(
        ocr_root=ocr_root,
        raw_root=raw_root,
        queue_mode=queue_mode,
        sample_size=sample_size,
        seed=seed,
        limit=limit,
        limit_reports=limit_reports,
    )
    return [item.to_manifest_record() for item in queue], index_summary, {}


def get_item_or_404(session_id: str, index: int) -> dict[str, Any]:
    manifest = cached_manifest(session_id)
    if index < 0 or index >= len(manifest):
        abort(404, description="item index out of range")
    return manifest[index]


def item_page_text(item: dict[str, Any]) -> str:
    if item.get("item_kind") == "table":
        return str(item.get("table_html") or "")
    pages = cached_pages(item["mmd_path"])
    page_index = int(item.get("page_index", 0))
    if page_index < 0 or page_index >= len(pages):
        return ""
    return pages[page_index]


def omit_markdown_image_refs(markdown_text: str) -> str:
    return IMAGE_REF_RE.sub(
        lambda match: f"_[image omitted: {match.group(2)}]_", markdown_text
    )


def rewrite_markdown_image_refs(markdown_text: str, session_id: str, index: int) -> str:
    def replace_md(match: re.Match[str]) -> str:
        rel_path = match.group(2).lstrip("./")
        src = f"/api/session/{session_id}/item/{index}/inline-image/{rel_path}"
        return f"{match.group(1)}{src}{match.group(3)}"

    return IMAGE_REF_RE.sub(replace_md, markdown_text)


def render_markdown_page(
    markdown_text: str,
    *,
    session_id: str,
    index: int,
    show_inline_images: bool,
) -> str:
    if show_inline_images:
        rewritten = rewrite_markdown_image_refs(markdown_text, session_id, index)
    else:
        rewritten = omit_markdown_image_refs(markdown_text)
    html = markdown_lib.markdown(
        rewritten,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        output_format="html5",
    )
    return bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols=["http", "https", "mailto", "data"],
    )


def safe_child_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        abort(400, description="unsafe path")
    resolved_root = root.resolve()
    target = (resolved_root / candidate).resolve()
    if not target.is_relative_to(resolved_root):
        abort(400, description="unsafe path")
    return target


def progress_payload(session_id: str) -> dict[str, Any]:
    metadata = load_metadata(session_id)
    manifest = cached_manifest(session_id)
    current = load_current_annotations(session_id)
    status_counts: dict[str, int] = {}
    for item in manifest:
        status = current.get(item["item_id"], {}).get("overall_status", "unreviewed")
        status_counts[status] = status_counts.get(status, 0) + 1

    next_unreviewed_index = None
    for index, item in enumerate(manifest):
        if item["item_id"] not in current:
            next_unreviewed_index = index
            break

    return {
        "metadata": metadata,
        "item_count": len(manifest),
        "reviewed_count": len(current),
        "status_counts": status_counts,
        "next_unreviewed_index": next_unreviewed_index,
    }


def create_app(default_session_id: str | None, build_defaults: dict[str, Any]) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["DEFAULT_SESSION_ID"] = default_session_id
    app.config["BUILD_DEFAULTS"] = build_defaults

    @app.get("/")
    def index() -> Any:
        # If ?session=<id> in URL, serve the annotation UI for that session
        session_from_url = request.args.get("session")
        if session_from_url:
            return render_template("index.html", session_id=session_from_url)
        # If server was started with a pre-created session, redirect to it
        if default_session_id:
            return redirect(f"/?session={default_session_id}")
        # Otherwise show the landing / session picker page
        return render_template("landing.html")

    @app.get("/api/sessions")
    def api_sessions() -> Any:
        return jsonify(
            {
                "sessions": list_sessions(),
                "default_session_id": default_session_id or None,
            }
        )

    @app.post("/api/sessions")
    def api_create_session() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        defaults = app.config["BUILD_DEFAULTS"]
        queue_mode = payload.get("queue_mode") or defaults["queue_mode"]
        study_bundle_value = payload.get("study_bundle_path") or defaults.get(
            "study_bundle_path"
        )
        study_bundle_path = Path(study_bundle_value) if study_bundle_value else None
        manifest_path_value = payload.get("manifest_path") or defaults.get(
            "manifest_path"
        )
        manifest_path = Path(manifest_path_value) if manifest_path_value else None
        manifest_items, index_summary, study_config = resolve_session_source(
            study_bundle_path=study_bundle_path,
            manifest_path=manifest_path,
            ocr_root=Path(payload.get("ocr_root") or defaults["ocr_root"]),
            raw_root=Path(payload.get("raw_root") or defaults["raw_root"]),
            queue_mode=queue_mode,
            sample_size=payload.get("sample_size", defaults.get("sample_size")),
            seed=int(payload.get("seed", defaults["seed"])),
            limit=payload.get("limit", defaults.get("limit")),
            limit_reports=payload.get("limit_reports", defaults.get("limit_reports")),
        )
        config = {
            "ocr_root": payload.get("ocr_root") or defaults["ocr_root"],
            "raw_root": payload.get("raw_root") or defaults["raw_root"],
            "study_bundle_path": str(study_bundle_path.resolve())
            if study_bundle_path
            else None,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "queue_mode": queue_mode,
            "sample_size": payload.get("sample_size", defaults.get("sample_size")),
            "seed": int(payload.get("seed", defaults["seed"])),
            "limit": payload.get("limit", defaults.get("limit")),
            "limit_reports": payload.get(
                "limit_reports", defaults.get("limit_reports")
            ),
            **study_config,
        }
        metadata = create_session(
            session_name=str(payload.get("session_name") or "OCR annotation session"),
            annotator=str(payload.get("annotator") or "anonymous"),
            manifest_items=manifest_items,
            index_summary=index_summary,
            config=config,
        )
        cached_manifest.cache_clear()
        return jsonify(
            {"metadata": metadata, "progress": progress_payload(metadata["session_id"])}
        )

    @app.get("/api/session/<session_id>")
    def api_session(session_id: str) -> Any:
        return jsonify(progress_payload(session_id))

    @app.get("/api/session/<session_id>/item/<int:index>")
    def api_item(session_id: str, index: int) -> Any:
        manifest = cached_manifest(session_id)
        item = get_item_or_404(session_id, index)
        text = item_page_text(item)
        annotations = load_current_annotations(session_id)
        show_inline_images = request.args.get("inline_images", "1") != "0"
        next_image_url = None
        if index + 1 < len(manifest) and manifest[index + 1].get("raw_png_path"):
            next_image_url = f"/api/session/{session_id}/item/{index + 1}/raw-image"
        return jsonify(
            {
                "index": index,
                "item_count": len(manifest),
                "item": item,
                "annotation": annotations.get(item["item_id"]),
                "page_text": text,
                "markdown_html": render_markdown_page(
                    text,
                    session_id=session_id,
                    index=index,
                    show_inline_images=show_inline_images,
                ),
                "inline_images": show_inline_images,
                "image_url": f"/api/session/{session_id}/item/{index}/raw-image",
                "next_image_url": next_image_url,
            }
        )

    @app.get("/api/session/<session_id>/item/<int:index>/raw-image")
    def api_raw_image(session_id: str, index: int) -> Any:
        item = get_item_or_404(session_id, index)
        raw_png_path = item.get("raw_png_path")
        if not raw_png_path:
            abort(404, description="raw page image missing")
        target = Path(raw_png_path).resolve()
        raw_root = Path(item.get("raw_root") or "/").resolve()
        if not target.is_relative_to(raw_root):
            abort(400, description="raw image outside raw root")
        if not target.is_file():
            abort(404, description="raw page image missing")
        return send_file(target, conditional=True, max_age=86400)

    @app.get("/api/session/<session_id>/item/<int:index>/inline-image/<path:rel_path>")
    def api_inline_image(session_id: str, index: int, rel_path: str) -> Any:
        item = get_item_or_404(session_id, index)
        report_dir = Path(item["report_dir"])
        target = safe_child_path(report_dir, rel_path)
        if not target.is_file():
            abort(404, description="inline OCR image missing")
        return send_file(target, conditional=True, max_age=86400)

    @app.post("/api/session/<session_id>/annotation")
    def api_save_annotation(session_id: str) -> Any:
        payload = request.get_json(force=True, silent=False) or {}
        item_id = payload.get("item_id")
        if not item_id:
            abort(400, description="missing item_id")
        record = save_annotation(
            session_id=session_id, item_id=str(item_id), payload=payload
        )
        return jsonify({"annotation": record, "progress": progress_payload(session_id)})

    @app.get("/api/session/<session_id>/progress")
    def api_progress(session_id: str) -> Any:
        return jsonify(progress_payload(session_id))

    @app.post("/api/session/<session_id>/summarize")
    def api_summarize(session_id: str) -> Any:
        paths = write_summary_files(session_id)
        return jsonify({"paths": paths, "progress": progress_payload(session_id)})

    @app.get("/api/session/<session_id>/summary.csv")
    def api_summary_csv(session_id: str) -> Any:
        write_summary_files(session_id)
        return send_file(session_dir(session_id) / "summary.csv", as_attachment=True)

    @app.get("/api/session/<session_id>/summary.md")
    def api_summary_md(session_id: str) -> Any:
        write_summary_files(session_id)
        return send_file(session_dir(session_id) / "summary.md", as_attachment=True)

    return app


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Session creation is now optional — if no --session-id given and
    # --session-name is the default placeholder, start headless so users
    # can create/resume sessions from the browser landing page.
    session_id: str | None = None
    if args.session_id:
        session_id = prepare_session(args)
    elif args.annotator != "anonymous" or args.session_name != "OCR annotation session":
        session_id = prepare_session(args)

    build_defaults = {
        "ocr_root": str(args.ocr_root),
        "raw_root": str(args.raw_root),
        "study_bundle_path": str(args.study_bundle.resolve())
        if args.study_bundle
        else None,
        "manifest_path": str(args.manifest_path) if args.manifest_path else None,
        "queue_mode": args.queue_mode,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "limit": args.limit,
        "limit_reports": args.limit_reports,
    }
    app = create_app(session_id, build_defaults)
    if session_id:
        print(f"Annotation session: {session_id}")
    else:
        print(
            "Starting in headless mode — users will create sessions from the browser."
        )
    print(f"Open: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
