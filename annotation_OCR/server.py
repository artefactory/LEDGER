"""Browser-based OCR page annotation server."""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import bleach
import markdown as markdown_lib
from flask import Flask, abort, jsonify, render_template, request, send_file

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
IMAGE_REF_RE = re.compile(r"(!\[[^\]]*\]\()((?:\./)?images/[^)\s]+)(\))")
HTML_IMAGE_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])(images/[^"\']+)(["\'])', re.I)

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
        "--queue-mode",
        choices=["all", "table-candidates", "sample"],
        default="table-candidates",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None, help="Maximum queued pages.")
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

    queue, index_summary = build_queue(
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
        "queue_mode": args.queue_mode,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "limit": args.limit,
        "limit_reports": args.limit_reports,
    }
    metadata = create_session(
        session_name=args.session_name,
        annotator=args.annotator,
        manifest_items=[item.to_manifest_record() for item in queue],
        index_summary=index_summary,
        config=config,
    )
    return metadata["session_id"]


@lru_cache(maxsize=64)
def cached_pages(mmd_path: str) -> tuple[str, ...]:
    return tuple(load_pages(Path(mmd_path)))


def get_item_or_404(session_id: str, index: int) -> dict[str, Any]:
    manifest = load_manifest(session_id)
    if index < 0 or index >= len(manifest):
        abort(404, description="item index out of range")
    return manifest[index]


def item_page_text(item: dict[str, Any]) -> str:
    pages = cached_pages(item["mmd_path"])
    page_index = int(item.get("page_index", 0))
    if page_index < 0 or page_index >= len(pages):
        return ""
    return pages[page_index]


def rewrite_markdown_image_refs(markdown_text: str, session_id: str, index: int) -> str:
    def replace_md(match: re.Match[str]) -> str:
        rel_path = match.group(2).lstrip("./")
        src = f"/api/session/{session_id}/item/{index}/inline-image/{rel_path}"
        return f"{match.group(1)}{src}{match.group(3)}"

    return IMAGE_REF_RE.sub(replace_md, markdown_text)


def rewrite_html_image_refs(html: str, session_id: str, index: int) -> str:
    def replace_html(match: re.Match[str]) -> str:
        rel_path = match.group(2).lstrip("./")
        src = f"/api/session/{session_id}/item/{index}/inline-image/{rel_path}"
        return f"{match.group(1)}{src}{match.group(3)}"

    return HTML_IMAGE_SRC_RE.sub(replace_html, html)


def render_markdown_page(markdown_text: str, session_id: str, index: int) -> str:
    rewritten = rewrite_markdown_image_refs(markdown_text, session_id, index)
    html = markdown_lib.markdown(
        rewritten,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        output_format="html5",
    )
    html = rewrite_html_image_refs(html, session_id, index)
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
    manifest = load_manifest(session_id)
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


def create_app(default_session_id: str, build_defaults: dict[str, Any]) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["DEFAULT_SESSION_ID"] = default_session_id
    app.config["BUILD_DEFAULTS"] = build_defaults

    @app.get("/")
    def index() -> str:
        return render_template("index.html", default_session_id=default_session_id)

    @app.get("/api/sessions")
    def api_sessions() -> Any:
        return jsonify(
            {"sessions": list_sessions(), "default_session_id": default_session_id}
        )

    @app.post("/api/sessions")
    def api_create_session() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        defaults = app.config["BUILD_DEFAULTS"]
        queue_mode = payload.get("queue_mode") or defaults["queue_mode"]
        queue, index_summary = build_queue(
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
            "queue_mode": queue_mode,
            "sample_size": payload.get("sample_size", defaults.get("sample_size")),
            "seed": int(payload.get("seed", defaults["seed"])),
            "limit": payload.get("limit", defaults.get("limit")),
            "limit_reports": payload.get(
                "limit_reports", defaults.get("limit_reports")
            ),
        }
        metadata = create_session(
            session_name=str(payload.get("session_name") or "OCR annotation session"),
            annotator=str(payload.get("annotator") or "anonymous"),
            manifest_items=[item.to_manifest_record() for item in queue],
            index_summary=index_summary,
            config=config,
        )
        return jsonify(
            {"metadata": metadata, "progress": progress_payload(metadata["session_id"])}
        )

    @app.get("/api/session/<session_id>")
    def api_session(session_id: str) -> Any:
        return jsonify(progress_payload(session_id))

    @app.get("/api/session/<session_id>/item/<int:index>")
    def api_item(session_id: str, index: int) -> Any:
        item = get_item_or_404(session_id, index)
        text = item_page_text(item)
        annotations = load_current_annotations(session_id)
        return jsonify(
            {
                "index": index,
                "item_count": len(load_manifest(session_id)),
                "item": item,
                "annotation": annotations.get(item["item_id"]),
                "page_text": text,
                "markdown_html": render_markdown_page(text, session_id, index),
                "image_url": f"/api/session/{session_id}/item/{index}/raw-image",
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
        return send_file(target)

    @app.get("/api/session/<session_id>/item/<int:index>/inline-image/<path:rel_path>")
    def api_inline_image(session_id: str, index: int, rel_path: str) -> Any:
        item = get_item_or_404(session_id, index)
        report_dir = Path(item["report_dir"])
        target = safe_child_path(report_dir, rel_path)
        if not target.is_file():
            abort(404, description="inline OCR image missing")
        return send_file(target)

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
    session_id = prepare_session(args)
    build_defaults = {
        "ocr_root": str(args.ocr_root),
        "raw_root": str(args.raw_root),
        "queue_mode": args.queue_mode,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "limit": args.limit,
        "limit_reports": args.limit_reports,
    }
    app = create_app(session_id, build_defaults)
    print(f"Annotation session: {session_id}")
    print(f"Open: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
