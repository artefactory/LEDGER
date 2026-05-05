"""OCR document loading for the KPI extraction benchmark.

Discovers report subdirectories under ``--root`` (the OCR tree at
``sample_data/subset_auto_parts_2017_2022/`` by default), picks the right
``.mmd`` file per report, and renders the text with ``[Page N]`` markers so
the LLM can reason about layout and we can later cite pages if we add
provenance.

For the rare reports that exceed the model's prompt budget (some are ~245k
tokens) we tail-truncate by dropping pages from the end. We never silently
drop content — every truncation is logged.

The discovery / mmd-selection logic mirrors
``KPI_analysis/validate_ocr_kpis.py:223-262`` so this module can be used
identically. Reimplemented here to keep the package self-contained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PAGE_SPLIT_RE = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)
REPORT_NAME_RE = re.compile(r"^([A-Z0-9-]+)_(.+)_(\d{4})(?:_[0-9a-fA-F]+)?$")


@dataclass(frozen=True)
class ReportInfo:
    name: str       # directory name e.g. "NYSE_AAP_2019"
    exchange: str
    ticker: str
    year: int
    mmd_path: Path


def parse_report_name(name: str) -> tuple[str, str, int] | None:
    m = REPORT_NAME_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def find_mmd(report_dir: Path) -> Path | None:
    """Pick the right ``.mmd`` for a report directory.

    Preference order:
    1. ``<dirname>.mmd`` if it exists.
    2. Any other ``*.mmd`` not ending in ``_det.mmd``.
    3. Fallback to ``*_det.mmd``.
    """
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


def discover_reports(root: Path) -> list[ReportInfo]:
    """Walk the OCR tree and return one ReportInfo per parseable subdir."""
    out: list[ReportInfo] = []
    seen: set[Path] = set()
    for mmd in root.rglob("*.mmd"):
        d = mmd.parent
        if d in seen:
            continue
        seen.add(d)
    for d in sorted(seen):
        parsed = parse_report_name(d.name)
        if parsed is None:
            continue
        mmd = find_mmd(d)
        if mmd is None:
            continue
        exchange, ticker, year = parsed
        out.append(
            ReportInfo(
                name=d.name,
                exchange=exchange,
                ticker=ticker,
                year=year,
                mmd_path=mmd,
            )
        )
    return out


@dataclass
class LoadedDocument:
    text: str            # final page-marked text passed to the LLM
    n_pages: int         # number of pages present in the source mmd
    n_pages_kept: int    # number of pages actually included after truncation
    n_chars: int         # length of `text`
    truncated: bool      # True if any pages were dropped


def split_pages(raw: str) -> list[str]:
    return [p.strip() for p in PAGE_SPLIT_RE.split(raw) if p.strip()]


def render_with_page_markers(pages: list[str]) -> str:
    """Prepend `[Page N]` to each page (1-indexed)."""
    chunks = []
    for i, page in enumerate(pages, start=1):
        chunks.append(f"[Page {i}]\n{page}")
    return "\n\n".join(chunks)


def load_document(mmd_path: Path, *, max_chars: int | None = None) -> LoadedDocument:
    """Read an `.mmd`, render with page markers, and tail-truncate if needed.

    ``max_chars`` is a soft cap on the rendered text length (a practical
    proxy for tokens — divide a target token budget by ~3.5 to get chars).
    When the rendered text exceeds ``max_chars``, we drop pages from the
    *end* one at a time until we fit, preserving headroom for the system
    prompt and the model's response.
    """
    raw = mmd_path.read_text(encoding="utf-8", errors="replace")
    pages = split_pages(raw)
    n_pages = len(pages)
    if max_chars is None:
        text = render_with_page_markers(pages)
        return LoadedDocument(
            text=text,
            n_pages=n_pages,
            n_pages_kept=n_pages,
            n_chars=len(text),
            truncated=False,
        )

    kept = list(pages)
    text = render_with_page_markers(kept)
    while len(text) > max_chars and len(kept) > 1:
        kept.pop()
        text = render_with_page_markers(kept)
    return LoadedDocument(
        text=text,
        n_pages=n_pages,
        n_pages_kept=len(kept),
        n_chars=len(text),
        truncated=len(kept) < n_pages,
    )
