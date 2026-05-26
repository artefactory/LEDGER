# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "pyserini",
#   "faiss-cpu",
#   "torch",
#   "transformers",
#   "absl-py",
#   "beartype",
#   "orjson",
# ]
# ///
"""Page-level retrieval over DeepSeek-OCR'd annual reports (BM25 + SPLADE).

Use case / intent
-----------------
Index an OCR'd annual-report dataset **page by page**, then query it. A query is
either a single ``--query`` string or a ``--queries_file`` batch; every query
returns ranked **pages**. You can search the whole dataset, or restrict to one
report with ``--report`` (KPI-page localization within a known filing).

The motivating goal: measure whether a retrieval method can surface the *page*
of a report that contains a given KPI. ``index`` builds the index once; ``query``
reuses it and emits a TREC run + a human-readable JSON.

Two subcommands, selected by ``--method {bm25,splade}``.

INDEX (build an index from a tree of DeepSeek .mmd files):

    uv run retrieval/retrieval.py index --method bm25   --root /path/to/mmd_tree
    uv run retrieval/retrieval.py index --method splade --root /path/to/mmd_tree
    # smoke on a few reports:
    uv run retrieval/retrieval.py index --method bm25 --root /path/to/mmd_tree --limit 3

QUERY (-> <output_dir>/<method>/{run.trec, results.jsonl}); a "page" is always
returned, identified by docid ``{EXCHANGE}_{TICKER}_{YEAR}#p{PAGE}``:

    # search the WHOLE dataset
    uv run retrieval/retrieval.py query --method bm25 \
        --query "stock based compensation expense" --top_k 10

    # search WITHIN one report only (KPI-page localization)
    uv run retrieval/retrieval.py query --method bm25 \
        --report NYSE_SLB_2018 --query "total revenue net sales"

    # batch queries from a file (qid<TAB>text, or one query per line)
    uv run retrieval/retrieval.py query --method splade \
        --queries_file queries.tsv --top_k 10

Self-contained: every helper lives here; nothing is imported from the repo.
Heavy engine deps (pyserini / torch) are lazy-imported inside the handlers.

Input format (DeepSeek OCR): a directory tree of ``.mmd`` files named
``{EXCHANGE}_{TICKER}_{YEAR}.mmd`` (optionally a ``_det.mmd`` sibling), either
flat or one-per-report-dir. Pages within a file are separated by the literal
marker ``<--- Page Split --->``. The tree is walked recursively (os.walk).

Notes
-----
* pyserini needs a JVM (JDK 21). No explicit check here — pyserini fast-fails
  with a clear error if the JVM is missing or the wrong version.
* ``--report`` is a post-filter: it pulls a deep ``--pool`` from the full index
  and keeps that report's pages. A too-small pool can miss a page (it warns).
* SPLADE query encoding runs on CPU inside LuceneImpactSearcher (fine for a few
  queries; slower for large ``--queries_file`` batches). ``--device`` controls
  the document encoder at index time, where the heavy work is.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Iterator

import orjson as json  # drop-in dumps/loads; dumps returns bytes (write in "wb")
from absl import app
from absl.flags import argparse_flags
from beartype import beartype
from beartype.vale import Is

# --- format constants (the marker is also exposed as --page_split_marker) ----
DEFAULT_PAGE_SPLIT = r"<---\s*Page Split\s*--->"
# Public/ungated SPLADE++ checkpoint (pyserini's reference model). Override with
# --splade_model; e.g. naver/splade-v3 is gated and needs HF access first.
DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
# {EXCHANGE}_{TICKER}_{YEAR} — ticker may contain '_'/'-'/'.'; YEAR is the
# trailing 4 digits, EXCHANGE the leading alnum token.
REPORT_RE = re.compile(r"^([A-Za-z0-9]+)_(.+)_(\d{4})$")

# 0-based page index, matching KPI_analysis/validate_ocr_kpis.py (which numbers
# pages with enumerate(pages) starting at 0). Keeps docids aligned with the
# KPI-extraction code so qrels derived from it need no off-by-one fix.
PageIndex = Annotated[int, Is[lambda n: n >= 0]]
# 1-based rank within a query's result list.
Rank = Annotated[int, Is[lambda n: n >= 1]]
# A non-blank string: rejects "" and whitespace-only fields at construction.
NonBlank = Annotated[str, Is[lambda s: bool(s.strip())]]
# Report cover year; annual-report corpus, so a sane four-digit range.
FiscalYear = Annotated[int, Is[lambda y: 1900 <= y <= 2100]]
# docid must carry the "#p<page>" page suffix the rest of the pipeline parses.
DocId = Annotated[str, Is[lambda s: re.fullmatch(r".+#p\d+", s) is not None]]


@beartype
@dataclass(frozen=True, slots=True)
class Page:
    docid: DocId        # "NYSE_AAP_2019#p11"  (page 11 = 12th page, 0-based)
    report: NonBlank    # "NYSE_AAP_2019"
    exchange: NonBlank
    ticker: NonBlank
    year: FiscalYear
    page: PageIndex
    text: NonBlank

    def __post_init__(self) -> None:
        # docid must be exactly "<report>#p<page>" — guards against the two ever
        # drifting (e.g. a renamed report or a renumbered page).
        if self.docid != f"{self.report}#p{self.page}":
            raise ValueError(
                f"docid {self.docid!r} != '{self.report}#p{self.page}'"
            )


@beartype
@dataclass(frozen=True, slots=True)
class Query:
    qid: NonBlank
    text: NonBlank


@beartype
@dataclass(frozen=True, slots=True)
class DocRecord:
    """A docstore record: page metadata + snippet, the searchable-page identity
    we own (one JSONL line in docstore.jsonl). ``text`` lives in the index, not
    here — only a snippet is kept for human-readable output."""
    docid: DocId
    report: NonBlank
    exchange: NonBlank
    ticker: NonBlank
    year: FiscalYear
    page: PageIndex
    snippet: str

    @classmethod
    @beartype
    def from_page(cls, page: Page, snippet_chars: int) -> "DocRecord":
        return cls(
            docid=page.docid, report=page.report, exchange=page.exchange,
            ticker=page.ticker, year=page.year, page=page.page,
            snippet=page.text[:snippet_chars],
        )


@beartype
@dataclass(frozen=True, slots=True)
class ScoredHit:
    """A pyserini hit's rank/score paired with our own ``DocRecord``. We read
    only ``.docid``/``.score`` off pyserini's hit; the rest is data we own."""
    rank: Rank
    score: float
    doc: DocRecord


@beartype
@dataclass(frozen=True, slots=True)
class QueryResult:
    """The result record we emit per query (one JSONL line via asdict())."""
    qid: NonBlank
    query: str
    report: str | None  # report searched within, or None = whole corpus
    hits: list[ScoredHit]

    @classmethod
    @beartype
    def from_hits(
        cls,
        query: Query,
        hits: list,
        docstore: dict[str, DocRecord],
        report: str | None,
    ) -> "QueryResult":
        """Build a record from a query's ranked pyserini hits. ``docstore`` is
        built in the same index run, so every hit docid is present (direct
        lookup, no fallback)."""
        scored = [
            ScoredHit(rank=rank, score=float(h.score), doc=docstore[h.docid])
            for rank, h in enumerate(hits, start=1)
        ]
        return cls(qid=query.qid, query=query.text, report=report, hits=scored)


# --------------------------------------------------------------------------- #
# Discovery + page splitting (DeepSeek .mmd)                                   #
# --------------------------------------------------------------------------- #
@beartype
def parse_report_stem(stem: str) -> tuple[str, str, str, int] | None:
    """``NYSE_AAP_2019`` / ``NYSE_AAP_2019_det`` -> (report, exchange, ticker, year)."""
    report = stem[:-4] if stem.endswith("_det") else stem
    m = REPORT_RE.match(report)
    if m is None:
        return None
    return report, m.group(1), m.group(2), int(m.group(3))


@beartype
def discover_mmd(root: Path) -> dict[str, Path]:
    """Walk ``root`` and map report name -> chosen ``.mmd`` path.

    Handles both layouts (flat ``NYSE_X_2019.mmd`` files and per-report dirs).
    When both ``{name}.mmd`` and ``{name}_det.mmd`` exist, prefer the non-_det
    variant (matches the existing repo convention in ``document.py``).
    """
    chosen: dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".mmd"):
                continue
            stem = Path(fn).stem  # strips only the .mmd suffix; safe for dotted tickers
            parsed = parse_report_stem(stem)
            if parsed is None:
                continue
            report = parsed[0]
            path = Path(dirpath) / fn
            if report not in chosen:
                chosen[report] = path
            elif chosen[report].stem.endswith("_det") and not stem.endswith("_det"):
                chosen[report] = path
    return chosen


@beartype
def split_pages(raw: str, marker_re: re.Pattern) -> list[tuple[int, str]]:
    """Return ``(page_index, text)`` for each non-empty page.

    ``page_index`` is the **0-based position in the raw page sequence** (matching
    ``KPI_analysis/validate_ocr_kpis.py``): empty / blank pages still advance the
    count, so an index always maps to the document's actual page. This defines the
    docid page space — any qrels builder must use the same 0-based splitting.
    """
    pages: list[tuple[int, str]] = []
    for i, seg in enumerate(marker_re.split(raw), start=0):
        text = seg.strip()
        if text:
            pages.append((i, text))
    return pages


@beartype
def iter_pages(root: Path, marker_re: re.Pattern, limit: int | None) -> Iterator[Page]:
    """Yield one ``Page`` per non-empty page of every discovered report. Logs counts."""
    chosen = discover_mmd(root)
    names = sorted(chosen)
    if limit is not None:
        names = names[:limit]
    if not names:
        return
    total = 0
    for report in names:
        path = chosen[report]
        _report, exchange, ticker, year = parse_report_stem(path.stem)  # type: ignore[misc]
        raw = path.read_text(encoding="utf-8", errors="replace")
        pages = split_pages(raw, marker_re)
        for page_no, text in pages:
            yield Page(
                docid=f"{report}#p{page_no}",
                report=report,
                exchange=exchange,
                ticker=ticker,
                year=year,
                page=page_no,
                text=text,
            )
        total += len(pages)
        print(f"[index] {report}: {len(pages)} pages", file=sys.stderr)
    print(f"[index] {len(names)} reports, {total} pages total", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Docstore (docid -> metadata + snippet, for human-readable query output)      #
# --------------------------------------------------------------------------- #
@beartype
def write_docstore(pages: list[Page], path: Path, snippet_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for p in pages:
            f.write(json.dumps(asdict(DocRecord.from_page(p, snippet_chars))))
            f.write(b"\n")


@beartype
def read_docstore(path: Path) -> dict[str, DocRecord]:
    out: dict[str, DocRecord] = {}
    if not path.exists():
        return out
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                rec = DocRecord(**json.loads(line))
                out[rec.docid] = rec
    return out


# --------------------------------------------------------------------------- #
# Queries + output writers                                                     #
# --------------------------------------------------------------------------- #
@beartype
def load_queries(query: list[str] | None, queries_file: str | None) -> list[Query]:
    """Build the query list.

    ``--query`` strings get qids q0,q1,...  A ``--queries_file`` is parsed by
    pyserini's ``DefaultQueryIterator`` (Anserini ``qid<TAB>text`` topic reader),
    so it must end in ``.tsv``/``.txt`` (optionally ``.gz``) or ``.jsonl``.
    """
    out: list[Query] = []
    if queries_file:
        from pyserini.query_iterator import TopicsFormat, get_query_iterator
        for qid, text in get_query_iterator(queries_file, TopicsFormat.DEFAULT):
            out.append(Query(qid=str(qid), text=text))
    for i, q in enumerate(query or []):
        out.append(Query(qid=f"q{i}", text=q))
    return out


@beartype
def write_results_jsonl(
    results: dict[str, list],
    queries: list[Query],
    docstore: dict[str, DocRecord],
    path: Path,
    report: str | None,
) -> None:
    """Human-readable companion to the TREC run, one ``QueryResult`` per line.
    ``results`` maps qid -> ranked pyserini hits; rows are built by
    ``QueryResult.from_hits`` and serialized via ``asdict``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for q in queries:
            record = QueryResult.from_hits(q, results[q.qid], docstore, report)
            f.write(json.dumps(asdict(record)))
            f.write(b"\n")


@beartype
def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# index subcommand                                                             #
# --------------------------------------------------------------------------- #
def cmd_index(args) -> None:
    root = Path(args.root)
    marker_re = re.compile(args.page_split_marker, re.IGNORECASE)
    base = Path(args.output_dir) / args.method
    index_dir = base / "index"
    docstore = index_dir / "docstore.jsonl"
    encoded_dir = base / "encoded"

    pages = list(iter_pages(root, marker_re, args.limit))
    if not pages:
        sys.exit(f"[index] no pages discovered under {root} — check --root and .mmd naming")
    write_docstore(pages, docstore, args.snippet_chars)

    if args.method == "bm25":
        _index_bm25(pages, index_dir, args.threads)
    else:
        _index_splade(
            pages, index_dir, encoded_dir, args.splade_model,
            resolve_device(args.device), args.threads, args.refresh,
        )
    print(f"[index] done: {len(pages)} pages -> {index_dir} (docstore: {docstore})", file=sys.stderr)


@beartype
def _index_bm25(pages: list[Page], index_dir: Path, threads: int) -> None:
    """Write pages as a JsonCollection, then build a Lucene inverted index."""
    with tempfile.TemporaryDirectory() as tmp:
        with (Path(tmp) / "docs.jsonl").open("wb") as f:
            for p in pages:
                f.write(json.dumps({"id": p.docid, "contents": p.text}))
                f.write(b"\n")
        index_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection", "--input", tmp,
            "--index", str(index_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ], check=True)


@beartype
def _index_splade(
    pages: list[Page], index_dir: Path, encoded_dir: Path, model: str,
    device: str, threads: int, refresh: bool,
) -> None:
    """Encode pages to SPLADE impact vectors (cached), then build an impact index.

    Uses pyserini's own SPLADE document encoder so the doc term-space +
    quantization match the query encoder used at search time.
    """
    corpus_dir = encoded_dir / "corpus"
    vectors_dir = encoded_dir / "vectors"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    have_cache = any(vectors_dir.glob("*.jsonl"))
    if refresh or not have_cache:
        with (corpus_dir / "docs.jsonl").open("wb") as f:
            for p in pages:
                f.write(json.dumps({"id": p.docid, "text": p.text}))
                f.write(b"\n")
        subprocess.run([
            sys.executable, "-m", "pyserini.encode",
            "input", "--corpus", str(corpus_dir), "--fields", "text",
            "output", "--embeddings", str(vectors_dir),
            "encoder", "--encoder", model, "--encoder-class", "splade",
            "--fields", "text", "--batch-size", "32", "--device", device,
        ], check=True)
    else:
        print(f"[index] reusing SPLADE vectors in {vectors_dir} (use --refresh to re-encode)", file=sys.stderr)

    index_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonVectorCollection", "--input", str(vectors_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--impact", "--pretokenized",
    ], check=True)


# --------------------------------------------------------------------------- #
# query subcommand                                                             #
# --------------------------------------------------------------------------- #
def cmd_query(args) -> None:
    queries = load_queries(args.query, args.queries_file)
    if not queries:
        sys.exit("[query] provide at least one --query or a --queries_file")

    report = args.report  # None = search the whole dataset; else only this report
    base = Path(args.output_dir) / args.method
    index_dir = base / "index"
    docstore_path = index_dir / "docstore.jsonl"
    docstore = read_docstore(docstore_path)
    run_file = base / "run.trec"
    results_jsonl = base / "results.jsonl"
    tag = args.run_tag or args.method

    if args.method == "bm25":
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(str(index_dir))
        searcher.set_bm25(args.k1, args.b)
    else:
        from pyserini.search.lucene import LuceneImpactSearcher
        # "splade" in the model name -> pyserini picks SpladeQueryEncoder.
        searcher = LuceneImpactSearcher(str(index_dir), args.splade_model)

    results: dict[str, list] = {}
    for q in queries:
        if report:
            # Restrict to one report: retrieve a deep pool from the whole index,
            # then keep only that report's pages (docids start with "<report>#p").
            # Uniform for BM25 + SPLADE; raise --pool if a page is ever missed.
            pool = max(args.pool, args.top_k)
            hits = searcher.search(q.text, k=pool)
            kept = [h for h in hits if h.docid.startswith(f"{report}#p")][: args.top_k]
            if not kept:
                print(f"[query] WARN qid={q.qid}: no pages of {report} in top {pool} — raise --pool",
                      file=sys.stderr)
        else:
            kept = list(searcher.search(q.text, k=args.top_k))
        results[q.qid] = kept

    # TREC run via pyserini's own writer (identical format, future-proof).
    from pyserini.output_writer import OutputFormat, get_output_writer
    with get_output_writer(str(run_file), OutputFormat.TREC, max_hits=args.top_k, tag=tag) as writer:
        for qid, hits in results.items():
            writer.write(qid, hits)
    write_results_jsonl(results, queries, docstore, results_jsonl, report)
    print(
        f"[query] {len(queries)} queries, report={report or 'WHOLE DATASET'}, "
        f"top_k={args.top_k} -> {run_file} , {results_jsonl}",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# CLI (abseil-py: argparse_flags subparsers wired via app.run)                 #
# --------------------------------------------------------------------------- #
def parse_flags(argv):
    parser = argparse_flags.ArgumentParser(
        description="Page-level retrieval over DeepSeek-OCR .mmd reports (BM25 + SPLADE)."
    )
    sub = parser.add_subparsers(help="index | query")

    pi = sub.add_parser("index", help="build an index from a tree of .mmd files")
    pi.add_argument("--method", choices=["bm25", "splade"], required=True)
    pi.add_argument("--root", required=True,
                    help="directory tree of DeepSeek .mmd files (walked recursively)")
    pi.add_argument("--output_dir", default="retrieval/output",
                    help="base dir for produced artifacts; everything lands under "
                         "<output_dir>/<method>/ (index, docstore.jsonl, encoded)")
    pi.add_argument("--page_split_marker", default=DEFAULT_PAGE_SPLIT,
                    help="regex (case-insensitive) separating pages")
    pi.add_argument("--snippet_chars", type=int, default=400)
    pi.add_argument("--limit", type=int, default=None,
                    help="index only the first N reports (smoke test)")
    pi.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    pi.add_argument("--splade_model", default=DEFAULT_SPLADE_MODEL,
                    help="SPLADE doc encoder (must match the query encoder)")
    pi.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"])
    pi.add_argument("--refresh", action="store_true", help="bypass the SPLADE encode cache")
    pi.set_defaults(command=cmd_index)

    pq = sub.add_parser("query", help="run queries against an index")
    pq.add_argument("--method", choices=["bm25", "splade"], required=True)
    pq.add_argument("--output_dir", default="retrieval/output",
                    help="base dir holding the index built by `index`; reads/writes "
                         "under <output_dir>/<method>/ (index, run.trec, results.json)")
    pq.add_argument("--query", action="append", default=None,
                    help="a query string (repeat for several)")
    pq.add_argument("--queries_file", default=None,
                    help="'qid<TAB>text' rows, parsed by pyserini's topic reader; "
                         "must end in .tsv/.txt (optionally .gz) or .jsonl")
    pq.add_argument("--report", default=None,
                    help="restrict search to this report, e.g. NYSE_AAP_2019 "
                         "(omit to search the whole dataset). Returns pages either way.")
    pq.add_argument("--pool", type=int, default=2000,
                    help="with --report: retrieval depth pulled from the full index "
                         "before filtering down to that report's pages")
    pq.add_argument("--top_k", type=int, default=10)
    pq.add_argument("--run_tag", default=None, help="TREC run tag (6th column); default <method>")
    pq.add_argument("--k1", type=float, default=0.9, help="BM25 k1")
    pq.add_argument("--b", type=float, default=0.4, help="BM25 b")
    pq.add_argument("--splade_model", default=DEFAULT_SPLADE_MODEL,
                    help="SPLADE query encoder (must match index-time model)")
    pq.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"])
    pq.set_defaults(command=cmd_query)

    args = parser.parse_args(argv[1:])
    if not getattr(args, "command", None):
        parser.error("specify a subcommand: 'index' or 'query'")
    return args


def main(args):
    args.command(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_flags)
