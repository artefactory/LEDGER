# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "bm25s",
#   "splade-index",
#   "pylate",
#   "sentence-transformers",
#   "simple-parsing",
#   "absl-py",
#   "beartype",
#   "orjson",
# ]
# ///
"""Page-level retrieval over DeepSeek-OCR'd annual reports (BM25 + SPLADE + ColBERT).

Index an OCR'd annual-report tree **page by page**, then rank pages for a query
— the goal is to measure whether a method surfaces the page that holds a given
KPI. ``index`` builds the index once; ``query`` reuses it and emits a TREC run
(``run.trec``) plus a human-readable ``results.jsonl``. A "page" is identified by
docid ``{EXCHANGE}_{TICKER}_{YEAR}#p{PAGE}`` (0-based page, matching
``KPI_analysis/validate_ocr_kpis.py`` so qrels need no off-by-one fix).

    uv run retrieval/retrieval.py index --method bm25 --root /path/to/mmd_tree
    uv run retrieval/retrieval.py query --method bm25 --query "total revenue net sales"
    uv run retrieval/retrieval.py query --method bm25 --report NYSE_SLB_2018 --query "..."
    uv run retrieval/retrieval.py query --method splade --queries_file queries.tsv --top_k 10
    uv run retrieval/retrieval.py index --method colbert --root /path/to/mmd_tree

No-JVM engines: BM25 via ``bm25s`` (Lucene-equivalent scoring), SPLADE via
``splade-index`` + a ``sentence_transformers.SparseEncoder``, and ColBERT
(late interaction) via ``pylate`` (a PLAID index + a ``models.ColBERT`` encoder;
no finance-specialised ColBERT exists on the Hub, so the default is the
long-context ``lightonai/GTE-ModernColBERT-v1``). All three share one docid
space, so their ``run.trec`` files score against the same qrels.

Structure: frozen dataclasses parse the data they represent via ``from_*``
classmethods; fields carry validated types (named beartype predicates, so a
violation reads e.g. ``Is[is_positive]``), which removes hand-written field
checks. ``--method`` is a simple_parsing *subgroup*: ``bm25`` exposes
``--k1/--b``, ``splade`` exposes ``--model/--device``, ``colbert`` exposes
``--model/--device/--doc_length/--query_length/--nbits/--kmeans_niters/
--batch_size/--show_progress`` — and the chosen engine *is* the retrieval engine. The ``index``/``query`` subcommands are the
``IndexConfig``/``QueryConfig`` dataclasses, run via ``.run()``; ``main`` is
``prog.command.run()``, wired into abseil via ``app.run(flags_parser=...)``.

The remaining explicit checks are the ones beartype can't express well: runtime
conditions (empty corpus, missing docid), cross-field invariants (docid↔page,
``--query`` xor ``--queries_file``), and missing-artifact paths where a tailored
message ("run `index` first") beats a generic type violation.

Input: ``.mmd`` files named ``{EXCHANGE}_{TICKER}_{YEAR}.mmd`` (a ``_det.mmd``
sibling is deprioritized), walked recursively; pages split on ``<--- Page Split
--->``. The retrieval engines are lazy-imported inside the handlers.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, ClassVar, Iterator, Literal, Union

import orjson as json  # drop-in dumps/loads; dumps returns bytes (write in "wb")
import simple_parsing
from absl import app, flags, logging
from beartype import beartype
from beartype.vale import Is
from simple_parsing import subgroups
from simple_parsing.helpers.fields import subparsers

DEFAULT_PAGE_SPLIT = r"<---\s*Page Split\s*--->"
# A SparseEncoder-compatible SPLADE checkpoint (encodes docs + queries to sparse
# term weights); the same model is used at index and query time.
DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
# A PyLate-native ColBERT checkpoint. No finance/KPI-specialised ColBERT exists on
# the Hub (the nearest domain fine-tune is legal); GTE-ModernColBERT is ModernBERT-
# based with an 8k context, so it can index whole long financial pages without the
# 512-token truncation a classic BERT ColBERT (e.g. colbertv2.0) would impose.
DEFAULT_COLBERT_MODEL = "lightonai/GTE-ModernColBERT-v1"
# {EXCHANGE}_{TICKER}_{YEAR}: ticker may contain '_'/'-'/'.'; YEAR = trailing 4.
REPORT_RE = re.compile(r"^([A-Za-z0-9]+)_(.+)_(\d{4})$")


# --- validated types -------------------------------------------------------- #
# Named predicates (not lambdas) so a beartype violation names the rule it broke,
# e.g. "violates validator Is[is_positive]" instead of "Is[<lambda>]".
def is_non_negative(n: int) -> bool:
    return n >= 0


def is_positive(n: int) -> bool:
    return n >= 1


def is_non_negative_float(x: float) -> bool:
    return x >= 0.0


def is_unit_interval(x: float) -> bool:
    return 0.0 <= x <= 1.0


def is_non_blank(s: str) -> bool:
    return bool(s.strip())


def is_fiscal_year(y: int) -> bool:
    return 1900 <= y <= 2100


def is_docid(s: str) -> bool:
    return re.fullmatch(r".+#p\d+", s) is not None


def is_existing_dir(p: Path) -> bool:
    return p.is_dir()


NonNeg = Annotated[int, Is[is_non_negative]]  # page positions, counts, budgets
Positive = Annotated[int, Is[is_positive]]  # rank, top_k, pool, limit
NonNegFloat = Annotated[float, Is[is_non_negative_float]]  # BM25 k1
UnitFloat = Annotated[float, Is[is_unit_interval]]  # BM25 b
NonBlank = Annotated[str, Is[is_non_blank]]
FiscalYear = Annotated[int, Is[is_fiscal_year]]
DocId = Annotated[str, Is[is_docid]]
ExistingDir = Annotated[Path, Is[is_existing_dir]]
Method = Literal["bm25", "splade", "colbert"]
Device = Literal["auto", "cpu", "cuda"]


# --- identity records: ReportRef -> PageMeta -> {Page, DocRecord} ----------- #
# A class-level @beartype type-checks every method (incl. classmethods), so no
# per-method decorator is needed.
@beartype
@dataclass(frozen=True, slots=True)
class ReportRef:
    """A report's identity, parsed from its filename stem."""

    report: NonBlank  # "NYSE_AAP_2019"
    exchange: NonBlank
    ticker: NonBlank
    year: FiscalYear

    @classmethod
    def from_stem(cls, stem: str) -> ReportRef | None:
        """``NYSE_AAP_2019``/``..._det`` -> ref, or None if it doesn't match."""
        report = stem[:-4] if stem.endswith("_det") else stem
        m = REPORT_RE.match(report)
        if m is None:
            return None
        return cls(report, m.group(1), m.group(2), int(m.group(3)))


@beartype
@dataclass(frozen=True, slots=True)
class PageMeta(ReportRef):
    """Identity shared by ``Page`` (indexed text) and ``DocRecord`` (stored
    snippet). The guard runs for both — including records read from disk — so
    docid and (report, page) can't drift apart."""

    page: NonNeg
    docid: DocId

    def __post_init__(self) -> None:
        if self.docid != f"{self.report}#p{self.page}":
            raise ValueError(f"docid {self.docid!r} != '{self.report}#p{self.page}'")


@beartype
@dataclass(frozen=True, slots=True)
class Page(PageMeta):
    """One non-empty page; full ``text`` is indexed, never persisted verbatim."""

    text: NonBlank

    @classmethod
    def from_segment(cls, ref: ReportRef, page: NonNeg, segment: str) -> Page | None:
        """A raw page segment -> ``Page``, or None if blank. ``page`` is the
        0-based raw split position (blanks still advance it), so qrels builders
        must split with the same marker and not collapse blanks."""
        text = segment.strip()
        if not text:
            return None
        return cls(
            ref.report, ref.exchange, ref.ticker, ref.year,
            page, f"{ref.report}#p{page}", text,
        )


@beartype
@dataclass(frozen=True, slots=True)
class DocRecord(PageMeta):
    """One docstore.jsonl line: identity + snippet (full text lives in the index)."""

    snippet: str

    @classmethod
    def from_page(cls, p: Page, snippet_chars: NonNeg) -> DocRecord:
        return cls(p.report, p.exchange, p.ticker, p.year, p.page, p.docid,
                   p.text[:snippet_chars])

    @classmethod
    def from_json(cls, line: bytes) -> DocRecord:
        """Parse one docstore line; ``__post_init__`` revalidates the docid."""
        return cls(**json.loads(line))


# --- query/result records --------------------------------------------------- #
@beartype
@dataclass(frozen=True, slots=True)
class Query:
    qid: NonBlank
    text: NonBlank

    @classmethod
    def from_cli(cls, index: NonNeg, text: str) -> Query:
        return cls(f"q{index}", text)

    @classmethod
    def from_topics_file(cls, path: Path) -> list[Query]:
        """Parse ``qid<TAB>text`` rows (one query per line; bare text -> ``q{i}``).
        Rejects an empty file or duplicate qids."""
        if not path.is_file():
            raise FileNotFoundError(f"--queries_file not found: {path}")
        out: list[Query] = []
        for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = raw.strip()
            if not line:
                continue
            qid, _, text = line.partition("\t")
            out.append(cls(qid, text.strip()) if text.strip() else cls(f"q{i}", qid))
        if not out:
            raise ValueError(f"no queries parsed from {path}")
        seen: set[str] = set()
        dupes = {q.qid for q in out if q.qid in seen or seen.add(q.qid)}
        if dupes:
            raise ValueError(f"duplicate query ids in {path}: {sorted(dupes)}")
        return out


@beartype
@dataclass(frozen=True, slots=True)
class ScoredHit:
    """A ranked hit: rank/score paired with our ``DocRecord``."""

    rank: Positive
    score: float
    doc: DocRecord

    @classmethod
    def at(cls, rank: Positive, docid: DocId, score: float,
           docstore: dict[DocId, DocRecord]) -> ScoredHit:
        rec = docstore.get(docid)
        if rec is None:
            raise KeyError(
                f"docid {docid!r} returned by the index has no docstore entry — "
                "index and docstore.jsonl are out of sync; rebuild the index."
            )
        return cls(rank, score, rec)


@beartype
@dataclass(frozen=True, slots=True)
class QueryResult:
    """One emitted JSONL line per query."""

    qid: NonBlank
    query: NonBlank
    report: str | None  # report searched within, or None = whole corpus
    hits: list[ScoredHit]


@beartype
@dataclass(frozen=True, slots=True)
class Layout:
    """All artifact paths for one (output_dir, method), so index and query agree."""

    index_dir: Path
    ids: Path
    docstore: Path
    run: Path
    results: Path

    @classmethod
    def under(cls, output_dir: Path, method: Method) -> Layout:
        base = output_dir / method
        return cls(base / "index", base / "ids.json", base / "docstore.jsonl",
                   base / "run.trec", base / "results.jsonl")


# --- discovery + page iteration --------------------------------------------- #
@beartype
def discover_mmd(root: Path) -> dict[str, tuple[ReportRef, Path]]:
    """Map report name -> (ref, .mmd path), preferring the non-_det variant."""
    chosen: dict[str, tuple[ReportRef, Path]] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".mmd"):
                continue
            ref = ReportRef.from_stem(Path(fn).stem)
            if ref is None:
                continue
            cur = chosen.get(ref.report)
            if cur is None or (cur[1].stem.endswith("_det")
                               and not Path(fn).stem.endswith("_det")):
                chosen[ref.report] = (ref, Path(dirpath) / fn)
    return chosen


@beartype
def iter_pages(root: Path, marker_re: re.Pattern, limit: Positive | None) -> Iterator[Page]:
    """Yield one ``Page`` per non-empty page of every discovered report."""
    chosen = discover_mmd(root)
    names = sorted(chosen)[:limit] if limit is not None else sorted(chosen)
    total = 0
    for report in names:
        ref, path = chosen[report]
        raw = path.read_text(encoding="utf-8", errors="replace")
        pages = [p for i, seg in enumerate(marker_re.split(raw))
                 if (p := Page.from_segment(ref, i, seg)) is not None]
        total += len(pages)
        logging.info("[index] %s: %d pages", report, len(pages))
        yield from pages
    logging.info("[index] %d reports, %d pages total", len(names), total)


# --- on-disk artifacts ------------------------------------------------------ #
@beartype
def write_docstore(pages: list[Page], path: Path, snippet_chars: NonNeg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for p in pages:
            f.write(json.dumps(asdict(DocRecord.from_page(p, snippet_chars))) + b"\n")


@beartype
def read_docstore(path: Path) -> dict[DocId, DocRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"docstore not found: {path} — run `index` first.")
    out = {(r := DocRecord.from_json(line)).docid: r
           for line in path.read_bytes().splitlines() if line.strip()}
    if not out:
        raise ValueError(f"docstore is empty: {path}")
    return out


@beartype
def write_ids(ids: list[DocId], path: Path) -> None:
    """Persist the index-order docid list; engines return positions into it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(ids))


@beartype
def read_ids(path: Path) -> list[DocId]:
    if not path.is_file():
        raise FileNotFoundError(f"id map not found: {path} — run `index` first.")
    return json.loads(path.read_bytes())


@beartype
def write_trec(queries: list[Query], results: dict[str, list[ScoredHit]],
               path: Path, tag: NonBlank) -> None:
    """TREC run: ``qid Q0 docid rank score tag`` (one line per hit)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in queries:
            for h in results.get(q.qid, []):
                f.write(f"{q.qid} Q0 {h.doc.docid} {h.rank} {h.score:.6f} {tag}\n")


@beartype
def write_results_jsonl(queries: list[Query], results: dict[str, list[ScoredHit]],
                        path: Path, report: str | None) -> None:
    """Human-readable companion to run.trec, one ``QueryResult`` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for q in queries:
            rec = QueryResult(q.qid, q.text, report, results.get(q.qid, []))
            f.write(json.dumps(asdict(rec)) + b"\n")


@beartype
def resolve_device(device: Device) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --- retrieval engines == the --method subgroups (lazy-imported, no JVM) ----- #
# Each engine is a subgroup option (its fields are the method-specific flags) AND
# the engine itself: it builds an on-disk index and searches it. ``search``
# returns, per query, ``[(corpus_position, score), ...]``; the caller maps
# positions -> docids via ids.json. Adding ColBERT = one more engine class.
Ranked = list[list[tuple[int, float]]]


@beartype
@dataclass(frozen=True)
class Bm25Engine:
    """bm25s index/search (Lucene-equivalent scoring; k1/b baked in at index)."""

    name: ClassVar[Method] = "bm25"
    k1: NonNegFloat = 0.9
    b: UnitFloat = 0.4

    def index(self, texts: list[str], index_dir: Path) -> None:
        import bm25s

        retriever = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        retriever.index(bm25s.tokenize(texts, stopwords="en"))
        index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(index_dir))

    def search(self, index_dir: Path, texts: list[str], k: Positive) -> Ranked:
        import bm25s

        retriever = bm25s.BM25.load(str(index_dir))
        pos, scores = retriever.retrieve(bm25s.tokenize(texts, stopwords="en"), k=k)
        return [[(int(p), float(s)) for p, s in zip(pos[q], scores[q])]
                for q in range(len(texts))]


@beartype
@dataclass(frozen=True)
class SpladeEngine:
    """splade-index + a SparseEncoder; the same model encodes docs and queries."""

    name: ClassVar[Method] = "splade"
    model: NonBlank = DEFAULT_SPLADE_MODEL
    device: Device = "auto"

    def _encoder(self):
        from sentence_transformers import SparseEncoder

        return SparseEncoder(self.model, device=resolve_device(self.device))

    def index(self, texts: list[str], index_dir: Path) -> None:
        from splade_index import SPLADE

        retriever = SPLADE()
        retriever.index(model=self._encoder(), documents=texts)
        index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(index_dir))

    def search(self, index_dir: Path, texts: list[str], k: Positive) -> Ranked:
        from splade_index import SPLADE

        res = SPLADE.load(str(index_dir), model=self._encoder()).retrieve(texts, k=k)
        return [[(int(p), float(s)) for p, s in zip(res.doc_ids[q], res.scores[q])]
                for q in range(len(texts))]


@beartype
@dataclass(frozen=True)
class ColbertEngine:
    """PyLate late-interaction (ColBERT): a PLAID index + one ``models.ColBERT``
    encoder shared by docs and queries. Unlike BM25/SPLADE (one vector per page),
    ColBERT keeps one vector *per token* and scores by MaxSim, so the index is
    larger and ``--doc_length`` directly trades recall (capture the whole page)
    against index size/latency. The default model is ModernBERT-based (8k ctx),
    so a high ``--doc_length`` indexes full pages instead of silently truncating
    them — the reason to prefer it over a 512-token classic ColBERT.

    Like the other engines, ``index`` writes to ``index_dir`` and ``search``
    *reopens* that on-disk index (``override=False``) — it never re-indexes."""

    name: ClassVar[Method] = "colbert"
    model: NonBlank = DEFAULT_COLBERT_MODEL
    device: Device = "auto"
    # Token cap per page (docs) / per query. doc_length high enough to swallow a
    # whole OCR'd 10-K page; tokens beyond it are dropped (silent truncation).
    doc_length: Positive = 2048
    query_length: Positive = 32
    nbits: Positive = 2  # PLAID residual-quantisation bits (2 = ColBERTv2 default)
    kmeans_niters: Positive = 4  # PLAID centroid-training iterations
    batch_size: Positive = 32  # encode batch size (docs at index, queries at search)
    show_progress: bool = True  # encode progress bar
    index_name: NonBlank = "plaid"  # subdir under index_dir holding the PLAID files

    def _encoder(self):
        from pylate import models

        return models.ColBERT(
            model_name_or_path=self.model,
            document_length=self.doc_length,
            query_length=self.query_length,
            device=resolve_device(self.device),
        )

    def _index(self, index_dir: Path, override: bool):
        from pylate import indexes

        return indexes.PLAID(index_folder=str(index_dir), index_name=self.index_name,
                             override=override, nbits=self.nbits,
                             kmeans_niters=self.kmeans_niters)

    def index(self, texts: list[str], index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        embeddings = self._encoder().encode(
            texts, batch_size=self.batch_size, is_query=False,
            show_progress_bar=self.show_progress)
        # Corpus position is the PLAID docid; the caller maps it back via ids.json.
        self._index(index_dir, override=True).add_documents(
            documents_ids=[str(i) for i in range(len(texts))],
            documents_embeddings=embeddings)

    def search(self, index_dir: Path, texts: list[str], k: Positive) -> Ranked:
        from pylate import retrieve

        embeddings = self._encoder().encode(
            texts, batch_size=self.batch_size, is_query=True,
            show_progress_bar=self.show_progress)
        index = self._index(index_dir, override=False)
        ranked = retrieve.ColBERT(index=index).retrieve(queries_embeddings=embeddings, k=k)
        return [[(int(h["id"]), float(h["score"])) for h in ranked[q]]
                for q in range(len(texts))]


Engine = Union[Bm25Engine, SpladeEngine, ColbertEngine]
ENGINES = {"bm25": Bm25Engine, "splade": SpladeEngine, "colbert": ColbertEngine}


@beartype
@dataclass(frozen=True, slots=True)
class Searcher:
    """A loaded index (engine + id map + docstore) that answers queries: it runs
    the engine, maps corpus positions to docids, applies the optional ``--report``
    filter, and builds ranked ``ScoredHit``s. Separate from the CLI config and the
    output writers."""

    engine: Engine
    ids: list[DocId]
    docstore: dict[DocId, DocRecord]
    index_dir: Path

    @classmethod
    def load(cls, layout: Layout, engine: Engine) -> Searcher:
        if not layout.index_dir.is_dir():
            sys.exit(f"[query] no index at {layout.index_dir} — run "
                     f"`index --method {engine.name}` first.")
        return cls(engine, read_ids(layout.ids), read_docstore(layout.docstore),
                   layout.index_dir)

    def retrieve(self, queries: list[Query], *, top_k: Positive, pool: Positive,
                 report: str | None) -> dict[str, list[ScoredHit]]:
        if report is not None and REPORT_RE.match(report) is None:
            logging.warning("[query] --report %r isn't EXCHANGE_TICKER_YEAR; "
                            "filtering on the docid prefix anyway.", report)
        # With --report we over-retrieve a deep pool, then keep that report's pages.
        depth = min(max(pool, top_k) if report else top_k, len(self.ids))
        ranked = self.engine.search(self.index_dir, [q.text for q in queries], depth)
        out: dict[str, list[ScoredHit]] = {}
        for q, hits in zip(queries, ranked):
            docid_hits = [(self.ids[p], s) for p, s in hits]
            if report:
                docid_hits = [(d, s) for d, s in docid_hits if d.startswith(f"{report}#p")]
                if not docid_hits:
                    logging.warning("[query] qid=%s: no %s pages in top %d — raise --pool",
                                    q.qid, report, depth)
            out[q.qid] = [ScoredHit.at(r, d, s, self.docstore)
                          for r, (d, s) in enumerate(docid_hits[:top_k], 1)]
        return out


# --- command configs == the subcommands (parsed by simple_parsing) ---------- #
@beartype
@dataclass(frozen=True)
class IndexConfig:
    """Build a page-level index from a tree of .mmd files."""

    root: ExistingDir  # tree of DeepSeek .mmd files (validated to exist)
    # --method {bm25,splade,colbert}: choosing one exposes only that engine's flags.
    method: Engine = subgroups(ENGINES, default_factory=Bm25Engine)
    output_dir: Path = Path("retrieval/output")  # artifacts -> <output_dir>/<method>/
    page_split_marker: NonBlank = DEFAULT_PAGE_SPLIT  # case-insensitive page separator
    snippet_chars: NonNeg = 400  # docstore snippet length
    limit: Positive | None = None  # index only the first N reports

    def run(self) -> None:
        lay = Layout.under(self.output_dir, self.method.name)
        pages = list(
            iter_pages(self.root, re.compile(self.page_split_marker, re.I), self.limit)
        )
        if not pages:
            sys.exit(f"[index] no pages under {self.root} — check --root and .mmd naming")
        write_docstore(pages, lay.docstore, self.snippet_chars)
        self.method.index([p.text for p in pages], lay.index_dir)
        write_ids([p.docid for p in pages], lay.ids)
        logging.info("[index] done: %d pages -> %s", len(pages), lay.index_dir)


@beartype
@dataclass(frozen=True)
class QueryConfig:
    """Run queries against an index (one --query, repeatable, or a --queries_file)."""

    # --method {bm25,splade,colbert}: splade/colbert also set the query --model/--device.
    method: Engine = subgroups(ENGINES, default_factory=Bm25Engine)
    output_dir: Path = Path("retrieval/output")  # reads <output_dir>/<method>/
    query: list[str] = field(default_factory=list)  # query string(s)
    queries_file: Path | None = None  # qid<TAB>text rows
    report: str | None = None  # restrict to one report e.g. NYSE_AAP_2019
    pool: Positive = 2000  # depth pulled before --report filter
    top_k: Positive = 10  # results per query
    run_tag: str | None = None  # TREC run tag; default <method>

    def _load_queries(self) -> list[Query]:
        # query and queries_file are mutually exclusive (one source required).
        if bool(self.query) == bool(self.queries_file):
            sys.exit("[query] provide exactly one of --query or --queries_file")
        if self.queries_file:
            return Query.from_topics_file(self.queries_file)
        return [Query.from_cli(i, q) for i, q in enumerate(self.query)]

    def run(self) -> None:
        lay = Layout.under(self.output_dir, self.method.name)
        queries = self._load_queries()
        searcher = Searcher.load(lay, self.method)
        results = searcher.retrieve(queries, top_k=self.top_k, pool=self.pool,
                                    report=self.report)
        write_trec(queries, results, lay.run, self.run_tag or self.method.name)
        write_results_jsonl(queries, results, lay.results, self.report)
        logging.info("[query] %d queries, report=%s, top_k=%d -> %s , %s",
                     len(queries), self.report or "WHOLE DATASET", self.top_k,
                     lay.run, lay.results)


# --- CLI: simple_parsing subcommands, wired into abseil's app.run ----------- #
@dataclass
class Program:
    """Page-level retrieval over DeepSeek-OCR .mmd reports (BM25 + SPLADE + ColBERT)."""

    command: Union[IndexConfig, QueryConfig] = subparsers(
        {"index": IndexConfig, "query": QueryConfig})


@beartype
def parse_flags(argv: list[str]) -> Program:
    """abseil ``flags_parser``: simple_parsing builds the index/query subcommands
    (each scoped to its own flags); leftover args go to absl.flags."""
    parser = simple_parsing.ArgumentParser(description=Program.__doc__)
    parser.add_arguments(Program, dest="prog")
    namespace, remaining = parser.parse_known_args(argv[1:])
    flags.FLAGS([""] + remaining)  # let absl consume --verbosity, --logtostderr, …
    return namespace.prog


@beartype
def main(prog: Program) -> None:
    logging.set_verbosity(logging.INFO)
    prog.command.run()


if __name__ == "__main__":
    app.run(main, flags_parser=parse_flags)
