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
#   "numpy",
#   "torch",
#   "ir-measures",
# ]
#
# # Pin torch to the CUDA 12.8 build: the default wheel is cu13.0, which the
# # local NVIDIA driver (570.x, CUDA 12.8) is too old to run. cu128 is
# # driver-compatible and still GPU-accelerated. Drop this block (and torch
# # above) to fall back to the default/CPU wheel.
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Per-document retrieval evaluation over DeepSeek-OCR'd annual reports.

For every query we build (or reuse) an index of **one report only** — the report
of the query's fiscal year — then rank that report's pages and score the ranking
against ``qrels_llm.txt``. This isolates the question "given the right report,
does the method surface the right page?" from the harder cross-document task.

    # BM25 (instant) and SPLADE (slower, GPU if available) can run in parallel:
    uv run retrieval/evaluate.py --method bm25
    uv run retrieval/evaluate.py --method splade            # device=auto -> cuda/cpu
    uv run retrieval/evaluate.py --method colbert           # late-interaction (MaxSim)

Inputs (defaults match the repo layout):
  --root     tree of {EX}_{TICKER}_{YEAR}/{...}.mmd reports   (sample_data subset)
  --queries  queries.csv with header ``query_id,query_text``
  --qrels    TREC qrels: ``query_id 0 {REPORT}/page_{NNNN} relevance`` (rel in 0/1/2)

Query -> report mapping. The query_id is ``{TICKER}_{kpi}_{YEAR}`` (no exchange),
so we read the report identity from the qrels themselves: among the report
prefixes a query is judged against, the **target report is the one whose trailing
year equals the query's year**. The other years appear because annual reports
print prior-year comparatives — those pages are *not* in the target report's
index and count as non-relevant (relevance 0), per the evaluation design.

Page numbering. ``retrieval.py`` splits pages on ``<--- Page Split --->`` with a
0-based raw-split index; the qrels use the same 0-based index zero-padded to 4
digits (``page_0042`` == ``#p42``). We reuse ``retrieval.py``'s splitter so the
two never drift; a self-check reports any qrels page index past a report's end.

Metrics (binary relevance = ``rel >= --rel-threshold`` for Recall/MRR; graded
0/1/2 for nDCG), macro-averaged over evaluated queries:
  Recall@1/3/5, MRR (over the full per-report ranking), nDCG@5, nDCG@10.

In addition to the hand-rolled metrics above, the *same* (qrels, run) pair is
scored with `ir_measures <https://ir-measur.es/>`_ — the reference TREC
implementation — as an independent cross-check and to expose extra measures
(AP, R-Precision, Success, …). The run feeds ir_measures docids
``{REPORT}/page_{NNNN}`` (matching the qrels) with descending per-rank scores,
and the qrels carry the target report's graded relevances unchanged. Binary
measures (AP / RR / Recall / P / Success) are built with ``rel=--rel-threshold``
so they agree with the hand-rolled Recall/MRR; nDCG stays graded.

Outputs under ``--output-dir`` (default ``retrieval/output/eval``):
  {method}_per_query.csv             one row per evaluated query with every metric
  {method}_summary.md                macro-averaged table + run metadata
  {method}_ir_measures.md            ir_measures aggregate table (unless --no-ir-measures)
  {method}_ir_measures_per_query.csv ir_measures per-query values
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Reuse the *exact* discovery + page-split convention so docids align with qrels.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieval as R  # noqa: E402

YEAR_RE = re.compile(r"(\d{4})$")


def docid(report: str, page: int) -> str:
    """Rebuild the qrels docid for a report page (``{REPORT}/page_{NNNN}``,
    0-based index zero-padded to 4 digits — must match ``load_qrels``)."""
    return f"{report}/page_{page:04d}"


# --- qrels / queries loading ------------------------------------------------ #
def load_queries(path: Path) -> dict[str, str]:
    """query_id -> query_text from a ``query_id,query_text`` CSV."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "query_id" not in rows[0] or "query_text" not in rows[0]:
        sys.exit(f"[eval] {path} must have header 'query_id,query_text'")
    return {r["query_id"].strip(): r["query_text"].strip() for r in rows}


@dataclass(frozen=True)
class QrelEntry:
    """A query's judged report and its per-page relevance within that report."""

    target_report: str            # e.g. "NYSE_AAP_2017"
    page_rel: dict[int, int]      # page index (0-based) -> relevance, target report only


def load_qrels(path: Path) -> dict[str, QrelEntry]:
    """Parse the qrels and, per query, pick the target report (matching the
    query's trailing year) and collect that report's page->relevance map.

    A query is dropped here only if it has *no* judged report for its own year;
    other-year (comparative) pages are simply not carried into ``page_rel``."""
    # qid -> report_prefix -> {page_idx: rel}
    raw: dict[str, dict[str, dict[int, int]]] = {}
    bad = 0
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split("\t") if "\t" in ln else ln.split()
        if len(parts) != 4:
            bad += 1
            continue
        qid, _q0, docid, rel = parts
        prefix, _, page_s = docid.partition("/page_")
        if not page_s or not page_s.isdigit():
            bad += 1
            continue
        raw.setdefault(qid, {}).setdefault(prefix, {})[int(page_s)] = int(rel)
    if bad:
        R.logging.warning("[eval] skipped %d malformed qrels lines", bad)

    out: dict[str, QrelEntry] = {}
    for qid, by_report in raw.items():
        m = YEAR_RE.search(qid)
        if m is None:
            continue
        year = m.group(1)
        # Target = the judged report whose trailing year matches the query year.
        cands = [rp for rp in by_report if (ym := YEAR_RE.search(rp)) and ym.group(1) == year]
        if not cands:
            continue
        # Normally exactly one; if several exchanges collide, prefer the report
        # holding the strongest judgment.
        target = max(cands, key=lambda rp: max(by_report[rp].values()))
        out[qid] = QrelEntry(target, by_report[target])
    return out


# --- per-report page text (reuses retrieval.py's splitter) ------------------ #
def report_pages(ref: R.ReportRef, path: Path, marker_re: re.Pattern) -> list[R.Page]:
    """0-based pages of one report, identical to retrieval.iter_pages's logic."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [p for i, seg in enumerate(marker_re.split(raw))
            if (p := R.Page.from_segment(ref, i, seg)) is not None]


# --- ranking backends (in-memory; SPLADE encoder loaded once) --------------- #
@dataclass
class Bm25Backend:
    k1: float = 0.9
    b: float = 0.75

    def rank(self, doc_texts: list[str], query_texts: list[str], k: int) -> list[list[int]]:
        import bm25s

        retriever = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        retriever.index(bm25s.tokenize(doc_texts, stopwords="en", show_progress=False),
                        show_progress=False)
        pos, _scores = retriever.retrieve(
            bm25s.tokenize(query_texts, stopwords="en", show_progress=False),
            k=min(k, len(doc_texts)), show_progress=False)
        return [[int(p) for p in pos[q]] for q in range(len(query_texts))]


@dataclass
class SpladeBackend:
    model: str = R.DEFAULT_SPLADE_MODEL
    device: str = "auto"

    def __post_init__(self) -> None:
        from sentence_transformers import SparseEncoder

        # Load the model ONCE and reuse it across every report.
        device = self.device if ":" in self.device else R.resolve_device(self.device)
        self._encoder = SparseEncoder(self.model, device=device)

    def rank(self, doc_texts: list[str], query_texts: list[str], k: int) -> list[list[int]]:
        from splade_index import SPLADE

        retriever = SPLADE()
        retriever.index(model=self._encoder, documents=doc_texts, show_progress=False)
        res = retriever.retrieve(query_texts, k=min(k, len(doc_texts)), show_progress=False)
        return [[int(p) for p in res.doc_ids[q]] for q in range(len(query_texts))]


@dataclass
class ColbertBackend:
    """PyLate late-interaction (ColBERT) backend — same encoder and scoring as
    retrieval.py.  Builds a temporary PLAID index per report (small corpus)."""

    model: str = R.DEFAULT_COLBERT_MODEL
    device: str = "auto"
    doc_length: int = 2048
    query_length: int = 32
    batch_size: int = 32
    nbits: int = 2
    kmeans_niters: int = 4

    def __post_init__(self) -> None:
        from pylate import models

        device = self.device if ":" in self.device else R.resolve_device(self.device)
        self._encoder = models.ColBERT(
            model_name_or_path=self.model,
            document_length=self.doc_length,
            query_length=self.query_length,
            device=device,
        )

    def rank(self, doc_texts: list[str], query_texts: list[str], k: int) -> list[list[int]]:
        from pylate import indexes, retrieve

        k = min(k, len(doc_texts))
        doc_embeddings = self._encoder.encode(
            doc_texts, batch_size=self.batch_size, is_query=False, show_progress_bar=False)
        query_embeddings = self._encoder.encode(
            query_texts, batch_size=self.batch_size, is_query=True, show_progress_bar=False)

        with tempfile.TemporaryDirectory() as tmp:
            index = indexes.PLAID(
                index_folder=tmp, index_name="eval",
                override=True, nbits=self.nbits,
                kmeans_niters=self.kmeans_niters)
            index.add_documents(
                documents_ids=[str(i) for i in range(len(doc_texts))],
                documents_embeddings=doc_embeddings)
            ranked = retrieve.ColBERT(index=index).retrieve(
                queries_embeddings=query_embeddings, k=k)

        return [[int(h["id"]) for h in ranked[q]] for q in range(len(query_texts))]


def make_backend(args: argparse.Namespace, device: str | None = None):
    dev = device or args.device
    if args.method == "bm25":
        return Bm25Backend(k1=args.k1, b=args.b)
    elif args.method == "colbert":
        return ColbertBackend(
            model=args.colbert_model, device=dev,
            doc_length=args.doc_length, query_length=args.query_length,
            batch_size=args.batch_size, nbits=args.nbits,
            kmeans_niters=args.kmeans_niters)
    return SpladeBackend(model=args.model, device=dev)


# --- metrics ---------------------------------------------------------------- #
def dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at(ranked_rels: list[int], all_rels: list[int], cut: int) -> float:
    ideal = sorted(all_rels, reverse=True)[:cut]
    idcg = dcg(ideal)
    return dcg(ranked_rels[:cut]) / idcg if idcg > 0 else 0.0


@dataclass
class QueryMetrics:
    qid: str
    report: str
    n_pages: int
    n_rel: int
    first_rel_rank: int          # 0 = no relevant page in ranking
    recall_at: dict[int, float]
    mrr: float
    ndcg: dict[int, float]


def score_query(qid: str, report: str, ranked_pages: list[int],
                page_rel: dict[int, int], threshold: int,
                k_values: tuple[int, ...], ndcg_cuts: tuple[int, ...]) -> QueryMetrics:
    """ranked_pages: page indices best-first (full report ranking)."""
    rel_pages = {p for p, r in page_rel.items() if r >= threshold}
    n_rel = len(rel_pages)
    ranked_rels = [page_rel.get(p, 0) for p in ranked_pages]  # graded, for nDCG
    all_rels = list(page_rel.values())

    first = next((i + 1 for i, p in enumerate(ranked_pages) if p in rel_pages), 0)
    recall = {k: (len({p for p in ranked_pages[:k] if p in rel_pages}) / n_rel
                  if n_rel else 0.0) for k in k_values}
    return QueryMetrics(
        qid=qid, report=report, n_pages=len(ranked_pages), n_rel=n_rel,
        first_rel_rank=first,
        recall_at=recall,
        mrr=(1.0 / first if first else 0.0),
        ndcg={c: ndcg_at(ranked_rels, all_rels, c) for c in ndcg_cuts},
    )


# --- ir_measures cross-check (reference TREC implementation) ---------------- #
def default_ir_measures(threshold: int, k_values: tuple[int, ...],
                        ndcg_cuts: tuple[int, ...]) -> list:
    """A default measure set mirroring the hand-rolled metrics (binary measures
    pinned to ``rel=threshold`` so they agree) plus a few extras."""
    from ir_measures import AP, P, R, RR, Success, nDCG

    measures = [AP(rel=threshold), RR(rel=threshold)]
    measures += [R(rel=threshold) @ k for k in k_values]
    measures += [P(rel=threshold) @ k for k in k_values]
    measures += [Success(rel=threshold) @ k for k in k_values]
    measures += [nDCG @ c for c in ndcg_cuts]  # nDCG is graded — no rel param
    return measures


def resolve_ir_measures(args: argparse.Namespace, k_values: tuple[int, ...],
                        ndcg_cuts: tuple[int, ...]) -> list:
    """User-supplied ``--ir-measures`` string (space/comma-separated) or the
    default set. Unparseable tokens abort with a clear error."""
    import ir_measures

    if not args.ir_measures:
        return default_ir_measures(args.rel_threshold, k_values, ndcg_cuts)
    tokens = [t for t in re.split(r"[,\s]+", args.ir_measures) if t]
    try:
        return [ir_measures.parse_measure(t) for t in tokens]
    except (ValueError, NameError, KeyError) as e:  # ir_measures raises these on bad tokens
        sys.exit(f"[eval] could not parse --ir-measures {args.ir_measures!r}: {e}")


def write_ir_measures(args: argparse.Namespace, measures: list,
                      ir_qrels: dict[str, dict[str, int]],
                      ir_run: dict[str, dict[str, float]]) -> None:
    """Score the (qrels, run) pair with ir_measures and write an aggregate
    markdown table + a per-query CSV alongside the hand-rolled outputs."""
    import ir_measures

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate = ir_measures.calc_aggregate(measures, ir_qrels, ir_run)
    # Stable column order = the order measures were requested in.
    cols = [str(m) for m in measures]
    agg_by_str = {str(m): v for m, v in aggregate.items()}

    # Per-query values -> {qid: {measure_str: value}}.
    per_query: dict[str, dict[str, float]] = {}
    for metric in ir_measures.iter_calc(measures, ir_qrels, ir_run):
        per_query.setdefault(metric.query_id, {})[str(metric.measure)] = metric.value

    csv_path = out_dir / f"{args.method}_ir_measures_per_query.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "qid", *cols])
        for qid in sorted(per_query):
            row = per_query[qid]
            w.writerow([args.method, qid, *[f"{row.get(c, 0.0):.6f}" for c in cols]])

    lines = [f"# ir_measures cross-check — `{args.method}`", "",
             f"- reports root: `{args.root}`",
             f"- queries scored: **{len(ir_qrels)}**  "
             f"(binary measures use rel ≥ {args.rel_threshold}; nDCG graded)",
             "", "| measure | value |", "|---|---|"]
    lines += [f"| {c} | {agg_by_str[c]:.4f} |" for c in cols]
    md_path = out_dir / f"{args.method}_ir_measures.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    R.logging.info("[eval] wrote %s and %s", md_path, csv_path)


# --- driver ----------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    import logging as _logging
    _logging.getLogger("bm25s").setLevel(_logging.WARNING)
    R.logging.set_verbosity(R.logging.INFO)
    k_values = tuple(int(x) for x in args.recall_k.split(","))
    ndcg_cuts = tuple(int(x) for x in args.ndcg_cuts.split(","))
    marker_re = re.compile(args.page_split_marker, re.I)

    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    discovered = R.discover_mmd(args.root)  # report -> (ref, path)
    R.logging.info("[eval] %d queries, %d qrels-judged queries, %d reports under %s",
                   len(queries), len(qrels), len(discovered), args.root)

    # Keep queries that (a) have text, (b) have a target report present locally,
    # (c) have >=1 relevant page in that report.
    todo: dict[str, list[str]] = {}  # report -> [qid, ...]
    for qid, entry in qrels.items():
        if qid not in queries:
            continue
        if entry.target_report not in discovered:
            continue
        if not any(r >= args.rel_threshold for r in entry.page_rel.values()):
            continue
        todo.setdefault(entry.target_report, []).append(qid)

    reports = sorted(todo)
    if args.limit_docs:
        reports = reports[:args.limit_docs]
    if not reports:
        sys.exit("[eval] no evaluable (query, report) pairs — check --root / --qrels")
    R.logging.info("[eval] evaluating %d queries across %d reports (method=%s)",
                   sum(len(todo[r]) for r in reports), len(reports), args.method)

    import queue

    # Determine number of workers and create a pool of backends (one per worker).
    if args.method == "bm25":
        n_workers = args.workers or os.cpu_count()
        backend_queue: queue.Queue = queue.Queue()
        for _ in range(n_workers):
            backend_queue.put(make_backend(args))
    else:
        import torch
        n_gpus = torch.cuda.device_count()
        n_workers = args.workers or n_gpus
        if n_workers > n_gpus:
            R.logging.warning("[eval] requested %d workers but only %d GPUs; capping", n_workers, n_gpus)
            n_workers = n_gpus
        R.logging.info("[eval] loading %d model instance(s) across %d GPU(s)...", n_workers, n_gpus)
        backend_queue = queue.Queue()
        for i in range(n_workers):
            backend_queue.put(make_backend(args, device=f"cuda:{i}"))

    results: list[QueryMetrics] = []
    ir_qrels: dict[str, dict[str, int]] = {}
    ir_run: dict[str, dict[str, float]] = {}
    misaligned = 0

    def _eval_report(report: str):
        backend = backend_queue.get()
        device_id = getattr(backend, 'device', '?')
        try:
            ref, path = discovered[report]
            pages = report_pages(ref, path, marker_re)
            if not pages:
                R.logging.warning("[eval] %s: no pages, skipping", report)
                return None
            R.logging.info("[eval] %s -> %s | %d pages, %d queries",
                           report, device_id, len(pages), len(todo[report]))
            page_ids = [p.page for p in pages]
            pos_to_page = {i: pg for i, pg in enumerate(page_ids)}
            qids = todo[report]
            if args.limit_queries:
                qids = qids[:args.limit_queries]
            ranked = backend.rank([p.text for p in pages], [queries[q] for q in qids], k=len(pages))

            valid_pages = set(page_ids)
            local_results = []
            local_ir_qrels: dict[str, dict[str, int]] = {}
            local_ir_run: dict[str, dict[str, float]] = {}
            local_misaligned = 0
            for qid, hit_positions in zip(qids, ranked):
                ranked_pages = [pos_to_page[p] for p in hit_positions]
                page_rel = qrels[qid].page_rel
                local_misaligned += sum(1 for p, r in page_rel.items()
                                        if r >= args.rel_threshold and p not in valid_pages)
                local_results.append(score_query(qid, report, ranked_pages, page_rel,
                                                 args.rel_threshold, k_values, ndcg_cuts))
                if not args.no_ir_measures:
                    local_ir_qrels[qid] = {docid(report, p): r for p, r in page_rel.items()}
                    n_ranked = len(ranked_pages)
                    local_ir_run[qid] = {docid(report, p): float(n_ranked - i)
                                         for i, p in enumerate(ranked_pages)}
            return local_results, local_ir_qrels, local_ir_run, local_misaligned
        finally:
            backend_queue.put(backend)

    R.logging.info("[eval] processing %d reports with %d worker(s)", len(reports), n_workers)

    done_count = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_eval_report, report): report for report in reports}
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            if result is None:
                continue
            local_results, local_ir_qrels, local_ir_run, local_misaligned = result
            results.extend(local_results)
            ir_qrels.update(local_ir_qrels)
            ir_run.update(local_ir_run)
            misaligned += local_misaligned
            if done_count % 20 == 0 or done_count == len(reports):
                R.logging.info("[eval] %d/%d reports done", done_count, len(reports))

    if misaligned:
        R.logging.warning("[eval] %d relevant qrels pages fall outside their report's "
                          "page range — possible page-numbering drift", misaligned)

    write_outputs(args, results, k_values, ndcg_cuts)
    if not args.no_ir_measures and ir_qrels:
        write_ir_measures(args, resolve_ir_measures(args, k_values, ndcg_cuts),
                          ir_qrels, ir_run)


def write_outputs(args: argparse.Namespace, results: list[QueryMetrics],
                  k_values: tuple[int, ...], ndcg_cuts: tuple[int, ...]) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query = out_dir / f"{args.method}_per_query.csv"
    rec_cols = [f"recall@{k}" for k in k_values]
    ndcg_cols = [f"ndcg@{c}" for c in ndcg_cuts]
    header = ["method", "qid", "report", "n_pages", "n_rel", "first_rel_rank",
              *rec_cols, "mrr", *ndcg_cols]
    with per_query.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in results:
            w.writerow([args.method, m.qid, m.report, m.n_pages, m.n_rel, m.first_rel_rank,
                        *[f"{m.recall_at[k]:.6f}" for k in k_values],
                        f"{m.mrr:.6f}", *[f"{m.ndcg[c]:.6f}" for c in ndcg_cuts]])

    n = len(results)
    means = {c: sum(getattr_metric(m, c) for m in results) / n for c in (*rec_cols, "mrr", *ndcg_cols)}
    lines = [f"# Per-document retrieval evaluation — `{args.method}`", "",
             f"- reports root: `{args.root}`",
             f"- queries evaluated: **{n}**  (rel threshold for Recall/MRR: rel ≥ {args.rel_threshold})",
             f"- avg pages/report ranked: {sum(m.n_pages for m in results)/n:.1f}",
             f"- avg relevant pages/query: {sum(m.n_rel for m in results)/n:.2f}",
             "", "| metric | macro-avg |", "|---|---|"]
    lines += [f"| {c} | {means[c]:.4f} |" for c in (*rec_cols, "mrr", *ndcg_cols)]
    summary = out_dir / f"{args.method}_summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    R.logging.info("[eval] wrote %s and %s", per_query, summary)


def getattr_metric(m: QueryMetrics, col: str) -> float:
    if col == "mrr":
        return m.mrr
    kind, _, k = col.partition("@")
    return (m.recall_at if kind == "recall" else m.ndcg)[int(k)]


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Per-document retrieval evaluation (BM25/SPLADE/ColBERT).")
    p.add_argument("--method", choices=["bm25", "splade", "colbert"], default="bm25")
    p.add_argument("--root", type=Path,
                   default=here.parent / "DeepSeekOCR_Ardian_pruned_1k")
    # p.add_argument("--queries", type=Path, default=here / "queries.csv")
    p.add_argument("--queries", type=Path, default=here / "test_set.csv")

    p.add_argument("--qrels", type=Path, default=here / "qrels_llm.txt")
    p.add_argument("--output-dir", type=Path, default=here / "output" / "eval")
    p.add_argument("--rel-threshold", type=int, default=1,
                   help="min qrels relevance counted as relevant for Recall/MRR")
    p.add_argument("--recall-k", default="1,3,5")
    p.add_argument("--ndcg-cuts", default="5,10")
    p.add_argument("--page-split-marker", default=R.DEFAULT_PAGE_SPLIT)
    p.add_argument("--limit-docs", type=int, default=None, help="evaluate only first N reports")
    p.add_argument("--limit-queries", type=int, default=None, help="cap queries per report")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel workers for report processing (default: cpu_count for bm25, 1 for GPU)")
    # ir_measures cross-check
    p.add_argument("--ir-measures", default=None,
                   help="space/comma-separated ir_measures spec (e.g. 'AP nDCG@10 R@5 "
                        "Rprec'); default mirrors the hand-rolled metrics")
    p.add_argument("--no-ir-measures", action="store_true",
                   help="skip the ir_measures cross-check entirely")
    # BM25 knobs
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)
    # SPLADE knobs
    p.add_argument("--model", default=R.DEFAULT_SPLADE_MODEL)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # ColBERT knobs (pylate — same defaults as retrieval.py)
    p.add_argument("--colbert-model", default=R.DEFAULT_COLBERT_MODEL,
                   help="ColBERT model name or path (default: lightonai/GTE-ModernColBERT-v1)")
    p.add_argument("--doc-length", type=int, default=2048,
                   help="max tokens per document page (ColBERT)")
    p.add_argument("--query-length", type=int, default=32,
                   help="max tokens per query (ColBERT)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="encode batch size (ColBERT)")
    p.add_argument("--nbits", type=int, default=2,
                   help="PLAID residual-quantisation bits (ColBERT)")
    p.add_argument("--kmeans-niters", type=int, default=4,
                   help="PLAID centroid-training iterations (ColBERT)")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())