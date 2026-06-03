"""Run the needle-in-a-haystack KPI benchmark against a vLLM server.

For each OCR'd annual report, this issues all of that report's queries against a
SINGLE prefilled-and-cached document prefix:

  1. Group the test-set queries by source report (``needle_data.group_by_report``).
  2. For each report (processed one at a time so only one big document prefix is
     resident in the KV cache):
       a. Load and page-mark the ``.mmd`` once.
       b. Send ONE warm-up query, blocking, to prefill + cache the prefix.
       c. Fire the remaining queries concurrently — each reuses the cached
          prefix and only decodes a ~80-token suffix + short JSON answer.
  3. Append every result (including the raw model text and token usage) to
     ``responses.jsonl``.

This is the whole point of launching vLLM with ``--enable-prefix-caching``:
~100k document tokens are prefilled once per report instead of once per query.

The server must be started separately, e.g.:

  vllm serve Qwen/Qwen3.6-27B-FP8 \\
      --enable-prefix-caching \\
      --max-model-len 131072 \\
      --guided-decoding-backend xgrammar \\
      --port 8000

Smoke test (prototype set, 3 reports / 76 queries):

  uv run python KPI_analysis/llm_benchmark/needle_haystack/run_needle.py \\
      --model Qwen/Qwen3.6-27B-FP8 --prototype

Plan the run offline (no server needed) — prints the prefix-cache plan and
estimated token savings:

  uv run python KPI_analysis/llm_benchmark/needle_haystack/run_needle.py \\
      --model X --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import needle_data as nd  # noqa: E402
from document import LoadedDocument, ReportInfo, load_document  # noqa: E402
from needle_client import NeedleResult, call_needle, make_client  # noqa: E402
from needle_data import QueryRecord  # noqa: E402
from needle_prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_message,
    build_question_block,
    prompt_version,
)

DEFAULT_OUTPUT_BASE = HERE / "output"


def model_slug(model: str) -> str:
    """Filesystem-safe slug for a model id (``openai/gpt-oss-20b`` -> ``openai__gpt-oss-20b``)."""
    safe = []
    for ch in model:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        elif ch == "/":
            safe.append("__")
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "model"


# ---------------------------------------------------------------------------
# Token estimation (cl100k proxy — for the too-long guard + savings estimate)
# ---------------------------------------------------------------------------

_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        import tiktoken

        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def estimate_tokens(text: str) -> int:
    return len(_enc().encode(text))


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def load_done_query_ids(responses_path: Path) -> set[str]:
    """Read query_ids already present with a terminal status in responses.jsonl."""
    done: set[str] = set()
    if not responses_path.is_file():
        return done
    with responses_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("query_id")
            if qid and rec.get("status") in ("ok", "failed", "skipped_too_long"):
                done.add(qid)
    return done


# ---------------------------------------------------------------------------
# Per-query record building
# ---------------------------------------------------------------------------


def build_record(
    rec: QueryRecord,
    report: ReportInfo,
    doc: LoadedDocument,
    doc_tokens: int,
    *,
    model: str,
    query_mode: str,
    is_warmup: bool,
    result: NeedleResult | None,
    status: str,
    error: str | None = None,
) -> dict:
    out: dict = {
        "query_id": rec.query_id,
        "ticker": rec.ticker,
        "kpi": rec.kpi,
        "year": rec.year,
        "unit_class": rec.unit_class,
        "query_mode": query_mode,
        "query_text": rec.query_text,
        "report_name": report.name,
        "exchange": report.exchange,
        "mmd_path": str(report.mmd_path),
        "n_pages": doc.n_pages,
        "n_chars": doc.n_chars,
        "doc_tokens_est": doc_tokens,
        "model": model,
        "status": status,
        "is_warmup": is_warmup,
        "error": error,
    }
    if result is not None:
        ans = result.answer
        out.update(
            {
                "attempts": result.attempts,
                "latency_s": round(result.latency_s, 3),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cached_tokens": result.cached_tokens,
                "found": ans.found if ans else None,
                "value": ans.value if ans else None,
                "value_verbatim": ans.value_verbatim if ans else None,
                "unit_scale": ans.unit_scale if ans else None,
                "page": ans.page if ans else None,
                "raw_response": result.raw_response,
                "error": error or result.error,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class JsonlWriter:
    """Thread-safe append-and-flush writer for responses.jsonl."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def run(args: argparse.Namespace) -> None:
    test_set_path = nd.DEFAULT_PROTOTYPE if args.prototype else args.test_set
    records, stats = nd.build_query_records(
        test_set_path=test_set_path,
        kpis_long_path=args.kpis_long,
        ocr_root=args.ocr_root,
        require_report=True,
    )
    sys.stderr.write(
        f"[setup] test set: {test_set_path.name} | resolved {stats['kept']}/"
        f"{stats['total']} queries "
        f"(dropped: no_ground_truth={stats['no_ground_truth']}, "
        f"no_report={stats['no_report']})\n"
    )

    groups = nd.group_by_report(records)
    if args.limit_reports is not None:
        groups = groups[: args.limit_reports]
    if args.limit_queries is not None:
        groups = [(r, qs[: args.limit_queries]) for (r, qs) in groups]
    n_queries = sum(len(qs) for _, qs in groups)
    sys.stderr.write(
        f"[setup] {len(groups)} reports, {n_queries} queries "
        f"(avg {n_queries / max(len(groups), 1):.1f} queries/report)\n"
    )

    out_dir = args.output_dir or (DEFAULT_OUTPUT_BASE / model_slug(args.model))
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.jsonl"

    done: set[str] = set()
    if args.resume:
        done = load_done_query_ids(responses_path)
        sys.stderr.write(f"[setup] --resume: {len(done)} queries already done\n")
    elif responses_path.exists() and not args.dry_run:
        sys.stderr.write(
            f"[setup] WARNING: {responses_path} exists and --resume not set; "
            "appending (may duplicate query_ids). Delete it for a clean run.\n"
        )

    # ----- dry run: print the prefix-cache plan, no server contact -----
    if args.dry_run:
        _dry_run_report(groups, args, out_dir)
        return

    client = make_client(args.base_url, args.api_key)
    writer = JsonlWriter(responses_path)
    counters = {"ok": 0, "failed": 0, "error": 0, "skipped_too_long": 0}
    started = time.monotonic()

    def call_one(
        rec: QueryRecord,
        report: ReportInfo,
        doc: LoadedDocument,
        doc_tokens: int,
        is_warmup: bool,
    ) -> dict:
        qb = build_question_block(
            company_name=(rec.gt.company_name if rec.gt else rec.ticker),
            ticker=rec.ticker,
            year=rec.year,
            kpi_label=rec.kpi_label,
            kpi_definition=rec.kpi_definition,
            unit_class=rec.unit_class,
            query_text=rec.query_text,
            query_mode=args.query_mode,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(doc.text, qb)},
        ]
        try:
            result = call_needle(
                client,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                enable_thinking=args.enable_thinking,
                reasoning_effort=args.reasoning_effort,
                retries=args.retries,
            )
        except Exception as e:  # noqa: BLE001
            return build_record(
                rec,
                report,
                doc,
                doc_tokens,
                model=args.model,
                query_mode=args.query_mode,
                is_warmup=is_warmup,
                result=None,
                status="error",
                error=f"{type(e).__name__}: {e}",
            )
        status = "ok" if result.answer is not None else "failed"
        return build_record(
            rec,
            report,
            doc,
            doc_tokens,
            model=args.model,
            query_mode=args.query_mode,
            is_warmup=is_warmup,
            result=result,
            status=status,
        )

    pbar = tqdm(total=n_queries, desc=model_slug(args.model), unit="q")
    for report, queries in groups:
        pending = [q for q in queries if q.query_id not in done]
        if not pending:
            pbar.update(len(queries))
            continue

        doc = load_document(report.mmd_path, max_chars=None)
        doc_tokens = estimate_tokens(doc.text) + estimate_tokens(SYSTEM_PROMPT)

        # Guard: never silently truncate. Too-long reports are recorded as skipped.
        if args.max_doc_tokens and doc_tokens > args.max_doc_tokens:
            for q in pending:
                writer.write(
                    build_record(
                        q,
                        report,
                        doc,
                        doc_tokens,
                        model=args.model,
                        query_mode=args.query_mode,
                        is_warmup=False,
                        result=None,
                        status="skipped_too_long",
                        error=f"doc_tokens_est {doc_tokens} > max_doc_tokens {args.max_doc_tokens}",
                    )
                )
                counters["skipped_too_long"] += 1
            pbar.update(len(queries))
            pbar.set_postfix(
                **{
                    k: counters[k]
                    for k in ("ok", "failed", "error", "skipped_too_long")
                }
            )
            continue

        # Already-done queries in a partially-resumed report.
        pbar.update(len(queries) - len(pending))

        # (b) warm-up: blocking, prefills + caches the document prefix.
        warm_rec = call_one(pending[0], report, doc, doc_tokens, is_warmup=True)
        writer.write(warm_rec)
        _bump(counters, warm_rec["status"])
        pbar.update(1)

        # (c) the rest: concurrent against the now-cached prefix.
        rest = pending[1:]
        if rest:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {
                    ex.submit(call_one, q, report, doc, doc_tokens, False): q
                    for q in rest
                }
                for fut in as_completed(futs):
                    rec = fut.result()
                    writer.write(rec)
                    _bump(counters, rec["status"])
                    pbar.update(1)
        pbar.set_postfix(
            ok=counters["ok"],
            fail=counters["failed"],
            err=counters["error"],
            skip=counters["skipped_too_long"],
            cached=warm_rec.get("cached_tokens"),
        )

    pbar.close()
    writer.close()
    elapsed = time.monotonic() - started

    meta = {
        "model": args.model,
        "base_url": args.base_url,
        "test_set": str(test_set_path),
        "query_mode": args.query_mode,
        "prompt_version": prompt_version(args.query_mode),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "retries": args.retries,
            "enable_thinking": args.enable_thinking,
            "reasoning_effort": args.reasoning_effort,
        },
        "max_doc_tokens": args.max_doc_tokens,
        "concurrency": args.concurrency,
        "n_reports": len(groups),
        "n_queries": n_queries,
        "counters": counters,
        "elapsed_s": round(elapsed, 1),
        "responses_path": str(responses_path),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    sys.stderr.write(
        f"\n[done] {n_queries} queries in {elapsed:.1f}s — ok={counters['ok']}, "
        f"failed={counters['failed']}, error={counters['error']}, "
        f"skipped_too_long={counters['skipped_too_long']}\n"
        f"[done] wrote {responses_path}\n"
    )


def _bump(counters: dict, status: str) -> None:
    counters[status if status in counters else "error"] = (
        counters.get(status if status in counters else "error", 0) + 1
    )


def _dry_run_report(groups, args, out_dir: Path) -> None:
    """Print the prefix-cache plan + estimated token savings, write a plan JSON."""
    sys_tokens = estimate_tokens(SYSTEM_PROMPT)
    suffix_probe = estimate_tokens(
        build_question_block(
            company_name="ACME Corporation",
            ticker="ACME",
            year=2020,
            kpi_label="Net income",
            kpi_definition="Net income attributable to parent.",
            unit_class="monetary",
            query_text="What was the net income in 2020?",
            query_mode=args.query_mode,
        )
    )
    rows = []
    total_doc_tokens = 0
    total_queries = 0
    n_skipped = 0
    for report, queries in groups:
        doc = load_document(report.mmd_path, max_chars=None)
        dtoks = estimate_tokens(doc.text) + sys_tokens
        nq = len(queries)
        too_long = bool(args.max_doc_tokens and dtoks > args.max_doc_tokens)
        if too_long:
            n_skipped += nq
        else:
            total_doc_tokens += dtoks
            total_queries += nq
        rows.append(
            {
                "report": report.name,
                "queries": nq,
                "prefix_tokens_est": dtoks,
                "too_long": too_long,
            }
        )

    # With prefix caching: 1 prefill per report. Without: 1 prefill per query.
    prefill_cached = total_doc_tokens  # sum over reports
    prefill_uncached = sum(
        r["prefix_tokens_est"] * r["queries"] for r in rows if not r["too_long"]
    )
    suffix_tokens = (suffix_probe + 8) * total_queries

    print("\n=== Prefix-cache plan (dry run, cl100k estimate) ===")
    print(
        f"reports: {len(rows)} | queries to run: {total_queries} "
        f"| skipped_too_long: {n_skipped}"
    )
    print(f"system prompt: ~{sys_tokens} tok | question suffix: ~{suffix_probe} tok")
    print(
        f"prompt version: {prompt_version(args.query_mode)} | query_mode: {args.query_mode}"
    )
    print("\nPrefill tokens (the expensive part):")
    print(f"  WITH prefix caching   (1 prefill/report): ~{prefill_cached:,.0f}")
    print(f"  WITHOUT prefix caching (1 prefill/query):  ~{prefill_uncached:,.0f}")
    if prefill_cached:
        print(
            f"  -> prefix caching saves ~{prefill_uncached - prefill_cached:,.0f} "
            f"prefill tokens ({prefill_uncached / max(prefill_cached, 1):.1f}x less prefill)"
        )
    print(f"suffix+decode prefill (varies/query): ~{suffix_tokens:,.0f}")
    longest = sorted(rows, key=lambda r: -r["prefix_tokens_est"])[:5]
    print("\nlargest reports:")
    for r in longest:
        flag = "  [SKIP too_long]" if r["too_long"] else ""
        print(
            f"  {r['report']:30s} {r['queries']:3d} q  ~{r['prefix_tokens_est']:>7,} tok{flag}"
        )
    plan = {
        "query_mode": args.query_mode,
        "prompt_version": prompt_version(args.query_mode),
        "system_tokens_est": sys_tokens,
        "suffix_tokens_est": suffix_probe,
        "n_reports": len(rows),
        "n_queries_to_run": total_queries,
        "n_skipped_too_long": n_skipped,
        "prefill_tokens_with_caching": prefill_cached,
        "prefill_tokens_without_caching": prefill_uncached,
        "reports": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dry_run_plan.json").write_text(json.dumps(plan, indent=2))
    print(f"\nwrote {out_dir / 'dry_run_plan.json'}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="vLLM model name.")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--test-set", type=Path, default=nd.DEFAULT_TEST_SET)
    p.add_argument(
        "--prototype",
        action="store_true",
        help="Use prototype_3_reports.csv instead of test_set.csv (smoke test).",
    )
    p.add_argument("--kpis-long", type=Path, default=nd.DEFAULT_KPIS_LONG)
    p.add_argument("--ocr-root", type=Path, default=nd.DEFAULT_OCR_ROOT)
    p.add_argument(
        "--output-dir", type=Path, default=None, help="Default: output/<model-slug>/."
    )
    p.add_argument(
        "--query-mode",
        choices=["defined", "plain"],
        default="defined",
        help="'defined' (default): include the canonical KPI definition + "
        "unit hint in each query. 'plain': informal NL question only.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="In-flight queries per report AFTER the warm-up (all share "
        "the cached prefix). Default 8.",
    )
    p.add_argument(
        "--max-doc-tokens",
        type=int,
        default=125_000,
        help="Skip (do not truncate) reports whose system+document "
        "exceeds this many cl100k tokens. 0 disables the guard.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Completion budget. The answer is a tiny JSON object.",
    )  # If no reasoning is impossible, we need more room for the thinking trace. Set this at 2048
    # gpt oss or Mistral.
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Enable thinking mode (Qwen3/Nemotron). "
        "Default: send no chat_template_kwargs at all.",
    )
    p.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Send enable_thinking=False — needed for Qwen3-family so it "
        "emits bare JSON under xgrammar. Do NOT use for gpt-oss / Mistral "
        "(their templates reject/ignore the kwarg); use --reasoning-effort instead.",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="gpt-oss (Harmony) reasoning effort. Use 'low' for needle lookups.",
    )
    p.add_argument(
        "--limit-reports",
        type=int,
        default=None,
        help="Process at most N reports (after grouping).",
    )
    p.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Process at most N queries per report.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip query_ids already terminal in responses.jsonl.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prefix-cache plan + token savings; no server calls.",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
