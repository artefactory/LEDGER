"""Main orchestrator: run the LLM KPI extraction benchmark.

For each (ticker, year) that has both an OCR'd report and ground-truth KPIs
in ``kpis_long.csv``, this script:

1. Loads the ``.mmd`` and renders it with ``[Page N]`` markers.
2. Calls the LLM (OpenAI-compatible, vLLM-served) with the schema-constrained
   prompt from ``prompts.py``.
3. Validates the response against the ``ReportExtraction`` Pydantic model.
4. Writes a per-report JSON record to ``--output-dir/raw/{TICKER}_{YEAR}.json``.

Default OCR root is ``DeepSeekOCR_Ardian_pruned_1k/`` (~994 reports, ~980
overlap with ``kpis_long.csv``). Override with ``--root``.

Smoke-test pattern:
    uv run python KPI_analysis/llm_benchmark/run_benchmark.py \\
        --model Qwen/Qwen2.5-72B-Instruct --limit 8

Scoring is a separate step — see ``score_benchmark.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
KPI_ANALYSIS_DIR = HERE.parent
REPO_ROOT = KPI_ANALYSIS_DIR.parent

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KPI_ANALYSIS_DIR))

from client import call_extraction, make_client  # noqa: E402
from document import LoadedDocument, ReportInfo, discover_reports, load_document  # noqa: E402
from prompts import build_messages  # noqa: E402


DEFAULT_OCR_ROOT = REPO_ROOT / "DeepSeekOCR_Ardian_pruned_1k"
DEFAULT_GROUND_TRUTH = KPI_ANALYSIS_DIR / "output" / "kpis_long.csv"
DEFAULT_OUTPUT_BASE = HERE / "output"
DEFAULT_IS_10K = REPO_ROOT / "doc_text_processing" / "10K_or_not" / "is_10k.txt"


def model_slug(model: str) -> str:
    """Filesystem-safe slug for a model id (e.g. ``openai/gpt-oss-20b`` ->
    ``openai__gpt-oss-20b``)."""
    safe = []
    for ch in model:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        elif ch == "/":
            safe.append("__")
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "model"


def load_ground_truth_pairs(csv_path: Path) -> set[tuple[str, int]]:
    """Return the set of (ticker, year) pairs present in ``kpis_long.csv``."""
    pairs: set[tuple[str, int]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip()
            try:
                year = int(row.get("year", "").strip())
            except ValueError:
                continue
            if ticker:
                pairs.add((ticker, year))
    return pairs


def load_us_10k_dirnames(path: Path) -> set[str]:
    """Read the list of report directory names flagged as US 10-Ks."""
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_report_names(path: Path) -> set[str]:
    """Read a newline-delimited list of report directory names to keep.

    One ``{EXCHANGE}_{TICKER}_{YEAR}`` name per line (e.g. ``NASDAQ_CAAS_2017``,
    matching ``ReportInfo.name``). Blank lines and ``#`` comments are skipped.
    """
    if not path.is_file():
        raise FileNotFoundError(f"--reports-file not found: {path}")
    names: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(line)
    return names


def write_record(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    tmp.replace(out_path)


def existing_ok(out_path: Path) -> bool:
    if not out_path.is_file():
        return False
    try:
        return json.loads(out_path.read_text()).get("status") == "ok"
    except Exception:
        return False


def run_one(
    report: ReportInfo,
    *,
    raw_dir: Path,
    client,
    model: str,
    max_chars: int | None,
    few_shot: bool,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool | None,
    reasoning_effort: str | None,
    retries: int,
) -> dict:
    out_path = raw_dir / f"{report.exchange}_{report.ticker}_{report.year}.json"
    started = time.monotonic()
    try:
        doc: LoadedDocument = load_document(report.mmd_path, max_chars=max_chars)
        messages = build_messages(doc.text, ticker=report.ticker, few_shot=few_shot)
        result = call_extraction(
            client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            retries=retries,
        )
    except Exception as e:  # noqa: BLE001 — we want any error captured
        record = {
            "ticker": report.ticker,
            "year": report.year,
            "exchange": report.exchange,
            "report_name": report.name,
            "mmd_path": str(report.mmd_path),
            "model": model,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": time.monotonic() - started,
        }
        write_record(out_path, record)
        return record

    status = "ok" if result.extraction is not None else "failed"
    record = {
        "ticker": report.ticker,
        "year": report.year,
        "exchange": report.exchange,
        "report_name": report.name,
        "mmd_path": str(report.mmd_path),
        "model": model,
        "status": status,
        "attempts": result.attempts,
        "latency_s": result.latency_s,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "n_pages": doc.n_pages,
        "n_pages_kept": doc.n_pages_kept,
        "n_chars": doc.n_chars,
        "truncated": doc.truncated,
        "few_shot": few_shot,
        "extraction": result.extraction.model_dump() if result.extraction else None,
        "error": result.error,
        "raw_response": result.raw_response if status != "ok" else None,
        "elapsed_s": time.monotonic() - started,
    }
    write_record(out_path, record)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_OCR_ROOT,
        help="OCR'd reports root directory.",
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Path to kpis_long.csv (defines which reports we benchmark).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: "
        "KPI_analysis/llm_benchmark/output/<model-slug>/ — keeps per-model "
        "results separate so --resume across models is safe.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="vLLM model name, e.g. Qwen/Qwen3.5-27B-FP8.",
    )
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N reports (sorted by ticker, year). "
        "Default: process all.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Parallel in-flight LLM requests. Default 2 — long "
        "contexts dominate KV cache.",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=380_000,
        help="Soft cap on rendered document length (chars). "
        "Translates to roughly 95k tokens at chars/4 — sized for "
        "a 128k-context model with headroom. Set to 0 to disable.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="LLM completion max_tokens. Default 4096 — generous "
        "for the slim values-only schema.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Enable thinking mode for templates that support it "
        "(Qwen3, Nemotron Nano 3). Default: do not send the kwarg.",
    )
    p.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Explicitly disable thinking mode (sends enable_thinking=False).",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort for gpt-oss (Harmony template). "
        "Default: do not send the kwarg.",
    )
    p.add_argument(
        "--few-shot",
        action="store_true",
        help="Prepend a snippet-level few-shot pair to the messages.",
    )
    p.add_argument(
        "--us-only",
        action="store_true",
        help="Restrict to reports flagged as US 10-Ks via "
        "doc_text_processing/10K_or_not/is_10k.txt.",
    )
    p.add_argument("--is-10k-list", type=Path, default=DEFAULT_IS_10K)
    p.add_argument(
        "--reports-file",
        type=Path,
        default=None,
        help="Restrict to the report directory names listed in this file "
        "(one {EXCHANGE}_{TICKER}_{YEAR} per line, e.g. "
        "needle_haystack/test_set_reports.txt). Blank lines and '#' comments "
        "are skipped. Applied before the ground-truth overlap filter.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip reports whose output JSON already exists with status=ok.",
    )
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_BASE / model_slug(args.model)
        sys.stderr.write(f"[setup] output_dir defaulted to {args.output_dir}\n")

    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    gt_pairs = load_ground_truth_pairs(args.ground_truth)
    sys.stderr.write(f"[setup] {len(gt_pairs)} (ticker, year) pairs in ground truth\n")

    reports = discover_reports(args.root)
    sys.stderr.write(
        f"[setup] {len(reports)} OCR'd reports discovered under {args.root}\n"
    )

    if args.reports_file is not None:
        wanted = load_report_names(args.reports_file)
        discovered_names = {r.name for r in reports}
        reports = [r for r in reports if r.name in wanted]
        missing = sorted(wanted - discovered_names)
        sys.stderr.write(
            f"[setup] {len(reports)}/{len(wanted)} reports from "
            f"{args.reports_file.name} found under {args.root}\n"
        )
        if missing:
            sys.stderr.write(
                f"[setup] WARNING: {len(missing)} listed report(s) not found "
                f"in the OCR tree: {', '.join(missing[:10])}"
                f"{' …' if len(missing) > 10 else ''}\n"
            )

    reports = [r for r in reports if (r.ticker, r.year) in gt_pairs]
    sys.stderr.write(f"[setup] {len(reports)} reports overlap with ground truth\n")

    if args.us_only:
        us_set = load_us_10k_dirnames(args.is_10k_list)
        reports = [r for r in reports if r.name in us_set]
        sys.stderr.write(f"[setup] {len(reports)} reports after --us-only filter\n")

    reports.sort(key=lambda r: (r.ticker, r.year))
    if args.limit is not None:
        reports = reports[: args.limit]
        sys.stderr.write(f"[setup] limiting to first {len(reports)} after --limit\n")

    if args.resume:
        before = len(reports)
        reports = [
            r
            for r in reports
            if not existing_ok(raw_dir / f"{r.exchange}_{r.ticker}_{r.year}.json")
        ]
        sys.stderr.write(
            f"[setup] --resume: {before - len(reports)} already done, "
            f"{len(reports)} to run\n"
        )

    if not reports:
        sys.stderr.write("[setup] nothing to do\n")
        return

    client = make_client(args.base_url, args.api_key)
    max_chars = args.max_chars if args.max_chars > 0 else None

    started = time.monotonic()
    sys.stderr.write(
        f"[run] {len(reports)} reports, model={args.model}, "
        f"concurrency={args.concurrency}, max_chars={max_chars}\n"
    )

    n_ok = 0
    n_failed = 0
    n_error = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        fut_to_report = {
            ex.submit(
                run_one,
                r,
                raw_dir=raw_dir,
                client=client,
                model=args.model,
                max_chars=max_chars,
                few_shot=args.few_shot,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
                reasoning_effort=args.reasoning_effort,
                retries=args.retries,
            ): r
            for r in reports
        }
        pbar = tqdm(
            as_completed(fut_to_report),
            total=len(reports),
            desc=args.model,
            unit="report",
        )
        for fut in pbar:
            r = fut_to_report[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                n_error += 1
                pbar.write(f"EXC {r.name}: {type(e).__name__}: {e}")
                continue
            status = rec.get("status")
            if status == "ok":
                n_ok += 1
                tag = "OK"
            elif status == "failed":
                n_failed += 1
                tag = "FAIL"
            else:
                n_error += 1
                tag = "ERR"
            pbar.set_postfix(ok=n_ok, fail=n_failed, err=n_error)
            pbar.write(
                f"{tag} {r.name} "
                f"({rec.get('latency_s', 0):.1f}s, "
                f"in={rec.get('prompt_tokens')}, out={rec.get('completion_tokens')}, "
                f"trunc={rec.get('truncated')})"
            )

    elapsed = time.monotonic() - started
    sys.stderr.write(
        f"\n[done] {len(reports)} reports in {elapsed:.1f}s — "
        f"ok={n_ok}, failed={n_failed}, error={n_error}\n"
    )

    # Write a small run-meta summary alongside the raw outputs.
    meta = {
        "model": args.model,
        "base_url": args.base_url,
        "few_shot": args.few_shot,
        "us_only": args.us_only,
        "reports_file": str(args.reports_file) if args.reports_file else None,
        "max_chars": max_chars,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "retries": args.retries,
        "n_reports": len(reports),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_error": n_error,
        "elapsed_s": elapsed,
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
