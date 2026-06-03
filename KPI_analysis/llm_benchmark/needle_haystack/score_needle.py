"""Score the needle-in-a-haystack benchmark outputs against ground truth.

Reads ``responses.jsonl`` (produced by ``run_needle.py``), joins each answer to
its ground-truth value in ``kpis_long.csv`` by ``query_id``, and classifies the
outcome of every query:

  matched      both present, |pred - gt| / |gt| <= tolerance.
  wrong        both present, outside tolerance.
  not_found    model answered found=false / value=null (an abstention).
  no_response  the call failed / errored (no valid JSON answer).
  skipped      report exceeded --max-doc-tokens and was not run.

For ``wrong`` rows it assigns a diagnostic bucket — the recurring,
*systematic* failure modes of this task, so a low score can be read rather than
just observed:

  year_shift    pred equals THIS metric's value for a neighbouring fiscal year
                (the model read the wrong column of a multi-year statement).
  sign_error    pred ~= -gt (sign / parenthesis convention).
  scale_error   pred ~= gt x 10^(+-3/6/9) (mis-applied the in-thousands/
                millions/billions scaling).
  scope_factor  |pred/gt| in [0.5, 2] and not matched (likely picked a related
                but differently-scoped line — e.g. total cost of sales vs COGS).
  other         none of the above.

Headline metrics (over the queries that produced a usable response, i.e.
matched + wrong + not_found):

  accuracy             matched / eval
  accuracy_strict      matched within 0.05% / eval  (rewards exact transcription)
  attempt_rate         (matched + wrong) / eval      (1 - abstention rate)
  precision_when_found matched / (matched + wrong)

Outputs (under the same dir as responses.jsonl, or --output-dir):
  scored.csv, per_kpi.csv, per_year.csv, per_source.csv, per_unit_class.csv,
  summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import needle_data as nd  # noqa: E402

DEFAULT_OUTPUT_BASE = HERE / "output"

# Outcomes that count toward the evaluation denominator (a usable response was
# produced — the model either answered or correctly abstained).
EVAL_OUTCOMES = ("matched", "wrong", "not_found")
SCALE_FACTORS = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)


def model_slug(model: str) -> str:
    safe = []
    for ch in model:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        elif ch == "/":
            safe.append("__")
        else:
            safe.append(ch if ch.isalnum() else "_")
    return "".join(safe).strip("_") or "model"


# ---------------------------------------------------------------------------
# Load predictions (last write wins per query_id)
# ---------------------------------------------------------------------------


def load_responses(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[score] skipping bad jsonl line: {e}\n")
                continue
            qid = rec.get("query_id")
            if qid:
                out[qid] = rec  # last occurrence wins
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def within(pred: float, target: float, tol: float, zero_eps: float) -> bool:
    if abs(target) < zero_eps:
        return abs(pred) < zero_eps
    return abs(pred - target) / abs(target) <= tol


def classify_outcome(rec: dict, gt_value: float | None) -> tuple[str, float | None, float | None]:
    """Return (outcome, rel_error, ratio) for one prediction record.

    ``rel_error`` and ``ratio`` are vs the requested year's ground truth and are
    None when not applicable.
    """
    status = rec.get("status")
    if status == "skipped_too_long":
        return "skipped", None, None
    if status in ("failed", "error"):
        return "no_response", None, None
    # status == ok
    found = rec.get("found")
    pred = rec.get("value")
    if not found or pred is None:
        return "not_found", None, None
    if gt_value is None:
        # We only run queries that have ground truth, so this is unexpected.
        return "wrong", None, None
    ratio = (pred / gt_value) if gt_value != 0 else None
    rel = (pred - gt_value) / abs(gt_value) if gt_value != 0 else None
    return ("matched" if within(pred, gt_value, _TOL, _ZERO_EPS) else "wrong"), rel, ratio


def wrong_bucket(
    rec: dict,
    gt: nd.GroundTruth,
    gt_index: dict[str, nd.GroundTruth],
    ratio: float | None,
) -> str:
    pred = rec.get("value")
    if pred is None:
        return "other"
    # 1) year shift: matches this metric's value for an adjacent fiscal year.
    for k in (-1, 1, -2, 2):
        neigh = gt_index.get(f"{gt.ticker}_{gt.kpi}_{gt.year + k}")
        if neigh is not None and within(pred, neigh.value, _TOL, _ZERO_EPS):
            return f"year_shift({k:+d})"
    # 2) sign error.
    if gt.value != 0 and within(pred, -gt.value, _TOL, _ZERO_EPS):
        return "sign_error"
    # 3) scale error (mis-applied thousands/millions/billions).
    if gt.value != 0:
        for fac in SCALE_FACTORS:
            if within(pred, gt.value * fac, _TOL, _ZERO_EPS):
                exp = int(round(math.log10(fac)))
                return f"scale_error(x1e{exp:+d})"
    # 4) related-but-different-scope line.
    if ratio is not None and 0.5 <= abs(ratio) <= 2.0:
        return "scope_factor"
    return "other"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(rows: list[dict], key: str) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is None or v == "":
            continue
        by[str(v)].append(r)
    out: list[dict] = []
    for k, items in by.items():
        out.append(_metrics_row(items) | {key: k})
    return out


def _metrics_row(items: list[dict]) -> dict:
    c = Counter(it["outcome"] for it in items)
    matched = c.get("matched", 0)
    wrong = c.get("wrong", 0)
    not_found = c.get("not_found", 0)
    no_response = c.get("no_response", 0)
    skipped = c.get("skipped", 0)
    eval_n = matched + wrong + not_found
    strict = sum(1 for it in items if it.get("matched_strict"))
    rels = [abs(it["rel_error"]) for it in items
            if it.get("rel_error") is not None and it["outcome"] in ("matched", "wrong")]
    return {
        "n": len(items),
        "eval_n": eval_n,
        "matched": matched,
        "wrong": wrong,
        "not_found": not_found,
        "no_response": no_response,
        "skipped": skipped,
        "accuracy": (matched / eval_n) if eval_n else None,
        "accuracy_strict": (strict / eval_n) if eval_n else None,
        "attempt_rate": ((matched + wrong) / eval_n) if eval_n else None,
        "precision_when_found": (matched / (matched + wrong)) if (matched + wrong) else None,
        "median_abs_rel_error": statistics.median(rels) if rels else None,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

SCORED_FIELDS = [
    "query_id", "ticker", "kpi", "year", "unit_class", "outcome", "wrong_bucket",
    "matched_strict", "gt_value", "pred_value", "value_verbatim", "unit_scale",
    "page", "rel_error", "ratio", "source", "industry", "exchange", "company_name",
    "model", "report_name", "n_pages", "doc_tokens_est", "latency_s",
    "prompt_tokens", "completion_tokens", "cached_tokens", "attempts", "run_status",
]


def write_scored(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORED_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


METRIC_COLS = [
    "n", "eval_n", "matched", "wrong", "not_found", "no_response", "skipped",
    "accuracy", "accuracy_strict", "attempt_rate", "precision_when_found",
    "median_abs_rel_error",
]


def write_metrics(rows: list[dict], path: Path, key: str) -> None:
    rows = sorted(rows, key=lambda r: (r.get("accuracy") if r.get("accuracy") is not None else -1))
    fields = [key] + METRIC_COLS
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def _fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.4f}"
    return "" if v is None else v


def _md_table(rows: list[dict], key: str) -> str:
    if not rows:
        return "_(no data)_"
    rows = sorted(rows, key=lambda r: (r.get("accuracy") if r.get("accuracy") is not None else -1))
    cols = [key, "n", "eval_n", "matched", "wrong", "not_found", "accuracy",
            "accuracy_strict", "precision_when_found", "median_abs_rel_error"]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_summary(*, model, overall, wrong_buckets, per_kpi, per_year, per_source,
                   per_unit, n_total, args) -> str:
    L: list[str] = []
    L.append(f"# Needle-in-a-haystack KPI benchmark — {model}\n")
    L.append(f"- Responses: `{args.responses}`")
    L.append(f"- Ground truth: `{args.kpis_long}`")
    L.append(f"- Match tolerance: ±{args.tolerance:.2%} (strict: ±{args.strict_tolerance:.3%})")
    L.append(f"- Queries scored: {n_total}")
    L.append("")
    L.append("## Headline\n")
    L.append("| metric | value |")
    L.append("| --- | --- |")
    for k, v in overall.items():
        L.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    L.append("")
    L.append("## Wrong-answer diagnostics\n")
    L.append("How the `wrong` answers break down (systematic failure modes):\n")
    if wrong_buckets:
        L.append("| bucket | count |")
        L.append("| --- | --- |")
        for b, n in wrong_buckets.most_common():
            L.append(f"| {b} | {n} |")
    else:
        L.append("_(no wrong answers)_")
    L.append("")
    L.append("## Per KPI\n")
    L.append(_md_table(per_kpi, "kpi"))
    L.append("")
    L.append("## Per fiscal year\n")
    L.append(_md_table(per_year, "year"))
    L.append("")
    L.append("## Per ground-truth source\n")
    L.append(_md_table(per_source, "source"))
    L.append("")
    L.append("## Per unit class\n")
    L.append(_md_table(per_unit, "unit_class"))
    L.append("")
    return "\n".join(L)


# Module-level tolerances (set in main, read by classify helpers).
_TOL = 0.01
_ZERO_EPS = 0.5
_STRICT_TOL = 0.0005


def main() -> None:
    global _TOL, _ZERO_EPS, _STRICT_TOL
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="Score output/<model-slug>/.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Dir containing responses.jsonl. Overrides --model.")
    p.add_argument("--responses", type=Path, default=None,
                   help="Explicit path to responses.jsonl. Overrides the above.")
    p.add_argument("--kpis-long", type=Path, default=nd.DEFAULT_KPIS_LONG)
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="Relative match tolerance. Default 1%%.")
    p.add_argument("--strict-tolerance", type=float, default=0.0005,
                   help="Tolerance for the 'exact' secondary metric. Default 0.05%%.")
    p.add_argument("--zero-eps", type=float, default=0.5,
                   help="Absolute eps when ground truth is ~0.")
    args = p.parse_args()

    if args.responses is not None:
        responses_path = args.responses
        out_dir = args.output_dir or responses_path.parent
    else:
        if args.output_dir is not None:
            out_dir = args.output_dir
        elif args.model is not None:
            out_dir = DEFAULT_OUTPUT_BASE / model_slug(args.model)
        else:
            sys.stderr.write("[score] pass --responses, --output-dir, or --model\n")
            sys.exit(2)
        responses_path = out_dir / "responses.jsonl"
    args.responses = responses_path

    if not responses_path.is_file():
        sys.stderr.write(f"[score] {responses_path} not found — run run_needle.py first\n")
        sys.exit(1)

    _TOL = args.tolerance
    _ZERO_EPS = args.zero_eps
    _STRICT_TOL = args.strict_tolerance

    preds = load_responses(responses_path)
    gt_index = nd.load_ground_truth(args.kpis_long)
    sys.stderr.write(f"[score] {len(preds)} predictions, {len(gt_index)} ground-truth cells\n")

    rows: list[dict] = []
    for qid, rec in preds.items():
        gt = gt_index.get(qid)
        gt_value = gt.value if gt else None
        outcome, rel, ratio = classify_outcome(rec, gt_value)
        bucket = ""
        if outcome == "wrong" and gt is not None:
            bucket = wrong_bucket(rec, gt, gt_index, ratio)
        matched_strict = bool(
            outcome in ("matched", "wrong")
            and gt_value is not None
            and rec.get("value") is not None
            and within(rec["value"], gt_value, _STRICT_TOL, _ZERO_EPS)
        )
        rows.append({
            "query_id": qid,
            "ticker": rec.get("ticker") or (gt.ticker if gt else None),
            "kpi": rec.get("kpi") or (gt.kpi if gt else None),
            "year": rec.get("year") or (gt.year if gt else None),
            "unit_class": rec.get("unit_class") or (nd.KPI_UNIT_CLASS.get(gt.kpi) if gt else None),
            "outcome": outcome,
            "wrong_bucket": bucket,
            "matched_strict": matched_strict,
            "gt_value": gt_value,
            "pred_value": rec.get("value"),
            "value_verbatim": rec.get("value_verbatim"),
            "unit_scale": rec.get("unit_scale"),
            "page": rec.get("page"),
            "rel_error": rel,
            "ratio": ratio,
            "source": gt.source if gt else None,
            "industry": gt.industry if gt else None,
            "exchange": gt.exchange if gt else None,
            "company_name": gt.company_name if gt else None,
            "model": rec.get("model"),
            "report_name": rec.get("report_name"),
            "n_pages": rec.get("n_pages"),
            "doc_tokens_est": rec.get("doc_tokens_est"),
            "latency_s": rec.get("latency_s"),
            "prompt_tokens": rec.get("prompt_tokens"),
            "completion_tokens": rec.get("completion_tokens"),
            "cached_tokens": rec.get("cached_tokens"),
            "attempts": rec.get("attempts"),
            "run_status": rec.get("status"),
        })

    write_scored(rows, out_dir / "scored.csv")
    per_kpi = aggregate(rows, "kpi")
    per_year = aggregate(rows, "year")
    per_source = aggregate(rows, "source")
    per_unit = aggregate(rows, "unit_class")
    write_metrics(per_kpi, out_dir / "per_kpi.csv", "kpi")
    write_metrics(per_year, out_dir / "per_year.csv", "year")
    write_metrics(per_source, out_dir / "per_source.csv", "source")
    write_metrics(per_unit, out_dir / "per_unit_class.csv", "unit_class")

    overall_metrics = _metrics_row(rows)
    c = Counter(r["outcome"] for r in rows)
    overall = {
        "queries_scored": len(rows),
        "eval_n (matched+wrong+not_found)": overall_metrics["eval_n"],
        "matched": c.get("matched", 0),
        "wrong": c.get("wrong", 0),
        "not_found": c.get("not_found", 0),
        "no_response": c.get("no_response", 0),
        "skipped": c.get("skipped", 0),
        "accuracy": overall_metrics["accuracy"] if overall_metrics["accuracy"] is not None else float("nan"),
        "accuracy_strict": overall_metrics["accuracy_strict"] if overall_metrics["accuracy_strict"] is not None else float("nan"),
        "attempt_rate": overall_metrics["attempt_rate"] if overall_metrics["attempt_rate"] is not None else float("nan"),
        "precision_when_found": overall_metrics["precision_when_found"] if overall_metrics["precision_when_found"] is not None else float("nan"),
        "median_abs_rel_error": overall_metrics["median_abs_rel_error"] if overall_metrics["median_abs_rel_error"] is not None else float("nan"),
    }
    wrong_buckets = Counter(r["wrong_bucket"] for r in rows if r["outcome"] == "wrong" and r["wrong_bucket"])

    model_name = rows[0]["model"] if rows else (args.model or "unknown")
    summary = render_summary(
        model=model_name, overall=overall, wrong_buckets=wrong_buckets,
        per_kpi=per_kpi, per_year=per_year, per_source=per_source, per_unit=per_unit,
        n_total=len(rows), args=args,
    )
    (out_dir / "summary.md").write_text(summary)

    sys.stderr.write(
        f"[score] wrote scored.csv, per_kpi/year/source/unit_class.csv, summary.md to {out_dir}\n"
        f"[score] accuracy={overall['accuracy']:.4f}  strict={overall['accuracy_strict']:.4f}  "
        f"(matched={overall['matched']}, wrong={overall['wrong']}, "
        f"not_found={overall['not_found']}, no_response={overall['no_response']}, "
        f"skipped={overall['skipped']})\n"
    )
    if wrong_buckets:
        sys.stderr.write("[score] wrong buckets: "
                         + ", ".join(f"{b}={n}" for b, n in wrong_buckets.most_common()) + "\n")


if __name__ == "__main__":
    main()
