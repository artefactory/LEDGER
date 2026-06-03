"""Score the LLM KPI extractions against the ground-truth ``kpis_long.csv``.

Walks ``--output-dir/raw/*.json``, flattens each ``ExtractedKPI`` into a
prediction row, joins against ``kpis_long.csv`` on ``(ticker, year, kpi)``,
and emits four CSVs plus a markdown summary:

- ``predictions_long.csv``  — full joined view, one row per (ticker, year, kpi).
- ``per_kpi_metrics.csv``    — counts + recall/precision per KPI.
- ``per_year_metrics.csv``   — same, sliced by year.
- ``per_source_metrics.csv`` — same, sliced by ground-truth source (edgar / yfinance / alphavantage).
- ``summary.md``             — top-line table + worst-performing KPIs/tickers.

Status buckets per (ticker, year, kpi):

- ``matched``  — both present, |pred − gt| / max(|gt|, ε) ≤ tolerance.
- ``wrong``    — both present, outside tolerance.
- ``missing``  — ground truth has a value, LLM emitted nothing.
- ``extra``    — LLM emitted a value, ground truth has a gap (informational
  signal only — ground truth itself has known gaps especially for
  yfinance / alphavantage rows).
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
BENCHMARK_DIR = HERE.parent
KPI_ANALYSIS_DIR = BENCHMARK_DIR.parent

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(KPI_ANALYSIS_DIR))

from schema import ReportExtraction  # noqa: E402  (validates raw extraction objs)


DEFAULT_GROUND_TRUTH = KPI_ANALYSIS_DIR / "output" / "kpis_long.csv"
DEFAULT_OUTPUT_DIR = HERE / "output"


def load_ground_truth(csv_path: Path) -> dict[tuple[str, int, str], dict]:
    """Index ``kpis_long.csv`` by (ticker, year, kpi)."""
    out: dict[tuple[str, int, str], dict] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"].strip()
            try:
                year = int(row["year"])
                value = float(row["value"])
            except (ValueError, KeyError):
                continue
            kpi = row["kpi"].strip()
            out[(ticker, year, kpi)] = {
                "value": value,
                "source": row.get("source", "").strip(),
                "tag": row.get("tag", "").strip(),
                "company_name": row.get("company_name", "").strip(),
                "exchange": row.get("exchange", "").strip(),
                "industry": row.get("industry", "").strip(),
            }
    return out


def load_predictions(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load all raw JSONs. Returns (prediction_rows, run_records).

    Prediction rows: one per (ticker, year, kpi) emitted by the LLM. Each
    has ``ticker``, ``year``, ``kpi``, ``value``, ``reporting_currency``,
    ``model``, ``report_name``.

    Run records: one per file — used to detect missing reports vs failed
    reports. Each has ``ticker``, ``year``, ``status``, ``model``,
    ``report_name``.
    """
    preds: list[dict] = []
    runs: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[score] could not parse {path}: {e}\n")
            continue
        runs.append({
            "ticker": rec.get("ticker"),
            "year": rec.get("year"),
            "status": rec.get("status"),
            "model": rec.get("model"),
            "report_name": rec.get("report_name"),
        })
        if rec.get("status") != "ok":
            continue
        extr = rec.get("extraction") or {}
        try:
            obj = ReportExtraction.model_validate(extr)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(
                f"[score] invalid extraction shape in {path.name}: {e}\n"
            )
            continue
        for k in obj.kpis:
            preds.append({
                "ticker": rec.get("ticker"),
                "year": k.fiscal_year,
                "kpi": k.kpi,
                "pred_value": k.value,
                "reporting_currency": obj.reporting_currency,
                "model": rec.get("model"),
                "report_name": rec.get("report_name"),
                "report_year": rec.get("year"),
            })
    return preds, runs


def classify(
    gt_value: float | None,
    pred_value: float | None,
    *,
    tolerance: float,
    zero_eps: float,
) -> tuple[str, float | None]:
    """Bucket a (gt, pred) pair and compute the relative error.

    ``rel_error`` is None if either side is missing or if gt is zero (use
    absolute comparison instead).
    """
    if gt_value is None and pred_value is None:
        return "absent", None
    if gt_value is None:
        return "extra", None
    if pred_value is None:
        return "missing", None
    if abs(gt_value) < zero_eps:
        if abs(pred_value) < zero_eps:
            return "matched", 0.0
        return "wrong", None
    rel = (pred_value - gt_value) / abs(gt_value)
    if abs(rel) <= tolerance:
        return "matched", rel
    return "wrong", rel


def write_predictions_long(rows: list[dict], path: Path) -> None:
    fields = [
        "ticker",
        "year",
        "kpi",
        "status",
        "gt_value",
        "pred_value",
        "rel_error",
        "source",
        "tag",
        "exchange",
        "industry",
        "company_name",
        "reporting_currency",
        "model",
        "report_name",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate(
    rows: list[dict],
    group_key: str,
) -> list[dict]:
    """Group rows by ``group_key`` and compute matched / wrong / missing / extra counts.

    Recall = matched / (matched + wrong + missing).
    Precision = matched / (matched + wrong).
    Median |rel_error| over the matched∪wrong intersection.
    """
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = r.get(group_key)
        if key is None:
            continue
        by[str(key)].append(r)

    out: list[dict] = []
    for key, items in by.items():
        c = Counter(it["status"] for it in items)
        matched = c.get("matched", 0)
        wrong = c.get("wrong", 0)
        missing = c.get("missing", 0)
        extra = c.get("extra", 0)
        gt_total = matched + wrong + missing
        pred_total = matched + wrong + extra
        recall = matched / gt_total if gt_total else None
        precision = matched / (matched + wrong) if (matched + wrong) else None
        rel_errors = [
            abs(it["rel_error"]) for it in items
            if it.get("rel_error") is not None and it["status"] in ("matched", "wrong")
        ]
        med_rel = statistics.median(rel_errors) if rel_errors else None
        out.append({
            group_key: key,
            "n_gt": gt_total,
            "n_pred": pred_total,
            "matched": matched,
            "wrong": wrong,
            "missing": missing,
            "extra": extra,
            "recall": recall,
            "precision": precision,
            "median_abs_rel_error": med_rel,
        })
    return out


def write_metrics_csv(rows: list[dict], path: Path, group_key: str) -> None:
    if not rows:
        path.write_text(f"{group_key},n_gt,n_pred,matched,wrong,missing,extra,recall,precision,median_abs_rel_error\n")
        return
    rows = sorted(rows, key=lambda r: r[group_key])
    fields = [
        group_key,
        "n_gt",
        "n_pred",
        "matched",
        "wrong",
        "missing",
        "extra",
        "recall",
        "precision",
        "median_abs_rel_error",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                k: (
                    f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else v
                )
                for k, v in r.items()
            })


def render_summary(
    *,
    overall: dict,
    per_kpi: list[dict],
    per_year: list[dict],
    per_source: list[dict],
    runs: list[dict],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("# LLM KPI extraction benchmark — summary\n")
    lines.append(f"- Tolerance: ±{args.tolerance:.1%}")
    lines.append(f"- Ground truth: `{args.ground_truth}`")
    lines.append(f"- Predictions root: `{args.output_dir / 'raw'}`")
    lines.append(
        f"- Reports loaded: {len(runs)} "
        f"(ok={sum(1 for r in runs if r['status']=='ok')}, "
        f"failed={sum(1 for r in runs if r['status']=='failed')}, "
        f"error={sum(1 for r in runs if r['status']=='error')})"
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k, v in overall.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Per KPI")
    lines.append("")
    lines.append(_md_table(per_kpi, key="kpi"))
    lines.append("")
    lines.append("## Per year")
    lines.append("")
    lines.append(_md_table(per_year, key="year"))
    lines.append("")
    lines.append("## Per ground-truth source")
    lines.append("")
    lines.append(_md_table(per_source, key="source"))
    lines.append("")
    return "\n".join(lines)


def _md_table(rows: list[dict], *, key: str) -> str:
    if not rows:
        return "_(no data)_"
    rows = sorted(
        rows,
        key=lambda r: (r.get("recall") if r.get("recall") is not None else -1),
    )
    cols = [
        key, "n_gt", "matched", "wrong", "missing", "extra",
        "recall", "precision", "median_abs_rel_error",
    ]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        formatted = []
        for c in cols:
            v = r.get(c)
            if v is None:
                formatted.append("—")
            elif isinstance(v, float):
                formatted.append(f"{v:.3f}")
            else:
                formatted.append(str(v))
        out.append("| " + " | ".join(formatted) + " |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    p.add_argument(
        "--model",
        default=None,
        help="If given, scores output/<model-slug>/. Mutually compatible with "
        "--output-dir; --output-dir wins if both are passed.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="Match if |pred-gt| / |gt| <= tolerance. Default 1%%.")
    p.add_argument("--zero-eps", type=float, default=0.5,
                   help="Absolute eps when ground-truth value is 0 "
                        "(e.g. zero dividends). Default 0.5 single units.")
    args = p.parse_args()

    if args.output_dir is None:
        if args.model is None:
            sys.stderr.write(
                "[score] pass --model <id> or --output-dir <path>\n"
            )
            sys.exit(2)
        sys.path.insert(0, str(HERE))
        from run_benchmark import model_slug  # noqa: E402
        args.output_dir = DEFAULT_OUTPUT_DIR / model_slug(args.model)

    raw_dir = args.output_dir / "raw"
    if not raw_dir.is_dir():
        sys.stderr.write(f"[score] {raw_dir} does not exist — run run_benchmark.py first\n")
        sys.exit(1)

    gt = load_ground_truth(args.ground_truth)
    preds, runs = load_predictions(raw_dir)
    sys.stderr.write(
        f"[score] loaded {len(preds)} prediction rows from {len(runs)} reports, "
        f"{len(gt)} ground-truth (ticker, year, kpi) cells\n"
    )

    # Build the joined view. Iterate over the union of (ticker, year, kpi)
    # keys appearing in either pred set or in the gt set restricted to the
    # tickers/years we actually ran.
    ran_pairs: set[tuple[str, int]] = {
        (r["ticker"], r["year"]) for r in runs if r["status"] == "ok"
        and r["ticker"] is not None and r["year"] is not None
    }

    pred_index: dict[tuple[str, int, str], dict] = {}
    for pr in preds:
        key = (pr["ticker"], pr["year"], pr["kpi"])
        # If the LLM emitted the same KPI twice for the same year (rare
        # under xgrammar but defensively handled), keep the first.
        pred_index.setdefault(key, pr)

    keys: set[tuple[str, int, str]] = set(pred_index.keys())
    for (ticker, year, kpi), gt_row in gt.items():
        if (ticker, year) in ran_pairs:
            keys.add((ticker, year, kpi))

    rows: list[dict] = []
    for ticker, year, kpi in sorted(keys):
        pr = pred_index.get((ticker, year, kpi))
        gt_row = gt.get((ticker, year, kpi))
        gt_value = gt_row["value"] if gt_row else None
        pred_value = pr["pred_value"] if pr else None
        status, rel = classify(
            gt_value,
            pred_value,
            tolerance=args.tolerance,
            zero_eps=args.zero_eps,
        )
        rows.append({
            "ticker": ticker,
            "year": year,
            "kpi": kpi,
            "status": status,
            "gt_value": gt_value,
            "pred_value": pred_value,
            "rel_error": rel,
            "source": gt_row.get("source") if gt_row else None,
            "tag": gt_row.get("tag") if gt_row else None,
            "exchange": gt_row.get("exchange") if gt_row else None,
            "industry": gt_row.get("industry") if gt_row else None,
            "company_name": gt_row.get("company_name") if gt_row else None,
            "reporting_currency": pr.get("reporting_currency") if pr else None,
            "model": pr.get("model") if pr else None,
            "report_name": pr.get("report_name") if pr else None,
        })

    write_predictions_long(rows, args.output_dir / "predictions_long.csv")

    per_kpi = aggregate(rows, "kpi")
    per_year = aggregate(rows, "year")
    per_source = aggregate(rows, "source")

    write_metrics_csv(per_kpi, args.output_dir / "per_kpi_metrics.csv", "kpi")
    write_metrics_csv(per_year, args.output_dir / "per_year_metrics.csv", "year")
    write_metrics_csv(per_source, args.output_dir / "per_source_metrics.csv", "source")

    # Overall scalar metrics
    counter = Counter(r["status"] for r in rows)
    matched = counter.get("matched", 0)
    wrong = counter.get("wrong", 0)
    missing = counter.get("missing", 0)
    extra = counter.get("extra", 0)
    gt_total = matched + wrong + missing
    pred_total = matched + wrong + extra
    overall = {
        "n_predictions": pred_total,
        "n_ground_truth": gt_total,
        "matched": matched,
        "wrong": wrong,
        "missing": missing,
        "extra": extra,
        "recall (matched/gt)": (matched / gt_total) if gt_total else float("nan"),
        "precision (matched/(matched+wrong))": (matched / (matched + wrong))
            if (matched + wrong) else float("nan"),
    }

    summary_text = render_summary(
        overall=overall,
        per_kpi=per_kpi,
        per_year=per_year,
        per_source=per_source,
        runs=runs,
        args=args,
    )
    (args.output_dir / "summary.md").write_text(summary_text)

    sys.stderr.write(
        f"[score] wrote predictions_long.csv, per_kpi/year/source_metrics.csv, summary.md\n"
        f"[score] overall: matched={matched}, wrong={wrong}, missing={missing}, extra={extra}\n"
    )


if __name__ == "__main__":
    main()
