"""Compute agreement between human and LLM qrels annotations.

Loads a human annotation session (from annotation_qrels/sessions/) and the
LLM audit CSV (from KPI_analysis/output/qrels/annotations_audit.csv), joins
them on item_id, and computes agreement metrics.

Metrics reported:
- Cohen's kappa (unweighted, linear-weighted, quadratic-weighted)
- Overall percent agreement
- Per-grade confusion matrix (human rows x LLM columns)
- Per-KPI and per-match-type breakdowns
- Flag rate: items where human grade < 2 but LLM gave 2 (or vice versa)

Usage::

    uv run python annotation_qrels/compute_agreement.py \\
        --session-id <session_id> \\
        --llm-audit KPI_analysis/output/qrels/annotations_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.metrics import (
        cohen_kappa_score,
        confusion_matrix,
    )
except ImportError:
    raise SystemExit("scikit-learn is required. Run: uv add scikit-learn")

HERE = Path(__file__).resolve().parent
DEFAULT_SESSIONS_DIR = HERE / "sessions"
DEFAULT_LLM_AUDIT = HERE.parent / "KPI_analysis" / "output" / "qrels" / "annotations_audit.csv"
DEFAULT_OUTPUT_DIR = HERE.parent / "KPI_analysis" / "output" / "qrels"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute human-LLM agreement on qrels annotations."
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Annotation session ID (directory name under sessions/).",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=DEFAULT_SESSIONS_DIR,
    )
    parser.add_argument(
        "--llm-audit",
        type=Path,
        default=DEFAULT_LLM_AUDIT,
        help="Path to annotations_audit.csv from llm_annotate_qrels.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <session-dir>/agreement/",
    )
    parser.add_argument(
        "--min-grade",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Only include items where at least one annotator gave grade >= this value. Default: 0 (all).",
    )
    return parser


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_human_annotations(session_dir: Path) -> dict[str, dict[str, Any]]:
    """Load current_annotations.json from a session directory."""
    current = load_json(session_dir / "current_annotations.json", default={})
    if not isinstance(current, dict):
        raise ValueError(f"invalid current_annotations.json in {session_dir}")
    return current


def load_manifest_items(session_dir: Path) -> dict[str, dict[str, Any]]:
    """Load manifest.json and index items by item_id."""
    manifest = load_json(session_dir / "manifest.json", default={}) or {}
    items = manifest.get("items", [])
    return {item["item_id"]: item for item in items}


def load_llm_annotations(audit_csv: Path) -> dict[str, int]:
    """Load LLM audit CSV and return {item_id: llm_grade}."""
    llm: dict[str, int] = {}
    with audit_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row["query_id"]
            doc_id = row["doc_id"]
            raw_grade = row.get("llm_grade", "").strip()
            if not raw_grade:
                continue
            try:
                grade = int(raw_grade)
            except ValueError:
                continue
            # Construct the same item_id as qrels_index.py
            item_id = f"{query_id}__{doc_id.replace('/', '_')}"
            llm[item_id] = grade
    return llm


def compute_weighted_kappa(
    human: np.ndarray, llm: np.ndarray, weights: str
) -> float:
    """Compute weighted Cohen's kappa using scikit-learn."""
    return float(cohen_kappa_score(human, llm, weights=weights))


def grade_label(grade: int) -> str:
    return {0: "0-NotRelevant", 1: "1-Contextual", 2: "2-Primary"}.get(grade, str(grade))


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    session_dir = args.sessions_dir / args.session_id
    if not session_dir.is_dir():
        raise FileNotFoundError(f"session directory not found: {session_dir}")

    # Load data
    human_ann = load_human_annotations(session_dir)
    manifest_items = load_manifest_items(session_dir)
    llm_ann = load_llm_annotations(args.llm_audit)

    print(f"Human annotations: {len(human_ann)}")
    print(f"LLM annotations: {len(llm_ann)}")
    print(f"Manifest items: {len(manifest_items)}")

    # Build paired dataset (only items annotated by both)
    paired: list[dict[str, Any]] = []
    skipped_unreviewed = 0
    skipped_no_llm = 0

    for item_id, human_rec in human_ann.items():
        status = human_rec.get("overall_status", "unreviewed")
        if status == "unreviewed":
            skipped_unreviewed += 1
            continue
        if item_id not in llm_ann:
            skipped_no_llm += 1
            continue

        human_grade = int(status)
        llm_grade = llm_ann[item_id]
        manifest_item = manifest_items.get(item_id, {})

        if human_grade < args.min_grade and llm_grade < args.min_grade:
            continue

        paired.append(
            {
                "item_id": item_id,
                "query_id": manifest_item.get("query_id", human_rec.get("query_id", "")),
                "doc_id": manifest_item.get("doc_id", human_rec.get("doc_id", "")),
                "kpi": manifest_item.get("kpi", human_rec.get("kpi", "")),
                "match_type": manifest_item.get("match_type", human_rec.get("match_type", "")),
                "ticker": manifest_item.get("ticker", ""),
                "year": manifest_item.get("year", ""),
                "human_grade": human_grade,
                "llm_grade": llm_grade,
                "agree": human_grade == llm_grade,
                "human_notes": human_rec.get("notes", ""),
                "llm_reasoning": "",  # could be enriched from audit CSV
            }
        )

    print(f"Skipped (unreviewed): {skipped_unreviewed}")
    print(f"Skipped (no LLM match): {skipped_no_llm}")
    print(f"Paired items for analysis: {len(paired)}")

    if len(paired) < 2:
        print("\nNot enough paired items for agreement analysis.")
        return 1

    human_grades = np.array([p["human_grade"] for p in paired])
    llm_grades = np.array([p["llm_grade"] for p in paired])

    # --- Overall agreement ---
    agree_count = int(np.sum(human_grades == llm_grades))
    total = len(paired)
    pct_agree = agree_count / total

    # --- Cohen's kappa ---
    kappa_unweighted = compute_weighted_kappa(human_grades, llm_grades, weights=None)
    kappa_linear = compute_weighted_kappa(human_grades, llm_grades, weights="linear")
    kappa_quadratic = compute_weighted_kappa(human_grades, llm_grades, weights="quadratic")

    # --- Confusion matrix ---
    labels = [0, 1, 2]
    cm = confusion_matrix(human_grades, llm_grades, labels=labels)

    # --- Per-grade breakdown ---
    per_grade = {}
    for g in labels:
        mask_h = human_grades == g
        mask_l = llm_grades == g
        per_grade[grade_label(g)] = {
            "human_count": int(np.sum(mask_h)),
            "llm_count": int(np.sum(mask_l)),
            "agree_on_grade": int(np.sum((human_grades == g) & (llm_grades == g))),
        }

    # --- Disagreement analysis ---
    disagreements = [p for p in paired if not p["agree"]]
    off_by_one = sum(1 for p in disagreements if abs(p["human_grade"] - p["llm_grade"]) == 1)
    off_by_two = sum(1 for p in disagreements if abs(p["human_grade"] - p["llm_grade"]) == 2)

    # --- Directional flags ---
    human_higher = sum(1 for p in paired if p["human_grade"] > p["llm_grade"])
    llm_higher = sum(1 for p in paired if p["llm_grade"] > p["human_grade"])

    # --- Per-KPI breakdown ---
    by_kpi: dict[str, dict[str, Any]] = defaultdict(lambda: {"agree": 0, "total": 0, "grades_h": [], "grades_l": []})
    for p in paired:
        kpi = p["kpi"] or "unknown"
        by_kpi[kpi]["total"] += 1
        if p["agree"]:
            by_kpi[kpi]["agree"] += 1
        by_kpi[kpi]["grades_h"].append(p["human_grade"])
        by_kpi[kpi]["grades_l"].append(p["llm_grade"])

    kpi_stats = {}
    for kpi, d in sorted(by_kpi.items(), key=lambda x: -x[1]["total"]):
        gh = np.array(d["grades_h"])
        gl = np.array(d["grades_l"])
        k = compute_weighted_kappa(gh, gl, weights="quadratic") if len(gh) >= 2 else None
        kpi_stats[kpi] = {
            "n": d["total"],
            "agree": d["agree"],
            "pct_agree": format_pct(d["agree"] / d["total"]),
            "quadratic_kappa": f"{k:.3f}" if k is not None else "n/a",
        }

    # --- Per-match-type breakdown ---
    by_mt: dict[str, dict[str, Any]] = defaultdict(lambda: {"agree": 0, "total": 0, "grades_h": [], "grades_l": []})
    for p in paired:
        mt = p["match_type"] or "unknown"
        by_mt[mt]["total"] += 1
        if p["agree"]:
            by_mt[mt]["agree"] += 1
        by_mt[mt]["grades_h"].append(p["human_grade"])
        by_mt[mt]["grades_l"].append(p["llm_grade"])

    mt_stats = {}
    for mt, d in sorted(by_mt.items(), key=lambda x: -x[1]["total"]):
        gh = np.array(d["grades_h"])
        gl = np.array(d["grades_l"])
        k = compute_weighted_kappa(gh, gl, weights="quadratic") if len(gh) >= 2 else None
        mt_stats[mt] = {
            "n": d["total"],
            "agree": d["agree"],
            "pct_agree": format_pct(d["agree"] / d["total"]),
            "quadratic_kappa": f"{k:.3f}" if k is not None else "n/a",
        }

    # --- Build output ---
    output_dir = args.output_dir or (session_dir / "agreement")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Disagreement CSV
    disagree_csv_path = output_dir / "disagreements.csv"
    disagree_fields = [
        "item_id", "query_id", "doc_id", "kpi", "match_type",
        "ticker", "year", "human_grade", "llm_grade", "human_notes",
    ]
    with disagree_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=disagree_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(disagreements, key=lambda p: abs(p["human_grade"] - p["llm_grade"]), reverse=True))

    # Paired CSV
    paired_csv_path = output_dir / "paired_annotations.csv"
    paired_fields = [
        "item_id", "query_id", "doc_id", "kpi", "match_type",
        "ticker", "year", "human_grade", "llm_grade", "agree", "human_notes",
    ]
    with paired_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paired_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paired)

    # Summary JSON
    summary = {
        "session_id": args.session_id,
        "llm_audit_csv": str(args.llm_audit),
        "total_human_annotations": len(human_ann),
        "total_llm_annotations": len(llm_ann),
        "paired_items": total,
        "skipped_unreviewed": skipped_unreviewed,
        "skipped_no_llm_match": skipped_no_llm,
        "overall_agreement": {
            "agree_count": agree_count,
            "total": total,
            "percent_agreement": format_pct(pct_agree),
        },
        "cohens_kappa": {
            "unweighted": round(kappa_unweighted, 4),
            "linear_weighted": round(kappa_linear, 4),
            "quadratic_weighted": round(kappa_quadratic, 4),
        },
        "confusion_matrix": {
            "labels": ["0-NotRelevant", "1-Contextual", "2-Primary"],
            "rows_are_human_cols_are_llm": cm.tolist(),
        },
        "per_grade": per_grade,
        "disagreement_analysis": {
            "total_disagreements": len(disagreements),
            "off_by_one": off_by_one,
            "off_by_two": off_by_two,
            "human_grade_higher": human_higher,
            "llm_grade_higher": llm_higher,
        },
        "per_kpi": kpi_stats,
        "per_match_type": mt_stats,
    }

    summary_json_path = output_dir / "agreement_summary.json"
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Summary Markdown
    md_lines = [
        f"# Human-LLM Agreement: {args.session_id}",
        "",
        f"- Paired items: **{total}**",
        f"- Human annotations loaded: {len(human_ann)}",
        f"- LLM annotations loaded: {len(llm_ann)}",
        f"- Skipped (unreviewed): {skipped_unreviewed}",
        f"- Skipped (no LLM match): {skipped_no_llm}",
        "",
        "## Overall Agreement",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Percent agreement | **{format_pct(pct_agree)}** ({agree_count}/{total}) |",
        f"| Cohen's kappa (unweighted) | **{kappa_unweighted:.4f}** |",
        f"| Cohen's kappa (linear-weighted) | **{kappa_linear:.4f}** |",
        f"| Cohen's kappa (quadratic-weighted) | **{kappa_quadratic:.4f}** |",
        "",
        "## Confusion Matrix (Human \\ LLM)",
        "",
        "|  | LLM=0 | LLM=1 | LLM=2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for i, label in enumerate(["Human=0", "Human=1", "Human=2"]):
        row = cm[i]
        md_lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")

    md_lines += [
        "",
        "## Per-Grade Counts",
        "",
        "| Grade | Human count | LLM count | Agreement count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, d in per_grade.items():
        md_lines.append(
            f"| {label} | {d['human_count']} | {d['llm_count']} | {d['agree_on_grade']} |"
        )

    md_lines += [
        "",
        "## Disagreement Analysis",
        "",
        f"- Total disagreements: {len(disagreements)}",
        f"- Off by one grade: {off_by_one}",
        f"- Off by two grades: {off_by_two}",
        f"- Human grade > LLM: {human_higher}",
        f"- LLM grade > Human: {llm_higher}",
        "",
        "## Per-KPI Breakdown",
        "",
        "| KPI | N | Agree | % Agree | Quadratic Kappa |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kpi, d in kpi_stats.items():
        md_lines.append(
            f"| {kpi} | {d['n']} | {d['agree']} | {d['pct_agree']} | {d['quadratic_kappa']} |"
        )

    md_lines += [
        "",
        "## Per-Match-Type Breakdown",
        "",
        "| Match type | N | Agree | % Agree | Quadratic Kappa |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mt, d in mt_stats.items():
        md_lines.append(
            f"| {mt} | {d['n']} | {d['agree']} | {d['pct_agree']} | {d['quadratic_kappa']} |"
        )

    md_lines.append("")
    summary_md_path = output_dir / "agreement_summary.md"
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Print to stdout
    print("\n" + "=" * 60)
    print("AGREEMENT RESULTS")
    print("=" * 60)
    print(f"Paired items: {total}")
    print(f"Percent agreement: {format_pct(pct_agree)}")
    print(f"Cohen's kappa (unweighted):    {kappa_unweighted:.4f}")
    print(f"Cohen's kappa (linear):        {kappa_linear:.4f}")
    print(f"Cohen's kappa (quadratic):     {kappa_quadratic:.4f}")
    print()
    print("Confusion matrix (rows=human, cols=LLM):")
    print(f"         LLM=0  LLM=1  LLM=2")
    for i, label in enumerate(["Human=0", "Human=1", "Human=2"]):
        print(f"  {label}  {cm[i][0]:5d}  {cm[i][1]:5d}  {cm[i][2]:5d}")
    print()
    print(f"Disagreements: {len(disagreements)} (off-by-1: {off_by_one}, off-by-2: {off_by_two})")
    print(f"Human > LLM: {human_higher}  |  LLM > Human: {llm_higher}")
    print()
    print(f"Output: {output_dir}/")
    print(f"  agreement_summary.json")
    print(f"  agreement_summary.md")
    print(f"  paired_annotations.csv")
    print(f"  disagreements.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
