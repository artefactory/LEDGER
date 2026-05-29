"""
Binary classification grid search with quantile-based thresholds.

Two separate binary tasks per horizon:
  1. Positive vs Neutral: top Q% of returns = positive, rest = neutral
  2. Negative vs Neutral: bottom Q% of returns = negative, rest = neutral

Quantiles: 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%

EmbeddingGemma (--mode gemma) is encoded with its built-in "Classification"
task prompt before the trained classifiers run — see threshold_grid_search.py.

Usage:
    uv run python KPI_analysis/threshold_grid_search_binary.py --mode minilm
    uv run python KPI_analysis/threshold_grid_search_binary.py --mode roberta
    uv run python KPI_analysis/threshold_grid_search_binary.py --mode bow
    uv run python KPI_analysis/threshold_grid_search_binary.py --mode gemma
    uv run python KPI_analysis/threshold_grid_search_binary.py --mode all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from threshold_grid_search import (
    ENCODER_PRESETS,
    HORIZONS,
    SURPRISE_THRESHOLDS,
    HERE,
    build_records,
    encode_bow,
    encode_embeddings,
    load_letter_texts,
    SELECTED_COMPANIES_JSON,
)

OUTPUT_DIR = HERE / "output" / "plots" / "threshold_grid_search"

QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def evaluate_binary(quantile: float, horizon: int, y_ret: np.ndarray,
                    features: dict, classifiers: dict, cv,
                    direction: str) -> dict:
    """Binary classification: positive-vs-neutral or negative-vs-neutral.

    direction: 'positive' (top quantile = 1, rest = 0)
               'negative' (bottom quantile = 1, rest = 0)
    """
    n = len(y_ret)
    if direction == "positive":
        cutoff = np.quantile(y_ret, 1.0 - quantile)
        y_bin = (y_ret >= cutoff).astype(int)
    else:
        cutoff = np.quantile(y_ret, quantile)
        y_bin = (y_ret <= cutoff).astype(int)

    n_pos = int(y_bin.sum())
    n_neg = n - n_pos
    baseline = max(n_pos, n_neg) / n

    result = {
        "quantile": quantile,
        "horizon": horizon,
        "direction": direction,
        "cutoff_value": float(cutoff),
        "n": n,
        "n_class_1": n_pos,
        "n_class_0": n_neg,
        "baseline": float(baseline),
    }

    if n_pos < 5 or n_neg < 5:
        result["skipped"] = True
        result["skip_reason"] = "class_too_small"
        return result

    result["skipped"] = False

    for clf_name, (estimator, feat_key) in classifiers.items():
        X = features[feat_key]
        try:
            acc = cross_val_score(estimator, X, y_bin, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        except ValueError:
            acc = float("nan")
        try:
            auc = cross_val_score(estimator, X, y_bin, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        except ValueError:
            auc = float("nan")
        try:
            pr_auc = cross_val_score(estimator, X, y_bin, cv=cv, scoring="average_precision", n_jobs=-1).mean()
        except ValueError:
            pr_auc = float("nan")
        result[f"{clf_name}_accuracy"] = float(acc)
        result[f"{clf_name}_roc_auc"] = float(auc)
        result[f"{clf_name}_pr_auc"] = float(pr_auc)

    return result


def run_grid_search_binary(mode: str, records: list, output_dir: Path):
    """Run binary grid search for a given mode on all targets: raw_return, residual, residual_inliers, surprise."""
    corpus = [r["text"] for r in records]

    if mode == "bow":
        features, classifiers, metadata = encode_bow(corpus)
    else:
        model_name = ENCODER_PRESETS.get(mode, mode)
        features, classifiers, metadata = encode_embeddings(corpus, model_name)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_names = list(classifiers.keys())

    all_target_results = {}

    # ===== TARGET 1: RAW RETURN =====
    target_name = "raw_return"
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET: {target_name} — {len(QUANTILES)} quantiles x {len(HORIZONS)} horizons x 2 directions")
    print(f"{'='*80}\n")
    all_target_results[target_name] = _run_binary_on_horizons(
        mode, records, features, classifiers, cv, clf_names,
        target_name, HORIZONS,
        get_y_and_idx=lambda h, recs: _idx_raw_return(h, recs),
    )

    # ===== TARGET 1b: UNBIASED RETURN (return - industry return) =====
    target_name = "unbiased_return"
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET: {target_name} — {len(QUANTILES)} quantiles x {len(HORIZONS)} horizons x 2 directions")
    print(f"{'='*80}\n")
    all_target_results[target_name] = _run_binary_on_horizons(
        mode, records, features, classifiers, cv, clf_names,
        target_name, HORIZONS,
        get_y_and_idx=lambda h, recs: _idx_unbiased_return(h, recs),
    )

    # ===== TARGET 2: RESIDUAL (return - f(surprise)) =====
    target_name = "residual"
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET: {target_name} — {len(QUANTILES)} quantiles x {len(HORIZONS)} horizons x 2 directions")
    print(f"{'='*80}\n")
    all_target_results[target_name] = _run_binary_on_horizons(
        mode, records, features, classifiers, cv, clf_names,
        target_name, HORIZONS,
        get_y_and_idx=lambda h, recs: _idx_residual(h, recs),
    )

    # ===== TARGET 2b: RESIDUAL INLIERS =====
    target_name = "residual_inliers"
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET: {target_name} — {len(QUANTILES)} quantiles x {len(HORIZONS)} horizons x 2 directions")
    print(f"{'='*80}\n")
    all_target_results[target_name] = _run_binary_on_horizons(
        mode, records, features, classifiers, cv, clf_names,
        target_name, HORIZONS,
        get_y_and_idx=lambda h, recs: _idx_residual_inliers(h, recs),
    )

    # ===== TARGET 3: SURPRISE =====
    target_name = "surprise"
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET: {target_name} — {len(QUANTILES)} quantiles x 2 directions")
    print(f"{'='*80}\n")

    surp_idx = []
    surp_vals = []
    for i, r in enumerate(records):
        if r["surprise"] is not None:
            surp_idx.append(i)
            surp_vals.append(r["surprise"])

    surprise_results = {"positive_vs_neutral": [], "negative_vs_neutral": []}
    if len(surp_idx) >= 20:
        s_features = {k: v[surp_idx] for k, v in features.items()}
        y_surp = np.array(surp_vals)

        for direction in ["positive", "negative"]:
            label = "Top" if direction == "positive" else "Bottom"
            for q in QUANTILES:
                result = evaluate_binary(q, 0, y_surp, s_features, classifiers, cv, direction)
                result["target"] = target_name
                result.pop("horizon", None)
                key = f"{direction}_vs_neutral"
                surprise_results[key].append(result)

                if not result.get("skipped"):
                    parts = [f"{cn}={result.get(f'{cn}_accuracy', float('nan')):.3f}" for cn in clf_names]
                    print(f"  {label} {q*100:.0f}%: {' '.join(parts)} "
                          f"(bl={result['baseline']:.3f}, cut={result['cutoff_value']:.4f})")
                else:
                    print(f"  {label} {q*100:.0f}%: SKIPPED")
    else:
        print(f"  Only {len(surp_idx)} samples with surprise, skipping")

    all_target_results[target_name] = surprise_results

    # ===== SAVE =====
    mode_slug = mode if mode in ENCODER_PRESETS or mode == "bow" else mode.split("/")[-1]
    output_path = output_dir / f"grid_search_binary_{mode_slug}.json"
    with open(output_path, "w") as f:
        json.dump({
            "mode": mode,
            "quantiles": QUANTILES,
            "horizons": HORIZONS,
            "n_samples_total": len(records),
            "classifiers": clf_names,
            "metadata": metadata,
            "targets": all_target_results,
        }, f, indent=2)
    print(f"\n[{mode}] Results saved to {output_path}")

    # ===== SUMMARY =====
    for target_name, target_data in all_target_results.items():
        for direction in ["positive", "negative"]:
            key = f"{direction}_vs_neutral"
            if key in target_data and target_data[key]:
                _print_binary_summary(f"[{mode}] {target_name} — {direction.upper()} vs NEUTRAL",
                                      target_data[key], clf_names)


def _idx_raw_return(h: int, records: list) -> tuple[list[int], np.ndarray] | None:
    """Get valid indices and y values for raw return at horizon h."""
    valid_idx = []
    y_returns = []
    for i, r in enumerate(records):
        if h in r["returns"]:
            valid_idx.append(i)
            y_returns.append(r["returns"][h])
    if len(valid_idx) < 20:
        return None
    return valid_idx, np.array(y_returns)


def _idx_unbiased_return(h: int, records: list) -> tuple[list[int], np.ndarray] | None:
    """Get valid indices and y values for unbiased return (stock - industry) at horizon h."""
    valid_idx = []
    y_returns = []
    for i, r in enumerate(records):
        ub = r.get("unbiased_returns", {})
        if h in ub:
            valid_idx.append(i)
            y_returns.append(ub[h])
    if len(valid_idx) < 20:
        return None
    return valid_idx, np.array(y_returns)


def _idx_residual(h: int, records: list) -> tuple[list[int], np.ndarray] | None:
    """Get valid indices and residual y values (return - f(surprise)) at horizon h."""
    valid_idx = []
    y_returns = []
    surprises = []
    for i, r in enumerate(records):
        if h in r["returns"] and r["surprise"] is not None:
            valid_idx.append(i)
            y_returns.append(r["returns"][h])
            surprises.append(r["surprise"])
    if len(valid_idx) < 20:
        return None
    y_ret = np.array(y_returns)
    surp_arr = np.array(surprises)
    lr = LinearRegression()
    lr.fit(surp_arr.reshape(-1, 1), y_ret)
    residuals = y_ret - lr.predict(surp_arr.reshape(-1, 1))
    return valid_idx, residuals


def _idx_residual_inliers(h: int, records: list) -> tuple[list[int], np.ndarray] | None:
    """Get valid indices and residual y values with surprise inliers (q2-q98)."""
    valid_idx = []
    y_returns = []
    surprises = []
    for i, r in enumerate(records):
        if h in r["returns"] and r["surprise"] is not None:
            valid_idx.append(i)
            y_returns.append(r["returns"][h])
            surprises.append(r["surprise"])
    if len(valid_idx) < 20:
        return None
    surp_arr = np.array(surprises)
    q_lo = np.quantile(surp_arr, 0.02)
    q_hi = np.quantile(surp_arr, 0.98)
    inlier_mask = (surp_arr >= q_lo) & (surp_arr <= q_hi)
    if inlier_mask.sum() < 20:
        return None
    inlier_idx = [valid_idx[j] for j in range(len(valid_idx)) if inlier_mask[j]]
    y_ret = np.array(y_returns)[inlier_mask]
    surp_in = surp_arr[inlier_mask]
    lr = LinearRegression()
    lr.fit(surp_in.reshape(-1, 1), y_ret)
    residuals = y_ret - lr.predict(surp_in.reshape(-1, 1))
    return inlier_idx, residuals


def _run_binary_on_horizons(mode, records, features, classifiers, cv, clf_names,
                            target_name, horizons, get_y_and_idx) -> dict:
    """Run binary grid for both directions over all horizons for a given target."""
    results = {"positive_vs_neutral": [], "negative_vs_neutral": []}

    total = len(QUANTILES) * len(horizons)

    for direction in ["positive", "negative"]:
        label = "Top" if direction == "positive" else "Bottom"
        key = f"{direction}_vs_neutral"
        done = 0

        for h in horizons:
            result_tuple = get_y_and_idx(h, records)
            if result_tuple is None:
                for q in QUANTILES:
                    results[key].append({
                        "quantile": q, "horizon": h, "direction": direction,
                        "target": target_name,
                        "n": 0, "skipped": True,
                        "skip_reason": "insufficient_samples",
                    })
                    done += 1
                continue

            valid_idx, y_ret = result_tuple
            h_features = {k: v[valid_idx] for k, v in features.items()}

            for q in QUANTILES:
                done += 1
                result = evaluate_binary(q, h, y_ret, h_features, classifiers, cv, direction)
                result["target"] = target_name
                results[key].append(result)

                if not result.get("skipped"):
                    parts = [f"{cn}={result.get(f'{cn}_accuracy', float('nan')):.3f}" for cn in clf_names]
                    print(f"  [{done}/{total}] h={h:>3d}, {label} {q*100:.0f}%: "
                          f"{' '.join(parts)} (bl={result['baseline']:.3f}, "
                          f"cut={result['cutoff_value']:.4f})")
                else:
                    print(f"  [{done}/{total}] h={h:>3d}, {label} {q*100:.0f}%: SKIPPED")

    return results


def _print_binary_summary(label: str, results: list, clf_names: list):
    """Print best quantile per horizon summary."""
    print(f"\n{'='*100}")
    print(f"{label} — Best quantile per horizon (by max ROC AUC)")
    print(f"{'Horizon':>8} {'Quantile':>9} {'Classifier':>12} {'Accuracy':>9} {'ROC AUC':>8} {'Baseline':>9} {'N':>5}")
    print(f"{'-'*100}")

    seen_horizons = sorted(set(r["horizon"] for r in results if "horizon" in r))
    for h in seen_horizons:
        h_results = [r for r in results if r.get("horizon") == h and not r.get("skipped")]
        if not h_results:
            print(f"{h:>8d} {'—':>9} {'—':>12} {'—':>9} {'—':>8} {'—':>9} {'—':>5}")
            continue

        best = None
        best_auc = -1
        best_clf = ""
        for r in h_results:
            for cn in clf_names:
                val = r.get(f"{cn}_roc_auc", float("nan"))
                if not np.isnan(val) and val > best_auc:
                    best_auc = val
                    best = r
                    best_clf = cn

        if best:
            acc = best.get(f"{best_clf}_accuracy", float("nan"))
            print(f"{h:>8d} {best['quantile']*100:>8.0f}% {best_clf:>12} {acc:>9.3f} {best_auc:>8.3f} "
                  f"{best['baseline']:>9.3f} {best['n']:>5d}")

    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description="Binary threshold grid search (quantile-based)")
    parser.add_argument("--mode", default="all",
                        help="Encoder preset (minilm, roberta, mpnet, eurobert, gemma, bow), "
                             "'all' for all presets, or a full model path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SELECTED_COMPANIES_JSON) as f:
        companies_data = json.load(f)

    all_tickers = set()
    for industry, exchanges in companies_data.items():
        for exchange, companies in exchanges.items():
            for company in companies:
                all_tickers.add(company["ticker"])

    print("Loading CEO letter texts...")
    letter_texts = load_letter_texts()
    print(f"  Loaded {len(letter_texts)} (ticker, year) letter texts")

    records = build_records(all_tickers, letter_texts)
    print(f"\nTotal samples with letter + returns: {len(records)}")
    if len(records) < 20:
        print("Not enough samples. Exiting.")
        return

    if args.mode == "all":
        modes = list(ENCODER_PRESETS.keys()) + ["bow"]
    else:
        modes = [args.mode]

    for mode in modes:
        run_grid_search_binary(mode, records, OUTPUT_DIR)


if __name__ == "__main__":
    main()