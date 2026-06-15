"""
Plot heatmaps from binary grid search results (quantile-based).

For each target (raw_return, residual, residual_inliers, surprise):
  - Single combined heatmap with percentile axis 5%..95%
    Left half (5-50%) = negative extremes, right half (55-95%) = positive extremes

Cross-mode comparison: best AUC per horizon for each target.

Output: threshold_grid_search/binary/

Usage:
    uv run python KPI_analysis/plot_grid_search_binary_heatmaps.py
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
GRID_DIR = HERE / "output" / "plots" / "threshold_grid_search"
OUTPUT_DIR = GRID_DIR / "binary"


def load_binary_grid(mode: str) -> dict:
    path = GRID_DIR / f"grid_search_binary_{mode}.json"
    with open(path) as f:
        return json.load(f)


def _build_percentile_axis(quantiles: list[float]):
    """Build combined percentile axis: positive quantiles 5-50% (left), negative 55-95% (right)."""
    # Left side: positive (top-Q%) mapped as q*100 -> 5,10,...,50
    pos_pcts = sorted([q * 100 for q in quantiles])
    # Right side: negative (bottom-Q%) mapped as 100-q*100 -> 95,90,...,55
    neg_pcts = sorted([100 - q * 100 for q in quantiles if q != 0.5])
    all_pcts = pos_pcts + neg_pcts
    return all_pcts


def _merge_results(neg_results: list, pos_results: list, quantiles: list[float], horizons: list, clf_names: list):
    """Merge negative + positive results into combined arrays indexed by (horizon, percentile)."""
    all_pcts = _build_percentile_axis(quantiles)
    pct_idx = {p: i for i, p in enumerate(all_pcts)}
    h_idx = {h: i for i, h in enumerate(horizons)}
    n_h = len(horizons)
    n_p = len(all_pcts)

    best_auc = np.full((n_h, n_p), np.nan)
    best_acc = np.full((n_h, n_p), np.nan)
    best_pr_auc = np.full((n_h, n_p), np.nan)
    baselines = np.full((n_h, n_p), np.nan)
    hover_data = [[None] * n_p for _ in range(n_h)]

    def _process(results, q_to_pct):
        for r in results:
            if r.get('skipped'):
                continue
            h = r.get('horizon')
            q = r.get('quantile')
            if h not in h_idx:
                continue
            pct = q_to_pct(q)
            if pct is None or pct not in pct_idx:
                continue
            hi, pi = h_idx[h], pct_idx[pct]
            baselines[hi, pi] = r['baseline']
            hover_data[hi][pi] = r

            for cn in clf_names:
                auc = r.get(f'{cn}_roc_auc', float('nan'))
                acc = r.get(f'{cn}_accuracy', float('nan'))
                pr_auc = r.get(f'{cn}_pr_auc', float('nan'))
                if not np.isnan(auc) and (np.isnan(best_auc[hi, pi]) or auc > best_auc[hi, pi]):
                    best_auc[hi, pi] = auc
                if not np.isnan(acc) and (np.isnan(best_acc[hi, pi]) or acc > best_acc[hi, pi]):
                    best_acc[hi, pi] = acc
                if not np.isnan(pr_auc) and (np.isnan(best_pr_auc[hi, pi]) or pr_auc > best_pr_auc[hi, pi]):
                    best_pr_auc[hi, pi] = pr_auc

    _process(pos_results, lambda q: q * 100)
    _process(neg_results, lambda q: 100 - q * 100 if q != 0.5 else None)

    return best_auc, best_acc, best_pr_auc, baselines, hover_data, all_pcts


def _merge_results_no_horizon(neg_results: list, pos_results: list, quantiles: list[float], clf_names: list):
    """Merge negative + positive results for horizon-less targets (surprise)."""
    all_pcts = _build_percentile_axis(quantiles)
    pct_idx = {p: i for i, p in enumerate(all_pcts)}
    n_p = len(all_pcts)

    best_auc = np.full(n_p, np.nan)
    best_acc = np.full(n_p, np.nan)
    best_pr_auc = np.full(n_p, np.nan)
    baselines = np.full(n_p, np.nan)
    hover_data = [None] * n_p

    def _process(results, q_to_pct):
        for r in results:
            if r.get('skipped'):
                continue
            q = r.get('quantile')
            pct = q_to_pct(q)
            if pct is None or pct not in pct_idx:
                continue
            pi = pct_idx[pct]
            baselines[pi] = r['baseline']
            hover_data[pi] = r
            for cn in clf_names:
                auc = r.get(f'{cn}_roc_auc', float('nan'))
                acc = r.get(f'{cn}_accuracy', float('nan'))
                pr_auc = r.get(f'{cn}_pr_auc', float('nan'))
                if not np.isnan(auc) and (np.isnan(best_auc[pi]) or auc > best_auc[pi]):
                    best_auc[pi] = auc
                if not np.isnan(acc) and (np.isnan(best_acc[pi]) or acc > best_acc[pi]):
                    best_acc[pi] = acc
                if not np.isnan(pr_auc) and (np.isnan(best_pr_auc[pi]) or pr_auc > best_pr_auc[pi]):
                    best_pr_auc[pi] = pr_auc

    _process(pos_results, lambda q: q * 100)
    _process(neg_results, lambda q: 100 - q * 100 if q != 0.5 else None)

    return best_auc, best_acc, best_pr_auc, baselines, hover_data, all_pcts


def _make_hover_matrix(hover_data, clf_names, metric):
    """Build hover text from stored result records (2D for heatmaps)."""
    n_h = len(hover_data)
    n_p = len(hover_data[0])
    text = [["" for _ in range(n_p)] for _ in range(n_h)]

    for hi in range(n_h):
        for pi in range(n_p):
            r = hover_data[hi][pi]
            if r is None:
                continue
            text[hi][pi] = _format_hover(r, clf_names, metric)

    return text


def _format_hover(r, clf_names, metric):
    """Format a single result record as hover text."""
    bl = r['baseline']
    n = r['n']
    n1 = r.get('n_class_1', 0)
    n0 = r.get('n_class_0', 0)
    cutoff = r.get('cutoff_value', float('nan'))
    direction = r.get('direction', '?')
    target = r.get('target', '?')
    h = r.get('horizon', '-')
    q = r.get('quantile', 0)

    lines = [
        f"<b>h={h}, q={q*100:.0f}%, n={n}</b>",
        f"Target: {target} | Direction: {direction}",
        f"Cutoff: {cutoff:.5f}",
        f"Class 1: {n1} ({n1/n:.0%}) / Class 0: {n0} ({n0/n:.0%})",
        f"Baseline: {bl:.3f}",
        "",
    ]

    best_val = -1
    best_name = ""
    key = f"_{metric}"
    for cn in clf_names:
        val = r.get(f'{cn}{key}', float('nan'))
        if np.isnan(val):
            lines.append(f"  {cn}: N/A")
        else:
            lines.append(f"  {cn}: {val:.4f}")
            if val > best_val:
                best_val = val
                best_name = cn

    lines.append("")
    if best_name:
        if metric == "accuracy":
            diff = best_val - bl
            lines.append(f"<b>Best: {best_name} ({best_val:.4f}, lift={diff:+.4f})</b>")
        elif metric == "pr_auc":
            prevalence = n1 / n if n > 0 else 0.5
            diff = best_val - prevalence
            lines.append(f"<b>Best: {best_name} ({best_val:.4f}, vs prevalence({prevalence:.2f})={diff:+.4f})</b>")
        else:
            diff = best_val - 0.5
            lines.append(f"<b>Best: {best_name} ({best_val:.4f}, vs random={diff:+.4f})</b>")

    return "<br>".join(lines)


def plot_combined_heatmaps(neg_results, pos_results, clf_names, horizons, quantiles, mode, target, out_dir,
                           bow_neg=None, bow_pos=None, bow_clf_names=None):
    """Create combined AUC, PR AUC, and accuracy heatmaps with percentile axis 5%..95%.

    If bow_neg/bow_pos are provided, heatmaps show lift over BOW instead of random/majority.
    """
    best_auc, best_acc, best_pr_auc, baselines, hover_data, all_pcts = _merge_results(
        neg_results, pos_results, quantiles, horizons, clf_names
    )

    # Build BOW reference arrays
    bow_auc_ref = None
    bow_acc_ref = None
    bow_pr_auc_ref = None
    if bow_neg is not None and bow_pos is not None and bow_clf_names:
        bow_auc_ref, bow_acc_ref, bow_pr_auc_ref, _, _, _ = _merge_results(
            bow_neg, bow_pos, quantiles, horizons, bow_clf_names
        )

    y_labels = [str(h) for h in horizons]
    x_labels = [f"{p:.0f}%" for p in all_pcts]
    slug = f"{mode}_{target}"

    # Separator ON the 50% column
    n_left = len([p for p in all_pcts if p <= 50])
    sep_pos = n_left - 1  # index of 50% column

    def _add_side_annotations(fig):
        fig.add_annotation(x=n_left / 2 - 0.5, y=-0.08, yref='paper',
                           text="<b>POS vs Neutre</b>", showarrow=False,
                           font=dict(size=13, color='green'))
        fig.add_annotation(x=n_left + (len(all_pcts) - n_left) / 2 - 0.5, y=-0.08, yref='paper',
                           text="<b>NEG vs Neutre</b>", showarrow=False,
                           font=dict(size=13, color='red'))

    # --- ROC AUC Heatmap ---
    if bow_auc_ref is not None:
        auc_lift = best_auc - bow_auc_ref
        auc_title = f"{mode.upper()} - {target}<br>ROC AUC lift over BOW"
        auc_cbar = "AUC - BOW"
    else:
        auc_lift = best_auc - 0.5
        auc_title = f"{mode.upper()} - {target}<br>ROC AUC lift over random"
        auc_cbar = "AUC - 0.5"
    hover_auc = _make_hover_matrix(hover_data, clf_names, "roc_auc")
    fig_auc = go.Figure(data=go.Heatmap(
        z=auc_lift,
        x=x_labels,
        y=y_labels,
        text=hover_auc,
        hoverinfo='text',
        colorscale='RdYlGn',
        zmid=0,
        zmin=-0.4,
        zmax=0.4,
        colorbar=dict(title=auc_cbar),
    ))
    fig_auc.add_vline(x=sep_pos, line_dash="dot", line_color="gray", line_width=2)
    _add_side_annotations(fig_auc)
    fig_auc.update_layout(
        title=auc_title,
        xaxis_title="Quantile",
        yaxis_title="Horizon (days)",
        width=1200, height=800,
        margin=dict(b=100),
    )
    (out_dir / "roc_auc").mkdir(parents=True, exist_ok=True)
    fig_auc.write_html(str(out_dir / "roc_auc" / f"heatmap_auc_{slug}.html"))

    # --- PR AUC Heatmap ---
    # Random PR AUC = prevalence per quantile (varies by column)
    random_pr_auc_row = np.array([p / 100 if p <= 50 else (100 - p) / 100 for p in all_pcts])
    random_pr_auc_2d = np.tile(random_pr_auc_row, (len(horizons), 1))
    if bow_pr_auc_ref is not None:
        pr_lift = best_pr_auc - bow_pr_auc_ref
        pr_title = f"{mode.upper()} - {target}<br>PR AUC lift over BOW"
        pr_cbar = "PR AUC - BOW"
    else:
        pr_lift = best_pr_auc - random_pr_auc_2d
        pr_title = f"{mode.upper()} - {target}<br>PR AUC lift over random (prevalence)"
        pr_cbar = "PR AUC - prev."
    hover_pr = _make_hover_matrix(hover_data, clf_names, "pr_auc")
    fig_pr = go.Figure(data=go.Heatmap(
        z=pr_lift,
        x=x_labels,
        y=y_labels,
        text=hover_pr,
        hoverinfo='text',
        colorscale='RdYlGn',
        zmid=0,
        zmin=-0.4,
        zmax=0.4,
        colorbar=dict(title=pr_cbar),
    ))
    fig_pr.add_vline(x=sep_pos, line_dash="dot", line_color="gray", line_width=2)
    _add_side_annotations(fig_pr)
    fig_pr.update_layout(
        title=pr_title,
        xaxis_title="Quantile",
        yaxis_title="Horizon (days)",
        width=1200, height=800,
        margin=dict(b=100),
    )
    (out_dir / "pr_auc").mkdir(parents=True, exist_ok=True)
    fig_pr.write_html(str(out_dir / "pr_auc" / f"heatmap_pr_auc_{slug}.html"))

    # --- Accuracy lift heatmap ---
    if bow_acc_ref is not None:
        lift_matrix = best_acc - bow_acc_ref
        acc_title = f"{mode.upper()} - {target}<br>Accuracy Lift over BOW"
        acc_cbar = "Lift (Acc - BOW)"
    else:
        lift_matrix = best_acc - baselines
        acc_title = f"{mode.upper()} - {target}<br>Accuracy Lift over Baseline"
        acc_cbar = "Lift (Acc - Baseline)"
    hover_acc = _make_hover_matrix(hover_data, clf_names, "accuracy")
    fig_acc = go.Figure(data=go.Heatmap(
        z=lift_matrix,
        x=x_labels,
        y=y_labels,
        text=hover_acc,
        hoverinfo='text',
        colorscale='RdBu',
        zmid=0,
        zmin=-0.4,
        zmax=0.4,
        colorbar=dict(title=acc_cbar),
    ))
    fig_acc.add_vline(x=sep_pos, line_dash="dot", line_color="gray", line_width=2)
    _add_side_annotations(fig_acc)
    fig_acc.update_layout(
        title=acc_title,
        xaxis_title="Quantile",
        yaxis_title="Horizon (days)",
        width=1200, height=800,
        margin=dict(b=100),
    )
    (out_dir / "accuracy").mkdir(parents=True, exist_ok=True)
    fig_acc.write_html(str(out_dir / "accuracy" / f"heatmap_lift_{slug}.html"))


def plot_combined_surprise(neg_results, pos_results, clf_names, quantiles, mode, target, out_dir,
                           bow_neg=None, bow_pos=None, bow_clf_names=None):
    """Grouped bar chart for surprise (no horizon) with combined percentile axis."""
    best_auc, best_acc, best_pr_auc, baselines, hover_data, all_pcts = _merge_results_no_horizon(
        neg_results, pos_results, quantiles, clf_names
    )

    x_labels = [f"{p:.0f}%" for p in all_pcts]
    slug = f"{mode}_{target}"
    n_p = len(all_pcts)
    pct_idx = {p: i for i, p in enumerate(all_pcts)}
    n_left = len([p for p in all_pcts if p <= 50])
    sep_pos = n_left - 1

    # Build per-classifier arrays
    auc_by_clf = {cn: np.full(n_p, np.nan) for cn in clf_names}
    acc_by_clf = {cn: np.full(n_p, np.nan) for cn in clf_names}
    pr_auc_by_clf = {cn: np.full(n_p, np.nan) for cn in clf_names}

    def _fill(results, q_to_pct):
        for r in results:
            if r.get('skipped'):
                continue
            q = r.get('quantile')
            pct = q_to_pct(q)
            if pct is None or pct not in pct_idx:
                continue
            pi = pct_idx[pct]
            for cn in clf_names:
                auc = r.get(f'{cn}_roc_auc', float('nan'))
                acc = r.get(f'{cn}_accuracy', float('nan'))
                pr_auc = r.get(f'{cn}_pr_auc', float('nan'))
                if not np.isnan(auc):
                    auc_by_clf[cn][pi] = auc
                if not np.isnan(acc):
                    acc_by_clf[cn][pi] = acc
                if not np.isnan(pr_auc):
                    pr_auc_by_clf[cn][pi] = pr_auc

    _fill(pos_results, lambda q: q * 100)
    _fill(neg_results, lambda q: 100 - q * 100 if q != 0.5 else None)

    # Build BOW reference if available
    bow_auc_arr = None
    bow_pr_auc_arr = None
    bow_acc_arr = None
    if bow_neg is not None and bow_pos is not None and bow_clf_names:
        bow_auc_1d, bow_acc_1d, bow_pr_1d, _, _, _ = _merge_results_no_horizon(
            bow_neg, bow_pos, quantiles, bow_clf_names
        )
        bow_auc_arr = bow_auc_1d
        bow_acc_arr = bow_acc_1d
        bow_pr_auc_arr = bow_pr_1d

    # --- AUC grouped bar ---
    fig_auc = go.Figure()
    for cn in clf_names:
        fig_auc.add_trace(go.Bar(
            name=cn,
            x=x_labels,
            y=auc_by_clf[cn].tolist(),
            text=[f"{v:.3f}" if not np.isnan(v) else "" for v in auc_by_clf[cn]],
            textposition='auto',
        ))
    fig_auc.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Random")
    if bow_auc_arr is not None:
        fig_auc.add_trace(go.Scatter(
            name="BOW baseline",
            x=x_labels,
            y=bow_auc_arr.tolist(),
            mode='lines+markers',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8),
        ))
    fig_auc.add_vline(x=sep_pos, line_dash="dot", line_color="gray", line_width=2)
    fig_auc.add_annotation(x=n_left / 2 - 0.5, y=-0.08, yref='paper',
                           text="<b>POS vs Neutre</b>", showarrow=False,
                           font=dict(size=13, color='green'))
    fig_auc.add_annotation(x=n_left + (n_p - n_left) / 2 - 0.5, y=-0.08, yref='paper',
                           text="<b>NEG vs Neutre</b>", showarrow=False,
                           font=dict(size=13, color='red'))
    fig_auc.update_layout(
        title=f"{mode.upper()} - {target}<br>ROC AUC",
        xaxis_title="Quantile",
        yaxis_title="ROC AUC",
        yaxis_range=[0.4, 0.85],
        barmode='group',
        width=1600, height=700,
        margin=dict(b=100),
    )
    (out_dir / "roc_auc").mkdir(parents=True, exist_ok=True)
    fig_auc.write_html(str(out_dir / "roc_auc" / f"surprise_auc_{slug}.html"))

    # --- PR AUC grouped bar ---
    fig_pr = go.Figure()
    for cn in clf_names:
        fig_pr.add_trace(go.Bar(
            name=cn,
            x=x_labels,
            y=pr_auc_by_clf[cn].tolist(),
            text=[f"{v:.3f}" if not np.isnan(v) else "" for v in pr_auc_by_clf[cn]],
            textposition='auto',
        ))
    # Random PR AUC baseline = prevalence of positive class at each quantile
    # Left side (pos direction): prevalence = pct/100; Right side (neg direction): prevalence = (100-pct)/100
    random_pr_auc = [p / 100 if p <= 50 else (100 - p) / 100 for p in all_pcts]
    fig_pr.add_trace(go.Scatter(
        name="Random baseline",
        x=x_labels,
        y=random_pr_auc,
        mode='lines',
        line=dict(color='red', width=2, dash='dot'),
    ))
    if bow_pr_auc_arr is not None:
        fig_pr.add_trace(go.Scatter(
            name="BOW baseline",
            x=x_labels,
            y=bow_pr_auc_arr.tolist(),
            mode='lines+markers',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8),
        ))
    fig_pr.add_vline(x=sep_pos, line_dash="dot", line_color="gray", line_width=2)
    fig_pr.add_annotation(x=n_left / 2 - 0.5, y=-0.08, yref='paper',
                          text="<b>POS vs Neutre</b>", showarrow=False,
                          font=dict(size=13, color='green'))
    fig_pr.add_annotation(x=n_left + (n_p - n_left) / 2 - 0.5, y=-0.08, yref='paper',
                          text="<b>NEG vs Neutre</b>", showarrow=False,
                          font=dict(size=13, color='red'))
    fig_pr.update_layout(
        title=f"{mode.upper()} - {target}<br>PR AUC",
        xaxis_title="Quantile",
        yaxis_title="PR AUC",
        yaxis_range=[0.2, 0.85],
        barmode='group',
        width=1600, height=700,
        margin=dict(b=100),
    )
    (out_dir / "pr_auc").mkdir(parents=True, exist_ok=True)
    fig_pr.write_html(str(out_dir / "pr_auc" / f"surprise_pr_auc_{slug}.html"))

    # --- Accuracy vs baseline ---
    fig_acc = go.Figure()
    for cn in clf_names:
        fig_acc.add_trace(go.Bar(
            name=cn,
            x=x_labels,
            y=acc_by_clf[cn].tolist(),
        ))
    fig_acc.add_trace(go.Scatter(
        name="Majority Baseline",
        x=x_labels,
        y=baselines.tolist(),
        mode='lines+markers',
        line=dict(color='gray', width=2, dash='dot'),
        marker=dict(size=6),
    ))
    if bow_acc_arr is not None:
        fig_acc.add_trace(go.Scatter(
            name="BOW baseline",
            x=x_labels,
            y=bow_acc_arr.tolist(),
            mode='lines+markers',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8),
        ))
    fig_acc.add_vline(x=sep_pos, line_dash="dot", line_color="gray", line_width=2)
    fig_acc.add_annotation(x=n_left / 2 - 0.5, y=-0.08, yref='paper',
                           text="<b>POS vs Neutre</b>", showarrow=False,
                           font=dict(size=13, color='green'))
    fig_acc.add_annotation(x=n_left + (n_p - n_left) / 2 - 0.5, y=-0.08, yref='paper',
                           text="<b>NEG vs Neutre</b>", showarrow=False,
                           font=dict(size=13, color='red'))
    fig_acc.update_layout(
        title=f"{mode.upper()} - {target}<br>Accuracy vs BOW Baseline",
        xaxis_title="Quantile",
        yaxis_title="Accuracy",
        yaxis_range=[0.3, 1.0],
        barmode='group',
        width=1600, height=700,
        margin=dict(b=100),
    )
    (out_dir / "accuracy").mkdir(parents=True, exist_ok=True)
    fig_acc.write_html(str(out_dir / "accuracy" / f"surprise_acc_{slug}.html"))


def plot_comparison_across_modes(modes, target, out_dir):
    """Best AUC/PR AUC per horizon across modes (combined neg+pos best). BOW as baseline."""
    colors = {
        'minilm': '#1f77b4', 'minilm-l12': '#aec7e8',
        'roberta': '#9467bd', 'eurobert': '#ff7f0e',
        'mpnet': '#d62728',
    }
    _palette = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    encoder_modes = [m for m in modes if m != 'bow']

    # Collect BOW baseline
    bow_best_auc = None
    bow_best_pr_auc = None
    bow_horizons = None
    if 'bow' in modes:
        bow_data = load_binary_grid('bow')
        bow_targets = bow_data.get("targets", {})
        bow_target_data = bow_targets.get(target, {})
        bow_all_results = bow_target_data.get("negative_vs_neutral", []) + bow_target_data.get("positive_vs_neutral", [])
        if bow_all_results:
            bow_clf_names = bow_data['classifiers']
            bow_horizons = bow_data['horizons']
            bow_best_auc = []
            bow_best_pr_auc = []
            for h in bow_horizons:
                h_results = [r for r in bow_all_results if r.get('horizon') == h and not r.get('skipped')]
                best_auc = float('nan')
                best_pr = float('nan')
                for r in h_results:
                    for cn in bow_clf_names:
                        val = r.get(f'{cn}_roc_auc', float('nan'))
                        if not np.isnan(val) and (np.isnan(best_auc) or val > best_auc):
                            best_auc = val
                        val_pr = r.get(f'{cn}_pr_auc', float('nan'))
                        if not np.isnan(val_pr) and (np.isnan(best_pr) or val_pr > best_pr):
                            best_pr = val_pr
                bow_best_auc.append(best_auc)
                bow_best_pr_auc.append(best_pr)

    # --- ROC AUC comparison ---
    fig = go.Figure()
    for i_mode, mode in enumerate(encoder_modes):
        data = load_binary_grid(mode)
        targets = data.get("targets", {})
        target_data = targets.get(target, {})
        neg_results = target_data.get("negative_vs_neutral", [])
        pos_results = target_data.get("positive_vs_neutral", [])
        all_results = neg_results + pos_results
        if not all_results:
            continue
        clf_names = data['classifiers']
        horizons = data['horizons']

        best_per_horizon = []
        for h in horizons:
            h_results = [r for r in all_results if r.get('horizon') == h and not r.get('skipped')]
            best = float('nan')
            for r in h_results:
                for cn in clf_names:
                    val = r.get(f'{cn}_roc_auc', float('nan'))
                    if not np.isnan(val) and (np.isnan(best) or val > best):
                        best = val
            best_per_horizon.append(best)

        color = colors.get(mode, _palette[i_mode % len(_palette)])
        fig.add_trace(go.Scatter(
            x=horizons,
            y=best_per_horizon,
            mode='lines+markers',
            name=mode,
            line=dict(color=color),
        ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="red")
    if bow_best_auc is not None and bow_horizons is not None:
        fig.add_trace(go.Scatter(
            x=bow_horizons,
            y=bow_best_auc,
            mode='lines',
            name="BOW",
            line=dict(color='black', width=3, dash='dash'),
        ))

    fig.update_layout(
        title=f"Binary - {target}<br>Best ROC AUC vs Horizon — Encoders vs BOW",
        xaxis_title="Horizon (days)",
        yaxis_title="ROC AUC",
        yaxis_range=[0.45, 0.75],
        width=1200, height=550,
    )
    (out_dir / "roc_auc").mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_dir / "roc_auc" / f"comparison_{target}.html"))

    # --- PR AUC comparison ---
    fig_pr = go.Figure()
    for i_mode, mode in enumerate(encoder_modes):
        data = load_binary_grid(mode)
        targets = data.get("targets", {})
        target_data = targets.get(target, {})
        neg_results = target_data.get("negative_vs_neutral", [])
        pos_results = target_data.get("positive_vs_neutral", [])
        all_results = neg_results + pos_results
        if not all_results:
            continue
        clf_names = data['classifiers']
        horizons = data['horizons']

        best_per_horizon = []
        for h in horizons:
            h_results = [r for r in all_results if r.get('horizon') == h and not r.get('skipped')]
            best = float('nan')
            for r in h_results:
                for cn in clf_names:
                    val = r.get(f'{cn}_pr_auc', float('nan'))
                    if not np.isnan(val) and (np.isnan(best) or val > best):
                        best = val
            best_per_horizon.append(best)

        color = colors.get(mode, _palette[i_mode % len(_palette)])
        fig_pr.add_trace(go.Scatter(
            x=horizons,
            y=best_per_horizon,
            mode='lines+markers',
            name=mode,
            line=dict(color=color),
        ))

    # For PR AUC comparison: no single random hline makes sense (varies by quantile)
    # Show a rough midpoint reference instead
    fig_pr.add_hline(y=0.25, line_dash="dot", line_color="red",
                     annotation_text="~Random (mid-quantile)")
    if bow_best_pr_auc is not None and bow_horizons is not None:
        fig_pr.add_trace(go.Scatter(
            x=bow_horizons,
            y=bow_best_pr_auc,
            mode='lines',
            name="BOW",
            line=dict(color='black', width=3, dash='dash'),
        ))

    fig_pr.update_layout(
        title=f"Binary - {target}<br>Best PR AUC vs Horizon — Encoders vs BOW",
        xaxis_title="Horizon (days)",
        yaxis_title="PR AUC",
        yaxis_range=[0.2, 0.75],
        width=1200, height=550,
    )
    (out_dir / "pr_auc").mkdir(parents=True, exist_ok=True)
    fig_pr.write_html(str(out_dir / "pr_auc" / f"comparison_pr_auc_{target}.html"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-discover grid_search_binary_*.json
    modes = []
    for path in sorted(GRID_DIR.glob("grid_search_binary_*.json")):
        mode = path.stem.replace("grid_search_binary_", "")
        modes.append(mode)

    if not modes:
        print(f"No grid_search_binary_*.json found in {GRID_DIR}")
        return

    print(f"Found binary grid search results for: {modes}")

    # Load BOW data as baseline reference
    bow_grid = None
    bow_clf_names = None
    if 'bow' in modes:
        bow_grid = load_binary_grid('bow')
        bow_clf_names = bow_grid['classifiers']
        print("  Using BOW as baseline for encoder plots")

    # Determine all targets from first file
    sample_data = load_binary_grid(modes[0])
    all_targets = list(sample_data.get("targets", {}).keys())
    print(f"Targets: {all_targets}")

    for mode in modes:
        print(f"\n--- {mode.upper()} ---")
        data = load_binary_grid(mode)
        clf_names = data['classifiers']
        horizons = data['horizons']
        quantiles = data['quantiles']
        targets = data.get("targets", {})

        # For BOW itself, don't subtract BOW from BOW
        use_bow = bow_grid if mode != 'bow' else None

        for target_name, target_data in targets.items():
            neg_results = target_data.get("negative_vs_neutral", [])
            pos_results = target_data.get("positive_vs_neutral", [])
            if not neg_results and not pos_results:
                continue

            # Get BOW reference for this target
            bow_neg = None
            bow_pos = None
            if use_bow:
                bow_targets = use_bow.get("targets", {})
                bow_td = bow_targets.get(target_name, {})
                bow_neg = bow_td.get("negative_vs_neutral", [])
                bow_pos = bow_td.get("positive_vs_neutral", [])

            # Route to per-target subdirectory
            target_dir = OUTPUT_DIR / target_name
            target_dir.mkdir(parents=True, exist_ok=True)

            has_horizon = any(
                r.get('horizon') is not None
                for r in (neg_results + pos_results)
            )
            if has_horizon:
                plot_combined_heatmaps(neg_results, pos_results, clf_names,
                                       horizons, quantiles, mode, target_name, target_dir)
                print(f"  + {target_name} (heatmap)")
            else:
                plot_combined_surprise(neg_results, pos_results, clf_names,
                                       quantiles, mode, target_name, target_dir,
                                       bow_neg=bow_neg, bow_pos=bow_pos, bow_clf_names=bow_clf_names if use_bow else None)
                print(f"  + {target_name} (surprise bar)")

    # Cross-mode comparisons
    if len(modes) > 1:
        comparison_dir = OUTPUT_DIR / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        for target_name in all_targets:
            sample_results = (
                sample_data["targets"].get(target_name, {}).get("positive_vs_neutral", [])
                + sample_data["targets"].get(target_name, {}).get("negative_vs_neutral", [])
            )
            has_horizon = any(r.get('horizon') is not None for r in sample_results)
            if not has_horizon:
                continue
            plot_comparison_across_modes(modes, target_name, comparison_dir)
        print(f"\n+ Cross-mode comparison plots")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
