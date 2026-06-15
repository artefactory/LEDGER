"""
Plot heatmaps from threshold grid search results.

For raw_return and residual targets: horizon × threshold heatmaps (best AUC, best accuracy).
For surprise target: bar charts by threshold.

Usage:
    uv run python KPI_analysis/plot_grid_search_heatmaps.py
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
GRID_DIR = HERE / "output" / "plots" / "threshold_grid_search"
OUTPUT_DIR = GRID_DIR / "heatmaps"


def load_grid(mode: str) -> dict:
    path = GRID_DIR / f"grid_search_{mode}.json"
    with open(path) as f:
        return json.load(f)


def build_heatmap_data(results: list, clf_names: list, horizons: list, thresholds: list):
    """Build 2D arrays (horizon × threshold) for best AUC, best PR AUC, and best accuracy."""
    n_h = len(horizons)
    n_t = len(thresholds)
    h_idx = {h: i for i, h in enumerate(horizons)}
    t_idx = {t: i for i, t in enumerate(thresholds)}

    best_auc = np.full((n_h, n_t), np.nan)
    best_acc = np.full((n_h, n_t), np.nan)
    best_pr_auc = np.full((n_h, n_t), np.nan)
    best_auc_clf = [['' for _ in range(n_t)] for _ in range(n_h)]
    best_acc_clf = [['' for _ in range(n_t)] for _ in range(n_h)]
    best_pr_auc_clf = [['' for _ in range(n_t)] for _ in range(n_h)]
    baselines = np.full((n_h, n_t), np.nan)

    for r in results:
        if r.get('skipped'):
            continue
        h = r.get('horizon')
        t = r.get('threshold')
        if h not in h_idx or t not in t_idx:
            continue
        hi, ti = h_idx[h], t_idx[t]
        baselines[hi, ti] = r['baseline']

        for cn in clf_names:
            auc = r.get(f'{cn}_roc_auc', float('nan'))
            acc = r.get(f'{cn}_accuracy', float('nan'))
            pr_auc = r.get(f'{cn}_pr_auc', float('nan'))
            if not np.isnan(auc) and (np.isnan(best_auc[hi, ti]) or auc > best_auc[hi, ti]):
                best_auc[hi, ti] = auc
                best_auc_clf[hi][ti] = cn
            if not np.isnan(acc) and (np.isnan(best_acc[hi, ti]) or acc > best_acc[hi, ti]):
                best_acc[hi, ti] = acc
                best_acc_clf[hi][ti] = cn
            if not np.isnan(pr_auc) and (np.isnan(best_pr_auc[hi, ti]) or pr_auc > best_pr_auc[hi, ti]):
                best_pr_auc[hi, ti] = pr_auc
                best_pr_auc_clf[hi][ti] = cn

    return best_auc, best_acc, best_pr_auc, best_auc_clf, best_acc_clf, best_pr_auc_clf, baselines


def make_hover_text(results: list, clf_names: list, horizons: list, thresholds: list, metric: str):
    """Build hover text matrix showing all classifiers' values + best + diff vs baseline + class proportions."""
    h_idx = {h: i for i, h in enumerate(horizons)}
    t_idx = {t: i for i, t in enumerate(thresholds)}
    n_h, n_t = len(horizons), len(thresholds)
    text = [["" for _ in range(n_t)] for _ in range(n_h)]

    for r in results:
        if r.get('skipped'):
            continue
        h = r.get('horizon')
        t = r.get('threshold')
        if h not in h_idx or t not in t_idx:
            continue
        hi, ti = h_idx[h], t_idx[t]

        bl = r['baseline']
        n = r['n']
        neg = r.get('class_neg', 0)
        neu = r.get('class_neu', 0)
        pos = r.get('class_pos', 0)

        lines = [
            f"<b>h={h}, thr={t:.3f}, n={n}</b>",
            f"Classes: neg={neg} ({neg/n:.0%}) / neu={neu} ({neu/n:.0%}) / pos={pos} ({pos/n:.0%})",
            f"Baseline (majority): {bl:.3f}",
            "",
        ]

        best_val = -1
        best_name = ""
        key = f"_{metric}"  # _roc_auc or _accuracy
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
            else:
                diff = best_val - 0.5
                lines.append(f"<b>Best: {best_name} ({best_val:.4f}, vs random={diff:+.4f})</b>")

        text[hi][ti] = "<br>".join(lines)

    return text


def plot_heatmaps_for_target(data: dict, target_key: str, target_label: str, mode: str,
                             bow_data: dict | None = None):
    """Create and save heatmap figures for a given target (raw_return or residual).

    If bow_data is provided, heatmaps show lift over BOW instead of random/majority.
    """
    results = data[target_key]
    clf_names = data['classifiers']
    horizons = data['horizons']
    thresholds = data['thresholds_return']

    best_auc, best_acc, best_pr_auc, best_auc_clf, best_acc_clf, best_pr_auc_clf, baselines = build_heatmap_data(
        results, clf_names, horizons, thresholds
    )

    # Build BOW reference arrays if available
    bow_auc = None
    bow_acc = None
    bow_pr_auc = None
    if bow_data and target_key in bow_data and bow_data[target_key]:
        bow_results = bow_data[target_key]
        bow_clf_names = bow_data['classifiers']
        _, bow_acc_arr, _, _, _, _, _ = build_heatmap_data(
            bow_results, bow_clf_names, horizons, thresholds
        )
        bow_auc_arr = np.full((len(horizons), len(thresholds)), np.nan)
        bow_pr_auc_arr = np.full((len(horizons), len(thresholds)), np.nan)
        h_idx = {h: i for i, h in enumerate(horizons)}
        t_idx = {t: i for i, t in enumerate(thresholds)}
        for r in bow_results:
            if r.get('skipped'):
                continue
            h = r.get('horizon')
            t = r.get('threshold')
            if h not in h_idx or t not in t_idx:
                continue
            hi, ti = h_idx[h], t_idx[t]
            for cn in bow_clf_names:
                val = r.get(f'{cn}_roc_auc', float('nan'))
                if not np.isnan(val) and (np.isnan(bow_auc_arr[hi, ti]) or val > bow_auc_arr[hi, ti]):
                    bow_auc_arr[hi, ti] = val
                val_pr = r.get(f'{cn}_pr_auc', float('nan'))
                if not np.isnan(val_pr) and (np.isnan(bow_pr_auc_arr[hi, ti]) or val_pr > bow_pr_auc_arr[hi, ti]):
                    bow_pr_auc_arr[hi, ti] = val_pr
        bow_auc = bow_auc_arr
        bow_acc = bow_acc_arr
        bow_pr_auc = bow_pr_auc_arr

    # Format axis labels
    y_labels = [str(h) for h in horizons]
    x_labels = [f"{t:.3f}" for t in thresholds]

    # --- ROC AUC Heatmap (lift over BOW or random) ---
    if bow_auc is not None:
        auc_lift = best_auc - bow_auc
        auc_title = f"{mode.upper()} — {target_label} — ROC AUC lift over BOW (horizon × threshold)"
        auc_cbar = "AUC - BOW"
    else:
        auc_lift = best_auc - 0.5
        auc_title = f"{mode.upper()} — {target_label} — ROC AUC lift over random (horizon × threshold)"
        auc_cbar = "AUC - 0.5"
    hover_auc = make_hover_text(results, clf_names, horizons, thresholds, "roc_auc")
    fig_auc = go.Figure(data=go.Heatmap(
        z=auc_lift,
        x=x_labels,
        y=y_labels,
        text=hover_auc,
        hoverinfo='text',
        colorscale='RdYlGn',
        zmid=0,
        zmin=-0.15,
        zmax=0.15,
        colorbar=dict(title=auc_cbar),
    ))
    fig_auc.update_layout(
        title=auc_title,
        xaxis_title="Threshold",
        yaxis_title="Horizon (days)",
        width=900, height=700,
    )
    fig_auc.write_html(str(OUTPUT_DIR / f"heatmap_auc_{mode}_{target_key}.html"))

    # --- Accuracy Heatmap (lift over BOW or majority baseline) ---
    if bow_acc is not None:
        lift_matrix = best_acc - bow_acc
        acc_title = f"{mode.upper()} — {target_label} — Accuracy Lift over BOW (horizon × threshold)"
        acc_cbar = "Lift (Acc - BOW)"
    else:
        lift_matrix = best_acc - baselines
        acc_title = f"{mode.upper()} — {target_label} — Accuracy Lift over Baseline (horizon × threshold)"
        acc_cbar = "Lift (Acc - Baseline)"
    hover_acc = make_hover_text(results, clf_names, horizons, thresholds, "accuracy")
    fig_acc = go.Figure(data=go.Heatmap(
        z=lift_matrix,
        x=x_labels,
        y=y_labels,
        text=hover_acc,
        hoverinfo='text',
        colorscale='RdBu',
        zmid=0,
        zmin=-0.15,
        zmax=0.15,
        colorbar=dict(title=acc_cbar),
    ))
    fig_acc.update_layout(
        title=acc_title,
        xaxis_title="Threshold",
        yaxis_title="Horizon (days)",
        width=900, height=700,
    )
    fig_acc.write_html(str(OUTPUT_DIR / f"heatmap_lift_{mode}_{target_key}.html"))

    # --- PR AUC Heatmap (lift over BOW or random 1/3) ---
    if bow_pr_auc is not None:
        pr_lift = best_pr_auc - bow_pr_auc
        pr_title = f"{mode.upper()} — {target_label} — PR AUC lift over BOW (horizon × threshold)"
        pr_cbar = "PR AUC - BOW"
    else:
        pr_lift = best_pr_auc - (1.0 / 3)
        pr_title = f"{mode.upper()} — {target_label} — PR AUC lift over random (horizon × threshold)"
        pr_cbar = "PR AUC - 0.33"
    hover_pr = make_hover_text(results, clf_names, horizons, thresholds, "pr_auc")
    fig_pr = go.Figure(data=go.Heatmap(
        z=pr_lift,
        x=x_labels,
        y=y_labels,
        text=hover_pr,
        hoverinfo='text',
        colorscale='RdYlGn',
        zmid=0,
        zmin=-0.15,
        zmax=0.15,
        colorbar=dict(title=pr_cbar),
    ))
    fig_pr.update_layout(
        title=pr_title,
        xaxis_title="Threshold",
        yaxis_title="Horizon (days)",
        width=900, height=700,
    )
    fig_pr.write_html(str(OUTPUT_DIR / f"heatmap_pr_auc_{mode}_{target_key}.html"))

    return fig_auc, fig_acc, fig_pr


def plot_surprise(data: dict, mode: str, bow_data: dict | None = None):
    """Create bar charts for surprise target (no horizon dimension)."""
    results = data['surprise_results']
    clf_names = data['classifiers']

    # Build arrays per classifier
    auc_by_clf = {cn: [] for cn in clf_names}
    acc_by_clf = {cn: [] for cn in clf_names}
    pr_auc_by_clf = {cn: [] for cn in clf_names}
    baselines = []
    valid_thresholds = []
    n_samples = []
    prop_neg = []
    prop_neu = []
    prop_pos = []

    for r in results:
        if r.get('skipped'):
            continue
        valid_thresholds.append(r['threshold'])
        baselines.append(r['baseline'])
        n = r.get('n', 0)
        neg = r.get('class_neg', 0)
        neu = r.get('class_neu', 0)
        pos = r.get('class_pos', 0)
        n_samples.append(n)
        if n > 0:
            prop_neg.append(neg / n)
            prop_neu.append(neu / n)
            prop_pos.append(pos / n)
        else:
            prop_neg.append(np.nan)
            prop_neu.append(np.nan)
            prop_pos.append(np.nan)
        for cn in clf_names:
            auc_by_clf[cn].append(r.get(f'{cn}_roc_auc', float('nan')))
            acc_by_clf[cn].append(r.get(f'{cn}_accuracy', float('nan')))
            pr_auc_by_clf[cn].append(r.get(f'{cn}_pr_auc', float('nan')))

    # Build BOW reference values per threshold
    bow_auc_vals = None
    bow_acc_vals = None
    bow_pr_auc_vals = None
    if bow_data and 'surprise_results' in bow_data:
        bow_results = bow_data['surprise_results']
        bow_clf_names = bow_data['classifiers']
        bow_auc_vals = []
        bow_acc_vals = []
        bow_pr_auc_vals = []
        bow_by_thr = {r['threshold']: r for r in bow_results if not r.get('skipped')}
        for t in valid_thresholds:
            r = bow_by_thr.get(t)
            if r:
                best_auc = max((r.get(f'{cn}_roc_auc', float('nan')) for cn in bow_clf_names), default=float('nan'))
                best_acc = max((r.get(f'{cn}_accuracy', float('nan')) for cn in bow_clf_names), default=float('nan'))
                best_pr = max((r.get(f'{cn}_pr_auc', float('nan')) for cn in bow_clf_names), default=float('nan'))
                bow_auc_vals.append(best_auc if not np.isnan(best_auc) else float('nan'))
                bow_acc_vals.append(best_acc if not np.isnan(best_acc) else float('nan'))
                bow_pr_auc_vals.append(best_pr if not np.isnan(best_pr) else float('nan'))
            else:
                bow_auc_vals.append(float('nan'))
                bow_acc_vals.append(float('nan'))
                bow_pr_auc_vals.append(float('nan'))

    x_labels = [f"{t:.1f}%" for t in valid_thresholds]
    customdata = np.column_stack([
        np.array(valid_thresholds, dtype=np.float64),
        np.array(n_samples, dtype=np.float64),
        np.array(baselines, dtype=np.float64),
        np.array(prop_neg, dtype=np.float64),
        np.array(prop_neu, dtype=np.float64),
        np.array(prop_pos, dtype=np.float64),
    ])

    # --- ROC AUC bar chart ---
    fig_auc = go.Figure()
    for cn in clf_names:
        fig_auc.add_trace(go.Bar(
            name=cn,
            x=x_labels,
            y=auc_by_clf[cn],
            customdata=customdata,
            text=[f"{v:.3f}" for v in auc_by_clf[cn]],
            textposition='auto',
            hovertemplate=(
                "<b>Threshold %{customdata[0]:.1f}%</b><br>"
                "Classifier: " + cn + "<br>"
                "ROC AUC: %{y:.4f}<br>"
                "Baseline: %{customdata[2]:.4f}<br>"
                "n: %{customdata[1]:.0f}<br>"
                "Class proportions: neg=%{customdata[3]:.1%}, "
                "neu=%{customdata[4]:.1%}, pos=%{customdata[5]:.1%}"
                "<extra></extra>"
            ),
        ))
    fig_auc.add_hline(y=0.5, line_dash="dot", line_color="red",
                      annotation_text="Random (0.5)")
    if bow_auc_vals is not None:
        fig_auc.add_trace(go.Scatter(
            name="BOW baseline",
            x=x_labels,
            y=bow_auc_vals,
            mode='lines+markers',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8),
        ))
    fig_auc.update_layout(
        title=f"{mode.upper()} — Surprise — ROC AUC by Threshold",
        xaxis_title="Surprise Threshold (%)",
        yaxis_title="ROC AUC",
        yaxis_range=[0.4, 0.75],
        barmode='group',
        width=1600, height=700,
    )
    fig_auc.write_html(str(OUTPUT_DIR / f"surprise_auc_{mode}.html"))

    # --- PR AUC bar chart ---
    fig_pr = go.Figure()
    for cn in clf_names:
        fig_pr.add_trace(go.Bar(
            name=cn,
            x=x_labels,
            y=pr_auc_by_clf[cn],
            customdata=customdata,
            text=[f"{v:.3f}" for v in pr_auc_by_clf[cn]],
            textposition='auto',
            hovertemplate=(
                "<b>Threshold %{customdata[0]:.1f}%</b><br>"
                "Classifier: " + cn + "<br>"
                "PR AUC: %{y:.4f}<br>"
                "n: %{customdata[1]:.0f}<br>"
                "Class proportions: neg=%{customdata[3]:.1%}, "
                "neu=%{customdata[4]:.1%}, pos=%{customdata[5]:.1%}"
                "<extra></extra>"
            ),
        ))
    fig_pr.add_hline(y=1.0/3, line_dash="dot", line_color="red",
                      annotation_text="Random (0.33)")
    if bow_pr_auc_vals is not None:
        fig_pr.add_trace(go.Scatter(
            name="BOW baseline",
            x=x_labels,
            y=bow_pr_auc_vals,
            mode='lines+markers',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8),
        ))
    fig_pr.update_layout(
        title=f"{mode.upper()} — Surprise — PR AUC by Threshold",
        xaxis_title="Surprise Threshold (%)",
        yaxis_title="PR AUC",
        yaxis_range=[0.2, 0.75],
        barmode='group',
        width=1600, height=700,
    )
    fig_pr.write_html(str(OUTPUT_DIR / f"surprise_pr_auc_{mode}.html"))

    # --- Accuracy vs baseline bar chart ---
    fig_acc = go.Figure()
    for cn in clf_names:
        fig_acc.add_trace(go.Bar(
            name=cn,
            x=x_labels,
            y=acc_by_clf[cn],
            customdata=customdata,
            hovertemplate=(
                "<b>Threshold %{customdata[0]:.1f}%</b><br>"
                "Classifier: " + cn + "<br>"
                "Accuracy: %{y:.4f}<br>"
                "Baseline: %{customdata[2]:.4f}<br>"
                "Lift: %{y-customdata[2]:+.4f}<br>"
                "n: %{customdata[1]:.0f}<br>"
                "Class proportions: neg=%{customdata[3]:.1%}, "
                "neu=%{customdata[4]:.1%}, pos=%{customdata[5]:.1%}"
                "<extra></extra>"
            ),
        ))
    fig_acc.add_trace(go.Scatter(
        name="Majority Baseline",
        x=x_labels,
        y=baselines,
        customdata=customdata,
        mode='lines+markers',
        line=dict(color='gray', width=2, dash='dot'),
        marker=dict(size=6),
    ))
    if bow_acc_vals is not None:
        fig_acc.add_trace(go.Scatter(
            name="BOW baseline",
            x=x_labels,
            y=bow_acc_vals,
            mode='lines+markers',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8),
        ))
    fig_acc.update_layout(
        title=f"{mode.upper()} — Surprise — Accuracy vs BOW Baseline by Threshold",
        xaxis_title="Surprise Threshold (%)",
        yaxis_title="Accuracy",
        yaxis_range=[0.3, 0.85],
        barmode='group',
        width=1600, height=700,
    )
    fig_acc.write_html(str(OUTPUT_DIR / f"surprise_acc_{mode}.html"))

    return fig_auc, fig_acc, fig_pr


def plot_comparison_across_modes(modes: list[str]):
    """Create a summary comparison of best AUC/PR AUC per (target, mode) at each horizon.

    BOW is plotted as the baseline reference line (dashed black).
    """
    # --- ROC AUC comparison ---
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Raw Return — Best ROC AUC vs Horizon",
                        "Residual — Best ROC AUC vs Horizon",
                        "Residual Inliers — Best ROC AUC vs Horizon"],
    )

    colors = {'minilm': '#1f77b4', 'roberta': '#9467bd', 'eurobert': '#ff7f0e',
              'mpnet': '#d62728', 'minilm-l12': '#8c564b'}
    _palette = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Collect BOW data first for baseline
    bow_best_per_target: dict[str, list[float]] = {}
    bow_horizons = None
    if 'bow' in modes:
        bow_grid = load_grid('bow')
        bow_horizons = bow_grid['horizons']
        bow_clf_names = bow_grid['classifiers']
        for target_key in ['raw_return_results', 'unbiased_return_results', 'residual_results', 'residual_inliers_results']:
            if target_key not in bow_grid or not bow_grid[target_key]:
                bow_best_per_target[target_key] = []
                continue
            results = bow_grid[target_key]
            best_per_h = []
            for h in bow_horizons:
                h_results = [r for r in results if r.get('horizon') == h and not r.get('skipped')]
                best = float('nan')
                for r in h_results:
                    for cn in bow_clf_names:
                        val = r.get(f'{cn}_roc_auc', float('nan'))
                        if not np.isnan(val) and (np.isnan(best) or val > best):
                            best = val
                best_per_h.append(best)
            bow_best_per_target[target_key] = best_per_h

    encoder_modes = [m for m in modes if m != 'bow']

    for col, target_key in enumerate(['raw_return_results', 'unbiased_return_results', 'residual_results', 'residual_inliers_results'], 1):
        for i_mode, mode in enumerate(encoder_modes):
            data = load_grid(mode)
            if target_key not in data or not data[target_key]:
                continue
            results = data[target_key]
            clf_names = data['classifiers']
            horizons = data['horizons']

            best_per_horizon = []
            for h in horizons:
                h_results = [r for r in results if r.get('horizon') == h and not r.get('skipped')]
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
                name=f"{mode}" if col == 1 else None,
                legendgroup=mode,
                showlegend=(col == 1),
                line=dict(color=color),
            ), row=1, col=col)

        # Add random baseline
        fig.add_hline(y=0.5, line_dash="dot", line_color="red", row=1, col=col)
        # Add BOW baseline
        if target_key in bow_best_per_target and bow_best_per_target[target_key]:
            fig.add_trace(go.Scatter(
                x=bow_horizons,
                y=bow_best_per_target[target_key],
                mode='lines',
                name="BOW" if col == 1 else None,
                legendgroup="bow",
                showlegend=(col == 1),
                line=dict(color='black', width=3, dash='dash'),
            ), row=1, col=col)

    fig.update_layout(
        title="Best ROC AUC vs Horizon — Encoders vs BOW Baseline (best threshold per horizon)",
        width=1500, height=500,
    )
    fig.update_yaxes(range=[0.45, 0.75])
    fig.update_xaxes(title_text="Horizon (days)")
    fig.write_html(str(OUTPUT_DIR / "comparison_auc_vs_horizon.html"))

    # --- PR AUC comparison ---
    fig_pr = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Raw Return — Best PR AUC vs Horizon",
                        "Residual — Best PR AUC vs Horizon",
                        "Residual Inliers — Best PR AUC vs Horizon"],
    )

    # Collect BOW PR AUC baseline
    bow_pr_per_target: dict[str, list[float]] = {}
    if 'bow' in modes:
        for target_key in ['raw_return_results', 'unbiased_return_results', 'residual_results', 'residual_inliers_results']:
            if target_key not in bow_grid or not bow_grid[target_key]:
                bow_pr_per_target[target_key] = []
                continue
            results = bow_grid[target_key]
            best_per_h = []
            for h in bow_horizons:
                h_results = [r for r in results if r.get('horizon') == h and not r.get('skipped')]
                best = float('nan')
                for r in h_results:
                    for cn in bow_clf_names:
                        val = r.get(f'{cn}_pr_auc', float('nan'))
                        if not np.isnan(val) and (np.isnan(best) or val > best):
                            best = val
                best_per_h.append(best)
            bow_pr_per_target[target_key] = best_per_h

    for col, target_key in enumerate(['raw_return_results', 'unbiased_return_results', 'residual_results', 'residual_inliers_results'], 1):
        for i_mode, mode in enumerate(encoder_modes):
            data = load_grid(mode)
            if target_key not in data or not data[target_key]:
                continue
            results = data[target_key]
            clf_names = data['classifiers']
            horizons = data['horizons']

            best_per_horizon = []
            for h in horizons:
                h_results = [r for r in results if r.get('horizon') == h and not r.get('skipped')]
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
                name=f"{mode}" if col == 1 else None,
                legendgroup=mode,
                showlegend=(col == 1),
                line=dict(color=color),
            ), row=1, col=col)

        # Add random baseline
        fig_pr.add_hline(y=1.0/3, line_dash="dot", line_color="red", row=1, col=col)
        # Add BOW baseline
        if target_key in bow_pr_per_target and bow_pr_per_target[target_key]:
            fig_pr.add_trace(go.Scatter(
                x=bow_horizons,
                y=bow_pr_per_target[target_key],
                mode='lines',
                name="BOW" if col == 1 else None,
                legendgroup="bow",
                showlegend=(col == 1),
                line=dict(color='black', width=3, dash='dash'),
            ), row=1, col=col)

    fig_pr.update_layout(
        title="Best PR AUC vs Horizon — Encoders vs BOW Baseline (best threshold per horizon)",
        width=1500, height=500,
    )
    fig_pr.update_yaxes(range=[0.2, 0.65])
    fig_pr.update_xaxes(title_text="Horizon (days)")
    fig_pr.write_html(str(OUTPUT_DIR / "comparison_pr_auc_vs_horizon.html"))

    return fig, fig_pr


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-discover all grid_search_*.json files
    modes = []
    for path in sorted(GRID_DIR.glob("grid_search_*.json")):
        mode = path.stem.replace("grid_search_", "")
        modes.append(mode)

    print(f"Found grid search results for: {modes}")

    # Load BOW data as baseline reference (if available)
    bow_data = None
    if 'bow' in modes:
        bow_data = load_grid('bow')
        print("  Using BOW as baseline for encoder plots")

    for mode in modes:
        print(f"\n--- {mode.upper()} ---")
        data = load_grid(mode)
        # For BOW itself, don't subtract BOW from BOW (use random/majority baseline)
        ref_data = bow_data if mode != 'bow' else None

        # Raw return heatmaps
        if 'raw_return_results' in data and data['raw_return_results']:
            plot_heatmaps_for_target(data, 'raw_return_results', 'Raw Return', mode)
            print(f"  ✓ Raw return heatmaps")

        # Unbiased return (industry-adjusted) heatmaps
        if 'unbiased_return_results' in data and data['unbiased_return_results']:
            plot_heatmaps_for_target(data, 'unbiased_return_results', 'Unbiased Return', mode)
            print(f"  ✓ Unbiased return heatmaps")

        # Residual heatmaps
        if 'residual_results' in data and data['residual_results']:
            plot_heatmaps_for_target(data, 'residual_results', 'Residual', mode)
            print(f"  ✓ Residual heatmaps")

        # Residual inliers heatmaps
        if 'residual_inliers_results' in data and data['residual_inliers_results']:
            plot_heatmaps_for_target(data, 'residual_inliers_results', 'Residual Inliers (q2-q98)', mode)
            print(f"  ✓ Residual inliers heatmaps")

        # Surprise bar charts
        if 'surprise_results' in data and data['surprise_results']:
            plot_surprise(data, mode, bow_data=ref_data)
            print(f"  ✓ Surprise bar charts")

    # Cross-mode comparison
    if len(modes) > 1:
        plot_comparison_across_modes(modes)
        print(f"\n✓ Cross-mode comparison plots")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
