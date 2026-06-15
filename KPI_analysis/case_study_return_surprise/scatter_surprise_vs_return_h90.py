"""
Scatter: trailing 90d return (Close[t0]/Close[t0-90]-1) vs earnings surprise.

t0 = earnings date + 2 trading days (same anchor as the grid search).
The return is the "true" trailing return (price now vs 90 trading days earlier),
derived from the stored return_t{-90} = Close[t0-90]/Close[t0]-1 via r_inv = -r/(r+1).

Question: does a very large positive surprise coincide with a very negative
trailing return (or vice-versa)? -> look at the joint distribution + correlation.

Usage:
    uv run python KPI_analysis/scatter_surprise_vs_return_h90.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from threshold_grid_search import (
    HERE,
    build_records,
    load_letter_texts,
    SELECTED_COMPANIES_JSON,
)

HORIZON = -90
OUTPUT_DIR = HERE / "output" / "plots" / "threshold_grid_search"


def invert_return(r: float) -> float:
    """Close[t0]/Close[t0-90]-1 from Close[t0-90]/Close[t0]-1."""
    denom = r + 1.0
    if denom == 0.0:
        return float("nan")
    return 1.0 / denom - 1.0


def main():
    with open(SELECTED_COMPANIES_JSON) as f:
        companies_data = json.load(f)
    all_tickers = set()
    for industry, exchanges in companies_data.items():
        for exchange, companies in exchanges.items():
            for company in companies:
                all_tickers.add(company["ticker"])

    print("Loading CEO letter texts...")
    letter_texts = load_letter_texts()
    records = build_records(all_tickers, letter_texts)
    print(f"Total records: {len(records)}")

    xs, ys, labels = [], [], []
    for r in records:
        s = r.get("surprise")
        if s is None or not np.isfinite(s):
            continue
        if HORIZON not in r["returns"]:
            continue
        rv = r["returns"][HORIZON]
        if rv is None or not np.isfinite(rv):
            continue
        trailing = invert_return(float(rv))  # Close[t0]/Close[t0-90]-1
        if not np.isfinite(trailing):
            continue
        xs.append(float(s))
        ys.append(trailing)
        labels.append(f"{r['ticker']}_{r['year']}")

    xs = np.array(xs)
    ys = np.array(ys)
    n = len(xs)
    print(f"Points with both surprise and return_t{HORIZON}: {n}")

    pear_r, pear_p = stats.pearsonr(xs, ys)
    spear_r, spear_p = stats.spearmanr(xs, ys)
    print(f"Pearson  r = {pear_r:+.3f} (p={pear_p:.3g})")
    print(f"Spearman r = {spear_r:+.3f} (p={spear_p:.3g})")

    # --- main scatter (clipped axes to keep outliers from squashing the cloud) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, clip in [(ax1, False), (ax2, True)]:
        ax.scatter(xs, ys, s=18, alpha=0.5, edgecolors="none", color="#2c6fbb")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        # OLS fit line
        b1, b0 = np.polyfit(xs, ys, 1)
        xg = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xg, b0 + b1 * xg, color="crimson", lw=1.5,
                label=f"OLS slope={b1:+.3f}")
        ax.set_xlabel("Earnings surprise")
        ax.set_ylabel(r"Trailing 90d return  $Close[t_0]/Close[t_0-90]-1$")
        ax.legend(loc="upper right", fontsize=9)
        if clip:
            xlo, xhi = np.quantile(xs, [0.02, 0.98])
            ylo, yhi = np.quantile(ys, [0.02, 0.98])
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.set_title("Zoom (2-98% quantile clip)")
        else:
            ax.set_title("Full range")

    fig.suptitle(
        f"Surprise vs trailing-90d return  (t0 = earnings+2, n={n})\n"
        f"Pearson r={pear_r:+.3f} (p={pear_p:.2g})   |   Spearman r={spear_r:+.3f} (p={spear_p:.2g})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUT_DIR / "scatter_surprise_vs_return_h90.png"
    fig.savefig(out_png, dpi=130)
    print(f"\nSaved {out_png}")

    # --- quadrant counts (does high surprise avoid very low return?) ---
    print("\nQuadrant counts (sign of surprise x sign of trailing return):")
    for sx, sy, name in [
        (1, 1, "surprise>0 & return>0"),
        (1, -1, "surprise>0 & return<0"),
        (-1, 1, "surprise<0 & return>0"),
        (-1, -1, "surprise<0 & return<0"),
    ]:
        m = ((np.sign(xs) == sx) & (np.sign(ys) == sy)).sum()
        print(f"  {name:28s}: {m:4d}  ({m/n*100:5.1f}%)")

    # extreme cells: top/bottom 10% surprise vs their return distribution
    q_hi = np.quantile(xs, 0.90)
    q_lo = np.quantile(xs, 0.10)
    hi_mask = xs >= q_hi
    lo_mask = xs <= q_lo
    print(f"\nTop 10% surprise (n={hi_mask.sum()}): trailing return mean={ys[hi_mask].mean():+.3f}, median={np.median(ys[hi_mask]):+.3f}")
    print(f"Bot 10% surprise (n={lo_mask.sum()}): trailing return mean={ys[lo_mask].mean():+.3f}, median={np.median(ys[lo_mask]):+.3f}")


if __name__ == "__main__":
    main()
