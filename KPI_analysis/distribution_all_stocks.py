"""
Distribution plots for all selected stocks at 10-K filing date:
  - Return at t+1
  - Mean volatility t+1 to t+5
  - Mean volume ATS t+1 to t+5

Uses the same infrastructure as event_study_earnings.py but focuses on
simple distributional views across ALL selected companies.

Outputs:
  output/plots/distribution_all_stocks/
"""

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from FinancialIndicators import GetIndicatorsForPrices, GetIndustryDataFrame
from fetch_filing_returns import fetch_prices
from plot_indicators import annual_publication_dates

HERE = Path(__file__).resolve().parent
SELECTED_COMPANIES_JSON = (
    HERE.parent / "tickers_lists" / "grouped" / "selected" / "companies.json"
)
OUTPUT_DIR = HERE / "output" / "plots" / "distribution_all_stocks"

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2023, 6, 30)
EVENT_HALF_WINDOW = 10


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all selected tickers
    with open(SELECTED_COMPANIES_JSON) as f:
        companies_data = json.load(f)

    all_entries = []
    for industry, exchanges in companies_data.items():
        for exchange, companies in exchanges.items():
            for company in companies:
                ticker = company["ticker"]
                for year in range(2017, 2023):
                    all_entries.append({
                        "industry": industry,
                        "ticker": ticker,
                        "year": year,
                    })

    print(f"Loaded {len(all_entries)} entries")

    processed_tickers: dict[str, tuple] = {}
    industry_dfs: dict[str, pd.DataFrame] = {}  # ticker -> industry_df
    records = []

    for i, entry in enumerate(all_entries):
        ticker = entry["ticker"]
        year = entry["year"]
        industry = entry["industry"]

        # Get filing date
        pubs = annual_publication_dates(ticker, originals_only=True)
        if pubs.empty:
            continue
        pubs = pubs.copy()
        pubs["report_date"] = pd.to_datetime(pubs["report_date"])
        pubs["fy"] = pubs["report_date"].apply(lambda d: d.year - 1 if d.month <= 3 else d.year)
        match = pubs[pubs["fy"] == year]
        if match.empty:
            continue

        filing_dt = pd.to_datetime(match.iloc[0]["publication_date_et"])
        if filing_dt.tzinfo is not None:
            filing_dt = filing_dt.tz_localize(None)
        filing_dt = filing_dt.normalize()

        # Get prices
        if ticker not in processed_tickers:
            prices = fetch_prices(ticker, BENCH_START, BENCH_END)
            if prices is None or prices.empty:
                processed_tickers[ticker] = (None,)
                industry_dfs[ticker] = None
                continue
            prices = GetIndicatorsForPrices(prices, max_lag=EVENT_HALF_WINDOW)
            processed_tickers[ticker] = (prices,)
            industry_dfs[ticker] = GetIndustryDataFrame(ticker, BENCH_START, BENCH_END, max_lag=EVENT_HALF_WINDOW)
        else:
            (prices,) = processed_tickers[ticker]

        if prices is None:
            continue

        # Snap to trading day
        pub_ts = pd.Timestamp(filing_dt)
        if pub_ts not in prices.index:
            mask = prices.index >= pub_ts
            if mask.sum() == 0:
                continue
            pub_ts = prices.index[mask][0]

        t0_pos = prices.index.get_loc(pub_ts)
        if t0_pos + 5 >= len(prices):
            continue
        if t0_pos < 10 or t0_pos + 10 >= len(prices):
            continue

        # Return at t+1
        return_t1 = prices.loc[pub_ts, "return_t1"] if "return_t1" in prices.columns else None

        # Mean volatility t+1 to t+5
        vols = []
        for d in range(1, 6):
            pos = t0_pos + d
            if pos < len(prices):
                vols.append(prices.iloc[pos]["Volatility"])
        mean_vol_1_5 = np.nanmean(vols) if vols else None

        # Mean volume ATS t+1 to t+5
        volumes = []
        for d in range(1, 6):
            pos = t0_pos + d
            if pos < len(prices):
                volumes.append(prices.iloc[pos]["Volume_ATS"])
        mean_volume_1_5 = np.nanmean(volumes) if volumes else None

        # Industry volatility at event window days
        ind_df = industry_dfs.get(ticker)
        ind_vol_dict = {}
        if ind_df is not None and not ind_df.empty and "volatility" in ind_df.columns:
            ind_aligned = ind_df.reindex(prices.index)
            for d in range(-10, 11):
                pos = t0_pos + d
                if 0 <= pos < len(ind_aligned):
                    val = ind_aligned.iloc[pos]["volatility"]
                    ind_vol_dict[f"ind_vol_d{d}"] = val

        records.append({
            "ticker": ticker,
            "year": year,
            "industry": industry,
            "return_t1": return_t1,
            "mean_vol_1_5": mean_vol_1_5,
            "mean_volume_1_5": mean_volume_1_5,
            **{f"vol_d{d}": prices.iloc[t0_pos + d]["Volatility"] for d in range(-10, 11)},
            **ind_vol_dict,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(all_entries)}] processed, {len(records)} records so far")

    print(f"\nCollected {len(records)} records with valid data")

    if not records:
        print("No records. Exiting.")
        return

    df = pd.DataFrame(records)
    for c in ["return_t1", "mean_vol_1_5", "mean_volume_1_5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.to_csv(OUTPUT_DIR / "distribution_data.csv", index=False)

    # --- Plot 1: Distribution of return at t+1 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ret = df["return_t1"].dropna()
    ax.hist(ret, bins=80, color="#3498db", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(ret.mean(), color="red", linewidth=2, linestyle="--",
               label=f"Mean = {ret.mean():.4f}")
    ax.axvline(ret.median(), color="orange", linewidth=2, linestyle=":",
               label=f"Median = {ret.median():.4f}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Return at t+1 (filing date)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Return at t+1 after 10-K Filing\n(n={len(ret)} events, all selected stocks)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "dist_return_t1.png", dpi=150)
    plt.close(fig)
    print(f"Saved dist_return_t1.png")

    # --- Plot 2: Distribution of mean volatility t+1 to t+5 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    vol = df["mean_vol_1_5"].dropna()
    ax.hist(vol, bins=80, color="#e74c3c", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(vol.mean(), color="blue", linewidth=2, linestyle="--",
               label=f"Mean = {vol.mean():.4f}")
    ax.axvline(vol.median(), color="orange", linewidth=2, linestyle=":",
               label=f"Median = {vol.median():.4f}")
    ax.set_xlabel("Mean Volatility (rolling 20d std) from t+1 to t+5")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Mean Volatility [t+1, t+5] after 10-K Filing\n(n={len(vol)} events, all selected stocks)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "dist_mean_volatility_1_5.png", dpi=150)
    plt.close(fig)
    print(f"Saved dist_mean_volatility_1_5.png")

    # --- Plot 3: Distribution of mean volume ATS t+1 to t+5 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    volume = df["mean_volume_1_5"].dropna()
    ax.hist(volume, bins=80, color="#27ae60", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(volume.mean(), color="red", linewidth=2, linestyle="--",
               label=f"Mean = {volume.mean():.2f}")
    ax.axvline(volume.median(), color="orange", linewidth=2, linestyle=":",
               label=f"Median = {volume.median():.2f}")
    ax.set_xlabel("Mean Volume ATS from t+1 to t+5")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Mean Volume ATS [t+1, t+5] after 10-K Filing\n(n={len(volume)} events, all selected stocks)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "dist_mean_volume_1_5.png", dpi=150)
    plt.close(fig)
    print(f"Saved dist_mean_volume_1_5.png")

    # --- Plot 4: By industry (boxplots) ---
    industries = sorted(df["industry"].dropna().unique())

    # Return by industry
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_ind = [df[df["industry"] == ind]["return_t1"].dropna().values for ind in industries]
    short_labels = [ind.split(" / ")[-1] if " / " in ind else ind for ind in industries]
    bp = ax.boxplot(data_by_ind, labels=short_labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Return at t+1")
    ax.set_title("Return at t+1 after 10-K Filing — by Industry")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "boxplot_return_t1_by_industry.png", dpi=150)
    plt.close(fig)
    print(f"Saved boxplot_return_t1_by_industry.png")

    # Volatility by industry
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_ind = [df[df["industry"] == ind]["mean_vol_1_5"].dropna().values for ind in industries]
    bp = ax.boxplot(data_by_ind, labels=short_labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#e74c3c")
        patch.set_alpha(0.6)
    ax.set_ylabel("Mean Volatility [t+1, t+5]")
    ax.set_title("Mean Volatility [t+1, t+5] after 10-K Filing — by Industry")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "boxplot_volatility_by_industry.png", dpi=150)
    plt.close(fig)
    print(f"Saved boxplot_volatility_by_industry.png")

    # Volume by industry
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_ind = [df[df["industry"] == ind]["mean_volume_1_5"].dropna().values for ind in industries]
    bp = ax.boxplot(data_by_ind, labels=short_labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#27ae60")
        patch.set_alpha(0.6)
    ax.set_ylabel("Mean Volume ATS [t+1, t+5]")
    ax.set_title("Mean Volume ATS [t+1, t+5] after 10-K Filing — by Industry")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "boxplot_volume_by_industry.png", dpi=150)
    plt.close(fig)
    print(f"Saved boxplot_volume_by_industry.png")

    # --- Plot 5: Volatility by industry, -10 to +10 days around filing date ---
    days_range = list(range(-10, 11))
    vol_cols = [f"vol_d{d}" for d in days_range]
    ind_vol_cols = [f"ind_vol_d{d}" for d in days_range]
    for c in vol_cols + ind_vol_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 6))
    industries = sorted(df["industry"].dropna().unique())
    cmap = plt.colormaps["tab10"]
    for idx, ind in enumerate(industries):
        df_ind = df[df["industry"] == ind]
        means = [df_ind[f"vol_d{d}"].mean() for d in days_range]
        short = ind.split(" / ")[-1] if " / " in ind else ind
        ax.plot(days_range, means, color=cmap(idx % 10), linewidth=2,
                marker="o", markersize=3, label=f"{short} (n={len(df_ind)})")

    # Weighted average across all industries (weights = number of events per industry)
    all_means = [df[f"vol_d{d}"].mean() for d in days_range]
    ax.plot(days_range, all_means, color="black", linewidth=3, linestyle="-",
            marker="D", markersize=4, alpha=0.8,
            label=f"Weighted avg all (n={len(df)})")

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7, label="Filing date")
    ax.set_xticks(days_range)
    ax.set_xlabel("Trading days relative to 10-K Filing Date")
    ax.set_ylabel("Mean Stock Volatility (rolling 20d std)")
    ax.set_title("Stock Volatility around 10-K Filing Date — by Industry")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "volatility_by_industry_around_filing.png", dpi=150)
    plt.close(fig)
    print(f"Saved volatility_by_industry_around_filing.png")

    # --- Plot 6: Industry Avg volatility around filing date (from GetIndustryDataFrame) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, ind in enumerate(industries):
        df_ind = df[df["industry"] == ind]
        ind_means = []
        for d in days_range:
            col_name = f"ind_vol_d{d}"
            if col_name in df_ind.columns:
                ind_means.append(df_ind[col_name].mean())
            else:
                ind_means.append(np.nan)
        short = ind.split(" / ")[-1] if " / " in ind else ind
        ax.plot(days_range, ind_means, color=cmap(idx % 10), linewidth=2,
                marker="s", markersize=3, label=f"{short} (n={len(df_ind)})")

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7, label="Filing date")
    ax.set_xticks(days_range)
    ax.set_xlabel("Trading days relative to 10-K Filing Date")
    ax.set_ylabel("Industry Avg Volatility (GetIndustryDataFrame)")
    ax.set_title("Industry Avg Volatility around 10-K Filing Date\n(from GetIndustryDataFrame, equal-weighted mean of all peers)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "industry_avg_volatility_around_filing.png", dpi=150)
    plt.close(fig)
    print(f"Saved industry_avg_volatility_around_filing.png")

    # --- Plot 7: Histogram of industry_volatility values at filing date (d=0) ---
    if "ind_vol_d0" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ind_vol_at_0 = df["ind_vol_d0"].dropna()
        ax.hist(ind_vol_at_0, bins=80, color="#9b59b6", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(ind_vol_at_0.mean(), color="red", linewidth=2, linestyle="--",
                   label=f"Mean = {ind_vol_at_0.mean():.4f}")
        ax.axvline(ind_vol_at_0.median(), color="orange", linewidth=2, linestyle=":",
                   label=f"Median = {ind_vol_at_0.median():.4f}")
        ax.set_xlabel("Industry Avg Volatility at Filing Date (d=0)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of Industry Avg Volatility at Filing Date\n(n={len(ind_vol_at_0)}, from GetIndustryDataFrame)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "dist_industry_volatility_at_filing.png", dpi=150)
        plt.close(fig)
        print(f"Saved dist_industry_volatility_at_filing.png")

        # Histogram by industry
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()
        for idx, ind in enumerate(industries[:6]):
            ax = axes[idx]
            vals = df[df["industry"] == ind]["ind_vol_d0"].dropna()
            short = ind.split(" / ")[-1] if " / " in ind else ind
            ax.hist(vals, bins=40, color=cmap(idx % 10), alpha=0.8, edgecolor="white", linewidth=0.3)
            ax.axvline(vals.mean(), color="red", linewidth=1.5, linestyle="--",
                       label=f"μ={vals.mean():.4f}")
            ax.axvline(vals.median(), color="orange", linewidth=1.5, linestyle=":",
                       label=f"med={vals.median():.4f}")
            ax.set_title(f"{short} (n={len(vals)})", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Industry Avg Volatility at Filing Date — by Industry\n(from GetIndustryDataFrame)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "dist_industry_volatility_by_industry.png", dpi=150)
        plt.close(fig)
        print(f"Saved dist_industry_volatility_by_industry.png")

    # --- Outlier detection (IQR method) and JSON export ---
    outliers_report = {}
    metrics_for_outliers = [
        ("return_t1", "Return at t+1"),
        ("mean_vol_1_5", "Mean Volatility [t+1, t+5]"),
        ("mean_volume_1_5", "Mean Volume ATS [t+1, t+5]"),
    ]

    for col, label in metrics_for_outliers:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[col] < lower) | (df[col] > upper)
        outlier_df = df[mask & df[col].notna()].copy()
        outlier_df = outlier_df.sort_values(col, key=abs, ascending=False)

        outlier_entries = []
        for _, row in outlier_df.iterrows():
            outlier_entries.append({
                "ticker": row["ticker"],
                "year": int(row["year"]),
                "industry": row["industry"],
                "value": float(row[col]),
            })

        outliers_report[col] = {
            "label": label,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "n_outliers": len(outlier_entries),
            "outliers": outlier_entries,
        }

    outliers_path = OUTPUT_DIR / "outliers.json"
    with open(outliers_path, "w") as f:
        json.dump(outliers_report, f, indent=2)
    print(f"\nSaved outliers: {outliers_path}")
    for col, label in metrics_for_outliers:
        n = outliers_report[col]["n_outliers"]
        print(f"  {label}: {n} outliers")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
