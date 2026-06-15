"""
Distribution plots of event-study metrics, split by sign of cumulative return at -90 days.
Two overlaid histograms per metric: return_t-90 > 0 vs return_t-90 < 0.
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
SENTIMENTS_JSON = (
    HERE.parent
    / "doc_text_processing"
    / "CEO_word_extraction"
    / "cleaning_extractions"
    / "cleaned"
    / "sentiments.json"
)
OUTPUT_DIR = HERE / "output" / "plots" / "distributions_by_return_m90"

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2023, 6, 30)
EVENT_HALF_WINDOW = 90

EVENT_DAYS = list(range(-90, -10, 10)) + list(range(-10, 11)) + list(range(20, 91, 10))


def _pub_date_for_fy(ticker: str, fy: int) -> pd.Timestamp | None:
    pubs = annual_publication_dates(ticker, originals_only=True)
    if pubs.empty:
        return None
    pubs = pubs.copy()
    pubs["report_date"] = pd.to_datetime(pubs["report_date"])
    pubs["fy"] = pubs["report_date"].apply(lambda d: d.year - 1 if d.month <= 3 else d.year)
    match = pubs[pubs["fy"] == fy]
    if match.empty:
        return None
    pub_dt = pd.to_datetime(match.iloc[0]["publication_date_et"])
    if pub_dt.tzinfo is not None:
        pub_dt = pub_dt.tz_localize(None)
    return pub_dt.normalize()


def _window_mean(series: pd.Series, target_date: pd.Timestamp, window: int = 5) -> float | None:
    series = series.dropna()
    if series.empty:
        return None
    mask = series.index >= target_date
    if mask.sum() == 0:
        return None
    start_idx = series.index[mask][0]
    start_pos = series.index.get_loc(start_idx)
    end_pos = min(start_pos + window, len(series) - 1)
    chunk = series.iloc[start_pos: end_pos + 1]
    return chunk.mean() if not chunk.empty else None


def main():
    with open(SENTIMENTS_JSON) as f:
        data = json.load(f)

    rows = []
    for industry, tickers in data.items():
        for ticker, years in tickers.items():
            for year, sentiment in years.items():
                if sentiment is not None:
                    rows.append({"industry": industry, "ticker": ticker, "year": int(year), "sentiment": sentiment})

    print(f"Loaded {len(rows)} entries")

    processed_tickers: dict[str, tuple] = {}
    records = []

    for i, row in enumerate(rows):
        ticker = row["ticker"]
        year = row["year"]

        print(f"[{i+1}/{len(rows)}] {ticker} {year} ... ", end="", flush=True)

        pub_date = _pub_date_for_fy(ticker, year)
        if pub_date is None:
            print("no pub date")
            continue

        if ticker not in processed_tickers:
            prices = fetch_prices(ticker, BENCH_START, BENCH_END)
            if prices is None or prices.empty:
                print("no prices")
                processed_tickers[ticker] = (None, None)
                continue
            prices = GetIndicatorsForPrices(prices, max_lag=EVENT_HALF_WINDOW)
            industry_df = GetIndustryDataFrame(ticker, BENCH_START, BENCH_END, max_lag=EVENT_HALF_WINDOW)
            processed_tickers[ticker] = (prices, industry_df)
        else:
            prices, industry_df = processed_tickers[ticker]

        if prices is None:
            print("no prices (cached)")
            continue

        pub_ts = pd.Timestamp(pub_date)
        if pub_ts not in prices.index:
            mask = prices.index >= pub_ts
            if mask.sum() == 0:
                print("pub date out of range")
                continue
            pub_ts = prices.index[mask][0]

        ind_aligned = industry_df.reindex(prices.index)

        # Get return_t-90 for the split
        ret_m90 = prices.loc[pub_ts, "return_t-90"] if "return_t-90" in prices.columns else None
        if ret_m90 is None or pd.isna(ret_m90):
            print("no return_t-90")
            continue

        # Industry return_t-90
        ind_ret_m90 = ind_aligned.loc[pub_ts, "return_t-90"] if (pub_ts in ind_aligned.index and "return_t-90" in ind_aligned.columns) else None
        ind_ret_m90_vw = ind_aligned.loc[pub_ts, "return_t-90_vw"] if (pub_ts in ind_aligned.index and "return_t-90_vw" in ind_aligned.columns) else None

        # Metrics at publication date
        unbiased_vol = prices["Volatility"] - ind_aligned["volatility"]
        unbiased_volume = prices["Volume_ATS"] - ind_aligned["volumes"]
        raw_vol = prices["Volatility"]
        raw_volume = prices["Volume_ATS"]

        rec = {
            "ticker": ticker,
            "year": year,
            "industry": row["industry"],
            "sentiment": row["sentiment"],
            "return_t-90": ret_m90,
            "return_t-90_unbiased": (ret_m90 - ind_ret_m90) if ind_ret_m90 is not None and not pd.isna(ind_ret_m90) else None,
            "return_t-90_unbiased_vw": (ret_m90 - ind_ret_m90_vw) if ind_ret_m90_vw is not None and not pd.isna(ind_ret_m90_vw) else None,
            # Return at various horizons
            "return_t1": prices.loc[pub_ts, "return_t1"] if "return_t1" in prices.columns else None,
            "return_t5": prices.loc[pub_ts, "return_t5"] if "return_t5" in prices.columns else None,
            "return_t10": prices.loc[pub_ts, "return_t10"] if "return_t10" in prices.columns else None,
            "return_t90": prices.loc[pub_ts, "return_t90"] if "return_t90" in prices.columns else None,
            "return_t-10": prices.loc[pub_ts, "return_t-10"] if "return_t-10" in prices.columns else None,
            # Volatility & volume around pub date
            "raw_volatility": _window_mean(raw_vol, pub_date),
            "raw_volume": _window_mean(raw_volume, pub_date),
            "unbiased_volatility": _window_mean(unbiased_vol, pub_date),
            "unbiased_volume": _window_mean(unbiased_volume, pub_date),
        }
        records.append(rec)
        print("ok")

    if not records:
        print("No records collected.")
        return

    df = pd.DataFrame(records)
    for c in df.columns:
        if c not in ("ticker", "year", "industry", "sentiment"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"\nCollected {len(df)} data points")
    print(f"  return_t-90 > 0: {(df['return_t-90'] > 0).sum()}")
    print(f"  return_t-90 < 0: {(df['return_t-90'] < 0).sum()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "records.csv", index=False)

    # Split by sign of pre-publication return (J-90)
    df_up = df[df["return_t-90"] > 0]
    df_down = df[df["return_t-90"] < 0]

    # Metrics to plot distributions for
    dist_metrics = [
        ("return_t-90", "Cumulative Return J-90"),
        ("return_t-90_unbiased", "Cum Return J-90 - Industry EW"),
        ("return_t-90_unbiased_vw", "Cum Return J-90 - Industry VW"),
        ("return_t1", "Cumulative Return J+1"),
        ("return_t5", "Cumulative Return J+5"),
        ("return_t10", "Cumulative Return J+10"),
        ("return_t90", "Cumulative Return J+90"),
        ("return_t-10", "Cumulative Return J-10"),
        ("raw_volatility", "Stock Volatility (5-day mean at pub)"),
        ("raw_volume", "Stock Volume ATS (5-day mean at pub)"),
        ("unbiased_volatility", "Unbiased Volatility (5-day mean at pub)"),
        ("unbiased_volume", "Unbiased Volume ATS (5-day mean at pub)"),
    ]

    for col, title in dist_metrics:
        vals_up = df_up[col].dropna()
        vals_down = df_down[col].dropna()
        if vals_up.empty and vals_down.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        # Common bins — ensure 0 is always a bin edge for return metrics
        all_vals = pd.concat([vals_up, vals_down])
        lo, hi = all_vals.quantile(0.02), all_vals.quantile(0.98)
        if col.startswith("return") and lo < 0 < hi:
            # Build bins so 0 is an edge: n_neg bins left of 0, n_pos bins right of 0
            n_bins = 24
            n_neg = max(1, int(n_bins * abs(lo) / (abs(lo) + hi)))
            n_pos = n_bins - n_neg
            bin_edges = np.concatenate([
                np.linspace(lo, 0, n_neg + 1)[:-1],  # left edges up to 0 (exclusive)
                np.linspace(0, hi, n_pos + 1),        # 0 to hi
            ])
        else:
            bin_edges = np.linspace(lo, hi, 25)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        if not vals_up.empty:
            ax.hist(vals_up, bins=bin_edges, color="#2ecc71", edgecolor="white", linewidth=0.5, alpha=0.7,
                    label=f"Return₋₉₀ > 0 (n={len(vals_up)}, μ={vals_up.mean():.4f})")
        if not vals_down.empty:
            ax.hist(vals_down, bins=bin_edges, color="#e74c3c", edgecolor="white", linewidth=0.5, alpha=0.7,
                    label=f"Return₋₉₀ < 0 (n={len(vals_down)}, μ={vals_down.mean():.4f})")

        # Vertical line at 0
        ax.axvline(0, color="black", linewidth=1.5, linestyle="-", alpha=0.8, label="0")

        # X-ticks: bin centers + always include 0 for return metrics
        ticks = list(bin_centers[::4])
        if col.startswith("return"):
            if all(abs(t) > (bin_edges[1] - bin_edges[0]) * 0.3 for t in ticks):
                ticks.append(0.0)
            ticks = sorted(ticks)
        ax.set_xticks(ticks)
        labels = [f"{v:.3f}" if v != 0 else "0" for v in ticks]
        ax.set_xticklabels(labels, fontsize=8, rotation=45)
        for lbl in ax.get_xticklabels():
            if lbl.get_text() == "0":
                lbl.set_fontweight("bold")
                lbl.set_fontsize(10)

        # Means
        if not vals_up.empty:
            ax.axvline(vals_up.mean(), color="#27ae60", linewidth=2, linestyle="--")
        if not vals_down.empty:
            ax.axvline(vals_down.mean(), color="#c0392b", linewidth=2, linestyle="--")

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {title}\nSplit by sign of cum. return at -90 days",
                      fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"dist_{col}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)

    print(f"\nAll distribution plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
