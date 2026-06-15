"""
Distribution plots of the immediate publication effect:
  Δvolatility = Volatility(t+1) - Volatility(t-1)
  Δvolume     = Volume_ATS(t+1) - Volume_ATS(t-1)
  Δreturn     = return_t{+1} - return_t{-1}

One histogram per metric across all (ticker, year) reports.
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
OUTPUT_DIR = HERE / "output" / "plots" / "distributions_delta_t1_vs_tm1"

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2023, 6, 30)
EVENT_HALF_WINDOW = 10


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

    processed_tickers: dict[str, pd.DataFrame | None] = {}
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
                processed_tickers[ticker] = None
                continue
            prices = GetIndicatorsForPrices(prices, max_lag=EVENT_HALF_WINDOW)
            processed_tickers[ticker] = prices
        else:
            prices = processed_tickers[ticker]

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

        t0_pos = prices.index.get_loc(pub_ts)

        # Need at least 1 trading day before and after
        if t0_pos < 1 or t0_pos >= len(prices) - 1:
            print("edge of data")
            continue

        t_minus_1 = prices.index[t0_pos - 1]
        t_plus_1 = prices.index[t0_pos + 1]

        # Extract values
        vol_tp1 = prices.loc[t_plus_1, "Volatility"] if "Volatility" in prices.columns else None
        vol_tm1 = prices.loc[t_minus_1, "Volatility"] if "Volatility" in prices.columns else None

        volume_tp1 = prices.loc[t_plus_1, "Volume_ATS"] if "Volume_ATS" in prices.columns else None
        volume_tm1 = prices.loc[t_minus_1, "Volume_ATS"] if "Volume_ATS" in prices.columns else None

        ret_tp1 = prices.loc[pub_ts, "return_t1"] if "return_t1" in prices.columns else None
        ret_tm1 = prices.loc[pub_ts, "return_t-1"] if "return_t-1" in prices.columns else None

        rec = {
            "ticker": ticker,
            "year": year,
            "industry": row["industry"],
            "sentiment": row["sentiment"],
            "delta_volatility": (vol_tp1 - vol_tm1) if vol_tp1 is not None and vol_tm1 is not None else None,
            "delta_volume": (volume_tp1 - volume_tm1) if volume_tp1 is not None and volume_tm1 is not None else None,
            "delta_return": (ret_tp1 - ret_tm1) if ret_tp1 is not None and ret_tm1 is not None else None,
        }
        records.append(rec)
        print("ok")

    if not records:
        print("No records collected.")
        return

    df = pd.DataFrame(records)
    for c in ["delta_volatility", "delta_volume", "delta_return"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"\nCollected {len(df)} data points")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "records.csv", index=False)

    # --- Plot distributions ---
    metrics = [
        ("delta_volatility", "ΔVolatility = Vol(t+1) − Vol(t−1)"),
        ("delta_volume", "ΔVolume ATS = Volume(t+1) − Volume(t−1)"),
        ("delta_return", "ΔReturn = return_t{+1} − return_t{−1}"),
    ]

    for col, title in metrics:
        vals = df[col].dropna()
        if vals.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        lo, hi = vals.quantile(0.02), vals.quantile(0.98)
        # Ensure 0 is a bin edge
        if lo < 0 < hi:
            n_bins = 30
            n_neg = max(1, int(n_bins * abs(lo) / (abs(lo) + hi)))
            n_pos = n_bins - n_neg
            bin_edges = np.concatenate([
                np.linspace(lo, 0, n_neg + 1)[:-1],
                np.linspace(0, hi, n_pos + 1),
            ])
        else:
            bin_edges = np.linspace(lo, hi, 31)

        ax.hist(vals, bins=bin_edges, color="#3498db", edgecolor="white",
                linewidth=0.5, alpha=0.8)

        # Vertical lines
        ax.axvline(0, color="black", linewidth=1.5, linestyle="-", alpha=0.8)
        ax.axvline(vals.mean(), color="#2c3e50", linewidth=2, linestyle="--",
                   label=f"μ = {vals.mean():.4f}")
        ax.axvline(vals.median(), color="#8e44ad", linewidth=2, linestyle=":",
                   label=f"median = {vals.median():.4f}")

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {title}\n(all reports, n={len(vals)})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Bold "0" tick
        ticks = list(ax.get_xticks())
        if 0.0 not in ticks:
            ticks.append(0.0)
        ticks = sorted(ticks)
        ax.set_xticks(ticks)
        labels = [f"{v:.3f}" if v != 0 else "0" for v in ticks]
        ax.set_xticklabels(labels, fontsize=8, rotation=45)
        for lbl in ax.get_xticklabels():
            if lbl.get_text() == "0":
                lbl.set_fontweight("bold")
                lbl.set_fontsize(10)

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"dist_{col}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    # Summary stats
    print("\n--- Summary ---")
    for col, title in metrics:
        vals = df[col].dropna()
        if vals.empty:
            continue
        pct_pos = (vals > 0).sum() / len(vals) * 100
        print(f"{title}:")
        print(f"  n={len(vals)}, mean={vals.mean():.4f}, median={vals.median():.4f}, std={vals.std():.4f}")
        print(f"  % positive: {pct_pos:.1f}%")


if __name__ == "__main__":
    main()
