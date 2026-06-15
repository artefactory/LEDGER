"""
Scatter plot: ΔVolatility(t+1 − t−1) vs ΔVolume(t+1 − t−1) at publication dates.

Each point = one (ticker, fiscal year) publication date.

Usage:
    uv run python KPI_analysis/scatter_vol_volume.py
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
from fetch_kpis import tickers_from_selected
from plot_indicators import annual_publication_dates

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output" / "plots"

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2023, 6, 30)

VOL_WINDOW = 20  # rolling window for volatility (default in pipeline is 20)


def _pub_date_for_fy(ticker: str, fy: int) -> pd.Timestamp | None:
    pubs = annual_publication_dates(ticker, originals_only=True)
    if pubs.empty:
        return None
    pubs = pubs.copy()
    pubs["report_date"] = pd.to_datetime(pubs["report_date"])
    pubs["fy"] = pubs["report_date"].apply(
        lambda d: d.year - 1 if d.month <= 3 else d.year
    )
    match = pubs[pubs["fy"] == fy]
    if match.empty:
        return None
    pub_dt = pd.to_datetime(match.iloc[0]["publication_date_et"])
    if pub_dt.tzinfo is not None:
        pub_dt = pub_dt.tz_localize(None)
    return pub_dt.normalize()


def main():
    entries = tickers_from_selected()
    print(f"Total tickers from selection: {len(entries)}")

    records = []
    prices_cache: dict[str, pd.DataFrame | None] = {}

    for i, entry in enumerate(entries):
        ticker = entry["ticker"]
        for year in range(2017, 2023):
            print(f"[{i+1}/{len(entries)}] {ticker} {year} ... ", end="", flush=True)

            pub_date = _pub_date_for_fy(ticker, year)
            if pub_date is None:
                print("no pub date")
                continue

            if ticker not in prices_cache:
                prices = fetch_prices(ticker, BENCH_START, BENCH_END)
                if prices is None or prices.empty:
                    print("no prices")
                    prices_cache[ticker] = None
                    continue
                prices = GetIndicatorsForPrices(prices)
                # Recompute volatility with shorter window
                prices["Volatility_short"] = prices["Close"].pct_change().rolling(window=VOL_WINDOW).std()
                prices_cache[ticker] = prices
            else:
                prices = prices_cache[ticker]

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
            if t0_pos < 1 or t0_pos >= len(prices) - 1:
                print("edge of data")
                continue

            t_minus_1 = prices.index[t0_pos - 1]
            t_plus_1 = prices.index[t0_pos + 1]

            vol_tp1 = prices.loc[t_plus_1, "Volatility_short"]
            vol_tm1 = prices.loc[t_minus_1, "Volatility_short"]
            volume_tp1 = prices.loc[t_plus_1, "Volume_ATS"]
            volume_tm1 = prices.loc[t_minus_1, "Volume_ATS"]

            if any(pd.isna(v) for v in [vol_tp1, vol_tm1, volume_tp1, volume_tm1]):
                print("NaN in indicators")
                continue

            records.append({
                "ticker": ticker,
                "year": year,
                "delta_volatility": vol_tp1 - vol_tm1,
                "delta_volume": volume_tp1 - volume_tm1,
            })
            print("ok")

    if not records:
        print("No records collected.")
        return

    df = pd.DataFrame(records)
    print(f"\nCollected {len(df)} points")

    # --- Scatter plot ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df["delta_volume"], df["delta_volatility"], alpha=0.5, s=20, edgecolors="none")

    # Fit line
    x = df["delta_volume"].values
    y = df["delta_volatility"].values
    m, b = np.polyfit(x, y, 1)
    x_range = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_range, m * x_range + b, color="red", linewidth=1.5,
            label=f"y = {m:.4f}x + {b:.4f}")

    # Correlation
    corr = df[["delta_volume", "delta_volatility"]].corr().iloc[0, 1]
    ax.legend(title=f"r = {corr:.3f}", fontsize=10)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("ΔVolume ATS = Volume(t+1) − Volume(t−1)", fontsize=11)
    ax.set_ylabel(f"ΔVolatility (rolling {VOL_WINDOW}d) = Vol(t+1) − Vol(t−1)", fontsize=11)
    ax.set_title(f"ΔVolatility ({VOL_WINDOW}d) vs ΔVolume around Publication Date\n(each point = one annual report)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Zoom on the dense region (clip outliers)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.04, 0.04)

    out_path = OUTPUT_DIR / "scatter_delta_vol_vs_delta_volume.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Correlation: {corr:.4f}")


if __name__ == "__main__":
    main()
