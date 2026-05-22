"""
Event study plots for a single stock across its annual report publication dates.
Creates one plot per metric with one line per fiscal year.
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
from plot_indicators import annual_publication_dates

sys.path.insert(0, str(Path(__file__).resolve().parent))

from FinancialIndicators import GetIndicatorsForPrices, GetIndustryDataFrame
from fetch_filing_returns import fetch_prices
from event_study_earnings import fetch_earnings_dates, find_q4_earnings_date

# ============================================================
# CONFIGURATION — change STOCKS here (list of tickers to process)
# ============================================================
STOCKS = ["BCPC", "GEVO", "AZO", "ORLY", "SLB", "CPB", "AGNC", "HWKN", "CLMT", "LOOP"]
# ============================================================

HERE = Path(__file__).resolve().parent
SENTIMENTS_JSON = (
    HERE.parent
    / "doc_text_processing"
    / "CEO_word_extraction"
    / "cleaning_extractions"
    / "cleaned"
    / "sentiments.json"
)

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2023, 6, 30)
EVENT_HALF_WINDOW = 90

# Days to sample: every 10 from -90 to -20, daily -10 to +10, every 10 from +20 to +90
EVENT_DAYS = list(range(-90, -10, 10)) + list(range(-10, 11)) + list(range(20, 91, 10))
_DAY_TO_POS = {d: i for i, d in enumerate(EVENT_DAYS)}

# Zoomed days: daily -10 to +10
EVENT_DAYS_ZOOM = list(range(-10, 11))
_DAY_TO_POS_ZOOM = {d: i for i, d in enumerate(EVENT_DAYS_ZOOM)}


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


def _event_window_values(series: pd.Series, target_date: pd.Timestamp, half_window: int, days: list[int]) -> dict[int, float]:
    series = series.dropna()
    if series.empty:
        return {}
    mask = series.index >= target_date
    if mask.sum() == 0:
        return {}
    t0_idx = series.index[mask][0]
    t0_pos = series.index.get_loc(t0_idx)
    result = {}
    for d in days:
        pos = t0_pos + d
        if 0 <= pos < len(series):
            result[d] = series.iloc[pos]
    return result


def _get_sentiment(ticker: str) -> dict[int, str]:
    """Return {year: sentiment} for the ticker from sentiments.json."""
    if not SENTIMENTS_JSON.exists():
        return {}
    with open(SENTIMENTS_JSON) as f:
        data = json.load(f)
    for industry, tickers in data.items():
        if ticker in tickers:
            return {int(y): s for y, s in tickers[ticker].items() if s is not None}
    return {}


def run_single_stock(ticker: str):
    print(f"\n{'='*60}\nEvent study for {ticker}\n{'='*60}")

    OUTPUT_DIR = HERE / "output" / "plots" / "sentiment_summary_augmented_lag" / ticker
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch prices and indicators
    prices = fetch_prices(ticker, BENCH_START, BENCH_END)
    if prices is None or prices.empty:
        print(f"No prices for {ticker}")
        return
    prices = GetIndicatorsForPrices(prices, max_lag=EVENT_HALF_WINDOW)
    industry_df = GetIndustryDataFrame(ticker, BENCH_START, BENCH_END, max_lag=EVENT_HALF_WINDOW)
    ind_aligned = industry_df.reindex(prices.index)

    # Derived series
    unbiased_vol = prices["Volatility"] - ind_aligned["volatility"]
    unbiased_volume = prices["Volume_ATS"] - ind_aligned["volumes"]
    raw_vol = prices["Volatility"]
    raw_volume = prices["Volume_ATS"]
    raw_price = prices["Close"]
    unbiased_volume_vw = (prices["Volume_ATS"] - ind_aligned["volumes_vw"]) if "volumes_vw" in ind_aligned.columns else pd.Series(dtype=float)

    # Get sentiment labels
    sentiments = _get_sentiment(ticker)

    # Find all publication dates (FY 2017-2022)
    years = list(range(2017, 2023))
    pub_dates = {}
    for fy in years:
        pub = _pub_date_for_fy(ticker, fy)
        if pub is not None:
            pub_dates[fy] = pub

    if not pub_dates:
        print(f"No publication dates found for {ticker}")
        return

    print(f"Found {len(pub_dates)} publication dates: {list(pub_dates.keys())}")

    # Collect event data per year
    event_data = {}  # year -> dict of metric -> {day: value}
    for fy, pub_date in pub_dates.items():
        pub_ts = pd.Timestamp(pub_date)
        sent = sentiments.get(fy, "unknown")

        evt = {
            "sentiment": sent,
            "unbiased_volatility": _event_window_values(unbiased_vol, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS),
            "unbiased_volume": _event_window_values(unbiased_volume, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS),
            "raw_volatility": _event_window_values(raw_vol, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS),
            "raw_volume": _event_window_values(raw_volume, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS),
            "raw_price": _event_window_values(raw_price, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS),
            "volume_unbiased_vw": _event_window_values(unbiased_volume_vw, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS),
        }

        # Cumulative returns
        cum_ret = {}
        cum_ret_ind = {}
        cum_ret_ind_vw = {}
        for d in EVENT_DAYS:
            col = f"return_t{d}"
            col_vw = f"return_t{d}_vw"
            if pub_ts in prices.index and col in prices.columns:
                cum_ret[d] = prices.loc[pub_ts, col]
            if pub_ts in ind_aligned.index and col in ind_aligned.columns:
                cum_ret_ind[d] = ind_aligned.loc[pub_ts, col]
            if pub_ts in ind_aligned.index and col_vw in ind_aligned.columns:
                cum_ret_ind_vw[d] = ind_aligned.loc[pub_ts, col_vw]

        evt["cum_return"] = cum_ret
        evt["cum_return_unbiased"] = {d: (cum_ret.get(d, np.nan) - cum_ret_ind.get(d, np.nan))
                                       for d in EVENT_DAYS
                                       if d in cum_ret and d in cum_ret_ind}
        evt["cum_return_unbiased_vw"] = {d: (cum_ret.get(d, np.nan) - cum_ret_ind_vw.get(d, np.nan))
                                          for d in EVENT_DAYS
                                          if d in cum_ret and d in cum_ret_ind_vw}

        # Normalize price to J0=1
        p0 = evt["raw_price"].get(0)
        if p0 and p0 != 0:
            evt["raw_price_norm"] = {d: v / p0 for d, v in evt["raw_price"].items()}
        else:
            evt["raw_price_norm"] = {}

        event_data[fy] = evt

    # Color palette: one distinct color per fiscal year
    year_cmap = plt.colormaps["tab10"]
    year_colors = {fy: year_cmap(i % 10) for i, fy in enumerate(sorted(event_data.keys()))}

    # Metrics to plot
    plot_metrics = [
        ("unbiased_volatility", "Unbiased Volatility"),
        ("unbiased_volume", "Unbiased Volume ATS"),
        ("raw_volatility", "Stock Volatility"),
        ("raw_volume", "Stock Volume ATS"),
        ("raw_price_norm", "Stock Price (normalized, J0=1)"),
        ("cum_return", "Cumulative Return from J0"),
        ("cum_return_unbiased", "Cumulative Return - Industry"),
        ("cum_return_unbiased_vw", "Cumulative Return - Industry VW"),
        ("volume_unbiased_vw", "Volume ATS - Industry Volume VW"),
    ]

    for col, title in plot_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Collect all year data for computing the average
        all_year_vals = {}  # day -> list of values
        for idx, (fy, evt) in enumerate(sorted(event_data.items())):
            data = evt.get(col, {})
            if not data:
                continue
            color = year_colors[fy]
            days = sorted(d for d in data.keys() if d in _DAY_TO_POS)
            positions = [_DAY_TO_POS[d] for d in days]
            vals = [data[d] for d in days]
            label = f"FY{fy}"
            ax.plot(positions, vals, color=color, linewidth=1, marker='o', markersize=2,
                    alpha=0.4, label=label)
            for d in days:
                all_year_vals.setdefault(d, []).append(data[d])

        # Plot average in bold black
        avg_days = sorted(d for d in all_year_vals if d in _DAY_TO_POS)
        if avg_days:
            avg_positions = [_DAY_TO_POS[d] for d in avg_days]
            avg_vals = [np.nanmean(all_year_vals[d]) for d in avg_days]
            ax.plot(avg_positions, avg_vals, color="black", linewidth=2.5, marker='o',
                    markersize=4, label="Average", zorder=10)

        ax.axvline(_DAY_TO_POS[0], color="black", linewidth=1, linestyle="--", alpha=0.7, label="Publication day")
        if col == "raw_price_norm":
            ax.axhline(1, color="grey", linewidth=0.5, linestyle=":")
        else:
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

        # X-axis labels
        tick_labels = [str(d) for d in EVENT_DAYS]
        ax.set_xticks(range(len(EVENT_DAYS)))
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
        ax.set_xlabel("Trading days relative to publication", fontsize=10)
        ax.set_title(f"{ticker} — Event Study: {title}\n(±{EVENT_HALF_WINDOW} trading days, one line per FY)",
                      fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(axis="both", alpha=0.3)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"event_{col}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)

    print(f"\nAll plots saved to {OUTPUT_DIR}")

    # --- Zoomed plots: -10 to +10 in sentiment_summary folder ---
    ZOOM_DIR = HERE / "output" / "plots" / "sentiment_summary" / ticker
    ZOOM_DIR.mkdir(parents=True, exist_ok=True)

    for col, title in plot_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        all_year_vals = {}
        for idx, (fy, evt) in enumerate(sorted(event_data.items())):
            data = evt.get(col, {})
            if not data:
                continue
            color = year_colors[fy]
            days = sorted(d for d in data.keys() if d in _DAY_TO_POS_ZOOM)
            positions = [_DAY_TO_POS_ZOOM[d] for d in days]
            vals = [data[d] for d in days]
            label = f"FY{fy}"
            ax.plot(positions, vals, color=color, linewidth=1, marker='o', markersize=2,
                    alpha=0.4, label=label)
            for d in days:
                all_year_vals.setdefault(d, []).append(data[d])

        # Average in bold
        avg_days = sorted(d for d in all_year_vals if d in _DAY_TO_POS_ZOOM)
        if avg_days:
            avg_positions = [_DAY_TO_POS_ZOOM[d] for d in avg_days]
            avg_vals = [np.nanmean(all_year_vals[d]) for d in avg_days]
            ax.plot(avg_positions, avg_vals, color="black", linewidth=2.5, marker='o',
                    markersize=4, label="Average", zorder=10)

        ax.axvline(_DAY_TO_POS_ZOOM[0], color="black", linewidth=1, linestyle="--", alpha=0.7, label="Publication day")
        if col == "raw_price_norm":
            ax.axhline(1, color="grey", linewidth=0.5, linestyle=":")
        else:
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

        tick_labels = [str(d) for d in EVENT_DAYS_ZOOM]
        ax.set_xticks(range(len(EVENT_DAYS_ZOOM)))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Trading days relative to publication", fontsize=10)
        ax.set_title(f"{ticker} — Event Study: {title}\n(±10 trading days, one line per FY)",
                      fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(axis="both", alpha=0.3)
        plt.tight_layout()
        out_path = ZOOM_DIR / f"event_{col}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)

    print(f"Zoomed plots saved to {ZOOM_DIR}")

    # --- Earnings-date anchored event study (same layout, T0 = earnings date) ---
    EARN_DIR = HERE / "output" / "plots" / "event_study_earnings" / "single_stock" / ticker
    EARN_DIR.mkdir(parents=True, exist_ok=True)

    earnings_df = fetch_earnings_dates(ticker)
    earn_dates = {}  # fy -> (earnings_date, surprise_pct)
    for fy in years:
        earn_date, surprise, _ = find_q4_earnings_date(ticker, fy, earnings_df)
        if earn_date is not None:
            earn_dates[fy] = (earn_date, surprise)

    if earn_dates:
        print(f"\nEarnings dates found for {ticker}: {list(earn_dates.keys())}")

        # Collect event data anchored on earnings date
        earn_event_data = {}
        for fy, (earn_date, surprise) in earn_dates.items():
            sent = sentiments.get(fy, "unknown")
            evt = {
                "sentiment": sent,
                "surprise_pct": surprise,
                "unbiased_volatility": _event_window_values(unbiased_vol, earn_date, EVENT_HALF_WINDOW, EVENT_DAYS),
                "unbiased_volume": _event_window_values(unbiased_volume, earn_date, EVENT_HALF_WINDOW, EVENT_DAYS),
                "raw_volatility": _event_window_values(raw_vol, earn_date, EVENT_HALF_WINDOW, EVENT_DAYS),
                "raw_volume": _event_window_values(raw_volume, earn_date, EVENT_HALF_WINDOW, EVENT_DAYS),
                "raw_price": _event_window_values(raw_price, earn_date, EVENT_HALF_WINDOW, EVENT_DAYS),
                "volume_unbiased_vw": _event_window_values(unbiased_volume_vw, earn_date, EVENT_HALF_WINDOW, EVENT_DAYS),
            }

            # Cumulative returns
            pub_ts = pd.Timestamp(earn_date)
            cum_ret = {}
            cum_ret_ind = {}
            cum_ret_ind_vw = {}
            for d in EVENT_DAYS:
                col_r = f"return_t{d}"
                col_vw = f"return_t{d}_vw"
                if pub_ts in prices.index and col_r in prices.columns:
                    cum_ret[d] = prices.loc[pub_ts, col_r]
                if pub_ts in ind_aligned.index and col_r in ind_aligned.columns:
                    cum_ret_ind[d] = ind_aligned.loc[pub_ts, col_r]
                if pub_ts in ind_aligned.index and col_vw in ind_aligned.columns:
                    cum_ret_ind_vw[d] = ind_aligned.loc[pub_ts, col_vw]

            evt["cum_return"] = cum_ret
            evt["cum_return_unbiased"] = {d: (cum_ret.get(d, np.nan) - cum_ret_ind.get(d, np.nan))
                                           for d in EVENT_DAYS
                                           if d in cum_ret and d in cum_ret_ind}
            evt["cum_return_unbiased_vw"] = {d: (cum_ret.get(d, np.nan) - cum_ret_ind_vw.get(d, np.nan))
                                              for d in EVENT_DAYS
                                              if d in cum_ret and d in cum_ret_ind_vw}

            p0 = evt["raw_price"].get(0)
            if p0 and p0 != 0:
                evt["raw_price_norm"] = {d: v / p0 for d, v in evt["raw_price"].items()}
            else:
                evt["raw_price_norm"] = {}

            earn_event_data[fy] = evt

        # Plot (full window)
        for col, title in plot_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            all_year_vals = {}
            for fy, evt in sorted(earn_event_data.items()):
                data = evt.get(col, {})
                if not data:
                    continue
                color = year_colors[fy]
                days_sorted = sorted(d for d in data.keys() if d in _DAY_TO_POS)
                positions = [_DAY_TO_POS[d] for d in days_sorted]
                vals = [data[d] for d in days_sorted]
                ax.plot(positions, vals, color=color, linewidth=1, marker='o', markersize=2,
                        alpha=0.4, label=f"FY{fy}")
                for d in days_sorted:
                    all_year_vals.setdefault(d, []).append(data[d])

            avg_days = sorted(d for d in all_year_vals if d in _DAY_TO_POS)
            if avg_days:
                avg_positions = [_DAY_TO_POS[d] for d in avg_days]
                avg_vals = [np.nanmean(all_year_vals[d]) for d in avg_days]
                ax.plot(avg_positions, avg_vals, color="black", linewidth=2.5, marker='o',
                        markersize=4, label="Average", zorder=10)

            ax.axvline(_DAY_TO_POS[0], color="black", linewidth=1, linestyle="--", alpha=0.7, label="Earnings date")
            if col == "raw_price_norm":
                ax.axhline(1, color="grey", linewidth=0.5, linestyle=":")
            else:
                ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

            tick_labels = [str(d) for d in EVENT_DAYS]
            ax.set_xticks(range(len(EVENT_DAYS)))
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
            ax.set_xlabel("Trading days relative to earnings date", fontsize=10)
            ax.set_title(f"{ticker} — Event Study: {title}\n(T0 = Earnings Date, ±{EVENT_HALF_WINDOW} trading days)",
                          fontsize=12, fontweight="bold")
            ax.legend(fontsize=9, loc="best")
            ax.grid(axis="both", alpha=0.3)
            plt.tight_layout()
            out_path = EARN_DIR / f"event_{col}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Zoomed plots (-10 to +10)
        for col, title in plot_metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            all_year_vals = {}
            for fy, evt in sorted(earn_event_data.items()):
                data = evt.get(col, {})
                if not data:
                    continue
                color = year_colors[fy]
                days_sorted = sorted(d for d in data.keys() if d in _DAY_TO_POS_ZOOM)
                positions = [_DAY_TO_POS_ZOOM[d] for d in days_sorted]
                vals = [data[d] for d in days_sorted]
                ax.plot(positions, vals, color=color, linewidth=1, marker='o', markersize=2,
                        alpha=0.4, label=f"FY{fy}")
                for d in days_sorted:
                    all_year_vals.setdefault(d, []).append(data[d])

            avg_days = sorted(d for d in all_year_vals if d in _DAY_TO_POS_ZOOM)
            if avg_days:
                avg_positions = [_DAY_TO_POS_ZOOM[d] for d in avg_days]
                avg_vals = [np.nanmean(all_year_vals[d]) for d in avg_days]
                ax.plot(avg_positions, avg_vals, color="black", linewidth=2.5, marker='o',
                        markersize=4, label="Average", zorder=10)

            ax.axvline(_DAY_TO_POS_ZOOM[0], color="black", linewidth=1, linestyle="--", alpha=0.7, label="Earnings date")
            if col == "raw_price_norm":
                ax.axhline(1, color="grey", linewidth=0.5, linestyle=":")
            else:
                ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

            tick_labels = [str(d) for d in EVENT_DAYS_ZOOM]
            ax.set_xticks(range(len(EVENT_DAYS_ZOOM)))
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_xlabel("Trading days relative to earnings date", fontsize=10)
            ax.set_title(f"{ticker} — Event Study: {title}\n(T0 = Earnings Date, ±10 trading days)",
                          fontsize=12, fontweight="bold")
            ax.legend(fontsize=9, loc="best")
            ax.grid(axis="both", alpha=0.3)
            plt.tight_layout()
            out_path = EARN_DIR / f"event_{col}_zoom.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"Earnings-date event study plots saved to {EARN_DIR}")
    else:
        print(f"No earnings dates found for {ticker}, skipping earnings event study")


def main():
    for ticker in STOCKS:
        run_single_stock(ticker)


if __name__ == "__main__":
    main()
