"""
Event study anchored on EARNINGS CALL dates (not 10-K filing dates).

For each (ticker, fiscal year), finds the Q4 earnings announcement date via
yfinance.get_earnings_dates(), then computes event-window metrics around that
date. This captures the *actual* market reaction to annual results — the 10-K
filing typically comes 3-6 weeks later and contains no new information.

Approach to map earnings → fiscal year:
  For each ticker, we know the 10-K filing date (from EDGAR). The Q4 earnings
  call is the yfinance earnings date that falls in the window
  [filing_date - 60 days, filing_date - 5 days]. This avoids hardcoding
  fiscal-year-end calendars per company.

Outputs:
  output/plots/event_study_earnings/
    event_at_earnings_date/              — event study plots with 95% CI
    event_earnings_date_same_as_filling/ — same-day filers (delta ≤ 1d)
    distribution/                        — distributions (return, volume, volatility) by surprise sign
  output/earnings_events.csv — one row per (ticker, year) with earnings date,
                               surprise, and event metrics

Usage:
    uv run python KPI_analysis/event_study_earnings.py
"""

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

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
SELECTED_COMPANIES_JSON = (
    HERE.parent / "tickers_lists" / "grouped" / "selected" / "companies.json"
)
OUTPUT_BASE = HERE / "output" / "plots" / "event_study_earnings"
DIR_EVENT = OUTPUT_BASE / "event_at_earnings_date"
DIR_SAMEDAY = OUTPUT_BASE / "event_earnings_date_same_as_filling"
DIR_DISTRIB = OUTPUT_BASE / "distribution"
DIR_COMPARE = OUTPUT_BASE / "earnings_vs_filing_event"
DIR_FILING_SURPRISE = OUTPUT_BASE / "event_at_filing_date_by_surprise"
CACHE_DIR = HERE / "cache" / "earnings_dates"

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2023, 6, 30)
EVENT_HALF_WINDOW = 10

EVENT_DAYS = list(range(-10, 11))
_DAY_TO_POS = {d: i for i, d in enumerate(EVENT_DAYS)}


def _cache_path(ticker: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker}.csv"


def fetch_earnings_dates(ticker: str, *, refresh: bool = False) -> pd.DataFrame:
    """Fetch earnings dates from yfinance, cached on disk."""
    path = _cache_path(ticker)
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["earnings_date"])
        return df

    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=60)
        time.sleep(0.5)
    except Exception as e:
        print(f"  yfinance earnings error for {ticker}: {e}")
        return pd.DataFrame()

    if ed is None or ed.empty:
        return pd.DataFrame()

    # Flatten: index is Earnings Date (tz-aware), columns are EPS Estimate, Reported EPS, Surprise(%)
    records = []
    for dt_idx, row in ed.iterrows():
        dt_naive = dt_idx.tz_localize(None) if dt_idx.tzinfo else dt_idx
        records.append({
            "earnings_date": dt_naive,
            "eps_estimate": row.get("EPS Estimate"),
            "reported_eps": row.get("Reported EPS"),
            "surprise_pct": row.get("Surprise(%)"),
        })

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return df


def find_q4_earnings_date(
    ticker: str, fy: int, earnings_df: pd.DataFrame
) -> tuple[pd.Timestamp | None, float | None, pd.Timestamp | None]:
    """Find the Q4 earnings date for a given fiscal year.

    Strategy: find the earnings date closest to (but before) the 10-K filing date,
    within a 60-day lookback window.

    Returns (earnings_date, surprise_pct, filing_date).
    """
    pubs = annual_publication_dates(ticker, originals_only=True)
    if pubs.empty:
        return None, None, None

    pubs = pubs.copy()
    pubs["report_date"] = pd.to_datetime(pubs["report_date"])
    pubs["fy"] = pubs["report_date"].apply(lambda d: d.year - 1 if d.month <= 3 else d.year)
    match = pubs[pubs["fy"] == fy]
    if match.empty:
        return None, None, None

    filing_date = pd.to_datetime(match.iloc[0]["filing_date"])

    # Find earnings date in [filing_date - 60, filing_date + 1]
    # Many companies file the 10-K on the same day as the earnings call (delta=0-1d),
    # while others have a 3-6 week gap. The +1d captures same-day filers.
    window_start = filing_date - pd.Timedelta(days=60)
    window_end = filing_date + pd.Timedelta(days=1)

    candidates = earnings_df[
        (earnings_df["earnings_date"] >= window_start)
        & (earnings_df["earnings_date"] <= window_end)
    ]

    if candidates.empty:
        return None, None, filing_date

    # Take the one closest to the filing date (= most recent before filing)
    candidates = candidates.sort_values("earnings_date", ascending=False)
    best = candidates.iloc[0]
    return pd.Timestamp(best["earnings_date"]).normalize(), best.get("surprise_pct"), filing_date


def _event_window_values(
    series: pd.Series, target_date: pd.Timestamp, days: list[int]
) -> dict[int, float]:
    """Extract values at specific trading-day offsets from target_date."""
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


def main():
    # Load all selected tickers from companies.json (full selection, all industries)
    with open(SELECTED_COMPANIES_JSON) as f:
        companies_data = json.load(f)

    # Load sentiments for enrichment (optional)
    sentiments_map: dict[tuple[str, int], str] = {}
    if SENTIMENTS_JSON.exists():
        with open(SENTIMENTS_JSON) as f:
            sent_data = json.load(f)
        for industry, tickers in sent_data.items():
            for ticker, years in tickers.items():
                for year, sentiment in years.items():
                    if sentiment is not None:
                        sentiments_map[(ticker, int(year))] = sentiment

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
                        "sentiment": sentiments_map.get((ticker, year)),
                    })

    # Deduplicate by (ticker, year) — a ticker can appear under multiple exchanges
    seen_ty = set()
    deduped = []
    for e in all_entries:
        key = (e["ticker"], e["year"])
        if key not in seen_ty:
            seen_ty.add(key)
            deduped.append(e)
    all_entries = deduped

    print(f"Loaded {len(all_entries)} unique (ticker, year) entries from companies.json")

    processed_tickers: dict[str, tuple] = {}
    earnings_cache: dict[str, pd.DataFrame] = {}
    records = []
    event_rows = []  # day-by-day event study data
    matched_reports = []  # reports with earnings date found
    unmatched_reports = []  # reports without earnings date

    for i, entry in enumerate(all_entries):
        ticker = entry["ticker"]
        year = entry["year"]
        sentiment = entry["sentiment"]

        print(f"[{i+1}/{len(all_entries)}] {ticker} {year} ... ", end="", flush=True)

        # Get earnings dates
        if ticker not in earnings_cache:
            earnings_cache[ticker] = fetch_earnings_dates(ticker)
        earnings_df = earnings_cache[ticker]
        if earnings_df.empty:
            print("no earnings dates")
            unmatched_reports.append({
                "ticker": ticker,
                "year": year,
                "industry": entry["industry"],
                "reason": "no earnings dates from yfinance",
            })
            continue

        # Find Q4 earnings for this FY
        earn_date, surprise, filing_date = find_q4_earnings_date(ticker, year, earnings_df)
        if earn_date is None:
            print("no Q4 earnings match")
            unmatched_reports.append({
                "ticker": ticker,
                "year": year,
                "industry": entry["industry"],
                "reason": "no Q4 earnings match in [filing-60d, filing-1d] window",
                "filing_date": str(filing_date.date()) if filing_date is not None else None,
            })
            continue

        # Track matched report
        delta_days_report = (filing_date - earn_date).days if filing_date is not None else None
        matched_reports.append({
            "ticker": ticker,
            "year": year,
            "industry": entry["industry"],
            "sentiment": sentiment,
            "earnings_date": str(earn_date.date()),
            "filing_date": str(filing_date.date()) if filing_date is not None else None,
            "delta_filing_earnings": delta_days_report,
            "surprise_pct": float(surprise) if surprise is not None and not pd.isna(surprise) else None,
        })

        # Get prices + indicators
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

        ind_aligned = industry_df.reindex(prices.index)

        # Compute metrics at earnings date
        pub_ts = pd.Timestamp(earn_date)
        if pub_ts not in prices.index:
            mask = prices.index >= pub_ts
            if mask.sum() == 0:
                print("earnings date out of range")
                continue
            pub_ts = prices.index[mask][0]

        t0_pos = prices.index.get_loc(pub_ts)
        if t0_pos < EVENT_HALF_WINDOW or t0_pos >= len(prices) - EVENT_HALF_WINDOW:
            print("edge of data")
            continue

        # Volume ATS at t+1
        volume_t1 = prices.iloc[t0_pos + 1]["Volume_ATS"] if t0_pos + 1 < len(prices) else None
        # Return t+1 (from the return_t1 column at pub_ts)
        return_t1 = prices.loc[pub_ts, "return_t1"] if "return_t1" in prices.columns else None
        return_t5 = prices.loc[pub_ts, "return_t5"] if "return_t5" in prices.columns else None

        # Industry-adjusted return
        ind_ret_t1 = ind_aligned.loc[pub_ts, "return_t1"] if (pub_ts in ind_aligned.index and "return_t1" in ind_aligned.columns) else None
        unbiased_ret_t1 = (return_t1 - ind_ret_t1) if return_t1 is not None and ind_ret_t1 is not None and not pd.isna(ind_ret_t1) else None

        # Compute delta between filing and earnings
        delta_days = (filing_date - earn_date).days if filing_date is not None else None

        rec = {
            "ticker": ticker,
            "year": year,
            "industry": entry["industry"],
            "sentiment": sentiment,
            "earnings_date": earn_date,
            "filing_date": filing_date,
            "delta_filing_earnings": delta_days,
            "surprise_pct": surprise,
            "volume_t1": volume_t1,
            "return_t1": return_t1,
            "return_t5": return_t5,
            "unbiased_return_t1": unbiased_ret_t1,
        }
        records.append(rec)

        # --- Collect day-by-day event study values ---
        for d in EVENT_DAYS:
            pos = t0_pos + d
            if 0 <= pos < len(prices):
                cum_ret_col = f"return_t{d}"
                cum_ret = prices.loc[pub_ts, cum_ret_col] if cum_ret_col in prices.columns else None
                ind_cum_ret = ind_aligned.loc[pub_ts, cum_ret_col] if (pub_ts in ind_aligned.index and cum_ret_col in ind_aligned.columns) else None

                event_rows.append({
                    "ticker": ticker,
                    "year": year,
                    "industry": entry["industry"],
                    "sentiment": sentiment,
                    "surprise_pct": surprise,
                    "delta_filing_earnings": delta_days,
                    "relative_day": d,
                    "raw_volatility": prices.iloc[pos]["Volatility"],
                    "raw_volume": prices.iloc[pos]["Volume_ATS"],
                    "cum_return": cum_ret,
                    "cum_return_unbiased": (cum_ret - ind_cum_ret) if (cum_ret is not None and ind_cum_ret is not None and not pd.isna(ind_cum_ret)) else None,
                })

        print(f"ok ({earn_date.date()}, surprise={surprise:+.1f}%)" if surprise and not pd.isna(surprise) else f"ok ({earn_date.date()})")

    if not records:
        print("No records collected.")
        return

    df = pd.DataFrame(records)
    for c in ["volume_t1", "return_t1", "return_t5", "unbiased_return_t1", "surprise_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for d in [DIR_EVENT, DIR_SAMEDAY, DIR_DISTRIB, DIR_COMPARE, DIR_FILING_SURPRISE, OUTPUT_BASE]:
        d.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_BASE / "earnings_events.csv", index=False)

    # Save matched/unmatched JSON
    earnings_mapping = {
        "matched": matched_reports,
        "unmatched": unmatched_reports,
    }
    mapping_path = OUTPUT_BASE / "earnings_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(earnings_mapping, f, indent=2)
    print(f"\nSaved earnings mapping: {mapping_path}")
    print(f"  Matched: {len(matched_reports)}, Unmatched: {len(unmatched_reports)}")

    print(f"\nCollected {len(df)} events")
    print(f"  With surprise data: {df['surprise_pct'].notna().sum()}")
    print(f"  Mean |return_t1| at earnings: {df['return_t1'].abs().mean():.4f}")
    print(f"  Mean volume_t1 at earnings: {df['volume_t1'].mean():.2f}")

    # --- Event study plots: Vol, Volume, Return over ±10 days ---
    df_event = pd.DataFrame(event_rows)
    for c in ["raw_volatility", "raw_volume", "cum_return", "cum_return_unbiased", "surprise_pct", "delta_filing_earnings"]:
        df_event[c] = pd.to_numeric(df_event[c], errors="coerce")

    event_metrics = [
        ("raw_volatility", "Stock Volatility"),
        ("raw_volume", "Stock Volume ATS"),
        ("cum_return", "Cumulative Return from Earnings Date"),
        ("cum_return_unbiased", "Cum Return − Industry Cum Return"),
    ]

    x_positions = list(range(len(EVENT_DAYS)))

    # --- Helper: plot event study with 95% CI ---
    def _plot_event_with_ci(ax, sub_df, col, color, label_prefix):
        """Plot mean line + 95% CI shading for a metric grouped by relative_day."""
        grouped = sub_df.groupby("relative_day")[col]
        means = grouped.mean()
        sems = grouped.sem()  # standard error of the mean
        y_mean = [means.get(d, np.nan) for d in EVENT_DAYS]
        y_sem = [sems.get(d, np.nan) for d in EVENT_DAYS]
        y_mean = np.array(y_mean, dtype=float)
        y_sem = np.array(y_sem, dtype=float)
        ci_lo = y_mean - 1.96 * y_sem
        ci_hi = y_mean + 1.96 * y_sem
        n = sub_df.groupby(["ticker", "year"]).ngroups
        ax.plot(x_positions, y_mean, color=color, linewidth=2,
                label=f"{label_prefix} (n={n})", marker="o", markersize=3)
        ax.fill_between(x_positions, ci_lo, ci_hi, color=color, alpha=0.15)

    # Plot 1: by surprise sign (positive surprise vs negative surprise)
    df_event_pos_surprise = df_event[df_event["surprise_pct"] > 0]
    df_event_neg_surprise = df_event[df_event["surprise_pct"] < 0]

    for col, title in event_metrics:
        fig, ax = plt.subplots(figsize=(12, 5))

        if not df_event_pos_surprise.empty:
            _plot_event_with_ci(ax, df_event_pos_surprise, col, "#2ecc71", "Surprise > 0")
        if not df_event_neg_surprise.empty:
            _plot_event_with_ci(ax, df_event_neg_surprise, col, "#e74c3c", "Surprise < 0")

        # Count observations per surprise sign per day
        n_per_day_surp = {}
        for label, sub in [("pos", df_event_pos_surprise), ("neg", df_event_neg_surprise)]:
            n_per_day_surp[label] = sub.groupby("relative_day")[col].count() if not sub.empty else pd.Series(dtype=int)
        tick_labels_surp = []
        for d in EVENT_DAYS:
            parts = [str(d)]
            parts.append(str(int(n_per_day_surp["pos"].get(d, 0))))
            parts.append(str(int(n_per_day_surp["neg"].get(d, 0))))
            tick_labels_surp.append("\n".join(parts))

        ax.axvline(_DAY_TO_POS[0], color="gray", linestyle="--", alpha=0.7, label="Earnings date")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels_surp, fontsize=7)
        ax.set_xlabel("Trading days relative to earnings call\n(n: surprise>0 / surprise<0)")
        ax.set_ylabel(title)
        ax.set_title(f"{title}\n(±{EVENT_HALF_WINDOW} days around earnings call, by EPS surprise sign, 95% CI)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(DIR_EVENT / f"event_{col}_by_surprise.png", dpi=150)
        plt.close(fig)

    print(f"\nSaved event study plots (by surprise) to {DIR_EVENT}/")

    # Plot 1b: by sentiment (positive vs negative CEO letter)
    df_event_sent_pos = df_event[df_event["sentiment"] == "positive"]
    df_event_sent_neg = df_event[df_event["sentiment"] == "negative"]

    if not df_event_sent_pos.empty and not df_event_sent_neg.empty:
        for col, title in event_metrics:
            fig, ax = plt.subplots(figsize=(12, 5))

            _plot_event_with_ci(ax, df_event_sent_pos, col, "#2ecc71", "Sentiment: positive")
            _plot_event_with_ci(ax, df_event_sent_neg, col, "#e74c3c", "Sentiment: negative")

            ax.axvline(_DAY_TO_POS[0], color="gray", linestyle="--", alpha=0.7, label="Earnings date")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(d) for d in EVENT_DAYS], fontsize=8)
            ax.set_xlabel("Trading days relative to earnings call")
            ax.set_ylabel(title)
            ax.set_title(f"{title}\n(±{EVENT_HALF_WINDOW} days around earnings call, by CEO letter sentiment, 95% CI)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(DIR_EVENT / f"event_{col}_by_sentiment.png", dpi=150)
            plt.close(fig)

        print(f"Saved event study plots (by sentiment) to {DIR_EVENT}/")

    # --- Plot 2: same-day filers only (delta ≤ 1 days) ---
    df_event_sameday = df_event[df_event["delta_filing_earnings"].notna() & (df_event["delta_filing_earnings"] <= 1)]
    n_sameday_events = df_event_sameday.groupby(["ticker", "year"]).ngroups
    print(f"\n--- Same-day filers (delta ≤ 1d): {n_sameday_events} events ---")

    if n_sameday_events > 5:
        df_sd_pos = df_event_sameday[df_event_sameday["surprise_pct"] > 0]
        df_sd_neg = df_event_sameday[df_event_sameday["surprise_pct"] < 0]

        for col, title in event_metrics:
            fig, ax = plt.subplots(figsize=(12, 5))

            if not df_sd_pos.empty:
                _plot_event_with_ci(ax, df_sd_pos, col, "#2ecc71", "Surprise > 0")
            if not df_sd_neg.empty:
                _plot_event_with_ci(ax, df_sd_neg, col, "#e74c3c", "Surprise < 0")

            n_per_day_surp = {}
            for label, sub in [("pos", df_sd_pos), ("neg", df_sd_neg)]:
                n_per_day_surp[label] = sub.groupby("relative_day")[col].count() if not sub.empty else pd.Series(dtype=int)
            tick_labels_surp = []
            for d in EVENT_DAYS:
                parts = [str(d)]
                parts.append(str(int(n_per_day_surp["pos"].get(d, 0))))
                parts.append(str(int(n_per_day_surp["neg"].get(d, 0))))
                tick_labels_surp.append("\n".join(parts))

            ax.axvline(_DAY_TO_POS[0], color="gray", linestyle="--", alpha=0.7, label="Earnings/Filing date")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(tick_labels_surp, fontsize=7)
            ax.set_xlabel("Trading days relative to earnings call\n(n: surprise>0 / surprise<0)")
            ax.set_ylabel(title)
            ax.set_title(f"{title}\n(Same-day filers: earnings = filing ±1d, n={n_sameday_events}, by surprise, 95% CI)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(DIR_SAMEDAY / f"event_{col}_by_surprise.png", dpi=150)
            plt.close(fig)

        print(f"Saved same-day filer plots to {DIR_SAMEDAY}/")

    # --- Distribution plots: return, volume, volatility by surprise sign ---
    # Use day t+1 values (relative_day == 1) for distributions
    df_t1 = df_event[df_event["relative_day"] == 1].copy()
    df_t1_pos = df_t1[df_t1["surprise_pct"] > 0]
    df_t1_neg = df_t1[df_t1["surprise_pct"] < 0]

    # Histogram of delta between filing date and earnings date
    df_delta = df[df["delta_filing_earnings"].notna()].copy()
    if not df_delta.empty:
        df_delta["is_us"] = ~df_delta["ticker"].str.contains(r"\.", regex=True)
        deltas_us = df_delta.loc[df_delta["is_us"], "delta_filing_earnings"]
        deltas_nonus = df_delta.loc[~df_delta["is_us"], "delta_filing_earnings"]
        deltas = df_delta["delta_filing_earnings"]

        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.arange(deltas.min() - 0.5, deltas.max() + 1.5, 1)
        if not deltas_us.empty:
            ax.hist(deltas_us, bins=bins, color="#3498db", edgecolor="white", alpha=0.8,
                    label=f"US (n={len(deltas_us)})")
        if not deltas_nonus.empty:
            ax.hist(deltas_nonus, bins=bins, color="#e67e22", edgecolor="white", alpha=0.8,
                    label=f"Non-US (n={len(deltas_nonus)})")
        ax.axvline(deltas.median(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Median = {deltas.median():.0f}d")
        ax.axvline(deltas.mean(), color="orange", linestyle="--", linewidth=1.5,
                   label=f"Mean = {deltas.mean():.1f}d")
        ax.set_xlabel("Delta (Filing Date − Earnings Date) [days]")
        ax.set_ylabel("Count")
        ax.set_xticks(np.arange(int(deltas.min()), int(deltas.max()) + 1, 1))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"Distribution of Filing−Earnings Date Gap\n(n={len(deltas)}, same-day ≤1d: {(deltas <= 1).sum()})")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(DIR_DISTRIB / "dist_delta_filing_earnings.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {DIR_DISTRIB / 'dist_delta_filing_earnings.png'}")

    # Histogram of EPS surprise distribution
    surprise_vals = df["surprise_pct"].dropna()
    if not surprise_vals.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        q01, q99 = surprise_vals.quantile(0.02), surprise_vals.quantile(0.98)
        inliers = surprise_vals[(surprise_vals >= q01) & (surprise_vals <= q99)]
        bins = np.linspace(q01, q99, 40)
        pos_vals = inliers[inliers >= 0]
        neg_vals = inliers[inliers < 0]
        ax.hist(neg_vals, bins=bins, alpha=0.7, color="#e74c3c", label=f"Surprise < 0 (n={len(neg_vals)})")
        ax.hist(pos_vals, bins=bins, alpha=0.7, color="#2ecc71", label=f"Surprise ≥ 0 (n={len(pos_vals)})")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(surprise_vals.median(), color="red", linestyle=":", linewidth=1.5,
                   label=f"Median = {surprise_vals.median():.1f}%")
        n_clipped = len(surprise_vals) - len(inliers)
        ax.set_xlabel("EPS Surprise (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of EPS Surprise\n(n={len(inliers)}, {n_clipped} outliers clipped outside [{q01:.0f}%, {q99:.0f}%])")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(DIR_DISTRIB / "dist_surprise_pct.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {DIR_DISTRIB / 'dist_surprise_pct.png'}")

    distrib_metrics = [
        ("cum_return", "Return t+1", None),
        ("raw_volume", "Volume ATS t+1", (0, 6)),
        ("raw_volatility", "Volatility t+1", None),
    ]

    for col, label, xlim in distrib_metrics:
        fig, ax = plt.subplots(figsize=(10, 5))

        vals_pos = df_t1_pos[col].dropna()
        vals_neg = df_t1_neg[col].dropna()

        if xlim:
            bins = np.linspace(xlim[0], xlim[1], 40)
        else:
            all_vals = pd.concat([vals_pos, vals_neg])
            if all_vals.empty:
                plt.close(fig)
                continue
            q01, q99 = all_vals.quantile(0.01), all_vals.quantile(0.99)
            bins = np.linspace(q01, q99, 40)

        if not vals_pos.empty:
            ax.hist(vals_pos, bins=bins, alpha=0.6, color="#2ecc71",
                    label=f"Surprise > 0 (n={len(vals_pos)}, μ={vals_pos.mean():.4f})")
        if not vals_neg.empty:
            ax.hist(vals_neg, bins=bins, alpha=0.6, color="#e74c3c",
                    label=f"Surprise < 0 (n={len(vals_neg)}, μ={vals_neg.mean():.4f})")

        ax.axvline(0, color="black", linewidth=0.8, alpha=0.5, linestyle="--")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label} at Earnings Date\n(by EPS surprise sign)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(DIR_DISTRIB / f"dist_{col}.png", dpi=150)
        plt.close(fig)

    print(f"Saved distribution plots to {DIR_DISTRIB}/")

    # --- Compare: volume at earnings vs volume at 10-K filing ---
    # Build surprise lookup from main records
    surprise_lookup = {}
    for _, row in df.iterrows():
        surprise_lookup[(row["ticker"], row["year"])] = row["surprise_pct"]

    filing_records = []
    filing_event_rows = []  # day-by-day event study anchored on filing date
    for i, entry in enumerate(all_entries):
        ticker = entry["ticker"]
        year = entry["year"]

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

        if ticker not in processed_tickers or processed_tickers[ticker][0] is None:
            continue
        prices, industry_df = processed_tickers[ticker]

        pub_ts = pd.Timestamp(filing_dt)
        if pub_ts not in prices.index:
            mask = prices.index >= pub_ts
            if mask.sum() == 0:
                continue
            pub_ts = prices.index[mask][0]

        t0_pos = prices.index.get_loc(pub_ts)
        if t0_pos + 1 >= len(prices):
            continue

        filing_records.append({
            "ticker": ticker,
            "year": year,
            "filing_volume_t1": prices.iloc[t0_pos + 1]["Volume_ATS"],
            "filing_return_t1": prices.loc[pub_ts, "return_t1"] if "return_t1" in prices.columns else None,
        })

        # Collect day-by-day event rows anchored on filing date
        if t0_pos >= EVENT_HALF_WINDOW and t0_pos < len(prices) - EVENT_HALF_WINDOW:
            ind_aligned_f = industry_df.reindex(prices.index)
            for d in EVENT_DAYS:
                pos = t0_pos + d
                if 0 <= pos < len(prices):
                    cum_ret_col = f"return_t{d}"
                    cum_ret = prices.loc[pub_ts, cum_ret_col] if cum_ret_col in prices.columns else None
                    ind_cum_ret = ind_aligned_f.loc[pub_ts, cum_ret_col] if (pub_ts in ind_aligned_f.index and cum_ret_col in ind_aligned_f.columns) else None
                    filing_event_rows.append({
                        "ticker": ticker,
                        "year": year,
                        "industry": entry["industry"],
                        "surprise_pct": surprise_lookup.get((ticker, year)),
                        "relative_day": d,
                        "raw_volatility": prices.iloc[pos]["Volatility"],
                        "raw_volume": prices.iloc[pos]["Volume_ATS"],
                        "cum_return": cum_ret,
                        "cum_return_unbiased": (cum_ret - ind_cum_ret) if (cum_ret is not None and ind_cum_ret is not None and not pd.isna(ind_cum_ret)) else None,
                    })

    df_filing = pd.DataFrame(filing_records)
    if not df_filing.empty:
        df_filing["filing_volume_t1"] = pd.to_numeric(df_filing["filing_volume_t1"], errors="coerce")
        df_filing["filing_return_t1"] = pd.to_numeric(df_filing["filing_return_t1"], errors="coerce")

        print(f"\n--- Comparison: Earnings Call vs 10-K Filing ---")
        print(f"  Mean |return_t1| at EARNINGS: {df['return_t1'].abs().mean():.4f}")
        print(f"  Mean |return_t1| at 10-K FILING: {df_filing['filing_return_t1'].abs().mean():.4f}")
        print(f"  Mean volume_t1 at EARNINGS: {df['volume_t1'].mean():.2f}")
        print(f"  Mean volume_t1 at 10-K FILING: {df_filing['filing_volume_t1'].mean():.2f}")

    # --- Plot: distribution of |return| and volume at earnings vs filing ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    earn_ret = df["return_t1"].dropna().abs()
    filing_ret = df_filing["filing_return_t1"].dropna().abs() if not df_filing.empty else pd.Series(dtype=float)
    bins = np.linspace(0, 0.15, 30)
    if not earn_ret.empty:
        ax.hist(earn_ret, bins=bins, alpha=0.7, color="#e74c3c", label=f"Earnings Call (μ={earn_ret.mean():.4f})")
    if not filing_ret.empty:
        ax.hist(filing_ret, bins=bins, alpha=0.7, color="#3498db", label=f"10-K Filing (μ={filing_ret.mean():.4f})")
    ax.set_xlabel("|Return t+1|")
    ax.set_ylabel("Count")
    ax.set_title("|Return| at t+1: Earnings vs 10-K Filing")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    earn_vol = df["volume_t1"].dropna()
    filing_vol = df_filing["filing_volume_t1"].dropna() if not df_filing.empty else pd.Series(dtype=float)
    bins = np.linspace(0, 6, 30)
    if not earn_vol.empty:
        ax.hist(earn_vol, bins=bins, alpha=0.7, color="#e74c3c", label=f"Earnings Call (μ={earn_vol.mean():.2f})")
    if not filing_vol.empty:
        ax.hist(filing_vol, bins=bins, alpha=0.7, color="#3498db", label=f"10-K Filing (μ={filing_vol.mean():.2f})")
    ax.set_xlabel("Volume ATS at t+1")
    ax.set_ylabel("Count")
    ax.set_title("Volume at t+1: Earnings vs 10-K Filing")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Market Reaction: Earnings Call vs 10-K Filing\n(same tickers & fiscal years)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = DIR_DISTRIB / "earnings_vs_filing_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # --- Overlay event study: T0=earnings vs T0=filing on same graph ---
    if filing_event_rows:
        df_filing_event = pd.DataFrame(filing_event_rows)
        for c in ["raw_volatility", "raw_volume", "cum_return", "cum_return_unbiased"]:
            df_filing_event[c] = pd.to_numeric(df_filing_event[c], errors="coerce")

        n_earn = df_event.groupby(["ticker", "year"]).ngroups
        n_filing = df_filing_event.groupby(["ticker", "year"]).ngroups

        for col, title in event_metrics:
            fig, ax = plt.subplots(figsize=(12, 5))

            # Earnings date curve
            earn_means = df_event.groupby("relative_day")[col].mean()
            earn_sems = df_event.groupby("relative_day")[col].sem()
            y_earn = np.array([earn_means.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
            y_earn_sem = np.array([earn_sems.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
            ax.plot(x_positions, y_earn, color="#e74c3c", linewidth=2,
                    label=f"T0 = Earnings Date (n={n_earn})", marker="o", markersize=3)
            ax.fill_between(x_positions, y_earn - 1.96 * y_earn_sem, y_earn + 1.96 * y_earn_sem,
                            color="#e74c3c", alpha=0.12)

            # Filing date curve
            fil_means = df_filing_event.groupby("relative_day")[col].mean()
            fil_sems = df_filing_event.groupby("relative_day")[col].sem()
            y_fil = np.array([fil_means.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
            y_fil_sem = np.array([fil_sems.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
            ax.plot(x_positions, y_fil, color="#3498db", linewidth=2,
                    label=f"T0 = 10-K Filing Date (n={n_filing})", marker="s", markersize=3)
            ax.fill_between(x_positions, y_fil - 1.96 * y_fil_sem, y_fil + 1.96 * y_fil_sem,
                            color="#3498db", alpha=0.12)

            ax.axvline(_DAY_TO_POS[0], color="gray", linestyle="--", alpha=0.7, label="T0")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(d) for d in EVENT_DAYS], fontsize=8)
            ax.set_xlabel("Trading days relative to T0")
            ax.set_ylabel(title)
            ax.set_title(f"{title}\n(T0 = Earnings Date vs T0 = 10-K Filing Date, 95% CI)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(DIR_COMPARE / f"compare_{col}.png", dpi=150)
            plt.close(fig)

        print(f"Saved earnings vs filing overlay plots to {DIR_COMPARE}/")

        # --- Per-industry overlay: T0=earnings vs T0=filing ---
        DIR_COMPARE_IND = DIR_COMPARE / "by_industry"
        DIR_COMPARE_IND.mkdir(parents=True, exist_ok=True)

        industries_earn = df_filing_event["industry"].dropna().unique() if "industry" in df_filing_event.columns else []
        for ind in sorted(industries_earn):
            df_e_ind = df_event[df_event["industry"] == ind] if "industry" in df_event.columns else pd.DataFrame()
            df_f_ind = df_filing_event[df_filing_event["industry"] == ind]
            if df_e_ind.empty and df_f_ind.empty:
                continue

            n_e = df_e_ind.groupby(["ticker", "year"]).ngroups if not df_e_ind.empty else 0
            n_f = df_f_ind.groupby(["ticker", "year"]).ngroups if not df_f_ind.empty else 0
            ind_slug = ind.lower().replace(" / ", "-").replace(" & ", "-").replace(" ", "-").replace("---", "-")

            for col, title in event_metrics:
                fig, ax = plt.subplots(figsize=(12, 5))

                if not df_e_ind.empty:
                    e_means = df_e_ind.groupby("relative_day")[col].mean()
                    e_sems = df_e_ind.groupby("relative_day")[col].sem()
                    y_e = np.array([e_means.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
                    y_e_sem = np.array([e_sems.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
                    ax.plot(x_positions, y_e, color="#e74c3c", linewidth=2,
                            label=f"T0 = Earnings (n={n_e})", marker="o", markersize=3)
                    ax.fill_between(x_positions, y_e - 1.96 * y_e_sem, y_e + 1.96 * y_e_sem,
                                    color="#e74c3c", alpha=0.12)

                if not df_f_ind.empty:
                    f_means = df_f_ind.groupby("relative_day")[col].mean()
                    f_sems = df_f_ind.groupby("relative_day")[col].sem()
                    y_f = np.array([f_means.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
                    y_f_sem = np.array([f_sems.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
                    ax.plot(x_positions, y_f, color="#3498db", linewidth=2,
                            label=f"T0 = Filing (n={n_f})", marker="s", markersize=3)
                    ax.fill_between(x_positions, y_f - 1.96 * y_f_sem, y_f + 1.96 * y_f_sem,
                                    color="#3498db", alpha=0.12)

                ax.axvline(_DAY_TO_POS[0], color="gray", linestyle="--", alpha=0.7, label="T0")
                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(d) for d in EVENT_DAYS], fontsize=8)
                ax.set_xlabel("Trading days relative to T0")
                ax.set_ylabel(title)
                ax.set_title(f"{title}\n{ind} — Earnings vs Filing Date, 95% CI")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(DIR_COMPARE_IND / f"{ind_slug}_compare_{col}.png", dpi=150)
                plt.close(fig)

            print(f"  {ind}: saved ({n_e} earnings, {n_f} filing)")
        print(f"Saved per-industry overlay plots to {DIR_COMPARE_IND}/")

    # --- Event study at filing date, split by surprise sign ---
    if filing_event_rows:
        df_fe = pd.DataFrame(filing_event_rows) if not isinstance(df_filing_event, pd.DataFrame) else df_filing_event
        # df_filing_event already built above; add surprise_pct conversion
        df_fe = pd.DataFrame(filing_event_rows)
        for c in ["raw_volatility", "raw_volume", "cum_return", "cum_return_unbiased", "surprise_pct"]:
            df_fe[c] = pd.to_numeric(df_fe[c], errors="coerce")

        df_fe_pos = df_fe[df_fe["surprise_pct"] > 0]
        df_fe_neg = df_fe[df_fe["surprise_pct"] < 0]
        n_pos = df_fe_pos.groupby(["ticker", "year"]).ngroups
        n_neg = df_fe_neg.groupby(["ticker", "year"]).ngroups

        for col, title in event_metrics:
            fig, ax = plt.subplots(figsize=(12, 5))

            # Positive surprise
            if not df_fe_pos.empty:
                pos_means = df_fe_pos.groupby("relative_day")[col].mean()
                pos_sems = df_fe_pos.groupby("relative_day")[col].sem()
                y_pos = np.array([pos_means.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
                y_pos_sem = np.array([pos_sems.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
                ax.plot(x_positions, y_pos, color="#27ae60", linewidth=2,
                        label=f"Surprise > 0 (n={n_pos})", marker="o", markersize=3)
                ax.fill_between(x_positions, y_pos - 1.96 * y_pos_sem, y_pos + 1.96 * y_pos_sem,
                                color="#27ae60", alpha=0.12)

            # Negative surprise
            if not df_fe_neg.empty:
                neg_means = df_fe_neg.groupby("relative_day")[col].mean()
                neg_sems = df_fe_neg.groupby("relative_day")[col].sem()
                y_neg = np.array([neg_means.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
                y_neg_sem = np.array([neg_sems.get(d, np.nan) for d in EVENT_DAYS], dtype=float)
                ax.plot(x_positions, y_neg, color="#e74c3c", linewidth=2,
                        label=f"Surprise < 0 (n={n_neg})", marker="s", markersize=3)
                ax.fill_between(x_positions, y_neg - 1.96 * y_neg_sem, y_neg + 1.96 * y_neg_sem,
                                color="#e74c3c", alpha=0.12)

            ax.axvline(_DAY_TO_POS[0], color="gray", linestyle="--", alpha=0.7, label="Filing date (T0)")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(d) for d in EVENT_DAYS], fontsize=8)
            ax.set_xlabel("Trading days relative to 10-K Filing Date")
            ax.set_ylabel(title)
            ax.set_title(f"{title}\n(T0 = 10-K Filing Date, split by EPS Surprise sign, 95% CI)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(DIR_FILING_SURPRISE / f"filing_{col}_by_surprise.png", dpi=150)
            plt.close(fig)

        print(f"Saved filing-date event study by surprise to {DIR_FILING_SURPRISE}/ (pos={n_pos}, neg={n_neg})")

    # --- Difference plot: (surprise>0 - surprise<0) at earnings vs filing ---
    # Shows how much the surprise sign separates the market reaction depending on anchor
    df_event_pos = df_event[df_event["surprise_pct"] > 0]
    df_event_neg = df_event[df_event["surprise_pct"] < 0]

    if filing_event_rows and not df_event_pos.empty and not df_event_neg.empty:
        df_fe = pd.DataFrame(filing_event_rows)
        for c in ["raw_volatility", "raw_volume", "cum_return", "cum_return_unbiased", "surprise_pct"]:
            df_fe[c] = pd.to_numeric(df_fe[c], errors="coerce")
        df_fe_pos = df_fe[df_fe["surprise_pct"] > 0]
        df_fe_neg = df_fe[df_fe["surprise_pct"] < 0]

        for col, title in event_metrics:
            fig, ax = plt.subplots(figsize=(12, 5))

            # Earnings-date difference: mean(pos) - mean(neg)
            earn_pos_means = df_event_pos.groupby("relative_day")[col].mean()
            earn_neg_means = df_event_neg.groupby("relative_day")[col].mean()
            y_earn_diff = np.array([earn_pos_means.get(d, np.nan) - earn_neg_means.get(d, np.nan)
                                    for d in EVENT_DAYS], dtype=float)
            ax.plot(x_positions, y_earn_diff, color="#e74c3c", linewidth=2,
                    label="T0 = Earnings Date", marker="o", markersize=3)

            # Filing-date difference: mean(pos) - mean(neg)
            if not df_fe_pos.empty and not df_fe_neg.empty:
                fil_pos_means = df_fe_pos.groupby("relative_day")[col].mean()
                fil_neg_means = df_fe_neg.groupby("relative_day")[col].mean()
                y_fil_diff = np.array([fil_pos_means.get(d, np.nan) - fil_neg_means.get(d, np.nan)
                                       for d in EVENT_DAYS], dtype=float)
                ax.plot(x_positions, y_fil_diff, color="#3498db", linewidth=2,
                        label="T0 = 10-K Filing Date", marker="s", markersize=3)

            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.axvline(_DAY_TO_POS[0], color="gray", linestyle="--", alpha=0.7, label="T0")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(d) for d in EVENT_DAYS], fontsize=8)
            ax.set_xlabel("Trading days relative to T0")
            ax.set_ylabel(f"Δ {title}")
            ax.set_title(f"Separation by EPS Surprise: mean(surprise>0) − mean(surprise<0)\n{title}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(DIR_COMPARE / f"surprise_separation_{col}.png", dpi=150)
            plt.close(fig)

        print(f"Saved surprise separation plots to {DIR_COMPARE}/")

    # --- Plot: surprise vs return ---
    df_surprise = df.dropna(subset=["surprise_pct", "return_t1"])
    if len(df_surprise) > 10:
        # Winsorize: clip extreme surprises for visualization (keep all for stats)
        q_lo, q_hi = df_surprise["surprise_pct"].quantile(0.02), df_surprise["surprise_pct"].quantile(0.98)
        xlim_lo, xlim_hi = q_lo, q_hi
        df_inlier = df_surprise[(df_surprise["surprise_pct"] >= xlim_lo) & (df_surprise["surprise_pct"] <= xlim_hi)]
        n_outliers = len(df_surprise) - len(df_inlier)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Color by surprise sign
        pos_mask = df_inlier["surprise_pct"] >= 0
        ax.scatter(df_inlier.loc[pos_mask, "surprise_pct"], df_inlier.loc[pos_mask, "return_t1"],
                   alpha=0.6, s=30, color="#2ecc71", edgecolors="white", linewidth=0.3, label=f"Surprise ≥ 0 (n={pos_mask.sum()})")
        ax.scatter(df_inlier.loc[~pos_mask, "surprise_pct"], df_inlier.loc[~pos_mask, "return_t1"],
                   alpha=0.6, s=30, color="#e74c3c", edgecolors="white", linewidth=0.3, label=f"Surprise < 0 (n={(~pos_mask).sum()})")

        # Regression on inliers only
        x = df_inlier["surprise_pct"].values
        y = df_inlier["return_t1"].values
        m, b = np.polyfit(x, y, 1)
        x_range = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_range, m * x_range + b, color="black", linewidth=2,
                label=f"OLS: y = {m:.5f}x + {b:.4f}")

        # Correlation (full dataset + inliers)
        corr_all = df_surprise[["surprise_pct", "return_t1"]].corr().iloc[0, 1]
        corr_inlier = df_inlier[["surprise_pct", "return_t1"]].corr().iloc[0, 1]

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlim(xlim_lo, xlim_hi)
        ax.set_xlabel("EPS Surprise (%)", fontsize=11)
        ax.set_ylabel("Return t+1 (next-day)", fontsize=11)
        ax.set_title(f"EPS Surprise vs Next-Day Return at Earnings Call\n"
                     f"(n={len(df_inlier)}, {n_outliers} outliers clipped outside [{xlim_lo:.0f}%, {xlim_hi:.0f}%])",
                     fontsize=12)
        ax.legend(title=f"r = {corr_inlier:.3f} (inliers)  |  r = {corr_all:.3f} (all {len(df_surprise)})",
                  fontsize=9, title_fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = DIR_DISTRIB / "surprise_vs_return.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
        print(f"Surprise-Return correlation: r_inlier={corr_inlier:.4f}, r_all={corr_all:.4f} ({n_outliers} outliers clipped)")

    # --- Comparison: Sentiment separation vs Surprise separation (both at earnings date) ---
    # SAME SAMPLE: restrict to events that have BOTH a sentiment label AND a surprise value
    # This avoids selection bias (sentiment is only available for ~71 CEO letters)
    DIR_DELTA_COMPARE = OUTPUT_BASE / "sentiment_vs_surprise_delta"
    DIR_DELTA_COMPARE.mkdir(parents=True, exist_ok=True)

    # Common sample: events with a sentiment label (positive or negative) AND surprise data
    df_event_with_sent = df_event[df_event["sentiment"].isin(["positive", "negative"])].copy()
    df_event_common = df_event_with_sent[df_event_with_sent["surprise_pct"].notna()].copy()

    # Sentiment split on the common sample
    df_common_sent_pos = df_event_common[df_event_common["sentiment"] == "positive"]
    df_common_sent_neg = df_event_common[df_event_common["sentiment"] == "negative"]

    # Surprise split on the SAME common sample
    df_common_surp_pos = df_event_common[df_event_common["surprise_pct"] > 0]
    df_common_surp_neg = df_event_common[df_event_common["surprise_pct"] < 0]

    has_sentiment = not df_common_sent_pos.empty and not df_common_sent_neg.empty
    has_surprise = not df_common_surp_pos.empty and not df_common_surp_neg.empty

    if has_sentiment or has_surprise:
        n_common = df_event_common.groupby(["ticker", "year"]).ngroups
        n_sent_pos = df_common_sent_pos.groupby(["ticker", "year"]).ngroups if has_sentiment else 0
        n_sent_neg = df_common_sent_neg.groupby(["ticker", "year"]).ngroups if has_sentiment else 0
        n_surp_pos = df_common_surp_pos.groupby(["ticker", "year"]).ngroups if has_surprise else 0
        n_surp_neg = df_common_surp_neg.groupby(["ticker", "year"]).ngroups if has_surprise else 0

        for col, title in event_metrics:
            fig, ax = plt.subplots(figsize=(12, 5))

            # Sentiment: |neg - pos|
            if has_sentiment:
                sent_pos_means = df_common_sent_pos.groupby("relative_day")[col].mean()
                sent_neg_means = df_common_sent_neg.groupby("relative_day")[col].mean()
                y_sent_delta = np.array([abs(sent_neg_means.get(d, np.nan) - sent_pos_means.get(d, np.nan))
                                         for d in EVENT_DAYS], dtype=float)
                ax.plot(x_positions, y_sent_delta, color="#9b59b6", linewidth=2.5,
                        label=f"Sentiment |neg−pos| (n={n_sent_neg}+{n_sent_pos})",
                        marker="o", markersize=4)

            # Surprise: |surprise<0 - surprise>0| on same sample
            if has_surprise:
                surp_pos_means = df_common_surp_pos.groupby("relative_day")[col].mean()
                surp_neg_means = df_common_surp_neg.groupby("relative_day")[col].mean()
                y_surp_delta = np.array([abs(surp_neg_means.get(d, np.nan) - surp_pos_means.get(d, np.nan))
                                         for d in EVENT_DAYS], dtype=float)
                ax.plot(x_positions, y_surp_delta, color="#e67e22", linewidth=2.5,
                        label=f"Surprise |neg−pos| (n={n_surp_neg}+{n_surp_pos})",
                        marker="s", markersize=4)

            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.axvline(_DAY_TO_POS[0], color="gray", linestyle="--", alpha=0.7, label="Earnings date")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(d) for d in EVENT_DAYS], fontsize=8)
            ax.set_xlabel("Trading days relative to Earnings Date")
            ax.set_ylabel(f"|Δ| {title}")
            ax.set_title(f"Discrimination power: Sentiment vs EPS Surprise (same sample, n={n_common})\n|negative − positive| for {title}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(DIR_DELTA_COMPARE / f"delta_compare_{col}.png", dpi=150)
            plt.close(fig)

        print(f"Saved sentiment vs surprise delta plots to {DIR_DELTA_COMPARE}/ (common sample: {n_common} events)")


if __name__ == "__main__":
    main()
