"""
Plot average unbiased return, volatility and volume at publication date,
grouped by CEO-letter sentiment (positive / negative / neutral).
"""

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_indicators import annual_publication_dates
# -- local imports (run from repo root or KPI_analysis/) --
sys.path.insert(0, str(Path(__file__).resolve().parent))

from FinancialIndicators import (
    GetIndicatorsForPrices,
    GetIndustryDataFrame,
)
from fetch_filing_returns import fetch_prices

HERE = Path(__file__).resolve().parent
SENTIMENTS_JSON = (
    HERE.parent
    / "doc_text_processing"
    / "CEO_word_extraction"
    / "cleaning_extractions"
    / "cleaned"
    / "sentiments.json"
)
OUTPUT_DIR = HERE / "output" / "plots" / "sentiment_summary_augmented_lag"

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2023, 6, 30)
WINDOW = 5  # trading days around publication to average over
EVENT_HALF_WINDOW = 90  # trading days before/after publication for event study

# Days to sample: every 10 days from -90 to -20, then daily -10 to +10, then every 10 days from +20 to +90
EVENT_DAYS = list(range(-90, -10, 10)) + list(range(-10, 11)) + list(range(20, 91, 10))
# Map real day values to evenly-spaced positions for plotting
_DAY_TO_POS = {d: i for i, d in enumerate(EVENT_DAYS)}


def load_sentiments() -> list[dict]:
    """Return flat list of {industry, ticker, year, sentiment}."""
    with open(SENTIMENTS_JSON) as f:
        data = json.load(f)
    rows = []
    for industry, tickers in data.items():
        for ticker, years in tickers.items():
            for year, sentiment in years.items():
                if sentiment is not None:
                    rows.append(
                        {
                            "industry": industry,
                            "ticker": ticker,
                            "year": int(year),
                            "sentiment": sentiment,
                        }
                    )
    return rows


def _pub_date_for_fy(ticker: str, fy: int) -> pd.Timestamp | None:
    """Return the publication date for the given fiscal year, or None."""
    pubs = annual_publication_dates(ticker, originals_only=True)
    if pubs.empty:
        return None
    pubs = pubs.copy()
    pubs["report_date"] = pd.to_datetime(pubs["report_date"])
    # Derive filer FY from report_date using the project convention
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


def _window_mean(series: pd.Series, target_date: pd.Timestamp, window: int) -> float | None:
    """Return mean of `series` in a [target_date, target_date + window trading days] window."""
    series = series.dropna()
    if series.empty:
        return None
    # Find the closest trading day >= target_date
    mask = series.index >= target_date
    if mask.sum() == 0:
        return None
    start_idx = series.index[mask][0]
    start_pos = series.index.get_loc(start_idx)
    end_pos = min(start_pos + window, len(series) - 1)
    chunk = series.iloc[start_pos : end_pos + 1]
    if chunk.empty:
        return None
    return chunk.mean()


def _event_window_values(series: pd.Series, target_date: pd.Timestamp, half_window: int, days: list[int] | None = None) -> dict[int, float]:
    """Return {relative_day: value} for selected trading days around target_date."""
    series = series.dropna()
    if series.empty:
        return {}
    mask = series.index >= target_date
    if mask.sum() == 0:
        return {}
    t0_idx = series.index[mask][0]
    t0_pos = series.index.get_loc(t0_idx)
    result = {}
    iter_days = days if days is not None else range(-half_window, half_window + 1)
    for d in iter_days:
        pos = t0_pos + d
        if 0 <= pos < len(series):
            result[d] = series.iloc[pos]
    return result


def main():
    sentiments = load_sentiments()
    print(f"Loaded {len(sentiments)} (ticker, year, sentiment) entries")

    # Collect per-sentiment indicator values
    records = []  # (sentiment, unbiased_return, unbiased_vol, unbiased_volume)
    event_rows = []  # (sentiment, relative_day, unbiased_return, unbiased_volatility, unbiased_volume)

    processed_tickers: dict[str, tuple] = {}  # ticker -> (prices_df, industry_df)

    for i, row in enumerate(sentiments):
        ticker = row["ticker"]
        year = row["year"]
        sentiment = row["sentiment"]

        print(f"[{i+1}/{len(sentiments)}] {ticker} {year} ({sentiment}) ... ", end="", flush=True)

        # Get publication date
        pub_date = _pub_date_for_fy(ticker, year)
        if pub_date is None:
            print("no publication date")
            continue

        # Fetch / cache prices + indicators
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

        # Compute unbiased indicators (reindex industry to ticker's dates to avoid index union)
        ind_aligned = industry_df.reindex(prices.index)
        unbiased_vol = prices["Volatility"] - ind_aligned["volatility"]
        unbiased_volume = prices["Volume_ATS"] - ind_aligned["volumes"]

        # Raw stock indicators
        raw_vol = prices["Volatility"]
        raw_volume = prices["Volume_ATS"]
        raw_price = prices["Close"]

        # Volume-weighted unbiased indicators
        unbiased_volume_vw = prices["Volume_ATS"] - ind_aligned["volumes_vw"] if "volumes_vw" in ind_aligned.columns else pd.Series(dtype=float)

        # Get values in the window around publication
        v = _window_mean(unbiased_vol, pub_date, WINDOW)
        vol = _window_mean(unbiased_volume, pub_date, WINDOW)
        rv = _window_mean(raw_vol, pub_date, WINDOW)
        rvol = _window_mean(raw_volume, pub_date, WINDOW)
        vol_vw = _window_mean(unbiased_volume_vw, pub_date, WINDOW)
        rp = _window_mean(raw_price, pub_date, WINDOW)

        pub_date_ts = pd.Timestamp(pub_date)

        cum_ret_1 = prices.loc[pub_date_ts, "return_t1"] if pub_date_ts in prices.index else None
        cum_ret_5 = prices.loc[pub_date_ts, "return_t5"] if pub_date_ts in prices.index else None
        ind_ret_1 = ind_aligned.loc[pub_date_ts, "return_t1"] if (pub_date_ts in ind_aligned.index and "return_t1" in ind_aligned.columns) else None
        ind_ret_5 = ind_aligned.loc[pub_date_ts, "return_t5"] if (pub_date_ts in ind_aligned.index and "return_t5" in ind_aligned.columns) else None
        ind_ret_1_vw = ind_aligned.loc[pub_date_ts, "return_t1_vw"] if (pub_date_ts in ind_aligned.index and "return_t1_vw" in ind_aligned.columns) else None
        ind_ret_5_vw = ind_aligned.loc[pub_date_ts, "return_t5_vw"] if (pub_date_ts in ind_aligned.index and "return_t5_vw" in ind_aligned.columns) else None
        cum_ret_1_unbiased = (cum_ret_1 - ind_ret_1) if (cum_ret_1 is not None and ind_ret_1 is not None) else None
        cum_ret_5_unbiased = (cum_ret_5 - ind_ret_5) if (cum_ret_5 is not None and ind_ret_5 is not None) else None
        cum_ret_1_unbiased_vw = (cum_ret_1 - ind_ret_1_vw) if (cum_ret_1 is not None and ind_ret_1_vw is not None) else None
        cum_ret_5_unbiased_vw = (cum_ret_5 - ind_ret_5_vw) if (cum_ret_5 is not None and ind_ret_5_vw is not None) else None


        # Collect event-window day-by-day values
        v_evt = _event_window_values(unbiased_vol, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS)
        vol_evt = _event_window_values(unbiased_volume, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS)
        rv_evt = _event_window_values(raw_vol, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS)
        rvol_evt = _event_window_values(raw_volume, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS)
        rp_evt = _event_window_values(raw_price, pub_date, EVENT_HALF_WINDOW, EVENT_DAYS)
        # Industry-level indicators for the same event window
        ind_v_evt = _event_window_values(industry_df["volatility"], pub_date, EVENT_HALF_WINDOW, EVENT_DAYS)
        ind_vol_evt = _event_window_values(industry_df["volumes"], pub_date, EVENT_HALF_WINDOW, EVENT_DAYS)
        # Volume-weighted industry indicators
        ind_vol_vw_evt = _event_window_values(ind_aligned["volumes_vw"], pub_date, EVENT_HALF_WINDOW, EVENT_DAYS) if "volumes_vw" in ind_aligned.columns else {}
        # Normalize price to J0 = 1 so tickers are comparable
        p0 = rp_evt.get(0)
        if p0 and p0 != 0:
            rp_evt_norm = {d: v / p0 for d, v in rp_evt.items()}
        else:
            rp_evt_norm = {d: None for d in rp_evt}

        # Normalize all metrics relative to day 0: (metric(t) - metric(0)) / metric(0)
        def _normalize_to_day0(evt_dict):
            v0 = evt_dict.get(0)
            if v0 is None or v0 == 0:
                return {d: None for d in evt_dict}
            return {d: ((v - v0) / abs(v0) if v is not None else None) for d, v in evt_dict.items()}

        v_evt_norm = _normalize_to_day0(v_evt)
        vol_evt_norm = _normalize_to_day0(vol_evt)
        rv_evt_norm = _normalize_to_day0(rv_evt)
        rvol_evt_norm = _normalize_to_day0(rvol_evt)



        

        # Volume unbiased VW dict (raw_volume - industry_volume_vw) then normalize
        vol_ub_vw_evt = {d: (rvol_evt.get(d) - ind_vol_vw_evt.get(d)) if (rvol_evt.get(d) is not None and ind_vol_vw_evt.get(d) is not None) else None for d in EVENT_DAYS}
        vol_ub_vw_evt_norm = _normalize_to_day0(vol_ub_vw_evt)

        for d in EVENT_DAYS:
            if d in v_evt or d in vol_evt:
                # Cumulative return from J0 to day d (from FinancialIndicators)
                cum_ret_col = f"return_t{d}"
                cum_ret_col_vw = f"return_t{d}_vw"
                cum_ret = prices.loc[pub_date_ts, cum_ret_col] if (pub_date_ts in prices.index and cum_ret_col in prices.columns) else None
                cum_ret_ind = ind_aligned.loc[pub_date_ts, cum_ret_col] if (pub_date_ts in ind_aligned.index and cum_ret_col in ind_aligned.columns) else None
                cum_ret_ind_vw = ind_aligned.loc[pub_date_ts, cum_ret_col_vw] if (pub_date_ts in ind_aligned.index and cum_ret_col_vw in ind_aligned.columns) else None

                event_rows.append(
                    {
                        "sentiment": sentiment,
                        "industry": row["industry"],
                        "ticker": ticker,
                        "year": year,
                        "relative_day": d,
                        "unbiased_volatility": v_evt.get(d),
                        "unbiased_volume": vol_evt.get(d),
                        "raw_volatility": rv_evt.get(d),
                        "raw_volume": rvol_evt.get(d),
                        "raw_price_norm": rp_evt_norm.get(d),
                        "industry_volatility": ind_v_evt.get(d),
                        "industry_volume": ind_vol_evt.get(d),
                        # Normalized to day 0
                        "norm_unbiased_volatility": v_evt_norm.get(d),
                        "norm_unbiased_volume": vol_evt_norm.get(d),
                        "norm_raw_volatility": rv_evt_norm.get(d),
                        "norm_raw_volume": rvol_evt_norm.get(d),
                        # Cumulative return from publication day
                        "cum_return": cum_ret,
                        # Cumulative return - industry cumulative 
                        "cum_return_unbiased": (cum_ret - cum_ret_ind) if (cum_ret is not None and cum_ret_ind is not None) else None,
                        # Cumulative return - industry cumulativeb volume-weighted
                        "cum_return_unbiased_vw": (cum_ret - cum_ret_ind_vw) if (cum_ret is not None and cum_ret_ind_vw is not None) else None,
                        # Volume - industry volume-weighted
                        "volume_unbiased_vw": vol_ub_vw_evt.get(d),
                        "norm_volume_unbiased_vw": vol_ub_vw_evt_norm.get(d),
                    }
                )

        if v is None and vol is None:
            print("no data at pub date")
            continue
        

        records.append(
            {
                "sentiment": sentiment,
                "industry": row["industry"],
                "ticker": ticker,
                "year": year,
                "unbiased_volatility": v,
                "unbiased_volume": vol,
                "raw_volatility": rv,
                "raw_volume": rvol,
                "raw_price": rp,
                "volume_unbiased_vw": vol_vw,
                "cum_return_1d": cum_ret_1,
                "cum_return_5d": cum_ret_5,
                "cum_return_1d_unbiased": cum_ret_1_unbiased,
                "cum_return_5d_unbiased": cum_ret_5_unbiased,
                "cum_return_1d_unbiased_vw": cum_ret_1_unbiased_vw,
                "cum_return_5d_unbiased_vw": cum_ret_5_unbiased_vw,
            }
        )

    if not records:
        print("No records to plot.")
        return

    df = pd.DataFrame(records)
    print(f"\nCollected {len(df)} data points")
    all_cols = ["unbiased_volatility", "unbiased_volume",
                "raw_volatility", "raw_volume", "raw_price", "volume_unbiased_vw",
                "cum_return_1d", "cum_return_5d",
                "cum_return_1d_unbiased", "cum_return_5d_unbiased",
                "cum_return_1d_unbiased_vw", "cum_return_5d_unbiased_vw"]

    # Save dataframes for inspection
    df.to_csv(OUTPUT_DIR.parent / "sentiment_records.csv", index=False)
    print(f"Saved {OUTPUT_DIR.parent / 'sentiment_records.csv'}")
    print(df.groupby("sentiment")[all_cols].mean().to_string())
    sys.stdout.flush()


    # --- Plot ---
    # Clean output directory structure
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "01_bar_charts").mkdir(exist_ok=True)
    (OUTPUT_DIR / "02_event_studies" / "aggregate" / "raw").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "02_event_studies" / "aggregate" / "normalized").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "02_event_studies" / "aggregate_no_outliers").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "02_event_studies" / "all_industries_overlay" / "raw").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "02_event_studies" / "all_industries_overlay" / "normalized").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "03_distributions" / "by_metric").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "03_distributions" / "no_outliers").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "03_distributions" / "pre_event").mkdir(parents=True, exist_ok=True)

    # sentiments_order = ["positive", "neutral", "negative"]
    sentiments_order = ["positive", "negative"]

    colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}

    metrics = [
        ("unbiased_volatility", "Unbiased Volatility (mean)"),
        ("unbiased_volume", "Unbiased Volume ATS (mean)"),
        ("raw_volatility", "Stock Volatility (mean)"),
        ("raw_volume", "Stock Volume ATS (mean)"),
        ("raw_price", "Stock Price (mean)"),
        ("volume_unbiased_vw", "Volume ATS - Industry VW (mean)"),
        ("cum_return_1d", "Cumulative Return J+1"),
        ("cum_return_5d", "Cumulative Return J+5"),
        ("cum_return_1d_unbiased", "Cum Return J+1 - Industry"),
        ("cum_return_5d_unbiased", "Cum Return J+5 - Industry"),
        ("cum_return_1d_unbiased_vw", "Cum Return J+1 - Industry VW"),
        ("cum_return_5d_unbiased_vw", "Cum Return J+5 - Industry VW"),
    ]

    # Event study uses normalized price
    event_metrics = [
        ("unbiased_volatility", "Unbiased Volatility (mean)"),
        ("unbiased_volume", "Unbiased Volume ATS (mean)"),
        ("raw_volatility", "Stock Volatility (mean)"),
        ("raw_volume", "Stock Volume ATS (mean)"),
        ("raw_price_norm", "Stock Price (normalized, J0=1)"),
        ("cum_return", "Cumulative Return from J0"),
        ("cum_return_unbiased", "Cumulative Return - Industry Cum Return"),
        ("cum_return_unbiased_vw", "Cumulative Return - Industry Cum Return VW"),
        ("volume_unbiased_vw", "Volume ATS - Industry Volume VW"),
    ]

    event_metrics_normalized = [
        ("norm_unbiased_volatility", "Unbiased Volatility (% change from J0)"),
        ("norm_unbiased_volume", "Unbiased Volume (% change from J0)"),
        ("norm_raw_volatility", "Stock Volatility (% change from J0)"),
        ("norm_raw_volume", "Stock Volume (% change from J0)"),
        ("norm_volume_unbiased_vw", "Volume ATS - Industry VW (% change from J0)"),
    ]

    all_event_metrics = event_metrics + event_metrics_normalized
    normalized_cols = {c for c, _ in event_metrics_normalized}

    # --- Bar charts: one figure per metric ---
    for col, title in metrics:
        fig, ax = plt.subplots(figsize=(6, 5))
        means = []
        stds = []
        labels = []
        bar_colors = []
        for s in sentiments_order:
            subset = df[df["sentiment"] == s][col].dropna()
            if len(subset) == 0:
                continue
            means.append(subset.mean())
            stds.append(subset.std() / np.sqrt(len(subset)))  # SEM
            labels.append(f"{s}\n(n={len(subset)})")
            bar_colors.append(colors[s])

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=bar_colors, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        if col.startswith("cum_return"):
            ax.set_title(f"{title}\n({len(df)} obs.)",
                          fontsize=12, fontweight="bold")
        else:
            ax.set_title(f"{title}\n(window = {WINDOW} trading days, {len(df)} obs.)",
                          fontsize=12, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = OUTPUT_DIR / "01_bar_charts" / f"bar_{col}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.show()

    # --- Event study plots: one figure per metric ---
    edf = pd.DataFrame(event_rows)
    if edf.empty:
        print("No event-window data to plot.")
        return

    # Force numeric dtype on all metric columns (None → NaN as float)
    metric_cols = [c for c, _ in all_event_metrics]
    for mc in metric_cols:
        if mc in edf.columns:
            edf[mc] = pd.to_numeric(edf[mc], errors="coerce")

    for col, title in all_event_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for s in sentiments_order:
            subset = edf[edf["sentiment"] == s]
            if subset.empty:
                continue
            grouped = subset.groupby("relative_day")[col]
            means = grouped.mean()
            sems = grouped.sem()
            n_docs = subset.drop_duplicates(subset=["ticker", "year"]).shape[0]
            days = means.index.values
            positions = [_DAY_TO_POS[d] for d in days if d in _DAY_TO_POS]
            vals = [means[d] for d in days if d in _DAY_TO_POS]
            lo_ci = [(means[d] - 1.96 * sems[d]) for d in days if d in _DAY_TO_POS]
            hi_ci = [(means[d] + 1.96 * sems[d]) for d in days if d in _DAY_TO_POS]
            ax.plot(positions, vals, color=colors[s], label=f"{s} (n={n_docs} docs)", linewidth=1.5, marker='o', markersize=4)
            ax.fill_between(positions, lo_ci, hi_ci, color=colors[s], alpha=0.15)
        # Count observations per sentiment per day
        n_per_day = {}
        for s in sentiments_order:
            subset_s = edf[edf["sentiment"] == s]
            if not subset_s.empty and col in subset_s.columns:
                n_per_day[s] = subset_s.groupby("relative_day")[col].count()
            else:
                n_per_day[s] = pd.Series(dtype=int)
        tick_labels = []
        for d in EVENT_DAYS:
            parts = [str(d)]
            for s in sentiments_order:
                parts.append(str(int(n_per_day[s].get(d, 0))))
            tick_labels.append("\n".join(parts))
        ax.axvline(_DAY_TO_POS[0], color="black", linewidth=1, linestyle="--", alpha=0.7, label="Publication day")
        if col == "raw_price_norm":
            ax.axhline(1, color="grey", linewidth=0.5, linestyle=":")
        else:
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xticks(range(len(EVENT_DAYS)))
        ax.set_xticklabels(tick_labels, fontsize=7)
        sent_labels = " / ".join(sentiments_order)
        ax.set_xlabel(f"Trading days relative to publication\n(n: {sent_labels})", fontsize=10)
        ax.set_title(f"Event Study: {title}\n(±{EVENT_HALF_WINDOW} trading days, 95% CI)",
                      fontsize=12, fontweight="bold")
        # Clip y-axis for normalized metrics using 2nd/98th percentile
        if col in normalized_cols:
            all_vals = edf[col].dropna()
            if not all_vals.empty:
                lo, hi = all_vals.quantile(0.02), all_vals.quantile(0.98)
                margin = (hi - lo) * 0.1
                ax.set_ylim(lo - margin, hi + margin)
        ax.legend(fontsize=9)
        ax.grid(axis="both", alpha=0.3)
        plt.tight_layout()
        subdir = "02_event_studies/aggregate/normalized" if col in normalized_cols else "02_event_studies/aggregate/raw"
        out_path = OUTPUT_DIR / subdir / f"event_{col}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.show()

    # --- Event study plots per industry (all industries + sentiments on one graph per metric) ---
    industries = sorted(edf["industry"].unique())
    ind_cmap = plt.colormaps["tab10"]
    ind_colors = {ind: ind_cmap(i % 10) for i, ind in enumerate(industries)}
    sent_linestyles = {"positive": "-", "negative": "--"}

    for col, title in all_event_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        for industry in industries:
            c = ind_colors[industry]
            for s in sentiments_order:
                subset = edf[(edf["industry"] == industry) & (edf["sentiment"] == s)]
                if subset.empty:
                    continue
                grouped = subset.groupby("relative_day")[col]
                means = grouped.mean()
                sems = grouped.sem()
                n_docs = subset.drop_duplicates(subset=["ticker", "year"]).shape[0]
                if n_docs == 0:
                    continue
                has_data = True
                days = means.index.values
                positions = [_DAY_TO_POS[d] for d in days if d in _DAY_TO_POS]
                vals = [means[d] for d in days if d in _DAY_TO_POS]
                lo_ci = [(means[d] - 1.96 * sems[d]) for d in days if d in _DAY_TO_POS]
                hi_ci = [(means[d] + 1.96 * sems[d]) for d in days if d in _DAY_TO_POS]
                ls = sent_linestyles.get(s, "-")
                ax.plot(positions, vals, color=c, linestyle=ls,
                        label=f"{industry} / {s} (n={n_docs} docs)",
                        linewidth=1.5, marker='o', markersize=3)
                ax.fill_between(positions, lo_ci, hi_ci, color=c, alpha=0.07)
        if not has_data:
            plt.close(fig)
            continue
        # Count observations per sentiment per day (all industries combined)
        n_per_day = {}
        for s in sentiments_order:
            subset_s = edf[edf["sentiment"] == s]
            if not subset_s.empty and col in subset_s.columns:
                n_per_day[s] = subset_s.groupby("relative_day")[col].count()
            else:
                n_per_day[s] = pd.Series(dtype=int)
        tick_labels = []
        for d in EVENT_DAYS:
            parts = [str(d)]
            for s in sentiments_order:
                parts.append(str(int(n_per_day[s].get(d, 0))))
            tick_labels.append("\n".join(parts))
        ax.axvline(_DAY_TO_POS[0], color="black", linewidth=1, linestyle="--", alpha=0.7, label="Publication day")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xticks(range(len(EVENT_DAYS)))
        ax.set_xticklabels(tick_labels, fontsize=7)
        sent_labels = " / ".join(sentiments_order)
        ax.set_xlabel(f"Trading days relative to publication\n(n: {sent_labels})", fontsize=10)
        ax.set_title(f"Event Study: {title}\n"
                      f"(±{EVENT_HALF_WINDOW} days, 95% CI, by industry — solid=positive, dashed=negative)",
                      fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.grid(axis="both", alpha=0.3)
        plt.tight_layout()
        subdir = "02_event_studies/all_industries_overlay/normalized" if col in normalized_cols else "02_event_studies/all_industries_overlay/raw"
        out_path = OUTPUT_DIR / subdir / f"event_{col}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.show()

    # Map raw metric columns to their industry-level counterpart
    raw_to_industry_col = {
        "raw_return": "industry_return",
        "raw_volatility": "industry_volatility",
        "raw_volume": "industry_volume",
    }

    # --- Event study plots: one graph per industry (positive + negative on same plot) ---
    for industry in industries:
        ind_edf = edf[edf["industry"] == industry]
        ind_slug = industry.replace("/", "_").replace(" ", "_").replace(":", "_")
        ind_dir = OUTPUT_DIR / "02_event_studies" / "per_industry" / ind_slug / "raw"
        ind_dir_norm = OUTPUT_DIR / "02_event_studies" / "per_industry" / ind_slug / "normalized"
        ind_dir.mkdir(parents=True, exist_ok=True)
        ind_dir_norm.mkdir(parents=True, exist_ok=True)
        for col, title in all_event_metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            has_data = False
            for s in sentiments_order:
                subset = ind_edf[ind_edf["sentiment"] == s]
                if subset.empty:
                    continue
                grouped = subset.groupby("relative_day")[col]
                means = grouped.mean()
                sems = grouped.sem()
                n_docs = subset.drop_duplicates(subset=["ticker", "year"]).shape[0]
                if n_docs == 0:
                    continue
                has_data = True
                days = means.index.values
                positions = [_DAY_TO_POS[d] for d in days if d in _DAY_TO_POS]
                vals = [means[d] for d in days if d in _DAY_TO_POS]
                lo_ci = [(means[d] - 1.96 * sems[d]) for d in days if d in _DAY_TO_POS]
                hi_ci = [(means[d] + 1.96 * sems[d]) for d in days if d in _DAY_TO_POS]
                ax.plot(positions, vals, color=colors[s], label=f"{s} (n={n_docs} docs)",
                        linewidth=1.5, marker='o', markersize=4)
                ax.fill_between(positions, lo_ci, hi_ci, color=colors[s], alpha=0.15)
            # Plot industry average for raw metrics
            ind_col = raw_to_industry_col.get(col)
            if ind_col and ind_col in ind_edf.columns:
                ind_grouped = ind_edf.groupby("relative_day")[ind_col]
                ind_means = ind_grouped.mean()
                if not ind_means.dropna().empty:
                    ind_days = ind_means.index.values
                    ind_positions = [_DAY_TO_POS[d] for d in ind_days if d in _DAY_TO_POS]
                    ind_vals = [ind_means[d] for d in ind_days if d in _DAY_TO_POS]
                    ax.plot(ind_positions, ind_vals, color="blue",
                            linestyle=":", linewidth=2, alpha=0.7, label="Industry avg")
            if not has_data:
                plt.close(fig)
                continue
            # Count observations per sentiment per day
            n_per_day = {}
            for s in sentiments_order:
                subset_s = ind_edf[ind_edf["sentiment"] == s]
                if not subset_s.empty and col in subset_s.columns:
                    counts = subset_s.groupby("relative_day")[col].count()
                    n_per_day[s] = counts
                else:
                    n_per_day[s] = pd.Series(dtype=int)
            tick_labels = []
            for d in EVENT_DAYS:
                parts = [str(d)]
                for s in sentiments_order:
                    n = int(n_per_day[s].get(d, 0))
                    parts.append(f"{n}")
                tick_labels.append("\n".join(parts))
            ax.axvline(_DAY_TO_POS[0], color="black", linewidth=1, linestyle="--", alpha=0.7, label="Publication day")
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
            ax.set_xticks(range(len(EVENT_DAYS)))
            ax.set_xticklabels(tick_labels, fontsize=7)
            # Legend for tick sub-labels
            sent_labels = " / ".join(sentiments_order)
            ax.set_xlabel(f"Trading days relative to publication\n(n: {sent_labels})", fontsize=10)
            ax.set_title(f"Event Study: {title}\n{industry} (±{EVENT_HALF_WINDOW} days, 95% CI)",
                          fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(axis="both", alpha=0.3)
            plt.tight_layout()
            target_dir = ind_dir_norm if col in normalized_cols else ind_dir
            out_path = target_dir / f"event_{col}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()

    # --- Distribution of each metric for positive and negative sentiment ---
    dist_metrics = [
        ("unbiased_volatility", "Unbiased Volatility"),
        ("unbiased_volume", "Unbiased Volume ATS"),
        ("raw_volatility", "Stock Volatility"),
        ("raw_volume", "Stock Volume ATS"),
        ("raw_price", "Stock Price"),
        ("volume_unbiased_vw", "Volume ATS - Industry Volume VW"),
        ("cum_return_1d", "Cumulative Return J+1"),
        ("cum_return_5d", "Cumulative Return J+5"),
        ("cum_return_1d_unbiased", "Cum Return J+1 - Industry"),
        ("cum_return_5d_unbiased", "Cum Return J+5 - Industry"),
        ("cum_return_1d_unbiased_vw", "Cum Return J+1 - Industry VW"),
        ("cum_return_5d_unbiased_vw", "Cum Return J+5 - Industry VW"),
    ]

    for col, label in dist_metrics:
        for sent in ["positive", "negative"]:
            vals = df[df["sentiment"] == sent][col].dropna()
            if vals.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(vals, bins=15, color=colors[sent], edgecolor="black", alpha=0.7)
            ax.axvline(vals.mean(), color="black", linewidth=1.5, linestyle="--",
                       label=f"mean = {vals.mean():.4f}")
            ax.axvline(vals.median(), color="orange", linewidth=1.5, linestyle=":",
                       label=f"median = {vals.median():.4f}")
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"Distribution of {label} — {sent.capitalize()} Sentiment\n(n={len(vals)})",
                          fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            out_path = OUTPUT_DIR / "03_distributions" / "by_metric" / f"dist_{col}_{sent}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()

    # --- Event study: unbiased + raw volatility without 2 outlier documents ---
    if not edf.empty:
        pre_window = edf[(edf["relative_day"] >= -10) & (edf["relative_day"] <= -7)]
        pre_window = pre_window.dropna(subset=["unbiased_volatility"])
        pre_window_neg = pre_window[pre_window["sentiment"] == "negative"]

        # Average per document, then find the 2 worst
        pre_doc_means = pre_window_neg.groupby(["ticker", "year"])["unbiased_volatility"].mean()
        outlier_docs = pre_doc_means.nsmallest(2)
        outliers_pre_window = [(t, y) for t, y in outlier_docs.index]
       

        # Identify the 3 outlier documents on negative side (lowest unbiased_volatility)
        neg_df = df[df["sentiment"] == "negative"]
        outlier_rows_neg = neg_df.nsmallest(3, "unbiased_volatility")
        outlier_pairs_neg = list(zip(outlier_rows_neg["ticker"], outlier_rows_neg["year"]))

        # Identify the 1 outlier document on positive side (lowest unbiased_volatility)
        pos_df = df[df["sentiment"] == "positive"]
        outlier_rows_pos = pos_df.nsmallest(1, "unbiased_volatility")
        outlier_pairs_pos = list(zip(outlier_rows_pos["ticker"], outlier_rows_pos["year"]))

        all_outlier_pairs = list(set(outlier_pairs_neg + outlier_pairs_pos + outliers_pre_window))
        outlier_desc = ", ".join(f"{t} {y}" for t, y in all_outlier_pairs)
        print(f"Outlier documents: {outlier_desc}, excluding from event study")

        # Remove all event rows belonging to those documents
        mask = pd.Series(False, index=edf.index)
        for t, y in all_outlier_pairs:
            mask |= (edf["ticker"] == t) & (edf["year"] == y)
        edf_clean = edf[~mask]

        # Also build cleaned df for distributions (same outliers removed)
        mask_df = pd.Series(False, index=df.index)
        for t, y in all_outlier_pairs:
            mask_df |= (df["ticker"] == t) & (df["year"] == y)
        df_clean = df[~mask_df]

        # Distribution of unbiased volatility without outliers, positive + negative
        for sent in ["positive", "negative"]:
            vals = df_clean[df_clean["sentiment"] == sent]["unbiased_volatility"].dropna()
            if vals.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(vals, bins=15, color=colors[sent], edgecolor="black", alpha=0.7)
            ax.axvline(vals.mean(), color="black", linewidth=1.5, linestyle="--",
                       label=f"mean = {vals.mean():.4f}")
            ax.axvline(vals.median(), color="orange", linewidth=1.5, linestyle=":",
                       label=f"median = {vals.median():.4f}")
            ax.set_xlabel("Unbiased Volatility", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"Distribution of Unbiased Volatility — {sent.capitalize()} (no outliers)\n"
                          f"(n={len(vals)}, removed: {outlier_desc})",
                          fontsize=11, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            out_path = OUTPUT_DIR / "03_distributions" / "no_outliers" / f"dist_unbiased_volatility_{sent}_no_outlier.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()

        # Event study plots without outliers
        for col, title in [("unbiased_volatility", "Unbiased Volatility"),
                           ("raw_volatility", "Stock Volatility")]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for s in sentiments_order:
                subset = edf_clean[edf_clean["sentiment"] == s]
                if subset.empty:
                    continue
                grouped = subset.groupby("relative_day")[col]
                means = grouped.mean()
                sems = grouped.sem()
                n_docs = subset.drop_duplicates(subset=["ticker", "year"]).shape[0]
                days = means.index.values
                positions = [_DAY_TO_POS[d] for d in days if d in _DAY_TO_POS]
                vals = [means[d] for d in days if d in _DAY_TO_POS]
                lo_ci = [(means[d] - 1.96 * sems[d]) for d in days if d in _DAY_TO_POS]
                hi_ci = [(means[d] + 1.96 * sems[d]) for d in days if d in _DAY_TO_POS]
                ax.plot(positions, vals, color=colors[s], label=f"{s} (n={n_docs} docs)",
                        linewidth=1.5, marker='o', markersize=4)
                ax.fill_between(positions, lo_ci, hi_ci, color=colors[s], alpha=0.15)
            # Count observations per sentiment per day (outlier-cleaned)
            n_per_day = {}
            for s in sentiments_order:
                subset_s = edf_clean[edf_clean["sentiment"] == s]
                if not subset_s.empty and col in subset_s.columns:
                    n_per_day[s] = subset_s.groupby("relative_day")[col].count()
                else:
                    n_per_day[s] = pd.Series(dtype=int)
            tick_labels = []
            for d in EVENT_DAYS:
                parts = [str(d)]
                for s in sentiments_order:
                    parts.append(str(int(n_per_day[s].get(d, 0))))
                tick_labels.append("\n".join(parts))
            ax.axvline(_DAY_TO_POS[0], color="black", linewidth=1, linestyle="--", alpha=0.7, label="Publication day")
            ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
            ax.set_xticks(range(len(EVENT_DAYS)))
            ax.set_xticklabels(tick_labels, fontsize=7)
            sent_labels = " / ".join(sentiments_order)
            ax.set_xlabel(f"Trading days relative to publication\n(n: {sent_labels})", fontsize=10)
            ax.set_title(f"Event Study: {title} ({len(all_outlier_pairs)} outlier docs removed)\n"
                          f"(±{EVENT_HALF_WINDOW} trading days, 95% CI, removed {outlier_desc})",
                          fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(axis="both", alpha=0.3)
            plt.tight_layout()
            out_path = OUTPUT_DIR / "02_event_studies" / "aggregate_no_outliers" / f"event_{col}_no_outlier.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()

    # --- Distribution of unbiased volatility between day -10 and day -7 ---
    if not edf.empty:
        pre_window = edf[(edf["relative_day"] >= -10) & (edf["relative_day"] <= -7)]
        for sent in ["positive", "negative"]:
            subset = pre_window[pre_window["sentiment"] == sent]
            if subset.empty:
                continue
            # Average per document (ticker, year), then plot distribution
            vals = subset.groupby(["ticker", "year"])["unbiased_volatility"].mean().dropna()
            if vals.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(vals, bins=15, color=colors[sent], edgecolor="black", alpha=0.7)
            ax.axvline(vals.mean(), color="black", linewidth=1.5, linestyle="--",
                       label=f"mean = {vals.mean():.4f}")
            ax.axvline(vals.median(), color="orange", linewidth=1.5, linestyle=":",
                       label=f"median = {vals.median():.4f}")
            ax.set_xlabel("Unbiased Volatility (mean over days -10 to -7)", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"Distribution of Unbiased Volatility (day -10 to -7) — {sent.capitalize()}\n"
                          f"(n={len(vals)} documents)",
                          fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            out_path = OUTPUT_DIR / "03_distributions" / "pre_event" / f"dist_unbiased_volatility_pre_event_{sent}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {out_path}")
            plt.show()


if __name__ == "__main__":
    main()
