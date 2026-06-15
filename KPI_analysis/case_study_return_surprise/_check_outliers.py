"""Quick script to identify outlier documents across all summary metrics."""
import json, sys, pandas as pd, numpy as np
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentiment_vs_indicators import load_sentiments, _pub_date_for_fy, _window_mean
from FinancialIndicators import GetIndicatorsForPrices, GetIndustryDataFrame
from fetch_filing_returns import fetch_prices

sentiments = load_sentiments()
BENCH_START, BENCH_END = date(2016, 6, 1), date(2023, 6, 30)
WINDOW = 5
cache = {}
records = []

for i, row in enumerate(sentiments):
    t, y, s = row["ticker"], row["year"], row["sentiment"]
    pub = _pub_date_for_fy(t, y)
    if pub is None:
        continue
    if t not in cache:
        prices = fetch_prices(t, BENCH_START, BENCH_END)
        if prices is None or prices.empty:
            cache[t] = (None, None)
            continue
        prices = GetIndicatorsForPrices(prices)
        ind = GetIndustryDataFrame(t, BENCH_START, BENCH_END)
        cache[t] = (prices, ind)
    prices, ind = cache[t]
    if prices is None:
        continue

    ind_aligned = ind.reindex(prices.index)
    unbiased_vol = prices["Volatility"] - ind_aligned["volatility"]
    unbiased_volume = prices["Volume_ATS"] - ind_aligned["volumes"]
    raw_vol = prices["Volatility"]
    raw_volume = prices["Volume_ATS"]
    raw_price = prices["Close"]
    unbiased_volume_vw = prices["Volume_ATS"] - ind_aligned["volumes_vw"] if "volumes_vw" in ind_aligned.columns else pd.Series(dtype=float)

    v = _window_mean(unbiased_vol, pub, WINDOW)
    vol = _window_mean(unbiased_volume, pub, WINDOW)
    rv = _window_mean(raw_vol, pub, WINDOW)
    rvol = _window_mean(raw_volume, pub, WINDOW)
    rp = _window_mean(raw_price, pub, WINDOW)
    vol_vw = _window_mean(unbiased_volume_vw, pub, WINDOW)

    pub_ts = pd.Timestamp(pub)
    cum_ret_1 = prices.loc[pub_ts, "return_t1"] if pub_ts in prices.index and "return_t1" in prices.columns else None
    cum_ret_5 = prices.loc[pub_ts, "return_t5"] if pub_ts in prices.index and "return_t5" in prices.columns else None
    cum_ret_1_unbiased = (cum_ret_1 - ind_aligned.loc[pub_ts, "return_t1"]) if (cum_ret_1 is not None and pub_ts in ind_aligned.index and "return_t1" in ind_aligned.columns) else None
    cum_ret_5_unbiased = (cum_ret_5 - ind_aligned.loc[pub_ts, "return_t5"]) if (cum_ret_5 is not None and pub_ts in ind_aligned.index and "return_t5" in ind_aligned.columns) else None
    cum_ret_1_unbiased_vw = (cum_ret_1 - ind_aligned.loc[pub_ts, "return_t1_vw"]) if (cum_ret_1 is not None and pub_ts in ind_aligned.index and "return_t1_vw" in ind_aligned.columns) else None
    cum_ret_5_unbiased_vw = (cum_ret_5 - ind_aligned.loc[pub_ts, "return_t5_vw"]) if (cum_ret_5 is not None and pub_ts in ind_aligned.index and "return_t5_vw" in ind_aligned.columns) else None

    if v is None and vol is None:
        continue

    records.append({
        "ticker": t, "year": y, "sentiment": s, "industry": row["industry"],
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
    })

df = pd.DataFrame(records)
print(f"Total documents: {len(df)}")
print(f"Sentiments: {df['sentiment'].value_counts().to_dict()}\n")

# --- IQR-based outlier detection for each metric ---
metrics = [
    "unbiased_volatility", "unbiased_volume", "raw_volatility", "raw_volume",
    "raw_price", "volume_unbiased_vw",
    "cum_return_1d", "cum_return_5d",
    "cum_return_1d_unbiased", "cum_return_5d_unbiased",
    "cum_return_1d_unbiased_vw", "cum_return_5d_unbiased_vw",
]

all_outliers = set()

for metric in metrics:
    vals = df[metric].dropna()
    if vals.empty:
        continue
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    mask = (df[metric] < lo) | (df[metric] > hi)
    outliers = df[mask].dropna(subset=[metric])
    if outliers.empty:
        continue
    print(f"{'='*60}")
    print(f"METRIC: {metric}")
    print(f"  Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f}")
    print(f"  Bounds: [{lo:.6f}, {hi:.6f}]")
    print(f"  Outliers: {len(outliers)}")
    print(outliers[["ticker", "year", "sentiment", "industry", metric]]
          .sort_values(metric).to_string(index=False))
    print()
    for _, r in outliers.iterrows():
        all_outliers.add((r["ticker"], int(r["year"])))

print(f"\n{'='*60}")
print(f"SUMMARY: {len(all_outliers)} unique outlier documents across all metrics")
for t, y in sorted(all_outliers):
    row_data = df[(df["ticker"] == t) & (df["year"] == y)].iloc[0]
    print(f"  {t} {y} ({row_data['sentiment']}, {row_data['industry']})")
