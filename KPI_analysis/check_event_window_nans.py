"""Check for NaN gaps within the ±10 trading-day event window around publication dates.

The concern: _event_window_values does series.dropna() BEFORE positional indexing,
so if there are NaN values within the window, positional offsets no longer correspond
to actual trading-day offsets. Day -5 could really be day -7 if 2 NaN days were dropped.

This script reports how many publications are affected and which ones.
"""

import json
import sys
from datetime import date
from pathlib import Path

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


def check_nans_in_window(series_raw: pd.Series, target_date: pd.Timestamp, half_window: int) -> dict:
    """Check for NaN in the event window WITHOUT dropping them first.

    Returns dict with:
      - has_nan: bool
      - nan_positions: list of relative day positions that are NaN
      - total_days: how many trading days exist in the window
    """
    # Find t0 position on the FULL index (with NaN values present)
    mask = series_raw.index >= target_date
    if mask.sum() == 0:
        return {"has_nan": False, "nan_positions": [], "total_days": 0, "missing_days": []}

    t0_idx = series_raw.index[mask][0]
    t0_pos = series_raw.index.get_loc(t0_idx)

    nan_positions = []
    missing_days = []
    total_days = 0

    for d in range(-half_window, half_window + 1):
        pos = t0_pos + d
        if 0 <= pos < len(series_raw):
            total_days += 1
            if pd.isna(series_raw.iloc[pos]):
                nan_positions.append(d)
        else:
            missing_days.append(d)

    return {
        "has_nan": len(nan_positions) > 0,
        "nan_positions": nan_positions,
        "total_days": total_days,
        "missing_days": missing_days,
    }


def check_nans_relative_series(series: pd.Series, half_window: int) -> dict:
    """Check NaN in a series indexed by relative day (int)."""
    nan_positions = []
    missing_days = []
    total_days = 0
    for d in range(-half_window, half_window + 1):
        if d in series.index:
            total_days += 1
            if pd.isna(series[d]):
                nan_positions.append(d)
        else:
            missing_days.append(d)
    return {
        "has_nan": len(nan_positions) > 0,
        "nan_positions": nan_positions,
        "total_days": total_days,
        "missing_days": missing_days,
    }


def main():
    with open(SENTIMENTS_JSON) as f:
        data = json.load(f)

    rows = []
    for industry, tickers in data.items():
        for ticker, years in tickers.items():
            for year, sentiment in years.items():
                if sentiment is not None:
                    rows.append({
                        "industry": industry,
                        "ticker": ticker,
                        "year": int(year),
                        "sentiment": sentiment,
                    })

    print(f"Loaded {len(rows)} (ticker, year, sentiment) entries")

    processed_tickers: dict[str, tuple] = {}
    affected = []
    total_checked = 0

    indicators = ["returns", "Volatility", "Volume_ATS"]

    for i, row in enumerate(rows):
        ticker = row["ticker"]
        year = row["year"]
        sentiment = row["sentiment"]

        pub_date = _pub_date_for_fy(ticker, year)
        if pub_date is None:
            continue

        if ticker not in processed_tickers:
            prices = fetch_prices(ticker, BENCH_START, BENCH_END)
            if prices is None or prices.empty:
                processed_tickers[ticker] = (None, None)
                continue
            prices = GetIndicatorsForPrices(prices)
            industry_df = GetIndustryDataFrame(ticker, BENCH_START, BENCH_END)
            processed_tickers[ticker] = (prices, industry_df)
        else:
            prices, industry_df = processed_tickers[ticker]

        if prices is None:
            continue

        total_checked += 1

        # Check each indicator for NaN in the window
        ind_aligned = industry_df.reindex(prices.index)
        series_to_check = {
            "unbiased_return": prices["returns"] - ind_aligned["returns"],
            "unbiased_volatility": prices["Volatility"] - ind_aligned["volatility"],
            "unbiased_volume": prices["Volume_ATS"] - ind_aligned["volumes"],
            "raw_return": prices["returns"],
            "raw_volatility": prices["Volatility"],
            "raw_volume": prices["Volume_ATS"],
        }

        # Cumulative return (ticker level)
        # Build a series where index=relative day, value=return_t{d} evaluated at t0
        pub_date_ts = pd.Timestamp(pub_date)
        # Find actual t0 (first trading day >= pub_date)
        mask_t0 = prices.index >= pub_date_ts
        if mask_t0.sum() > 0:
            t0_date = prices.index[mask_t0][0]
            cum_ret_vals = {}
            for d in range(-EVENT_HALF_WINDOW, EVENT_HALF_WINDOW + 1):
                col = f"return_t{d}"
                if col in prices.columns:
                    cum_ret_vals[d] = prices.loc[t0_date, col]
                else:
                    cum_ret_vals[d] = float("nan")
            # Build series indexed by relative day for check
            cum_ret_series = pd.Series(cum_ret_vals)
            series_to_check["cum_return"] = cum_ret_series

            # Cumulative return industry VW
            cum_ret_vw_vals = {}
            for d in range(-EVENT_HALF_WINDOW, EVENT_HALF_WINDOW + 1):
                col = f"return_t{d}_vw"
                if col in ind_aligned.columns and t0_date in ind_aligned.index:
                    cum_ret_vw_vals[d] = ind_aligned.loc[t0_date, col]
                else:
                    cum_ret_vw_vals[d] = float("nan")
            cum_ret_vw_series = pd.Series(cum_ret_vw_vals)
            # cum_return_unbiased_vw = cum_return - cum_return_vw
            cum_ret_ub_vw_series = cum_ret_series - cum_ret_vw_series
            series_to_check["cum_return_unbiased_vw"] = cum_ret_ub_vw_series

            # return_unbiased_vw (daily return - industry return VW)
            if "returns_vw" in ind_aligned.columns:
                series_to_check["return_unbiased_vw"] = prices["returns"] - ind_aligned["returns_vw"]
            # volume_unbiased_vw
            if "volumes_vw" in ind_aligned.columns:
                series_to_check["volume_unbiased_vw"] = prices["Volume_ATS"] - ind_aligned["volumes_vw"]

        pub_issues = {}
        # Series indexed by relative day (int) use a different checker
        relative_day_series = {"cum_return", "cum_return_unbiased_vw"}
        for name, series in series_to_check.items():
            if name in relative_day_series:
                result = check_nans_relative_series(series, EVENT_HALF_WINDOW)
            else:
                result = check_nans_in_window(series, pub_date, EVENT_HALF_WINDOW)
            if result["has_nan"] or result["missing_days"]:
                pub_issues[name] = result

        if pub_issues:
            affected.append({
                "ticker": ticker,
                "year": year,
                "sentiment": sentiment,
                "industry": row["industry"],
                "pub_date": str(pub_date.date()),
                "issues": pub_issues,
            })

    # --- Report ---
    print(f"\n{'='*70}")
    print(f"Total publications checked: {total_checked}")
    print(f"Publications with NaN in ±{EVENT_HALF_WINDOW} day window: {len(affected)}")
    print(f"{'='*70}\n")

    # Save JSON report
    out_path = HERE / "output" / "event_window_nans.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "total_checked": total_checked,
            "affected_count": len(affected),
            "affected": affected,
        }, f, indent=2)
    print(f"Saved {out_path}")

    if not affected:
        print("No NaN issues found.")
        return

    # Summary by indicator
    indicator_counts = {}
    for pub in affected:
        for ind_name, info in pub["issues"].items():
            if info["has_nan"]:
                indicator_counts[ind_name] = indicator_counts.get(ind_name, 0) + 1
    print("Affected publications per indicator:")
    for ind, count in sorted(indicator_counts.items(), key=lambda x: -x[1]):
        print(f"  {ind}: {count} publications")

    print(f"\n{'─'*70}")
    print("Detailed list of affected publications:")
    print(f"{'─'*70}")
    for pub in affected:
        print(f"\n  {pub['ticker']} {pub['year']} ({pub['sentiment']}) — pub: {pub['pub_date']}")
        print(f"    Industry: {pub['industry']}")
        for ind_name, info in pub["issues"].items():
            if info["has_nan"]:
                print(f"    {ind_name}: NaN at relative days {info['nan_positions']}")
            if info["missing_days"]:
                print(f"    {ind_name}: missing days (out of bounds) {info['missing_days']}")


if __name__ == "__main__":
    main()
