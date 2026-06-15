"""
Verify selection bias: tickers with a sentiment label vs those without.

For each industry, compare the average indicators of:
  - tickers WITH a sentiment label (positive or negative)
  - tickers WITHOUT a sentiment label (in the industry but absent from sentiments.json)

Indicators: returns, volatility, volume ATS.
Expected result: unlabelled tickers are more volatile → industry avg > labelled avg.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_filing_returns import fetch_prices
from fetch_kpis import tickers_from_selected
from FinancialIndicators import GetIndicatorsForPrices

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

INDICATORS = ["return_t1", "return_t5", "Volatility", "Volume"]


def _ticker_means(ticker: str) -> dict[str, float] | None:
    """Return mean of each indicator for a ticker, or None if no usable data."""
    prices = fetch_prices(ticker, BENCH_START, BENCH_END)
    if prices is None or prices.empty:
        return None
    prices = GetIndicatorsForPrices(prices)
    result = {}
    for col in INDICATORS:
        v = prices[col].mean()
        result[col] = v if not pd.isna(v) else np.nan
    return result


def main():
    # Load sentiment labels
    with open(SENTIMENTS_JSON) as f:
        data = json.load(f)

    labelled: dict[str, set[str]] = {}
    for industry, tickers in data.items():
        labelled[industry] = set()
        for ticker, years in tickers.items():
            if any(v is not None for v in years.values()):
                labelled[industry].add(ticker)

    # All tickers per industry
    all_entries = tickers_from_selected()
    industry_tickers: dict[str, set[str]] = {}
    for e in all_entries:
        industry_tickers.setdefault(e["industry"], set()).add(e["ticker"])

    rows = []
    for industry in sorted(industry_tickers):
        all_tickers = industry_tickers[industry]
        lab = labelled.get(industry, set()) & all_tickers
        unlab = all_tickers - lab

        print(f"\n=== {industry} ===")
        print(f"  Total: {len(all_tickers)}, labelled: {len(lab)}, unlabelled: {len(unlab)}")

        def group_means(tickers: set[str], label: str) -> dict[str, float]:
            all_vals: dict[str, list[float]] = {c: [] for c in INDICATORS}
            for t in sorted(tickers):
                m = _ticker_means(t)
                if m is None:
                    print(f"    {t}: no prices")
                    continue
                skip = [c for c in INDICATORS if pd.isna(m[c])]
                if skip:
                    print(f"    {t}: not enough data for {skip} (skipped)")
                for c in INDICATORS:
                    if not pd.isna(m[c]):
                        all_vals[c].append(m[c])
                vals_str = ", ".join(f"{c}={m[c]:.6f}" for c in INDICATORS if not pd.isna(m[c]))
                if vals_str:
                    print(f"    {t}: {vals_str}")
            result = {}
            for c in INDICATORS:
                result[c] = np.mean(all_vals[c]) if all_vals[c] else np.nan
            return result

        print(f"  --- Labelled ---")
        lab_means = group_means(lab, "labelled")
        print(f"  --- Unlabelled ---")
        unlab_means = group_means(unlab, "unlabelled")

        row = {
            "industry": industry,
            "n_labelled": len(lab),
            "n_unlabelled": len(unlab),
        }
        print()
        for c in INDICATORS:
            lv = lab_means[c]
            uv = unlab_means[c]
            row[f"{c}_labelled"] = lv
            row[f"{c}_unlabelled"] = uv
            if not pd.isna(lv) and not pd.isna(uv) and lv != 0:
                ratio = uv / lv
                row[f"{c}_ratio"] = ratio
                print(f"  >> {c}: labelled={lv:.6f}, unlabelled={uv:.6f}, ratio={ratio:.2f}x")
            else:
                row[f"{c}_ratio"] = np.nan
                print(f"  >> {c}: labelled={lv}, unlabelled={uv}")
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        print("\n\n========== SUMMARY ==========")
        print(df.to_string(index=False, float_format="%.6f"))
        for c in INDICATORS:
            col = f"{c}_ratio"
            if col in df.columns:
                print(f"\nMean ratio {c} (unlabelled / labelled): {df[col].mean():.2f}x")
        out = HERE / "output" / "selection_bias_check.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
