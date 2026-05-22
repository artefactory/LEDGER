"""
Multinomial Naive Bayes classifier using CEO letter bag-of-words to predict:
  1) Sign of cumulative return at horizon h from earnings date
  2) Sign of residual (return - linear_prediction_from_surprise) at horizon h

Horizons: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90

Features: CountVectorizer on the cleaned CEO letter texts.

Usage:
    uv run python KPI_analysis/predict_target.py
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, str(Path(__file__).resolve().parent))

from FinancialIndicators import GetIndicatorsForPrices
from fetch_filing_returns import fetch_prices
from event_study_earnings import fetch_earnings_dates, find_q4_earnings_date

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

CLEANED_LETTERS_DIR = (
    REPO_ROOT
    / "doc_text_processing"
    / "CEO_word_extraction"
    / "cleaning_extractions"
    / "cleaned"
)
SELECTED_COMPANIES_JSON = (
    REPO_ROOT / "tickers_lists" / "grouped" / "selected" / "companies.json"
)
OUTPUT_DIR = HERE / "output" / "plots" / "predict_target"

HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
MAX_LAG = max(HORIZONS)

# Threshold for neutral class: returns in [-NEUTRAL_THR, +NEUTRAL_THR] are "neutral"
NEUTRAL_THR = 0.01  # 1%
# For surprise: surprise_pct in [-SURPRISE_NEUTRAL_THR, +SURPRISE_NEUTRAL_THR] → neutral
SURPRISE_NEUTRAL_THR = 2.0  # 2% EPS surprise

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2024, 6, 30)


def load_letter_texts() -> dict[tuple[str, int], str]:
    """Load cleaned CEO letter texts, keyed by (ticker, year).

    For a given (ticker, year) with multiple extractions, concatenate them.
    Filename pattern: {EXCHANGE}_{TICKER}_{YEAR}__{NN}_{slug}.md
    """
    texts: dict[tuple[str, int], list[str]] = {}
    for path in sorted(CLEANED_LETTERS_DIR.glob("*.md")):
        name = path.stem
        # Parse: EXCHANGE_TICKER_YEAR__NN_slug
        parts = name.split("__")
        if len(parts) < 2:
            continue
        prefix = parts[0]  # e.g. NYSE_APD_2017
        # Split prefix into exchange, ticker, year
        segments = prefix.split("_")
        if len(segments) < 3:
            continue
        try:
            year = int(segments[-1])
        except ValueError:
            continue
        # Ticker may contain dots (e.g. ELM.L) -> rejoin middle parts
        ticker = "_".join(segments[1:-1])

        text = path.read_text(encoding="utf-8")
        # Strip the header (title + source line + ---) 
        # Find first "---" separator and take text after it
        sep_idx = text.find("\n---\n")
        if sep_idx != -1:
            text = text[sep_idx + 5:]

        key = (ticker, year)
        if key not in texts:
            texts[key] = []
        texts[key].append(text)

    return {k: "\n\n".join(v) for k, v in texts.items()}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load companies ---
    with open(SELECTED_COMPANIES_JSON) as f:
        companies_data = json.load(f)

    all_tickers = set()
    for industry, exchanges in companies_data.items():
        for exchange, companies in exchanges.items():
            for company in companies:
                all_tickers.add(company["ticker"])

    # --- Load CEO letter texts ---
    print("Loading CEO letter texts...")
    letter_texts = load_letter_texts()
    print(f"  Loaded {len(letter_texts)} (ticker, year) letter texts")

    # --- Build dataset: for each (ticker, year) with a letter, get returns at earnings date ---
    print("Fetching earnings dates and prices...")
    earnings_cache: dict[str, pd.DataFrame] = {}
    prices_cache: dict[str, pd.DataFrame | None] = {}

    records = []  # (ticker, year, text, {return_h: val}, surprise)
    for (ticker, year), text in sorted(letter_texts.items()):
        if ticker not in all_tickers:
            continue

        # Fetch earnings date
        if ticker not in earnings_cache:
            earnings_cache[ticker] = fetch_earnings_dates(ticker)
        earnings_df = earnings_cache[ticker]
        if earnings_df.empty:
            continue

        earn_date, surprise, filing_date = find_q4_earnings_date(ticker, year, earnings_df)
        if earn_date is None:
            continue

        # Fetch prices
        if ticker not in prices_cache:
            prices = fetch_prices(ticker, BENCH_START, BENCH_END)
            if prices is not None and not prices.empty:
                prices = GetIndicatorsForPrices(prices, max_lag=MAX_LAG)
            else:
                prices = None
            prices_cache[ticker] = prices
        else:
            prices = prices_cache[ticker]

        if prices is None:
            continue

        # Locate earnings date in price index
        pub_ts = pd.Timestamp(earn_date)
        if pub_ts not in prices.index:
            mask = prices.index >= pub_ts
            if mask.sum() == 0:
                continue
            pub_ts = prices.index[mask][0]

        t0_pos = prices.index.get_loc(pub_ts)

        # Check we have enough data for all horizons
        if t0_pos + MAX_LAG >= len(prices):
            continue

        # Extract returns at each horizon
        returns = {}
        for h in HORIZONS:
            col = f"return_t{h}"
            if col in prices.columns:
                val = prices.loc[pub_ts, col]
                if pd.notna(val):
                    returns[h] = float(val)

        if not returns:
            continue

        records.append({
            "ticker": ticker,
            "year": year,
            "text": text,
            "returns": returns,
            "surprise": float(surprise) if surprise is not None and not pd.isna(surprise) else None,
        })

        print(f"  {ticker} {year}: ok (earn={earn_date.date()}, {len(returns)} horizons)")

    print(f"\nTotal samples with letter + returns: {len(records)}")
    if len(records) < 20:
        print("Not enough samples to train. Exiting.")
        return

    # --- Build feature matrix X (bag of words) ---
    print("\nBuilding CountVectorizer features...")
    corpus = [r["text"] for r in records]
    vectorizer = CountVectorizer(
        min_df=3,
        stop_words="english",
        max_features=5000,
    )
    X = vectorizer.fit_transform(corpus)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Feature matrix: {X.shape}")

    # Save vocabulary to JSON for inspection (word -> total frequency across corpus)
    vocab_path = OUTPUT_DIR / "vocabulary.json"
    total_freq = np.asarray(X.sum(axis=0)).flatten()
    vocab_freq = {word: int(total_freq[idx]) for word, idx in vectorizer.vocabulary_.items()}
    vocab_freq_sorted = dict(sorted(vocab_freq.items(), key=lambda x: -x[1]))
    with open(vocab_path, "w") as f:
        json.dump(vocab_freq_sorted, f, indent=2)
    print(f"  Vocabulary saved to {vocab_path}")

    # --- Target 3 (horizon-independent): predict sign(surprise) from BoW ---
    # 3 classes: negative (< -thr), neutral ([-thr, +thr]), positive (> +thr)
    surprise_result = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    surp_idx = [i for i, r in enumerate(records) if r["surprise"] is not None and not np.isnan(r["surprise"])]
    if len(surp_idx) >= 20:
        X_surp_all = X[surp_idx]
        y_surp_all = np.array([records[i]["surprise"] for i in surp_idx])
        # 3-class labeling: 0=negative, 1=neutral, 2=positive
        y_surp_3 = np.where(y_surp_all > SURPRISE_NEUTRAL_THR, 2,
                            np.where(y_surp_all < -SURPRISE_NEUTRAL_THR, 0, 1))
        class_counts_s = np.bincount(y_surp_3, minlength=3)
        baseline_s = class_counts_s.max() / len(y_surp_3)

        # Need at least 5 in each class for stable CV
        if all(c >= 5 for c in class_counts_s):
            scores_surp = cross_val_score(MultinomialNB(), X_surp_all, y_surp_3, cv=cv, scoring="accuracy")
            y_pred_surp = cross_val_predict(MultinomialNB(), X_surp_all, y_surp_3, cv=cv)
            n_unique_pred_s = len(np.unique(y_pred_surp))
            # Silhouette score (needs >=2 predicted classes)
            sil_s = (silhouette_score(X_surp_all.toarray(), y_pred_surp)
                     if n_unique_pred_s >= 2 else float("nan"))
            surprise_result = {
                "accuracy": scores_surp.mean(),
                "accuracy_std": scores_surp.std(),
                "baseline": baseline_s,
                "n": len(surp_idx),
                "n_classes_predicted": int(n_unique_pred_s),
                "silhouette": sil_s,
                "class_dist": f"neg={class_counts_s[0]} neu={class_counts_s[1]} pos={class_counts_s[2]}",
            }
            print(f"\nTarget 3 — surprise (3-class): acc={surprise_result['accuracy']:.3f} "
                  f"(baseline={baseline_s:.3f}, n={len(surp_idx)}, "
                  f"classes_predicted={n_unique_pred_s}, silhouette={sil_s:.3f})")
            print(f"  Class distribution: {surprise_result['class_dist']}")
        else:
            print(f"\nTarget 3 — surprise (3-class): class too small "
                  f"(neg={class_counts_s[0]} neu={class_counts_s[1]} pos={class_counts_s[2]})")
    else:
        print(f"\nTarget 3 — surprise (3-class): not enough samples ({len(surp_idx)})")

    # --- For each horizon, build targets and evaluate ---
    results_raw = []      # accuracy for raw return sign
    results_residual = []  # accuracy for residual sign

    for h in HORIZONS:
        # Collect returns at this horizon
        y_returns = []
        valid_idx = []
        surprises_h = []
        for i, r in enumerate(records):
            if h in r["returns"]:
                y_returns.append(r["returns"][h])
                valid_idx.append(i)
                surprises_h.append(r["surprise"])

        if len(valid_idx) < 20:
            print(f"  h={h}: only {len(valid_idx)} samples, skipping")
            results_raw.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            continue

        X_h = X[valid_idx]
        y_ret = np.array(y_returns)
        surprises_arr = np.array(surprises_h, dtype=float)

        # --- Target 1: sign of raw return (3-class) ---
        # 0=negative (<-thr), 1=neutral ([-thr,+thr]), 2=positive (>+thr)
        y_3class = np.where(y_ret > NEUTRAL_THR, 2,
                            np.where(y_ret < -NEUTRAL_THR, 0, 1))
        class_counts = np.bincount(y_3class, minlength=3)
        baseline = class_counts.max() / len(y_3class)

        if any(c < 5 for c in class_counts):
            print(f"  h={h}: class too small (neg={class_counts[0]} neu={class_counts[1]} pos={class_counts[2]}), skipping")
            results_raw.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            continue

        scores_raw = cross_val_score(MultinomialNB(), X_h, y_3class, cv=cv, scoring="accuracy")
        y_pred_raw = cross_val_predict(MultinomialNB(), X_h, y_3class, cv=cv)
        n_unique_pred = len(np.unique(y_pred_raw))
        sil_raw = (silhouette_score(X_h.toarray(), y_pred_raw)
                   if n_unique_pred >= 2 else float("nan"))
        acc_raw = scores_raw.mean()
        results_raw.append({
            "horizon": h,
            "accuracy": acc_raw,
            "accuracy_std": scores_raw.std(),
            "baseline": baseline,
            "n": len(valid_idx),
            "n_classes_predicted": int(n_unique_pred),
            "silhouette": sil_raw,
            "class_dist": f"neg={class_counts[0]} neu={class_counts[1]} pos={class_counts[2]}",
        })

        # --- Target 2: sign of residual (return - f(surprise)) ---
        # Only for samples with surprise data
        has_surprise = ~np.isnan(surprises_arr)
        if has_surprise.sum() >= 20:
            idx_surp = np.where(has_surprise)[0]
            X_surp = X_h[idx_surp]
            y_ret_surp = y_ret[idx_surp]
            surp_surp = surprises_arr[idx_surp]

            # Fit linear regression: return = a * surprise + b
            lr = LinearRegression()
            lr.fit(surp_surp.reshape(-1, 1), y_ret_surp)
            y_pred = lr.predict(surp_surp.reshape(-1, 1))
            residuals = y_ret_surp - y_pred

            # 3-class residual: neg / neutral / pos
            # Use residual std as adaptive threshold (or fixed NEUTRAL_THR)
            y_resid_3 = np.where(residuals > NEUTRAL_THR, 2,
                                 np.where(residuals < -NEUTRAL_THR, 0, 1))
            class_counts_r = np.bincount(y_resid_3, minlength=3)
            baseline_r = class_counts_r.max() / len(y_resid_3)

            if all(c >= 5 for c in class_counts_r):
                scores_resid = cross_val_score(
                    MultinomialNB(), X_surp, y_resid_3, cv=cv, scoring="accuracy"
                )
                y_pred_resid = cross_val_predict(MultinomialNB(), X_surp, y_resid_3, cv=cv)
                n_unique_pred_r = len(np.unique(y_pred_resid))
                sil_resid = (silhouette_score(X_surp.toarray(), y_pred_resid)
                             if n_unique_pred_r >= 2 else float("nan"))
                acc_resid = scores_resid.mean()
                results_residual.append({
                    "horizon": h,
                    "accuracy": acc_resid,
                    "accuracy_std": scores_resid.std(),
                    "baseline": baseline_r,
                    "n": len(idx_surp),
                    "n_classes_predicted": int(n_unique_pred_r),
                    "silhouette": sil_resid,
                    "class_dist": f"neg={class_counts_r[0]} neu={class_counts_r[1]} pos={class_counts_r[2]}",
                    "lr_coef": float(lr.coef_[0]),
                    "lr_intercept": float(lr.intercept_),
                })
            else:
                results_residual.append({"horizon": h, "accuracy": np.nan, "n": len(idx_surp), "baseline": baseline_r})
        else:
            results_residual.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})

        sil_str = f", sil={results_raw[-1].get('silhouette', float('nan')):.3f}" if not np.isnan(results_raw[-1].get('silhouette', float('nan'))) else ""
        print(f"  h={h:3d}: raw acc={acc_raw:.3f} (baseline={baseline:.3f}, n={len(valid_idx)}{sil_str})"
              + (f" | resid acc={results_residual[-1].get('accuracy', 'N/A')}" if 'accuracy' in results_residual[-1] and not np.isnan(results_residual[-1].get('accuracy', np.nan)) else ""))

    # --- Save results ---
    df_raw = pd.DataFrame(results_raw)
    df_resid = pd.DataFrame(results_residual)
    df_raw.to_csv(OUTPUT_DIR / "nb_raw_return.csv", index=False)
    df_resid.to_csv(OUTPUT_DIR / "nb_residual_return.csv", index=False)

    # --- Plot: accuracy vs horizon ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: raw return (3-class)
    ax = axes[0]
    mask = df_raw["accuracy"].notna()
    ax.plot(df_raw.loc[mask, "horizon"], df_raw.loc[mask, "accuracy"],
            "o-", color="#2ecc71", linewidth=2, label="MultinomialNB accuracy")
    ax.plot(df_raw.loc[mask, "horizon"], df_raw.loc[mask, "baseline"],
            "--", color="gray", linewidth=1.5, label="Majority baseline")
    if "silhouette" in df_raw.columns:
        mask_sil = df_raw["silhouette"].notna() & mask
        if mask_sil.any():
            ax.plot(df_raw.loc[mask_sil, "horizon"], df_raw.loc[mask_sil, "silhouette"],
                    "x--", color="#f39c12", linewidth=1.5, label="Silhouette score")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Score")
    ax.set_title(f"Return class (neg/neu/pos, thr=\u00b1{NEUTRAL_THR:.0%})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 0.9)

    # Plot 2: residual (3-class)
    ax = axes[1]
    mask = df_resid["accuracy"].notna()
    if mask.any():
        ax.plot(df_resid.loc[mask, "horizon"], df_resid.loc[mask, "accuracy"],
                "s-", color="#e74c3c", linewidth=2, label="MultinomialNB accuracy")
        ax.plot(df_resid.loc[mask, "horizon"], df_resid.loc[mask, "baseline"],
                "--", color="gray", linewidth=1.5, label="Majority baseline")
        if "silhouette" in df_resid.columns:
            mask_sil = df_resid["silhouette"].notna() & mask
            if mask_sil.any():
                ax.plot(df_resid.loc[mask_sil, "horizon"], df_resid.loc[mask_sil, "silhouette"],
                        "x--", color="#f39c12", linewidth=1.5, label="Silhouette score")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Score")
    ax.set_title(f"Residual class (neg/neu/pos, thr=\u00b1{NEUTRAL_THR:.0%})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 0.9)

    # Plot 3: surprise (3-class bar)
    ax = axes[2]
    if surprise_result:
        acc_s = surprise_result["accuracy"]
        bl_s = surprise_result["baseline"]
        sil_s = surprise_result.get("silhouette", float("nan"))
        bars = ax.bar(["NB Accuracy", "Baseline", "Silhouette"],
                      [acc_s, bl_s, sil_s if not np.isnan(sil_s) else 0],
                      color=["#9b59b6", "gray", "#f39c12"], width=0.5)
        ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
        ax.set_ylim(-0.2, 0.9)
        ax.set_title(f"Predicting surprise class (neg/neu/pos)\n(n={surprise_result['n']}, {surprise_result['class_dist']})")
        ax.set_ylabel("Score")
        for bar, val in zip(bars, [acc_s, bl_s, sil_s if not np.isnan(sil_s) else 0]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                    ha="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Predicting surprise class (neg/neu/pos)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Multinomial Naive Bayes: CEO Letter → Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "nb_accuracy_vs_horizon.png", dpi=150)
    plt.close(fig)

    # Save surprise result
    if surprise_result:
        pd.DataFrame([surprise_result]).to_csv(OUTPUT_DIR / "nb_surprise.csv", index=False)

    print(f"\nSaved results to {OUTPUT_DIR}/")
    print(f"  nb_raw_return.csv")
    print(f"  nb_residual_return.csv")
    print(f"  nb_surprise.csv")
    print(f"  nb_accuracy_vs_horizon.png")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Horizon':>8} {'Raw Acc':>8} {'Baseline':>9} {'Sil(raw)':>9} {'Resid Acc':>10} {'Baseline':>9} {'Sil(res)':>9} {'N':>5}")
    print("-" * 100)
    for i, h in enumerate(HORIZONS):
        raw_acc = results_raw[i].get("accuracy", np.nan)
        raw_bl = results_raw[i].get("baseline", np.nan)
        raw_sil = results_raw[i].get("silhouette", np.nan)
        res_acc = results_residual[i].get("accuracy", np.nan)
        res_bl = results_residual[i].get("baseline", np.nan)
        res_sil = results_residual[i].get("silhouette", np.nan)
        n = results_raw[i].get("n", 0)
        raw_sil_s = f"{raw_sil:>9.3f}" if not np.isnan(raw_sil) else f"{'—':>9}"
        res_sil_s = f"{res_sil:>9.3f}" if not np.isnan(res_sil) else f"{'—':>9}"
        print(f"{h:>8d} {raw_acc:>8.3f} {raw_bl:>9.3f} {raw_sil_s} {res_acc:>10.3f} {res_bl:>9.3f} {res_sil_s} {n:>5d}")
    print("-" * 100)
    if surprise_result:
        sil_s_str = f"{surprise_result['silhouette']:.3f}" if not np.isnan(surprise_result.get('silhouette', float('nan'))) else "—"
        print(f"{'surprise':>8} {surprise_result['accuracy']:>8.3f} {surprise_result['baseline']:>9.3f} {sil_s_str:>9} {'—':>10} {'—':>9} {'—':>9} {surprise_result['n']:>5d}")
    print("=" * 100)


if __name__ == "__main__":
    main()
