"""
Naive Bayes classifiers (MultinomialNB + GaussianNB + RQ-MNB + PCA-LogReg)
using sentence embeddings of CEO letters to predict:
  1) Return class (neg/neutral/pos) at horizon h from earnings date
  2) Residual class (return - f(surprise)) at horizon h
  3) Surprise class (neg/neutral/pos EPS surprise)

Encoder is configurable via --encoder (minilm, roberta, eurobert, or any
sentence-transformers model path). Output is written to a per-encoder folder.

Usage:
    uv run python KPI_analysis/predict_target_embed_encoder.py --encoder minilm
    uv run python KPI_analysis/predict_target_embed_encoder.py --encoder roberta
    uv run python KPI_analysis/predict_target_embed_encoder.py --encoder eurobert
    uv run python KPI_analysis/predict_target_embed_encoder.py --encoder sentence-transformers/all-mpnet-base-v2
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, silhouette_score
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler, label_binarize

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

# --- Encoder presets ---
ENCODER_PRESETS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
    "roberta": "sentence-transformers/all-roberta-large-v1",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "eurobert": "EuroBERT/EuroBERT-2.1B",
}

HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
MAX_LAG = max(HORIZONS)

# Threshold for neutral class
NEUTRAL_THR = 0.01  # 1% return
SURPRISE_NEUTRAL_THR = 2.0  # 2% EPS surprise
SURPRISE_INLIER_Q_LOW = 0.02
SURPRISE_INLIER_Q_HIGH = 0.98

# PCA + Logistic Regression params
PCA_DIM = 50

# Residual Quantization params
RQ_N_LEVELS = 8       # number of quantization levels
RQ_CODEBOOK_SIZE = 32  # centroids per level (K)

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2024, 6, 30)


def residual_quantize_fit(X: np.ndarray, n_levels: int, codebook_size: int,
                          random_state: int = 42) -> list:
    """Fit Residual Quantization codebooks (list of KMeans models)."""
    codebooks = []
    residual = X.copy()
    for level in range(n_levels):
        km = KMeans(n_clusters=codebook_size, random_state=random_state,
                    n_init=3, max_iter=100)
        km.fit(residual)
        codebooks.append(km)
        residual = residual - km.cluster_centers_[km.labels_]
    return codebooks


def residual_quantize_transform(X: np.ndarray, codebooks: list) -> np.ndarray:
    """Transform embeddings to bag-of-codes via RQ.

    Returns a (n_samples, n_levels * codebook_size) matrix of code counts
    (one-hot per level, concatenated).
    """
    n_levels = len(codebooks)
    codebook_size = codebooks[0].n_clusters
    n = X.shape[0]
    codes = np.zeros((n, n_levels * codebook_size), dtype=np.float64)
    residual = X.copy()
    for level, km in enumerate(codebooks):
        labels = km.predict(residual)
        offset = level * codebook_size
        for i, lbl in enumerate(labels):
            codes[i, offset + lbl] = 1.0
        residual = residual - km.cluster_centers_[labels]
    return codes


def pr_auc_ovr_scorer(estimator, X, y):
    """PR AUC (One-vs-Rest, macro) scorer for multiclass cross_val_score."""
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    y_proba = estimator.predict_proba(X)
    y_bin = label_binarize(y, classes=classes)
    if y_bin.shape[1] == 1:
        return average_precision_score(y_bin.ravel(), y_proba[:, 1])
    return average_precision_score(y_bin, y_proba, average="macro")


def load_letter_texts() -> dict[tuple[str, int], str]:
    """Load cleaned CEO letter texts, keyed by (ticker, year)."""
    texts: dict[tuple[str, int], list[str]] = {}
    for path in sorted(CLEANED_LETTERS_DIR.glob("*.md")):
        name = path.stem
        parts = name.split("__")
        if len(parts) < 2:
            continue
        prefix = parts[0]
        segments = prefix.split("_")
        if len(segments) < 3:
            continue
        try:
            year = int(segments[-1])
        except ValueError:
            continue
        ticker = "_".join(segments[1:-1])

        text = path.read_text(encoding="utf-8")
        sep_idx = text.find("\n---\n")
        if sep_idx != -1:
            text = text[sep_idx + 5:]

        key = (ticker, year)
        if key not in texts:
            texts[key] = []
        texts[key].append(text)

    return {k: "\n\n".join(v) for k, v in texts.items()}


def encode_corpus(corpus: list[str], model_name: str) -> tuple[np.ndarray, int]:
    """Encode corpus with the given model. Returns (X_raw, embed_dim).

    For sentence-transformers models: uses SentenceTransformer.encode()
    For EuroBERT: uses transformers AutoModel with mean pooling + L2 norm.
    """
    if "eurobert" in model_name.lower():
        return _encode_eurobert(corpus, model_name)
    else:
        return _encode_sentence_transformer(corpus, model_name)


def _encode_sentence_transformer(corpus: list[str], model_name: str) -> tuple[np.ndarray, int]:
    """Encode with any sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    print(f"\nEncoding with {model_name}...")
    model = SentenceTransformer(model_name)
    X_raw = model.encode(corpus, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    X_raw = np.array(X_raw, dtype=np.float64)
    embed_dim = X_raw.shape[1]
    print(f"  Embedding shape: {X_raw.shape}")
    return X_raw, embed_dim


def _encode_eurobert(corpus: list[str], model_name: str) -> tuple[np.ndarray, int]:
    """Encode with EuroBERT (transformers AutoModel + mean pooling + L2 norm)."""
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope_init(config, device=None):
            dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            if hasattr(config, "rope_theta"):
                base = config.rope_theta
            elif hasattr(config, "rope_parameters") and "rope_theta" in config.rope_parameters:
                base = config.rope_parameters["rope_theta"]
            else:
                base = getattr(config, "default_theta", 10000.0)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

    from transformers import AutoTokenizer, AutoModel

    print(f"\nEncoding with {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lm_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    batch_size = 4
    embeddings = []
    n_batches = (len(corpus) + batch_size - 1) // batch_size
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = lm_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embeddings.append(emb.cpu().numpy())
        batch_num = i // batch_size + 1
        if batch_num % 10 == 0 or batch_num == n_batches:
            print(f"  batch {batch_num}/{n_batches}")

    X_raw = np.vstack(embeddings).astype(np.float64)
    np.nan_to_num(X_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # L2 normalize
    norms = np.linalg.norm(X_raw, axis=1, keepdims=True).clip(min=1e-9)
    X_raw = X_raw / norms
    embed_dim = X_raw.shape[1]
    print(f"  Embedding shape: {X_raw.shape}")
    return X_raw, embed_dim


def model_slug(model_name: str) -> str:
    """Derive a filesystem-safe slug from the model name."""
    # e.g. "sentence-transformers/all-MiniLM-L6-v2" -> "all-MiniLM-L6-v2"
    # e.g. "EuroBERT/EuroBERT-2.1B" -> "EuroBERT-2.1B"
    name = model_name.split("/")[-1]
    return name.replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Predict targets from CEO letter embeddings")
    parser.add_argument("--encoder", default="roberta",
                        help="Encoder preset (minilm, roberta, eurobert) or full model path")
    args = parser.parse_args()

    # Resolve encoder
    EMBED_MODEL = ENCODER_PRESETS.get(args.encoder, args.encoder)
    slug = model_slug(EMBED_MODEL)
    EMBED_DIM = None  # will be set after encoding

    OUTPUT_DIR = HERE / "output" / "plots" / f"predict_target_embed_{slug}"
    INLIERS_DIR = OUTPUT_DIR / "surprise_inliers"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INLIERS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Encoder: {EMBED_MODEL}")
    print(f"Output:  {OUTPUT_DIR}")

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

    records = []
    for (ticker, year), text in sorted(letter_texts.items()):
        if ticker not in all_tickers:
            continue

        if ticker not in earnings_cache:
            earnings_cache[ticker] = fetch_earnings_dates(ticker)
        earnings_df = earnings_cache[ticker]
        if earnings_df.empty:
            continue

        earn_date, surprise, filing_date = find_q4_earnings_date(ticker, year, earnings_df)
        if earn_date is None:
            continue

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

        pub_ts = pd.Timestamp(earn_date)
        if pub_ts not in prices.index:
            mask = prices.index >= pub_ts
            if mask.sum() == 0:
                continue
            pub_ts = prices.index[mask][0]

        t0_pos = prices.index.get_loc(pub_ts)
        if t0_pos + MAX_LAG >= len(prices):
            continue

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

    # --- Encode texts ---
    corpus = [r["text"] for r in records]
    X_raw, EMBED_DIM = encode_corpus(corpus, EMBED_MODEL)

    # X_raw: for GaussianNB (works on any real-valued features)
    # X_mnb: scaled to [0, 1] for MultinomialNB
    scaler = MinMaxScaler()
    X_mnb = scaler.fit_transform(X_raw)
    print(f"  Scaled copy to [0, 1] for MultinomialNB")

    # --- BOW baseline (TF-IDF + MultinomialNB) ---
    tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    X_bow = tfidf.fit_transform(corpus)
    print(f"  BOW baseline: TF-IDF shape {X_bow.shape}")

    # Save embeddings for reuse
    np.save(OUTPUT_DIR / "embeddings.npy", X_raw)
    print(f"  Embeddings saved to {OUTPUT_DIR / 'embeddings.npy'}")

    # --- PCA ---
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X_pca = pca.fit_transform(X_raw)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"\nPCA: {EMBED_DIM}d -> {PCA_DIM}d (explained variance = {explained_var:.1%})")

    # --- Residual Quantization ---
    print(f"\nResidual Quantization: {RQ_N_LEVELS} levels, K={RQ_CODEBOOK_SIZE} "
          f"-> {RQ_N_LEVELS * RQ_CODEBOOK_SIZE}-dim bag-of-codes")
    rq_codebooks = residual_quantize_fit(X_raw, RQ_N_LEVELS, RQ_CODEBOOK_SIZE)
    X_rq = residual_quantize_transform(X_raw, rq_codebooks)
    print(f"  RQ feature shape: {X_rq.shape}")

    # LogReg classifier (used on PCA features)
    logreg = LogisticRegression(max_iter=1000, random_state=42)

    # --- Target 3 (horizon-independent): predict surprise class ---
    surprise_result_mnb = {}
    surprise_result_gnb = {}
    surprise_result_rq = {}
    surprise_result_lr = {}
    surprise_result_bow = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    surp_idx = [i for i, r in enumerate(records) if r["surprise"] is not None and not np.isnan(r["surprise"])]
    if len(surp_idx) >= 20:
        X_surp_mnb = X_mnb[surp_idx]
        X_surp_raw = X_raw[surp_idx]
        X_surp_rq = X_rq[surp_idx]
        X_surp_pca = X_pca[surp_idx]
        X_surp_bow = X_bow[surp_idx]
        y_surp_all = np.array([records[i]["surprise"] for i in surp_idx])
        y_surp_3 = np.where(y_surp_all > SURPRISE_NEUTRAL_THR, 2,
                            np.where(y_surp_all < -SURPRISE_NEUTRAL_THR, 0, 1))
        class_counts_s = np.bincount(y_surp_3, minlength=3)
        baseline_s = class_counts_s.max() / len(y_surp_3)

        if all(c >= 5 for c in class_counts_s):
            # MultinomialNB
            scores_surp_mnb = cross_val_score(MultinomialNB(), X_surp_mnb, y_surp_3, cv=cv, scoring="accuracy")
            y_pred_surp_mnb = cross_val_predict(MultinomialNB(), X_surp_mnb, y_surp_3, cv=cv)
            n_unique_mnb = len(np.unique(y_pred_surp_mnb))
            sil_mnb = silhouette_score(X_surp_mnb, y_pred_surp_mnb) if n_unique_mnb >= 2 else float("nan")
            try:
                auc_surp_mnb = cross_val_score(MultinomialNB(), X_surp_mnb, y_surp_3, cv=cv, scoring="roc_auc_ovr").mean()
            except ValueError:
                auc_surp_mnb = float("nan")
            try:
                pr_auc_surp_mnb = cross_val_score(MultinomialNB(), X_surp_mnb, y_surp_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
            except ValueError:
                pr_auc_surp_mnb = float("nan")
            surprise_result_mnb = {
                "accuracy": scores_surp_mnb.mean(),
                "accuracy_std": scores_surp_mnb.std(),
                "baseline": baseline_s,
                "n": len(surp_idx),
                "n_classes_predicted": int(n_unique_mnb),
                "silhouette": sil_mnb,
                "roc_auc": auc_surp_mnb,
                "pr_auc": pr_auc_surp_mnb,
                "class_dist": f"neg={class_counts_s[0]} neu={class_counts_s[1]} pos={class_counts_s[2]}",
            }
            # GaussianNB
            scores_surp_gnb = cross_val_score(GaussianNB(), X_surp_raw, y_surp_3, cv=cv, scoring="accuracy")
            y_pred_surp_gnb = cross_val_predict(GaussianNB(), X_surp_raw, y_surp_3, cv=cv)
            n_unique_gnb = len(np.unique(y_pred_surp_gnb))
            sil_gnb = silhouette_score(X_surp_raw, y_pred_surp_gnb) if n_unique_gnb >= 2 else float("nan")
            try:
                auc_surp_gnb = cross_val_score(GaussianNB(), X_surp_raw, y_surp_3, cv=cv, scoring="roc_auc_ovr").mean()
            except ValueError:
                auc_surp_gnb = float("nan")
            try:
                pr_auc_surp_gnb = cross_val_score(GaussianNB(), X_surp_raw, y_surp_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
            except ValueError:
                pr_auc_surp_gnb = float("nan")
            surprise_result_gnb = {
                "accuracy": scores_surp_gnb.mean(),
                "accuracy_std": scores_surp_gnb.std(),
                "baseline": baseline_s,
                "n": len(surp_idx),
                "n_classes_predicted": int(n_unique_gnb),
                "silhouette": sil_gnb,
                "roc_auc": auc_surp_gnb,
                "pr_auc": pr_auc_surp_gnb,
                "class_dist": f"neg={class_counts_s[0]} neu={class_counts_s[1]} pos={class_counts_s[2]}",
            }
            # RQ + MNB
            scores_surp_rq = cross_val_score(MultinomialNB(), X_surp_rq, y_surp_3, cv=cv, scoring="accuracy")
            y_pred_surp_rq = cross_val_predict(MultinomialNB(), X_surp_rq, y_surp_3, cv=cv)
            n_unique_rq = len(np.unique(y_pred_surp_rq))
            sil_rq = silhouette_score(X_surp_rq, y_pred_surp_rq) if n_unique_rq >= 2 else float("nan")
            try:
                auc_surp_rq = cross_val_score(MultinomialNB(), X_surp_rq, y_surp_3, cv=cv, scoring="roc_auc_ovr").mean()
            except ValueError:
                auc_surp_rq = float("nan")
            try:
                pr_auc_surp_rq = cross_val_score(MultinomialNB(), X_surp_rq, y_surp_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
            except ValueError:
                pr_auc_surp_rq = float("nan")
            surprise_result_rq = {
                "accuracy": scores_surp_rq.mean(),
                "accuracy_std": scores_surp_rq.std(),
                "baseline": baseline_s,
                "n": len(surp_idx),
                "n_classes_predicted": int(n_unique_rq),
                "silhouette": sil_rq,
                "roc_auc": auc_surp_rq,
                "pr_auc": pr_auc_surp_rq,
                "class_dist": f"neg={class_counts_s[0]} neu={class_counts_s[1]} pos={class_counts_s[2]}",
            }
            # PCA + LogReg
            scores_surp_lr = cross_val_score(logreg, X_surp_pca, y_surp_3, cv=cv, scoring="accuracy")
            y_pred_surp_lr = cross_val_predict(logreg, X_surp_pca, y_surp_3, cv=cv)
            n_unique_lr = len(np.unique(y_pred_surp_lr))
            sil_lr = silhouette_score(X_surp_pca, y_pred_surp_lr) if n_unique_lr >= 2 else float("nan")
            try:
                auc_surp_lr = cross_val_score(logreg, X_surp_pca, y_surp_3, cv=cv, scoring="roc_auc_ovr").mean()
            except ValueError:
                auc_surp_lr = float("nan")
            try:
                pr_auc_surp_lr = cross_val_score(logreg, X_surp_pca, y_surp_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
            except ValueError:
                pr_auc_surp_lr = float("nan")
            surprise_result_lr = {
                "accuracy": scores_surp_lr.mean(),
                "accuracy_std": scores_surp_lr.std(),
                "baseline": baseline_s,
                "n": len(surp_idx),
                "n_classes_predicted": int(n_unique_lr),
                "silhouette": sil_lr,
                "roc_auc": auc_surp_lr,
                "pr_auc": pr_auc_surp_lr,
                "class_dist": f"neg={class_counts_s[0]} neu={class_counts_s[1]} pos={class_counts_s[2]}",
            }
            # BOW baseline (TF-IDF + MNB)
            scores_surp_bow = cross_val_score(MultinomialNB(), X_surp_bow, y_surp_3, cv=cv, scoring="accuracy")
            try:
                auc_surp_bow = cross_val_score(MultinomialNB(), X_surp_bow, y_surp_3, cv=cv, scoring="roc_auc_ovr").mean()
            except ValueError:
                auc_surp_bow = float("nan")
            try:
                pr_auc_surp_bow = cross_val_score(MultinomialNB(), X_surp_bow, y_surp_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
            except ValueError:
                pr_auc_surp_bow = float("nan")
            surprise_result_bow = {
                "accuracy": scores_surp_bow.mean(),
                "accuracy_std": scores_surp_bow.std(),
                "baseline": baseline_s,
                "n": len(surp_idx),
                "roc_auc": auc_surp_bow,
                "pr_auc": pr_auc_surp_bow,
                "class_dist": f"neg={class_counts_s[0]} neu={class_counts_s[1]} pos={class_counts_s[2]}",
            }
            print(f"\nTarget 3 — surprise (3-class):")
            print(f"  BOW baseline:  acc={surprise_result_bow['accuracy']:.3f} (auc={auc_surp_bow:.3f}, pr_auc={pr_auc_surp_bow:.3f})")
            print(f"  MultinomialNB: acc={surprise_result_mnb['accuracy']:.3f} (auc={auc_surp_mnb:.3f}, pr_auc={pr_auc_surp_mnb:.3f})")
            print(f"  GaussianNB:    acc={surprise_result_gnb['accuracy']:.3f} (auc={auc_surp_gnb:.3f}, pr_auc={pr_auc_surp_gnb:.3f})")
            print(f"  RQ+MNB:        acc={surprise_result_rq['accuracy']:.3f} (auc={auc_surp_rq:.3f}, pr_auc={pr_auc_surp_rq:.3f})")
            print(f"  PCA+LogReg:    acc={surprise_result_lr['accuracy']:.3f} (auc={auc_surp_lr:.3f}, pr_auc={pr_auc_surp_lr:.3f})")
            print(f"  majority_baseline={baseline_s:.3f}, n={len(surp_idx)}")
            print(f"  Class distribution: neg={class_counts_s[0]} neu={class_counts_s[1]} pos={class_counts_s[2]}")
        else:
            print(f"\nTarget 3 — surprise (3-class): class too small "
                  f"(neg={class_counts_s[0]} neu={class_counts_s[1]} pos={class_counts_s[2]})")
    else:
        print(f"\nTarget 3 — surprise (3-class): not enough samples ({len(surp_idx)})")

    # --- For each horizon, build targets and evaluate ---
    results_raw_mnb = []
    results_raw_gnb = []
    results_raw_rq = []
    results_raw_lr = []
    results_raw_bow = []
    results_residual_mnb = []
    results_residual_gnb = []
    results_residual_rq = []
    results_residual_lr = []
    results_residual_bow = []
    results_residual_inliers_mnb = []
    results_residual_inliers_gnb = []
    results_residual_inliers_rq = []
    results_residual_inliers_lr = []

    for h in HORIZONS:
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
            results_raw_mnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_raw_gnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_raw_rq.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_raw_lr.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_raw_bow.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual_mnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual_gnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual_rq.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual_lr.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual_bow.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual_inliers_mnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual_inliers_gnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual_inliers_rq.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            results_residual_inliers_lr.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx)})
            continue

        X_h_mnb = X_mnb[valid_idx]
        X_h_raw = X_raw[valid_idx]
        X_h_rq = X_rq[valid_idx]
        X_h_pca = X_pca[valid_idx]
        X_h_bow = X_bow[valid_idx]
        y_ret = np.array(y_returns)
        surprises_arr = np.array(surprises_h, dtype=float)

        # --- Target 1: return class (3-class) ---
        y_3class = np.where(y_ret > NEUTRAL_THR, 2,
                            np.where(y_ret < -NEUTRAL_THR, 0, 1))
        class_counts = np.bincount(y_3class, minlength=3)
        baseline = class_counts.max() / len(y_3class)

        if any(c < 5 for c in class_counts):
            print(f"  h={h}: class too small (neg={class_counts[0]} neu={class_counts[1]} pos={class_counts[2]}), skipping")
            results_raw_mnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_raw_gnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_raw_rq.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_raw_lr.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_raw_bow.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual_mnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual_gnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual_rq.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual_lr.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual_bow.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual_inliers_mnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual_inliers_gnb.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual_inliers_rq.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            results_residual_inliers_lr.append({"horizon": h, "accuracy": np.nan, "n": len(valid_idx), "baseline": baseline})
            continue

        # MultinomialNB on scaled features
        scores_mnb = cross_val_score(MultinomialNB(), X_h_mnb, y_3class, cv=cv, scoring="accuracy")
        y_pred_mnb = cross_val_predict(MultinomialNB(), X_h_mnb, y_3class, cv=cv)
        n_unique_mnb = len(np.unique(y_pred_mnb))
        sil_mnb = silhouette_score(X_h_mnb, y_pred_mnb) if n_unique_mnb >= 2 else float("nan")
        try:
            auc_mnb = cross_val_score(MultinomialNB(), X_h_mnb, y_3class, cv=cv, scoring="roc_auc_ovr").mean()
        except ValueError:
            auc_mnb = float("nan")
        try:
            pr_auc_mnb = cross_val_score(MultinomialNB(), X_h_mnb, y_3class, cv=cv, scoring=pr_auc_ovr_scorer).mean()
        except ValueError:
            pr_auc_mnb = float("nan")
        acc_mnb = scores_mnb.mean()
        results_raw_mnb.append({
            "horizon": h, "accuracy": acc_mnb, "accuracy_std": scores_mnb.std(),
            "baseline": baseline, "n": len(valid_idx),
            "n_classes_predicted": int(n_unique_mnb), "silhouette": sil_mnb,
            "roc_auc": auc_mnb, "pr_auc": pr_auc_mnb,
            "class_dist": f"neg={class_counts[0]} neu={class_counts[1]} pos={class_counts[2]}",
        })

        # GaussianNB on raw embeddings
        scores_gnb = cross_val_score(GaussianNB(), X_h_raw, y_3class, cv=cv, scoring="accuracy")
        y_pred_gnb = cross_val_predict(GaussianNB(), X_h_raw, y_3class, cv=cv)
        n_unique_gnb = len(np.unique(y_pred_gnb))
        sil_gnb = silhouette_score(X_h_raw, y_pred_gnb) if n_unique_gnb >= 2 else float("nan")
        try:
            auc_gnb = cross_val_score(GaussianNB(), X_h_raw, y_3class, cv=cv, scoring="roc_auc_ovr").mean()
        except ValueError:
            auc_gnb = float("nan")
        try:
            pr_auc_gnb = cross_val_score(GaussianNB(), X_h_raw, y_3class, cv=cv, scoring=pr_auc_ovr_scorer).mean()
        except ValueError:
            pr_auc_gnb = float("nan")
        acc_gnb = scores_gnb.mean()
        results_raw_gnb.append({
            "horizon": h, "accuracy": acc_gnb, "accuracy_std": scores_gnb.std(),
            "baseline": baseline, "n": len(valid_idx),
            "n_classes_predicted": int(n_unique_gnb), "silhouette": sil_gnb,
            "roc_auc": auc_gnb, "pr_auc": pr_auc_gnb,
            "class_dist": f"neg={class_counts[0]} neu={class_counts[1]} pos={class_counts[2]}",
        })

        # RQ + MNB on bag-of-codes
        scores_rq = cross_val_score(MultinomialNB(), X_h_rq, y_3class, cv=cv, scoring="accuracy")
        y_pred_rq = cross_val_predict(MultinomialNB(), X_h_rq, y_3class, cv=cv)
        n_unique_rq = len(np.unique(y_pred_rq))
        sil_rq = silhouette_score(X_h_rq, y_pred_rq) if n_unique_rq >= 2 else float("nan")
        try:
            auc_rq = cross_val_score(MultinomialNB(), X_h_rq, y_3class, cv=cv, scoring="roc_auc_ovr").mean()
        except ValueError:
            auc_rq = float("nan")
        try:
            pr_auc_rq = cross_val_score(MultinomialNB(), X_h_rq, y_3class, cv=cv, scoring=pr_auc_ovr_scorer).mean()
        except ValueError:
            pr_auc_rq = float("nan")
        acc_rq = scores_rq.mean()
        results_raw_rq.append({
            "horizon": h, "accuracy": acc_rq, "accuracy_std": scores_rq.std(),
            "baseline": baseline, "n": len(valid_idx),
            "n_classes_predicted": int(n_unique_rq), "silhouette": sil_rq,
            "roc_auc": auc_rq, "pr_auc": pr_auc_rq,
            "class_dist": f"neg={class_counts[0]} neu={class_counts[1]} pos={class_counts[2]}",
        })

        # PCA + LogReg
        scores_lr = cross_val_score(logreg, X_h_pca, y_3class, cv=cv, scoring="accuracy")
        y_pred_lr = cross_val_predict(logreg, X_h_pca, y_3class, cv=cv)
        n_unique_lr = len(np.unique(y_pred_lr))
        sil_lr = silhouette_score(X_h_pca, y_pred_lr) if n_unique_lr >= 2 else float("nan")
        try:
            auc_lr = cross_val_score(logreg, X_h_pca, y_3class, cv=cv, scoring="roc_auc_ovr").mean()
        except ValueError:
            auc_lr = float("nan")
        try:
            pr_auc_lr = cross_val_score(logreg, X_h_pca, y_3class, cv=cv, scoring=pr_auc_ovr_scorer).mean()
        except ValueError:
            pr_auc_lr = float("nan")
        acc_lr = scores_lr.mean()
        results_raw_lr.append({
            "horizon": h, "accuracy": acc_lr, "accuracy_std": scores_lr.std(),
            "baseline": baseline, "n": len(valid_idx),
            "n_classes_predicted": int(n_unique_lr), "silhouette": sil_lr,
            "roc_auc": auc_lr, "pr_auc": pr_auc_lr,
            "class_dist": f"neg={class_counts[0]} neu={class_counts[1]} pos={class_counts[2]}",
        })

        # BOW baseline (TF-IDF + MNB)
        scores_bow = cross_val_score(MultinomialNB(), X_h_bow, y_3class, cv=cv, scoring="accuracy")
        try:
            auc_bow = cross_val_score(MultinomialNB(), X_h_bow, y_3class, cv=cv, scoring="roc_auc_ovr").mean()
        except ValueError:
            auc_bow = float("nan")
        try:
            pr_auc_bow = cross_val_score(MultinomialNB(), X_h_bow, y_3class, cv=cv, scoring=pr_auc_ovr_scorer).mean()
        except ValueError:
            pr_auc_bow = float("nan")
        acc_bow = scores_bow.mean()
        results_raw_bow.append({
            "horizon": h, "accuracy": acc_bow, "accuracy_std": scores_bow.std(),
            "baseline": baseline, "n": len(valid_idx),
            "roc_auc": auc_bow, "pr_auc": pr_auc_bow,
            "class_dist": f"neg={class_counts[0]} neu={class_counts[1]} pos={class_counts[2]}",
        })

        # --- Target 2: residual class (3-class) ---
        has_surprise = ~np.isnan(surprises_arr)
        if has_surprise.sum() >= 20:
            idx_surp = np.where(has_surprise)[0]
            X_surp_mnb = X_h_mnb[idx_surp]
            X_surp_raw = X_h_raw[idx_surp]
            X_surp_rq = X_h_rq[idx_surp]
            X_surp_pca = X_h_pca[idx_surp]
            X_surp_bow = X_h_bow[idx_surp]
            y_ret_surp = y_ret[idx_surp]
            surp_surp = surprises_arr[idx_surp]

            lr = LinearRegression()
            lr.fit(surp_surp.reshape(-1, 1), y_ret_surp)
            y_pred_lr = lr.predict(surp_surp.reshape(-1, 1))
            residuals = y_ret_surp - y_pred_lr

            y_resid_3 = np.where(residuals > NEUTRAL_THR, 2,
                                 np.where(residuals < -NEUTRAL_THR, 0, 1))
            class_counts_r = np.bincount(y_resid_3, minlength=3)
            baseline_r = class_counts_r.max() / len(y_resid_3)

            if all(c >= 5 for c in class_counts_r):
                # MultinomialNB
                sc_resid_mnb = cross_val_score(MultinomialNB(), X_surp_mnb, y_resid_3, cv=cv, scoring="accuracy")
                yp_resid_mnb = cross_val_predict(MultinomialNB(), X_surp_mnb, y_resid_3, cv=cv)
                nu_mnb = len(np.unique(yp_resid_mnb))
                sil_r_mnb = silhouette_score(X_surp_mnb, yp_resid_mnb) if nu_mnb >= 2 else float("nan")
                try:
                    auc_r_mnb = cross_val_score(MultinomialNB(), X_surp_mnb, y_resid_3, cv=cv, scoring="roc_auc_ovr").mean()
                except ValueError:
                    auc_r_mnb = float("nan")
                try:
                    pr_auc_r_mnb = cross_val_score(MultinomialNB(), X_surp_mnb, y_resid_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
                except ValueError:
                    pr_auc_r_mnb = float("nan")
                results_residual_mnb.append({
                    "horizon": h, "accuracy": sc_resid_mnb.mean(), "accuracy_std": sc_resid_mnb.std(),
                    "baseline": baseline_r, "n": len(idx_surp),
                    "n_classes_predicted": int(nu_mnb), "silhouette": sil_r_mnb,
                    "roc_auc": auc_r_mnb, "pr_auc": pr_auc_r_mnb,
                    "class_dist": f"neg={class_counts_r[0]} neu={class_counts_r[1]} pos={class_counts_r[2]}",
                    "lr_coef": float(lr.coef_[0]), "lr_intercept": float(lr.intercept_),
                })
                # GaussianNB
                sc_resid_gnb = cross_val_score(GaussianNB(), X_surp_raw, y_resid_3, cv=cv, scoring="accuracy")
                yp_resid_gnb = cross_val_predict(GaussianNB(), X_surp_raw, y_resid_3, cv=cv)
                nu_gnb = len(np.unique(yp_resid_gnb))
                sil_r_gnb = silhouette_score(X_surp_raw, yp_resid_gnb) if nu_gnb >= 2 else float("nan")
                try:
                    auc_r_gnb = cross_val_score(GaussianNB(), X_surp_raw, y_resid_3, cv=cv, scoring="roc_auc_ovr").mean()
                except ValueError:
                    auc_r_gnb = float("nan")
                try:
                    pr_auc_r_gnb = cross_val_score(GaussianNB(), X_surp_raw, y_resid_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
                except ValueError:
                    pr_auc_r_gnb = float("nan")
                results_residual_gnb.append({
                    "horizon": h, "accuracy": sc_resid_gnb.mean(), "accuracy_std": sc_resid_gnb.std(),
                    "baseline": baseline_r, "n": len(idx_surp),
                    "n_classes_predicted": int(nu_gnb), "silhouette": sil_r_gnb,
                    "roc_auc": auc_r_gnb, "pr_auc": pr_auc_r_gnb,
                    "class_dist": f"neg={class_counts_r[0]} neu={class_counts_r[1]} pos={class_counts_r[2]}",
                    "lr_coef": float(lr.coef_[0]), "lr_intercept": float(lr.intercept_),
                })
                # RQ + MNB
                sc_resid_rq = cross_val_score(MultinomialNB(), X_surp_rq, y_resid_3, cv=cv, scoring="accuracy")
                yp_resid_rq = cross_val_predict(MultinomialNB(), X_surp_rq, y_resid_3, cv=cv)
                nu_rq = len(np.unique(yp_resid_rq))
                sil_r_rq = silhouette_score(X_surp_rq, yp_resid_rq) if nu_rq >= 2 else float("nan")
                try:
                    auc_r_rq = cross_val_score(MultinomialNB(), X_surp_rq, y_resid_3, cv=cv, scoring="roc_auc_ovr").mean()
                except ValueError:
                    auc_r_rq = float("nan")
                try:
                    pr_auc_r_rq = cross_val_score(MultinomialNB(), X_surp_rq, y_resid_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
                except ValueError:
                    pr_auc_r_rq = float("nan")
                results_residual_rq.append({
                    "horizon": h, "accuracy": sc_resid_rq.mean(), "accuracy_std": sc_resid_rq.std(),
                    "baseline": baseline_r, "n": len(idx_surp),
                    "n_classes_predicted": int(nu_rq), "silhouette": sil_r_rq,
                    "roc_auc": auc_r_rq, "pr_auc": pr_auc_r_rq,
                    "class_dist": f"neg={class_counts_r[0]} neu={class_counts_r[1]} pos={class_counts_r[2]}",
                    "lr_coef": float(lr.coef_[0]), "lr_intercept": float(lr.intercept_),
                })
                # PCA + LogReg
                sc_resid_lr = cross_val_score(logreg, X_surp_pca, y_resid_3, cv=cv, scoring="accuracy")
                yp_resid_lr = cross_val_predict(logreg, X_surp_pca, y_resid_3, cv=cv)
                nu_lr = len(np.unique(yp_resid_lr))
                sil_r_lr = silhouette_score(X_surp_pca, yp_resid_lr) if nu_lr >= 2 else float("nan")
                try:
                    auc_r_lr = cross_val_score(logreg, X_surp_pca, y_resid_3, cv=cv, scoring="roc_auc_ovr").mean()
                except ValueError:
                    auc_r_lr = float("nan")
                try:
                    pr_auc_r_lr = cross_val_score(logreg, X_surp_pca, y_resid_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
                except ValueError:
                    pr_auc_r_lr = float("nan")
                results_residual_lr.append({
                    "horizon": h, "accuracy": sc_resid_lr.mean(), "accuracy_std": sc_resid_lr.std(),
                    "baseline": baseline_r, "n": len(idx_surp),
                    "n_classes_predicted": int(nu_lr), "silhouette": sil_r_lr,
                    "roc_auc": auc_r_lr, "pr_auc": pr_auc_r_lr,
                    "class_dist": f"neg={class_counts_r[0]} neu={class_counts_r[1]} pos={class_counts_r[2]}",
                    "lr_coef": float(lr.coef_[0]), "lr_intercept": float(lr.intercept_),
                })
                # BOW baseline (TF-IDF + MNB) on residual
                sc_resid_bow = cross_val_score(MultinomialNB(), X_surp_bow, y_resid_3, cv=cv, scoring="accuracy")
                try:
                    auc_r_bow = cross_val_score(MultinomialNB(), X_surp_bow, y_resid_3, cv=cv, scoring="roc_auc_ovr").mean()
                except ValueError:
                    auc_r_bow = float("nan")
                try:
                    pr_auc_r_bow = cross_val_score(MultinomialNB(), X_surp_bow, y_resid_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
                except ValueError:
                    pr_auc_r_bow = float("nan")
                results_residual_bow.append({
                    "horizon": h, "accuracy": sc_resid_bow.mean(), "accuracy_std": sc_resid_bow.std(),
                    "baseline": baseline_r, "n": len(idx_surp),
                    "roc_auc": auc_r_bow, "pr_auc": pr_auc_r_bow,
                    "class_dist": f"neg={class_counts_r[0]} neu={class_counts_r[1]} pos={class_counts_r[2]}",
                    "lr_coef": float(lr.coef_[0]), "lr_intercept": float(lr.intercept_),
                })
            else:
                results_residual_mnb.append({"horizon": h, "accuracy": np.nan, "n": len(idx_surp), "baseline": baseline_r})
                results_residual_gnb.append({"horizon": h, "accuracy": np.nan, "n": len(idx_surp), "baseline": baseline_r})
                results_residual_rq.append({"horizon": h, "accuracy": np.nan, "n": len(idx_surp), "baseline": baseline_r})
                results_residual_lr.append({"horizon": h, "accuracy": np.nan, "n": len(idx_surp), "baseline": baseline_r})
                results_residual_bow.append({"horizon": h, "accuracy": np.nan, "n": len(idx_surp), "baseline": baseline_r})

            # --- Residual class with surprise inliers only (q2-q98) ---
            q_lo = np.quantile(surp_surp, SURPRISE_INLIER_Q_LOW)
            q_hi = np.quantile(surp_surp, SURPRISE_INLIER_Q_HIGH)
            inlier_mask = (surp_surp >= q_lo) & (surp_surp <= q_hi)
            n_inliers = int(inlier_mask.sum())
            n_outliers = int(len(surp_surp) - n_inliers)

            if n_inliers >= 20:
                X_in_mnb = X_surp_mnb[inlier_mask]
                X_in_raw = X_surp_raw[inlier_mask]
                X_in_rq = X_surp_rq[inlier_mask]
                X_in_pca = X_surp_pca[inlier_mask]
                y_in = y_ret_surp[inlier_mask]
                surp_in = surp_surp[inlier_mask]

                lr_in = LinearRegression()
                lr_in.fit(surp_in.reshape(-1, 1), y_in)
                y_pred_in = lr_in.predict(surp_in.reshape(-1, 1))
                residuals_in = y_in - y_pred_in

                y_resid_in_3 = np.where(residuals_in > NEUTRAL_THR, 2,
                                        np.where(residuals_in < -NEUTRAL_THR, 0, 1))
                class_counts_in = np.bincount(y_resid_in_3, minlength=3)
                baseline_in = class_counts_in.max() / len(y_resid_in_3)

                if all(c >= 5 for c in class_counts_in):
                    # MNB
                    sc_in_mnb = cross_val_score(MultinomialNB(), X_in_mnb, y_resid_in_3, cv=cv, scoring="accuracy")
                    yp_in_mnb = cross_val_predict(MultinomialNB(), X_in_mnb, y_resid_in_3, cv=cv)
                    nu_in_mnb = len(np.unique(yp_in_mnb))
                    sil_in_mnb = silhouette_score(X_in_mnb, yp_in_mnb) if nu_in_mnb >= 2 else float("nan")
                    try:
                        auc_in_mnb = cross_val_score(MultinomialNB(), X_in_mnb, y_resid_in_3, cv=cv, scoring="roc_auc_ovr").mean()
                    except ValueError:
                        auc_in_mnb = float("nan")
                    try:
                        pr_auc_in_mnb = cross_val_score(MultinomialNB(), X_in_mnb, y_resid_in_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
                    except ValueError:
                        pr_auc_in_mnb = float("nan")
                    results_residual_inliers_mnb.append({
                        "horizon": h, "accuracy": sc_in_mnb.mean(), "accuracy_std": sc_in_mnb.std(),
                        "baseline": baseline_in, "n": n_inliers,
                        "n_outliers_removed": n_outliers,
                        "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi),
                        "n_classes_predicted": int(nu_in_mnb), "silhouette": sil_in_mnb,
                        "roc_auc": auc_in_mnb, "pr_auc": pr_auc_in_mnb,
                        "class_dist": f"neg={class_counts_in[0]} neu={class_counts_in[1]} pos={class_counts_in[2]}",
                        "lr_coef": float(lr_in.coef_[0]), "lr_intercept": float(lr_in.intercept_),
                    })

                    # GNB
                    sc_in_gnb = cross_val_score(GaussianNB(), X_in_raw, y_resid_in_3, cv=cv, scoring="accuracy")
                    yp_in_gnb = cross_val_predict(GaussianNB(), X_in_raw, y_resid_in_3, cv=cv)
                    nu_in_gnb = len(np.unique(yp_in_gnb))
                    sil_in_gnb = silhouette_score(X_in_raw, yp_in_gnb) if nu_in_gnb >= 2 else float("nan")
                    try:
                        auc_in_gnb = cross_val_score(GaussianNB(), X_in_raw, y_resid_in_3, cv=cv, scoring="roc_auc_ovr").mean()
                    except ValueError:
                        auc_in_gnb = float("nan")
                    try:
                        pr_auc_in_gnb = cross_val_score(GaussianNB(), X_in_raw, y_resid_in_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
                    except ValueError:
                        pr_auc_in_gnb = float("nan")
                    results_residual_inliers_gnb.append({
                        "horizon": h, "accuracy": sc_in_gnb.mean(), "accuracy_std": sc_in_gnb.std(),
                        "baseline": baseline_in, "n": n_inliers,
                        "n_outliers_removed": n_outliers,
                        "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi),
                        "n_classes_predicted": int(nu_in_gnb), "silhouette": sil_in_gnb,
                        "roc_auc": auc_in_gnb, "pr_auc": pr_auc_in_gnb,
                        "class_dist": f"neg={class_counts_in[0]} neu={class_counts_in[1]} pos={class_counts_in[2]}",
                        "lr_coef": float(lr_in.coef_[0]), "lr_intercept": float(lr_in.intercept_),
                    })

                    # RQ+MNB
                    sc_in_rq = cross_val_score(MultinomialNB(), X_in_rq, y_resid_in_3, cv=cv, scoring="accuracy")
                    yp_in_rq = cross_val_predict(MultinomialNB(), X_in_rq, y_resid_in_3, cv=cv)
                    nu_in_rq = len(np.unique(yp_in_rq))
                    sil_in_rq = silhouette_score(X_in_rq, yp_in_rq) if nu_in_rq >= 2 else float("nan")
                    try:
                        auc_in_rq = cross_val_score(MultinomialNB(), X_in_rq, y_resid_in_3, cv=cv, scoring="roc_auc_ovr").mean()
                    except ValueError:
                        auc_in_rq = float("nan")
                    try:
                        pr_auc_in_rq = cross_val_score(MultinomialNB(), X_in_rq, y_resid_in_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
                    except ValueError:
                        pr_auc_in_rq = float("nan")
                    results_residual_inliers_rq.append({
                        "horizon": h, "accuracy": sc_in_rq.mean(), "accuracy_std": sc_in_rq.std(),
                        "baseline": baseline_in, "n": n_inliers,
                        "n_outliers_removed": n_outliers,
                        "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi),
                        "n_classes_predicted": int(nu_in_rq), "silhouette": sil_in_rq,
                        "roc_auc": auc_in_rq, "pr_auc": pr_auc_in_rq,
                        "class_dist": f"neg={class_counts_in[0]} neu={class_counts_in[1]} pos={class_counts_in[2]}",
                        "lr_coef": float(lr_in.coef_[0]), "lr_intercept": float(lr_in.intercept_),
                    })

                    # PCA+LR
                    sc_in_lr = cross_val_score(logreg, X_in_pca, y_resid_in_3, cv=cv, scoring="accuracy")
                    yp_in_lr = cross_val_predict(logreg, X_in_pca, y_resid_in_3, cv=cv)
                    nu_in_lr = len(np.unique(yp_in_lr))
                    sil_in_lr = silhouette_score(X_in_pca, yp_in_lr) if nu_in_lr >= 2 else float("nan")
                    try:
                        auc_in_lr = cross_val_score(logreg, X_in_pca, y_resid_in_3, cv=cv, scoring="roc_auc_ovr").mean()
                    except ValueError:
                        auc_in_lr = float("nan")
                    try:
                        pr_auc_in_lr = cross_val_score(logreg, X_in_pca, y_resid_in_3, cv=cv, scoring=pr_auc_ovr_scorer).mean()
                    except ValueError:
                        pr_auc_in_lr = float("nan")
                    results_residual_inliers_lr.append({
                        "horizon": h, "accuracy": sc_in_lr.mean(), "accuracy_std": sc_in_lr.std(),
                        "baseline": baseline_in, "n": n_inliers,
                        "n_outliers_removed": n_outliers,
                        "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi),
                        "n_classes_predicted": int(nu_in_lr), "silhouette": sil_in_lr,
                        "roc_auc": auc_in_lr, "pr_auc": pr_auc_in_lr,
                        "class_dist": f"neg={class_counts_in[0]} neu={class_counts_in[1]} pos={class_counts_in[2]}",
                        "lr_coef": float(lr_in.coef_[0]), "lr_intercept": float(lr_in.intercept_),
                    })
                else:
                    results_residual_inliers_mnb.append({"horizon": h, "accuracy": np.nan, "n": n_inliers, "baseline": baseline_in, "n_outliers_removed": n_outliers, "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi)})
                    results_residual_inliers_gnb.append({"horizon": h, "accuracy": np.nan, "n": n_inliers, "baseline": baseline_in, "n_outliers_removed": n_outliers, "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi)})
                    results_residual_inliers_rq.append({"horizon": h, "accuracy": np.nan, "n": n_inliers, "baseline": baseline_in, "n_outliers_removed": n_outliers, "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi)})
                    results_residual_inliers_lr.append({"horizon": h, "accuracy": np.nan, "n": n_inliers, "baseline": baseline_in, "n_outliers_removed": n_outliers, "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi)})
            else:
                results_residual_inliers_mnb.append({"horizon": h, "accuracy": np.nan, "n": n_inliers, "n_outliers_removed": n_outliers, "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi)})
                results_residual_inliers_gnb.append({"horizon": h, "accuracy": np.nan, "n": n_inliers, "n_outliers_removed": n_outliers, "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi)})
                results_residual_inliers_rq.append({"horizon": h, "accuracy": np.nan, "n": n_inliers, "n_outliers_removed": n_outliers, "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi)})
                results_residual_inliers_lr.append({"horizon": h, "accuracy": np.nan, "n": n_inliers, "n_outliers_removed": n_outliers, "surprise_q_low": float(q_lo), "surprise_q_high": float(q_hi)})
        else:
            results_residual_mnb.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})
            results_residual_gnb.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})
            results_residual_rq.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})
            results_residual_lr.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})
            results_residual_bow.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})
            results_residual_inliers_mnb.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})
            results_residual_inliers_gnb.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})
            results_residual_inliers_rq.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})
            results_residual_inliers_lr.append({"horizon": h, "accuracy": np.nan, "n": int(has_surprise.sum())})

        print(f"  h={h:3d}: MNB={acc_mnb:.3f} GNB={acc_gnb:.3f} RQ={acc_rq:.3f} LR={acc_lr:.3f} BOW={acc_bow:.3f} (baseline={baseline:.3f}, n={len(valid_idx)})")

    # --- Save results ---
    df_raw_mnb = pd.DataFrame(results_raw_mnb)
    df_raw_gnb = pd.DataFrame(results_raw_gnb)
    df_raw_rq = pd.DataFrame(results_raw_rq)
    df_raw_lr = pd.DataFrame(results_raw_lr)
    df_raw_bow = pd.DataFrame(results_raw_bow)
    df_resid_mnb = pd.DataFrame(results_residual_mnb)
    df_resid_gnb = pd.DataFrame(results_residual_gnb)
    df_resid_rq = pd.DataFrame(results_residual_rq)
    df_resid_lr = pd.DataFrame(results_residual_lr)
    df_resid_bow = pd.DataFrame(results_residual_bow)
    df_resid_in_mnb = pd.DataFrame(results_residual_inliers_mnb)
    df_resid_in_gnb = pd.DataFrame(results_residual_inliers_gnb)
    df_resid_in_rq = pd.DataFrame(results_residual_inliers_rq)
    df_resid_in_lr = pd.DataFrame(results_residual_inliers_lr)
    df_raw_mnb.to_csv(OUTPUT_DIR / "mnb_raw_return.csv", index=False)
    df_raw_gnb.to_csv(OUTPUT_DIR / "gnb_raw_return.csv", index=False)
    df_raw_rq.to_csv(OUTPUT_DIR / "rq_mnb_raw_return.csv", index=False)
    df_raw_lr.to_csv(OUTPUT_DIR / "lr_pca_raw_return.csv", index=False)
    df_raw_bow.to_csv(OUTPUT_DIR / "bow_raw_return.csv", index=False)
    df_resid_mnb.to_csv(OUTPUT_DIR / "mnb_residual_return.csv", index=False)
    df_resid_gnb.to_csv(OUTPUT_DIR / "gnb_residual_return.csv", index=False)
    df_resid_rq.to_csv(OUTPUT_DIR / "rq_mnb_residual_return.csv", index=False)
    df_resid_lr.to_csv(OUTPUT_DIR / "lr_pca_residual_return.csv", index=False)
    df_resid_bow.to_csv(OUTPUT_DIR / "bow_residual_return.csv", index=False)
    df_resid_in_mnb.to_csv(INLIERS_DIR / "mnb_residual_return_inliers.csv", index=False)
    df_resid_in_gnb.to_csv(INLIERS_DIR / "gnb_residual_return_inliers.csv", index=False)
    df_resid_in_rq.to_csv(INLIERS_DIR / "rq_mnb_residual_return_inliers.csv", index=False)
    df_resid_in_lr.to_csv(INLIERS_DIR / "lr_pca_residual_return_inliers.csv", index=False)

    # --- Plot 1: raw return accuracy vs horizon ---
    fig, ax = plt.subplots(figsize=(8, 5))
    mask_mnb = df_raw_mnb["accuracy"].notna()
    mask_gnb = df_raw_gnb["accuracy"].notna()
    mask_rq = df_raw_rq["accuracy"].notna()
    mask_lr = df_raw_lr["accuracy"].notna()
    mask_bow = df_raw_bow["accuracy"].notna()
    xpos = np.arange(len(df_raw_mnb))
    ax.plot(xpos[mask_mnb], df_raw_mnb.loc[mask_mnb, "accuracy"].values,
            "o-", color="#2ecc71", linewidth=2, label="MultinomialNB")
    ax.plot(xpos[mask_gnb], df_raw_gnb.loc[mask_gnb, "accuracy"].values,
            "^-", color="#3498db", linewidth=2, label="GaussianNB")
    ax.plot(xpos[mask_rq], df_raw_rq.loc[mask_rq, "accuracy"].values,
            "v-", color="#e67e22", linewidth=2, label="RQ+MNB")
    ax.plot(xpos[mask_lr], df_raw_lr.loc[mask_lr, "accuracy"].values,
            "P-", color="#c0392b", linewidth=2, label=f"PCA({PCA_DIM})+LogReg")
    ax.plot(xpos[mask_bow], df_raw_bow.loc[mask_bow, "accuracy"].values,
            "x--", color="#7f8c8d", linewidth=2, label="BOW (TF-IDF)")
    ax.plot(xpos[mask_mnb], df_raw_mnb.loc[mask_mnb, "baseline"].values,
            "--", color="gray", linewidth=1.5, label="Majority baseline")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=7, rotation=45)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Return class (neg/neu/pos, thr=\u00b1{NEUTRAL_THR:.0%})\n{slug} ({EMBED_DIM}d)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_raw_return.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: residual accuracy vs horizon ---
    fig, ax = plt.subplots(figsize=(8, 5))
    mask_mnb = df_resid_mnb["accuracy"].notna()
    mask_gnb = df_resid_gnb["accuracy"].notna()
    mask_rq_r = df_resid_rq["accuracy"].notna()
    mask_lr_r = df_resid_lr["accuracy"].notna()
    mask_bow_r = df_resid_bow["accuracy"].notna()
    xpos_r = np.arange(len(df_resid_mnb))
    if mask_mnb.any():
        ax.plot(xpos_r[mask_mnb], df_resid_mnb.loc[mask_mnb, "accuracy"].values,
                "s-", color="#e74c3c", linewidth=2, label="MultinomialNB")
    if mask_gnb.any():
        ax.plot(xpos_r[mask_gnb], df_resid_gnb.loc[mask_gnb, "accuracy"].values,
                "D-", color="#9b59b6", linewidth=2, label="GaussianNB")
    if mask_rq_r.any():
        ax.plot(xpos_r[mask_rq_r], df_resid_rq.loc[mask_rq_r, "accuracy"].values,
                "v-", color="#e67e22", linewidth=2, label="RQ+MNB")
    if mask_lr_r.any():
        ax.plot(xpos_r[mask_lr_r], df_resid_lr.loc[mask_lr_r, "accuracy"].values,
                "P-", color="#c0392b", linewidth=2, label=f"PCA({PCA_DIM})+LogReg")
    if mask_bow_r.any():
        ax.plot(xpos_r[mask_bow_r], df_resid_bow.loc[mask_bow_r, "accuracy"].values,
                "x--", color="#7f8c8d", linewidth=2, label="BOW (TF-IDF)")
    if mask_mnb.any():
        ax.plot(xpos_r[mask_mnb], df_resid_mnb.loc[mask_mnb, "baseline"].values,
                "--", color="gray", linewidth=1.5, label="Majority baseline")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xticks(xpos_r)
    ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=7, rotation=45)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Residual class (neg/neu/pos, thr=\u00b1{NEUTRAL_THR:.0%})\n{slug} ({EMBED_DIM}d)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_residual_return.png", dpi=150)
    plt.close(fig)

    # --- Plot 2b: residual accuracy vs horizon (surprise inliers only) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    mask_mnb = df_resid_in_mnb["accuracy"].notna()
    mask_gnb = df_resid_in_gnb["accuracy"].notna()
    mask_rq_r = df_resid_in_rq["accuracy"].notna()
    mask_lr_r = df_resid_in_lr["accuracy"].notna()
    xpos_r = np.arange(len(df_resid_in_mnb))
    if mask_mnb.any():
        ax.plot(xpos_r[mask_mnb], df_resid_in_mnb.loc[mask_mnb, "accuracy"].values,
                "s-", color="#8e44ad", linewidth=2, label="MultinomialNB")
        ax.plot(xpos_r[mask_mnb], df_resid_in_mnb.loc[mask_mnb, "baseline"].values,
                "--", color="gray", linewidth=1.5, label="Majority baseline")
    if mask_gnb.any():
        ax.plot(xpos_r[mask_gnb], df_resid_in_gnb.loc[mask_gnb, "accuracy"].values,
                "D-", color="#3498db", linewidth=2, label="GaussianNB")
    if mask_rq_r.any():
        ax.plot(xpos_r[mask_rq_r], df_resid_in_rq.loc[mask_rq_r, "accuracy"].values,
                "v-", color="#e67e22", linewidth=2, label="RQ+MNB")
    if mask_lr_r.any():
        ax.plot(xpos_r[mask_lr_r], df_resid_in_lr.loc[mask_lr_r, "accuracy"].values,
                "P-", color="#c0392b", linewidth=2, label=f"PCA({PCA_DIM})+LogReg")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xticks(xpos_r)
    ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=7, rotation=45)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Residual class (surprise inliers only)\n"
        f"q[{SURPRISE_INLIER_Q_LOW:.0%}, {SURPRISE_INLIER_Q_HIGH:.0%}] + {slug}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.9)
    plt.tight_layout()
    fig.savefig(INLIERS_DIR / "accuracy_residual_return_inliers.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: surprise (3-class bar) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    if surprise_result_mnb or surprise_result_gnb or surprise_result_rq or surprise_result_lr:
        labels = ["MNB", "GNB", "RQ+MNB", "PCA+LR", "BOW", "Baseline"]
        acc_mnb_s = surprise_result_mnb.get("accuracy", 0) if surprise_result_mnb else 0
        acc_gnb_s = surprise_result_gnb.get("accuracy", 0) if surprise_result_gnb else 0
        acc_rq_s = surprise_result_rq.get("accuracy", 0) if surprise_result_rq else 0
        acc_lr_s = surprise_result_lr.get("accuracy", 0) if surprise_result_lr else 0
        acc_bow_s = surprise_result_bow.get("accuracy", 0) if surprise_result_bow else 0
        bl_s = surprise_result_mnb.get("baseline", 0) if surprise_result_mnb else 0
        bars = ax.bar(labels, [acc_mnb_s, acc_gnb_s, acc_rq_s, acc_lr_s, acc_bow_s, bl_s],
                      color=["#2ecc71", "#3498db", "#e67e22", "#c0392b", "#7f8c8d", "gray"], width=0.5)
        ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
        ax.set_ylim(0, 0.9)
        n_s = surprise_result_mnb.get("n", 0) if surprise_result_mnb else 0
        dist_s = surprise_result_mnb.get("class_dist", "") if surprise_result_mnb else ""
        ax.set_title(f"Surprise class (neg/neu/pos)\n(n={n_s}, {dist_s})")
        ax.set_ylabel("Accuracy")
        for bar, val in zip(bars, [acc_mnb_s, acc_gnb_s, acc_rq_s, acc_lr_s, acc_bow_s, bl_s]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                    ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Surprise class (neg/neu/pos)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_surprise.png", dpi=150)
    plt.close(fig)

    # --- Plot 4: ROC AUC raw return vs horizon ---
    fig, ax = plt.subplots(figsize=(8, 5))
    xpos2 = np.arange(len(df_raw_mnb))
    if "roc_auc" in df_raw_mnb.columns:
        m = df_raw_mnb["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2[m], df_raw_mnb.loc[m, "roc_auc"].values,
                    "o-", color="#2ecc71", linewidth=2, label="MultinomialNB")
    if "roc_auc" in df_raw_gnb.columns:
        m = df_raw_gnb["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2[m], df_raw_gnb.loc[m, "roc_auc"].values,
                    "^-", color="#3498db", linewidth=2, label="GaussianNB")
    if "roc_auc" in df_raw_rq.columns:
        m = df_raw_rq["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2[m], df_raw_rq.loc[m, "roc_auc"].values,
                    "v-", color="#e67e22", linewidth=2, label="RQ+MNB")
    if "roc_auc" in df_raw_lr.columns:
        m = df_raw_lr["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2[m], df_raw_lr.loc[m, "roc_auc"].values,
                    "P-", color="#c0392b", linewidth=2, label=f"PCA({PCA_DIM})+LogReg")
    if "roc_auc" in df_raw_bow.columns:
        m = df_raw_bow["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2[m], df_raw_bow.loc[m, "roc_auc"].values,
                    "x--", color="#7f8c8d", linewidth=2, label="BOW (TF-IDF)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Random (0.5)")
    ax.set_xticks(xpos2)
    ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=7, rotation=45)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"ROC AUC — Return class (neg/neu/pos, thr=\u00b1{NEUTRAL_THR:.0%})\n{slug} ({EMBED_DIM}d)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_auc_raw_return.png", dpi=150)
    plt.close(fig)

    # --- Plot 5: ROC AUC residual vs horizon ---
    fig, ax = plt.subplots(figsize=(8, 5))
    xpos2_r = np.arange(len(df_resid_mnb))
    if "roc_auc" in df_resid_mnb.columns:
        m = df_resid_mnb["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2_r[m], df_resid_mnb.loc[m, "roc_auc"].values,
                    "s-", color="#e74c3c", linewidth=2, label="MultinomialNB")
    if "roc_auc" in df_resid_gnb.columns:
        m = df_resid_gnb["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2_r[m], df_resid_gnb.loc[m, "roc_auc"].values,
                    "D-", color="#9b59b6", linewidth=2, label="GaussianNB")
    if "roc_auc" in df_resid_rq.columns:
        m = df_resid_rq["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2_r[m], df_resid_rq.loc[m, "roc_auc"].values,
                    "v-", color="#e67e22", linewidth=2, label="RQ+MNB")
    if "roc_auc" in df_resid_lr.columns:
        m = df_resid_lr["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2_r[m], df_resid_lr.loc[m, "roc_auc"].values,
                    "P-", color="#c0392b", linewidth=2, label=f"PCA({PCA_DIM})+LogReg")
    if "roc_auc" in df_resid_bow.columns:
        m = df_resid_bow["roc_auc"].notna()
        if m.any():
            ax.plot(xpos2_r[m], df_resid_bow.loc[m, "roc_auc"].values,
                    "x--", color="#7f8c8d", linewidth=2, label="BOW (TF-IDF)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Random (0.5)")
    ax.set_xticks(xpos2_r)
    ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=7, rotation=45)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"ROC AUC — Residual class (neg/neu/pos, thr=\u00b1{NEUTRAL_THR:.0%})\n{slug} ({EMBED_DIM}d)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_auc_residual_return.png", dpi=150)
    plt.close(fig)

    # --- Plot 6: ROC AUC surprise bar ---
    fig, ax = plt.subplots(figsize=(6, 5))
    auc_mnb_s = surprise_result_mnb.get("roc_auc", float("nan")) if surprise_result_mnb else float("nan")
    auc_gnb_s = surprise_result_gnb.get("roc_auc", float("nan")) if surprise_result_gnb else float("nan")
    auc_rq_s = surprise_result_rq.get("roc_auc", float("nan")) if surprise_result_rq else float("nan")
    auc_lr_s = surprise_result_lr.get("roc_auc", float("nan")) if surprise_result_lr else float("nan")
    auc_bow_s = surprise_result_bow.get("roc_auc", float("nan")) if surprise_result_bow else float("nan")
    if not (np.isnan(auc_mnb_s) and np.isnan(auc_gnb_s) and np.isnan(auc_rq_s) and np.isnan(auc_lr_s)):
        labels = ["MNB", "GNB", "RQ+MNB", "PCA+LR", "BOW", "Random"]
        vals = [auc_mnb_s if not np.isnan(auc_mnb_s) else 0,
                auc_gnb_s if not np.isnan(auc_gnb_s) else 0,
                auc_rq_s if not np.isnan(auc_rq_s) else 0,
                auc_lr_s if not np.isnan(auc_lr_s) else 0,
                auc_bow_s if not np.isnan(auc_bow_s) else 0, 0.5]
        bars = ax.bar(labels, vals, color=["#2ecc71", "#3498db", "#e67e22", "#c0392b", "#7f8c8d", "gray"], width=0.5)
        ax.set_ylim(0.3, 1.0)
        n_s = surprise_result_mnb.get("n", 0) if surprise_result_mnb else 0
        ax.set_title(f"ROC AUC — Surprise class (n={n_s})")
        ax.set_ylabel("ROC AUC")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                    ha="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("ROC AUC — Surprise class")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_auc_surprise.png", dpi=150)
    plt.close(fig)

    # --- Plot 7: PR AUC raw return vs horizon ---
    fig, ax = plt.subplots(figsize=(8, 5))
    xpos3 = np.arange(len(df_raw_mnb))
    if "pr_auc" in df_raw_mnb.columns:
        m = df_raw_mnb["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3[m], df_raw_mnb.loc[m, "pr_auc"].values,
                    "o-", color="#2ecc71", linewidth=2, label="MultinomialNB")
    if "pr_auc" in df_raw_gnb.columns:
        m = df_raw_gnb["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3[m], df_raw_gnb.loc[m, "pr_auc"].values,
                    "^-", color="#3498db", linewidth=2, label="GaussianNB")
    if "pr_auc" in df_raw_rq.columns:
        m = df_raw_rq["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3[m], df_raw_rq.loc[m, "pr_auc"].values,
                    "v-", color="#e67e22", linewidth=2, label="RQ+MNB")
    if "pr_auc" in df_raw_lr.columns:
        m = df_raw_lr["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3[m], df_raw_lr.loc[m, "pr_auc"].values,
                    "P-", color="#c0392b", linewidth=2, label=f"PCA({PCA_DIM})+LogReg")
    if "pr_auc" in df_raw_bow.columns:
        m = df_raw_bow["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3[m], df_raw_bow.loc[m, "pr_auc"].values,
                    "x--", color="#7f8c8d", linewidth=2, label="BOW (TF-IDF)")
    ax.axhline(1.0 / 3, color="gray", linestyle="--", linewidth=1.5, label="Random (0.33)")
    ax.set_xticks(xpos3)
    ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=7, rotation=45)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("PR AUC")
    ax.set_title(f"PR AUC — Return class (neg/neu/pos, thr=\u00b1{NEUTRAL_THR:.0%})\n{slug} ({EMBED_DIM}d)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pr_auc_raw_return.png", dpi=150)
    plt.close(fig)

    # --- Plot 8: PR AUC residual vs horizon ---
    fig, ax = plt.subplots(figsize=(8, 5))
    xpos3_r = np.arange(len(df_resid_mnb))
    if "pr_auc" in df_resid_mnb.columns:
        m = df_resid_mnb["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3_r[m], df_resid_mnb.loc[m, "pr_auc"].values,
                    "s-", color="#e74c3c", linewidth=2, label="MultinomialNB")
    if "pr_auc" in df_resid_gnb.columns:
        m = df_resid_gnb["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3_r[m], df_resid_gnb.loc[m, "pr_auc"].values,
                    "D-", color="#9b59b6", linewidth=2, label="GaussianNB")
    if "pr_auc" in df_resid_rq.columns:
        m = df_resid_rq["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3_r[m], df_resid_rq.loc[m, "pr_auc"].values,
                    "v-", color="#e67e22", linewidth=2, label="RQ+MNB")
    if "pr_auc" in df_resid_lr.columns:
        m = df_resid_lr["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3_r[m], df_resid_lr.loc[m, "pr_auc"].values,
                    "P-", color="#c0392b", linewidth=2, label=f"PCA({PCA_DIM})+LogReg")
    if "pr_auc" in df_resid_bow.columns:
        m = df_resid_bow["pr_auc"].notna()
        if m.any():
            ax.plot(xpos3_r[m], df_resid_bow.loc[m, "pr_auc"].values,
                    "x--", color="#7f8c8d", linewidth=2, label="BOW (TF-IDF)")
    ax.axhline(1.0 / 3, color="gray", linestyle="--", linewidth=1.5, label="Random (0.33)")
    ax.set_xticks(xpos3_r)
    ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=7, rotation=45)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("PR AUC")
    ax.set_title(f"PR AUC — Residual class (neg/neu/pos, thr=\u00b1{NEUTRAL_THR:.0%})\n{slug} ({EMBED_DIM}d)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pr_auc_residual_return.png", dpi=150)
    plt.close(fig)

    # --- Plot 9: PR AUC surprise bar ---
    fig, ax = plt.subplots(figsize=(6, 5))
    pr_mnb_s = surprise_result_mnb.get("pr_auc", float("nan")) if surprise_result_mnb else float("nan")
    pr_gnb_s = surprise_result_gnb.get("pr_auc", float("nan")) if surprise_result_gnb else float("nan")
    pr_rq_s = surprise_result_rq.get("pr_auc", float("nan")) if surprise_result_rq else float("nan")
    pr_lr_s = surprise_result_lr.get("pr_auc", float("nan")) if surprise_result_lr else float("nan")
    pr_bow_s = surprise_result_bow.get("pr_auc", float("nan")) if surprise_result_bow else float("nan")
    if not (np.isnan(pr_mnb_s) and np.isnan(pr_gnb_s) and np.isnan(pr_rq_s) and np.isnan(pr_lr_s)):
        labels = ["MNB", "GNB", "RQ+MNB", "PCA+LR", "BOW", "Random"]
        vals = [pr_mnb_s if not np.isnan(pr_mnb_s) else 0,
                pr_gnb_s if not np.isnan(pr_gnb_s) else 0,
                pr_rq_s if not np.isnan(pr_rq_s) else 0,
                pr_lr_s if not np.isnan(pr_lr_s) else 0,
                pr_bow_s if not np.isnan(pr_bow_s) else 0, 1.0 / 3]
        bars = ax.bar(labels, vals, color=["#2ecc71", "#3498db", "#e67e22", "#c0392b", "#7f8c8d", "gray"], width=0.5)
        ax.set_ylim(0.0, 1.0)
        n_s = surprise_result_mnb.get("n", 0) if surprise_result_mnb else 0
        ax.set_title(f"PR AUC — Surprise class (n={n_s})")
        ax.set_ylabel("PR AUC")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                    ha="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("PR AUC — Surprise class")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pr_auc_surprise.png", dpi=150)
    plt.close(fig)

    # Save surprise results
    if surprise_result_mnb:
        pd.DataFrame([surprise_result_mnb]).to_csv(OUTPUT_DIR / "mnb_surprise.csv", index=False)
    if surprise_result_gnb:
        pd.DataFrame([surprise_result_gnb]).to_csv(OUTPUT_DIR / "gnb_surprise.csv", index=False)
    if surprise_result_rq:
        pd.DataFrame([surprise_result_rq]).to_csv(OUTPUT_DIR / "rq_mnb_surprise.csv", index=False)
    if surprise_result_lr:
        pd.DataFrame([surprise_result_lr]).to_csv(OUTPUT_DIR / "lr_pca_surprise.csv", index=False)

    print(f"\nSaved results to {OUTPUT_DIR}/")
    print(f"  mnb_raw_return.csv / gnb_raw_return.csv / rq_mnb_raw_return.csv / lr_pca_raw_return.csv")
    print(f"  mnb_residual_return.csv / gnb_residual_return.csv / rq_mnb_residual_return.csv / lr_pca_residual_return.csv")
    print(f"  surprise_inliers/mnb_residual_return_inliers.csv / surprise_inliers/gnb_residual_return_inliers.csv")
    print(f"  surprise_inliers/rq_mnb_residual_return_inliers.csv / surprise_inliers/lr_pca_residual_return_inliers.csv")
    print(f"  mnb_surprise.csv / gnb_surprise.csv / rq_mnb_surprise.csv / lr_pca_surprise.csv")
    print(f"  accuracy_raw_return.png / accuracy_residual_return.png / surprise_inliers/accuracy_residual_return_inliers.png / accuracy_surprise.png")
    print(f"  roc_auc_raw_return.png / roc_auc_residual_return.png / roc_auc_surprise.png")
    print(f"  pr_auc_raw_return.png / pr_auc_residual_return.png / pr_auc_surprise.png")
    print(f"  embeddings.npy")

    # Print summary table
    print("\n" + "=" * 200)
    print(f"{'Horizon':>8} {'MNB Acc':>8} {'MNB AUC':>8} {'MNB PR':>7} {'GNB Acc':>8} {'GNB AUC':>8} {'GNB PR':>7} {'RQ Acc':>7} {'RQ AUC':>7} {'RQ PR':>6} {'LR Acc':>7} {'LR AUC':>7} {'LR PR':>6} {'Baseline':>9} {'MNB Res':>8} {'GNB Res':>8} {'RQ Res':>7} {'LR Res':>7} {'N':>5}")
    print("-" * 200)
    for i, h in enumerate(HORIZONS):
        mnb_acc = results_raw_mnb[i].get("accuracy", np.nan)
        mnb_auc = results_raw_mnb[i].get("roc_auc", np.nan)
        mnb_pr = results_raw_mnb[i].get("pr_auc", np.nan)
        gnb_acc = results_raw_gnb[i].get("accuracy", np.nan)
        gnb_auc = results_raw_gnb[i].get("roc_auc", np.nan)
        gnb_pr = results_raw_gnb[i].get("pr_auc", np.nan)
        rq_acc = results_raw_rq[i].get("accuracy", np.nan)
        rq_auc = results_raw_rq[i].get("roc_auc", np.nan)
        rq_pr = results_raw_rq[i].get("pr_auc", np.nan)
        lr_acc = results_raw_lr[i].get("accuracy", np.nan)
        lr_auc = results_raw_lr[i].get("roc_auc", np.nan)
        lr_pr = results_raw_lr[i].get("pr_auc", np.nan)
        bl = results_raw_mnb[i].get("baseline", np.nan)
        mnb_res = results_residual_mnb[i].get("accuracy", np.nan)
        gnb_res = results_residual_gnb[i].get("accuracy", np.nan)
        rq_res = results_residual_rq[i].get("accuracy", np.nan)
        lr_res = results_residual_lr[i].get("accuracy", np.nan)
        n = results_raw_mnb[i].get("n", 0)
        print(f"{h:>8d} {mnb_acc:>8.3f} {mnb_auc:>8.3f} {mnb_pr:>7.3f} {gnb_acc:>8.3f} {gnb_auc:>8.3f} {gnb_pr:>7.3f} {rq_acc:>7.3f} {rq_auc:>7.3f} {rq_pr:>6.3f} {lr_acc:>7.3f} {lr_auc:>7.3f} {lr_pr:>6.3f} {bl:>9.3f} {mnb_res:>8.3f} {gnb_res:>8.3f} {rq_res:>7.3f} {lr_res:>7.3f} {n:>5d}")
    print("-" * 200)
    if surprise_result_mnb:
        mnb_auc_s = surprise_result_mnb.get('roc_auc', np.nan)
        mnb_pr_s = surprise_result_mnb.get('pr_auc', np.nan)
        gnb_auc_s = surprise_result_gnb.get('roc_auc', np.nan) if surprise_result_gnb else np.nan
        gnb_pr_s = surprise_result_gnb.get('pr_auc', np.nan) if surprise_result_gnb else np.nan
        rq_acc_s = surprise_result_rq.get('accuracy', np.nan) if surprise_result_rq else np.nan
        rq_auc_s = surprise_result_rq.get('roc_auc', np.nan) if surprise_result_rq else np.nan
        rq_pr_s = surprise_result_rq.get('pr_auc', np.nan) if surprise_result_rq else np.nan
        lr_acc_s = surprise_result_lr.get('accuracy', np.nan) if surprise_result_lr else np.nan
        lr_auc_s = surprise_result_lr.get('roc_auc', np.nan) if surprise_result_lr else np.nan
        lr_pr_s = surprise_result_lr.get('pr_auc', np.nan) if surprise_result_lr else np.nan
        print(f"{'surprise':>8} {surprise_result_mnb['accuracy']:>8.3f} {mnb_auc_s:>8.3f} {mnb_pr_s:>7.3f} {surprise_result_gnb.get('accuracy', np.nan):>8.3f} {gnb_auc_s:>8.3f} {gnb_pr_s:>7.3f} {rq_acc_s:>7.3f} {rq_auc_s:>7.3f} {rq_pr_s:>6.3f} {lr_acc_s:>7.3f} {lr_auc_s:>7.3f} {lr_pr_s:>6.3f} {surprise_result_mnb['baseline']:>9.3f} {'\u2014':>8} {'\u2014':>8} {'\u2014':>7} {'\u2014':>7} {surprise_result_mnb['n']:>5d}")
    print("=" * 200)


if __name__ == "__main__":
    main()
