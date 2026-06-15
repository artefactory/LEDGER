"""
Grid search over neutral-class thresholds to find optimal classification boundaries.

For each threshold in the grid and each horizon, evaluates classifiers
via 5-fold stratified CV. Supports 5 modes:
  - minilm:   all-MiniLM-L6-v2 embeddings (MNB, GNB, RQ+MNB, PCA+LogReg)
  - roberta:  all-roberta-large-v1 embeddings (MNB, GNB, RQ+MNB, PCA+LogReg)
  - baai:     BAAI/bge-large-en-v1.5 embeddings (1024d, best classification perf)
  - eurobert: EuroBERT-2.1B embeddings (MNB, GNB, PCA+LogReg)
  - bow:      CountVectorizer bag-of-words (MNB only)

Results saved as JSON.

Usage:
    uv run python KPI_analysis/threshold_grid_search.py --mode minilm
    uv run python KPI_analysis/threshold_grid_search.py --mode roberta
    uv run python KPI_analysis/threshold_grid_search.py --mode baai
    uv run python KPI_analysis/threshold_grid_search.py --mode eurobert
    uv run python KPI_analysis/threshold_grid_search.py --mode bow
    uv run python KPI_analysis/threshold_grid_search.py --mode all
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from FinancialIndicators import (
    GetIndicatorsForPrices,
    GetIndustryDataFrame,
    find_ticker_industry,
)
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
OUTPUT_DIR = HERE / "output" / "plots" / "threshold_grid_search"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

# --- Grid search parameters ---
THRESHOLDS = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
SURPRISE_THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
HORIZONS = [-90, -80, -70, -60, -50, -40, -30, -20, -10, -5, -1, 1, 2, 3, 4, 5,]
MAX_LAG = max(max(HORIZONS), abs(min(HORIZONS)))

# Industry benchmark weighting used to build the "unbiased" return target
# (stock cumulative return minus industry cumulative return at the same horizon).
# "_vw" = value-weighted (matches cum_return_unbiased_vw elsewhere); "" = equal-weighted.
INDUSTRY_WEIGHTING = ""

# MiniLM params
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIM = 384

# RoBERTa params
ROBERTA_MODEL = "sentence-transformers/all-roberta-large-v1"
ROBERTA_DIM = 768

PCA_DIM = 50
RQ_N_LEVELS = 8
RQ_CODEBOOK_SIZE = 32

# EuroBERT params
EUROBERT_MODEL = "EuroBERT/EuroBERT-2.1B"
EUROBERT_DIM = 2048

# Encoder presets (mode -> model name)
ENCODER_PRESETS = {
    "minilm": MINILM_MODEL,
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
    "roberta": ROBERTA_MODEL,
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "eurobert": EUROBERT_MODEL,
    "baai": "BAAI/bge-large-en-v1.5",
    # "pplx": "perplexity-ai/pplx-embed-v1-0.6b",
    "baai-m3": "BAAI/bge-m3",
    "mpnet_nli": "sentence-transformers/nli-mpnet-base-v2",
    "modernbert": "nomic-ai/modernbert-embed-base",
    "gemma": "google/embeddinggemma-300m",
}

BENCH_START = date(2016, 6, 1)
BENCH_END = date(2024, 6, 30)


def residual_quantize_fit(X: np.ndarray, n_levels: int, codebook_size: int,
                          random_state: int = 42) -> list:
    codebooks = []
    residual = X.copy()
    for level in range(n_levels):
        km = KMeans(n_clusters=codebook_size, random_state=random_state + level, n_init=3)
        km.fit(residual)
        codebooks.append(km)
        residual = residual - km.cluster_centers_[km.labels_]
    return codebooks


def residual_quantize_transform(X: np.ndarray, codebooks: list) -> np.ndarray:
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


def _pr_auc_ovr_scorer(estimator, X, y):
    """PR AUC (average precision) with OVR macro averaging for multiclass."""
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    Y_bin = label_binarize(y, classes=classes)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
    else:
        return float("nan")
    # Align proba columns with Y_bin columns
    if proba.shape[1] != Y_bin.shape[1]:
        return float("nan")
    scores = []
    for i in range(Y_bin.shape[1]):
        if Y_bin[:, i].sum() == 0:
            continue
        scores.append(average_precision_score(Y_bin[:, i], proba[:, i]))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_threshold(thr: float, horizon: int, y_ret: np.ndarray,
                       features: dict, classifiers: dict, cv) -> dict:
    """Evaluate classifiers at a given threshold and horizon.

    features: dict of name -> X array
    classifiers: dict of name -> (sklearn estimator, feature_key)
    """
    y_3class = np.where(y_ret > thr, 2, np.where(y_ret < -thr, 0, 1))
    class_counts = np.bincount(y_3class, minlength=3)
    baseline = class_counts.max() / len(y_3class)
    n = len(y_3class)

    result = {
        "threshold": thr,
        "horizon": horizon,
        "n": n,
        "baseline": float(baseline),
        "class_neg": int(class_counts[0]),
        "class_neu": int(class_counts[1]),
        "class_pos": int(class_counts[2]),
    }

    if any(c < 5 for c in class_counts):
        result["skipped"] = True
        result["skip_reason"] = "class_too_small"
        return result

    result["skipped"] = False

    import warnings as _warnings

    for clf_name, (estimator, feat_key) in classifiers.items():
        X = features[feat_key]
        # Use n_jobs=1 for SAGA solvers so convergence warnings are captured
        is_saga = "l1" in clf_name
        njobs = 1 if is_saga else -1
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            try:
                acc = cross_val_score(estimator, X, y_3class, cv=cv, scoring="accuracy", n_jobs=njobs).mean()
            except ValueError:
                acc = float("nan")
            try:
                auc = cross_val_score(estimator, X, y_3class, cv=cv, scoring="roc_auc_ovr", n_jobs=njobs).mean()
            except ValueError:
                auc = float("nan")
            try:
                pr_auc = cross_val_score(estimator, X, y_3class, cv=cv, scoring=_pr_auc_ovr_scorer, n_jobs=njobs).mean()
            except ValueError:
                pr_auc = float("nan")
        convergence_warns = [w for w in caught if "ConvergenceWarning" in str(w.category.__name__ if hasattr(w.category, '__name__') else w.category)]
        if convergence_warns:
            print(f"    ⚠ CONVERGENCE WARNING: clf={clf_name}, h={horizon}, thr={thr} ({len(convergence_warns)} warns)")
        result[f"{clf_name}_accuracy"] = float(acc)
        result[f"{clf_name}_roc_auc"] = float(auc)
        result[f"{clf_name}_pr_auc"] = float(pr_auc)

    return result


def build_records(all_tickers, letter_texts):
    """Build dataset records: fetch earnings dates and prices."""
    print("Fetching earnings dates and prices...")
    earnings_cache: dict[str, pd.DataFrame] = {}
    prices_cache: dict[str, pd.DataFrame | None] = {}
    # Industry benchmark, aligned to each stock's price index, keyed by ticker.
    industry_aligned_cache: dict[str, pd.DataFrame | None] = {}

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

        # Shift t0 to +2 trading days after earnings date (post-reaction anchor)
        t0_pos = prices.index.get_loc(pub_ts) + 2
        if t0_pos >= len(prices):
            continue
        pub_ts = prices.index[t0_pos]

        if t0_pos + MAX_LAG >= len(prices):
            continue

        # Industry benchmark aligned to this stock's index (cached per ticker).
        if ticker not in industry_aligned_cache:
            industry_df = GetIndustryDataFrame(ticker, BENCH_START, BENCH_END, max_lag=MAX_LAG)
            if industry_df is not None and not industry_df.empty:
                industry_aligned_cache[ticker] = industry_df.reindex(prices.index)
            else:
                industry_aligned_cache[ticker] = None
        ind_aligned = industry_aligned_cache[ticker]

        returns = {}
        unbiased_returns = {}
        for h in HORIZONS:
            col = f"return_t{h}"
            if col not in prices.columns:
                continue
            val = prices.loc[pub_ts, col]
            if pd.isna(val):
                continue
            returns[h] = float(val)

            # Unbiased return = stock return - industry return at the same horizon.
            if ind_aligned is None:
                continue
            ind_col = f"return_t{h}{INDUSTRY_WEIGHTING}"
            if ind_col in ind_aligned.columns and pub_ts in ind_aligned.index:
                ind_val = ind_aligned.loc[pub_ts, ind_col]
                if pd.notna(ind_val):
                    unbiased_returns[h] = float(val) - float(ind_val)

        if not returns:
            continue

        records.append({
            "ticker": ticker,
            "year": year,
            "text": text,
            "returns": returns,
            "unbiased_returns": unbiased_returns,
            "surprise": float(surprise) if surprise is not None and not pd.isna(surprise) else None,
        })

    return records


# EmbeddingGemma is a task-conditioned embedder: the input must be wrapped with a
# task prompt. For peer classification we use its "Classification" task so the
# vectors are tuned for a downstream classifier (the trained estimators below).
GEMMA_CLASSIFICATION_PROMPT = "task: classification | query: "


def _is_gemma(model_name: str) -> bool:
    return "embeddinggemma" in model_name.lower() or "/gemma" in model_name.lower()


def _embedding_cache_path(model_name: str, corpus: list[str],
                          prompt: str | None = None) -> Path:
    """Return cache path for raw embeddings, keyed by model slug + corpus hash.

    The task prompt (if any) is folded into the hash so prompt-conditioned
    embeddings (e.g. EmbeddingGemma classification) never collide with plain ones.
    """
    import hashlib
    slug = model_name.split("/")[-1]
    payload = "\n".join(corpus)
    if prompt:
        payload = f"<<prompt:{prompt}>>\n{payload}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return EMBEDDINGS_DIR / f"{slug}_{h}.npy"


def get_raw_embeddings(corpus: list[str], model_name: str) -> tuple[np.ndarray, int]:
    """Return raw (L2-normalised) embeddings for a corpus, cached on disk.

    Dispatches to SentenceTransformers or EuroBERT (transformers) based on the
    model name, and caches the result under EMBEDDINGS_DIR keyed by model slug +
    corpus hash so repeated runs reuse the same vectors. EmbeddingGemma is
    encoded through its "Classification" task prompt.
    """
    prompt = GEMMA_CLASSIFICATION_PROMPT if _is_gemma(model_name) else None
    cache_path = _embedding_cache_path(model_name, corpus, prompt)
    if cache_path.exists():
        print(f"\nLoading cached embeddings from {cache_path.name}")
        X_raw = np.load(cache_path)
        print(f"  Embedding shape: {X_raw.shape}")
        return X_raw, X_raw.shape[1]
    if "eurobert" in model_name.lower():
        X_raw, embed_dim = _encode_eurobert_raw(corpus, model_name)
    else:
        X_raw, embed_dim = _encode_sentence_transformer_raw(corpus, model_name, prompt)
    np.save(cache_path, X_raw)
    print(f"  Saved embeddings to {cache_path.name}")
    return X_raw, embed_dim


def encode_embeddings(corpus: list[str], model_name: str) -> tuple[dict, dict, dict]:
    """Encode corpus with any model, return (features, classifiers, metadata).

    Dispatches to SentenceTransformers or EuroBERT (transformers) based on model name.
    Always produces: MNB, GNB, RQ+MNB, PCA+LogReg classifiers.
    Caches raw embeddings to disk (EMBEDDINGS_DIR) for reuse.
    """
    X_raw, embed_dim = get_raw_embeddings(corpus, model_name)

    # Build features
    scaler = MinMaxScaler()
    X_mnb = scaler.fit_transform(X_raw)

    pca = PCA(n_components=PCA_DIM, random_state=42)
    X_pca = pca.fit_transform(X_raw)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {embed_dim}d -> {PCA_DIM}d (explained variance = {explained_var:.1%})")


    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))

    # logreg_l2_001 = make_pipeline(StandardScaler(), LogisticRegression(
    #     C=0.001, l1_ratio=0, max_iter=1000, random_state=42))
    # logreg_l2_01 = make_pipeline(StandardScaler(), LogisticRegression(
    #     C=0.01, l1_ratio=0, max_iter=1000, random_state=42))
    # logreg_l2_1 = make_pipeline(StandardScaler(), LogisticRegression(
    #     C=0.1, l1_ratio=0, max_iter=1000, random_state=42))
    
    
    # logreg_l1_01 = make_pipeline(StandardScaler(), LogisticRegression(
    #     C=0.01, l1_ratio=1, solver="saga", max_iter=2000, random_state=42))
    # logreg_l1_1 = make_pipeline(StandardScaler(), LogisticRegression(
    #     C=0.1, l1_ratio=1, solver="saga", max_iter=2000, random_state=42))
    # logreg_l1_10 = make_pipeline(StandardScaler(), LogisticRegression(
    #     C=1.0, l1_ratio=1, solver="saga", max_iter=10000, random_state=42))
    
    logreg_l2 = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.01, l1_ratio=0, max_iter=2000, random_state=42))

    logreg_l1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, l1_ratio=1, solver="saga", max_iter=2000, random_state=42))
    
    features = {"mnb": X_mnb, "raw": X_raw,  "pca": X_pca}
    # classifiers = {
    #     "mnb": (MultinomialNB(), "mnb"),
    #     "gnb": (GaussianNB(var_smoothing=1e-6), "raw"),
    #     "lr_pca": (logreg, "pca"),

    #     "lr_l2_001": (logreg_l2_001, "raw"),
    #     "lr_l2_01": (logreg_l2_01, "raw"),
    #     "lr_l2_1": (logreg_l2_1, "raw"),

    #     "lr_l1_01": (logreg_l1_01, "raw"),
    #     "lr_l1_1": (logreg_l1_1, "raw"),
    #     "lr_l1_10": (logreg_l1_10, "raw"),

    #     "lda": (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "raw"),
    # }
    classifiers = {
        # "mnb": (MultinomialNB(), "mnb"),
        # "gnb": (GaussianNB(var_smoothing=1e-6), "raw"),
        # "lr_pca": (logreg, "pca"),
        "lr_l2": (logreg_l2, "raw"),
        # "lr_l1": (logreg_l1, "raw"),
        # "lda": (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "raw"),
    }
    metadata = {
        "embed_model": model_name, "embed_dim": embed_dim,
        "pca_dim": PCA_DIM, "pca_explained_var": float(explained_var),
    }
    return features, classifiers, metadata


def _encode_sentence_transformer_raw(corpus: list[str], model_name: str,
                                     prompt: str | None = None) -> tuple[np.ndarray, int]:
    """Encode with any sentence-transformers model.

    `prompt` (a raw task-prompt string) is prepended to every input by
    sentence-transformers — used to drive EmbeddingGemma's Classification task.
    """
    from sentence_transformers import SentenceTransformer

    print(f"\nEncoding with {model_name}...")
    if prompt:
        print(f"  Using task prompt: {prompt!r}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    encode_kwargs = {"show_progress_bar": True, "batch_size": 32, "normalize_embeddings": True}
    if prompt:
        encode_kwargs["prompt"] = prompt
    X_raw = model.encode(corpus, **encode_kwargs)
    X_raw = np.array(X_raw, dtype=np.float64)
    print(f"  Embedding shape: {X_raw.shape}")
    return X_raw, X_raw.shape[1]


def _encode_eurobert_raw(corpus: list[str], model_name: str) -> tuple[np.ndarray, int]:
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
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
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
    print(f"  Embedding shape: {X_raw.shape}")
    return X_raw, X_raw.shape[1]


def encode_bow(corpus: list[str]) -> tuple[dict, dict, dict]:
    """Encode with CountVectorizer, return (features, classifiers, metadata)."""
    print("\nEncoding with CountVectorizer (BoW)...")
    vectorizer = CountVectorizer(max_features=10000, stop_words="english")
    X_bow = vectorizer.fit_transform(corpus).toarray().astype(np.float64)
    print(f"  BoW shape: {X_bow.shape}")

    features = {"bow": X_bow}
    classifiers = {
        "mnb": (MultinomialNB(), "bow"),
    }
    metadata = {
        "feature_type": "bow_countvectorizer",
        "max_features": 10000,
        "vocab_size": X_bow.shape[1],
    }
    return features, classifiers, metadata


def _run_return_target(target_name: str, return_key: str, records: list,
                       features: dict, classifiers: dict, cv, clf_names: list) -> list:
    """Grid-search a per-horizon return target (raw or unbiased) over thresholds.

    `return_key` selects which per-record dict holds the y values:
    "returns" for the raw return, "unbiased_returns" for the industry-adjusted one.
    """
    results = []
    total = len(THRESHOLDS) * len(HORIZONS)
    done = 0

    for h in HORIZONS:
        valid_idx = []
        y_returns = []
        for i, r in enumerate(records):
            d = r.get(return_key, {})
            if h in d:
                valid_idx.append(i)
                y_returns.append(d[h])

        if len(valid_idx) < 20:
            for thr in THRESHOLDS:
                results.append({
                    "threshold": thr, "horizon": h, "n": len(valid_idx),
                    "skipped": True, "skip_reason": "insufficient_samples",
                    "target": target_name,
                })
                done += 1
            continue

        h_features = {k: v[valid_idx] for k, v in features.items()}
        y_ret = np.array(y_returns)

        for thr in THRESHOLDS:
            done += 1
            result = evaluate_threshold(thr, h, y_ret, h_features, classifiers, cv)
            result["target"] = target_name
            results.append(result)

            if not result.get("skipped"):
                parts = [f"{cn}={result.get(f'{cn}_accuracy', float('nan')):.3f}" for cn in clf_names]
                print(f"  [{done}/{total}] h={h:>3d}, thr={thr:.3f}: "
                      f"{' '.join(parts)} (bl={result['baseline']:.3f})")
            else:
                print(f"  [{done}/{total}] h={h:>3d}, thr={thr:.3f}: SKIPPED")

    return results


def run_grid_search(mode: str, records: list, output_dir: Path):
    """Run grid search for a given mode on targets: raw return, unbiased return,
    residual, residual inliers, surprise."""
    corpus = [r["text"] for r in records]

    if mode == "bow":
        features, classifiers, metadata = encode_bow(corpus)
    else:
        # Resolve mode to model name (preset or raw model path)
        model_name = ENCODER_PRESETS.get(mode, mode)
        features, classifiers, metadata = encode_embeddings(corpus, model_name)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_names = list(classifiers.keys())

    # ===== TARGET 1: RAW RETURN =====
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET 1: Raw Return — {len(THRESHOLDS)} thresholds x {len(HORIZONS)} horizons")
    print(f"Classifiers: {clf_names}")
    print(f"{'='*80}\n")

    raw_results = _run_return_target(
        "raw_return", "returns", records, features, classifiers, cv, clf_names)

    # ===== TARGET 1b: UNBIASED RETURN (return - industry return) =====
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET 1b: Unbiased Return (industry-adjusted, weighting='{INDUSTRY_WEIGHTING or 'ew'}') "
          f"— {len(THRESHOLDS)} thresholds x {len(HORIZONS)} horizons")
    print(f"{'='*80}\n")

    unbiased_results = _run_return_target(
        "unbiased_return", "unbiased_returns", records, features, classifiers, cv, clf_names)

    # ===== TARGET 2: RESIDUAL (return - f(surprise)) =====
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET 2: Residual — {len(THRESHOLDS)} thresholds x {len(HORIZONS)} horizons")
    print(f"{'='*80}\n")

    residual_results = []
    total = len(THRESHOLDS) * len(HORIZONS)
    done = 0

    for h in HORIZONS:
        # Only samples with both returns at this horizon AND surprise
        valid_idx = []
        y_returns = []
        surprises = []
        for i, r in enumerate(records):
            if h in r["returns"] and r["surprise"] is not None:
                valid_idx.append(i)
                y_returns.append(r["returns"][h])
                surprises.append(r["surprise"])

        if len(valid_idx) < 20:
            for thr in THRESHOLDS:
                residual_results.append({
                    "threshold": thr, "horizon": h, "n": len(valid_idx),
                    "skipped": True, "skip_reason": "insufficient_samples",
                    "target": "residual",
                })
                done += 1
            continue

        h_features = {k: v[valid_idx] for k, v in features.items()}
        y_ret = np.array(y_returns)
        surp_arr = np.array(surprises)

        # Compute residuals: return - linear_prediction(surprise)
        lr = LinearRegression()
        lr.fit(surp_arr.reshape(-1, 1), y_ret)
        y_pred = lr.predict(surp_arr.reshape(-1, 1))
        residuals = y_ret - y_pred

        for thr in THRESHOLDS:
            done += 1
            result = evaluate_threshold(thr, h, residuals, h_features, classifiers, cv)
            result["target"] = "residual"
            residual_results.append(result)

            if not result.get("skipped"):
                parts = [f"{cn}={result.get(f'{cn}_accuracy', float('nan')):.3f}" for cn in clf_names]
                print(f"  [{done}/{total}] h={h:>3d}, thr={thr:.3f}: "
                      f"{' '.join(parts)} (bl={result['baseline']:.3f})")
            else:
                print(f"  [{done}/{total}] h={h:>3d}, thr={thr:.3f}: SKIPPED")

    # ===== TARGET 2b: RESIDUAL INLIERS (surprise clipped to q2-q98, LR fit on inliers) =====
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET 2b: Residual Inliers — {len(THRESHOLDS)} thresholds x {len(HORIZONS)} horizons")
    print(f"{'='*80}\n")

    residual_inliers_results = []
    total = len(THRESHOLDS) * len(HORIZONS)
    done = 0

    for h in HORIZONS:
        valid_idx = []
        y_returns = []
        surprises = []
        for i, r in enumerate(records):
            if h in r["returns"] and r["surprise"] is not None:
                valid_idx.append(i)
                y_returns.append(r["returns"][h])
                surprises.append(r["surprise"])

        if len(valid_idx) < 20:
            for thr in THRESHOLDS:
                residual_inliers_results.append({
                    "threshold": thr, "horizon": h, "n": len(valid_idx),
                    "skipped": True, "skip_reason": "insufficient_samples",
                    "target": "residual_inliers",
                })
                done += 1
            continue

        surp_arr = np.array(surprises)
        q_lo = np.quantile(surp_arr, 0.02)
        q_hi = np.quantile(surp_arr, 0.98)
        inlier_mask = (surp_arr >= q_lo) & (surp_arr <= q_hi)
        n_inliers = int(inlier_mask.sum())

        if n_inliers < 20:
            for thr in THRESHOLDS:
                residual_inliers_results.append({
                    "threshold": thr, "horizon": h, "n": n_inliers,
                    "skipped": True, "skip_reason": "insufficient_inliers",
                    "target": "residual_inliers",
                })
                done += 1
            continue

        # Keep only inlier indices
        inlier_idx = np.array(valid_idx)[inlier_mask]
        h_features = {k: v[inlier_idx] for k, v in features.items()}
        y_ret = np.array(y_returns)[inlier_mask]
        surp_in = surp_arr[inlier_mask]

        # Fit LR on inliers only
        lr = LinearRegression()
        lr.fit(surp_in.reshape(-1, 1), y_ret)
        y_pred = lr.predict(surp_in.reshape(-1, 1))
        residuals = y_ret - y_pred

        for thr in THRESHOLDS:
            done += 1
            result = evaluate_threshold(thr, h, residuals, h_features, classifiers, cv)
            result["target"] = "residual_inliers"
            result["surprise_q_low"] = float(q_lo)
            result["surprise_q_high"] = float(q_hi)
            result["n_outliers_removed"] = len(valid_idx) - n_inliers
            residual_inliers_results.append(result)

            if not result.get("skipped"):
                parts = [f"{cn}={result.get(f'{cn}_accuracy', float('nan')):.3f}" for cn in clf_names]
                print(f"  [{done}/{total}] h={h:>3d}, thr={thr:.3f}: "
                      f"{' '.join(parts)} (bl={result['baseline']:.3f}, n_in={n_inliers})")
            else:
                print(f"  [{done}/{total}] h={h:>3d}, thr={thr:.3f}: SKIPPED")

    # ===== TARGET 3: SURPRISE (EPS surprise class) =====
    print(f"\n{'='*80}")
    print(f"[{mode}] TARGET 3: Surprise — {len(SURPRISE_THRESHOLDS)} thresholds")
    print(f"{'='*80}\n")

    surprise_results = []

    # Samples with surprise
    surp_idx = []
    surp_vals = []
    for i, r in enumerate(records):
        if r["surprise"] is not None:
            surp_idx.append(i)
            surp_vals.append(r["surprise"])

    if len(surp_idx) < 20:
        print(f"  Only {len(surp_idx)} samples with surprise, skipping")
        for thr in SURPRISE_THRESHOLDS:
            surprise_results.append({
                "threshold": thr, "n": len(surp_idx),
                "skipped": True, "skip_reason": "insufficient_samples",
                "target": "surprise",
            })
    else:
        s_features = {k: v[surp_idx] for k, v in features.items()}
        y_surp = np.array(surp_vals)

        for thr in SURPRISE_THRESHOLDS:
            result = evaluate_threshold(thr, 0, y_surp, s_features, classifiers, cv)
            result["target"] = "surprise"
            # Remove 'horizon' key, not relevant for surprise
            result.pop("horizon", None)
            surprise_results.append(result)

            if not result.get("skipped"):
                parts = [f"{cn}={result.get(f'{cn}_accuracy', float('nan')):.3f}" for cn in clf_names]
                print(f"  thr={thr:>5.1f}%: {' '.join(parts)} "
                      f"(bl={result['baseline']:.3f}, n={result['n']}, "
                      f"neg={result['class_neg']} neu={result['class_neu']} pos={result['class_pos']})")
            else:
                print(f"  thr={thr:>5.1f}%: SKIPPED ({result.get('skip_reason', '')})")

    # ===== SAVE ALL RESULTS =====
    # Use a filesystem-safe slug for the output filename
    mode_slug = mode if mode in ENCODER_PRESETS or mode == "bow" else mode.split("/")[-1]
    output_path = output_dir / f"grid_search_{mode_slug}.json"
    with open(output_path, "w") as f:
        json.dump({
            "mode": mode,
            "thresholds_return": THRESHOLDS,
            "thresholds_surprise": SURPRISE_THRESHOLDS,
            "horizons": HORIZONS,
            "n_samples_total": len(records),
            "classifiers": clf_names,
            "metadata": metadata,
            "industry_weighting": INDUSTRY_WEIGHTING or "ew",
            "raw_return_results": raw_results,
            "unbiased_return_results": unbiased_results,
            "residual_results": residual_results,
            "residual_inliers_results": residual_inliers_results,
            "surprise_results": surprise_results,
        }, f, indent=2)
    print(f"\n[{mode}] Results saved to {output_path}")

    # ===== SUMMARY TABLES =====
    _print_summary(f"[{mode}] Raw Return", raw_results, clf_names, HORIZONS)
    _print_summary(f"[{mode}] Unbiased Return", unbiased_results, clf_names, HORIZONS)
    _print_summary(f"[{mode}] Residual", residual_results, clf_names, HORIZONS)
    _print_summary(f"[{mode}] Residual Inliers", residual_inliers_results, clf_names, HORIZONS)

    # Surprise summary
    print(f"\n{'='*120}")
    print(f"[{mode}] Surprise — Best threshold (by max ROC AUC)")
    print(f"{'Threshold':>10} {'Classifier':>12} {'Accuracy':>9} {'ROC AUC':>8} {'PR AUC':>7} {'Baseline':>9} {'N':>5}")
    print(f"{'-'*120}")
    best = None
    best_auc = -1
    best_clf = ""
    for r in surprise_results:
        if r.get("skipped"):
            continue
        for cn in clf_names:
            val = r.get(f"{cn}_roc_auc", float("nan"))
            if not np.isnan(val) and val > best_auc:
                best_auc = val
                best = r
                best_clf = cn
    if best:
        acc = best.get(f"{best_clf}_accuracy", float("nan"))
        pr_auc = best.get(f"{best_clf}_pr_auc", float("nan"))
        print(f"{best['threshold']:>10.1f}% {best_clf:>12} {acc:>9.3f} {best_auc:>8.3f} "
              f"{pr_auc:>7.3f} {best['baseline']:>9.3f} {best['n']:>5d}")
    print(f"{'='*120}")


def _print_summary(label: str, results: list, clf_names: list, horizons: list):
    """Print best threshold per horizon summary table."""
    print(f"\n{'='*120}")
    print(f"{label} — Best threshold per horizon (by max ROC AUC)")
    print(f"{'Horizon':>8} {'Best Thr':>9} {'Classifier':>12} {'Accuracy':>9} {'ROC AUC':>8} {'PR AUC':>7} {'Baseline':>9} {'N':>5}")
    print(f"{'-'*120}")

    for h in horizons:
        h_results = [r for r in results if r.get("horizon") == h and not r.get("skipped")]
        if not h_results:
            print(f"{h:>8d} {'—':>9} {'—':>12} {'—':>9} {'—':>8} {'—':>7} {'—':>9} {'—':>5}")
            continue

        best = None
        best_auc = -1
        best_clf = ""
        for r in h_results:
            for cn in clf_names:
                val = r.get(f"{cn}_roc_auc", float("nan"))
                if not np.isnan(val) and val > best_auc:
                    best_auc = val
                    best = r
                    best_clf = cn

        if best:
            acc = best.get(f"{best_clf}_accuracy", float("nan"))
            pr_auc = best.get(f"{best_clf}_pr_auc", float("nan"))
            print(f"{h:>8d} {best['threshold']:>9.3f} {best_clf:>12} {acc:>9.3f} {best_auc:>8.3f} "
                  f"{pr_auc:>7.3f} {best['baseline']:>9.3f} {best['n']:>5d}")

    print(f"{'='*120}")


def main():
    parser = argparse.ArgumentParser(description="Threshold grid search for classification")
    parser.add_argument("--mode", default="all",
                        help="Encoder preset (minilm, roberta, mpnet, eurobert, bow), "
                             "'all' for all presets, or a full model path "
                             "(e.g. sentence-transformers/all-mpnet-base-v2)")
    args = parser.parse_args()

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

    # --- Build dataset (shared across all modes) ---
    records = build_records(all_tickers, letter_texts)
    print(f"\nTotal samples with letter + returns: {len(records)}")
    if len(records) < 20:
        print("Not enough samples. Exiting.")
        return

    # --- Run grid search ---
    if args.mode == "all":
        modes = list(ENCODER_PRESETS.keys()) + ["bow"]
    else:
        modes = [args.mode]
    for mode in modes:
        run_grid_search(mode, records, OUTPUT_DIR)


if __name__ == "__main__":
    main()
