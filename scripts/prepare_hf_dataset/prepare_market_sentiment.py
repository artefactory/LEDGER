"""
Prepare the MarketSentimentPredictionData for HuggingFace upload.

Output layout (one HF Dataset repository):
    hf_output/market_sentiment/
    ├── README.md                          # dataset card (generated)
    ├── letters/
    │   └── data.parquet                   # CEO letters with sentiment labels
    ├── eps_surprise/
    │   └── data.parquet                   # per-ticker earnings surprise data
    ├── stock_prices/
    │   └── data-NNNNN-of-NNNNN.parquet    # per-ticker daily prices (sharded)
    └── industry_indicators/
        └── data.parquet                   # per-industry daily aggregate indicators

Four configs:
  - letters:              one row per CEO letter (~465 letters)
  - eps_surprise:         one row per earnings event (~5k events)
  - stock_prices:         one row per ticker-day (~600k rows, sharded)
  - industry_indicators:  one row per industry-day (~15k rows)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "MarketSentimentPredictionData"
DEFAULT_OUTPUT = REPO_ROOT / "hf_output" / "market_sentiment"

# ── Friendly industry names for the industry_indicators filenames ─────────────
INDUSTRY_SLUG_MAP = {
    "Basic_Materials___Specialty_Chemicals": "Basic Materials / Specialty Chemicals",
    "Consumer_Cyclical___Auto_Parts": "Consumer Cyclical / Auto Parts",
    "Consumer_Defensive___Packaged_Foods": "Consumer Defensive / Packaged Foods",
    "Energy___Oil_&_Gas_E&P": "Energy / Oil & Gas E&P",
    "Energy___Oil_&_Gas_Equipment_&_Services": "Energy / Oil & Gas Equipment & Services",
    "Real_Estate___REIT_-_Mortgage": "Real Estate / REIT - Mortgage",
}

# ── Letters config ────────────────────────────────────────────────────────────

# Filename pattern: {EX}_{TICKER}_{YEAR}__{NN}_{slug}.md
# The double underscore separates report ID from letter metadata.
LETTER_FILENAME_RE = re.compile(
    r"^(?P<exchange>[A-Z]+)_(?P<ticker>.+?)_(?P<year>\d{4})__(?P<num>\d+)_(?P<slug>.+)\.md$"
)


def build_letters_config(source_dir: Path, output_dir: Path) -> int:
    """Build the letters config parquet. Returns row count."""
    letters_dir = source_dir / "letter_of_ceo"
    sentiments_path = letters_dir / "sentiments.json"

    # Load sentiments: industry -> ticker -> year -> label
    with sentiments_path.open(encoding="utf-8") as f:
        sentiments_raw = json.load(f)

    # Build a flat lookup: (ticker, year_str) -> (industry, sentiment)
    sentiment_lookup: dict[tuple[str, str], tuple[str, str | None]] = {}
    for industry, tickers in sentiments_raw.items():
        for ticker, years in tickers.items():
            for year_str, label in years.items():
                sentiment_lookup[(ticker, year_str)] = (industry, label)

    # Parse all .md files
    rows = []
    md_files = sorted(letters_dir.glob("*.md"))
    for md_path in tqdm(md_files, desc="Reading CEO letters", unit="file"):
        m = LETTER_FILENAME_RE.match(md_path.name)
        if m is None:
            continue

        exchange = m.group("exchange")
        ticker = m.group("ticker")
        year = int(m.group("year"))
        letter_number = int(m.group("num"))
        slug = m.group("slug")

        # Read file content
        content = md_path.read_text(encoding="utf-8")

        # Extract title from first heading line
        title = slug.replace("_", " ")
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        # Extract source citation and page range
        source_report = ""
        start_page = None
        end_page = None
        source_match = re.search(
            r"_Source:\s*`(.+?)`\s*·\s*pages\s*(\d+)-(\d+)_", content
        )
        if source_match:
            source_report = source_match.group(1)
            start_page = int(source_match.group(2))
            end_page = int(source_match.group(3))

        # Look up sentiment and industry
        industry, sentiment = sentiment_lookup.get((ticker, str(year)), ("", None))

        rows.append(
            {
                "ticker": ticker,
                "exchange": exchange,
                "industry": industry,
                "year": year,
                "letter_number": letter_number,
                "title": title,
                "start_page": start_page,
                "end_page": end_page,
                "sentiment": sentiment,
                "text": content,
                "source_report": source_report,
            }
        )

    df = pd.DataFrame(rows)
    # Sort for deterministic output
    df = df.sort_values(["industry", "ticker", "year", "letter_number"]).reset_index(
        drop=True
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "data.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  letters: {len(df):,} rows → {parquet_path}")
    return len(df)


# ── EPS surprise config ───────────────────────────────────────────────────────


def build_eps_surprise_config(source_dir: Path, output_dir: Path) -> int:
    """Build the eps_surprise config parquet. Returns row count."""
    eps_dir = source_dir / "eps_surprise"
    frames = []
    for csv_path in tqdm(
        sorted(eps_dir.glob("*.csv")), desc="Reading EPS surprise CSVs", unit="file"
    ):
        ticker = csv_path.stem  # e.g. "AAP" or "ELM.L"
        df = pd.read_csv(csv_path)
        df.insert(0, "ticker", ticker)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Normalize column names
    combined.columns = [c.strip() for c in combined.columns]
    # Parse date
    combined["earnings_date"] = pd.to_datetime(
        combined["earnings_date"], errors="coerce"
    )
    # Sort
    combined = combined.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "data.parquet"
    combined.to_parquet(parquet_path, index=False)
    print(f"  eps_surprise: {len(combined):,} rows → {parquet_path}")
    return len(combined)


# ── Stock prices config (sharded) ─────────────────────────────────────────────

ROWS_PER_SHARD = 100_000


def build_stock_prices_config(source_dir: Path, output_dir: Path) -> int:
    """Build the stock_prices config as sharded parquets. Returns row count."""
    prices_dir = source_dir / "stock_prices_enhanced"
    frames = []
    for csv_path in tqdm(
        sorted(prices_dir.glob("*.csv")), desc="Reading stock price CSVs", unit="file"
    ):
        ticker = csv_path.stem
        df = pd.read_csv(csv_path)
        df.insert(0, "ticker", ticker)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [c.strip() for c in combined.columns]
    # Parse date
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    # Sort
    combined = combined.sort_values(["ticker", "Date"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write sharded parquet
    n_shards = max(1, (len(combined) + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD)
    for i in range(n_shards):
        shard = combined.iloc[i * ROWS_PER_SHARD : (i + 1) * ROWS_PER_SHARD]
        shard_path = output_dir / f"data-{i:05d}-of-{n_shards:05d}.parquet"
        shard.to_parquet(shard_path, index=False)

    print(f"  stock_prices: {len(combined):,} rows → {n_shards} shards in {output_dir}")
    return len(combined)


# ── Industry indicators config ────────────────────────────────────────────────


def build_industry_indicators_config(source_dir: Path, output_dir: Path) -> int:
    """Build the industry_indicators config parquet. Returns row count."""
    ind_dir = source_dir / "industry_indicators"
    frames = []
    for csv_path in tqdm(
        sorted(ind_dir.glob("*.csv")),
        desc="Reading industry indicator CSVs",
        unit="file",
    ):
        slug = csv_path.stem
        industry = INDUSTRY_SLUG_MAP.get(slug, slug)
        df = pd.read_csv(csv_path)
        df.insert(0, "industry", industry)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [c.strip() for c in combined.columns]
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.sort_values(["industry", "Date"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "data.parquet"
    combined.to_parquet(parquet_path, index=False)
    print(f"  industry_indicators: {len(combined):,} rows → {parquet_path}")
    return len(combined)


# ── Dataset card ──────────────────────────────────────────────────────────────


def build_readme(
    output_dir: Path,
    n_letters: int,
    n_eps: int,
    n_prices: int,
    n_indicators: int,
):
    """Write a HuggingFace dataset card README.md."""
    readme = f"""---
configs:
- config_name: letters
  data_files:
  - split: train
    path: letters/data.parquet
- config_name: eps_surprise
  data_files:
  - split: train
    path: eps_surprise/data.parquet
- config_name: stock_prices
  data_files:
  - split: train
    path: stock_prices/data-*.parquet
- config_name: industry_indicators
  data_files:
  - split: train
    path: industry_indicators/data.parquet
task_categories:
- text-classification
- time-series-forecasting
language:
- en
tags:
- finance
- market-sentiment
- ceo-letters
- earnings
- stock-prices
- annual-reports
license: cc-by-4.0
---

# LEDGER Market Sentiment Prediction Data

Data used for the market sentiment prediction case study in the LEDGER paper,
linking CEO-letter rhetoric to EPS surprises and post-publication market reactions.

## Dataset Description

This dataset supports research on whether the rhetoric in corporate annual report
CEO letters carries signal about future fundamentals and market reaction. It covers
six highly liquid industries (specialty chemicals, auto parts, packaged foods, oil &
gas E&P, oil & gas equipment & services, and mortgage REITs) spanning fiscal years
2017–2022.

## Configs

| Config | Rows | Description |
|--------|------|-------------|
| `letters` | {n_letters:,} | CEO/chairman letters extracted from annual reports, with sentiment labels |
| `eps_surprise` | {n_eps:,} | Earnings per share surprise data (consensus vs. reported) |
| `stock_prices` | {n_prices:,} | Daily stock prices with rolling return windows (t-90 to t+90) |
| `industry_indicators` | {n_indicators:,} | Aggregate industry-level daily indicators |

## Schema

### `letters`

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Stock ticker symbol |
| `exchange` | string | Stock exchange (NYSE, NASDAQ, LSE, AMEX) |
| `industry` | string | Industry classification |
| `year` | int | Fiscal year of the annual report |
| `letter_number` | int | Letter index within the report (1-based) |
| `title` | string | Letter heading (e.g. "Chairman's Statement") |
| `start_page` | int | First page in the source report |
| `end_page` | int | Last page in the source report |
| `sentiment` | string | Overall sentiment label: `positive`, `negative`, or null |
| `text` | string | Full letter text (Markdown) |
| `source_report` | string | Path to the source OCR'd report |

### `eps_surprise`

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Stock ticker symbol |
| `earnings_date` | datetime | Earnings announcement date (UTC) |
| `eps_estimate` | float | Consensus analyst EPS estimate |
| `reported_eps` | float | Actual reported EPS (null for future dates) |
| `surprise_pct` | float | Surprise percentage: (actual - estimate) / estimate × 100 |

### `stock_prices`

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Stock ticker symbol |
| `Date` | date | Trading date |
| `Open` | float | Opening price |
| `High` | float | Daily high |
| `Low` | float | Daily low |
| `Close` | float | Closing price |
| `Volume` | float | Trading volume |
| `Volume_ATS` | float | Alternative trading system volume (normalized) |
| `returns` | float | Daily return |
| `Volatility` | float | Rolling volatility |
| `return_t-90` … `return_t90` | float | Rolling returns from t-90 to t+90 trading days |

### `industry_indicators`

| Column | Type | Description |
|--------|------|-------------|
| `industry` | string | Industry name |
| `Date` | date | Trading date |
| `returns` | float | Equal-weighted industry return |
| `volumes` | float | Normalized aggregate volume |
| `volatility` | float | Industry volatility |
| `returns_vw` | float | Value-weighted industry return |
| `volumes_vw` | float | Value-weighted aggregate volume |
| `volatility_vw` | float | Value-weighted volatility |
| `return_t-90_vw` … `return_t42_vw` | float | Value-weighted rolling returns (t-90 to t+42) |

## Usage

```python
from datasets import load_dataset

# Load CEO letters with sentiment
letters = load_dataset("artefactory/LEDGER-market-sentiment", "letters")

# Load EPS surprises
eps = load_dataset("artefactory/LEDGER-market-sentiment", "eps_surprise")

# Load daily stock prices
prices = load_dataset("artefactory/LEDGER-market-sentiment", "stock_prices")

# Load industry indicators
indicators = load_dataset("artefactory/LEDGER-market-sentiment", "industry_indicators")

# Example: filter to positive-sentiment letters
positive_letters = letters["train"].filter(lambda x: x["sentiment"] == "positive")
```

## Industries Covered

1. Basic Materials / Specialty Chemicals
2. Consumer Cyclical / Auto Parts
3. Consumer Defensive / Packaged Foods
4. Energy / Oil & Gas E&P
5. Energy / Oil & Gas Equipment & Services
6. Real Estate / REIT - Mortgage

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{{moslonka2026ledger,
  title={{LEDGER: A Long-Context Benchmark of Corporate Annual Reports for Grounded Financial Retrieval and Extraction}},
  author={{Moslonka, Charles and de Vitry, Amaury and Garnier, Arthur and Randrianarivo, Hicham and Malherbe, Emmanuel}},
  booktitle={{Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM)}},
  year={{2026}}
}}
```

## License

CC-BY-4.0
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"  Wrote {output_dir / 'README.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MarketSentimentPredictionData for HuggingFace upload."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to MarketSentimentPredictionData/ directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for the HF dataset",
    )
    parser.add_argument(
        "--skip-letters", action="store_true", help="Skip letters config"
    )
    parser.add_argument(
        "--skip-eps", action="store_true", help="Skip eps_surprise config"
    )
    parser.add_argument(
        "--skip-prices", action="store_true", help="Skip stock_prices config"
    )
    parser.add_argument(
        "--skip-indicators", action="store_true", help="Skip industry_indicators config"
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir

    if not source_dir.is_dir():
        print(f"ERROR: source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source:  {source_dir}")
    print(f"Output:  {output_dir}")

    n_letters = n_eps = n_prices = n_indicators = 0

    if not args.skip_letters:
        print(f"\n{'=' * 60}")
        print("Building config: letters")
        print(f"{'=' * 60}")
        n_letters = build_letters_config(source_dir, output_dir / "letters")

    if not args.skip_eps:
        print(f"\n{'=' * 60}")
        print("Building config: eps_surprise")
        print(f"{'=' * 60}")
        n_eps = build_eps_surprise_config(source_dir, output_dir / "eps_surprise")

    if not args.skip_prices:
        print(f"\n{'=' * 60}")
        print("Building config: stock_prices")
        print(f"{'=' * 60}")
        n_prices = build_stock_prices_config(source_dir, output_dir / "stock_prices")

    if not args.skip_indicators:
        print(f"\n{'=' * 60}")
        print("Building config: industry_indicators")
        print(f"{'=' * 60}")
        n_indicators = build_industry_indicators_config(
            source_dir, output_dir / "industry_indicators"
        )

    print(f"\n{'=' * 60}")
    print("Building README.md")
    print(f"{'=' * 60}")
    build_readme(output_dir, n_letters, n_eps, n_prices, n_indicators)

    print(f"\nDone. Output at: {output_dir}")


if __name__ == "__main__":
    main()
