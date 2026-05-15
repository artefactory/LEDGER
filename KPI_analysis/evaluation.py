from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel
from enum import Enum
import json
import sys


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

HERE = Path(__file__).resolve().parent
CEO_LETTERS_DIR = HERE.parent / "doc_text_processing" / "CEO_word_extraction" / "cleaning_extractions" / "cleaned"
COMPANIES_JSON = HERE.parent / "tickers_lists" / "grouped" / "selected" / "companies.json"
OUTPUT_JSON = CEO_LETTERS_DIR / "sentiments.json"
YEARS = range(2017, 2023)


def load_ceo_letter(ticker: str, year: int) -> str | None:
    """Return the markdown text of the first CEO letter for (ticker, year), or None."""
    pattern = f"*_{ticker}_{year}__*.md"
    matches = sorted(CEO_LETTERS_DIR.glob(pattern)) # retourne le nom du doc qui  match avec le pattern
    if not matches:
        return None
    return matches[0].read_text()


class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class SentimentResponse(BaseModel):
    sentiment: Sentiment


def classify_sentiment(ticker: str, year: int) -> Sentiment | None:
    """Return the sentiment of the CEO letter for (ticker, year), or None if no letter."""
    ceo_letter = load_ceo_letter(ticker, year)
    if ceo_letter is None:
        return None
    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-9B",
        messages=[
            {"role": "system", "content": (
                "You are a financial analyst evaluating a company's annual performance. "
                "Read the CEO letter below and classify the company's FINANCIAL RESULTS for that year. "
                "Ignore the optimistic tone — CEO letters are always written positively, even in bad years. "
                "Focus strictly on the reported numbers: revenue, earnings, profit, margins, cash flow, debt. "
                "Pay attention to hedging language: if the CEO emphasizes future improvement, recovery, "
                "or 'turning a corner', that usually signals the current year was poor. "
                "Phrases like 'despite challenges', 'headwinds', 'difficult environment', or 'we expect to do better' "
                "are red flags — classify as negative unless hard numbers clearly show growth. "
                "Rules:\n"
                "- positive: revenue or earnings grew year-over-year, margins improved, or clear financial progress backed by numbers\n"
                "- negative: revenue or earnings declined, losses reported, impairments, restructuring charges, "
                "deteriorating metrics, OR the letter is mostly forward-looking promises without reporting actual good numbers\n"
                "- neutral: genuinely mixed results with some metrics clearly up and others clearly down, backed by specific numbers\n"
                "When in doubt between neutral and negative, lean negative. "
                "Respond in JSON."
            )},
            {"role": "user", "content": ceo_letter},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_response",
                "schema": SentimentResponse.model_json_schema(),
                "strict": True,
            },
        },
    )
    result = SentimentResponse.model_validate_json(response.choices[0].message.content)
    return result.sentiment


def run_all_industries() -> dict:
    """Classify sentiment for all tickers across the 6 selected industries, save to JSON."""
    with open(COMPANIES_JSON) as f:
        companies = json.load(f) # load companies 

    # Load existing results to resume
    results: dict = {}
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON) as f:
            results = json.load(f)

    total = sum(
        len(entries)
        for industry_exchanges in companies.values()
        for entries in industry_exchanges.values()
    )
    done = 0

    for industry, exchanges in companies.items():
        if industry not in results:
            results[industry] = {}

        for exchange, entries in exchanges.items():
            for entry in entries:
                ticker = entry["ticker"]
                done += 1

                if ticker in results[industry]:
                    print(f"[{done}/{total}] {ticker} already done, skipping")
                    continue

                results[industry][ticker] = {}
                for year in YEARS:
                    sentiment = classify_sentiment(ticker, year) 
                    results[industry][ticker][str(year)] = sentiment.value if sentiment else None
                    status = sentiment.value if sentiment else "no letter"
                    print(f"[{done}/{total}] {ticker} {year}: {status}")

                # Save after each ticker (resume-safe)
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_all_industries()


