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
OUTPUT_DIR = HERE / "output"
OUTPUT_JSON = OUTPUT_DIR / "sentiments.json"
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
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# KPI-based performance evaluation
# ---------------------------------------------------------------------------

KPI_EXTRACTIONS_DIR = HERE / "llm_benchmark" / "output" / "raw"
KPI_PERFORMANCE_JSON = OUTPUT_DIR / "kpi_performance.json"


class Performance(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class PerformanceResponse(BaseModel):
    performance: Performance


def load_kpis_for_report(ticker: str, year: int, exchange: str | None = None) -> dict | None:
    """Load LLM-extracted KPIs for (ticker, year) from the benchmark output.
    
    Returns the extraction dict or None if not available.
    """
    # Try with exchange prefix if provided
    if exchange:
        path = KPI_EXTRACTIONS_DIR / f"{exchange}_{ticker}_{year}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("status") == "ok" and data.get("extraction"):
                return data["extraction"]
    # Otherwise scan for any matching file
    pattern = f"*_{ticker}_{year}.json"
    matches = sorted(KPI_EXTRACTIONS_DIR.glob(pattern))
    for m in matches:
        data = json.loads(m.read_text())
        if data.get("status") == "ok" and data.get("extraction"):
            return data["extraction"]
    return None


def format_kpis_for_prompt(extraction: dict, year: int) -> str:
    """Format KPI extraction into a readable table for the LLM prompt."""
    kpis = extraction.get("kpis", [])
    currency = extraction.get("reporting_currency", "USD")
    ticker = extraction.get("ticker", "?")

    # Filter KPIs for the target year and year-1 (for YoY comparison)
    current_year = [k for k in kpis if k.get("fiscal_year") == year]
    prev_year = [k for k in kpis if k.get("fiscal_year") == year - 1]

    lines = [f"Company: {ticker} | Currency: {currency} | Fiscal Year: {year}"]
    lines.append("")
    lines.append(f"{'KPI':<30} {'FY' + str(year):<20} {'FY' + str(year-1):<20}")
    lines.append("-" * 70)

    # Build lookup for previous year
    prev_lookup = {k["kpi"]: k["value"] for k in prev_year}

    for k in current_year:
        kpi_name = k["kpi"]
        val = k["value"]
        prev_val = prev_lookup.get(kpi_name)
        val_str = f"{val:,.0f}" if val is not None else "N/A"
        prev_str = f"{prev_val:,.0f}" if prev_val is not None else "N/A"
        lines.append(f"{kpi_name:<30} {val_str:<20} {prev_str:<20}")

    return "\n".join(lines)


def classify_kpi_performance(ticker: str, year: int, exchange: str | None = None) -> Performance | None:
    """Ask the LLM to judge company performance based on extracted KPIs."""
    extraction = load_kpis_for_report(ticker, year, exchange)
    if extraction is None:
        return None

    kpi_table = format_kpis_for_prompt(extraction, year)
    if not kpi_table.strip():
        return None

    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-9B",
        messages=[
            {"role": "system", "content": (
                "You are a financial analyst evaluating a company's annual performance. "
                "You are given the key financial indicators (KPIs) extracted from the company's annual report. "
                "Based ONLY on these numbers, classify whether the company had a good or bad year. "
                "Rules:\n"
                "- positive: revenue grew, profits improved, margins stable or better, no major losses\n"
                "- negative: revenue declined, net loss, operating income dropped significantly, "
                "impairments, or clear deterioration in key metrics\n"
                "- neutral: genuinely mixed — some metrics up, others down, no clear overall direction\n"
                "Focus on year-over-year change when both years are available. "
                "If only one year is available, judge absolute performance (profitable = positive, loss = negative). "
                "Respond in JSON."
            )},
            {"role": "user", "content": kpi_table},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "performance_response",
                "schema": PerformanceResponse.model_json_schema(),
                "strict": True,
            },
        },
    )
    result = PerformanceResponse.model_validate_json(response.choices[0].message.content)
    return result.performance


def run_kpi_performance() -> dict:
    """Classify KPI-based performance for all reports with LLM extractions."""
    with open(COMPANIES_JSON) as f:
        companies = json.load(f)

    # Load existing results to resume
    results: dict = {}
    if KPI_PERFORMANCE_JSON.exists():
        with open(KPI_PERFORMANCE_JSON) as f:
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
                    perf = classify_kpi_performance(ticker, year, exchange)
                    results[industry][ticker][str(year)] = perf.value if perf else None
                    status = perf.value if perf else "no KPIs"
                    print(f"[{done}/{total}] {ticker} {year}: {status}")

                # Save after each ticker (resume-safe)
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(KPI_PERFORMANCE_JSON, "w") as f:
                    json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("=== CEO letter sentiment classification ===")
    run_all_industries()
    print("\n=== KPI-based performance classification ===")
    run_kpi_performance()


