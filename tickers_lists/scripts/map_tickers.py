import argparse
import os
import time

import pandas as pd
import yfinance as yf

SLEEP_TIME = 1  # Seconds between requests, to respect rate limits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKERS_DIR = os.path.join(ROOT, "tickers")
OUTPUT_DIR = os.path.join(ROOT, "mapped")


def load_tickers(filename: str) -> list[str]:
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return []

    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip()]


def map_tickers(exchange: str) -> None:
    input_file = os.path.join(TICKERS_DIR, f"{exchange}_tickers.txt")
    output_file = os.path.join(OUTPUT_DIR, f"{exchange}_mapped.csv")

    tickers = load_tickers(input_file)
    total = len(tickers)

    if total == 0:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Found {total} tickers for {exchange}. Starting safe data fetch...")

    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{total}] Fetching {ticker}...", end=" ")

        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", "N/A")
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            yahoo_exchange = info.get("fullExchangeName") or info.get("exchange", "N/A")
            status = "Success"
        except Exception:
            company_name = "Error"
            sector = "Error"
            industry = "Error"
            yahoo_exchange = "Error"
            status = "Failed"

        df_row = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Company Name": company_name,
                    "Sector": sector,
                    "Industry": industry,
                    "Exchange (Yahoo)": yahoo_exchange,
                }
            ]
        )

        write_header = not os.path.exists(output_file)
        df_row.to_csv(output_file, mode="a", header=write_header, index=False)

        print(f"- {status}")
        time.sleep(SLEEP_TIME)

    print(f"\nFinished! All data safely saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich tickers for one exchange with yfinance metadata."
    )
    parser.add_argument(
        "exchange",
        help="Exchange label matching {EXCHANGE}_tickers.txt (e.g. NASDAQ, NYSE, LSE).",
    )
    args = parser.parse_args()
    map_tickers(args.exchange)
