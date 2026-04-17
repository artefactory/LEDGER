import os
import time

import pandas as pd
import yfinance as yf

# --- Configuration ---
EXCHANGE = "NASDAQ"
SLEEP_TIME = 1  # Seconds between requests, to respect rate limits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(ROOT, "tickers", f"{EXCHANGE}_tickers.txt")
OUTPUT_DIR = os.path.join(ROOT, "mapped")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{EXCHANGE}_mapped.csv")


def load_tickers(filename: str) -> list[str]:
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return []

    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip()]


def map_tickers() -> None:
    tickers = load_tickers(INPUT_FILE)
    total = len(tickers)

    if total == 0:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Found {total} tickers. Starting safe data fetch...")

    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{total}] Fetching {ticker}...", end=" ")

        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", "N/A")
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            status = "Success"
        except Exception:
            company_name = "Error"
            sector = "Error"
            industry = "Error"
            status = "Failed"

        df_row = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Company Name": company_name,
                    "Sector": sector,
                    "Industry": industry,
                }
            ]
        )

        write_header = not os.path.exists(OUTPUT_FILE)
        df_row.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)

        print(f"- {status}")
        time.sleep(SLEEP_TIME)

    print(f"\nFinished! All data safely saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    map_tickers()
