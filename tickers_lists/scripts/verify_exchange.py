"""Add the Yahoo exchange field to an existing cleaned CSV.

For each row in `cleaned/{EXCHANGE}_mapped_clean.csv`, queries yfinance only
to fetch `fullExchangeName`, and writes
`cleaned/{EXCHANGE}_mapped_clean_verified.csv` with one extra column. This
lets you detect rows where yfinance silently redirected a ticker to a
different exchange (e.g. the LSE → NYSE resolutions we saw).

Resumable: on restart, tickers already present in the output file are
skipped, so interruptions don't cost progress. 1 s sleep between requests.

Usage:
    uv run python tickers_lists/scripts/verify_exchange.py LSE
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import yfinance as yf

SLEEP_TIME = 1
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(ROOT, "cleaned")


def already_done_tickers(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    with open(output_path, newline="") as f:
        return {row["Ticker"] for row in csv.DictReader(f)}


def verify(exchange: str) -> int:
    input_path = os.path.join(CLEANED_DIR, f"{exchange}_mapped_clean.csv")
    output_path = os.path.join(
        CLEANED_DIR, f"{exchange}_mapped_clean_verified.csv"
    )

    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"{exchange}: input is empty")
        return 0

    done = already_done_tickers(output_path)
    todo = [r for r in rows if r["Ticker"] not in done]

    print(
        f"{exchange}: {len(rows)} rows total, "
        f"{len(done)} already verified, {len(todo)} to fetch"
    )

    if not todo:
        return 0

    fieldnames = list(rows[0].keys()) + ["Exchange (Yahoo)"]
    write_header = not os.path.exists(output_path)

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        for i, row in enumerate(todo, 1):
            ticker = row["Ticker"]
            print(f"  [{i}/{len(todo)}] {ticker}...", end=" ", flush=True)
            try:
                info = yf.Ticker(ticker).info
                yahoo_exchange = (
                    info.get("fullExchangeName") or info.get("exchange") or "N/A"
                )
            except Exception:
                yahoo_exchange = "Error"
            print(yahoo_exchange)

            writer.writerow({**row, "Exchange (Yahoo)": yahoo_exchange})
            f.flush()
            time.sleep(SLEEP_TIME)

    print(f"{exchange}: wrote {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "exchange",
        help="Exchange label matching {EXCHANGE}_mapped_clean.csv (e.g. LSE).",
    )
    sys.exit(verify(parser.parse_args().exchange))
