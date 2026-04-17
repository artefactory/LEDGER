import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(ROOT, "file_list.txt")
OUTPUT_DIR = os.path.join(ROOT, "tickers")


def process_filenames(input_file: str) -> None:
    exchange_data: dict[str, set[str]] = {}

    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found.")
        return

    print(f"Reading from {input_file}...\n")

    with open(input_file, "r") as file:
        for line in file:
            clean_line = line.strip()
            if not clean_line:
                continue

            parts = clean_line.split("_")
            if len(parts) >= 3:
                exchange = parts[0]
                ticker = "_".join(parts[1:-1])
                exchange_data.setdefault(exchange, set()).add(ticker)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for exchange, tickers in exchange_data.items():
        output_path = os.path.join(OUTPUT_DIR, f"{exchange}_tickers.txt")
        with open(output_path, "w") as out_file:
            for ticker in sorted(tickers):
                out_file.write(f"{ticker}\n")
        print(f"Created '{output_path}' containing {len(tickers)} unique tickers.")

    print("\nExtraction complete!")


if __name__ == "__main__":
    process_filenames(INPUT_FILE)
