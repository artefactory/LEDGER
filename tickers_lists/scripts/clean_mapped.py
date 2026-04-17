import csv
import glob
import os

INVALID = {"N/A", "Error", ""}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPED_DIR = os.path.join(ROOT, "mapped")
CLEANED_DIR = os.path.join(ROOT, "cleaned")


def clean_file(path: str) -> int:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and not any(cell.strip() in INVALID for cell in r)]

    name = os.path.basename(path).replace("_mapped.csv", "_mapped_clean.csv")
    out_path = os.path.join(CLEANED_DIR, name)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    os.makedirs(CLEANED_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(MAPPED_DIR, "*_mapped.csv")))

    for path in files:
        kept = clean_file(path)
        print(f"{os.path.basename(path)}: kept {kept} rows")


if __name__ == "__main__":
    main()
