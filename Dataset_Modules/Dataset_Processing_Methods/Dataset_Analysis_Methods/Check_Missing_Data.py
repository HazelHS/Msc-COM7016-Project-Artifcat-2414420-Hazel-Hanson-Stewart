"""
Check_Missing_Data.py
---------------------
Analyses a dataset CSV for missing values and duplicate rows.

Usage (standalone):
    python Check_Missing_Data.py --dataset <path_to_csv>

When launched from the Model Designer GUI the --dataset argument is
populated automatically from the dropdown selection.
"""

import argparse
import sys
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset for missing / duplicate data.")
    parser.add_argument("--dataset", required=True, help="Absolute path to the input CSV file.")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    try:
        df = pd.read_csv(args.dataset, index_col=0, parse_dates=True)
    except Exception as exc:
        print(f"[ERROR] Could not load CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")

    # ── Missing values ────────────────────────────────────────────────
    missing = df.isna().sum()
    total_missing = missing.sum()
    print("=" * 50)
    print("MISSING VALUES BY COLUMN")
    print("=" * 50)
    print(missing.to_string())
    print(f"\nTotal missing cells : {total_missing}")
    pct = total_missing / (df.shape[0] * df.shape[1]) * 100
    print(f"Missing percentage  : {pct:.2f}%")

    # Columns with no missing data
    complete_cols = missing[missing == 0].index.tolist()
    print(f"\nColumns with NO missing data ({len(complete_cols)}): {complete_cols}")

    # Columns with any missing data
    incomplete_cols = missing[missing > 0]
    if not incomplete_cols.empty:
        print("\nColumns WITH missing data:")
        for col, count in incomplete_cols.items():
            pct_col = count / df.shape[0] * 100
            print(f"  {col:40s}  {count:6d} missing  ({pct_col:.1f}%)")
    else:
        print("\nNo missing values found.")

    # ── Duplicates ────────────────────────────────────────────────────
    dup_mask = df.duplicated(keep="first")
    dup_count = dup_mask.sum()
    print("\n" + "=" * 50)
    print("DUPLICATE ROWS")
    print("=" * 50)
    print(f"Duplicate rows (keeping first occurrence): {dup_count}")
    if dup_count > 0:
        print("\nDuplicate rows:")
        print(df[dup_mask].to_string())

    # ── Unique values per column ──────────────────────────────────────
    print("\n" + "=" * 50)
    print("UNIQUE VALUES PER COLUMN")
    print("=" * 50)
    print(df.nunique().to_string())

    print("\n=== Check Missing Data complete ===")


if __name__ == "__main__":
    main()
