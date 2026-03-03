"""
Interpolate_Missing_Data.py
---------------------------
Fills gaps in a dataset CSV using linear interpolation (limit = 5
consecutive missing values per gap) and saves the result alongside
the original file as ``<stem>_interpolated.csv``.

Usage (standalone):
    python Interpolate_Missing_Data.py --dataset <path_to_csv>

When launched from the Model Designer GUI the --dataset argument is
populated automatically from the dropdown selection.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate missing values in a dataset CSV.")
    parser.add_argument("--dataset", required=True, help="Absolute path to the input CSV file.")
    args = parser.parse_args()

    input_path = Path(args.dataset)
    print(f"Loading dataset: {input_path}")
    if not input_path.is_file():
        print(f"[ERROR] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    except Exception as exc:
        print(f"[ERROR] Could not load CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDataset shape (before): {df.shape[0]} rows x {df.shape[1]} columns")

    missing_before = df.isna().sum().sum()
    print(f"Total missing values before interpolation: {missing_before}")

    # ── Interpolation ─────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("No numeric columns to interpolate. Exiting.")
        sys.exit(0)

    df_out = df.copy()
    for col in numeric_cols:
        before = df_out[col].isna().sum()
        df_out[col] = df_out[col].interpolate(method="linear", limit=5, limit_direction="both")
        after = df_out[col].isna().sum()
        filled = before - after
        if before > 0:
            print(f"  {col:40s}  {before:5d} missing  ->  {filled:5d} filled  ({after} remain)")

    missing_after = df_out.isna().sum().sum()
    print(f"\nTotal missing values after  interpolation: {missing_after}")
    print(f"Cells filled: {missing_before - missing_after}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = input_path.parent / (input_path.stem + "_interpolated.csv")
    try:
        df_out.to_csv(out_path)
        print(f"\nInterpolated dataset saved -> {out_path}")
        print(f"Shape: {df_out.shape[0]} rows x {df_out.shape[1]} columns")
    except Exception as exc:
        print(f"[ERROR] Could not save output: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Interpolate Missing Data complete ===")


if __name__ == "__main__":
    main()
