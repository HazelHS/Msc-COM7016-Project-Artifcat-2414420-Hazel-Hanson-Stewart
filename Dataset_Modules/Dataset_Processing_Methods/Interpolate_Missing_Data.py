# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Interpolate_Missing_Data.py fills gaps in a dataset CSV using linear interpolation (limit = 5
consecutive missing values per gap) and saves the result alongside
the original file.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

def main() -> None: # (Anthropic, 2026)
    """Fills missing values in a CSV file using linear interpolation.

    Reads the dataset specified by the ``--dataset`` CLI argument and applies
    linear interpolation to every numeric column, filling gaps of up to 5
    consecutive missing values in either direction. Prints a per-column
    summary of how many missing values were filled and how many remain, then
    writes the result to a new file named ``<stem>_interpolated.csv`` in the
    same directory as the input.

    Gaps longer than 5 consecutive missing values will only be partially
    filled from their edges; remaining NaN cells are preserved in the output.

    Raises:
        SystemExit: If the input file does not exist, cannot be parsed as CSV,
          or the output file cannot be written.
    """
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

    # Interpolation
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

    # Save
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
