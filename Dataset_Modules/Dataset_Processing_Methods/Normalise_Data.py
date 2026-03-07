# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Normalise_Data.py normalises all numeric columns in a dataset CSV using Min-Max scaling
so every value falls in the range [0, 1], then saves the result alongside
the original file.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

def main() -> None: # (Anthropic, 2026)
    """Normalises numeric columns in a CSV file using Min-Max scaling.

    Reads the dataset specified by the ``--dataset`` CLI argument, applies
    Min-Max normalisation to every numeric column so that all values fall in
    [0, 1], prints a per-column summary and a post-normalisation range check,
    then writes the result to a new file named ``<stem>_normalised.csv`` in the
    same directory as the input.

    Constant columns (min == max) are set to 0.0 rather than producing a
    division-by-zero error.

    Raises:
        SystemExit: If the input file does not exist, cannot be parsed as CSV,
          or the output file cannot be written.
    """
    parser = argparse.ArgumentParser(description="Min-Max normalise numeric columns in a dataset CSV.")
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

    print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("No numeric columns found. Normalisation skipped.")
        sys.exit(0)

    df_out = df.copy()

    print("=" * 60)
    print("MIN-MAX NORMALISATION SUMMARY")
    print("=" * 60)
    print(f"{'Column':<40}  {'Min':>12}  {'Max':>12}  {'Status'}")
    print("-" * 72)

    for col in numeric_cols:
        col_min = df_out[col].min()
        col_max = df_out[col].max()
        span    = col_max - col_min
        if span == 0:
            df_out[col] = 0.0
            status = "constant -> set to 0"
        else:
            df_out[col] = (df_out[col] - col_min) / span
            status = "normalised"
        print(f"  {col:<38}  {col_min:>12.4f}  {col_max:>12.4f}  {status}")

    # Validation
    print(f"\nPost-normalisation range check (numeric columns):")
    print(df_out[numeric_cols].describe().loc[["min", "max"]].to_string())

    # Save
    out_path = input_path.parent / (input_path.stem + "_normalised.csv")
    try:
        df_out.to_csv(out_path)
        print(f"\nNormalised dataset saved -> {out_path}")
        print(f"Shape: {df_out.shape[0]} rows x {df_out.shape[1]} columns")
    except Exception as exc:
        print(f"[ERROR] Could not save output: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Normalise Data complete ===")

if __name__ == "__main__":
    main()
