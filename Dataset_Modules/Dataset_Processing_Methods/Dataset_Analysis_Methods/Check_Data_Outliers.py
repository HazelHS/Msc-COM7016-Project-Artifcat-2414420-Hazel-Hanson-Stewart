# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Check_Data_Outliers.py script detects and reports statistical outliers in a dataset CSV using the
Inter-Quartile Range (IQR) method.
"""

import argparse
import sys
import pandas as pd

def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments and detect IQR-based outliers in the CSV.

    Loads the CSV at --dataset and prints two sections to stdout: a
    per-column summary table showing Q1, Q3, IQR, lower and upper fences,
    and counts of values below and above each fence; followed by the
    first 20 outlier rows for each column that contains at least one
    outlier.  Outliers are defined as values outside Q1 - 1.5*IQR or
    Q3 + 1.5*IQR.
    """
    parser = argparse.ArgumentParser(description="Detect outliers via IQR method.")
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

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("No numeric columns found. Outlier detection skipped.")
        return

    print("=" * 60)
    print("OUTLIER SUMMARY  (IQR method, multiplier = 1.5)")
    print("=" * 60)

    total_outliers = 0
    report_rows: list[dict] = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1  = series.quantile(0.25)
        q3  = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        n_low  = int((series < lower).sum())
        n_high = int((series > upper).sum())
        n_out  = n_low + n_high
        total_outliers += n_out

        report_rows.append({
            "Column" : col,
            "Q1"     : round(q1,  4),
            "Q3"     : round(q3,  4),
            "IQR"    : round(iqr, 4),
            "Lower fence" : round(lower, 4),
            "Upper fence" : round(upper, 4),
            "# Below fence": n_low,
            "# Above fence": n_high,
            "Total outliers": n_out,
        })

    summary = pd.DataFrame(report_rows).set_index("Column")
    print(summary.to_string())
    print(f"\nTotal outlier cells detected: {total_outliers}")

    # Per-column listing for columns that have outliers
    flagged = summary[summary["Total outliers"] > 0].index.tolist()
    if flagged:
        print("\n" + "=" * 60)
        print("OUTLIER ROWS (first 20 per column)")
        print("=" * 60)
        for col in flagged:
            series   = df[col].dropna()
            q1  = series.quantile(0.25)
            q3  = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask    = (df[col] < lower) | (df[col] > upper)
            print(f"\n--- {col} ---")
            print(df.loc[mask, [col]].head(20).to_string())
    else:
        print("\nNo outliers detected in any numeric column.")

    print("\n=== Check Data Outliers complete ===")

if __name__ == "__main__":
    main()
