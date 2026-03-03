"""
Plot_Distribution_Outliers.py
------------------------------
Generates violin plots (with embedded box plots) for every numeric column
in a dataset CSV, overlaying red scatter points for values that are more
than 3 standard deviations from the column mean.

Opens the chart in an interactive window.  The interactive matplotlib
window includes a save button (floppy-disk icon) in the toolbar — click
it to export the figure to any folder and format you choose.

An optional --output-dir argument will also auto-save the figure to
that directory as ``distribution_outliers.png`` before the window opens.

Usage (standalone):
    python Plot_Distribution_Outliers.py --dataset <path_to_csv>
    python Plot_Distribution_Outliers.py --dataset <path_to_csv> --output-dir <path_to_folder>

When launched from the Model Designer GUI the --dataset argument is
populated automatically from the dropdown selection.

References:
    Anthropic. (2024). Claude (claude-sonnet) [Large language model].
    https://www.anthropic.com
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot violin/outlier distributions for each column in a dataset CSV."
    )
    parser.add_argument(
        "--dataset", required=True, help="Absolute path to the input CSV file."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional folder to auto-save the PNG. If omitted, use the toolbar save button.",
    )
    args = parser.parse_args()

    input_path = Path(args.dataset)
    if not input_path.is_file():
        print(f"[ERROR] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {input_path}")
    try:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    except Exception as exc:
        print(f"[ERROR] Could not load CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        print("[ERROR] No numeric columns found.", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Plotting distributions for {len(numeric_df.columns)} numeric column(s)...")

    # ── Build plot ────────────────────────────────────────────────────
    n_cols = len(numeric_df.columns)
    n_rows = (n_cols + 2) // 3   # arrange in rows of 3 plots

    fig = plt.figure(figsize=(18, n_rows * 5))
    gs = GridSpec(n_rows, 3, figure=fig)

    for i, column in enumerate(numeric_df.columns):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])

        sns.violinplot(y=numeric_df[column], ax=ax, inner="box", color="skyblue")
        ax.set_title(f"Distribution of {column}", fontsize=12)
        ax.set_ylabel("Value")

        # scatter points for outliers (> 3 std deviations from mean)
        outliers = numeric_df[
            abs(numeric_df[column] - numeric_df[column].mean()) > (3 * numeric_df[column].std())
        ]
        if not outliers.empty:
            sns.stripplot(y=outliers[column], ax=ax, color="red", size=4, jitter=True)
            print(f"  {column}: {len(outliers)} outlier(s) highlighted in red.")

    plt.tight_layout()

    # ── Optional auto-save ────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "distribution_outliers.png"
        try:
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved -> {out_path}")
        except Exception as exc:
            print(f"[WARNING] Could not save figure: {exc}", file=sys.stderr)
    else:
        print(
            "Tip: use the save icon in the toolbar to export, "
            "or pass --output-dir to auto-save."
        )

    plt.show()
    print("=== Plot Distribution & Outliers complete ===")


if __name__ == "__main__":
    main()
