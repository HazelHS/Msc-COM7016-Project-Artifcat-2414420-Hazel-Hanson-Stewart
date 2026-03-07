# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Plot_Pairplot.py script generates a seaborn pairplot showing pairwise relationships 
between numeric features in a dataset CSV file.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments and render a seaborn pairplot.

    Loads the CSV at --dataset and creates a scatter-matrix for all numeric
    columns.  When the dataset has 10 or fewer numeric columns the full
    pairplot is produced; for larger datasets, features with pairwise
    correlation above 0.8 are dropped and at most 8 columns are kept to
    preserve readability.  The figure is displayed interactively.  If
    --output-dir is supplied the figure is also auto-saved to that
    directory as pairplot.png (or pairplot_selected.png when column
    subsetting is applied) before the window opens.
    """
    parser = argparse.ArgumentParser(
        description="Plot a pairplot for a dataset CSV."
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

    n_cols = len(numeric_df.columns)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Numeric columns to plot: {n_cols}")

    # Column selection
    if n_cols <= 10:
        plot_df = numeric_df
        title = "Pairwise Relationships Between Features"
        filename = "pairplot.png"
        print("Plotting full pairplot (≤10 columns).")
    else:
        print(
            f"{n_cols} columns detected — subsetting to the most distinct features "
            "(dropping columns with pairwise correlation > 0.8, capping at 8)."
        )
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]
        important_cols = list(set(numeric_df.columns) - set(to_drop))
        if len(important_cols) > 8:
            important_cols = important_cols[:8]
        plot_df = numeric_df[important_cols]
        title = "Pairwise Relationships Between Key Features"
        filename = "pairplot_selected.png"
        print(f"Selected {len(important_cols)} column(s): {important_cols}")

    # Build plot
    plt.figure(figsize=(20, 20))
    pair_grid = sns.pairplot(
        plot_df,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 30, "edgecolor": "k"},
    )
    pair_grid.figure.suptitle(title, y=1.02, fontsize=20)
    plt.tight_layout()

    # Optional auto-save 
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        try:
            pair_grid.figure.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved -> {out_path}")
        except Exception as exc:
            print(f"[WARNING] Could not save figure: {exc}", file=sys.stderr)
    else:
        print(
            "Tip: use the save icon in the toolbar to export, "
            "or pass --output-dir to auto-save."
        )

    plt.show()
    print("=== Plot Pairplot complete ===")

if __name__ == "__main__":
    main()
