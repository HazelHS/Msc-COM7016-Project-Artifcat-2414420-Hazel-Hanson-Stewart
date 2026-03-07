# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Plot_Correlation_Matrix.py script generates a lower-triangle correlation-matrix heatmap for 
all numeric features in a dataset CSV file and opens it in an interactive window.

The interactive matplotlib window includes a save button (floppy-disk
icon) in the toolbar.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments and render a correlation matrix heatmap.

    Loads the CSV at --dataset, computes pairwise Pearson correlations
    for all numeric columns, and displays a lower-triangle heatmap with
    annotated coefficient values.  The figure is displayed interactively.
    If --output-dir is supplied the figure is also auto-saved to that
    directory as correlation_matrix.png before the window opens.
    """
    parser = argparse.ArgumentParser(
        description="Plot a correlation matrix heatmap for a dataset CSV."
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

    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        print("[ERROR] No numeric columns found — cannot compute correlation.", file=sys.stderr)
        sys.exit(1)

    # Build plot
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=0.8)

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
    )
    plt.title("Correlation Matrix of Features", fontsize=16)
    plt.tight_layout()

    # Optional auto-save
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "correlation_matrix.png"
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
    print("=== Plot Correlation Matrix complete ===")

if __name__ == "__main__":
    main()
