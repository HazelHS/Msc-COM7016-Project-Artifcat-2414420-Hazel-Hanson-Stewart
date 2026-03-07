# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Plot_Missing_Values_Heatmap.py script generates a heatmap highlighting missing (NaN) cells across all columns
in a dataset CSV file and opens it in an interactive window.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments and render a missing-values heatmap.

    Loads the CSV at --dataset and produces a seaborn boolean heatmap
    where each cell is coloured to indicate whether the corresponding
    value is missing.  The figure is displayed interactively.  If
    --output-dir is supplied the figure is also auto-saved to that
    directory as missing_values_heatmap.png before the window opens.
    """
    parser = argparse.ArgumentParser(
        description="Plot a missing-values heatmap for a dataset CSV."
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
    total_missing = df.isna().sum().sum()
    print(f"Total missing cells: {total_missing}")

    # Build plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isna(), cmap="viridis", cbar_kws={"label": "Missing Values"})
    plt.title("Missing Values Heatmap", fontsize=16)
    plt.tight_layout()

    # Optional auto-save
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "missing_values_heatmap.png"
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
    print("=== Plot Missing Values Heatmap complete ===")

if __name__ == "__main__":
    main()
