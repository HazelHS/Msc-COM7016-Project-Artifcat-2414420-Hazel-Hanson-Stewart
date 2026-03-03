"""
Plot_Time_Series.py
-------------------
Plots a stacked time-series chart for every numeric column in a dataset
CSV file, sharing the same x-axis across all sub-plots. Year markers and
quarterly minor ticks are applied to the bottom axis.

Opens the chart in an interactive window. The interactive matplotlib
window includes a save button (floppy-disk icon) in the toolbar — click
it to export the figure to any folder and format you choose.

An optional --output-dir argument will also auto-save the figure to
that directory as ``time_series.png`` before the window opens.

Usage (standalone):
    python Plot_Time_Series.py --dataset <path_to_csv>
    python Plot_Time_Series.py --dataset <path_to_csv> --output-dir <path_to_folder>

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
import matplotlib.dates as mdates


# ── Reusable helpers (adapted from dataset.ipynb) ─────────────────────────────

def plot_time_series(series, title, ax, color="#1f77b4", alpha=0.8, linewidth=1.5):
    """Plot a single Pandas Series on ax as a time series line."""
    # (Anthropic, 2024)
    ax.plot(series.index, series, color=color, alpha=alpha, linewidth=linewidth)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Value", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim(series.index.min(), series.index.max())
    return ax


def format_time_axis(ax, is_last=False):
    """Apply date formatting to the x-axis of ax."""
    # (Anthropic, 2024)
    if not is_last:
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel("Date", fontsize=10)
        locator = mdates.YearLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        minor_locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
        ax.xaxis.set_minor_locator(minor_locator)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stacked time-series charts for each column in a dataset CSV."
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
    print(f"Plotting time series for {len(numeric_df.columns)} numeric column(s)...")

    if not isinstance(df.index, pd.DatetimeIndex):
        print(
            "[WARNING] Index is not a DatetimeIndex — date formatting may not render correctly.",
            file=sys.stderr,
        )

    # ── Build plot ────────────────────────────────────────────────────
    n_cols = len(numeric_df.columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, n_cols * 2.5), sharex=True)

    # handle single-column case
    if n_cols == 1:
        axes = [axes]

    for i, column in enumerate(numeric_df.columns):
        plot_time_series(
            numeric_df[column],
            f"Time Series: {column}",
            axes[i],
            color="#1f77b4",
            alpha=0.8,
            linewidth=1.5,
        )
        format_time_axis(axes[i], is_last=(i == n_cols - 1))

    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout(pad=1.2)

    # ── Optional auto-save ────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "time_series.png"
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
    print("=== Plot Time Series complete ===")


if __name__ == "__main__":
    main()
