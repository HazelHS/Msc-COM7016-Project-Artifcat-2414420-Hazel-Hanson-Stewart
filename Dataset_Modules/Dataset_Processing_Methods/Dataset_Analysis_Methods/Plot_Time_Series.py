# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Plot_Time_Series.py script plots a stacked time-series chart for every numeric column in a dataset
CSV file, sharing the same x-axis across all sub-plots. Year markers and
quarterly minor ticks are applied to the bottom axis.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#Reusable helpers
def plot_time_series(series, title, ax, color="#1f77b4", alpha=0.8, linewidth=1.5): # (Anthropic, 2026)
    """Plot a single pandas Series on ax as a time-series line.

    Draws the series against its index, sets the axis title, y-label,
    gridlines, and x-limits.  The axes object is modified in place.

    Args:
        series: A pandas Series with a DatetimeIndex as its index.
        title: Title string displayed above the axes.
        ax: The matplotlib Axes to draw onto.
        color: Line colour as a hex string.  Defaults to "#1f77b4".
        alpha: Line opacity between 0 and 1.  Defaults to 0.8.
        linewidth: Line width in points.  Defaults to 1.5.

    Returns:
        The ax object passed in, after plotting.
    """
    ax.plot(series.index, series, color=color, alpha=alpha, linewidth=linewidth)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Value", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim(series.index.min(), series.index.max())
    return ax

def format_time_axis(ax, is_last=False): # (Anthropic, 2026)
    """Apply date formatting to the x-axis of ax.

    When is_last is False, x-tick labels are hidden so that only the
    bottom axis in a stacked layout carries date labels.  When is_last
    is True, annual major ticks with year labels and quarterly minor
    ticks are applied, and labels are rotated 45 degrees.

    Args:
        ax: The matplotlib Axes whose x-axis to format.
        is_last: If True, render full date labels with annual major and
            quarterly minor ticks.  If False, suppress x-tick labels.
            Defaults to False.
    """
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

# Main
def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments and plot a stacked time-series chart for each column.

    Loads the CSV at --dataset, selects all numeric columns, and renders
    a vertically stacked line chart with a shared x-axis.  The figure is
    displayed interactively.  If --output-dir is supplied the figure is
    also auto-saved to that directory as time_series.png before the
    window opens.
    """
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

    # Build plot 
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

    # Optional auto-save
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
