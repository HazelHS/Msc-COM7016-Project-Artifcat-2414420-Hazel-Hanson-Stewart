# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Denoise_Data_Plot.py script applies wavelet denoising to all numeric columns in a 
dataset CSV, saves the denoised result and opens an interactive time-series plot showing 
the original signal (blue, semi-transparent) with the Denoised signal (orange) and 
the removed noise component (red, below each signal pair).

The formula for wavelet denoising is derived from Lopez et al. (2024):
    1. Multi-level decomposition:   x(t) = A_J + D_J + D_{J-1} + … + D_1
    2. Universal threshold:         λ = σ·√(2·log(N))·0.8
                                    σ = MAD(D_1) / 0.6745
    3. Soft thresholding:           T_λ(d) = sign(d)·max(|d|−λ, 0)
    4. Inverse wavelet transform + boundary correction.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pywt
from statsmodels.robust import mad

# Time-series plot helpers
def plot_time_series(series, title, ax, color="blue", alpha=0.8, linewidth=1.5): # (Anthropic, 2026)
    """Plot a pandas Series as a line on a matplotlib Axes.

    Adds a line plot of ``series`` to ``ax``, sets grid lines, locks the
    x-axis limits to the extent of the series index, and optionally sets
    the axes title.

    Args:
        series: A pandas Series with a DatetimeIndex to plot.
        title: Title string for the axes, or ``None`` to leave it unset.
        ax: The matplotlib Axes to draw on.
        color: Line colour (default ``'blue'``).
        alpha: Line opacity in [0, 1] (default ``0.8``).
        linewidth: Line width in points (default ``1.5``).

    Returns:
        The ``ax`` object passed in, after the line has been added.
    """
    ax.plot(series.index, series, color=color, alpha=alpha, linewidth=linewidth)
    if title is not None:
        ax.set_title(title, fontsize=11)
    ax.set_ylabel("Value", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim(series.index.min(), series.index.max())
    return ax

def format_time_axis(ax, is_last=False): # (Anthropic, 2026)
    """Apply date formatting to the x-axis of a matplotlib Axes.

    When used in stacked subplot layouts, pass ``is_last=False`` for all
    rows except the bottom one to suppress redundant x-tick labels.
    The bottom row (``is_last=True``) receives year major ticks, quarterly
    minor ticks, and rotated date labels.

    Args:
        ax: The matplotlib Axes whose x-axis to format.
        is_last: If ``True``, render full date labels, year major ticks,
          and quarterly minor ticks. If ``False``, hide x-tick labels
          (default ``False``).
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

# Core denoising 
def wavelet_denoising(df: pd.DataFrame, wavelet: str = "db4", level: int = 3) -> pd.DataFrame: # (Anthropic, 2026)
    """Apply wavelet denoising to all numeric columns in a DataFrame.

    Decomposes each numeric column using a multi-level discrete wavelet
    transform, suppresses noise via universal soft thresholding on the detail
    coefficients, then reconstructs the signal. The approximation coefficients
    are left untouched. Non-numeric columns are copied through unchanged.
    Boundary length mismatches after reconstruction are corrected by truncation
    or edge-padding to match the original row count.

    Args:
        df: Input DataFrame whose numeric columns are to be denoised.
        wavelet: PyWavelets wavelet family name (default ``'db4'``).
        level: Number of decomposition levels (default ``3``).

    Returns:
        A DataFrame of the same shape and index as ``df`` with denoised values
        in every numeric column.
    """
    # (Gil et al., 2024), (Anthropic, 2024)
    df_denoised = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for column in numeric_cols:
        # 1. Decompose
        coeffs = pywt.wavedec(df[column].values.copy(), wavelet, level=level)

        # 2. Threshold estimation
        sigma = mad(coeffs[-1])
        n = len(df[column])
        threshold = sigma * np.sqrt(2 * np.log(n)) * 0.8

        # 3. Soft threshold detail coefficients; keep approximation untouched
        coeffs_modified = [coeffs[0]]
        for i in range(1, len(coeffs)):
            coeffs_modified.append(pywt.threshold(coeffs[i], threshold, "soft"))

        # 4. Reconstruct
        denoised_data = pywt.waverec(coeffs_modified, wavelet)

        # 5. Boundary correction
        if len(denoised_data) > n:
            denoised_data = denoised_data[:n]
        elif len(denoised_data) < n:
            denoised_data = np.pad(denoised_data, (0, n - len(denoised_data)), "edge")

        df_denoised[column] = denoised_data

    return df_denoised

# Plotting helpers
def plot_denoising_results(original_df: pd.DataFrame, denoised_df: pd.DataFrame, column_name: str): # (Anthropic, 2026)
    """Create a two-panel comparison figure for a single column.

    The top panel overlays the original signal (blue, semi-transparent) with
    the denoised signal (orange). The bottom panel shows the noise component
    removed by denoising (original minus denoised, red). Both panels share
    the same x-axis; only the bottom panel shows date labels.

    Args:
        original_df: DataFrame containing the raw signal column.
        denoised_df: DataFrame containing the denoised signal column,
          with the same index as ``original_df``.
        column_name: Name of the column to visualise. Must exist in both
          DataFrames.

    Returns:
        The matplotlib Figure containing the two panels.
    """
    # (Anthropic, 2024)
    noise = original_df[column_name] - denoised_df[column_name]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Panel 1 — original vs denoised
    plot_time_series(
        original_df[column_name],
        f"Denoising Results: {column_name}",
        axes[0],
        color="#1f77b4",
        alpha=0.5,
        linewidth=1.5,
    )
    plot_time_series(
        denoised_df[column_name],
        None,
        axes[0],
        color="#ff7f0e",
        alpha=0.8,
        linewidth=1.5,
    )
    axes[0].legend(["Original", "Denoised"], loc="upper right", fontsize=9)

    # Panel 2 — removed noise
    plot_time_series(
        noise,
        "Removed Noise Component",
        axes[1],
        color="#d62728",
        alpha=0.5,
        linewidth=1,
    )

    format_time_axis(axes[0], is_last=False)
    format_time_axis(axes[1], is_last=True)

    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout(pad=1.2)

    return fig

def plot_all_denoised_columns(original_df: pd.DataFrame, denoised_df: pd.DataFrame): # (Anthropic, 2026)
    """Create a stacked figure showing denoising results for all numeric columns.

    Produces two subplot rows per numeric column: the first overlays the
    original (blue) and denoised (orange) signals; the second shows the
    removed noise component (red) with y-limits clamped to ±3 standard
    deviations of the noise. All subplots share a common x-axis; date labels
    appear only on the bottom row.

    Args:
        original_df: DataFrame containing the raw signals.
        denoised_df: DataFrame containing the denoised signals, with the same
          index and numeric columns as ``original_df``.

    Returns:
        The matplotlib Figure containing all stacked panels.
    """
    # (Anthropic, 2024)
    numeric_cols = original_df.select_dtypes(include="number").columns.tolist()
    n_cols = len(numeric_cols)

    fig, axes = plt.subplots(n_cols * 2, 1, figsize=(15, n_cols * 4), sharex=True)

    # Ensure axes is always a 1-D array even for a single column
    axes = np.atleast_1d(axes)

    for i, column in enumerate(numeric_cols):
        noise = original_df[column] - denoised_df[column]
        signal_idx = i * 2
        noise_idx  = i * 2 + 1

        # Signal subplot
        plot_time_series(
            original_df[column],
            f"Time Series: {column}",
            axes[signal_idx],
            color="#1f77b4",
            alpha=0.5,
            linewidth=1.5,
        )
        plot_time_series(
            denoised_df[column],
            None,
            axes[signal_idx],
            color="#ff7f0e",
            alpha=0.8,
            linewidth=1.5,
        )
        axes[signal_idx].legend(["Original", "Denoised"], loc="upper right", fontsize=8)

        # Noise subplot
        plot_time_series(
            noise,
            f"Noise Component: {column}",
            axes[noise_idx],
            color="#d62728",
            alpha=0.7,
            linewidth=1,
        )

        is_last_row = i == n_cols - 1
        format_time_axis(axes[signal_idx], is_last=False)
        format_time_axis(axes[noise_idx],  is_last=is_last_row)

        # Auto-scale noise y-axis using ±3σ, guarding against inf/NaN
        valid_noise = noise.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_noise) > 0:
            noise_std  = valid_noise.std()
            noise_mean = valid_noise.mean()
            if np.isfinite(noise_std) and np.isfinite(noise_mean) and noise_std > 0:
                axes[noise_idx].set_ylim(
                    [noise_mean - 3 * noise_std, noise_mean + 3 * noise_std]
                )
            else:
                axes[noise_idx].set_ylim([-1, 1])
        else:
            axes[noise_idx].set_ylim([-1, 1])

    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout(pad=1.2)

    return fig

# Main 
def main() -> None: # (Anthropic, 2026)
    """Wavelet-denoise a CSV and display interactive denoising comparison plots.

    Reads the dataset specified by the ``--dataset`` CLI argument, applies
    :func:`wavelet_denoising` to all numeric columns, saves the denoised data
    as ``<stem>_denoised.csv`` alongside the input, then opens an interactive
    matplotlib window with per-column and combined denoising plots.

    If ``--output-dir`` is provided, one PNG per column and one combined PNG
    are auto-saved to that directory at 300 dpi before the window opens.

    Raises:
        SystemExit: If the input file does not exist or cannot be parsed as
          CSV, or if no numeric columns are found.
    """
    parser = argparse.ArgumentParser(
        description="Wavelet-denoise a dataset CSV and display time-series comparison plots."
    )
    parser.add_argument(
        "--dataset", required=True, help="Absolute path to the input CSV file."
    )
    parser.add_argument(
        "--wavelet",
        default="db4",
        help="PyWavelets wavelet family to use (default: db4).",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        help="Decomposition level (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional folder to auto-save PNGs. "
            "Saves one PNG per column and one combined PNG. "
            "If omitted, use the toolbar save button in the plot window."
        ),
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

    print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nWavelet : {args.wavelet}  |  Level : {args.level}\n")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("[ERROR] No numeric columns found.", file=sys.stderr)
        sys.exit(1)

    # Apply denoising 
    print("Applying wavelet denoising...")
    df_denoised = wavelet_denoising(df, wavelet=args.wavelet, level=args.level)

    # Save denoised CSV 
    out_csv = input_path.parent / (input_path.stem + "_denoised.csv")
    try:
        df_denoised.to_csv(out_csv)
        print(f"Denoised dataset saved -> {out_csv}")
    except Exception as exc:
        print(f"[WARNING] Could not save denoised CSV: {exc}", file=sys.stderr)

    # Per-column comparison plots
    print(f"\nGenerating per-column denoising plots ({len(numeric_cols)} column(s))...")
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for column in numeric_cols:
        print(f"  Plotting: {column}")
        fig = plot_denoising_results(df, df_denoised, column)
        if out_dir:
            safe_name = column.replace("/", "_").replace("\\", "_")
            out_path = out_dir / f"denoising_{safe_name}.png"
            try:
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"    Saved -> {out_path}")
            except Exception as exc:
                print(f"    [WARNING] Could not save: {exc}", file=sys.stderr)
        plt.close(fig)  # close individual before opening combined

    # Combined all-columns plot
    print("\nGenerating combined plot for all columns...")
    fig_all = plot_all_denoised_columns(df, df_denoised)

    if out_dir:
        combined_path = out_dir / "all_denoised_columns_with_noise.png"
        try:
            fig_all.savefig(combined_path, dpi=300, bbox_inches="tight")
            print(f"Combined figure saved -> {combined_path}")
        except Exception as exc:
            print(f"[WARNING] Could not save combined figure: {exc}", file=sys.stderr)
    else:
        print(
            "Tip: use the save icon in the toolbar to export, "
            "or pass --output-dir to auto-save all PNGs."
        )

    plt.show()
    print("\n=== Denoise Data Plot complete ===")

if __name__ == "__main__":
    main()
