"""
Denoise_Data_Plot.py
--------------------
Applies wavelet denoising to all numeric columns in a dataset CSV,
saves the denoised result as ``<stem>_denoised.csv``, and opens an
interactive time-series plot showing:

  • Original signal (blue, semi-transparent)
  • Denoised signal (orange)
  • Removed noise component (red, below each signal pair)

The interactive matplotlib window includes a save button (floppy-disk
icon) in the toolbar — click it to export the figure to any folder and
format you choose.

An optional --output-dir argument will also auto-save:
  • One PNG per column:               denoising_<column>.png
  • One combined PNG for all columns: all_denoised_columns_with_noise.png

Mathematical foundation:
    1. Multi-level decomposition:   x(t) = A_J + D_J + D_{J-1} + … + D_1
    2. Universal threshold:         λ = σ·√(2·log(N))·0.8
                                    σ = MAD(D_1) / 0.6745
    3. Soft thresholding:           T_λ(d) = sign(d)·max(|d|−λ, 0)
    4. Inverse wavelet transform + boundary correction.

Usage (standalone):
    python Denoise_Data_Plot.py --dataset <path_to_csv>
    python Denoise_Data_Plot.py --dataset <path_to_csv> --output-dir <folder>
    python Denoise_Data_Plot.py --dataset <path_to_csv> --wavelet db4 --level 3

When launched from the Model Designer GUI the --dataset argument is
populated automatically from the dropdown selection.

References:
    Gil, M., et al. (2024). Stock index forecasting based on multivariate
    empirical mode decomposition and temporal convolutional networks.
    Anthropic. (2024). Claude (claude-sonnet) [Large language model].
    https://www.anthropic.com
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


# ── Time-series plot helpers (adapted from dataset.ipynb) ─────────────────────

def plot_time_series(series, title, ax, color="blue", alpha=0.8, linewidth=1.5):
    """Plot *series* as a line on *ax*.

    Args:
        series: pandas Series with a DatetimeIndex.
        title: Axis title string, or None to skip.
        ax: matplotlib Axes to plot onto.
        color: Line colour (default ``'blue'``).
        alpha: Line opacity (default 0.8).
        linewidth: Line width in points (default 1.5).

    Returns:
        The same *ax* object after plotting.
    """
    ax.plot(series.index, series, color=color, alpha=alpha, linewidth=linewidth)
    if title is not None:
        ax.set_title(title, fontsize=11)
    ax.set_ylabel("Value", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim(series.index.min(), series.index.max())
    return ax


def format_time_axis(ax, is_last=False):
    """Apply date formatting to the x-axis of *ax*.

    Args:
        ax: matplotlib Axes whose x-axis to format.
        is_last: If True, show full date labels and tick marks;
            if False, hide x-tick labels (for stacked plots).
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


# ── Core denoising ────────────────────────────────────────────────────────────

def wavelet_denoising(df: pd.DataFrame, wavelet: str = "db4", level: int = 3) -> pd.DataFrame:
    """
    Apply wavelet denoising to all numeric columns in *df*.

    Steps (per column):
      1. Multi-level decomposition:  x(t) = A_J + D_J + … + D_1
      2. Universal threshold:        λ = σ·√(2·log(N))·0.8
                                     σ = MAD(D_1) / 0.6745
      3. Soft thresholding:          T_λ(d) = sign(d)·max(|d|−λ, 0)
      4. Inverse transform + boundary correction.

    Args:
        df: Input DataFrame with a DatetimeIndex and numeric columns.
        wavelet: PyWavelets wavelet name (default ``'db4'``).
        level: Decomposition level (default 3).

    Returns:
        DataFrame of the same shape with denoised values.
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


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_denoising_results(original_df: pd.DataFrame, denoised_df: pd.DataFrame, column_name: str):
    """Create a two-panel figure for *column_name* showing original vs denoised.

    Top panel: original (blue) vs denoised (orange) signal.
    Bottom panel: removed noise component (red).

    Args:
        original_df: DataFrame containing the raw signal.
        denoised_df: DataFrame containing the denoised signal.
        column_name: Column to visualise.

    Returns:
        The matplotlib Figure object.
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


def plot_all_denoised_columns(original_df: pd.DataFrame, denoised_df: pd.DataFrame):
    """
    Create a stacked figure with two rows per column:
      Row 1: original (blue) vs denoised (orange) signal.
      Row 2: noise component (red) with auto-scaled y-limits.

    Returns the matplotlib Figure.
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
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

    # ── Apply denoising ───────────────────────────────────────────────
    print("Applying wavelet denoising...")
    df_denoised = wavelet_denoising(df, wavelet=args.wavelet, level=args.level)

    # ── Save denoised CSV ─────────────────────────────────────────────
    out_csv = input_path.parent / (input_path.stem + "_denoised.csv")
    try:
        df_denoised.to_csv(out_csv)
        print(f"Denoised dataset saved -> {out_csv}")
    except Exception as exc:
        print(f"[WARNING] Could not save denoised CSV: {exc}", file=sys.stderr)

    # ── Per-column comparison plots ───────────────────────────────────
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

    # ── Combined all-columns plot ─────────────────────────────────────
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
