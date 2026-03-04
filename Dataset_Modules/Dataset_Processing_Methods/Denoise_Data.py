"""
Denoise_Data.py
---------------
Applies wavelet denoising to all numeric columns in a dataset CSV and
saves the result alongside the original file as ``<stem>_denoised.csv``.

Mathematical foundation:
    The denoising pipeline consists of four steps applied per column:

    1. Multi-level wavelet decomposition (default: Daubechies-4, 3 levels):
           x(t) = A_J + D_J + D_{J-1} + ... + D_1

    2. Universal soft-threshold estimation:
           σ  = MAD(D_1) / 0.6745          (robust noise estimate)
           λ  = σ · √(2·log(N)) · 0.8      (conservative threshold)

    3. Soft thresholding of detail coefficients:
           T_λ(d) = sign(d) · max(|d| − λ, 0)

    4. Inverse wavelet transform to reconstruct the denoised signal.
       Boundary effects are handled by truncating or edge-padding the
       reconstructed signal to match the original length.

Usage (standalone):
    python Denoise_Data.py --dataset <path_to_csv>
    python Denoise_Data.py --dataset <path_to_csv> --wavelet db4 --level 3

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
import pywt
from statsmodels.robust import mad


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
        sigma = mad(coeffs[-1])           # MAD of finest detail coefficients
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run wavelet denoising on the specified CSV.

    Loads the dataset, applies :func:`wavelet_denoising` to all numeric
    columns, prints a per-column noise/signal summary, and saves the
    denoised result as ``<stem>_denoised.csv`` in the same directory.
    """
    parser = argparse.ArgumentParser(
        description="Wavelet-denoise numeric columns in a dataset CSV."
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
        print("No numeric columns found. Denoising skipped.")
        sys.exit(0)

    # ── Apply denoising ───────────────────────────────────────────────
    print("=" * 60)
    print("WAVELET DENOISING")
    print("=" * 60)

    df_denoised = wavelet_denoising(df, wavelet=args.wavelet, level=args.level)

    # ── Per-column summary ────────────────────────────────────────────
    print(f"\n{'Column':<40}  {'Noise std':>12}  {'Signal std (orig)':>18}  {'Signal std (denoised)':>22}")
    print("-" * 100)
    for col in numeric_cols:
        noise = df[col] - df_denoised[col]
        print(
            f"  {col:<38}  "
            f"{noise.std():>12.6f}  "
            f"{df[col].std():>18.6f}  "
            f"{df_denoised[col].std():>22.6f}"
        )

    # ── Save ──────────────────────────────────────────────────────────
    out_path = input_path.parent / (input_path.stem + "_denoised.csv")
    try:
        df_denoised.to_csv(out_path)
        print(f"\nDenoised dataset saved -> {out_path}")
        print(f"Shape: {df_denoised.shape[0]} rows x {df_denoised.shape[1]} columns")
    except Exception as exc:
        print(f"[ERROR] Could not save output: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Denoise Data complete ===")


if __name__ == "__main__":
    main()
