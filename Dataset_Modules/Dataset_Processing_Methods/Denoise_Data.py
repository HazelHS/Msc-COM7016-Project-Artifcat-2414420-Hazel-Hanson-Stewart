# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Denoise_Data.py script applies wavelet denoising to all numeric columns in a 
dataset CSV, saves the denoised result except this time without ploting the result in a graph.

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
import pywt
from statsmodels.robust import mad

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

# Main
def main() -> None: # (Anthropic, 2026)
    """Wavelet-denoise numeric columns in a CSV file and save the result.

    Reads the dataset specified by the ``--dataset`` CLI argument, applies
    :func:`wavelet_denoising` to all numeric columns, prints a per-column
    summary of noise standard deviation against original and denoised signal
    standard deviations, then writes the result to ``<stem>_denoised.csv``
    in the same directory as the input.

    Raises:
        SystemExit: If the input file does not exist, cannot be parsed as CSV,
          or the output file cannot be written.
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

    # Apply denoising 
    print("=" * 60)
    print("WAVELET DENOISING")
    print("=" * 60)

    df_denoised = wavelet_denoising(df, wavelet=args.wavelet, level=args.level)

    # Per-column summary 
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

    # Save 
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
