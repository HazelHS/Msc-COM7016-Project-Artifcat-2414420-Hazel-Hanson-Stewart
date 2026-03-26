# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
Unit tests for dataset processing transformations.

Covers three processing methods:
  - wavelet_denoising  (Denoise_Data.py — standalone function, imported directly)
  - Min-Max normalisation logic (mirrors Normalise_Data.main transform)
  - Linear interpolation logic  (mirrors Interpolate_Missing_Data.main transform)

The normalisation and interpolation scripts expose their transform logic only
inside main(), so the logic is replicated inline here to keep tests fast and
dependency-free; the tests validate the correctness of that transform contract.
"""

import numpy as np
import pandas as pd
import pytest
from Denoise_Data import wavelet_denoising

# wavelet_denoising 
class TestWaveletDenoising: # (Anthropic, 2026)
    """Tests for the standalone wavelet_denoising() function in Denoise_Data.py."""

    @pytest.fixture
    def noisy_df(self):
        """A 512-row DataFrame containing a sinusoidal signal with additive Gaussian noise.

        Returns:
          A DataFrame with columns 'price' and 'volume'. Each column contains a
          sine wave scaled to ±100, with zero-mean Gaussian noise (std=20)
          superimposed. The RNG is seeded at 0 for reproducibility.
        """
        rng = np.random.default_rng(0)
        t = np.linspace(0, 4 * np.pi, 512)
        signal = np.sin(t) * 100
        noise = rng.standard_normal(512) * 20
        return pd.DataFrame({
            "price": signal + noise,
            "volume": signal * 2 + noise,
        })

    def test_output_shape_matches_input(self, noisy_df): # (Anthropic, 2026)
        """Asserts that wavelet_denoising preserves the input DataFrame shape."""
        result = wavelet_denoising(noisy_df)
        assert result.shape == noisy_df.shape

    def test_output_columns_match_input(self, noisy_df): # (Anthropic, 2026)
        """Asserts that wavelet_denoising returns the same column names as the input."""
        result = wavelet_denoising(noisy_df)
        assert list(result.columns) == list(noisy_df.columns)

    def test_denoising_reduces_high_frequency_variation(self, noisy_df): # (Anthropic, 2026)
        """Asserts that denoising attenuates high-frequency noise.

        Compares the standard deviation of first differences between the noisy
        input and the denoised output for the 'price' column. A lower value after
        denoising confirms that high-frequency variation was suppressed rather than
        amplified.
        """
        result = wavelet_denoising(noisy_df)
        noisy_var = np.diff(noisy_df["price"].values).std()
        clean_var = np.diff(result["price"].values).std()
        assert clean_var < noisy_var

    def test_non_numeric_columns_pass_through_unchanged(self): # (Anthropic, 2026)
        """Asserts that non-numeric columns are copied without modification.

        Constructs a DataFrame with a string 'date' column and a numeric 'price'
        column, then verifies that the string column values are identical in the
        denoised output.
        """
        rng = np.random.default_rng(1)
        n = 128
        df = pd.DataFrame({
            "date": [f"2020-{i:04d}" for i in range(n)],
            "price": rng.standard_normal(n) * 10 + 100,
        })
        result = wavelet_denoising(df)
        assert list(result["date"]) == list(df["date"])

# Min-Max normalisation
class TestMinMaxNormalisation: # (Anthropic, 2026)
    """Tests for the normalisation contract defined in Normalise_Data.main().

    The transform scales each numeric column to [0, 1] using the formula
    x_norm = (x - min) / (max - min). Constant columns (where min == max)
    are set to 0.0 rather than producing NaN.
    """

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame: # (Anthropic, 2026)
        """Applies Min-Max normalisation to all numeric columns in df.

        Replicates the transform logic from Normalise_Data.main(). Each numeric
        column is scaled independently to [0, 1]. Columns with zero span (all
        values equal) are set to 0.0 to avoid NaN propagation.

        Args:
          df: Input DataFrame. Non-numeric columns are left unchanged.

        Returns:
          A copy of df with each numeric column scaled to the interval [0, 1].
        """
        df_out = df.copy()
        for col in df_out.select_dtypes(include="number").columns:
            col_min, col_max = df_out[col].min(), df_out[col].max()
            span = col_max - col_min
            df_out[col] = 0.0 if span == 0 else (df_out[col] - col_min) / span
        return df_out

    def test_all_values_in_unit_interval(self): # (Anthropic, 2026)
        """Asserts that all normalised values lie within [0, 1]."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 10.0],
            "b": [-5.0, 0.0, 5.0, 100.0],
        })
        out = self._normalise(df)
        for col in ["a", "b"]:
            assert out[col].min() >= 0.0
            assert out[col].max() <= 1.0

    def test_min_and_max_become_zero_and_one(self): # (Anthropic, 2026)
        """Asserts that the column minimum maps to exactly 0 and the maximum to exactly 1."""
        df = pd.DataFrame({"x": [3.0, 7.0, 1.0, 9.0, 5.0]})
        out = self._normalise(df)
        assert out["x"].min() == pytest.approx(0.0)
        assert out["x"].max() == pytest.approx(1.0)

    def test_constant_column_set_to_zero_not_nan(self): # (Anthropic, 2026)
        """Asserts that a constant column produces 0.0 rather than NaN after normalisation."""
        df = pd.DataFrame({"a": [7.0, 7.0, 7.0], "b": [1.0, 2.0, 3.0]})
        out = self._normalise(df)
        assert (out["a"] == 0.0).all()
        assert not out["a"].isna().any()

    def test_output_shape_unchanged(self): # (Anthropic, 2026)
        """Asserts that normalisation does not alter the row or column count."""
        df = pd.DataFrame({"x": range(50), "y": range(50, 100)})
        assert self._normalise(df).shape == df.shape

# Linear interpolation
class TestLinearInterpolation: # (Anthropic, 2026)
    """Tests for the interpolation contract defined in Interpolate_Missing_Data.main().

    The transform fills missing values using linear interpolation with a maximum
    run of 5 consecutive NaNs, applied in both directions along each numeric column.
    """

    @staticmethod
    def _interpolate(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        """Applies linear interpolation to all numeric columns in df.

        Replicates the transform logic from Interpolate_Missing_Data.main(). NaN
        runs no longer than limit are filled from both ends; interior NaNs in gaps
        exceeding limit remain unfilled.

        Args:
          df: Input DataFrame. Non-numeric columns are left unchanged.
          limit: Maximum number of consecutive NaN values to fill. Defaults to 5.

        Returns:
          A copy of df with NaN values in numeric columns filled where the gap
          length does not exceed limit.
        """
        df_out = df.copy()
        for col in df_out.select_dtypes(include="number").columns:
            df_out[col] = df_out[col].interpolate(
                method="linear", limit=limit, limit_direction="both"
            )
        return df_out

    def test_small_gap_fully_filled(self): # (Anthropic, 2026)
        """Asserts that an interior gap of 3 consecutive NaNs is completely filled.

        The series length exceeds limit+1, which is required to satisfy
        pandas' internal interpolation window.
        """
        df = pd.DataFrame({"price": [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0]})
        out = self._interpolate(df)
        assert out["price"].isna().sum() == 0

    def test_large_interior_gap_partially_filled(self): # (Anthropic, 2026)
        """Asserts that a 15-NaN interior gap leaves central NaNs unfilled.

        A gap of 15 consecutive NaNs exceeds limit*2, so the central positions
        cannot be reached from either boundary and must remain NaN in the output.
        """
        values = [1.0] + [np.nan] * 15 + [16.0]
        df = pd.DataFrame({"price": values})
        out = self._interpolate(df, limit=5)
        assert out["price"].isna().sum() > 0

    def test_gap_free_series_unchanged(self): # (Anthropic, 2026)
        """Asserts that a series containing no missing values is returned identical."""
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = self._interpolate(df)
        pd.testing.assert_series_equal(out["price"], df["price"])

    def test_output_shape_unchanged_after_interpolation(self): # (Anthropic, 2026)
        """Asserts that interpolation does not alter the row or column count.

        Uses series of length 7 (greater than limit+1) to satisfy pandas'
        internal interpolation window requirement.
        """
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0],
        })
        assert self._interpolate(df).shape == df.shape