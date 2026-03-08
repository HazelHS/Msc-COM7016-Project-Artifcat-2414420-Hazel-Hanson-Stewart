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


# ── wavelet_denoising ─────────────────────────────────────────────────────────

class TestWaveletDenoising:
    """Tests for the standalone wavelet_denoising() function in Denoise_Data.py."""

    @pytest.fixture
    def noisy_df(self):
        """DataFrame with a known sinusoidal signal plus additive Gaussian noise."""
        rng = np.random.default_rng(0)
        t = np.linspace(0, 4 * np.pi, 512)
        signal = np.sin(t) * 100
        noise = rng.standard_normal(512) * 20
        return pd.DataFrame({
            "price": signal + noise,
            "volume": signal * 2 + noise,
        })

    def test_output_shape_matches_input(self, noisy_df):
        result = wavelet_denoising(noisy_df)
        assert result.shape == noisy_df.shape

    def test_output_columns_match_input(self, noisy_df):
        result = wavelet_denoising(noisy_df)
        assert list(result.columns) == list(noisy_df.columns)

    def test_denoising_reduces_high_frequency_variation(self, noisy_df):
        """Denoised signal must have a smaller first-difference standard deviation
        than the original noisy signal, confirming that high-frequency noise
        was suppressed rather than amplified."""
        result = wavelet_denoising(noisy_df)
        noisy_var = np.diff(noisy_df["price"].values).std()
        clean_var = np.diff(result["price"].values).std()
        assert clean_var < noisy_var

    def test_non_numeric_columns_pass_through_unchanged(self):
        """String columns must be copied without modification."""
        rng = np.random.default_rng(1)
        n = 128
        df = pd.DataFrame({
            "date": [f"2020-{i:04d}" for i in range(n)],
            "price": rng.standard_normal(n) * 10 + 100,
        })
        result = wavelet_denoising(df)
        assert list(result["date"]) == list(df["date"])


# ── Min-Max normalisation ─────────────────────────────────────────────────────

class TestMinMaxNormalisation:
    """Tests for the normalisation contract defined in Normalise_Data.main().

    The transform: for each numeric column, x_norm = (x - min) / (max - min).
    Constant columns (min == max) are set to 0.0.
    """

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Inline replica of the Min-Max transform from Normalise_Data.main()."""
        df_out = df.copy()
        for col in df_out.select_dtypes(include="number").columns:
            col_min, col_max = df_out[col].min(), df_out[col].max()
            span = col_max - col_min
            df_out[col] = 0.0 if span == 0 else (df_out[col] - col_min) / span
        return df_out

    def test_all_values_in_unit_interval(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 10.0],
            "b": [-5.0, 0.0, 5.0, 100.0],
        })
        out = self._normalise(df)
        for col in ["a", "b"]:
            assert out[col].min() >= 0.0
            assert out[col].max() <= 1.0

    def test_min_and_max_become_zero_and_one(self):
        """The column minimum must map to exactly 0 and the maximum to exactly 1."""
        df = pd.DataFrame({"x": [3.0, 7.0, 1.0, 9.0, 5.0]})
        out = self._normalise(df)
        assert out["x"].min() == pytest.approx(0.0)
        assert out["x"].max() == pytest.approx(1.0)

    def test_constant_column_set_to_zero_not_nan(self):
        """A column where all values are equal must produce 0.0, not NaN."""
        df = pd.DataFrame({"a": [7.0, 7.0, 7.0], "b": [1.0, 2.0, 3.0]})
        out = self._normalise(df)
        assert (out["a"] == 0.0).all()
        assert not out["a"].isna().any()

    def test_output_shape_unchanged(self):
        df = pd.DataFrame({"x": range(50), "y": range(50, 100)})
        assert self._normalise(df).shape == df.shape


# ── Linear interpolation ──────────────────────────────────────────────────────

class TestLinearInterpolation:
    """Tests for the interpolation contract defined in Interpolate_Missing_Data.main().

    The transform: linear interpolation, limit=5 consecutive NaNs, both directions.
    """

    @staticmethod
    def _interpolate(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        """Inline replica of the interpolation logic from Interpolate_Missing_Data.main()."""
        df_out = df.copy()
        for col in df_out.select_dtypes(include="number").columns:
            df_out[col] = df_out[col].interpolate(
                method="linear", limit=limit, limit_direction="both"
            )
        return df_out

    def test_small_gap_fully_filled(self):
        """A 3-NaN interior gap (below the limit of 5) must be completely filled.
        Series must be longer than limit+1 to satisfy pandas' internal window."""
        df = pd.DataFrame({"price": [1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0]})
        out = self._interpolate(df)
        assert out["price"].isna().sum() == 0

    def test_large_interior_gap_partially_filled(self):
        """A 15-NaN interior gap exceeds limit*2; the middle NaNs must survive."""
        values = [1.0] + [np.nan] * 15 + [16.0]
        df = pd.DataFrame({"price": values})
        out = self._interpolate(df, limit=5)
        assert out["price"].isna().sum() > 0

    def test_gap_free_series_unchanged(self):
        """A series with no missing values must be returned identical."""
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = self._interpolate(df)
        pd.testing.assert_series_equal(out["price"], df["price"])

    def test_output_shape_unchanged_after_interpolation(self):
        """Row and column count must not change regardless of NaN pattern.
        Series length must exceed limit+1 for pandas' internal window."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0],
        })
        assert self._interpolate(df).shape == df.shape
