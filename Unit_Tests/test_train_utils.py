# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
Unit tests for AI_Modules/Training_Methods/train_utils.py

Covers the four shared data-preparation helpers used by both training scripts:
  - temporal_train_val_test_split  (temporal ordering, no leakage, correct sizes)
  - fit_and_scale                  (scaler anchored to train, expected keys, range)
  - make_sequences                 (sliding-window shape and stride correctness)
  - SequenceDataset                (PyTorch Dataset interface compliance)
"""

import numpy as np
import pandas as pd
import pytest
import torch

from train_utils import (
    SequenceDataset,
    create_dataloaders,
    fit_and_scale,
    make_sequences,
    temporal_train_val_test_split,
)

# Shared fixture
@pytest.fixture
def sample_df(): # (Anthropic, 2026)
    """A 200-row DataFrame with two feature columns and one target column.

    Each column is a cumulative random walk so the series drifts realistically,
    making it likely that validation and test splits fall outside the training
    min/max range — required for scaler-leakage assertions.

    Returns:
      A DataFrame with columns 'feat_a', 'feat_b', and 'target', indexed 0–199.
    """
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "feat_a": rng.standard_normal(n).cumsum(),
        "feat_b": rng.standard_normal(n).cumsum() + 5,
        "target": rng.standard_normal(n).cumsum() + 10,
    })

# temporal_train_val_test_split
class TestTemporalSplit: # (Anthropic, 2026)
    """Tests for the temporal_train_val_test_split() function from train_utils.py."""

    def test_sizes_sum_to_total(self, sample_df): # (Anthropic, 2026)
        """Asserts that the combined row count of all splits equals the input length."""
        train, val, test = temporal_train_val_test_split(sample_df)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_no_row_index_overlap(self, sample_df): # (Anthropic, 2026)
        """Asserts that the index sets of all three splits are mutually exclusive."""
        train, val, test = temporal_train_val_test_split(sample_df)
        assert set(train.index).isdisjoint(val.index)
        assert set(val.index).isdisjoint(test.index)
        assert set(train.index).isdisjoint(test.index)

    def test_temporal_order_preserved(self, sample_df): # (Anthropic, 2026)
        """Asserts that train, val, and test splits appear in strict chronological order."""
        train, val, test = temporal_train_val_test_split(sample_df)
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_custom_ratios_respected(self, sample_df): # (Anthropic, 2026)
        """Asserts that a custom 80/10/10 ratio on 200 rows yields approximately 160/20/20 rows."""
        train, val, test = temporal_train_val_test_split(
            sample_df, split_ratios=(0.80, 0.10, 0.10)
        )
        assert len(train) == pytest.approx(160, abs=2)
        assert len(val) == pytest.approx(20, abs=2)

# fit_and_scale
class TestFitAndScale: # (Anthropic, 2026)
    """Tests for the fit_and_scale() function from train_utils.py."""

    @pytest.fixture
    def scaled(self, sample_df): # (Anthropic, 2026)
        """The full output of fit_and_scale() together with the raw split DataFrames.

        Returns:
          A tuple (train, val, test, result) where train, val, and test are the
          raw split DataFrames from temporal_train_val_test_split, and result is
          the dict returned by fit_and_scale().
        """
        train, val, test = temporal_train_val_test_split(sample_df)
        return train, val, test, fit_and_scale(
            train, val, test, ["feat_a", "feat_b"], "target"
        )

    def test_returns_all_expected_keys(self, scaled): # (Anthropic, 2026)
        """Asserts that the result dict contains all eight expected keys."""
        _, _, _, result = scaled
        expected = {
            "feature_scaler", "target_scaler",
            "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
        }
        assert expected.issubset(result.keys())

    def test_training_features_in_unit_interval(self, scaled): # (Anthropic, 2026)
        """Asserts that all scaled training feature values lie within [0, 1]."""
        _, _, _, result = scaled
        vals = result["X_train"].values
        assert vals.min() >= -1e-9
        assert vals.max() <= 1.0 + 1e-9

    def test_scaler_anchored_to_training_data(self, scaled): # (Anthropic, 2026)
        """Asserts that the feature scaler was fitted exclusively on training data.

        Compares the scaler's data_min_ and data_max_ attributes against the
        per-column extremes of the raw training split. A mismatch would indicate
        the scaler was re-fitted on validation or test data, causing leakage.
        """
        train, _, _, result = scaled
        scaler = result["feature_scaler"]
        np.testing.assert_allclose(
            scaler.data_min_,
            [train["feat_a"].min(), train["feat_b"].min()],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            scaler.data_max_,
            [train["feat_a"].max(), train["feat_b"].max()],
            rtol=1e-6,
        )

    def test_original_index_preserved_in_output(self, scaled): # (Anthropic, 2026)
        """Asserts that scaled output DataFrames retain the same index as their source splits."""
        train, val, test, result = scaled
        assert list(result["X_train"].index) == list(train.index)
        assert list(result["X_val"].index) == list(val.index)
        assert list(result["X_test"].index) == list(test.index)

# make_sequences 
class TestMakeSequences: # (Anthropic, 2026)
    """Tests for the make_sequences() function from train_utils.py."""

    @pytest.fixture
    def scaled_train(self, sample_df):
        """The scaled training feature DataFrame and target Series.

        Returns:
          A tuple (X_train, y_train) taken from the fit_and_scale() output for
          the training split of sample_df.
        """
        train, val, test = temporal_train_val_test_split(sample_df)
        result = fit_and_scale(train, val, test, ["feat_a", "feat_b"], "target")
        return result["X_train"], result["y_train"]

    def test_x_shape(self, scaled_train): # (Anthropic, 2026)
        """Asserts that the returned X array has shape [N, seq_len, n_features]."""
        X_df, y_s = scaled_train
        seq_len, horizon = 10, 5
        X, _ = make_sequences(X_df, y_s, seq_len, horizon)
        expected_n = len(X_df) - seq_len - horizon + 1
        assert X.shape == (expected_n, seq_len, 2)

    def test_y_shape(self, scaled_train): # (Anthropic, 2026)
        """Asserts that the returned y array has shape [N, forecast_horizon]."""
        X_df, y_s = scaled_train
        seq_len, horizon = 10, 5
        _, y = make_sequences(X_df, y_s, seq_len, horizon)
        expected_n = len(X_df) - seq_len - horizon + 1
        assert y.shape == (expected_n, horizon)

    def test_window_slides_by_one_row(self, scaled_train): # (Anthropic, 2026)
        """Asserts that each successive window advances by exactly one row.

        For a stride of 1, window[0] covers rows[0:seq_len] and window[1] covers
        rows[1:seq_len+1], so X[0, 1:] must be element-wise equal to X[1, :-1].
        """
        X_df, y_s = scaled_train
        X, _ = make_sequences(X_df, y_s, sequence_length=5, forecast_horizon=1)
        np.testing.assert_array_equal(X[0, 1:], X[1, :-1])

# SequenceDataset 
class TestSequenceDataset: # (Anthropic, 2026)
    """Tests for the SequenceDataset class from train_utils.py."""

    @pytest.fixture
    def dataset(self): # (Anthropic, 2026)
        """A SequenceDataset containing 50 samples of shape (10, 3) with 7-step targets.

        Returns:
          A SequenceDataset built from randomly generated float32 arrays with
          50 samples, a sequence length of 10, 3 features, and a horizon of 7.
        """
        X = np.random.randn(50, 10, 3).astype(np.float32)
        y = np.random.randn(50, 7).astype(np.float32)
        return SequenceDataset(X, y)

    def test_len_matches_sample_count(self, dataset): # (Anthropic, 2026)
        """Asserts that len() returns the number of samples passed to the constructor."""
        assert len(dataset) == 50

    def test_getitem_returns_float_tensors(self, dataset): # (Anthropic, 2026)
        """Asserts that __getitem__ returns a pair of float32 Tensors."""
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_getitem_shape(self, dataset): # (Anthropic, 2026)
        """Asserts that __getitem__ returns tensors with the expected feature and target shapes."""
        x, y = dataset[0]
        assert x.shape == (10, 3)
        assert y.shape == (7,)

    def test_last_index_accessible(self, dataset): # (Anthropic, 2026)
        """Asserts that the final valid index (len - 1) is accessible without error."""
        x, y = dataset[len(dataset) - 1]
        assert x.shape == (10, 3)
