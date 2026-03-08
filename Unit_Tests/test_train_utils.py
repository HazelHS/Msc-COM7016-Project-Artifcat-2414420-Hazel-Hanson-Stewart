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


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """200-row DataFrame with two feature columns and one target column.

    Uses a cumulative random walk so the series drifts realistically,
    ensuring validation and test splits are likely to fall outside the
    training min/max range (needed for leakage tests).
    """
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "feat_a": rng.standard_normal(n).cumsum(),
        "feat_b": rng.standard_normal(n).cumsum() + 5,
        "target": rng.standard_normal(n).cumsum() + 10,
    })


# ── temporal_train_val_test_split ─────────────────────────────────────────────

class TestTemporalSplit:

    def test_sizes_sum_to_total(self, sample_df):
        """All rows must be assigned to exactly one split — no rows lost or duplicated."""
        train, val, test = temporal_train_val_test_split(sample_df)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_no_row_index_overlap(self, sample_df):
        """Index sets of each split must be mutually exclusive."""
        train, val, test = temporal_train_val_test_split(sample_df)
        assert set(train.index).isdisjoint(val.index)
        assert set(val.index).isdisjoint(test.index)
        assert set(train.index).isdisjoint(test.index)

    def test_temporal_order_preserved(self, sample_df):
        """Splits must appear in chronological succession with no reordering."""
        train, val, test = temporal_train_val_test_split(sample_df)
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_custom_ratios_respected(self, sample_df):
        """An 80/10/10 split on 200 rows must yield approximately 160/20/20."""
        train, val, test = temporal_train_val_test_split(
            sample_df, split_ratios=(0.80, 0.10, 0.10)
        )
        assert len(train) == pytest.approx(160, abs=2)
        assert len(val) == pytest.approx(20, abs=2)


# ── fit_and_scale ─────────────────────────────────────────────────────────────

class TestFitAndScale:

    @pytest.fixture
    def scaled(self, sample_df):
        train, val, test = temporal_train_val_test_split(sample_df)
        return train, val, test, fit_and_scale(
            train, val, test, ["feat_a", "feat_b"], "target"
        )

    def test_returns_all_expected_keys(self, scaled):
        _, _, _, result = scaled
        expected = {
            "feature_scaler", "target_scaler",
            "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
        }
        assert expected.issubset(result.keys())

    def test_training_features_in_unit_interval(self, scaled):
        """Min-Max scaler must map every training feature value to [0, 1]."""
        _, _, _, result = scaled
        vals = result["X_train"].values
        assert vals.min() >= -1e-9
        assert vals.max() <= 1.0 + 1e-9

    def test_scaler_anchored_to_training_data(self, scaled):
        """data_min_ and data_max_ must equal the training split's extremes,
        proving the scaler was not re-fitted on validation or test data."""
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

    def test_original_index_preserved_in_output(self, scaled):
        """Scaled DataFrames must retain the same index as their source split."""
        train, val, test, result = scaled
        assert list(result["X_train"].index) == list(train.index)
        assert list(result["X_val"].index) == list(val.index)
        assert list(result["X_test"].index) == list(test.index)


# ── make_sequences ────────────────────────────────────────────────────────────

class TestMakeSequences:

    @pytest.fixture
    def scaled_train(self, sample_df):
        train, val, test = temporal_train_val_test_split(sample_df)
        result = fit_and_scale(train, val, test, ["feat_a", "feat_b"], "target")
        return result["X_train"], result["y_train"]

    def test_x_shape(self, scaled_train):
        """X array must be [N, seq_len, n_features]."""
        X_df, y_s = scaled_train
        seq_len, horizon = 10, 5
        X, _ = make_sequences(X_df, y_s, seq_len, horizon)
        expected_n = len(X_df) - seq_len - horizon + 1
        assert X.shape == (expected_n, seq_len, 2)

    def test_y_shape(self, scaled_train):
        """y array must be [N, forecast_horizon]."""
        X_df, y_s = scaled_train
        seq_len, horizon = 10, 5
        _, y = make_sequences(X_df, y_s, seq_len, horizon)
        expected_n = len(X_df) - seq_len - horizon + 1
        assert y.shape == (expected_n, horizon)

    def test_window_slides_by_one_row(self, scaled_train):
        """Consecutive windows must overlap by (seq_len - 1) rows.

        Window[0] = rows[0:seq_len], Window[1] = rows[1:seq_len+1]
        so X[0, 1:] must equal X[1, :-1] element-wise.
        """
        X_df, y_s = scaled_train
        X, _ = make_sequences(X_df, y_s, sequence_length=5, forecast_horizon=1)
        np.testing.assert_array_equal(X[0, 1:], X[1, :-1])


# ── SequenceDataset ───────────────────────────────────────────────────────────

class TestSequenceDataset:

    @pytest.fixture
    def dataset(self):
        X = np.random.randn(50, 10, 3).astype(np.float32)
        y = np.random.randn(50, 7).astype(np.float32)
        return SequenceDataset(X, y)

    def test_len_matches_sample_count(self, dataset):
        assert len(dataset) == 50

    def test_getitem_returns_float_tensors(self, dataset):
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_getitem_shape(self, dataset):
        x, y = dataset[0]
        assert x.shape == (10, 3)
        assert y.shape == (7,)

    def test_last_index_accessible(self, dataset):
        """Boundary check: index len-1 must not raise IndexError."""
        x, y = dataset[len(dataset) - 1]
        assert x.shape == (10, 3)
