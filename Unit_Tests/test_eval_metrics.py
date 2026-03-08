"""
Unit tests for evaluation metric computations.

Covers:
  - calculate_mase (Evaluation_Modules/Evaluation_Metrics/eval_utils.py)
      A MASE of 0 for a perfect predictor, ~1 for a naive persistence
      predictor, >1 for a predictor worse than naive, and graceful
      fallback when y_train is None.

  - Directional binary classification accuracy logic used in
      eval_classification_metrics.py — tested directly on numpy arrays
      without requiring a trained model checkpoint.
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from eval_utils import calculate_mase


# ── calculate_mase ────────────────────────────────────────────────────────────

class TestCalculateMase:

    def test_perfect_predictor_gives_zero(self):
        """When predictions exactly match actuals, MAE is 0, so MASE must be 0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        mase = calculate_mase(y_true, y_true.copy(), y_train)
        assert mase == pytest.approx(0.0, abs=1e-9)

    def test_naive_persistence_predictor_gives_mase_near_one(self):
        """For a linear series, a one-step persistence forecast has the same
        absolute error as the naive training-set benchmark, so MASE == 1.0."""
        y_train = np.arange(1, 101, dtype=float)          # 1, 2, …, 100
        y_true  = np.arange(101, 121, dtype=float)         # 101, …, 120
        # Shift y_true right by one step — the naive persistence prediction
        y_pred  = np.concatenate([[100.0], y_true[:-1]])
        mase = calculate_mase(y_true, y_pred, y_train)
        assert mase == pytest.approx(1.0, rel=0.05)

    def test_worse_than_naive_gives_mase_greater_than_one(self):
        """A constant-zero predictor on a rising series must yield MASE > 1."""
        y_true  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred  = np.zeros_like(y_true)
        mase = calculate_mase(y_true, y_pred, y_train)
        assert mase > 1.0

    def test_none_y_train_fallback_returns_finite_value(self):
        """calculate_mase must not raise when y_train is None, and must return
        a finite float using y_true as the fallback denominator."""
        y_true = np.array([10.0, 12.0, 11.0, 13.0, 15.0])
        y_pred = np.array([10.5, 11.5, 11.0, 12.5, 14.5])
        result = calculate_mase(y_true, y_pred, y_train=None)
        assert np.isfinite(result)


# ── Directional classification accuracy ──────────────────────────────────────

class TestDirectionalClassification:
    """Tests the binary directional labelling and accuracy logic used in
    eval_classification_metrics.py, extracted as a pure numpy computation."""

    @staticmethod
    def _directional_accuracy(actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Mirror of the binary-direction accuracy computation in eval_classification_metrics.main()."""
        return float(accuracy_score(np.diff(actuals) > 0, np.diff(predictions) > 0))

    def test_identical_sequences_give_perfect_accuracy(self):
        """When predictions equal actuals exactly, all directions match → accuracy = 1."""
        series = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
        assert self._directional_accuracy(series, series.copy()) == pytest.approx(1.0)

    def test_inverted_directions_give_zero_accuracy(self):
        """When every predicted direction is opposite to the actual direction,
        accuracy must be 0.0."""
        actuals     = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # all up
        predictions = np.array([5.0, 4.0, 3.0, 2.0, 1.0])   # all down
        assert self._directional_accuracy(actuals, predictions) == pytest.approx(0.0)

    def test_accuracy_bounded_in_unit_interval(self):
        """For any pair of sequences, accuracy must lie in [0, 1]."""
        rng = np.random.default_rng(99)
        actuals     = rng.standard_normal(50).cumsum()
        predictions = rng.standard_normal(50).cumsum()
        acc = self._directional_accuracy(actuals, predictions)
        assert 0.0 <= acc <= 1.0
