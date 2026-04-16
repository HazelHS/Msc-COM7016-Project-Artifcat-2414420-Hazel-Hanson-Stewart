# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

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
from __eval_utils import calculate_mase

# calculate_mase
class TestCalculateMase: # (Anthropic, 2026)
    """Tests for the calculate_mase() function from eval_utils.py."""

    def test_perfect_predictor_gives_zero(self): # (Anthropic, 2026)
        """Asserts that a perfect predictor produces a MASE of exactly 0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        mase = calculate_mase(y_true, y_true.copy(), y_train)
        assert mase == pytest.approx(0.0, abs=1e-9)

    def test_naive_persistence_predictor_gives_mase_near_one(self): # (Anthropic, 2026)
        """Asserts that a one-step persistence forecast on a linear series gives MASE ≈ 1.

        For a linearly increasing series, the naive persistence prediction has the
        same absolute error as the training-set benchmark, so MASE should equal
        1.0 within a 5% relative tolerance.
        """
        y_train = np.arange(1, 101, dtype=float)          # 1, 2, …, 100
        y_true  = np.arange(101, 121, dtype=float)         # 101, …, 120
        # Shift y_true right by one step — the naive persistence prediction
        y_pred  = np.concatenate([[100.0], y_true[:-1]])
        mase = calculate_mase(y_true, y_pred, y_train)
        assert mase == pytest.approx(1.0, rel=0.05)

    def test_worse_than_naive_gives_mase_greater_than_one(self): # (Anthropic, 2026)
        """Asserts that a constant-zero predictor on a rising series yields MASE > 1."""
        y_true  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred  = np.zeros_like(y_true)
        mase = calculate_mase(y_true, y_pred, y_train)
        assert mase > 1.0

    def test_none_y_train_fallback_returns_finite_value(self): # (Anthropic, 2026)
        """Asserts that passing y_train=None does not raise and returns a finite float.

        When y_train is None, calculate_mase must fall back to using y_true as
        the denominator and still return a valid, finite numeric result.
        """
        y_true = np.array([10.0, 12.0, 11.0, 13.0, 15.0])
        y_pred = np.array([10.5, 11.5, 11.0, 12.5, 14.5])
        result = calculate_mase(y_true, y_pred, y_train=None)
        assert np.isfinite(result)

# Directional classification accuracy
class TestDirectionalClassification: # (Anthropic, 2026)
    """Tests for the binary directional accuracy logic used in eval_classification_metrics.py.

    The logic is extracted and exercised directly on numpy arrays without
    requiring a trained model checkpoint.
    """

    @staticmethod
    def _directional_accuracy(actuals: np.ndarray, predictions: np.ndarray) -> float: # (Anthropic, 2026)
        """Computes binary directional classification accuracy.

        Mirrors the direction-accuracy computation in eval_classification_metrics.main().
        Converts both series to binary up/down labels via first differences, then
        scores label agreement using sklearn's accuracy_score.

        Args:
          actuals: 1-D array of ground-truth values in chronological order.
          predictions: 1-D array of predicted values, same length as actuals.

        Returns:
          The fraction of time steps where the predicted direction (up or down)
          matches the actual direction, as a float in [0, 1].
        """
        return float(accuracy_score(np.diff(actuals) > 0, np.diff(predictions) > 0))

    def test_identical_sequences_give_perfect_accuracy(self): # (Anthropic, 2026)
        """Asserts that identical actuals and predictions yield an accuracy of 1.0."""
        series = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
        assert self._directional_accuracy(series, series.copy()) == pytest.approx(1.0)

    def test_inverted_directions_give_zero_accuracy(self): # (Anthropic, 2026)
        """Asserts that predictions whose directions are all opposite to actuals yield accuracy 0.0."""
        actuals     = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # all up
        predictions = np.array([5.0, 4.0, 3.0, 2.0, 1.0])   # all down
        assert self._directional_accuracy(actuals, predictions) == pytest.approx(0.0)

    def test_accuracy_bounded_in_unit_interval(self): # (Anthropic, 2026)
        """Asserts that directional accuracy always lies within [0, 1] for any input pair."""
        rng = np.random.default_rng(99)
        actuals     = rng.standard_normal(50).cumsum()
        predictions = rng.standard_normal(50).cumsum()
        acc = self._directional_accuracy(actuals, predictions)
        assert 0.0 <= acc <= 1.0
