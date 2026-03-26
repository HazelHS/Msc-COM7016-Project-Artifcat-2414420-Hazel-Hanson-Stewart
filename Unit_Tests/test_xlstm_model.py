# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
Unit tests for AI_Modules/Model_Designs/xLSTM_TS.py

Validates:
  - xLSTM_TS_Model forward pass output shape across a range of hyperparameter
    configurations (parametrised).
  - Model output is finite (no NaN / Inf from unstable initialisations).
  - directional_loss returns a non-negative scalar tensor.
  - Perfect predictions produce zero loss.

No training is performed; all tests operate on randomly-initialised weights
and random input tensors.
"""

import pytest
import torch
from xLSTM_TS import xLSTM_TS_Model, directional_loss

# xLSTM_TS_Model forward pass
class TestXLSTMForwardPass: # (Anthropic, 2026)
    """Tests for the xLSTM_TS_Model forward pass from xLSTM_TS.py."""

    @pytest.mark.parametrize("batch,seq_len,n_feats,embed_dim,out_size", [
        (8,  30, 3, 32, 1),   # small config, single-step
        (4,  60, 5, 64, 7),   # default-scale config, 7-step horizon
        (1,  10, 1, 16, 3),   # single sample, minimal features
        (16, 20, 2, 32, 5),   # larger batch
    ])
    def test_output_shape(self, batch, seq_len, n_feats, embed_dim, out_size):
        """Asserts that the forward pass produces a tensor of shape [batch, output_s # (Anthropic, 2026)ize].

        Covers a range of hyperparameter configurations via parametrize to catch
        shape regressions introduced by changes to any internal projection layer.
        """
        model = xLSTM_TS_Model(
            input_shape=(seq_len, n_feats),
            embedding_dim=embed_dim,
            output_size=out_size,
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(batch, seq_len, n_feats))
        assert out.shape == (batch, out_size), (
            f"Expected ({batch}, {out_size}), got {out.shape}"
        )

    def test_output_is_finite(self): # (Anthropic, 2026)
        """Asserts that a randomly-initialised model produces only finite values on the first forward pass."""
        model = xLSTM_TS_Model(input_shape=(20, 2), embedding_dim=32, output_size=5)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 20, 2))
        assert torch.isfinite(out).all(), (
            "Model output contains NaN or Inf with default initialisation"
        )

    def test_batch_size_one_does_not_raise(self): # (Anthropic, 2026)
        """Asserts that a single-sample batch passes through every layer without shape errors."""
        model = xLSTM_TS_Model(input_shape=(15, 3), embedding_dim=16, output_size=4)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 15, 3))
        assert out.shape == (1, 4)

    def test_model_is_nn_module(self): # (Anthropic, 2026)
        """Asserts that xLSTM_TS_Model is a subclass of torch.nn.Module."""
        import torch.nn as nn
        model = xLSTM_TS_Model(input_shape=(10, 2), embedding_dim=16, output_size=1)
        assert isinstance(model, nn.Module)

# directional_loss 
class TestDirectionalLoss: # (Anthropic, 2026)
    """Tests for the directional_loss() function from xLSTM_TS.py."""

    def test_returns_scalar_tensor(self): # (Anthropic, 2026)
        """Asserts that directional_loss returns a zero-dimensional scalar tensor."""
        loss = directional_loss(torch.rand(8, 7), torch.rand(8, 7))
        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"

    def test_loss_is_non_negative(self): # (Anthropic, 2026)
        """Asserts that directional_loss never returns a negative value."""
        loss = directional_loss(torch.rand(8, 7), torch.rand(8, 7))
        assert loss.item() >= 0.0

    def test_perfect_prediction_produces_zero_loss(self): # (Anthropic, 2026)
        """Asserts that identical predictions and targets produce a loss of exactly 0.0."""
        y = torch.rand(4, 7)
        loss = directional_loss(y, y.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_worse_predictions_give_higher_loss(self): # (Anthropic, 2026)
        """Asserts that predictions with a large constant offset produce higher loss than perfect predictions."""
        y_true = torch.rand(8, 7)
        y_perfect = y_true.clone()
        y_bad = y_true + 10.0
        assert directional_loss(y_true, y_bad).item() > directional_loss(y_true, y_perfect).item()
