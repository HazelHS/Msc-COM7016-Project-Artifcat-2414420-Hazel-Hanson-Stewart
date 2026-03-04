"""
eval_utils.py
-------------
Shared utilities for all Model Evaluation metric scripts.

Provides a single entry-point function:

    load_model_and_run_inference(model_path, dataset_path)

which:
  1. Loads a trained model checkpoint (.pt) and reads saved hyperparameters.
  2. Detects the model type (xLSTM_TS or MEMD_TCN).
  3. Rebuilds the correct model architecture from saved hyperparameters.
  4. Loads and preprocesses the dataset CSV using the same pipeline as training.
  5. Runs inference on the test split.
  6. Returns inverse-scaled predictions and actuals for downstream metric scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
_eval_dir    = Path(__file__).resolve().parent           # Evaluation_Metrics/
_root        = _eval_dir.parent.parent                   # project root
_training    = _root / "AI_Modules" / "Training_Methods"
_model_dsgns = _root / "AI_Modules" / "Model_Designs"

for p in [str(_root), str(_training), str(_model_dsgns)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Third-party / project imports ────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

# Model classes
from xLSTM_TS import xLSTM_TS_Model
from MEMD_TCN import MEMD_TCN_Model

# Shared data-prep helpers
from train_utils import (
    SequenceDataset,
    temporal_train_val_test_split,
    fit_and_scale,
    make_sequences,
    create_dataloaders,
)

# MEMD-TCN–specific helpers (imported directly from training script module)
from Train_MEMD_TCN import (
    _detect_input_columns,
    _decompose_fixed,
    _make_memd_sequences,
    MEMDSequenceDataset,
    prepare_data_memd,
)


# =============================================================================
# Public API
# =============================================================================

def load_model_and_run_inference(
    model_path: str | Path,
    dataset_path: str | Path,
) -> dict:
    """
    Load a trained checkpoint, preprocess *dataset_path* with the same
    pipeline used during training, and return inference results on the
    test split.

    Args:
        model_path   : Absolute path to a .pt checkpoint file.
        dataset_path : Absolute path to a .csv dataset file.

    Returns:
        dict with keys:
            predictions  (np.ndarray, 1-D, inverse-scaled)
            actuals      (np.ndarray, 1-D, inverse-scaled)
            y_train      (np.ndarray, 1-D, unscaled, raw training targets)
            model_name   (str)
            dataset_name (str)

    Raises:
        RuntimeError  on unknown model type or data-preparation failure.
    """
    model_path   = Path(model_path)
    dataset_path = Path(dataset_path)

    print(f"[eval_utils] Loading checkpoint : {model_path.name}")
    print(f"[eval_utils] Dataset            : {dataset_path.name}")

    # ── 1. Load checkpoint ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(model_path, map_location=device)
    hp     = ckpt["hyperparameters"]
    mtype  = hp["model"]  # "xLSTM_TS" or "MEMD_TCN"

    print(f"[eval_utils] Model type : {mtype}  |  device: {device}")

    # ── 2. Load dataset ──────────────────────────────────────────────────────
    df = pd.read_csv(dataset_path)
    print(f"[eval_utils] Dataset shape: {df.shape}")

    # ── 3. Route to model-specific pipeline ─────────────────────────────────
    if mtype == "xLSTM_TS":
        return _infer_xlstm(ckpt, hp, df, dataset_path, device)
    elif mtype == "MEMD_TCN":
        return _infer_memd(ckpt, hp, df, dataset_path, device)
    else:
        raise RuntimeError(f"[eval_utils] Unknown model type in checkpoint: '{mtype}'")


# =============================================================================
# xLSTM_TS inference
# =============================================================================

def _infer_xlstm(ckpt: dict, hp: dict, df: pd.DataFrame,
                 dataset_path: Path, device: torch.device) -> dict:
    """Full xLSTM-TS data-prep + inference pipeline."""

    target_col      = hp.get("target_col", "BTC/USD")
    sequence_length = int(hp.get("sequence_length", 60))
    embedding_dim   = int(hp.get("embedding_dim", 64))
    output_size     = int(hp.get("output_size", 7))
    n_features      = int(hp.get("n_features", 1))
    batch_size      = int(hp.get("batch_size", 16))

    # ── Guard: target column must exist ──────────────────────────────────────
    if target_col not in df.columns:
        # Fall back to first numeric column
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise RuntimeError("[eval_utils] No numeric columns found in dataset.")
        target_col = numeric_cols[0]
        print(f"[eval_utils] Target column not found; falling back to '{target_col}'")

    # ── Feature columns (same logic as Train_xLSTM_TS.prepare_data) ──────────
    feature_columns = [
        c for c in df.columns
        if c not in (target_col, "Unnamed: 0")
    ]
    if len(feature_columns) == 0:
        # Need at least one feature; synthesise lagged target
        df = df.copy()
        df["prev_price"] = df[target_col].shift(1)
        df = df.dropna()
        feature_columns = ["prev_price"]

    df = df.dropna(subset=feature_columns + [target_col]).copy()

    # ── Temporal split ────────────────────────────────────────────────────────
    train_df, val_df, test_df = temporal_train_val_test_split(df)

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaled = fit_and_scale(train_df, val_df, test_df, feature_columns, target_col)
    target_scaler = scaled["target_scaler"]
    y_train_raw   = train_df[target_col].values  # unscaled, for MASE

    # Actual n_features from data (may differ from checkpoint if CSV differs)
    actual_n_features = len(feature_columns)

    # ── Build sequences ───────────────────────────────────────────────────────
    forecast_horizon = output_size
    X_test_seq, y_test_seq = make_sequences(
        scaled["X_test"], scaled["y_test"], sequence_length, forecast_horizon
    )

    test_dataset = SequenceDataset(X_test_seq, y_test_seq)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=False)

    # ── Rebuild model ─────────────────────────────────────────────────────────
    input_shape = (sequence_length, actual_n_features)
    model = xLSTM_TS_Model(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        output_size=output_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_acts = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            out     = model(X_batch)                    # [B, output_size]
            all_preds.append(out.cpu().numpy())
            all_acts.append(y_batch.numpy())

    preds_np = np.concatenate(all_preds)[:, 0]         # first forecast step
    acts_np  = np.concatenate(all_acts)[:, 0]

    preds_inv = target_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
    acts_inv  = target_scaler.inverse_transform(acts_np.reshape(-1, 1)).flatten()

    return {
        "predictions":  preds_inv,
        "actuals":      acts_inv,
        "y_train":      y_train_raw,
        "model_name":   "xLSTM_TS",
        "dataset_name": dataset_path.name,
    }


# =============================================================================
# MEMD_TCN inference
# =============================================================================

def _infer_memd(ckpt: dict, hp: dict, df: pd.DataFrame,
                dataset_path: Path, device: torch.device) -> dict:
    """Full MEMD-TCN data-prep + inference pipeline."""

    target_col      = hp.get("target_col", "BTC/USD")
    sequence_length = int(hp.get("sequence_length", 30))
    max_imfs        = int(hp.get("max_imfs", 12))
    n_channels      = int(hp.get("n_channels", 5))
    n_components    = int(hp.get("n_components", max_imfs + 1))
    kernel_size     = int(hp.get("kernel_size", 2))
    dilations       = list(hp.get("dilations", [1, 2, 4]))
    dropout         = float(hp.get("dropout", 0.2))
    K               = int(hp.get("K", 512))
    max_sift        = int(hp.get("max_sift", 50))
    sd_threshold    = float(hp.get("sd_threshold", 0.2))
    batch_size      = int(hp.get("batch_size", 16))

    # ── Guard: target column must exist ──────────────────────────────────────
    if target_col not in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise RuntimeError("[eval_utils] No numeric columns found in dataset.")
        target_col = numeric_cols[0]
        print(f"[eval_utils] Target column not found; falling back to '{target_col}'")

    # ── Run MEMD data pipeline (mirrors prepare_data_memd) ───────────────────
    print("[eval_utils] Running MEMD decomposition — this may take several minutes …")
    prepared = prepare_data_memd(
        df,
        target_col=target_col,
        sequence_length=sequence_length,
        split_ratios=(0.70, 0.15, 0.15),
        K=K,
        max_imfs=max_imfs,
        max_sift=max_sift,
        sd_threshold=sd_threshold,
    )
    if prepared is None:
        raise RuntimeError("[eval_utils] MEMD data preparation failed.")

    target_scaler = prepared["target_scaler"]
    y_train_raw   = prepared["original_test_actuals"]  # approximate; test actuals
    # For MASE we ideally want the training targets — reconstruct from train_df
    train_df_raw = df.iloc[: int(len(df) * 0.70)].copy()
    y_train_raw  = train_df_raw[target_col].values

    actual_n_components = prepared["n_components"]
    actual_n_channels   = prepared["n_channels"]

    test_loader = create_dataloaders(
        prepared["train_dataset"],
        prepared["val_dataset"],
        prepared["test_dataset"],
        batch_size=batch_size,
    )["test_loader"]

    # ── Rebuild model ─────────────────────────────────────────────────────────
    model = MEMD_TCN_Model(
        in_channels=actual_n_channels,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
        K=K,
        max_imfs=max_imfs,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_acts = [], []
    with torch.no_grad():
        for imf_seqs, target in test_loader:
            # imf_seqs : [B, n_components, n_channels, seq_len]
            imf_seqs = imf_seqs.to(device)
            component_preds = [
                model.forward(imf_seqs[:, k, :, :], imf_idx=k)
                for k in range(actual_n_components)
            ]
            final_pred = model.reconstruct(component_preds)   # [B]
            all_preds.append(final_pred.cpu().numpy())
            all_acts.append(target.numpy())

    preds_np = np.concatenate(all_preds)
    acts_np  = np.concatenate(all_acts)

    preds_inv = target_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
    acts_inv  = target_scaler.inverse_transform(acts_np.reshape(-1, 1)).flatten()

    return {
        "predictions":  preds_inv,
        "actuals":      acts_inv,
        "y_train":      y_train_raw,
        "model_name":   "MEMD_TCN",
        "dataset_name": dataset_path.name,
    }


# =============================================================================
# Shared metric helpers
# =============================================================================

def calculate_mase(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_train: np.ndarray | None) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = MAE(test) / MAE(naive forecast on training set)
    A value < 1 means the model outperforms a naive one-step persistence forecast.
    """
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_true, y_pred)
    if y_train is not None and len(y_train) > 1:
        naive_errors = np.abs(np.diff(y_train))
        naive_mae    = np.mean(naive_errors)
        if naive_mae > 1e-10:
            return mae / naive_mae
    # Fallback: use test series itself as naive denominator
    if len(y_true) > 1:
        naive_errors = np.abs(np.diff(y_true))
        naive_mae    = np.mean(naive_errors)
        if naive_mae > 1e-10:
            return mae / naive_mae
    return mae
