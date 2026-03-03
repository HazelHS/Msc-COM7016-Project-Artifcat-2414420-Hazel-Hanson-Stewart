"""
Train_xLSTM_TS.py
-----------------
Self-contained training script for the xLSTM-TS model.
Based on Lopez et al. (2024).

Launched by the AI Training Method stage in Model Designer via:
    python Train_xLSTM_TS.py [CLI args]

All hyperparameter arguments are populated by the TrainingConfigureWindow
in Interface_Modules/main_window.py (xLSTM-TS panel).
"""

import sys
import argparse
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
root_path          = Path(__file__).resolve().parent.parent.parent
model_designs_path = Path(__file__).resolve().parent.parent / "Model_Designs"
training_path      = Path(__file__).resolve().parent

sys.path.insert(0, str(root_path))           # for dependency_checker
sys.path.insert(0, str(model_designs_path))  # for xLSTM_TS
sys.path.insert(0, str(training_path))       # for train_utils

# ── Shared & model imports ────────────────────────────────────────────────────
from train_utils import (
    SequenceDataset,
    temporal_train_val_test_split,
    fit_and_scale,
    make_sequences,
    create_dataloaders,
)
from dependency_checker import *   # torch, nn, optim, F, pd, np, tqdm, os …
from xLSTM_TS import xLSTM_TS_Model, directional_loss, TrainingProgressTracker

import torch
import pandas as pd


# =============================================================================
# 1 — Data preparation
# =============================================================================

def prepare_data(
    df: pd.DataFrame,
    target_col: str = "BTC/USD",
    sequence_length: int = 60,
    forecast_horizon: int = 7,
    split_ratios: tuple = (0.70, 0.15, 0.15),
) -> dict | None:
    """
    Prepare data for xLSTM-TS training.

    Args:
        df               : Raw DataFrame with time series data
        target_col       : Column to forecast
        sequence_length  : Look-back window (timesteps)
        forecast_horizon : Steps ahead to predict
        split_ratios     : (train, val, test) fractions

    Returns:
        Dict with datasets, scalers, original actuals, and n_features.
        Returns None on error.
    """
    if df.empty:
        print("[ERROR] Input DataFrame is empty.")
        return None

    print(f"Input shape : {df.shape}")
    print(f"Columns     : {df.columns.tolist()}")

    # Guard: need at least a target column + one feature
    if len(df.columns) <= 1:
        df = df.copy()
        df["prev_price"] = df[target_col].shift(1)
        df = df.dropna()
        print("Added lagged target as a synthetic feature.")

    # Feature columns = everything except target and unnamed index artefacts
    feature_columns = [
        c for c in df.columns if c not in (target_col, "Unnamed: 0")
    ]
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

    # ── Temporal split ────────────────────────────────────────────────
    train_df, val_df, test_df = temporal_train_val_test_split(df, split_ratios)
    print(f"Splits — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    # ── Scale ─────────────────────────────────────────────────────────
    scaled = fit_and_scale(train_df, val_df, test_df, feature_columns, target_col)
    feature_scaler = scaled["feature_scaler"]
    target_scaler  = scaled["target_scaler"]

    # Keep raw training targets for MASE calculation downstream
    y_train_original       = train_df[target_col].values
    original_test_actuals  = test_df[target_col].values

    # ── Build sliding-window sequences ────────────────────────────────
    print("\nBuilding sequences …")
    seq_progress = tqdm(total=3, desc="Sequence preparation")

    X_train_seq, y_train_seq = make_sequences(
        scaled["X_train"], scaled["y_train"], sequence_length, forecast_horizon
    )
    seq_progress.update(1)

    X_val_seq, y_val_seq = make_sequences(
        scaled["X_val"], scaled["y_val"], sequence_length, forecast_horizon
    )
    seq_progress.update(1)

    X_test_seq, y_test_seq = make_sequences(
        scaled["X_test"], scaled["y_test"], sequence_length, forecast_horizon
    )
    seq_progress.update(1)
    seq_progress.close()

    print(f"Sequence shapes — train: {X_train_seq.shape}  val: {X_val_seq.shape}  test: {X_test_seq.shape}")

    return {
        "train_dataset":        SequenceDataset(X_train_seq, y_train_seq),
        "val_dataset":          SequenceDataset(X_val_seq,   y_val_seq),
        "test_dataset":         SequenceDataset(X_test_seq,  y_test_seq),
        "feature_scaler":       feature_scaler,
        "target_scaler":        target_scaler,
        "y_train_original":     y_train_original,
        "original_test_actuals": original_test_actuals,
        "n_features":           len(feature_columns),
    }


# =============================================================================
# 2 — Model setup
# =============================================================================

def setup_model(
    n_features: int,
    sequence_length: int = 60,
    embedding_dim: int = 64,
    output_size: int = 7,
    learning_rate: float = 1e-4,
) -> dict:
    """
    Instantiate xLSTM-TS model, Adam optimiser, and ReduceLROnPlateau scheduler.

    Args:
        n_features      : Number of input feature channels
        sequence_length : Look-back window length
        embedding_dim   : xLSTM embedding dimension
        output_size     : Forecast horizon
        learning_rate   : Initial Adam learning rate

    Returns:
        Dict with model, optimizer, scheduler, device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = xLSTM_TS_Model(
        input_shape=(sequence_length, n_features),
        embedding_dim=embedding_dim,
        output_size=output_size,
    ).to(device)

    # Adam with Lopez et al. (2024) paper parameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=True,
    )

    return {
        "model":     model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device":    device,
    }


# =============================================================================
# 3 — Training loop
# =============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs: int = 200,
    early_stopping_patience: int = 30,
):
    """
    Full training loop with:
      - directional_loss (MSE + directional BCE, Lopez et al. 2024)
      - gradient clipping (max_norm=1.0)
      - early stopping
      - best-model restore

    Args:
        model                   : xLSTM-TS PyTorch model
        train_loader            : Training DataLoader
        val_loader              : Validation DataLoader
        optimizer               : Adam optimiser
        scheduler               : ReduceLROnPlateau scheduler
        device                  : cuda / cpu
        epochs                  : Maximum number of training epochs
        early_stopping_patience : Epochs without val improvement before stopping

    Returns:
        model : Best model (restored from checkpoint)
    """
    progress_tracker = TrainingProgressTracker(epochs, len(train_loader))

    best_val_loss         = float("inf")
    epochs_no_improve     = 0
    best_model_state      = None
    max_grad_norm         = 1.0

    for epoch in range(epochs):
        # ── Training phase ────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_mae  = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss   = directional_loss(target, output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            batch_mae  = F.l1_loss(output, target).item()
            train_loss += loss.item()
            train_mae  += batch_mae
            progress_tracker.update_batch(loss.item(), batch_mae)

        train_loss /= len(train_loader)
        train_mae  /= len(train_loader)

        # ── Validation phase ──────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output    = model(data)
                val_loss += directional_loss(target, output).item()
                val_mae  += F.l1_loss(output, target).item()

        val_loss /= len(val_loader)
        val_mae  /= len(val_loader)

        # ── Early stopping ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_model_state  = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

        scheduler.step(val_loss)
        progress_tracker.update_epoch(train_loss, val_loss, train_mae)

    progress_tracker.close()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest model restored  (val_loss = {best_val_loss:.6f})")

    return model


# =============================================================================
# 4 — CLI argument parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments supplied by the TrainingConfigureWindow (xLSTM-TS panel).
    Argument names must match exactly what _build_cli_args() passes.
    """
    parser = argparse.ArgumentParser(
        description="Train the xLSTM-TS model (Lopez et al. 2024)."
    )
    # ── Dataset ───────────────────────────────────────────────────────
    parser.add_argument("--dataset",    type=str,   default=None,
                        help="Absolute path to the input dataset CSV.")
    parser.add_argument("--target_col", type=str,   default="BTC/USD",
                        help="Column name of the forecast target.")
    # ── Sequence / split ──────────────────────────────────────────────
    parser.add_argument("--sequence_length",  type=int,   default=60,
                        help="Look-back window length (timesteps).")
    parser.add_argument("--train_split",      type=float, default=0.70,
                        help="Fraction of data for training.")
    parser.add_argument("--val_split",        type=float, default=0.15,
                        help="Fraction of data for validation.")
    # ── Training loop ─────────────────────────────────────────────────
    parser.add_argument("--epochs",                   type=int,   default=200)
    parser.add_argument("--batch_size",               type=int,   default=16)
    parser.add_argument("--early_stopping_patience",  type=int,   default=30)
    # ── Model architecture ────────────────────────────────────────────
    parser.add_argument("--learning_rate",  type=float, default=1e-4,
                        help="Initial Adam learning rate.")
    parser.add_argument("--embedding_dim",  type=int,   default=64,
                        help="Embedding dimension for xLSTM blocks.")
    parser.add_argument("--output_size",    type=int,   default=7,
                        help="Forecast horizon in days.")
    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "Trained_Model_Files"),
        help="Directory where the trained .pt file will be saved.",
    )
    return parser.parse_args()


# =============================================================================
# 5 — Orchestration
# =============================================================================

def train_and_return_model(args: argparse.Namespace = None):
    """
    Full pipeline: load data → prepare → build model → train → save.

    Args:
        args : Parsed CLI args (uses parse_args() if None)

    Returns:
        (model, test_loader, target_scaler, device)  or  None on failure
    """
    if args is None:
        args = parse_args()

    # ── Resolve dataset path ──────────────────────────────────────────
    if args.dataset:
        data_path = Path(args.dataset)
    else:
        data_path = (
            root_path
            / "Dataset_Modules"
            / "dataset_output"
            / "2015-2025_dataset_denoised.csv"
        )

    print(f"\n{'='*60}")
    print(f"  xLSTM-TS Training  —  Lopez et al. (2024)")
    print(f"{'='*60}")
    print(f"  Dataset          : {data_path.name}")
    print(f"  Target column    : {args.target_col}")
    print(f"  Sequence length  : {args.sequence_length}")
    print(f"  Forecast horizon : {args.output_size} days")
    print(f"  Epochs           : {args.epochs}  |  Batch size: {args.batch_size}")
    print(f"  Learning rate    : {args.learning_rate}  |  Embedding dim: {args.embedding_dim}")
    print(f"  Early stop pat.  : {args.early_stopping_patience}")
    print(f"  Splits           : train={args.train_split}  val={args.val_split}")
    print(f"  Save directory   : {args.save_dir}")
    print(f"{'='*60}\n")

    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")

    split_ratios = (
        args.train_split,
        args.val_split,
        round(1.0 - args.train_split - args.val_split, 6),
    )

    # ── Prepare data ──────────────────────────────────────────────────
    print("\n--- Preparing data ---")
    prepared = prepare_data(
        df,
        target_col=args.target_col,
        sequence_length=args.sequence_length,
        forecast_horizon=args.output_size,
        split_ratios=split_ratios,
    )
    if prepared is None:
        return None

    loaders = create_dataloaders(
        prepared["train_dataset"],
        prepared["val_dataset"],
        prepared["test_dataset"],
        batch_size=args.batch_size,
    )

    # ── Set up model ──────────────────────────────────────────────────
    print("\n--- Setting up model ---")
    model_setup = setup_model(
        n_features=prepared["n_features"],
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        output_size=args.output_size,
        learning_rate=args.learning_rate,
    )
    model     = model_setup["model"]
    device    = model_setup["device"]
    optimizer = model_setup["optimizer"]
    scheduler = model_setup["scheduler"]

    # ── Train ─────────────────────────────────────────────────────────
    print("\n--- Training ---")
    train_model(
        model,
        loaders["train_loader"],
        loaders["val_loader"],
        optimizer,
        scheduler,
        device,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
    )

    print("\nTraining complete.")

    # ── Save ──────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_stem  = data_path.stem
    model_filename = (
        f"xLSTM_TS"
        f"_ep{args.epochs}"
        f"_seq{args.sequence_length}"
        f"_emb{args.embedding_dim}"
        f"_{dataset_stem}"
        f".pt"
    )
    save_path = save_dir / model_filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": {
                "model":                   "xLSTM_TS",
                "sequence_length":         args.sequence_length,
                "embedding_dim":           args.embedding_dim,
                "output_size":             args.output_size,
                "n_features":              prepared["n_features"],
                "epochs_trained":          args.epochs,
                "batch_size":              args.batch_size,
                "learning_rate":           args.learning_rate,
                "early_stopping_patience": args.early_stopping_patience,
                "train_split":             args.train_split,
                "val_split":               args.val_split,
                "target_col":              args.target_col,
                "dataset":                 str(data_path),
            },
        },
        save_path,
    )
    print(f"Model saved -> {save_path}")

    return model, loaders["test_loader"], prepared["target_scaler"], device


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    train_and_return_model()
