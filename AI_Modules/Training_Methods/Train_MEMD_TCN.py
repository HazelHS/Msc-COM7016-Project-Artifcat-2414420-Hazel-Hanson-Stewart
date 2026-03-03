"""
Train_MEMD_TCN.py
-----------------
Self-contained training script for the MEMD-TCN model.
Based on Yao et al. (2023) / Rehman & Mandic (2010) / Bai et al. (2018).

Pipeline:
  1. Load CSV  →  auto-detect input channels from numeric columns
  2. Temporal split  →  min-max scale each split independently
  3. MEMD decomposition per split  →  fixed n_components = max_imfs + 1 (zero-padded)
  4. Build MEMDSequenceDataset storing [n_components, n_channels, seq_len] per sample
  5. Joint optimisation of all TCN blocks via reconstruction MSE
  6. Save best model checkpoint

Launched by the AI Training Method stage in Model Designer via:
    python Train_MEMD_TCN.py [CLI args]

All hyperparameter arguments are populated by the TrainingConfigureWindow
in Interface_Modules/main_window.py (MEMD-TCN panel).
"""

import sys
import argparse
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
root_path          = Path(__file__).resolve().parent.parent.parent
model_designs_path = Path(__file__).resolve().parent.parent / "Model_Designs"
training_path      = Path(__file__).resolve().parent

sys.path.insert(0, str(root_path))           # for dependency_checker
sys.path.insert(0, str(model_designs_path))  # for MEMD_TCN
sys.path.insert(0, str(training_path))       # for train_utils

# ── Shared & model imports ────────────────────────────────────────────────────
from train_utils import (
    temporal_train_val_test_split,
    create_dataloaders,
)
from dependency_checker import *   # torch, nn, optim, F, Dataset, DataLoader,
                                   # ReduceLROnPlateau, pd, np, tqdm, os

from MEMD_TCN import MEMD_TCN_Model, memd

import torch
import pandas as pd


# =============================================================================
# Custom Dataset for pre-computed MEMD components
# =============================================================================

class MEMDSequenceDataset(Dataset):
    """
    PyTorch Dataset for MEMD-TCN training.

    Stores pre-computed, fixed-length IMF + residual windows so that MEMD
    (which is expensive) is run only once per data split.

    Args:
        imf_sequences : ndarray [N, n_components, n_channels, seq_len]
                        One entry per sliding window.
        targets       : ndarray [N]
                        Scaled closing price at the next timestep for each window.
    """
    def __init__(self, imf_sequences: np.ndarray, targets: np.ndarray) -> None:
        # Store as float32 tensors for direct GPU transfer
        self.imf_sequences = torch.FloatTensor(imf_sequences)  # [N, C, ch, L]
        self.targets        = torch.FloatTensor(targets)         # [N]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.imf_sequences[idx], self.targets[idx]


# =============================================================================
# 1 — OHLCV column detection
# =============================================================================

def _detect_input_columns(df: pd.DataFrame, target_col: str) -> list:
    """
    Attempt to identify Open, High, Low, Close, Volume columns via name
    pattern matching.  Falls back to all numeric columns if fewer than 5
    are identified.

    The target_col is always treated as the Close channel.
    Returns an ordered list of column names to feed as MEMD input channels.
    """
    all_numeric = [
        c for c in df.select_dtypes(include="number").columns
        if c not in ("Unnamed: 0",)
    ]

    patterns = {
        "open":   ["open"],
        "high":   ["high"],
        "low":    ["low"],
        "volume": ["volume", "vol"],
    }

    found: dict = {"close": target_col}
    for key, pats in patterns.items():
        for col in all_numeric:
            if col == target_col:
                continue
            col_lower = col.lower()
            if any(pat in col_lower for pat in pats):
                found[key] = col
                break

    # If we have all 5 OHLCV-like columns, return them in canonical order
    if len(found) == 5:
        ohlcv = [
            found["open"], found["high"], found["low"],
            found["close"], found["volume"],
        ]
        print(f"Auto-detected OHLCV columns: {ohlcv}")
        return ohlcv

    # Fallback: all numeric columns (target_col first for consistency)
    ordered = [target_col] + [c for c in all_numeric if c != target_col]
    print(
        f"Could not identify a clean OHLCV set; using {len(ordered)} numeric "
        f"column(s) as input channels."
    )
    if len(ordered) > 10:
        print(
            f"[WARNING] {len(ordered)} channels will make MEMD very slow "
            f"(K={{}}) — consider reducing the dataset to OHLCV columns only."
        )
    return ordered


# =============================================================================
# 2 — MEMD decomposition helper
# =============================================================================

def _decompose_fixed(
    ohlcv_np: np.ndarray,
    max_imfs: int,
    K: int,
    max_sift: int,
    sd_threshold: float,
) -> list:
    """
    Run MEMD on *ohlcv_np* and return exactly ``max_imfs + 1`` component
    arrays (zero-padding missing IMF slots so the TCN model dimensions are
    always consistent across train / val / test splits).

    Args:
        ohlcv_np     : ndarray [n_channels, T] — scaled OHLCV, channels-first
        max_imfs     : Maximum IMFs requested (matches model architecture)
        K            : Number of Hammersley direction vectors for MEMD
        max_sift     : Maximum sifting iterations per IMF
        sd_threshold : Huang et al. (2003) SD stopping criterion

    Returns:
        all_components : list of (max_imfs + 1) ndarrays [n_channels, T]
                         Indices 0..max_imfs-1 → IMFs, index max_imfs → residual
    """
    n_channels, T = ohlcv_np.shape
    imfs, residual = memd(
        ohlcv_np,
        K=K,
        max_imfs=max_imfs,
        max_sift=max_sift,
        sd_threshold=sd_threshold,
    )

    # Pad shorter IMF lists with zero arrays to reach exactly max_imfs entries
    while len(imfs) < max_imfs:
        imfs.append(np.zeros((n_channels, T), dtype=np.float32))

    all_components = imfs + [residual]          # length = max_imfs + 1
    return all_components


# =============================================================================
# 3 — Sequence builder for one MEMD-decomposed split
# =============================================================================

def _make_memd_sequences(
    all_components: list,
    y_scaled: np.ndarray,
    close_channel_idx: int,
    sequence_length: int,
) -> tuple:
    """
    Build sliding-window IMF sequences and scalar close-price targets.

    For each window starting at index *i*:
      - X[i] : ndarray [n_components, n_channels, sequence_length]
      - y[i] : scalar — scaled close price at timestep i + sequence_length

    Args:
        all_components     : list of n_components arrays [n_channels, T]
        y_scaled           : 1-D ndarray of scaled close prices [T]
        close_channel_idx  : Which channel in each component is the close price
                             (used only for documentation; TCN sees all channels)
        sequence_length    : Look-back window length

    Returns:
        (X_seq, y_seq) : ndarrays [N, n_components, n_channels, seq_len] and [N]
    """
    n_components = len(all_components)
    n_channels   = all_components[0].shape[0]
    T            = all_components[0].shape[1]

    max_start = T - sequence_length - 1   # need at least 1 future step for target
    if max_start <= 0:
        raise ValueError(
            f"Not enough timesteps ({T}) for sequence_length={sequence_length}. "
            "Reduce sequence_length or use more data."
        )

    X_seq = np.empty(
        (max_start, n_components, n_channels, sequence_length), dtype=np.float32
    )
    y_seq = np.empty(max_start, dtype=np.float32)

    for i in range(max_start):
        for j, comp in enumerate(all_components):
            X_seq[i, j] = comp[:, i : i + sequence_length]       # [n_ch, L]
        y_seq[i] = y_scaled[i + sequence_length]                  # next-step close

    return X_seq, y_seq


# =============================================================================
# 4 — Data preparation
# =============================================================================

def prepare_data_memd(
    df: pd.DataFrame,
    target_col: str     = "BTC/USD",
    sequence_length: int = 30,
    split_ratios: tuple  = (0.70, 0.15, 0.15),
    K: int               = 512,
    max_imfs: int        = 12,
    max_sift: int        = 50,
    sd_threshold: float  = 0.2,
) -> dict | None:
    """
    Full data pipeline for MEMD-TCN:
      1. Auto-detect input (OHLCV) channels
      2. Temporal split
      3. Fit MinMaxScaler on training split; transform all three splits
      4. Run MEMD on each split independently (no future leakage)
      5. Build MEMDSequenceDatasets

    Args:
        df              : Raw time-series DataFrame
        target_col      : Close-price column name
        sequence_length : Look-back window size
        split_ratios    : (train_frac, val_frac, test_frac)
        K               : MEMD direction vectors (Rehman & Mandic eq. 3.2)
        max_imfs        : Maximum IMF components per decomposition
        max_sift        : Max sifting iterations (Huang et al. 2003)
        sd_threshold    : Sifting stopping criterion

    Returns:
        dict with datasets, scalers, n_components, n_channels, close_channel_idx
        Returns None on error.
    """
    if df.empty:
        print("[ERROR] Input DataFrame is empty.")
        return None

    # ── Detect input channels ─────────────────────────────────────────
    input_columns = _detect_input_columns(df, target_col)
    n_channels    = len(input_columns)
    # Index of the close/target channel within the ordered column list
    try:
        close_channel_idx = input_columns.index(target_col)
    except ValueError:
        close_channel_idx = 0

    n_components = max_imfs + 1     # fixed: max_imfs IMFs + 1 residual slot

    print(f"Input shape     : {df.shape}")
    print(f"Input channels  : {input_columns}")
    print(f"Close channel   : index {close_channel_idx} ({target_col})")
    print(f"n_components    : {n_components}  (max_imfs={max_imfs} + 1 residual)")

    # Drop rows with any NaN in the used columns
    df = df[input_columns].dropna().copy()

    # ── Temporal split ────────────────────────────────────────────────
    train_df, val_df, test_df = temporal_train_val_test_split(df, split_ratios)
    print(f"Splits — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    # ── Fit scaler on training data only  →  transform all splits ────
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df.values)

    train_scaled = scaler.transform(train_df.values)   # [T_tr, n_channels]
    val_scaled   = scaler.transform(val_df.values)
    test_scaled  = scaler.transform(test_df.values)

    # Build a target_scaler (single channel = close price col) for evaluation
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(train_df[[target_col]].values)   # fit same slice

    # Scaled close prices as 1-D arrays for target construction
    y_train = train_scaled[:, close_channel_idx]
    y_val   = val_scaled[:,   close_channel_idx]
    y_test  = test_scaled[:,  close_channel_idx]

    # Original (unscaled) test actuals for evaluation
    original_test_actuals = test_df[target_col].values

    # ── MEMD decomposition (channels-first: [n_channels, T]) ─────────
    print("\nRunning MEMD decomposition on each split …")
    print("  (This may take several minutes for K=512 directions)")

    def _decompose_split(scaled_data: np.ndarray, label: str) -> list:
        ohlcv_np = scaled_data.T.astype(float)   # [n_channels, T]
        print(f"  Decomposing {label} split  [{ohlcv_np.shape}] …")
        return _decompose_fixed(ohlcv_np, max_imfs, K, max_sift, sd_threshold)

    components_train = _decompose_split(train_scaled, "train")
    components_val   = _decompose_split(val_scaled,   "val")
    components_test  = _decompose_split(test_scaled,  "test")

    # ── Build sliding-window datasets ─────────────────────────────────
    print("\nBuilding IMF sequences …")
    seq_progress = tqdm(total=3, desc="Sequence preparation")

    X_train_seq, y_train_seq = _make_memd_sequences(
        components_train, y_train, close_channel_idx, sequence_length
    )
    seq_progress.update(1)

    X_val_seq, y_val_seq = _make_memd_sequences(
        components_val, y_val, close_channel_idx, sequence_length
    )
    seq_progress.update(1)

    X_test_seq, y_test_seq = _make_memd_sequences(
        components_test, y_test, close_channel_idx, sequence_length
    )
    seq_progress.update(1)
    seq_progress.close()

    print(
        f"Sequence shapes — "
        f"train: {X_train_seq.shape}  val: {X_val_seq.shape}  test: {X_test_seq.shape}"
    )

    return {
        "train_dataset":         MEMDSequenceDataset(X_train_seq, y_train_seq),
        "val_dataset":           MEMDSequenceDataset(X_val_seq,   y_val_seq),
        "test_dataset":          MEMDSequenceDataset(X_test_seq,  y_test_seq),
        "target_scaler":         target_scaler,
        "original_test_actuals": original_test_actuals,
        "n_components":          n_components,
        "n_channels":            n_channels,
        "close_channel_idx":     close_channel_idx,
    }


# =============================================================================
# 5 — Model setup
# =============================================================================

def setup_model_memd(
    n_channels: int,
    n_components: int,
    kernel_size: int   = 2,
    dilations: list    = None,
    dropout: float     = 0.2,
    K: int             = 512,
    max_imfs: int      = 12,
    learning_rate: float = 1e-4,
) -> dict:
    """
    Instantiate MEMD_TCN_Model, Adam optimiser, and ReduceLROnPlateau scheduler.

    The model is built with n_components TCN blocks (max_imfs IMF slots + 1
    residual) so that each pre-computed component has a dedicated TCN.

    Args:
        n_channels    : Number of OHLCV-like input channels
        n_components  : Total components = max_imfs + 1 (incl. residual)
        kernel_size   : TCN causal-conv kernel size (Yao et al.: k=2)
        dilations     : TCN dilation sequence  (Yao et al.: [1, 2, 4])
        dropout       : TCN dropout rate
        K             : Number of MEMD direction vectors (stored in model)
        max_imfs      : Maximum IMFs (stored in model)
        learning_rate : Initial Adam learning rate

    Returns:
        dict with model, optimizer, scheduler, device.
    """
    if dilations is None:
        dilations = [1, 2, 4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MEMD_TCN_Model(
        in_channels=n_channels,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
        K=K,
        max_imfs=max_imfs,       # tcn_blocks are built with max_imfs + 1 slots
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
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
        "model":        model,
        "optimizer":    optimizer,
        "scheduler":    scheduler,
        "device":       device,
        "n_components": n_components,
    }


# =============================================================================
# 6 — Training loop
# =============================================================================

def train_model_memd(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    n_components: int,
    epochs: int             = 100,
    early_stopping_patience: int = 20,
    max_grad_norm: float    = 1.0,
):
    """
    Joint MEMD-TCN training loop.

    For each batch the pipeline is:
      1. Forward each IMF component through its dedicated TCN  →  scalar [B]
      2. Reconstruct final prediction by summing all components  →  [B]
      3. MSE loss against scaled close price target
      4. Backward + gradient clip + Adam step

    All TCN blocks share one Adam optimiser (joint optimisation).

    Args:
        model                   : MEMD_TCN_Model
        train_loader            : DataLoader for MEMDSequenceDataset
        val_loader              : DataLoader for MEMDSequenceDataset
        optimizer               : Adam optimiser
        scheduler               : ReduceLROnPlateau scheduler
        device                  : cuda / cpu
        n_components            : Number of IMF + residual components
        epochs                  : Maximum training epochs
        early_stopping_patience : Epochs without val-loss improvement before stop
        max_grad_norm           : Gradient clipping norm

    Returns:
        model : Best model (restored from checkpoint)
    """
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_model_state  = None

    for epoch in range(epochs):
        # ── Training phase ────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_mae  = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [train]",
            leave=False,
            unit="batch",
        )
        for imf_seqs, target in pbar:
            # imf_seqs : [B, n_components, n_channels, seq_len]
            # target   : [B]
            imf_seqs, target = imf_seqs.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward each component through its dedicated TCN
            component_preds = [
                model.forward(imf_seqs[:, k, :, :], imf_idx=k)  # → [B]
                for k in range(n_components)
            ]
            # Reconstruct: element-wise sum  →  [B]
            final_pred = model.reconstruct(component_preds)

            loss = F.mse_loss(final_pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_loss += loss.item()
            train_mae  += F.l1_loss(final_pred.detach(), target).item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        pbar.close()
        train_loss /= len(train_loader)
        train_mae  /= len(train_loader)

        # ── Validation phase ──────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0

        with torch.no_grad():
            for imf_seqs, target in val_loader:
                imf_seqs, target = imf_seqs.to(device), target.to(device)
                component_preds  = [
                    model.forward(imf_seqs[:, k, :, :], imf_idx=k)
                    for k in range(n_components)
                ]
                final_pred = model.reconstruct(component_preds)
                val_loss  += F.mse_loss(final_pred, target).item()
                val_mae   += F.l1_loss(final_pred, target).item()

        val_loss /= len(val_loader)
        val_mae  /= len(val_loader)

        print(
            f"Epoch {epoch + 1:>4}/{epochs}  "
            f"train_loss={train_loss:.5f}  train_mae={train_mae:.5f}  "
            f"val_loss={val_loss:.5f}  val_mae={val_mae:.5f}"
        )

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

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest model restored  (val_loss = {best_val_loss:.6f})")

    return model


# =============================================================================
# 7 — CLI argument parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments supplied by the TrainingConfigureWindow (MEMD-TCN panel).
    Argument names match exactly what _build_cli_args() passes for MEMD scripts.
    """
    parser = argparse.ArgumentParser(
        description="Train the MEMD-TCN model (Yao et al. 2023)."
    )
    # ── Dataset ───────────────────────────────────────────────────────
    parser.add_argument("--dataset",    type=str, default=None,
                        help="Absolute path to the input dataset CSV.")
    parser.add_argument("--target_col", type=str, default="BTC/USD",
                        help="Column name of the closing price target.")
    # ── Sequence ──────────────────────────────────────────────────────
    parser.add_argument("--sequence_length", type=int, default=30,
                        help="Look-back window (lag) length in timesteps.")
    # ── Training loop ─────────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=100,
                        help="Training epochs (applied jointly across all TCNs).")
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # ── TCN architecture ──────────────────────────────────────────────
    parser.add_argument("--kernel_size", type=int,   default=2,
                        help="Causal conv kernel size (Yao et al.: k=2).")
    parser.add_argument("--dilations",   type=str,   default="1,2,4",
                        help="Comma-separated dilation factors (Yao et al.: 1,2,4).")
    parser.add_argument("--dropout",     type=float, default=0.2)
    # ── MEMD parameters ───────────────────────────────────────────────
    parser.add_argument("--K",            type=int,   default=512,
                        help="Number of Hammersley direction vectors for MEMD.")
    parser.add_argument("--max_imfs",     type=int,   default=12,
                        help="Maximum IMFs to extract (Yao et al.: ≤12).")
    parser.add_argument("--max_sift",     type=int,   default=50,
                        help="Maximum sifting iterations per IMF.")
    parser.add_argument("--sd_threshold", type=float, default=0.2,
                        help="Huang et al. (2003) SD sifting stopping criterion.")
    # ── Output ────────────────────────────────────────────────────────
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "Trained_Model_Files"),
        help="Directory where the trained .pt file will be saved.",
    )
    return parser.parse_args()


# =============================================================================
# 8 — Orchestration
# =============================================================================

def train_and_return_model(args: argparse.Namespace = None):
    """
    Full MEMD-TCN pipeline: load → prepare → decompose → build → train → save.

    Args:
        args : Parsed CLI args (uses parse_args() if None)

    Returns:
        (model, test_loader, target_scaler, device)  or  None on failure
    """
    if args is None:
        args = parse_args()

    # Parse dilations string  →  list[int]
    dilations = [int(d.strip()) for d in args.dilations.split(",")]

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
    print(f"  MEMD-TCN Training  —  Yao et al. (2023)")
    print(f"{'='*60}")
    print(f"  Dataset          : {data_path.name}")
    print(f"  Target column    : {args.target_col}")
    print(f"  Sequence length  : {args.sequence_length}")
    print(f"  Epochs           : {args.epochs}  |  Batch size: {args.batch_size}")
    print(f"  Learning rate    : {args.learning_rate}")
    print(f"  Kernel size      : {args.kernel_size}  |  Dilations: {dilations}")
    print(f"  Dropout          : {args.dropout}")
    print(f"  MEMD K           : {args.K}  |  max_imfs: {args.max_imfs}")
    print(f"  max_sift         : {args.max_sift}  |  sd_threshold: {args.sd_threshold}")
    print(f"  Save directory   : {args.save_dir}")
    print(f"{'='*60}\n")

    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")

    # ── Prepare data ──────────────────────────────────────────────────
    print("\n--- Preparing & decomposing data ---")
    prepared = prepare_data_memd(
        df,
        target_col=args.target_col,
        sequence_length=args.sequence_length,
        split_ratios=(0.70, 0.15, 0.15),        # fixed splits for MEMD — no GUI args
        K=args.K,
        max_imfs=args.max_imfs,
        max_sift=args.max_sift,
        sd_threshold=args.sd_threshold,
    )
    if prepared is None:
        return None

    n_components = prepared["n_components"]
    n_channels   = prepared["n_channels"]

    loaders = create_dataloaders(
        prepared["train_dataset"],
        prepared["val_dataset"],
        prepared["test_dataset"],
        batch_size=args.batch_size,
    )

    # ── Set up model ──────────────────────────────────────────────────
    print("\n--- Setting up MEMD-TCN model ---")
    model_setup = setup_model_memd(
        n_channels=n_channels,
        n_components=n_components,
        kernel_size=args.kernel_size,
        dilations=dilations,
        dropout=args.dropout,
        K=args.K,
        max_imfs=args.max_imfs,
        learning_rate=args.learning_rate,
    )
    model     = model_setup["model"]
    device    = model_setup["device"]
    optimizer = model_setup["optimizer"]
    scheduler = model_setup["scheduler"]

    # ── Train ─────────────────────────────────────────────────────────
    print("\n--- Training ---")
    train_model_memd(
        model,
        loaders["train_loader"],
        loaders["val_loader"],
        optimizer,
        scheduler,
        device,
        n_components=n_components,
        epochs=args.epochs,
        early_stopping_patience=max(10, args.epochs // 10),
    )

    print("\nTraining complete.")

    # ── Save ──────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_stem   = data_path.stem
    model_filename = (
        f"MEMD_TCN"
        f"_ep{args.epochs}"
        f"_seq{args.sequence_length}"
        f"_imfs{args.max_imfs}"
        f"_{dataset_stem}"
        f".pt"
    )
    save_path = save_dir / model_filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": {
                "model":           "MEMD_TCN",
                "n_channels":      n_channels,
                "n_components":    n_components,
                "max_imfs":        args.max_imfs,
                "kernel_size":     args.kernel_size,
                "dilations":       dilations,
                "dropout":         args.dropout,
                "K":               args.K,
                "max_sift":        args.max_sift,
                "sd_threshold":    args.sd_threshold,
                "sequence_length": args.sequence_length,
                "epochs_trained":  args.epochs,
                "batch_size":      args.batch_size,
                "learning_rate":   args.learning_rate,
                "target_col":      args.target_col,
                "dataset":         str(data_path),
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
