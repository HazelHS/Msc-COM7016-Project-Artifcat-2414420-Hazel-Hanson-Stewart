"""
train_utils.py
--------------
Shared training utilities for both xLSTM-TS and MEMD-TCN training scripts.

Contains model-agnostic helpers:
  - SequenceDataset        : generic PyTorch Dataset for (X_seq, y_seq) pairs
  - temporal_train_val_test_split : order-preserving DataFrame split
  - fit_and_scale          : fit MinMaxScalers on train, transform all splits
  - make_sequences         : sliding-window sequence builder
  - create_dataloaders     : wraps three Datasets in DataLoaders
"""

import sys
from pathlib import Path

# Add the project root directory to sys.path so dependency_checker can be found
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_path))

from dependency_checker import *   # torch, nn, optim, F, Dataset, DataLoader,
                                   # ReduceLROnPlateau, pd, np, MinMaxScaler, tqdm, os


# =============================================================================
# Generic Dataset
# =============================================================================

class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sliding-window time series sequences.

    Args:
        X : ndarray [N, sequence_length, n_features]
        y : ndarray [N, forecast_horizon]  or  [N] for scalar targets
    """
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# =============================================================================
# Data splitting
# =============================================================================

def temporal_train_val_test_split(
    df: pd.DataFrame,
    split_ratios: tuple = (0.70, 0.15, 0.15),
) -> tuple:
    """
    Split a DataFrame into train / validation / test subsets while
    preserving temporal order (no shuffling).

    Args:
        df           : Input DataFrame (rows = timesteps)
        split_ratios : (train_frac, val_frac, test_frac)  — must sum to ≤ 1

    Returns:
        (train_df, val_df, test_df)
    """
    n         = len(df)
    train_end = int(n * split_ratios[0])
    val_end   = train_end + int(n * split_ratios[1])
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


# =============================================================================
# Scaling
# =============================================================================

def fit_and_scale(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_columns: list,
    target_col: str,
) -> dict:
    """
    Fit MinMaxScaler(0,1) on training data then transform all three splits.

    Args:
        train_df        : Training DataFrame
        val_df          : Validation DataFrame
        test_df         : Test DataFrame
        feature_columns : List of feature column names (must not include target)
        target_col      : Name of the target column

    Returns:
        dict with keys:
            feature_scaler, target_scaler,
            X_train, y_train, X_val, y_val, X_test, y_test
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler  = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(train_df[feature_columns])
    target_scaler.fit(train_df[[target_col]])

    def _transform(df_split: pd.DataFrame):
        X = pd.DataFrame(
            feature_scaler.transform(df_split[feature_columns]),
            columns=feature_columns,
            index=df_split.index,
        )
        y = pd.Series(
            target_scaler.transform(df_split[[target_col]]).ravel(),
            index=df_split.index,
        )
        return X, y

    X_train, y_train = _transform(train_df)
    X_val,   y_val   = _transform(val_df)
    X_test,  y_test  = _transform(test_df)

    return {
        "feature_scaler": feature_scaler,
        "target_scaler":  target_scaler,
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }


# =============================================================================
# Sequence builder
# =============================================================================

def make_sequences(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    sequence_length: int,
    forecast_horizon: int = 7,
) -> tuple:
    """
    Build (X_seq, y_seq) arrays using a sliding window.

    Args:
        X_df             : Scaled feature DataFrame  [T, n_features]
        y_series         : Scaled target Series      [T]
        sequence_length  : Look-back window size
        forecast_horizon : Number of future steps to predict

    Returns:
        X_seq : ndarray [N, sequence_length, n_features]
        y_seq : ndarray [N, forecast_horizon]
    """
    X_seq, y_seq = [], []
    limit = len(X_df) - sequence_length - forecast_horizon + 1
    for i in range(limit):
        X_seq.append(X_df.iloc[i : i + sequence_length].values)
        y_seq.append(
            y_series.iloc[i + sequence_length : i + sequence_length + forecast_horizon]
            .values
            .reshape(-1)
        )
    return np.array(X_seq), np.array(y_seq)


# =============================================================================
# DataLoaders
# =============================================================================

def create_dataloaders(
    train_dataset: Dataset,
    val_dataset:   Dataset,
    test_dataset:  Dataset,
    batch_size: int = 16,
) -> dict:
    """
    Wrap three Datasets in DataLoaders.
    shuffle=False everywhere to preserve temporal ordering.

    Args:
        train_dataset : Training Dataset
        val_dataset   : Validation Dataset
        test_dataset  : Test Dataset
        batch_size    : Mini-batch size

    Returns:
        dict with keys: train_loader, val_loader, test_loader
    """
    def _loader(ds: Dataset) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return {
        "train_loader": _loader(train_dataset),
        "val_loader":   _loader(val_dataset),
        "test_loader":  _loader(test_dataset),
    }
