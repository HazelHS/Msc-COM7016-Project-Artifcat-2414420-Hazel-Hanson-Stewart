# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
Shared training utilities for both xLSTM-TS and MEMD-TCN training scripts, containing model-agnostic 
helpers for data preparation, scaling, sequence building, and DataLoader creation.
"""

import sys
from pathlib import Path

# Add the project root directory to sys.path so dependency_checker can be found
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_path))

from dependency_checker import *

# Generic Dataset
class SequenceDataset(Dataset):
    """PyTorch Dataset for sliding-window time series sequences.

    Args:
        X: Feature array of shape [N, sequence_length, n_features].
        y: Target array of shape [N, forecast_horizon], or [N] for scalar targets.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None: # (Anthropic, 2026)
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int: # (Anthropic, 2026)
        """Returns the total number of sliding-window samples."""
        return len(self.X)

    def __getitem__(self, idx: int): # (Anthropic, 2026)
        """Returns the (features, target) pair at index idx.

        Args:
            idx: Integer sample index.

        Returns:
            A tuple (X[idx], y[idx]) of float32 tensors.
        """
        return self.X[idx], self.y[idx]

# Data splitting
def temporal_train_val_test_split(
    df: pd.DataFrame,
    split_ratios: tuple = (0.70, 0.15, 0.15),
) -> tuple: # (Anthropic, 2026)
    """Splits a DataFrame into train, validation, and test subsets in temporal order.

    No shuffling is performed; row order is fully preserved to prevent
    data leakage across splits.

    Args:
        df: Input DataFrame where each row represents one timestep.
        split_ratios: A tuple (train_frac, val_frac, test_frac) whose values
          must sum to no more than 1.0.

    Returns:
        A tuple (train_df, val_df, test_df) of non-overlapping DataFrame slices.
    """
    n         = len(df)
    train_end = int(n * split_ratios[0])
    val_end   = train_end + int(n * split_ratios[1])
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

# Scaling
def fit_and_scale(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_columns: list,
    target_col: str,
) -> dict: # (Anthropic, 2026)
    """Fits MinMaxScaler instances on training data and transforms all three splits.

    Fits separate scalers for features and the target column using only the
    training split, then applies those fitted scalers to all three splits to
    prevent data leakage.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        feature_columns: List of feature column names; must not include target_col.
        target_col: Name of the target column to scale separately.

    Returns:
        A dict with keys feature_scaler, target_scaler, X_train, y_train,
        X_val, y_val, X_test, and y_test. Scaled features are DataFrames and
        scaled targets are Series, both with their original indices preserved.
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler  = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(train_df[feature_columns])
    target_scaler.fit(train_df[[target_col]])

    def _transform(df_split: pd.DataFrame): # (Anthropic, 2026)
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

# Sequence builder
def make_sequences(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    sequence_length: int,
    forecast_horizon: int = 7,
) -> tuple:
    """Builds sliding-window (X_seq, y_seq) arrays from scaled feature and target data.

    Args:
        X_df: Scaled feature DataFrame of shape [T, n_features].
        y_series: Scaled target Series of length T.
        sequence_length: Number of past timesteps in each input window.
        forecast_horizon: Number of future timesteps to predict per sample.

    Returns:
        A tuple (X_seq, y_seq) where X_seq has shape
        [N, sequence_length, n_features] and y_seq has shape
        [N, forecast_horizon], with N = T - sequence_length - forecast_horizon + 1.
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

# DataLoaders
def create_dataloaders(
    train_dataset: Dataset,
    val_dataset:   Dataset,
    test_dataset:  Dataset,
    batch_size: int = 16,
) -> dict: # (Anthropic, 2026)
    """Wraps three Datasets in DataLoaders with temporal ordering preserved.

    shuffle=False is enforced for all loaders to maintain the original
    timestep order, which is required for valid time-series evaluation.

    Args:
        train_dataset: Training Dataset.
        val_dataset: Validation Dataset.
        test_dataset: Test Dataset.
        batch_size: Number of samples per mini-batch.

    Returns:
        A dict with keys train_loader, val_loader, and test_loader, each a
        DataLoader configured with the given batch_size and shuffle=False.
    """
    def _loader(ds: Dataset) -> DataLoader: # (Anthropic, 2026)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return {
        "train_loader": _loader(train_dataset),
        "val_loader":   _loader(val_dataset),
        "test_loader":  _loader(test_dataset),
    }
