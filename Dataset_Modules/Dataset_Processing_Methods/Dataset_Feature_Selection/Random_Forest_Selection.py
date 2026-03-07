# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The Random_Forest_Selection.py script performs window-based permutation importance feature selection using a
Random Forest regressor.  Optionally applies Bayesian hyperparameter
tuning via BayesSearchCV.

The 75th-percentile of mean importance is used as the selection threshold
(top 25 % most important features).
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# Optional Bayesian search – only required when --tune is passed
try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Categorical
    _SKOPT_AVAILABLE = True
except ImportError:
    _SKOPT_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths 
SCRIPT_DIR  = Path(__file__).resolve().parent
# Output saved alongside the source dataset (dataset_output folder)
OUTPUT_DIR  = SCRIPT_DIR.parents[1] / "dataset_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helpers 
def load_data(filepath: Path) -> pd.DataFrame: # (Anthropic, 2026)
    """Load a CSV dataset from disk.

    Args:
        filepath: Absolute path to the CSV file. The first column is
          used as the DatetimeIndex.

    Returns:
        A DataFrame with a DatetimeIndex and one column per feature.
    """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def plot_feature_importance(scores_df: pd.DataFrame, method_name: str,
                             output_path: Path) -> None: # (Anthropic, 2026)
    """Draw a horizontal bar chart of selected feature importance scores.

    Filters ``scores_df`` to the selected features only, then renders an
    interactive matplotlib bar chart with inline score labels. The chart is
    displayed via ``plt.show()`` by the caller; this function does not save
    to disk.

    Args:
        scores_df: DataFrame with columns ``Feature``, ``Selected``, and
          either ``Score`` or ``Importance``.
        method_name: Human-readable label for the selection method, used
          as the chart title suffix.
        output_path: Intended destination path for the figure. Currently
          unused — saving is left to the caller or the interactive toolbar.
    """
    selected = scores_df[scores_df["Selected"]].copy()
    score_col = "Score" if "Score" in selected.columns else "Importance"

    fig, ax = plt.subplots(figsize=(12, max(6, len(selected) * 0.35)))
    bars = ax.barh(range(len(selected)), selected[score_col], color="steelblue")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected["Feature"])
    ax.set_xlabel("Mean Permutation Importance")
    ax.set_ylabel("Features")
    ax.set_title(f"Feature Importance Scores – {method_name}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + max(selected[score_col]) * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.4f}",
            va="center", ha="left", fontsize=8,
        )

    plt.tight_layout()
    print("Selected-features chart ready.")

def save_selected_features(df: pd.DataFrame, selected_features: list[str],
                            target: str, method_name: str,
                            dataset_stem: str) -> Path: # (Anthropic, 2026)
    """Persist the selected feature subset as a CSV file.

    Writes a CSV containing only the selected feature columns plus the target
    column to ``OUTPUT_DIR/<dataset_stem>_<method_name>_selected.csv``.

    Args:
        df: Full input DataFrame before feature selection.
        selected_features: List of feature column names to retain.
        target: Name of the target column to append to the output.
        method_name: Short label for the selection method (e.g.
          ``'RandomForest'``), used to construct the output filename.
        dataset_stem: Stem of the input filename, used to construct
          the output filename.

    Returns:
        The :class:`pathlib.Path` of the written CSV file.
    """
    final_cols = selected_features + [target]
    out_df = df[final_cols]
    out_path = OUTPUT_DIR / f"{dataset_stem}_{method_name}_selected.csv"
    out_df.to_csv(out_path)
    print(f"[save_selected_features] Saved {out_df.shape} -> {out_path}")
    return out_path

def random_forest_selection(
    df: pd.DataFrame,
    target: str = "BTC/USD",
    window_size: int = 30,
    n_estimators: int = 10,
    perform_tuning: bool = False,
    n_iterations: int = 10,
    n_jobs: int = -1,
) -> tuple[list[str], pd.DataFrame]: # (Anthropic, 2026)
    """Select features using window-based permutation importance with a Random Forest.

    Slides a rolling window over the time-series, fits a Random Forest
    regressor on each window, and measures permutation importance per feature.
    The mean importance across all windows is computed and features at or above
    the 75th percentile are retained (top 25%). Two matplotlib charts are
    prepared during the call — one showing all features with a threshold line,
    and one showing only selected features — but neither is displayed until
    ``plt.show()`` is called by the caller.

    Optionally tunes the Random Forest hyperparameters via ``BayesSearchCV``
    before the importance sweep. If ``scikit-optimize`` is not installed,
    tuning is silently skipped and the default estimator is used.

    Args:
        df: Input DataFrame containing feature and target columns.
        target: Name of the target column (default ``'BTC/USD'``).
        window_size: Number of rows in each rolling window (default ``30``).
        n_estimators: Number of trees per Random Forest (default ``10``).
        perform_tuning: If ``True``, tune hyperparameters with
          ``BayesSearchCV`` before computing importances (default ``False``).
        n_iterations: Number of Bayesian search iterations; only used when
          ``perform_tuning`` is ``True`` (default ``10``).
        n_jobs: Number of parallel jobs for fitting and the window sweep.
          ``-1`` uses all available CPUs (default ``-1``).

    Returns:
        A tuple ``(selected_features, importance_scores)``, where
        ``selected_features`` is a list of retained column names and
        ``importance_scores`` is a DataFrame with columns ``Feature``,
        ``Score``, and ``Selected`` for every input feature.

    Raises:
        ValueError: If ``df`` is empty.
    """
    print(f"\nInitialising Random Forest feature selection...")
    print(f"Input shape   : {df.shape}")
    print(f"Target column : {target}")
    print(f"Window size   : {window_size}")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    start_time = time.time()
    X = df.drop(columns=[target])
    y = df[target]
    feature_names = X.columns.tolist()
    n_windows = len(X) - window_size
    print(f"Features      : {len(feature_names)}")
    print(f"Windows       : {n_windows}")

    # Optional Bayesian hyperparameter tuning 
    if perform_tuning:
        if not _SKOPT_AVAILABLE:
            print("[WARN] scikit-optimize not installed – skipping tuning.")
            perform_tuning = False
        else:
            print("\nRunning Bayesian hyperparameter tuning...")
            sample_slice = slice(0, min(window_size * 3, len(X)))
            X_s, y_s = X.iloc[sample_slice], y.iloc[sample_slice]
            param_space = {
                "n_estimators":     Integer(10, 100),
                "max_depth":        Integer(3, 20),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 5),
                "max_features":     Categorical(["sqrt", "log2", None]),
                "bootstrap":        Categorical([True, False]),
            }
            bayes_search = BayesSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=n_jobs),
                param_space,
                n_iter=n_iterations,
                cv=3,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs,
                verbose=1,
                random_state=42,
            )
            bayes_search.fit(X_s, y_s)
            best_params = bayes_search.best_params_
            print(f"Best params : {best_params}")
            rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=n_jobs)
    
    if not perform_tuning:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            random_state=42,
            n_jobs=n_jobs,
        )

    def process_window(i: int) -> np.ndarray: # (Anthropic, 2026)
        """Compute permutation importances for one rolling window.

        Fits a cloned Random Forest on the window slice, records the baseline
        R² score, then shuffles each feature column in turn and measures the
        drop in R² as that feature's importance.

        Args:
            i: Starting row index of the window within the full dataset.

        Returns:
            A 1-D NumPy array of length ``n_features`` containing the
            permutation importance for each feature in the window.
        """
        window_X = X.iloc[i : i + window_size]
        window_y = y.iloc[i : i + window_size]
        rf_local = clone(rf)
        rf_local.fit(window_X, window_y)
        baseline = rf_local.score(window_X, window_y)
        importances = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            X_perm = window_X.copy()
            X_perm.iloc[:, j] = np.random.permutation(X_perm.iloc[:, j])
            importances[j] = baseline - rf_local.score(X_perm, window_y)
        return importances

    print("\nProcessing windows in parallel...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_window)(i)
        for i in tqdm(range(n_windows), desc="Processing windows")
    )

    importance_matrix = np.array(results)
    mean_importances = importance_matrix.mean(axis=0)
    threshold = np.quantile(mean_importances, 0.75)
    selected_features = [
        feature_names[i]
        for i, imp in enumerate(mean_importances)
        if imp > threshold
    ]

    results_df = pd.DataFrame({
        "Feature":  feature_names,
        "Score":    mean_importances,
        "Selected": mean_importances > threshold,
    }).sort_values("Score", ascending=False)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Selected {len(selected_features)} / {len(feature_names)} features "
          f"(threshold = {threshold:.6f}  [75th percentile])")

    # All-features bar chart (with threshold line)
    sorted_idx = np.argsort(mean_importances)
    pos = np.arange(len(sorted_idx)) + 0.5
    fig, ax = plt.subplots(figsize=(12, max(6, len(feature_names) * 0.28)))
    ax.barh(pos, mean_importances[sorted_idx], align="center",
            color=["steelblue" if mean_importances[i] > threshold else "lightgrey"
                   for i in sorted_idx])
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(feature_names)[sorted_idx], fontsize=8)
    ax.axvline(x=threshold, color="red", linestyle="--",
               label=f"Threshold (75th pct = {threshold:.5f})")
    ax.set_xlabel("Mean Permutation Importance")
    ax.set_title("Feature Importances – Random Forest (selected in blue, rejected in grey)")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    print("All-features chart ready.")

    return selected_features, results_df

# Entry point 
def main() -> None: # (Anthropic, 2026)
    """Run Random Forest feature selection from the command line and save outputs.

    Reads the dataset specified by ``--dataset``, auto-detects or accepts a
    ``--target`` column, runs :func:`random_forest_selection`, saves the
    selected-features chart and the filtered CSV to ``OUTPUT_DIR``, prints the
    list of selected features, then opens the interactive matplotlib window.

    Raises:
        SystemExit: If the input file does not exist or the target column is
          not found in the dataset.
    """
    parser = argparse.ArgumentParser(
        description="Window-based Random Forest permutation importance feature selection."
    )
    parser.add_argument("--dataset",     required=True,       help="Absolute path to the input CSV.")
    parser.add_argument("--target",      default=None,        help="Target column name (default: auto-detected).")
    parser.add_argument("--window-size", type=int, default=30, help="Rolling window size (default: 30).")
    parser.add_argument("--n-estimators",type=int, default=10, help="Number of trees (default: 10).")
    parser.add_argument("--tune",        action="store_true", help="Enable Bayesian hyperparameter tuning.")
    parser.add_argument("--n-iter",      type=int, default=10, help="Tuning iterations (default: 10).")
    args = parser.parse_args()

    input_path = Path(args.dataset)
    if not input_path.is_file():
        print(f"[ERROR] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {input_path}")
    df = load_data(input_path)
    print(f"Shape: {df.shape}   Columns: {list(df.columns[:8])}{'...' if df.shape[1] > 8 else ''}")

    # Auto-detect target column
    target = args.target
    if target is None:
        target = df.columns[0]
        print(f"[INFO] No --target given; defaulting to first column: '{target}'")
    if target not in df.columns:
        print(f"[ERROR] Target column '{target}' not in dataset.", file=sys.stderr)
        sys.exit(1)

    selected, scores_df = random_forest_selection(
        df,
        target=target,
        window_size=args.window_size,
        n_estimators=args.n_estimators,
        perform_tuning=args.tune,
        n_iterations=args.n_iter,
    )

    # Save selected-features chart
    chart_out = OUTPUT_DIR / "RF_selected_feature_importance"
    plot_feature_importance(scores_df, "Random Forest", chart_out)

    # Save selected-features CSV 
    out_csv = save_selected_features(df, selected, target, "RandomForest", input_path.stem)
    print(f"\nSelected features CSV saved -> {out_csv}")
    print(f"\nSelected features ({len(selected)}):")
    for f in selected:
        print(f"  - {f}")
    plt.show()

if __name__ == "__main__":
    main()
