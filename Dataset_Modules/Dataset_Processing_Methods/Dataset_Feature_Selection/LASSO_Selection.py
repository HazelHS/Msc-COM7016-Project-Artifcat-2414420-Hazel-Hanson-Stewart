# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The LASSO_Selection.py script performs feature selection using LASSO (Least Absolute Shrinkage and
Selection Operator) regularisation.  Features with non-zero LASSO
coefficients after fitting are retained.

Cross-validation (LassoCV) is used by default to find the optimal
regularisation strength alpha automatically.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "dataset_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helpers 
def load_data(filepath: Path) -> pd.DataFrame: # (Anthropic, 2026)
    """Load a CSV dataset from disk.

    The first column of the CSV is interpreted as a DatetimeIndex.

    Args:
        filepath: Absolute path to the CSV file.

    Returns:
        A DataFrame with a DatetimeIndex and one column per feature.
    """
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def plot_feature_importance(scores_df: pd.DataFrame, method_name: str,
                             output_path: Path) -> None: # (Anthropic, 2026)
    """Draw a horizontal bar chart of selected feature importance scores.

    Renders only the selected features, using their absolute LASSO
    coefficients as bar lengths.  The chart is displayed interactively
    and is not written to disk.

    Args:
        scores_df: DataFrame with columns Feature, Selected, and either
            Score or Importance containing per-feature coefficient data.
        method_name: Human-readable name of the selection method, used
            as the chart title suffix.
        output_path: Reserved for a future file-save option; currently
            unused.
    """
    selected = scores_df[scores_df["Selected"]].copy()
    score_col = "Score" if "Score" in selected.columns else "Importance"

    fig, ax = plt.subplots(figsize=(12, max(6, len(selected) * 0.35)))
    bars = ax.barh(range(len(selected)), selected[score_col], color="steelblue")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected["Feature"])
    ax.set_xlabel("|LASSO Coefficient|")
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

    Writes a CSV containing the selected feature columns plus the target
    column to OUTPUT_DIR.  The output filename follows the pattern
    <dataset_stem>_<method_name>_selected.csv.

    Args:
        df: Full input DataFrame before feature selection.
        selected_features: List of column names to retain.
        target: Name of the target column to append to the output.
        method_name: Short label for the selection method, e.g. "LASSO".
        dataset_stem: Stem of the input filename, used to construct the
            output filename.

    Returns:
        The Path of the written CSV file.
    """
    final_cols = selected_features + [target]
    out_df = df[final_cols]
    out_path = OUTPUT_DIR / f"{dataset_stem}_{method_name}_selected.csv"
    out_df.to_csv(out_path)
    print(f"[save_selected_features] Saved {out_df.shape} -> {out_path}")
    return out_path

def lasso_feature_selection(
    df: pd.DataFrame,
    target: str = "BTC/USD",
    alpha: str | float = "auto",
) -> tuple[list[str], pd.DataFrame]: # (Anthropic, 2026)
    """Select features using LASSO regularisation.

    Fits a LASSO (L1-penalised) linear model and retains features with
    non-zero coefficients.  When alpha is "auto", five-fold
    cross-validation (LassoCV) selects the optimal regularisation
    strength before the final model is fitted.  All features are
    standardised beforehand so the penalty is scale-independent.  Two
    matplotlib figures are produced: one for selected features only, and
    one for all features with selected ones highlighted in blue.

    Args:
        df: Input DataFrame containing both feature columns and the
            target column.
        target: Name of the target column.  Defaults to "BTC/USD".
        alpha: Regularisation strength.  Pass "auto" to select the
            optimal value via LassoCV, or supply a positive float to fix
            the penalty.

    Returns:
        A tuple (selected_features, importance_scores), where
        selected_features is a list of retained column names and
        importance_scores is a DataFrame with columns Feature, Score,
        Raw_Coef, and Selected for every input feature, sorted by
        descending absolute coefficient.
    """
    print(f"\nInitialising LASSO feature selection...")
    print(f"Input shape   : {df.shape}")
    print(f"Target column : {target}")

    X = df.drop(columns=[target])
    y = df[target]

    # Standardise – essential for LASSO so that penalty is scale-independent
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Features standardised ({X.shape[1]} columns).")

    # Optimal alpha via cross-validation 
    if alpha == "auto":
        print("Running LassoCV (5-fold) to find optimal alpha...")
        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10_000)
        lasso_cv.fit(X_scaled, y)
        alpha = lasso_cv.alpha_
        print(f"Optimal alpha : {alpha:.6f}")
    else:
        alpha = float(alpha)
        print(f"Using fixed alpha : {alpha}")

    # Fit final LASSO 
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10_000)
    lasso.fit(X_scaled, y)

    # Non-zero coefficients → selected features
    selected_features = X.columns[lasso.coef_ != 0].tolist()
    importance_scores = pd.DataFrame({
        "Feature":  X.columns,
        "Score":    np.abs(lasso.coef_),
        "Raw_Coef": lasso.coef_,
        "Selected": lasso.coef_ != 0,
    }).sort_values("Score", ascending=False)

    print(f"\nSelected {len(selected_features)} / {X.shape[1]} features")
    print("\nTop selected features and |coefficient|:")
    print(importance_scores[importance_scores["Selected"]].head(10).to_string(index=False))

    # All-features bar chart
    sorted_df = importance_scores.sort_values("Score", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_df) * 0.28)))
    colors = ["steelblue" if sel else "lightgrey"
              for sel in sorted_df["Selected"]]
    bars = ax.barh(range(len(sorted_df)), sorted_df["Score"], color=colors)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["Feature"], fontsize=8)
    ax.set_xlabel("|LASSO Coefficient| (standardised)")
    ax.set_title(f"LASSO Feature Importances  [alpha={alpha:.6f}]  (selected in blue)")

    # Value labels
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(
                w + sorted_df["Score"].max() * 0.005,
                bar.get_y() + bar.get_height() / 2.0,
                f"{w:.4f}",
                va="center", ha="left", fontsize=7,
            )

    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    print("All-features chart ready.")

    return selected_features, importance_scores

# Entry point 
def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments, run LASSO selection, and save outputs.

    Reads the dataset at the path given by --dataset, auto-detects or
    accepts the target column via --target, runs lasso_feature_selection,
    saves the selected-features bar chart and filtered CSV to OUTPUT_DIR,
    then displays all matplotlib figures.
    """
    parser = argparse.ArgumentParser(
        description="LASSO regularisation-based feature selection."
    )
    parser.add_argument("--dataset", required=True, help="Absolute path to the input CSV.")
    parser.add_argument("--target",  default=None,  help="Target column name (default: auto-detected).")
    parser.add_argument("--alpha",   default="auto",
                        help="Regularisation strength: 'auto' (LassoCV) or a float value (default: auto).")
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

    alpha_arg: str | float = args.alpha if args.alpha == "auto" else float(args.alpha)
    selected, scores_df = lasso_feature_selection(df, target=target, alpha=alpha_arg)

    # Save selected-features chart
    chart_out = OUTPUT_DIR / "LASSO_selected_feature_importance"
    plot_feature_importance(scores_df, "LASSO", chart_out)

    # Save selected-features CSV
    out_csv = save_selected_features(df, selected, target, "LASSO", input_path.stem)
    print(f"\nSelected features CSV saved -> {out_csv}")
    print(f"\nSelected features ({len(selected)}):")
    for f in selected:
        print(f"  - {f}")
    plt.show()

if __name__ == "__main__":
    main()
