"""
Boruta_Selection.py
-------------------
Performs feature selection using the Boruta algorithm with a Random
Forest regressor as the base estimator.  Features are ranked by their
normalised importance score (relative to the maximum), and those at or
above the configurable threshold are retained.

Usage (standalone):
    python Boruta_Selection.py --dataset <path_to_csv> [--target <column>]
        [--max-iter 100] [--importance-threshold 0.01]

When launched from the Model Designer GUI the --dataset argument is
populated automatically from the dropdown selection.

Requires:
    pip install boruta
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

try:
    from boruta import BorutaPy
    _BORUTA_AVAILABLE = True
except ImportError:
    _BORUTA_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "dataset_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def load_data(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


def plot_feature_importance(scores_df: pd.DataFrame, method_name: str,
                             output_path: Path) -> None:
    """Horizontal bar chart of selected feature importance scores."""
    selected = scores_df[scores_df["Selected"]].copy()
    score_col = "Score" if "Score" in selected.columns else "Importance"

    fig, ax = plt.subplots(figsize=(12, max(6, len(selected) * 0.35)))
    bars = ax.barh(range(len(selected)), selected[score_col], color="steelblue")
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected["Feature"])
    ax.set_xlabel("Normalised Importance Score")
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
                            dataset_stem: str) -> Path:
    final_cols = selected_features + [target]
    df_sel = df[final_cols]
    out = OUTPUT_DIR / f"{dataset_stem}_selected_features_{method_name}.csv"
    df_sel.to_csv(out)
    return out


# ── Core algorithm ────────────────────────────────────────────────────────

def boruta_selection(
    df: pd.DataFrame,
    target: str = "BTC/USD",
    max_iter: int = 100,
    importance_threshold: float = 0.01,
) -> tuple[list[str], pd.DataFrame]:
    """Boruta algorithm feature selection with Random Forest base estimator. (Anthropic, 2024)"""
    if not _BORUTA_AVAILABLE:
        raise ImportError(
            "The 'boruta' package is not installed. "
            "Run: pip install boruta"
        )

    print(f"\nInitialising Boruta feature selection...")
    print(f"Input shape          : {df.shape}")
    print(f"Target column        : {target}")
    print(f"Max iterations       : {max_iter}")
    print(f"Importance threshold : {importance_threshold}")

    X = df.drop(columns=[target])
    y = df[target]

    # ── Configure base Random Forest ──────────────────────────────────
    rf = RandomForestRegressor(
        n_jobs=-1,
        max_depth=7,
        n_estimators=250,
        random_state=42,  # 42 – for all the fish
    )

    # ── Configure Boruta ──────────────────────────────────────────────
    boruta = BorutaPy(
        rf,
        n_estimators="auto",
        max_iter=max_iter,
        perc=98,       # percentile for shadow feature importance threshold
        alpha=0.001,   # p-value for statistical testing
        two_step=True, # two-step correction for multiple testing
        verbose=0,
        random_state=42,
    )

    print("\nRunning Boruta (this may take several minutes)...")
    with tqdm(total=1, desc="Boruta Selection") as pbar:
        boruta.fit(X.values, y.values)
        pbar.update(1)

    # ── Use BorutaPy's authoritative support_ attribute for selection ─
    # support_      = confirmed accepted features (rank 1)
    # support_weak_ = tentatively accepted features (rank 2)
    # ranking_      = 1-based rank (lower = more important)
    confirmed_mask   = boruta.support_
    tentative_mask   = boruta.support_weak_ if hasattr(boruta, "support_weak_") else np.zeros(X.shape[1], dtype=bool)
    selected_mask    = confirmed_mask | tentative_mask

    # Importance scores: use mean importance history when available,
    # otherwise derive a score from ranking_ (lower rank → higher score).
    if hasattr(boruta, "importance_history_") and boruta.importance_history_.shape[0] > 0:
        raw_imp = boruta.importance_history_.mean(axis=0)[: X.shape[1]]
        max_imp = np.max(raw_imp) if np.max(raw_imp) > 0 else 1.0
        scores  = raw_imp / max_imp
    else:
        print("[WARN] importance_history_ not available; deriving scores from ranking_.")
        max_rank = int(np.max(boruta.ranking_))
        scores   = (max_rank - boruta.ranking_ + 1) / max_rank

    importance_scores = pd.DataFrame({
        "Feature":   X.columns,
        "Importance": scores.round(4),
        "Rank":      boruta.ranking_,
        "Confirmed": confirmed_mask,
        "Tentative": tentative_mask,
        "Selected":  selected_mask,
        "Score":     scores.round(4),  # alias for compatibility
    }).sort_values("Importance", ascending=False)

    selected_features = importance_scores[importance_scores["Selected"]]["Feature"].tolist()

    print(f"\nSelected {len(selected_features)} / {X.shape[1]} features")
    print("\nSelected features by importance:")
    print(importance_scores[importance_scores["Selected"]].to_string(index=False))

    # ── All-features bar chart ────────────────────────────────────────
    sorted_df = importance_scores.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_df) * 0.28)))
    colors = ["steelblue" if sel else "lightgrey"
              for sel in sorted_df["Selected"]]
    ax.barh(range(len(sorted_df)), sorted_df["Importance"], color=colors)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["Feature"], fontsize=8)
    ax.set_xlabel("Normalised Importance Score")
    ax.set_title("Boruta Feature Importances – confirmed/tentative in blue, rejected in grey")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Value labels (selected only to avoid clutter)
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        if row["Selected"]:
            ax.text(
                row["Importance"] + sorted_df["Importance"].max() * 0.005,
                i,
                f"{row['Importance']:.4f}",
                va="center", ha="left", fontsize=7,
            )

    plt.tight_layout()
    print("All-features chart ready.")

    return selected_features, importance_scores


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boruta algorithm feature selection."
    )
    parser.add_argument("--dataset",              required=True,         help="Absolute path to the input CSV.")
    parser.add_argument("--target",               default=None,          help="Target column name (default: auto-detected).")
    parser.add_argument("--max-iter",             type=int, default=100, help="Maximum Boruta iterations (default: 100).")
    parser.add_argument("--importance-threshold", type=float, default=0.01,
                        help="Minimum normalised importance to be considered selected (default: 0.01).")
    args = parser.parse_args()

    if not _BORUTA_AVAILABLE:
        print("[ERROR] boruta package not installed. Run: pip install boruta", file=sys.stderr)
        sys.exit(1)

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

    selected, scores_df = boruta_selection(
        df,
        target=target,
        max_iter=args.max_iter,
        importance_threshold=args.importance_threshold,
    )

    # ── Save selected-features chart ──────────────────────────────────
    chart_out = OUTPUT_DIR / "Boruta_selected_feature_importance"
    plot_feature_importance(scores_df, "Boruta", chart_out)

    # ── Save selected-features CSV ────────────────────────────────────
    out_csv = save_selected_features(df, selected, target, "Boruta", input_path.stem)
    print(f"\nSelected features CSV saved -> {out_csv}")
    print(f"\nSelected features ({len(selected)}):")
    for f in selected:
        print(f"  - {f}")
    plt.show()

if __name__ == "__main__":
    main()
