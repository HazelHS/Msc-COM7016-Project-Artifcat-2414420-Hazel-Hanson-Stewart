"""
eval_regression_metrics.py
---------------------------
Evaluation script — Regression Metrics (MAE & RMSE).

Loads a trained model checkpoint and a dataset CSV, runs inference on the
test split, then displays a bar chart of regression error metrics:
  • MAE  — Mean Absolute Error  (lower is better)
  • RMSE — Root Mean Squared Error  (lower is better)

Usage (standalone):
    python eval_regression_metrics.py --model <path/to/model.pt> --dataset <path/to/data.csv>

Usage (via Model Designer):
    Launched automatically with --model and --dataset flags by the
    "Model Evaluation Method" stage in Model Designer.
"""

import argparse
import sys
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

# ── Project imports ───────────────────────────────────────────────────────────
from eval_utils import load_model_and_run_inference

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the regression metrics evaluation script.

    Returns:
        argparse.Namespace with ``--model`` and ``--dataset`` paths.
    """
    p = argparse.ArgumentParser(
        description="Evaluate regression error metrics (MAE, RMSE) for a trained model."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Absolute path to the trained .pt checkpoint file.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Absolute path to the dataset .csv file for evaluation.",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Load checkpoint, run inference on the test split, and display regression metric bars."""
    args = parse_args()

    # ── Load model and run inference ─────────────────────────────────────────
    result = load_model_and_run_inference(args.model, args.dataset)

    predictions  = result["predictions"]
    actuals      = result["actuals"]
    model_name   = result["model_name"]
    dataset_name = result["dataset_name"]

    # ── Regression metrics ────────────────────────────────────────────────────
    mae  = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # ── Print metrics ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Regression Metrics")
    print(f"  Model  : {model_name}")
    print(f"  Data   : {dataset_name}")
    print(f"{'=' * 55}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"{'=' * 55}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    labels  = ["MAE", "RMSE"]
    values  = [mae, rmse]
    colours = ["#2ecc71", "#27ae60"]   # greens — error metrics

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.6,
                  width=0.45)

    max_val = max(values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylim(0, max_val * 1.25)
    ax.set_title(
        f"Regression Error Metrics\n"
        f"{model_name}  |  {dataset_name}",
        fontsize=13, pad=14,
    )
    ax.set_ylabel("Error (original scale)")
    ax.set_xlabel("Metric")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.01,
        "Note: MAE and RMSE are in the same units as the target variable.\n"
        "Lower values indicate better model performance.",
        ha="center", fontsize=8, style="italic", color="dimgrey",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
