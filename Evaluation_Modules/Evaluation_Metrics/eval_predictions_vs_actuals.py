"""
eval_predictions_vs_actuals.py
-------------------------------
Evaluation script — Predictions vs Actuals Time-Series Chart.

Loads a trained model checkpoint and a dataset CSV, runs inference on the
test split, then displays an overlaid line chart comparing:
  • Actual values    (solid blue line)
  • Predicted values (dashed red line)

A sample of up to 200 test-set steps is shown for visual clarity.

Usage (standalone):
    python eval_predictions_vs_actuals.py --model <path/to/model.pt> --dataset <path/to/data.csv>

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
    """Parse CLI arguments for the predictions-vs-actuals plot script.

    Returns:
        argparse.Namespace with ``--model``, ``--dataset``, and ``--sample`` values.
    """
    p = argparse.ArgumentParser(
        description="Plot predictions vs actuals for a trained model on a test dataset."
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
    p.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Maximum number of test-set timesteps to display (default: 200).",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Load checkpoint, run inference on the test split, and plot predictions vs actuals."""
    args = parse_args()

    # ── Load model and run inference ─────────────────────────────────────────
    result = load_model_and_run_inference(args.model, args.dataset)

    predictions  = result["predictions"]
    actuals      = result["actuals"]
    model_name   = result["model_name"]
    dataset_name = result["dataset_name"]

    # ── Summary metrics for subtitle ─────────────────────────────────────────
    mae  = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    print(f"\n{'=' * 55}")
    print(f"  Predictions vs Actuals")
    print(f"  Model  : {model_name}")
    print(f"  Data   : {dataset_name}")
    print(f"  MAE    : {mae:.4f}   RMSE : {rmse:.4f}")
    print(f"  Test samples plotted : {min(args.sample, len(actuals))}")
    print(f"{'=' * 55}\n")

    # ── Build display sample ──────────────────────────────────────────────────
    sample_size = min(args.sample, len(actuals))
    indices = np.linspace(0, len(actuals) - 1, sample_size, dtype=int)
    act_sample  = actuals[indices]
    pred_sample = predictions[indices]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(indices, act_sample,  "b-",  linewidth=1.2, label="Actual",    alpha=0.9)
    ax.plot(indices, pred_sample, "r--", linewidth=1.2, label="Predicted", alpha=0.85)

    ax.fill_between(
        indices,
        act_sample,
        pred_sample,
        alpha=0.08,
        color="purple",
        label="Error area",
    )

    ax.set_title(
        f"Predictions vs Actuals  (test split sample)\n"
        f"{model_name}  |  {dataset_name}  |  MAE={mae:.4f}  RMSE={rmse:.4f}",
        fontsize=12, pad=12,
    )
    ax.set_xlabel("Test-set timestep index")
    ax.set_ylabel("Value (original scale)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.01,
        "Note: Up to 200 evenly-spaced test-set timesteps are shown.\n"
        "The shaded area represents the prediction error at each point.",
        ha="center", fontsize=8, style="italic", color="dimgrey",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
