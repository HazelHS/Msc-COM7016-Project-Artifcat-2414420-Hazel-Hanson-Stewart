"""
eval_mase_metrics.py
---------------------
Evaluation script — Time-Series Specific Metric: MASE.

Loads a trained model checkpoint and a dataset CSV, runs inference on the
test split, then displays a bar chart of the Mean Absolute Scaled Error (MASE):

  MASE = MAE(model) / MAE(naive one-step persistence forecast on training set)

  • MASE < 1  — model outperforms the naive forecast  (good)
  • MASE ≈ 1  — model is about the same as naive      (fair)
  • MASE > 1  — naive forecast is better              (poor)

Colour zones:
  Green  : 0 – 1   (MASE < 1, model beats naive)
  Yellow : 1 – 2   (fair range)
  Red    : 2 – 10  (poor range)

Usage (standalone):
    python eval_mase_metrics.py --model <path/to/model.pt> --dataset <path/to/data.csv>

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
from eval_utils import load_model_and_run_inference, calculate_mase

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MASE evaluation script.

    Returns:
        argparse.Namespace with ``--model`` and ``--dataset`` paths.
    """
    p = argparse.ArgumentParser(
        description="Evaluate MASE (Mean Absolute Scaled Error) for a trained model."
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
    """Load checkpoint, run inference on the test split, and display a MASE bar chart."""
    args = parse_args()

    # ── Load model and run inference ─────────────────────────────────────────
    result = load_model_and_run_inference(args.model, args.dataset)

    predictions  = result["predictions"]
    actuals      = result["actuals"]
    y_train      = result["y_train"]
    model_name   = result["model_name"]
    dataset_name = result["dataset_name"]

    # ── MASE ─────────────────────────────────────────────────────────────────
    mase      = calculate_mase(actuals, predictions, y_train)
    safe_mase = min(mase, 10.0)   # cap display at 10 to avoid huge axis

    # Qualitative rating
    if mase < 1.0:
        rating    = "Good  (model beats naive forecast)"
        bar_color = "forestgreen"
    elif mase < 2.0:
        rating    = "Fair  (model comparable to naive)"
        bar_color = "goldenrod"
    else:
        rating    = "Poor  (naive forecast is better)"
        bar_color = "crimson"

    # ── Print metrics ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Time-Series Metric: MASE")
    print(f"  Model  : {model_name}")
    print(f"  Data   : {dataset_name}")
    print(f"{'=' * 55}")
    print(f"  MASE   : {mase:.4f}")
    print(f"  Rating : {rating}")
    print(f"{'=' * 55}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    y_upper = max(2.5, safe_mase * 1.30)

    fig, ax = plt.subplots(figsize=(5, 6))

    # Colour-coded background zones
    ax.axhspan(0,   1,   alpha=0.12, color="green",  label="Good  (MASE < 1)")
    ax.axhspan(1,   2,   alpha=0.12, color="yellow", label="Fair  (1 ≤ MASE < 2)")
    ax.axhspan(2,   y_upper, alpha=0.10, color="red", label="Poor  (MASE ≥ 2)")

    bar = ax.bar(["MASE"], [safe_mase], color=bar_color, edgecolor="white",
                 linewidth=0.6, width=0.35, zorder=3)

    # Annotate
    ax.text(
        0, safe_mase + y_upper * 0.02,
        f"{mase:.4f}" + ("  (capped at 10)" if mase > 10 else ""),
        ha="center", va="bottom", fontsize=13, fontweight="bold",
    )

    # Reference line at MASE = 1
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55,
               label="Naive forecast baseline (MASE = 1)")

    ax.set_ylim(0, y_upper)
    ax.set_title(
        f"Mean Absolute Scaled Error (MASE)\n"
        f"{model_name}  |  {dataset_name}",
        fontsize=12, pad=12,
    )
    ax.set_ylabel("MASE")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.01,
        f"Rating: {rating}\n"
        "MASE < 1 means the model outperforms persistence (naive) forecasting.",
        ha="center", fontsize=8, style="italic", color="dimgrey",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
