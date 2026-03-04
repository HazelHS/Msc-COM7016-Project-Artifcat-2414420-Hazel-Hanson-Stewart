"""
eval_classification_metrics.py
-------------------------------
Evaluation script — Directional Classification Metrics.

Loads a trained model checkpoint and a dataset CSV, runs inference on the
test split, then displays a bar chart of directional classification metrics:
  • Accuracy   — % of correct up/down movement predictions
  • Precision  — of predicted "up" moves, how many were actually "up"
  • Recall     — of actual "up" moves, how many were correctly predicted
  • F1 Score   — harmonic mean of Precision and Recall

Usage (standalone):
    python eval_classification_metrics.py --model <path/to/model.pt> --dataset <path/to/data.csv>

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
matplotlib.use("TkAgg")          # interactive window
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate directional classification metrics for a trained model."
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
    args = parse_args()

    # ── Load model and run inference ─────────────────────────────────────────
    result = load_model_and_run_inference(args.model, args.dataset)

    predictions  = result["predictions"]
    actuals      = result["actuals"]
    model_name   = result["model_name"]
    dataset_name = result["dataset_name"]

    # ── Directional (binary) classification ──────────────────────────────────
    binary_pred = np.diff(predictions) > 0   # True = predicted "up"
    binary_true = np.diff(actuals)     > 0   # True = actual "up"

    accuracy  = accuracy_score( binary_true, binary_pred)
    precision = precision_score(binary_true, binary_pred, zero_division=0)
    recall    = recall_score(   binary_true, binary_pred, zero_division=0)
    f1        = f1_score(       binary_true, binary_pred, zero_division=0)

    # ── Print metrics ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Directional Classification Metrics")
    print(f"  Model  : {model_name}")
    print(f"  Data   : {dataset_name}")
    print(f"{'=' * 55}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(f"  Precision : {precision:.4f}  ({precision * 100:.1f}%)")
    print(f"  Recall    : {recall:.4f}  ({recall * 100:.1f}%)")
    print(f"  F1 Score  : {f1:.4f}  ({f1 * 100:.1f}%)")
    print(f"{'=' * 55}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    labels  = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values  = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
    colours = ["royalblue", "steelblue", "cornflowerblue", "mediumblue"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.6)

    # Annotate each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_title(
        f"Directional Classification Metrics\n"
        f"{model_name}  |  {dataset_name}",
        fontsize=13, pad=14,
    )
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Metric")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.01,
        "Note: Metrics are based on next-step directional (up/down) movement prediction.\n"
        "Accuracy = % of correctly predicted directions. Higher is better.",
        ha="center", fontsize=8, style="italic", color="dimgrey",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save button via matplotlib toolbar (Toolbar → floppy / save icon)
    plt.show()


if __name__ == "__main__":
    main()
