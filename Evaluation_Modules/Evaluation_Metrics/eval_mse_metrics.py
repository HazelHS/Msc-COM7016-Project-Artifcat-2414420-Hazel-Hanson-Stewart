# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The eval_mse_metrics.py script evaluates Mean Squared Error (MSE) and
loads a trained model checkpoint and a dataset CSV, runs inference on the test split, then
displays a bar chart of the MSE metric:
    MSE — Mean Squared Error (lower is better)

MSE penalises large errors more heavily than MAE because errors are squared
before being averaged, making it sensitive to outliers.
"""

DESCRIPTION = "Evaluates Mean Squared Error (MSE) for model predictions."

import argparse
import sys
from pathlib import Path

# Path setup
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

# Project imports
from __eval_utils import load_model_and_run_inference

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# CLI
def parse_args() -> argparse.Namespace:  # (Anthropic, 2026)
    """Configure and parse CLI arguments for the MSE evaluation script.

    Returns:
        An argparse.Namespace with attributes ``model`` (str, path to the
        .pt checkpoint file), ``dataset`` (str, path to the .csv dataset
        file), and ``forecast_step`` (int, forecast step to evaluate;
        default 1).
    """
    p = argparse.ArgumentParser(
        description="Evaluate Mean Squared Error (MSE) for a trained model."
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
        "--forecast_step",
        type=int,
        default=1,
        help="Which forecast step to evaluate: 1=t+1, 2=t+2, etc. (default: 1).",
    )
    return p.parse_args()


# Main
def main() -> None:  # (Anthropic, 2026)
    """Run the MSE evaluation pipeline.

    Parses CLI arguments, loads the checkpoint and dataset via
    ``load_model_and_run_inference``, computes Mean Squared Error on the
    inverse-scaled test-split predictions and actuals, prints a formatted
    summary to stdout, and displays a labelled bar chart via Matplotlib.
    """
    args = parse_args()

    # Load model and run inference
    result = load_model_and_run_inference(
        args.model, args.dataset, forecast_step=args.forecast_step
    )

    predictions   = result["predictions"]
    actuals       = result["actuals"]
    model_name    = result["model_name"]
    dataset_name  = result["dataset_name"]
    forecast_step = result.get("forecast_step", args.forecast_step)

    # MSE metric
    mse = mean_squared_error(actuals, predictions)

    # Print metrics
    print(f"\n{'=' * 55}")
    print(f"  Mean Squared Error (MSE)")
    print(f"  Model  : {model_name}")
    print(f"  Data   : {dataset_name}")
    print(f"  Step   : t+{forecast_step}")
    print(f"{'=' * 55}")
    print(f"  MSE    : {mse:.4f}")
    print(f"{'=' * 55}\n")

    # Plot
    labels  = ["MSE"]
    values  = [mse]
    colours = ["#3498db"]   # blue — squared error metric

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.6,
                  width=0.35)

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
        f"Mean Squared Error (MSE)  —  t+{forecast_step}\n"
        f"{model_name}  |  {dataset_name}",
        fontsize=13, pad=14,
    )
    ax.set_ylabel("MSE (original scale squared)")
    ax.set_xlabel("Metric")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.01,
        "Note: MSE is in squared units of the target variable.\n"
        "Lower values indicate better model performance.",
        ha="center", fontsize=8, style="italic", color="dimgrey",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
