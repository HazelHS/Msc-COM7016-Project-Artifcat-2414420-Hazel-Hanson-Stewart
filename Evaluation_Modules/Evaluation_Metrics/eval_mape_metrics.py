# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The eval_mape_metrics.py script evaluates Mean Absolute Percentage Error (MAPE) and
loads a trained model checkpoint and a dataset CSV, runs inference on the test split, then
displays a bar chart of the MAPE metric:
    MAPE — Mean Absolute Percentage Error (lower is better, expressed as %)

MAPE expresses prediction error as a percentage of the actual values, making it
scale-independent and straightforward to interpret across different assets.

    MAPE = (1/n) * Σ |( actual - predicted ) / actual| * 100

Note: samples where actual == 0 are excluded to avoid division-by-zero.
"""

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


# CLI
def parse_args() -> argparse.Namespace:  # (Anthropic, 2026)
    """Configure and parse CLI arguments for the MAPE evaluation script.

    Returns:
        An argparse.Namespace with attributes ``model`` (str, path to the
        .pt checkpoint file), ``dataset`` (str, path to the .csv dataset
        file), and ``forecast_step`` (int, forecast step to evaluate;
        default 1).
    """
    p = argparse.ArgumentParser(
        description="Evaluate Mean Absolute Percentage Error (MAPE) for a trained model."
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


# MAPE calculation
def calculate_mape(actuals: np.ndarray, predictions: np.ndarray) -> float:  # (Anthropic, 2026)
    """Calculate Mean Absolute Percentage Error, excluding zero-valued actuals.

    Computes MAPE as the mean of absolute percentage errors across all samples
    where the actual value is non-zero, to prevent division-by-zero. The result
    is returned as a percentage (0–100+ range).

    Args:
        actuals: 1-D array of inverse-scaled ground-truth target values.
        predictions: 1-D array of inverse-scaled model predictions, same
            length as ``actuals``.

    Returns:
        MAPE expressed as a float percentage. Returns ``float('nan')`` if all
        actual values are zero (no valid samples remain after masking).
    """
    mask = actuals != 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100.0)


# Main
def main() -> None:  # (Anthropic, 2026)
    """Run the MAPE evaluation pipeline.

    Parses CLI arguments, loads the checkpoint and dataset via
    ``load_model_and_run_inference``, computes MAPE via ``calculate_mape``
    on the inverse-scaled test-split predictions and actuals (excluding
    zero-valued actuals), prints a formatted summary to stdout, and displays
    a labelled bar chart via Matplotlib with qualitative interpretation bands.
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

    # MAPE metric
    mape = calculate_mape(actuals, predictions)

    # Qualitative rating
    if np.isnan(mape):
        rating    = "N/A  (all actuals are zero)"
        bar_color = "grey"
    elif mape < 10.0:
        rating    = "Excellent  (< 10 %)"
        bar_color = "forestgreen"
    elif mape < 20.0:
        rating    = "Good  (10 – 20 %)"
        bar_color = "steelblue"
    elif mape < 50.0:
        rating    = "Fair  (20 – 50 %)"
        bar_color = "goldenrod"
    else:
        rating    = "Poor  (> 50 %)"
        bar_color = "crimson"

    # Print metrics
    print(f"\n{'=' * 55}")
    print(f"  Mean Absolute Percentage Error (MAPE)")
    print(f"  Model  : {model_name}")
    print(f"  Data   : {dataset_name}")
    print(f"  Step   : t+{forecast_step}")
    print(f"{'=' * 55}")
    print(f"  MAPE   : {mape:.2f} %" if not np.isnan(mape) else "  MAPE   : N/A")
    print(f"  Rating : {rating}")
    print(f"{'=' * 55}\n")

    # Plot
    display_mape = mape if not np.isnan(mape) else 0.0

    fig, ax = plt.subplots(figsize=(5, 5))
    bar = ax.bar(["MAPE"], [display_mape], color=bar_color, edgecolor="white",
                 linewidth=0.6, width=0.35)

    ax.text(
        bar[0].get_x() + bar[0].get_width() / 2,
        bar[0].get_height() + display_mape * 0.02,
        f"{display_mape:.2f} %" if not np.isnan(mape) else "N/A",
        ha="center", va="bottom", fontsize=12, fontweight="bold",
    )

    # Reference lines
    for level, label, colour in [(10, "Excellent (10%)", "forestgreen"),
                                  (20, "Good (20%)",     "steelblue"),
                                  (50, "Fair (50%)",     "goldenrod")]:
        ax.axhline(level, color=colour, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(0.98, level, label, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=7, color=colour)

    ax.set_ylim(0, max(display_mape * 1.3, 60))
    ax.set_title(
        f"Mean Absolute Percentage Error (MAPE)  —  t+{forecast_step}\n"
        f"{model_name}  |  {dataset_name}",
        fontsize=13, pad=14,
    )
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Metric")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.01,
        "Note: MAPE is scale-independent (expressed as %). Lower values indicate\n"
        "better model performance. Samples with actual == 0 are excluded.",
        ha="center", fontsize=8, style="italic", color="dimgrey",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
