# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
The eval_r2_metrics.py script evaluates the R-squared (R²) coefficient of
determination and loads a trained model checkpoint and a dataset CSV, runs
inference on the test split, then displays a gauge-style bar chart of R²:

    R² — Coefficient of Determination (higher is better, range ≤ 1)

    R² = 1 - SS_res / SS_tot
        where SS_res = Σ(actual - predicted)²
              SS_tot = Σ(actual - mean(actual))²

    R² = 1.0   — perfect fit
    R² > 0.9   — excellent
    R² > 0.7   — good
    R² > 0.5   — fair
    R² ≤ 0.5   — poor
    R² < 0     — model is worse than predicting the mean
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
from sklearn.metrics import r2_score


# CLI
def parse_args() -> argparse.Namespace:  # (Anthropic, 2026)
    """Configure and parse CLI arguments for the R² evaluation script.

    Returns:
        An argparse.Namespace with attributes ``model`` (str, path to the
        .pt checkpoint file), ``dataset`` (str, path to the .csv dataset
        file), and ``forecast_step`` (int, forecast step to evaluate;
        default 1).
    """
    p = argparse.ArgumentParser(
        description="Evaluate R-squared (R²) coefficient of determination for a trained model."
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
    """Run the R-squared evaluation pipeline.

    Parses CLI arguments, loads the checkpoint and dataset via
    ``load_model_and_run_inference``, computes the R² coefficient of
    determination on the inverse-scaled test-split predictions and actuals
    via ``sklearn.metrics.r2_score``, prints a rated summary to stdout, and
    renders a colour-coded horizontal bar chart with qualitative threshold
    bands (excellent / good / fair / poor) via Matplotlib.

    Raises:
        SystemExit: Exits with a non-zero status if argument parsing fails.
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

    # R² metric
    r2 = r2_score(actuals, predictions)

    # Qualitative rating
    if r2 >= 0.9:
        rating    = "Excellent  (R² ≥ 0.9)"
        bar_color = "forestgreen"
    elif r2 >= 0.7:
        rating    = "Good  (0.7 ≤ R² < 0.9)"
        bar_color = "steelblue"
    elif r2 >= 0.5:
        rating    = "Fair  (0.5 ≤ R² < 0.7)"
        bar_color = "goldenrod"
    elif r2 >= 0.0:
        rating    = "Poor  (0 ≤ R² < 0.5)"
        bar_color = "tomato"
    else:
        rating    = "Very Poor  (R² < 0, worse than mean predictor)"
        bar_color = "crimson"

    # Print metrics
    print(f"\n{'=' * 55}")
    print(f"  R-squared (R²) — Coefficient of Determination")
    print(f"  Model  : {model_name}")
    print(f"  Data   : {dataset_name}")
    print(f"  Step   : t+{forecast_step}")
    print(f"{'=' * 55}")
    print(f"  R²     : {r2:.4f}")
    print(f"  Rating : {rating}")
    print(f"{'=' * 55}\n")

    # Plot — horizontal bar so the 0–1 scale reads naturally left-to-right
    fig, ax = plt.subplots(figsize=(7, 4))

    # Background quality bands
    band_specs = [
        (0.0, 0.5,  "#fce4e4", "Poor"),
        (0.5, 0.7,  "#fff9e6", "Fair"),
        (0.7, 0.9,  "#e6f2ff", "Good"),
        (0.9, 1.0,  "#e6f9ec", "Excellent"),
    ]
    for x_start, x_end, colour, _ in band_specs:
        ax.axvspan(x_start, x_end, color=colour, alpha=0.6, zorder=0)

    # Clamp display value to [-0.5, 1.0] so very negative R² doesn't distort the axis
    display_r2 = max(r2, -0.5)
    ax.barh(["R²"], [display_r2], color=bar_color, edgecolor="white",
            linewidth=0.6, height=0.4, zorder=2)

    ax.axvline(0, color="dimgrey", linewidth=1.0, linestyle="-", zorder=3)

    label_x = display_r2 + 0.01 if display_r2 >= 0 else display_r2 - 0.01
    ha = "left" if display_r2 >= 0 else "right"
    ax.text(label_x, 0, f"R² = {r2:.4f}", va="center", ha=ha,
            fontsize=13, fontweight="bold", color=bar_color, zorder=4)

    # Threshold markers
    for thresh, label in [(0.5, "0.5"), (0.7, "0.7"), (0.9, "0.9")]:
        ax.axvline(thresh, color="grey", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)
        ax.text(thresh, -0.32, label, ha="center", va="top", fontsize=8, color="grey")

    # Band labels along the top
    for x_start, x_end, _, band_label in band_specs:
        ax.text((x_start + x_end) / 2, 0.28, band_label,
                ha="center", va="bottom", fontsize=8, color="dimgrey",
                transform=ax.get_xaxis_transform())

    ax.set_xlim(-0.5, 1.05)
    ax.set_xlabel("R² Value")
    ax.set_title(
        f"R-squared (R²)  —  t+{forecast_step}\n"
        f"{model_name}  |  {dataset_name}",
        fontsize=13, pad=14,
    )
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.text(
        0.5, 0.01,
        "Note: R² = 1 is a perfect fit; R² = 0 means the model predicts the mean;\n"
        "R² < 0 means the model performs worse than predicting the mean.",
        ha="center", fontsize=8, style="italic", color="dimgrey",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
