"""
constants.py
------------
Project-wide constants and pipeline stage definitions for the
Model Designer application.
"""

import os

# ── Project root (two levels up from this file) ──────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Pipeline stage definitions ────────────────────────────────────────
# Each tuple: (display_label, path_relative_to_ROOT_DIR, multi_select,
#              diagram_script_relative_to_ROOT_DIR | None)
# multi_select=True  → Configure button with checkboxes
# multi_select=False → single Combobox
# diagram_script     → if set, renders a "Create Diagram" button that
#                       runs that script
PIPELINE_STAGES: list[tuple[str, str, bool, str | None]] = [
    (
        "Dataset Collection Method",
        os.path.join("Dataset_Modules", "Dataset_Collection"),
        True,
        None,
    ),
    (
        "Dataset Processing Method",
        os.path.join("Dataset_Modules", "Dataset_Processing_Methods"),
        True,
        None,
    ),
    (
        "AI Model Designs",
        os.path.join("AI_Modules", "Model_Designs"),
        False,
        os.path.join("AI_Modules", "Model_Map_Diagram", "model_node_display.py"),
    ),
    (
        "AI Training Method",
        os.path.join("AI_Modules", "Training_Methods"),
        False,
        None,
    ),
    (
        "Model Evaluation Method",
        os.path.join("Evaluation_Modules", "Evaluation_Metrics"),
        True,
        None,
    ),
]

# ── Key output directories ────────────────────────────────────────────

# Folder that holds the collected CSV datasets
DATASET_OUTPUT_DIR = os.path.join(ROOT_DIR, "Dataset_Modules", "dataset_output")

# Folder that holds trained model checkpoint files
TRAINED_MODEL_DIR = os.path.join(
    ROOT_DIR, "AI_Modules", "Training_Methods", "Trained_Model_Files"
)

# ── Data-frequency options (used by Dataset Collection configure panel) ──

FREQ_OPTIONS: list[str] = [
    # ── Intraday ─────────────────────────────────────────────────────
    "1m", "2m", "5m", "15m", "30m", "90m", "1h",
    # ── Daily / coarser ──────────────────────────────────────────────
    "1d", "5d", "1wk", "1mo", "3mo",
]

# ── Stage-label sets that drive conditional UI in ConfigureWindow ─────

# Labels for which the Configure window shows the date/frequency panel.
DATE_CONFIG_STAGES: set[str] = {"Dataset Collection Method"}

# Labels for which the Configure window shows the CSV dataset picker.
DATASET_SELECT_STAGES: set[str] = {"Dataset Processing Method"}

# Labels for which the Configure window shows trained-model + dataset pickers.
EVAL_STAGES: set[str] = {"Model Evaluation Method"}
