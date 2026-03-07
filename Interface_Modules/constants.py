# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
constants.py is the project-wide constants and pipeline stage definitions for the
Model Designer application.
"""

import os

# (Anthropic, 2026)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Absolute path to the project root directory, resolved two levels up from this file."""

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
"""Ordered pipeline stage definitions used to populate the main-window UI.

Each entry is a 4-tuple with the following fields:

    display_label:  Human-readable stage name shown in the UI.
    relative_path:  Path to the stage's module directory, relative to ROOT_DIR.
    multi_select:   If True, the Configure button presents checkboxes allowing
                    multiple selections.  If False, a single Combobox is shown.
    diagram_script: Path to a diagram-rendering script relative to ROOT_DIR, or
                    None if the stage has no associated diagram view.  When set,
                    a "Create Diagram" button is rendered that executes the script.
"""

DATASET_OUTPUT_DIR = os.path.join(ROOT_DIR, "Dataset_Modules", "dataset_output")
"""Absolute path to the directory that stores collected CSV dataset files."""

TRAINED_MODEL_DIR = os.path.join(
    ROOT_DIR, "AI_Modules", "Training_Methods", "Trained_Model_Files"
)
"""Absolute path to the directory that stores trained model checkpoint files."""

FREQ_OPTIONS: list[str] = [
    "1m", "2m", "5m", "15m", "30m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
]
"""Ordered list of data-frequency strings available in the Dataset Collection configure panel.

Values map directly to yfinance interval identifiers.  The list is ordered
from finest granularity (1-minute intraday) to coarsest (3-month).
"""

DATE_CONFIG_STAGES: set[str] = {"Dataset Collection Method"}
"""Set of pipeline stage labels whose Configure window displays the date-range and frequency panel."""

DATASET_SELECT_STAGES: set[str] = {"Dataset Processing Method"}
"""Set of pipeline stage labels whose Configure window displays the CSV dataset picker."""

EVAL_STAGES: set[str] = {"Model Evaluation Method"}
"""Set of pipeline stage labels whose Configure window displays the trained-model and dataset pickers."""
