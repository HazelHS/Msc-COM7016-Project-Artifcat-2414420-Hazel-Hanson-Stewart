"""
Interface_Modules package
=========================
Public API for the GUI layer.  Import from here rather than from individual
sub-modules so that callers remain insulated from internal file layout.
"""

from .constants import (
    ROOT_DIR,
    PIPELINE_STAGES,
    DATASET_OUTPUT_DIR,
    TRAINED_MODEL_DIR,
    FREQ_OPTIONS,
    DATE_CONFIG_STAGES,
    DATASET_SELECT_STAGES,
    EVAL_STAGES,
)
from .utils import discover_scripts, discover_csvs, discover_models
from .main_window import MainWindow
from .configure_window import ConfigureWindow
from .analysis_window import AnalysisWindow
from .process_dataset_window import ProcessDatasetWindow
from .feature_selection_window import FeatureSelectionWindow
from .training_configure_window import TrainingConfigureWindow

__all__ = [
    # constants
    "ROOT_DIR",
    "PIPELINE_STAGES",
    "DATASET_OUTPUT_DIR",
    "TRAINED_MODEL_DIR",
    "FREQ_OPTIONS",
    "DATE_CONFIG_STAGES",
    "DATASET_SELECT_STAGES",
    "EVAL_STAGES",
    # utils
    "discover_scripts",
    "discover_csvs",
    "discover_models",
    # windows
    "MainWindow",
    "ConfigureWindow",
    "AnalysisWindow",
    "ProcessDatasetWindow",
    "FeatureSelectionWindow",
    "TrainingConfigureWindow",
]
