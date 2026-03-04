"""
utils.py
--------
Discovery helper functions shared across the Interface_Modules package.
These functions scan directories for scripts, CSVs and trained-model
checkpoint files used throughout the GUI.
"""

import os


def discover_scripts(directory: str) -> list[str]:
    """Return a sorted list of .py filenames found in *directory*.

    Dunder files (``__init__.py``, etc.) are excluded.
    Returns an empty list when *directory* does not exist.
    """
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".py") and not f.startswith("__")
    )


def discover_csvs(directory: str) -> list[str]:
    """Return a sorted list of .csv filenames found in *directory*.

    Returns an empty list when *directory* does not exist.
    """
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".csv")
    )


def discover_models(directory: str) -> list[str]:
    """Return a sorted list of .pt model-checkpoint filenames found in *directory*.

    Returns an empty list when *directory* does not exist.
    """
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".pt")
    )
