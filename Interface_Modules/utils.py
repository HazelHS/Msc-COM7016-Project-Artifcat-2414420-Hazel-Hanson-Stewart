# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
utils.py is a discovery helper functions shared across the Interface_Modules package.
These functions scan directories for scripts, CSVs and trained-model
checkpoint files used throughout the GUI.
"""

import os

def discover_scripts(directory: str) -> list[str]: # (Anthropic, 2026)
    """Return a sorted list of .py filenames found in directory.

    Dunder files (e.g. __init__.py) are excluded. Returns an empty list
    if directory does not exist.

    Args:
        directory: Absolute or relative path to the folder to scan.
    """
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".py") and not f.startswith("__")
    )

def discover_csvs(directory: str) -> list[str]: # (Anthropic, 2026)
    """Return a sorted list of .csv filenames found in directory.

    Returns an empty list if directory does not exist.

    Args:
        directory: Absolute or relative path to the folder to scan.
    """
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".csv")
    )

def discover_models(directory: str) -> list[str]: # (Anthropic, 2026)
    """Return a sorted list of .pt model-checkpoint filenames found in directory.

    Returns an empty list if directory does not exist.

    Args:
        directory: Absolute or relative path to the folder to scan.
    """
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".pt")
    )
