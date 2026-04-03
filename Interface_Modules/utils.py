# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
utils.py is a discovery helper functions shared across the Interface_Modules package.
These functions scan directories for scripts, CSVs and trained-model
checkpoint files used throughout the GUI.
"""

import os
import re

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

def read_script_description(filepath: str) -> str: # (Anthropic, 2026)
    """Read the DESCRIPTION variable from a script file without importing it.

    Scans only the first 3000 characters of the file for a top-level
    ``DESCRIPTION = "..."`` or ``DESCRIPTION = '...'`` assignment.

    Args:
        filepath: Absolute path to the script file.

    Returns:
        The description string, or an empty string if not found.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            head = fh.read(3000)
        match = re.search(
            r'^DESCRIPTION\s*=\s*["\'](.+?)["\']',
            head,
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    except OSError:
        pass
    return ""

def discover_scripts_with_descriptions(directory: str) -> list[tuple[str, str]]: # (Anthropic, 2026)
    """Return sorted (filename, description) pairs for scripts in directory.

    Calls ``discover_scripts`` then reads the ``DESCRIPTION`` variable from
    each file without importing it.

    Args:
        directory: Absolute or relative path to the folder to scan.

    Returns:
        List of ``(filename, description)`` tuples.
    """
    return [
        (script, read_script_description(os.path.join(directory, script)))
        for script in discover_scripts(directory)
    ]
