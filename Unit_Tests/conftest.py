"""
Pytest configuration for Unit_Tests.

Adds all project module directories to sys.path before any test module is
collected, so test files can import project source without fragile relative
path hacks inside each test file.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # project root


def _add(path: Path) -> None:
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


_add(_ROOT)
_add(_ROOT / "AI_Modules" / "Model_Designs")
_add(_ROOT / "AI_Modules" / "Training_Methods")
_add(_ROOT / "Dataset_Modules" / "Dataset_Processing_Methods")
_add(_ROOT / "Evaluation_Modules" / "Evaluation_Metrics")
_add(_ROOT / "Interface_Modules")
