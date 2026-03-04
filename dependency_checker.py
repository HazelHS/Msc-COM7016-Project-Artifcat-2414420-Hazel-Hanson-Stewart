"""Shared dependency bootstrap for all AI module scripts.

This module is imported via ``from dependency_checker import *`` at the
top of every model design, training, and evaluation script.  It performs
two responsibilities:

  1. Centralise all heavy third-party imports so they are resolved once
     and star-exported to the calling module's global namespace.
  2. Print GPU availability on import so every subprocess launched by the
     Model Designer GUI shows the hardware context immediately.

Typical usage example::

    from dependency_checker import *   # pulls in torch, nn, pd, np …
"""

# ── Deep learning framework ───────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Data manipulation ────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Machine learning utilities ────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler

# ── Standard library helpers ──────────────────────────────────────────────
from collections import defaultdict
from tqdm import tqdm        # progress-bar wrapper for training loops
import os
from pathlib import Path

# ── GPU diagnostics printed once at import time ───────────────────────────
# Lets every subprocess launched from the GUI show its compute device.
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")