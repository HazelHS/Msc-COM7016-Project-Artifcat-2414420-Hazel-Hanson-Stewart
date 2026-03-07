# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

""" 
Primarily used to make sure that the pytorch dependancy is correctly installed, working and can access the GPU.
It is also imported in the MEMD-TCN and xLSTM-TS and model training scripts too, for the same reason.

"""

# Deep learning framework
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np

# Machine learning utilities
from sklearn.preprocessing import MinMaxScaler

# Standard library helpers
from collections import defaultdict
from tqdm import tqdm        # progress-bar wrapper for training loops
import os
from pathlib import Path

# GPU diagnostics printed once at import time, lets every subprocess launched from the GUI show if it is using and recognising the GPU.
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")