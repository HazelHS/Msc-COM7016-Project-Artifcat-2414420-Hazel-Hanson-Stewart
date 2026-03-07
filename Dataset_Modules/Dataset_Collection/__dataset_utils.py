# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
__dataset_utils.py is a shared utilities for Dataset_Collection feature scripts, using "__" so 
it is excluded from the UI discover_scripts list.
"""

import os
from pathlib import Path
import pandas as pd

# Defaults 
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE   = "2025-02-01"
DEFAULT_FREQ       = "1d"

# Custom exception 
class UnsupportedIntervalError(ValueError): # (Anthropic, 2026)
    """Raised when a requested data frequency is not supported by the data source."""
# yfinance timezone normalisation 
def strip_yf_tz(df: "pd.DataFrame") -> "pd.DataFrame": # (Anthropic, 2026)
    """Remove timezone information from a yfinance download result.

    Call this immediately after yf.download() to ensure the index can be
    aligned against the timezone-naive DatetimeIndex produced by
    init_project_paths(). Safe to call on daily or coarser data — it is a
    no-op when the index is already timezone-naive or the DataFrame is empty.

    Args:
        df: DataFrame returned by yf.download().

    Returns:
        The same DataFrame with a timezone-naive DatetimeIndex.
    """
    import pandas as pd
    if not df.empty and getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df

# Frequency / interval maps
# Maps every accepted freq label → pandas date_range offset alias.
FREQ_MAP: dict[str, str] = {
    "1m":  "1min",  "2m":  "2min",  "5m":  "5min",
    "15m": "15min", "30m": "30min", "90m": "90min",
    "1h":  "h",
    "1d":  "D",   "5d":  "5D",  "1wk": "W",
    "1mo": "MS",  "3mo": "QS",
}

# Maps every accepted freq label → yfinance ``interval`` parameter string.
YF_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",   "2m": "2m",    "5m": "5m",
    "15m": "15m", "30m": "30m",  "90m": "90m",
    "1h": "1h",
    "1d": "1d",   "5d": "5d",    "1wk": "1wk",
    "1mo": "1mo", "3mo": "3mo",
}

# Freq labels that sources with only daily-granularity data (e.g. Blockchain.info)
# are able to fulfil.  Anything finer than "1d" is rejected.
BLOCKCHAIN_SUPPORTED_FREQS: frozenset[str] = frozenset({
    "1d", "5d", "1wk", "1mo", "3mo",
})

def get_yf_interval(freq: str) -> str: # (Anthropic, 2026)
    """Return the yfinance-compatible interval string for freq.

    Args:
        freq: A yfinance-style frequency label (e.g. "1d", "1h", "1wk").

    Returns:
        The matching yfinance interval string. Defaults to "1d" for
        unrecognised inputs.
    """
    return YF_INTERVAL_MAP.get(freq.lower(), "1d")

def init_project_paths(
    start_date: str = DEFAULT_START_DATE,
    end_date: str   = DEFAULT_END_DATE,
    freq: str       = DEFAULT_FREQ,
) -> dict: # (Anthropic, 2026)
    """Resolve project directories, create missing folders, and return a paths dict.

    Creates dataset_output/ and chart sub-directories under the project root
    if they do not already exist.

    Args:
        start_date: ISO date string (YYYY-MM-DD) for the start of the
          collection window. Defaults to "2015-01-01".
        end_date: ISO date string (YYYY-MM-DD) for the end of the collection
          window. Defaults to "2025-02-01".
        freq: Data frequency — one of the yfinance-style labels: "1m", "2m",
          "5m", "15m", "30m", "90m", "1h", "1d", "5d", "1wk", "1mo",
          "3mo". Defaults to "1d".

    Returns:
        A dict with the following keys:
          "project_root": Path to the workspace root.
          "output_dir": Path to the dataset_output/ directory.
          "output_path": Same as output_dir (append a filename to get a file path).
          "start_date": The start_date string as supplied.
          "end_date": The end_date string as supplied.
          "freq": The freq string as supplied.
          "pd_freq": pandas offset alias for the frequency (e.g. "D").
          "yf_interval": yfinance interval string (e.g. "1d").
          "date_range": pandas DatetimeIndex spanning start_date to end_date.
          "df": Empty DataFrame indexed by date_range.
    """
    # Navigate from  …/Dataset_Modules/Dataset_Collection/ → project root
    root = Path(__file__).resolve().parent.parent.parent

    output_dir = root / "Dataset_Modules" / "dataset_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chart sub-directories
    for sub in ("data_visualizations", "denoising", "evaluation_metrics"):
        (root / "Dataset_Modules" / "dataset_charts" / sub).mkdir(
            parents=True, exist_ok=True
        )

    # Resolve pandas and yfinance frequency / interval aliases
    pd_freq     = FREQ_MAP.get(freq.lower(), "D")
    yf_interval = get_yf_interval(freq)

    date_range = pd.date_range(start=start_date, end=end_date, freq=pd_freq)
    df = pd.DataFrame(index=date_range)

    return {
        "project_root": root,
        "output_dir":   output_dir,
        "output_path":  output_dir,     # base dir; each feature appends its filename
        "start_date":   start_date,
        "end_date":     end_date,
        "freq":         freq,
        "pd_freq":      pd_freq,
        "yf_interval":  yf_interval,
        "date_range":   date_range,
        "df":           df,
    }
