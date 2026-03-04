"""
__dataset_utils.py
------------------
Shared utilities for Dataset_Collection feature scripts.
This file starts with __ so it is excluded from the UI discover_scripts list.
"""

import os
from pathlib import Path
import pandas as pd


# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE   = "2025-02-01"
DEFAULT_FREQ       = "1d"


# ── Custom exception ──────────────────────────────────────────────────────────

class UnsupportedIntervalError(ValueError):
    """
    Raised by a dataset-collection script when the requested data frequency /
    interval is not supported by the underlying data source.
    """


# ── yfinance timezone normalisation ───────────────────────────────────────────

def strip_yf_tz(df: "pd.DataFrame") -> "pd.DataFrame":
    """Remove timezone information from a yfinance download result.

    yfinance returns UTC-aware timestamps for intraday intervals.  pandas
    cannot align a timezone-aware Series/DataFrame against the timezone-naive
    ``date_range`` produced by :func:`init_project_paths`, so every value
    would become NaN.  Calling this helper immediately after every
    ``yf.download()`` fixes the misalignment.

    Safe to call on daily/coarser data too – it is a no-op when the index
    is already timezone-naive.

    Args:
        df: DataFrame returned by ``yf.download()``.

    Returns:
        The same DataFrame with a timezone-naive DatetimeIndex.
    """
    import pandas as pd
    if not df.empty and getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


# ── Frequency / interval maps ─────────────────────────────────────────────────

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


def get_yf_interval(freq: str) -> str:
    """Return the yfinance-compatible interval string for *freq*.

    Args:
        freq: One of the yfinance-style frequency labels (e.g. ``'1d'``,
            ``'1h'``, ``'1wk'``).

    Returns:
        The matching yfinance interval string, defaulting to ``'1d'`` for
        unrecognised inputs.
    """
    return YF_INTERVAL_MAP.get(freq.lower(), "1d")


def init_project_paths(
    start_date: str = DEFAULT_START_DATE,
    end_date: str   = DEFAULT_END_DATE,
    freq: str       = DEFAULT_FREQ,
) -> dict:
    """Resolve project directories, create missing folders, and return paths.

    Creates ``dataset_output/`` and chart sub-directories if they do not
    exist, then returns a dictionary of resolved paths and the common
    date-range / empty DataFrame.

    Args:
        start_date: ISO date string (``YYYY-MM-DD``) for the beginning of
            the collection window.  Defaults to ``'2015-01-01'``.
        end_date: ISO date string (``YYYY-MM-DD``) for the end of the
            collection window.  Defaults to ``'2025-02-01'``.
        freq: Frequency / interval – one of the yfinance-style labels:
            ``'1m'``, ``'2m'``, ``'5m'``, ``'15m'``, ``'30m'``, ``'90m'``,
            ``'1h'``, ``'1d'``, ``'5d'``, ``'1wk'``, ``'1mo'``, ``'3mo'``.
            Defaults to ``'1d'``.

    Returns:
        dict containing:
            ``'project_root'``, ``'output_dir'``, ``'output_path'`` (Path),
            ``'start_date'``, ``'end_date'``, ``'freq'`` (as supplied),
            ``'pd_freq'`` (pandas offset alias, e.g. ``'D'``),
            ``'yf_interval'`` (yfinance interval string, e.g. ``'1d'``),
            ``'date_range'`` (pandas DatetimeIndex),
            ``'df'`` (empty DataFrame indexed by date_range).
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
