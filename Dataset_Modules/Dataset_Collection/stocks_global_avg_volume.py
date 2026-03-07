# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
col_stocks_global_avg_volume.py, creates a dataset of the global average stock trading volume, sourced from Yahoo Finance.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE
from __market_utils import fetch_index, GLOBAL_INDICES

OUTPUT_FILENAME = "stocks_global_avg_volume.csv"
COLUMN_NAME     = "Global averaged stocks (volume)"

def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, interval: str = "1d") -> pd.DataFrame: # (Anthropic, 2026)
    """Fetch seven global market indices and return their average trading volume.

    Downloads each index defined in GLOBAL_INDICES and averages the
    per-index volume columns (those ending in _Volume_M) across all
    indices that have data on each day.  If no volume columns are
    produced the output column is filled with NaN.

    Args:
        start_date: Download start date as an ISO string (YYYY-MM-DD).
        end_date: Download end date as an ISO string (YYYY-MM-DD).
        date_range: DatetimeIndex to re-index the downloaded data onto.
            Dates absent from all indices are filled with NaN.
        interval: yfinance interval string, e.g. "1d" for daily data.
            Defaults to "1d".

    Returns:
        A single-column DataFrame indexed by date_range with column
        "Global averaged stocks (volume)" containing the mean trading
        volume across all available global indices for each date.
    """
    combined = pd.DataFrame(index=date_range)

    for info in GLOBAL_INDICES.values():
        data = fetch_index(info["symbol"], info["currency"], start_date, end_date, date_range, interval=interval)
        if not data.empty and len(data.columns) > 0:
            combined = pd.concat([combined, data], axis=1)

    volume_cols = [c for c in combined.columns if str(c).endswith("_Volume_M")]
    out = pd.DataFrame(index=date_range)
    if volume_cols:
        out[COLUMN_NAME] = combined[volume_cols].mean(axis=1, skipna=True)
    else:
        out[COLUMN_NAME] = float("nan")
    return out

def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments, collect global average volume data, and save the output CSV.

    Accepts optional --start, --end, and --freq arguments to control the
    date range and data frequency.  Defaults are sourced from
    DEFAULT_START_DATE, DEFAULT_END_DATE, and "1d" respectively.  The
    resulting CSV is written to the dataset output directory as
    stocks_global_avg_volume.csv.
    """
    import argparse
    parser = argparse.ArgumentParser(description=OUTPUT_FILENAME)
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: DEFAULT_START_DATE)")
    parser.add_argument("--end",   default=None,
                        help="End date YYYY-MM-DD (default: DEFAULT_END_DATE)")
    parser.add_argument("--freq",  default="1d",
                        help="Data frequency: 1m 2m 5m 15m 30m 90m 1h 1d 5d 1wk 1mo 3mo")
    args = parser.parse_args()

    paths       = init_project_paths(
        start_date=args.start or DEFAULT_START_DATE,
        end_date=args.end   or DEFAULT_END_DATE,
        freq=args.freq,
    )
    output_path = paths["output_dir"] / OUTPUT_FILENAME
    print(f"[{OUTPUT_FILENAME}] Collecting ...")
    try:
        df = collect(paths["start_date"], paths["end_date"], paths["date_range"], interval=paths["yf_interval"])
        df.to_csv(output_path)
        print(f"[{OUTPUT_FILENAME}] Saved {df.shape} -> {output_path}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"Error: {exc}")
if __name__ == "__main__":
    main()
