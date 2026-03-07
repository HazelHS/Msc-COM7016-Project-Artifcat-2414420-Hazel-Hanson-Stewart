# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
col_volatility_vix.py, creates a dataset of the CBOE Volatility Index (VIX), sourced from Yahoo Finance.
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE, strip_yf_tz

OUTPUT_FILENAME    = "volatility_vix.csv"
COLUMN_NAME        = "Volatility_CBOE Volatility Index (VIX)"
RATE_LIMIT_TIMEOUT = 2

def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, interval: str = "1d") -> pd.DataFrame: # (Anthropic, 2026)
    """Download the CBOE VIX index from Yahoo Finance and re-index to date_range.

    A short sleep is applied before the request to respect Yahoo Finance
    rate limits.

    Args:
        start_date: Download start date as an ISO string (YYYY-MM-DD).
        end_date: Download end date as an ISO string (YYYY-MM-DD).
        date_range: DatetimeIndex to re-index the downloaded data onto.
            Dates absent from the download are filled with NaN.
        interval: yfinance interval string, e.g. "1d" for daily data.
            Defaults to "1d".

    Returns:
        A single-column DataFrame indexed by date_range with column
        "Volatility_CBOE Volatility Index (VIX)" containing the daily
        S&P 500 implied 30-day volatility readings.
    """
    time.sleep(RATE_LIMIT_TIMEOUT)
    raw = yf.download("^VIX", start=start_date, end=end_date, interval=interval, progress=False)
    raw = strip_yf_tz(raw)
    out = pd.DataFrame(index=date_range)
    out[COLUMN_NAME] = raw["Close"].squeeze()
    return out

def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments, collect CBOE VIX data, and save the output CSV.

    Accepts optional --start, --end, and --freq arguments to control the
    date range and data frequency.  Defaults are sourced from
    DEFAULT_START_DATE, DEFAULT_END_DATE, and "1d" respectively.  The
    resulting CSV is written to the dataset output directory as
    volatility_vix.csv.
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
