# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
col_currency_gold_btc_ratio.py, creates a single-column dataset feature dataset for the ratio of Gold Futures to 
BTC/USD closing prices, sourced from Yahoo Finance.
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE, strip_yf_tz

OUTPUT_FILENAME    = "currency_gold_btc_ratio.csv"
COLUMN_NAME        = "Gold/BTC Ratio"
RATE_LIMIT_TIMEOUT = 2

def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, interval: str = "1d") -> pd.DataFrame: # (Anthropic, 2026)
    """Download Gold Futures and BTC/USD closing prices, then compute the Gold/BTC ratio.

    Fetches GC=F and BTC-USD separately from Yahoo Finance with a rate-limit
    delay between each request. Both series are aligned to date_range before
    dividing. Days where BTC price is zero or missing produce NaN in the output.

    Args:
        start_date: ISO date string (YYYY-MM-DD) for the download start.
        end_date: ISO date string (YYYY-MM-DD) for the download end.
        date_range: pandas DatetimeIndex to re-index the result onto.
        interval: yfinance interval string (e.g. "1d" for daily).

    Returns:
        A single-column DataFrame indexed by date_range with the column
        "Gold/BTC Ratio". Dates where BTC price is zero or missing are NaN.
    """
    out = pd.DataFrame(index=date_range)

    print("  Fetching GC=F (Gold Futures) ...")
    time.sleep(RATE_LIMIT_TIMEOUT)
    gold = yf.download("GC=F", start=start_date, end=end_date, interval=interval, progress=False)
    gold = strip_yf_tz(gold)
    gold_close = pd.Series(index=date_range, dtype=float)
    gold_close.update(gold["Close"].squeeze())

    print("  Fetching BTC-USD (Close) ...")
    time.sleep(RATE_LIMIT_TIMEOUT)
    btc = yf.download("BTC-USD", start=start_date, end=end_date, interval=interval, progress=False)
    btc = strip_yf_tz(btc)
    btc_close = pd.Series(index=date_range, dtype=float)
    btc_close.update(btc["Close"].squeeze())

    out[COLUMN_NAME] = gold_close.div(btc_close.replace(0, float("nan")))
    return out

def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments, collect the Gold/BTC ratio data, and save the result to CSV.

    Reads --start, --end, and --freq from the command line, calls collect(),
    and writes the output to dataset_output/currency_gold_btc_ratio.csv.
    Prints a confirmation message on success or an error message on failure.
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
