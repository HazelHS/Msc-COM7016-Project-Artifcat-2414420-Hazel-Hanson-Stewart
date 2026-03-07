# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
col_currency_btc_price.py, creates a single-column feature dataset for the trading price of BTC/USD, sourced 
from Yahoo Finance.
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE, strip_yf_tz

OUTPUT_FILENAME    = "currency_btc_price.csv"
COLUMN_NAME        = "BTC/USD"
RATE_LIMIT_TIMEOUT = 2

def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, interval: str = "1d") -> pd.DataFrame: # (Anthropic, 2026)
    """Download BTC/USD closing price from Yahoo Finance and re-index to date_range.

    Applies a brief rate-limit delay before downloading. Timezone information
    is stripped from the Yahoo Finance response before re-indexing. Dates in
    date_range not covered by the source are left as NaN.

    Args:
        start_date: ISO date string (YYYY-MM-DD) for the download start.
        end_date: ISO date string (YYYY-MM-DD) for the download end.
        date_range: pandas DatetimeIndex to re-index the result onto.
        interval: yfinance interval string (e.g. "1d" for daily).

    Returns:
        A single-column DataFrame indexed by date_range with the column
        "BTC/USD" holding the Bitcoin closing price in US Dollars.
    """
    time.sleep(RATE_LIMIT_TIMEOUT)
    raw = yf.download("BTC-USD", start=start_date, end=end_date, interval=interval, progress=False)
    raw = strip_yf_tz(raw)
    out = pd.DataFrame(index=date_range)
    out[COLUMN_NAME] = raw["Close"].squeeze()
    return out


def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments, collect BTC/USD price data, and save the result to CSV.

    Reads --start, --end, and --freq from the command line, calls collect(),
    and writes the output to dataset_output/currency_btc_price.csv.
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
