"""
col_currency_gold_btc_ratio.py
-------------------------------
Single-column dataset feature.

Column : Gold/BTC Ratio
Source : Yahoo Finance – GC=F (Gold Futures) and BTC-USD (Bitcoin price).
         Ratio = Gold price / BTC price.  Days where BTC price is zero or
         missing are left as NaN.
Output : Dataset_Modules/dataset_output/2015-2025_currency_gold_btc_ratio.csv

Run standalone:
    python col_currency_gold_btc_ratio.py
Or select via Model Designer → Dataset Collection Method → Configure.
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


def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, interval: str = "1d") -> pd.DataFrame:
    """Download Gold Futures and BTC/USD then compute the Gold/BTC price ratio.

    Fetches GC=F (Gold Futures) and BTC-USD closing prices from Yahoo Finance
    separately, aligns both series to *date_range*, then divides gold by bitcoin
    to produce the Gold/BTC ratio.

    Args:
        start_date: ISO date string (``YYYY-MM-DD``) for the download start.
        end_date: ISO date string (``YYYY-MM-DD``) for the download end.
        date_range: Full pandas DatetimeIndex to re-index the result onto.
        interval: yfinance interval string (e.g. ``"1d"`` for daily).

    Returns:
        Single-column DataFrame indexed by *date_range* with column
        ``Gold/BTC Ratio``.  Dates where BTC price is zero or missing
        are set to NaN.
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


def main() -> None:
    """Parse CLI arguments, collect the Gold/BTC ratio data, and save the output CSV."""
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
