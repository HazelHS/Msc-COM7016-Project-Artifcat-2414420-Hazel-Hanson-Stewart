"""
col_currency_btc_volume.py
---------------------------
Single-column dataset feature.

Column : BTC Volume
Source : Yahoo Finance – BTC-USD  (Bitcoin daily trading volume in USD)
Output : Dataset_Modules/dataset_output/2015-2025_currency_btc_volume.csv

Run standalone:
    python col_currency_btc_volume.py
Or select via Model Designer -> Dataset Collection Method -> Configure.
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE, strip_yf_tz

OUTPUT_FILENAME    = "currency_btc_volume.csv"
COLUMN_NAME        = "BTC Volume"
RATE_LIMIT_TIMEOUT = 2


def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, interval: str = "1d") -> pd.DataFrame:
    """Download BTC daily trading volume from Yahoo Finance and re-index to *date_range*.

    Args:
        start_date: ISO date string (``YYYY-MM-DD``) for the download start.
        end_date: ISO date string (``YYYY-MM-DD``) for the download end.
        date_range: Full pandas DatetimeIndex to re-index the result onto.
        interval: yfinance interval string (e.g. ``"1d"`` for daily).

    Returns:
        Single-column DataFrame indexed by *date_range* with column
        ``BTC Volume`` holding the daily trading volume in USD.
    """
    time.sleep(RATE_LIMIT_TIMEOUT)
    raw = yf.download("BTC-USD", start=start_date, end=end_date, interval=interval, progress=False)
    raw = strip_yf_tz(raw)
    out = pd.DataFrame(index=date_range)
    out[COLUMN_NAME] = raw["Volume"].squeeze()
    return out


def main() -> None:
    """Parse CLI arguments, collect BTC trading volume data, and save the output CSV."""
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
