"""
col_stocks_global_avg_price.py
-------------------------------
Single-column dataset feature.

Column : Global averaged stocks(USD)
Source : Yahoo Finance – 7 major global indices, Close prices converted to USD
         and averaged across all indices with data on each day.
Output : Dataset_Modules/dataset_output/2015-2025_stocks_global_avg_price.csv

Run standalone:
    python col_stocks_global_avg_price.py
Or select via Model Designer → Dataset Collection Method → Configure.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE
from __market_utils import fetch_index, GLOBAL_INDICES

OUTPUT_FILENAME = "stocks_global_avg_price.csv"
COLUMN_NAME     = "Global averaged stocks(USD)"


def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, interval: str = "1d") -> pd.DataFrame:
    """Fetch seven global market indices and return their average USD closing price.

    Iterates over ``GLOBAL_INDICES``, downloading each index via
    :func:`__market_utils.fetch_index` which also performs currency conversion
    to USD.  Columns ending in ``_Close_USD`` are averaged across all available
    indices to produce a single blended global equity price series.

    Args:
        start_date: ISO date string (``YYYY-MM-DD``) for the download start.
        end_date: ISO date string (``YYYY-MM-DD``) for the download end.
        date_range: Full pandas DatetimeIndex to re-index the result onto.
        interval: yfinance interval string (e.g. ``"1d"`` for daily).

    Returns:
        Single-column DataFrame indexed by *date_range* with column
        ``Global averaged stocks(USD)``.  Dates with no data are NaN.
    """
    combined = pd.DataFrame(index=date_range)

    for info in GLOBAL_INDICES.values():
        data = fetch_index(info["symbol"], info["currency"], start_date, end_date, date_range, interval=interval)
        if not data.empty and len(data.columns) > 0:
            combined = pd.concat([combined, data], axis=1)

    close_cols = [c for c in combined.columns if str(c).endswith("_Close_USD")]
    out = pd.DataFrame(index=date_range)
    if close_cols:
        out[COLUMN_NAME] = combined[close_cols].mean(axis=1, skipna=True)
    else:
        out[COLUMN_NAME] = float("nan")
    return out


def main() -> None:
    """Parse CLI arguments, collect global average price data, and save the output CSV."""
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
