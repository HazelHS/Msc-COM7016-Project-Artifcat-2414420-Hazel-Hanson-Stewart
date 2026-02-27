"""
col_stocks_global_avg_volume.py
--------------------------------
Single-column dataset feature.

Column : Global averaged stocks (volume)
Source : Yahoo Finance – 7 major global indices, trading volumes averaged
         (in millions, or thousands for Asian markets) across all indices
         with data on each day.
Output : Dataset_Modules/dataset_output/2015-2025_stocks_global_avg_volume.csv

Run standalone:
    python col_stocks_global_avg_volume.py
Or select via Model Designer → Dataset Collection Method → Configure.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE
from __market_utils import fetch_index, GLOBAL_INDICES

OUTPUT_FILENAME = "stocks_global_avg_volume.csv"
COLUMN_NAME     = "Global averaged stocks (volume)"


def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, interval: str = "1d") -> pd.DataFrame:
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


def main() -> None:
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
