"""
col_onchain_tx_count.py
------------------------
Single-column dataset feature.

Column : Onchain Transaction Count
Source : Blockchain.info API – charts/n-transactions
         (number of confirmed Bitcoin transactions per day)
Output : Dataset_Modules/dataset_output/2015-2025_onchain_tx_count.csv

Run standalone:
    python col_onchain_tx_count.py
Or select via Model Designer → Dataset Collection Method → Configure.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE, BLOCKCHAIN_SUPPORTED_FREQS, UnsupportedIntervalError
from __blockchain_utils import fetch_blockchain_metric

OUTPUT_FILENAME = "onchain_tx_count.csv"
COLUMN_NAME     = "Onchain Transaction Count"
ENDPOINT        = "charts/n-transactions"


def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, freq: str = "1d") -> pd.DataFrame:
    if freq.lower() not in BLOCKCHAIN_SUPPORTED_FREQS:
        raise UnsupportedIntervalError(
            f"Blockchain.info API only provides daily-granularity data.  "
            f"Requested interval '{freq}' is not supported.\n"
            "Supported: 1d, 5d, 1wk, 1mo, 3mo  "
            "(or legacy labels: Daily, Weekly, Monthly, Quarterly, Yearly)."
        )
    print(f"  Fetching {COLUMN_NAME} …")
    out = pd.DataFrame(index=date_range)
    out[COLUMN_NAME] = fetch_blockchain_metric(ENDPOINT, start_date, end_date, date_range)
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
        df = collect(paths["start_date"], paths["end_date"], paths["date_range"], freq=paths["freq"])
        df.to_csv(output_path)
        print(f"[{OUTPUT_FILENAME}] Saved {df.shape} -> {output_path}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"Error: {exc}")
if __name__ == "__main__":
    main()
