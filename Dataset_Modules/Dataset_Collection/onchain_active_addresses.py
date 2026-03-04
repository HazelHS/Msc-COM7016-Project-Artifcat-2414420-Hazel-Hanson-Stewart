"""
col_onchain_active_addresses.py
--------------------------------
Single-column dataset feature.

Column : Onchain Active Addresses
Source : Blockchain.info API – charts/n-unique-addresses
         (number of unique Bitcoin addresses active per day)
Output : Dataset_Modules/dataset_output/2015-2025_onchain_active_addresses.csv

Run standalone:
    python col_onchain_active_addresses.py
Or select via Model Designer → Dataset Collection Method → Configure.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE, BLOCKCHAIN_SUPPORTED_FREQS, UnsupportedIntervalError
from __blockchain_utils import fetch_blockchain_metric

OUTPUT_FILENAME = "onchain_active_addresses.csv"
COLUMN_NAME     = "Onchain Active Addresses"
ENDPOINT        = "charts/n-unique-addresses"


def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, freq: str = "1d") -> pd.DataFrame:
    """Fetch the count of unique active Bitcoin addresses from the Blockchain.info API.

    Args:
        start_date: ISO date string (``YYYY-MM-DD``) for the request start.
        end_date: ISO date string (``YYYY-MM-DD``) for the request end.
        date_range: pandas DatetimeIndex to re-index the result onto.
        freq: Data frequency string.  Must be a member of
            ``BLOCKCHAIN_SUPPORTED_FREQS`` (``"1d"``, ``"5d"``,
            ``"1wk"``, ``"1mo"``, ``"3mo"``).

    Returns:
        Single-column DataFrame indexed by *date_range* with column
        ``Onchain Active Addresses``.

    Raises:
        UnsupportedIntervalError: If *freq* requests sub-daily granularity
            that the Blockchain.info API does not support.
    """
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
    """Parse CLI arguments, collect active address data, and save the output CSV."""
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
