
# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
col_onchain_active_addresses.py, creates a single-column dataset feature dataset for the count of unique active Bitcoin addresses, sourced from the Blockchain.info API.
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

def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, freq: str = "1d") -> pd.DataFrame: # (Anthropic, 2026)
    """Fetch the count of unique active Bitcoin addresses from the Blockchain.info API.

    Calls fetch_blockchain_metric with the n-unique-addresses endpoint
    and re-indexes the result onto date_range before returning.

    Args:
        start_date: ISO date string (YYYY-MM-DD) for the request start.
        end_date: ISO date string (YYYY-MM-DD) for the request end.
        date_range: pandas DatetimeIndex to re-index the result onto.
        freq: Data frequency string. Must be one of the values in
          BLOCKCHAIN_SUPPORTED_FREQS ("1d", "5d", "1wk", "1mo", "3mo").

    Returns:
        A single-column DataFrame indexed by date_range with the column
        "Onchain Active Addresses".

    Raises:
        UnsupportedIntervalError: If freq is not in BLOCKCHAIN_SUPPORTED_FREQS,
          indicating sub-daily granularity that the API does not support.
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

def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments, collect active address data, and save the result to CSV.

    Reads --start, --end, and --freq from the command line, calls collect(),
    and writes the output to dataset_output/onchain_active_addresses.csv.
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
        df = collect(paths["start_date"], paths["end_date"], paths["date_range"], freq=paths["freq"])
        df.to_csv(output_path)
        print(f"[{OUTPUT_FILENAME}] Saved {df.shape} -> {output_path}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"Error: {exc}")
if __name__ == "__main__":
    main()
