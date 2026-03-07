# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
col_btc_stock_to_flow.py creates a single-column dataset feature for the "stock to flow" (STF) model price of 
Bitcoin, based on the total supply and halving schedule. The S2F model is a popular framework in the crypto community 
that relates scarcity (stock) to new supply (flow) to estimate Bitcoin's value.
"""

import os
import sys
import requests
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from __dataset_utils import init_project_paths, DEFAULT_START_DATE, DEFAULT_END_DATE, BLOCKCHAIN_SUPPORTED_FREQS, UnsupportedIntervalError

OUTPUT_FILENAME     = "btc_stock_to_flow.csv"
COLUMN_NAME         = "S2F Model"
BLOCKCHAIN_API_BASE = "https://api.blockchain.info"

# Bitcoin schedule constants
GENESIS_DATE       = pd.Timestamp("2009-01-03")
HALVING_INTERVAL   = 210_000   # blocks
BLOCK_TIME_MINUTES = 10
BLOCKS_PER_DAY     = (24 * 60) // BLOCK_TIME_MINUTES   # 144
INITIAL_REWARD     = 50.0                               # BTC per block

def _block_reward(block_height: int) -> float: # (Anthropic, 2026)
    """Compute the BTC block reward at block_height using the halving schedule.

    Args:
        block_height: Cumulative block count since the genesis block.

    Returns:
        The BTC reward per block as a float, halving every 210,000 blocks
        from an initial reward of 50 BTC.
    """
    halvings = block_height // HALVING_INTERVAL
    return INITIAL_REWARD / (2 ** halvings)

def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, freq: str = "1d") -> pd.DataFrame: # (Anthropic, 2026)
    """Compute the Bitcoin Stock-to-Flow model price series over date_range.

    Fetches cumulative total Bitcoin supply from the Blockchain.info
    charts/total-bitcoins endpoint, estimates daily new supply from the
    halving schedule, then applies the PlanB S2F model price formula
    (exp(-1.84) * S2F ** 3.36). Returns an all-NaN DataFrame if the
    API request fails or returns no data.

    Args:
        start_date: ISO date string (YYYY-MM-DD) for the request start.
        end_date: ISO date string (YYYY-MM-DD) for the request end.
        date_range: pandas DatetimeIndex to re-index the result onto.
        freq: Data frequency string. Must be one of the values in
          BLOCKCHAIN_SUPPORTED_FREQS ("1d", "5d", "1wk", "1mo", "3mo").

    Returns:
        A single-column DataFrame indexed by date_range with the column
        "S2F Model" holding the PlanB model price in USD. All values are
        NaN if the upstream API request fails.

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
    params = {
        "timespan": "all",
        "start":    int(pd.Timestamp(start_date).timestamp()),
        "end":      int(pd.Timestamp(end_date).timestamp()),
        "format":   "json",
        "sampled":  "false",
    }

    print(f"  Fetching total-bitcoin supply from Blockchain.info …")
    try:
        resp = requests.get(
            f"{BLOCKCHAIN_API_BASE}/charts/total-bitcoins",
            params=params,
            timeout=60,
        )
    except requests.RequestException as exc:
        print(f"  Request failed: {exc}")
        return pd.DataFrame(index=date_range, columns=[COLUMN_NAME])

    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} – cannot compute S2F.")
        return pd.DataFrame(index=date_range, columns=[COLUMN_NAME])

    raw_values = resp.json().get("values", [])
    if not raw_values:
        return pd.DataFrame(index=date_range, columns=[COLUMN_NAME])

    supply_df = pd.DataFrame(raw_values, columns=["x", "y"])
    supply_df["timestamp"] = pd.to_datetime(supply_df["x"], unit="s").dt.normalize()
    stock = supply_df.groupby("timestamp")["y"].mean()
    stock = stock.reindex(date_range).interpolate(method="linear")

    s2f = pd.DataFrame(index=date_range)
    elapsed_min       = (date_range - GENESIS_DATE) / pd.Timedelta(minutes=1)
    s2f["block_height"]     = elapsed_min.astype(int) // BLOCK_TIME_MINUTES
    s2f["daily_production"] = s2f["block_height"].apply(_block_reward) * BLOCKS_PER_DAY
    s2f["s2f_ratio"]        = stock / (s2f["daily_production"] * 365)
    s2f[COLUMN_NAME]        = np.exp(-1.84) * (s2f["s2f_ratio"] ** 3.36)

    return s2f[[COLUMN_NAME]]

def main() -> None: # (Anthropic, 2026)
    """Parse CLI arguments, collect the BTC Stock-to-Flow series, and save the result to CSV.

    Reads --start, --end, and --freq from the command line, calls collect(),
    and writes the output to dataset_output/btc_stock_to_flow.csv.
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
