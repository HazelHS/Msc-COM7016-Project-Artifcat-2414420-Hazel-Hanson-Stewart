"""
col_btc_stock_to_flow.py
-------------------------
Single-column dataset feature.

Column : S2F Model
Source : Blockchain.info API – total bitcoin supply, combined with the
         theoretical daily block production derived from Bitcoin's halving
         schedule (genesis block: 2009-01-03, halving every 210,000 blocks).

         S2F ratio  = total supply / (daily production × 365)
         Model price = exp(-1.84) × S2F^3.36   (PlanB original model)

Output : Dataset_Modules/dataset_output/2015-2025_btc_stock_to_flow.csv

Run standalone:
    python col_btc_stock_to_flow.py
Or select via Model Designer → Dataset Collection Method → Configure.
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


def _block_reward(block_height: int) -> float:
    halvings = block_height // HALVING_INTERVAL
    return INITIAL_REWARD / (2 ** halvings)


def collect(start_date: str, end_date: str, date_range: pd.DatetimeIndex, freq: str = "1d") -> pd.DataFrame:
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
