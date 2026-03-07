# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

"""
blockchain_utils.py is a shared helper for fetching a single time-series chart from the
Blockchain.info public API and aligning it to a daily DatetimeIndex.
Hidden from the UI discover_scripts list because it starts with __.
"""

from datetime import datetime
import requests
import pandas as pd

BLOCKCHAIN_API_BASE = "https://api.blockchain.info"

def fetch_blockchain_metric(
    metric_endpoint: str,
    start_date: str,
    end_date: str,
    date_range: pd.DatetimeIndex,
) -> pd.Series: # (Anthropic, 2026)
    """Fetch a time-series chart from the Blockchain.info public API.

    Requests the given endpoint, parses the JSON response, and re-indexes
    the result onto date_range. Returns a NaN-filled Series if the request
    fails, the response status is not 200, or the payload contains no values.

    Args:
        metric_endpoint: Blockchain.info chart endpoint path
          (e.g. "charts/hash-rate").
        start_date: ISO date string (YYYY-MM-DD) for the request start.
        end_date: ISO date string (YYYY-MM-DD) for the request end.
        date_range: pandas DatetimeIndex to re-index the result onto.

    Returns:
        A float64 Series aligned to date_range. Entries with no
        corresponding API data are NaN.
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts   = int(datetime.strptime(end_date,   "%Y-%m-%d").timestamp())

    url    = f"{BLOCKCHAIN_API_BASE}/{metric_endpoint}"
    params = {
        "timespan": "all",
        "start":    start_ts,
        "end":      end_ts,
        "format":   "json",
        "sampled":  "true",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.RequestException as exc:
        print(f"    Request failed for {metric_endpoint}: {exc}")
        return pd.Series(index=date_range, dtype=float)

    if response.status_code != 200:
        print(f"    HTTP {response.status_code} for {metric_endpoint}")
        return pd.Series(index=date_range, dtype=float)

    data = response.json()
    if not isinstance(data, dict) or "values" not in data:
        return pd.Series(index=date_range, dtype=float)

    timestamps, values = [], []
    for entry in data["values"]:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            timestamps.append(entry[0])
            values.append(float(entry[1]))
        elif isinstance(entry, dict) and "x" in entry and "y" in entry:
            timestamps.append(entry["x"])
            values.append(float(entry["y"]))

    if not values:
        return pd.Series(index=date_range, dtype=float)

    tmp = pd.DataFrame({"timestamp": timestamps, "value": values})
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], unit="s").dt.normalize()
    tmp = tmp.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    tmp["value"] = tmp["value"].astype("float64")

    return tmp["value"].reindex(date_range)
