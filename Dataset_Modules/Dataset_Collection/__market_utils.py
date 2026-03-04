"""
__market_utils.py
-----------------
Shared helper for downloading and normalising a single global market index
from Yahoo Finance (with optional FX conversion to USD).
Hidden from the UI discover_scripts list because it starts with __.
"""

import time
import pandas as pd
import yfinance as yf

RATE_LIMIT_TIMEOUT = 2  # seconds between Yahoo Finance requests


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Remove UTC timezone from a yfinance intraday result.

    Ensures the index aligns with the timezone-naive date_range produced by
    :func:`init_project_paths`.

    Args:
        df: DataFrame returned by ``yf.download()``.

    Returns:
        The same DataFrame with a timezone-naive DatetimeIndex.
    """
    if not df.empty and getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_index(
    symbol: str,
    currency_pair: str | None,
    start_date: str,
    end_date: str,
    date_range: pd.DatetimeIndex,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV for *symbol* and optionally convert Close to USD.

    Uses *currency_pair* (e.g. ``'EURUSD=X'``) for FX conversion when
    provided.  Volume is scaled to millions (thousands for Nikkei / HSI).
    Sanity checks reject implausible price or volume ranges.

    Args:
        symbol: Yahoo Finance ticker symbol (e.g. ``'^GDAXI'``).
        currency_pair: Yahoo Finance FX pair symbol for USD conversion
            (e.g. ``'EURUSD=X'``), or None to skip conversion.
        start_date: ISO date string ``'YYYY-MM-DD'`` for download start.
        end_date: ISO date string ``'YYYY-MM-DD'`` for download end.
        date_range: pandas DatetimeIndex to re-index the result onto.
        interval: yfinance interval string (e.g. ``'1d'``, ``'1wk'``,
            ``'1mo'``).  Note: intraday intervals require *start_date*
            within the last 7–730 days depending on the interval.

    Returns:
        DataFrame with columns ``<symbol>_Close_USD`` and
        ``<symbol>_Volume_M`` re-indexed to *date_range*, or an empty
        DataFrame on download error or failed sanity checks.
    """
    print(f"    Fetching {symbol} …")
    time.sleep(RATE_LIMIT_TIMEOUT)

    raw = yf.download(symbol, start=start_date, end=end_date,
                      interval=interval, progress=False)
    raw = _strip_tz(raw)
    if raw.empty:
        return pd.DataFrame(index=date_range)

    result = pd.DataFrame(index=raw.index)
    result["Close"]  = raw["Close"].squeeze()
    result["Volume"] = raw["Volume"].squeeze()

    # ── FX conversion ────────────────────────────────────────────────
    if currency_pair:
        time.sleep(RATE_LIMIT_TIMEOUT)
        fx_raw = yf.download(currency_pair, start=start_date, end=end_date,
                             interval=interval, progress=False)
        fx_raw = _strip_tz(fx_raw)
        if fx_raw.empty:
            return pd.DataFrame(index=date_range)

        fx_rate = fx_raw["Close"].squeeze()
        result.index  = pd.to_datetime(result.index)
        fx_rate.index = pd.to_datetime(fx_rate.index)

        common = result.index.intersection(fx_rate.index)
        result  = result.loc[common]
        fx_rate = fx_rate.loc[common]
        result["Close"] = result["Close"].values * fx_rate.values

    # ── Volume scaling ───────────────────────────────────────────────
    if symbol in ["^N225", "^HSI"]:
        result["Volume"] = result["Volume"] / 1_000      # thousands
    else:
        result["Volume"] = result["Volume"] / 1_000_000  # millions

    # ── Sanity checks ────────────────────────────────────────────────
    if result["Close"].max() > 50_000 or result["Close"].min() < 1:
        return pd.DataFrame(index=date_range)
    if result["Volume"].min() == 0 or result["Volume"].max() / result["Volume"].min() > 1_000:
        return pd.DataFrame(index=date_range)

    result = result.rename(columns={
        "Close":  f"{symbol}_Close_USD",
        "Volume": f"{symbol}_Volume_M",
    })
    return result.reindex(date_range)


# Indices configuration shared by both averaging scripts.
GLOBAL_INDICES: dict[str, dict] = {
    "GDAXI":    {"symbol": "^GDAXI", "currency": "EURUSD=X"},   # Germany DAX
    "IXIC":     {"symbol": "^IXIC",  "currency": None},          # NASDAQ
    "DJI":      {"symbol": "^DJI",   "currency": None},          # Dow Jones
    "N225":     {"symbol": "^N225",  "currency": "JPYUSD=X"},    # Nikkei
    "STOXX50E": {"symbol": "^STOXX", "currency": "EURUSD=X"},    # Euro STOXX 50
    "HSI":      {"symbol": "^HSI",   "currency": "HKDUSD=X"},    # Hang Seng
    "FTSE":     {"symbol": "^FTSE",  "currency": "GBPUSD=X"},    # FTSE 100
}
