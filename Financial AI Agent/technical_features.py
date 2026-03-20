"""
technical_features.py  –  Add SMA, Returns, Volatility, RSI to a stock DataFrame.

Handles yfinance MultiIndex columns (e.g. ("Close", "GOOGL")) by flattening
them to simple string column names before any computation.
"""

import numpy as np
import pandas as pd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns a MultiIndex column like:
        [("Open","AAPL"), ("Close","AAPL"), ...]
    Flatten to simple strings: "Open", "Close", etc.
    If the columns are already flat strings, this is a no-op.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Remove duplicate columns that can appear after flattening
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SMA_20, Returns, Volatility, RSI to *df* and return it.
    Safe against yfinance MultiIndex column layouts.
    """
    df = df.copy()
    df = _flatten_columns(df)

    # Ensure Close is a plain Series (not a single-column DataFrame)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        df["Close"] = close

    df["SMA_20"]     = close.rolling(20).mean()
    df["Returns"]    = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df
