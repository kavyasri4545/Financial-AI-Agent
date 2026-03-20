"""
data_loader.py  -  Cached stock data loader.

Two-layer cache
---------------
1. In-memory dict  - instant, lives for MEMORY_TTL seconds.
2. On-disk parquet - survives restarts, lives for DISK_TTL seconds.

yfinance is only called when both caches are stale / missing.
"""

import logging
import threading
import time
from pathlib import Path
import tempfile

import pandas as pd

logger = logging.getLogger(__name__)

MEMORY_TTL = 300          # 5 minutes
DISK_TTL   = 3600         # 1 hour

_CACHE_DIR = Path(tempfile.gettempdir()) / "finai_stock_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MEM_CACHE: dict = {}     # symbol -> (timestamp, df)
_MEM_LOCK = threading.Lock()


def _disk_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_").replace("/", "_")
    return _CACHE_DIR / f"{safe}.parquet"


def _load_from_disk(symbol: str):
    path = _disk_path(symbol)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > DISK_TTL:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _save_to_disk(symbol: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_disk_path(symbol))
    except Exception as e:
        logger.debug("Disk cache write failed for %s: %s", symbol, e)


def _sanitise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten yfinance MultiIndex columns and ensure every OHLCV column
    is a plain 1-D float Series (not a sub-DataFrame).

    yfinance >= 0.2.x returns columns like:
        MultiIndex([('Close', 'AAPL'), ('High', 'AAPL'), ...])
    After get_level_values(0) those become:
        Index(['Close', 'Close', 'High', ...])   <-- possible duplicates
    We keep only the first occurrence of each name.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Drop duplicate column names (keep first)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # If any OHLCV column is still a DataFrame, squeeze it to a Series
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    return df


def _fetch_yfinance(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(
        symbol,
        period="6mo",
        interval="1d",
        progress=False,
        auto_adjust=True,
        group_by="column",   # keeps column layout predictable
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    return _sanitise(df)


def load_stock(symbol: str) -> pd.DataFrame:
    """
    Return a DataFrame for *symbol*. Hits memory cache -> disk cache -> yfinance.
    Always returns a DataFrame (may be empty on complete failure).
    """
    now = time.time()

    # 1. Memory cache
    with _MEM_LOCK:
        entry = _MEM_CACHE.get(symbol)
    if entry and (now - entry[0]) < MEMORY_TTL:
        logger.debug("Memory cache hit for %s", symbol)
        return entry[1].copy()

    # 2. Disk cache
    df = _load_from_disk(symbol)
    if df is not None and not df.empty:
        logger.debug("Disk cache hit for %s", symbol)
        df = _sanitise(df)   # re-sanitise in case old cache has MultiIndex
        with _MEM_LOCK:
            _MEM_CACHE[symbol] = (now, df)
        return df.copy()

    # 3. Network fetch
    try:
        logger.debug("Fetching %s from yfinance ...", symbol)
        df = _fetch_yfinance(symbol)
        with _MEM_LOCK:
            _MEM_CACHE[symbol] = (now, df)
        _save_to_disk(symbol, df)
        return df.copy()
    except Exception as e:
        logger.error("Error loading %s: %s", symbol, e)
        return pd.DataFrame()


def prefetch_symbols(symbols: list, max_workers: int = 8) -> None:
    """
    Pre-warm the cache for a list of symbols in parallel background threads.
    Call this at app startup so the first user click is instant.
    """
    def _load(sym):
        try:
            load_stock(sym)
        except Exception:
            pass

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        ex.map(_load, symbols)
