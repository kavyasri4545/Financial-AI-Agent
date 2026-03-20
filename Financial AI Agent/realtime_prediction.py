"""
realtime_prediction.py  –  Real-time prediction with a persistent cache.

Uses the optimised data_loader (two-layer cache) so repeated calls are fast.
"""

import logging
import threading
import time

import pandas as pd

from data_loader import load_stock
from technical_features import add_technical_features
from prediction_model import predict_price

logger = logging.getLogger(__name__)

# ─── In-memory engineered-DataFrame cache ─────────────────────────────────────
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
CACHE_TTL = 300          # 5 minutes


def _refresh_cache(symbol: str) -> None:
    try:
        df = load_stock(symbol)
        if df is None or df.empty:
            return
        df = add_technical_features(df)
        if df is None or df.empty:
            return
        with _CACHE_LOCK:
            _CACHE[symbol] = (time.time(), df)
        logger.debug("Engineered cache refreshed for %s", symbol)
    except Exception as e:
        logger.debug("Cache refresh failed for %s: %s", symbol, e)


def get_cached_df(
    symbol: str,
    force_refresh: bool = False,
    wait_for_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a feature-engineered DataFrame for *symbol*.

    - Fresh cache  → immediate return (no network call).
    - Stale cache  → return stale data, kick off background refresh.
    - No cache     → refresh synchronously if wait_for_refresh, else background.
    """
    now = time.time()

    with _CACHE_LOCK:
        entry = _CACHE.get(symbol)

    if entry:
        ts, df = entry
        if (now - ts) < CACHE_TTL and not force_refresh:
            return df.copy()
        if not force_refresh:
            # serve stale; refresh in background
            threading.Thread(target=_refresh_cache, args=(symbol,), daemon=True).start()
            return df.copy()

    # No entry or forced refresh
    if force_refresh or wait_for_refresh:
        _refresh_cache(symbol)
        with _CACHE_LOCK:
            entry = _CACHE.get(symbol)
        return entry[1].copy() if entry else pd.DataFrame()

    threading.Thread(target=_refresh_cache, args=(symbol,), daemon=True).start()
    return pd.DataFrame()


def run_realtime_prediction_streamlit(symbol: str):
    """
    Single-shot prediction for Streamlit / GUI use.

    Returns
    -------
    (current_price, predicted_price, signal)  – signal is 'BUY', 'SELL', or 'ERROR'.
    """
    try:
        df = get_cached_df(symbol)
        if df is None or df.empty:
            df = get_cached_df(symbol, force_refresh=True, wait_for_refresh=True)
        if df is None or df.empty:
            raise ValueError(f"No data for {symbol}")

        latest_price = float(df["Close"].iloc[-1])
        try:
            predicted_price = predict_price(df, symbol=symbol)
        except Exception as e:
            logger.warning("Prediction failed for %s: %s", symbol, e)
            predicted_price = latest_price

        signal = "BUY" if predicted_price > latest_price else "SELL"
        return latest_price, predicted_price, signal

    except Exception as e:
        logger.error("Real-time prediction error for %s: %s", symbol, e)
        return None, None, "ERROR"


def run_realtime_prediction(symbol: str, interval: int = 60) -> None:
    """Console loop – runs until KeyboardInterrupt."""
    try:
        while True:
            cp, pp, sig = run_realtime_prediction_streamlit(symbol)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            if cp is None:
                logger.info("[%s] %s | No data", ts, symbol)
            else:
                logger.info("[%s] %s | Current: %.2f | Predicted: %.2f | Signal: %s",
                            ts, symbol, cp, pp, sig)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Real-time prediction stopped.")
