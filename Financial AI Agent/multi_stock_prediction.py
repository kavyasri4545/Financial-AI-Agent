"""
multi_stock_prediction.py  –  Parallel multi-stock prediction.

Uses ThreadPoolExecutor so all symbols are processed concurrently instead of
one at a time. Also uses the shared cache from realtime_prediction so already-
loaded data is never re-fetched.
"""

import concurrent.futures
import logging

from realtime_prediction import get_cached_df
from data_loader import load_stock
from technical_features import add_technical_features
from prediction_model import predict_price

logger = logging.getLogger(__name__)


def _analyse_one(symbol: str):
    try:
        # Try cached engineered DataFrame first
        df = get_cached_df(symbol)
        if df is None or df.empty:
            df = load_stock(symbol)
            if df is None or df.empty:
                return None
            df = add_technical_features(df)

        prediction   = predict_price(df, symbol=symbol)
        latest_price = float(df["Close"].iloc[-1])
        exp_return   = (prediction - latest_price) / latest_price if latest_price else 0.0
        return (symbol, latest_price, prediction, exp_return)
    except Exception as e:
        logger.debug("Failed to analyse %s: %s", symbol, e)
        return None


def multi_stock_prediction(stock_list: list, max_workers: int = 8) -> list:
    """
    Predict prices for all symbols in parallel.
    Returns list of dicts sorted by expected return (descending).
    """
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_analyse_one, sym): sym for sym in stock_list}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    results.sort(key=lambda x: x[3], reverse=True)

    return [
        {
            "Rank":           rank,
            "Symbol":         sym,
            "Current Price":  cur,
            "Predicted Price": pred,
            "Expected Return": ret,
        }
        for rank, (sym, cur, pred, ret) in enumerate(results, start=1)
    ]
