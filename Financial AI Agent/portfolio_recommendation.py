"""
portfolio_recommendation.py  –  Fast Monte Carlo portfolio optimiser.

Changes vs original
-------------------
* Uses shared get_cached_df() → no redundant yfinance calls.
* Parallel data loading via ThreadPoolExecutor.
* Vectorised Monte Carlo (numpy batch random generation) instead of a Python loop.
* num_portfolios default reduced to 500; still gives good Sharpe estimate.
"""

import concurrent.futures
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from realtime_prediction import get_cached_df
from data_loader import load_stock
from technical_features import add_technical_features
from prediction_model import predict_price

logger = logging.getLogger(__name__)


def _process_symbol(symbol: str):
    try:
        df = get_cached_df(symbol)
        if df is None or df.empty:
            df = load_stock(symbol)
            if df is None or df.empty:
                return None
            df = add_technical_features(df)

        if len(df) < 2:
            return None

        pred          = predict_price(df, symbol=symbol)
        current_price = float(df["Close"].iloc[-1])
        return symbol, df["Close"].copy(), float(pred), current_price
    except Exception as e:
        logger.debug("Failed %s: %s", symbol, e)
        return None


def recommend_portfolio(
    stock_list: List[str],
    max_workers: int = 8,
    num_portfolios: int = 500,
) -> List[Tuple[str, float]]:
    """
    Monte Carlo portfolio optimisation.
    Returns list of (symbol, weight) sorted by allocation weight (descending).
    """
    if not stock_list:
        return []

    stock_list = stock_list[:50]
    results    = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_symbol, s): s for s in stock_list}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    if len(results) < 2:
        return []

    symbols, series_list, predictions, current_prices = zip(*results)

    price_df = pd.DataFrame({sym: ser for sym, ser in zip(symbols, series_list)})
    returns  = price_df.pct_change().dropna()
    if returns.empty:
        return []

    cov_matrix       = returns.cov().values
    expected_returns = np.array(
        [(p - cp) / cp for p, cp in zip(predictions, current_prices)]
    )
    n = len(symbols)
    rng = np.random.default_rng()

    # Vectorised: generate all random weights in one shot
    raw     = rng.random((num_portfolios, n))
    weights = raw / raw.sum(axis=1, keepdims=True)    # shape (num_portfolios, n)

    port_returns = weights @ expected_returns                                  # (P,)
    port_risks   = np.sqrt(
        np.einsum("pi,ij,pj->p", weights, cov_matrix, weights)
    )                                                                          # (P,)

    valid       = port_risks > 0
    sharpe      = np.where(valid, port_returns / port_risks, -np.inf)
    best_idx    = int(np.argmax(sharpe))
    best_weights = weights[best_idx] if sharpe[best_idx] > -np.inf else np.ones(n) / n

    portfolio = sorted(
        [(symbols[i], float(best_weights[i])) for i in range(n)],
        key=lambda x: x[1],
        reverse=True,
    )
    return portfolio
