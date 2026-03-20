# ...existing code...
import logging
import concurrent.futures
from typing import List, Dict, Optional

from stock_database import get_stock_list
from data_loader import load_stock
from technical_features import add_technical_features
from sentiment_analysis import get_sentiment
from macro_loader import get_macro_data
from prediction_model import predict_price
from genai_engine import generate_explanation
from portfolio_recommendation import recommend_portfolio
from realtime_prediction import get_cached_df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_symbol(symbol: str, use_cache: bool = True) -> Optional[Dict]:
    """
    Fast analysis for a single symbol:
      - tries cached engineered DataFrame for quick response
      - falls back to synchronous load if cache miss
      - computes prediction and explanation
    Returns dict with summary or None on failure.
    """
    try:
        df = None
        if use_cache:
            df = get_cached_df(symbol)
        if df is None or df.empty:
            # force synchronous refresh for first-time requests
            df = get_cached_df(symbol, force_refresh=True, wait_for_refresh=True)
        if df is None or df.empty:
            # final fallback to direct load
            df = load_stock(symbol)
        if df is None or df.empty:
            logger.debug("No data for %s", symbol)
            return None

        df = add_technical_features(df)
        if df is None or df.empty:
            logger.debug("Not enough data after features for %s", symbol)
            return None

        prediction = predict_price(df)
        latest = df.iloc[-1]
        sentiment = get_sentiment()
        inflation, interest = get_macro_data()

        explanation = generate_explanation(
            symbol,
            float(latest["Close"]),
            float(latest.get("SMA_20", 0)),
            float(latest.get("RSI", 0)),
            float(latest.get("Volatility", 0)),
            sentiment,
            inflation,
            prediction
        )

        current_price = float(latest["Close"])
        expected_return = (float(prediction) - current_price) / current_price if current_price != 0 else 0.0

        return {
            "Symbol": symbol,
            "DataFrame": df,
            "Current Price": current_price,
            "Predicted Price": float(prediction),
            "Expected Return": expected_return,
            "Explanation": explanation,
        }

    except Exception as e:
        logger.exception("Failed to analyze %s: %s", symbol, e)
        return None


def analyze_many(symbols: List[str], max_workers: int = 6, use_cache: bool = True) -> List[Dict]:
    """
    Analyze multiple symbols in parallel and return sorted list by expected return.
    Lightweight and responsive: uses cached data where possible.
    """
    results = []
    if not symbols:
        return results

    symbols = symbols[:50]  # safety cap
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_symbol, s, use_cache): s for s in symbols}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    # sort by expected return descending
    results.sort(key=lambda r: r.get("Expected Return", 0), reverse=True)
    # attach rank
    for i, r in enumerate(results, start=1):
        r["Rank"] = i

    return results


def recommend_portfolio_fast(stock_list: List[str], max_workers: int = 6, num_portfolios: int = 500):
    """
    Thin wrapper around existing recommend_portfolio. Kept to provide a single entry point.
    """
    # recommend_portfolio is already parallel and optimized; call directly
    return recommend_portfolio(stock_list)


if __name__ == "__main__":
    # simple CLI test: analyze top 8 symbols and print summary
    syms = get_stock_list()[:8]
    summary = analyze_many(syms, max_workers=6)
    for item in summary:
        print(f"{item['Rank']}. {item['Symbol']} | Current: {item['Current Price']:.2f} | Predicted: {item['Predicted Price']:.2f} | Return: {item['Expected Return']*100:.2f}%")
# ...existing code...