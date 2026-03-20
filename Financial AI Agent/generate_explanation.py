def generate_explanation(symbol, close, sma, rsi, volatility, sentiment, inflation, interest, prediction):
    """
    Generate a textual explanation for a stock based on indicators, sentiment, macro data, and predicted price.

    Parameters:
        symbol (str): Stock ticker symbol
        close (float): Latest closing price
        sma (float): 20-day SMA
        rsi (float): RSI value
        volatility (float): Volatility
        sentiment (float): Sentiment score
        inflation (float): Current inflation rate
        interest (float): Current interest rate
        prediction (float): Predicted next-day price

    Returns:
        explanation (str): AI-style textual explanation
    """

    # Simple logic-based explanation
    trend = "upward" if prediction > close else "downward"
    signal = "Buy" if prediction > close else "Sell"

    explanation = (
        f"Stock {symbol} Analysis:\n"
        f"- Latest Close Price: {close:.2f}\n"
        f"- 20-day SMA: {sma:.2f}\n"
        f"- RSI: {rsi:.2f}\n"
        f"- Volatility: {volatility:.4f}\n"
        f"- Market Sentiment: {sentiment:.2f}\n"
        f"- Inflation: {inflation:.2f}%, Interest Rate: {interest:.2f}%\n"
        f"- Predicted Price: {prediction:.2f}\n"
        f"- Predicted Trend: {trend}\n"
        f"- Recommended Action: {signal}\n"
    )

    return explanation