import numpy as np


def generate_explanation(
    symbol,
    close_price,
    sma20,
    rsi,
    volatility,
    sentiment,
    inflation,
    prediction
):

    explanation = "AI Financial Analysis Report\n"
    explanation += "-----------------------------------\n\n"

    explanation += "Stock: " + symbol + "\n"
    explanation += "Current Price: " + str(round(close_price, 2)) + "\n"
    explanation += "Next Day AI Prediction (LSTM): " + str(round(prediction, 2)) + "\n\n"

    # -------- Trend Analysis ----------
    if close_price > sma20:
        trend = "Bullish"
        explanation += "Trend Analysis: Stock is trading above SMA20 indicating bullish momentum.\n"
    else:
        trend = "Bearish"
        explanation += "Trend Analysis: Stock is trading below SMA20 indicating bearish momentum.\n"

    # -------- RSI Analysis ----------
    if rsi > 70:
        explanation += "RSI Analysis: RSI indicates overbought condition. Price correction possible.\n"
    elif rsi < 30:
        explanation += "RSI Analysis: RSI indicates oversold condition. Price rebound possible.\n"
    else:
        explanation += "RSI Analysis: RSI indicates neutral market condition.\n"

    # -------- Volatility ----------
    if volatility > 0.03:
        explanation += "Risk Analysis: High volatility detected. Investment carries higher risk.\n"
    else:
        explanation += "Risk Analysis: Volatility is moderate suggesting stable movement.\n"

    # -------- Sentiment ----------
    if sentiment == "Positive":
        explanation += "Market Sentiment: Positive sentiment may support price growth.\n"
    elif sentiment == "Negative":
        explanation += "Market Sentiment: Negative sentiment may pressure prices.\n"
    else:
        explanation += "Market Sentiment: Neutral market sentiment detected.\n"

    # -------- Macro Factors ----------
    explanation += "Macro Factor: Inflation rate considered in analysis = " + str(round(inflation, 2)) + "%\n"

    # -------- LSTM Prediction Insight ----------
    price_change = prediction - close_price
    percent_change = (price_change / close_price) * 100

    explanation += "\nAI Prediction Insight (LSTM Model):\n"

    if prediction > close_price:
        explanation += "Predicts the stock price may increase tomorrow.\n"
        suggestion = "Possible BUY opportunity."
    else:
        explanation += "Predicts the stock price may decline tomorrow.\n"
        suggestion = "Possible SELL or HOLD strategy."

    explanation += "Expected Price Change: " + str(round(price_change, 2)) + "\n"
    explanation += "Expected Return: " + str(round(percent_change, 2)) + "%\n"

    explanation += "\nOverall Market Trend: " + trend
    explanation += "\nInvestment Suggestion: " + suggestion

    return explanation