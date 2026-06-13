import sys
import os
import math
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import backend

def test_ticker(ticker):
    print(f"\n=== Testing Ticker: {ticker} ===")
    
    # 1. Fetch Prices
    prices, price_logs = backend.fetch_stock_prices(ticker, days=45)
    print(f"Prices fetched: {len(prices)} days. Log count: {len(price_logs)}")
    
    # Calculate trend
    closes = [p["close"] for p in prices]
    price_trend = (closes[-1] - closes[0]) / closes[0]
    print(f"Price trend (last vs first close): {price_trend:+.2%}")
    
    # 2. Fetch News
    news, news_logs = backend.fetch_stock_news(ticker)
    print(f"News fetched: {len(news)} items.")
    
    # 3. Sentiment Score
    sentiment = backend.analyze_sentiment_local(ticker, news)
    print(f"Sentiment Score: {sentiment['sentiment_score']:.2f}, Direction: {sentiment['direction']}")
    
    # 4. Price signal
    sma5_val = sum(closes[-5:]) / 5 if len(closes) >= 5 else closes[-1]
    sma15_val = sum(closes[-15:]) / 15 if len(closes) >= 15 else closes[-1]
    trend_direction = 1.0 if sma5_val >= sma15_val else -1.0
    
    price_sig = trend_direction * 0.4 + price_trend * 3.0
    price_sig = max(-1.0, min(1.0, price_sig))
    print(f"SMA5: ${sma5_val:.2f}, SMA15: ${sma15_val:.2f}")
    print(f"Price Signal: {price_sig:+.2f}")
    
    combined_score = price_sig * 0.45 + sentiment['sentiment_score'] * 0.55
    pred_prob = 0.5 + combined_score * 0.45
    print(f"Combined Score: {combined_score:+.2f}, Prediction Probability: {pred_prob:.2%}")
    print(f"Prediction: {'UP' if pred_prob >= 0.5 else 'DOWN'}")

test_ticker("AAPL")
test_ticker("NFLX")
test_ticker("COIN")
test_ticker("INTC")
test_ticker("NVDA")
