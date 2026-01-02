import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

all_data = []

for ticker in tickers:
    df = yf.download(ticker, start="2022-01-01", end="2024-01-01")
    df.reset_index(inplace=True)
    
    df["Ticker"] = ticker
    df["Daily_Return"] = df["Close"].pct_change()
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_14"] = df["Close"].rolling(14).mean()
    
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv("dataset_stock_prices.csv", index=False)

print("✅ Dataset 1 created correctly (multiple stocks)")
