import pandas as pd

df = pd.read_csv("dataset_stock_prices.csv")

# Ensure numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["Daily_Return"] = pd.to_numeric(df["Daily_Return"], errors="coerce")

# Feature Engineering
df["Lag_1"] = df["Close"].shift(1)
df["Lag_5"] = df["Close"].shift(5)

df["Rolling_Mean"] = df["Close"].rolling(10).mean()
df["Rolling_Std"] = df["Close"].rolling(10).std()

# RSI
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

df["Volatility"] = df["Daily_Return"].rolling(10).std()

# ✅ KEEP Ticker column
df_tech = df[
    [
        "Date",
        "Ticker",
        "Lag_1",
        "Lag_5",
        "Rolling_Mean",
        "Rolling_Std",
        "RSI",
        "Volatility",
        "Daily_Return"
    ]
]

df_tech.to_csv("dataset_technical_indicators.csv", index=False)

print("✅ Dataset 2 created successfully (with Ticker)")
