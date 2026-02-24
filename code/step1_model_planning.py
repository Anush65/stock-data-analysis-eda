import pandas as pd

df = pd.read_csv(r"C:\Users\AnushV\Documents\GitHub\stock-data-analysis-eda\data\final_combined_dataset.csv")

features = [
    "Open", "High", "Low", "Volume",
    "Lag_1", "Lag_5", "Rolling_Mean",
    "Rolling_Std", "RSI", "Volatility"
]

X = df[features]
y = df["Close"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
