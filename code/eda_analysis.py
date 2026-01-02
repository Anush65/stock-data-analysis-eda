import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("final_combined_dataset.csv", low_memory=False)

# -----------------------------
# CLEAN & FIX DATA TYPES
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_cols = ["Close", "Open", "High", "Low", "Volume"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Date", "Close"])
df = df.sort_values("Date")

# -----------------------------
# CREATE Daily_Return if missing
# -----------------------------
if "Daily_Return" not in df.columns:
    df["Daily_Return"] = df["Close"].pct_change()

# -----------------------------
# PLOTS
# -----------------------------

# 1. Price Trend
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Close"])
plt.title("Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("price_trend.png")
plt.show()

# 2. Volume Trend
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Volume"])
plt.title("Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("volume_trend.png")
plt.show()

# 3. Returns Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Daily_Return"].dropna(), bins=30)
plt.title("Daily Returns Distribution")
plt.tight_layout()
plt.savefig("returns_distribution.png")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.select_dtypes(include=["float64", "int64"]).corr()
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

print("✅ EDA completed successfully!")
