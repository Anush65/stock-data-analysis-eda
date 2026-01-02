import pandas as pd

df_price = pd.read_csv("dataset_stock_prices.csv")
df_tech = pd.read_csv("dataset_technical_indicators.csv")
df_company = pd.read_csv("dataset_company_metadata.csv")

# Merge price + technical data
df_merged = df_price.merge(
    df_tech,
    on=["Date", "Ticker"],
    how="inner"
)

# Merge company metadata
df_final = df_merged.merge(
    df_company,
    on="Ticker",
    how="left"
)

df_final.to_csv("final_combined_dataset.csv", index=False)

print("✅ Final dataset created successfully!")
