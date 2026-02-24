import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# STEP 1: LOAD DATA + MODEL PLANNING
# --------------------------------------------------

# Load dataset (explicit path)
df = pd.read_csv(
    r"C:\Users\AnushV\Documents\GitHub\stock-data-analysis-eda\data\final_combined_dataset.csv",
    low_memory=False
)

# Explicit numeric feature list
features = [
    "Open", "High", "Low", "Volume",
    "Lag_1", "Lag_5",
    "Rolling_Mean", "Rolling_Std",
    "RSI", "Volatility"
]

target = "Close"

# Force numeric conversion (CRITICAL FIX)
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with invalid values
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Time-based split (80-20)
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

#print("STEP 1 COMPLETED")
#print("Features used:", list(X.columns))
#print("Training samples:", len(X_train))
#print("Testing samples:", len(X_test))
#print("Data types:\n", X.dtypes)
#print("-" * 50)

# --------------------------------------------------
# STEP 2: MODEL SELECTION
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=100
    )
}

#print("STEP 2 COMPLETED")
#print("Models selected:")
#for name in models:
#    print("-", name)
#print("-" * 50)

# --------------------------------------------------
# STEP 3: TRAINING, TESTING & EVALUATION
# --------------------------------------------------

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2"]
)

# Save results (for screenshots)
results_df.to_csv(
    r"C:\Users\AnushV\Documents\GitHub\stock-data-analysis-eda\data\model_comparison_results.csv",
    index=False
)

#print("STEP 3 COMPLETED")
#print("Model evaluation results:")
#print(results_df)

#best_model = results_df.sort_values("RMSE").iloc[0]
#print("Best Model Selected:")
#print(best_model)

# ----------------------------------------------------
# STEP 7: HYPERPARAMETER TUNING FOR LASSO REGRESSION
# ----------------------------------------------------

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0]
}

lasso = Lasso(max_iter=10000)

grid = GridSearchCV(
    lasso,
    param_grid,
    scoring="neg_mean_squared_error",
    cv=5
)

grid.fit(X_train, y_train)

best_lasso = grid.best_estimator_

#print("Best alpha:", grid.best_params_)

tuned_preds = best_lasso.predict(X_test)

#print("Tuned MAE:", mean_absolute_error(y_test, tuned_preds))
#print("Tuned RMSE:", np.sqrt(mean_squared_error(y_test, tuned_preds)))
#print("Tuned R2:", r2_score(y_test, tuned_preds))

#print("Before Tuning:")
#print(results_df[results_df["Model"] == "Lasso Regression"])

#print("\nAfter Tuning:")
#print("RMSE:", np.sqrt(mean_squared_error(y_test, tuned_preds)))

import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Actual vs Predicted Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Close Price", linewidth=2)
plt.plot(tuned_preds, label="Predicted Close Price", linestyle="--")
plt.xlabel("Test Data Index")
plt.ylabel("Close Price")
plt.title("Actual vs Predicted Close Price (Tuned Lasso Regression)")
plt.legend()
plt.tight_layout()
plt.show()

# 2️⃣ Prediction Error Distribution
errors = y_test.values - tuned_preds

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution (Tuned Lasso Regression)")
plt.tight_layout()
plt.show()