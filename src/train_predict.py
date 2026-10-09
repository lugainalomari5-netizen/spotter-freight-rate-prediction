
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Read data
data = pd.read_csv("data/train_test.csv")
data["date"] = pd.to_datetime(data["date"])


# 2. Time-based split
train = data[data["date"] < "2025-10-01"].copy()
test = data[data["date"] >= "2025-10-01"].copy()

print("Training rows:", len(train))
print("Testing rows:", len(test))


# 3. Clean invalid weights
train.loc[train["weight"] <= 0, "weight"] = np.nan
test.loc[test["weight"] <= 0, "weight"] = np.nan

# Use training medians only
weight_median = train["weight"].median()
market_median = train["market_index"].median()

for df in [train, test]:
    df["weight"] = df["weight"].fillna(weight_median)
    df["market_index"] = df["market_index"].fillna(market_median)


# 4. Date features
for df in [train, test]:
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day


# 5. Features
features = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day_of_month",
    "equipment",
]

X_train = pd.get_dummies(
    train[features],
    columns=["equipment"]
)

X_test = pd.get_dummies(
    test[features],
    columns=["equipment"]
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

y_train = train["posted_rate"]
y_test = test["posted_rate"]


# 6. Naive baseline: predict training mean
naive_predictions = np.full(
    len(y_test),
    y_train.mean()
)

naive_mae = mean_absolute_error(
    y_test,
    naive_predictions
)

naive_rmse = np.sqrt(
    mean_squared_error(y_test, naive_predictions)
)


# 7. Train Random Forest
print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)


# 8. Predict
predictions = model.predict(X_test)


# 9. Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))


# 10. Print comparison
print("\n" + "=" * 55)
print("MODEL COMPARISON - OCTOBER 2025")
print("=" * 55)

print("\nNaive Baseline (Training Mean)")
print(f"MAE:  ${naive_mae:,.2f}")
print(f"RMSE: ${naive_rmse:,.2f}")

print("\nRandom Forest + Equipment")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")

improvement = (
    (naive_mae - mae) / naive_mae
) * 100

print(f"\nMAE Improvement: {improvement:.2f}%")
# 11. Analyze expensive loads in October

results = pd.DataFrame({
    "actual_rate": y_test.to_numpy(),
    "predicted_rate": predictions
})

results["absolute_error"] = abs(
    results["actual_rate"] - results["predicted_rate"]
)

results["error"] = (
    results["predicted_rate"] - results["actual_rate"]
)

print("\n" + "=" * 55)
print("HIGH-PRICE LOAD ANALYSIS")
print("=" * 55)

print("\nTop 10 most expensive actual loads:")
print(
    results.nlargest(10, "actual_rate").to_string(index=False)
)

# Define expensive loads using training data only
high_price_threshold = y_train.quantile(0.95)

expensive = results[
    results["actual_rate"] >= high_price_threshold
]

print(f"\nHigh-price threshold: ${high_price_threshold:,.2f}")
print("Expensive loads in October:", len(expensive))

if len(expensive) > 0:
    print(
        "High-price MAE: "
        f"${expensive['absolute_error'].mean():,.2f}"
    )

    print(
        "Average prediction bias: "
        f"${expensive['error'].mean():,.2f}"
    )

print(
    "\nOverall MAE: "
    f"${results['absolute_error'].mean():,.2f}"
)
# 12. Investigate extreme pricing errors

extreme_results = test[
    [
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "market_index",
        "quote_signal",
        "posted_rate",
    ]
].copy()

extreme_results["predicted_rate"] = predictions

extreme_results["error"] = (
    extreme_results["predicted_rate"]
    - extreme_results["posted_rate"]
)

print("\n" + "=" * 55)
print("TOP 15 MOST UNDERPREDICTED LOADS")
print("=" * 55)

print(
    extreme_results.nsmallest(15, "error").to_string(index=False)
)

print("\n" + "=" * 55)
print("EXTREME LOADS ABOVE $10,000")
print("=" * 55)

extreme_loads = extreme_results[
    extreme_results["posted_rate"] > 10000
]

print("Number of extreme loads:", len(extreme_loads))

print(
    extreme_loads[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
            "predicted_rate",
        ]
    ].describe().round(2)
)
