
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Load data
data = pd.read_csv("data/train_test.csv")
december = pd.read_csv("data/december_chart_inputs.csv")

data["date"] = pd.to_datetime(data["date"])
december["date"] = pd.to_datetime(december["date"])

print("Historical rows:", len(data))
print("December rows:", len(december))


# 2. Prepare features
def prepare_features(df, weight_median):
    df = df.copy()

    # Clean invalid or missing weights
    df.loc[df["weight"] <= 0, "weight"] = np.nan
    df["weight"] = df["weight"].fillna(weight_median)

    # Date features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day

    return df


features = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "month",
    "day_of_week",
    "day_of_month",
]

categorical_columns = [
    "pickup",
    "delivery",
    "equipment",
]


# 3. Split by time
train = data[data["date"] < "2025-10-01"].copy()
test = data[data["date"] >= "2025-10-01"].copy()

weight_median = train.loc[
    train["weight"] > 0, "weight"
].median()

train = prepare_features(train, weight_median)
test = prepare_features(test, weight_median)


# 4. Encode categorical features
X_train = pd.get_dummies(
    train[features],
    columns=categorical_columns
)

X_test = pd.get_dummies(
    test[features],
    columns=categorical_columns
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

y_train = train["posted_rate"]
y_test = test["posted_rate"]


# 5. Evaluate December-compatible model
print("\nTraining December-compatible model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print("\n" + "=" * 55)
print("DECEMBER MODEL - OCTOBER VALIDATION")
print("=" * 55)

print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")


# 6. Retrain using all historical data
print("\nRetraining on all historical data...")

full_weight_median = data.loc[
    data["weight"] > 0, "weight"
].median()

full_train = prepare_features(data, full_weight_median)
december_ready = prepare_features(
    december, full_weight_median
)

X_full = pd.get_dummies(
    full_train[features],
    columns=categorical_columns
)

X_december = pd.get_dummies(
    december_ready[features],
    columns=categorical_columns
)

X_december = X_december.reindex(
    columns=X_full.columns,
    fill_value=0
)

final_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

final_model.fit(X_full, full_train["posted_rate"])


# 7. Predict all 31 December dates
december_predictions = final_model.predict(X_december)

if not np.isfinite(december_predictions).all():
    raise ValueError("December predictions contain invalid values.")

if (december_predictions <= 0).any():
    raise ValueError("December predictions must be positive.")


# 8. Fill the original December file
december["predicted_rate"] = december_predictions

expected_columns = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]

december = december[expected_columns]

assert len(december) == 31
assert december["predicted_rate"].notna().all()

# Save a backup of the completed predictions
december.to_csv(
    "outputs/december_predictions.csv",
    index=False
)

# Update the file required by score.py
december.to_csv(
    "data/december_chart_inputs.csv",
    index=False
)


# 9. Print results
print("\n" + "=" * 55)
print("DECEMBER PREDICTIONS CREATED")
print("=" * 55)

print("Rows:", len(december))
print("\nFirst 10 days:")
print(
    december[
        ["date", "predicted_rate"]
    ].head(10).to_string(index=False)
)

print("\nPrediction statistics:")
print(december["predicted_rate"].describe())
