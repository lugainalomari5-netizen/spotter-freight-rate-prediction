
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


# 1. Load data
train = pd.read_csv("data/train_test.csv")
validation = pd.read_csv("data/validation.csv")
template = pd.read_csv("data/validation_predictions_template.csv")

print("Training rows:", len(train))
print("Validation rows:", len(validation))


# 2. Convert dates
train["date"] = pd.to_datetime(train["date"])
validation["date"] = pd.to_datetime(validation["date"])


# 3. Clean invalid weights
train.loc[train["weight"] <= 0, "weight"] = np.nan
validation.loc[validation["weight"] <= 0, "weight"] = np.nan

# Calculate medians from training data only
weight_median = train["weight"].median()
market_median = train["market_index"].median()

for df in [train, validation]:
    df["weight"] = df["weight"].fillna(weight_median)
    df["market_index"] = df["market_index"].fillna(market_median)


# 4. Create date features
for df in [train, validation]:
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day


# 5. Select model features
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


# 6. Encode equipment
X_train = pd.get_dummies(
    train[features],
    columns=["equipment"]
)

X_validation = pd.get_dummies(
    validation[features],
    columns=["equipment"]
)

# Match columns to training
X_validation = X_validation.reindex(
    columns=X_train.columns,
    fill_value=0
)

y_train = train["posted_rate"]


# 7. Train final model on all 48,000 rows
print("\nTraining final Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)


# 8. Predict validation prices
print("Predicting validation prices...")

predictions = model.predict(X_validation)

# Ensure predictions are finite and positive
if not np.isfinite(predictions).all():
    raise ValueError("Predictions contain NaN or infinity.")

if (predictions <= 0).any():
    raise ValueError("Predictions must be positive.")


# 9. Create submission file using the official template
if len(validation) != len(template):
    raise ValueError("Validation and template row counts differ.")

if not validation["load_id"].equals(template["load_id"]):
    raise ValueError("Validation IDs do not match the template.")

submission = template[["load_id"]].copy()
submission["predicted_rate"] = predictions

submission.to_csv(
    "validation_predictions.csv",
    index=False
)


# 10. Final checks
assert len(submission) == 12000
assert submission["load_id"].is_unique
assert submission.columns.tolist() == [
    "load_id",
    "predicted_rate",
]

print("\n" + "=" * 50)
print("VALIDATION PREDICTIONS CREATED SUCCESSFULLY")
print("=" * 50)

print("Output file: validation_predictions.csv")
print("Rows:", len(submission))
print("Columns:", submission.columns.tolist())

print("\nFirst 5 predictions:")
print(submission.head())

print("\nPrediction statistics:")
print(submission["predicted_rate"].describe())
