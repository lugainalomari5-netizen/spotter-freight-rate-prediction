import pandas as pd
import numpy as np


# Load data
train = pd.read_csv("data/train_test.csv")
validation = pd.read_csv("data/validation.csv")


print("=" * 60)
print("DATASET SHAPES")
print("=" * 60)

print("Train:", train.shape)
print("Validation:", validation.shape)


print("\n" + "=" * 60)
print("TRAIN COLUMNS")
print("=" * 60)

print(train.columns.tolist())


print("\n" + "=" * 60)
print("FIRST 5 TRAIN ROWS")
print("=" * 60)

print(train.head())


print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(train.dtypes)


print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = pd.DataFrame({
    "missing_count": train.isna().sum(),
    "missing_pct": train.isna().mean() * 100
})

print(missing[missing["missing_count"] > 0])


print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print("Duplicate rows:", train.duplicated().sum())


print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(train.describe())
# ============================================================
# TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TARGET: POSTED RATE")
print("=" * 60)

print(train["posted_rate"].describe())

print("\nHighest 10 posted rates:")
print(
    train.nlargest(10, "posted_rate")[
        ["pickup", "delivery", "distance", "weight", "date", "posted_rate"]
    ]
)


# ============================================================
# WEIGHT CHECK
# ============================================================

print("\n" + "=" * 60)
print("WEIGHT CHECK")
print("=" * 60)

print(train["weight"].describe())

print("\nNon-positive weights:")
print((train["weight"] <= 0).sum())

print("\nLowest 10 weights:")
print(
    train.nsmallest(10, "weight")[
        ["pickup", "delivery", "distance", "weight", "posted_rate"]
    ]
)


# ============================================================
# DATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DATE ANALYSIS")
print("=" * 60)

train["date"] = pd.to_datetime(train["date"])
validation["date"] = pd.to_datetime(validation["date"])

print("Train start:", train["date"].min())
print("Train end:", train["date"].max())

print("Validation start:", validation["date"].min())
print("Validation end:", validation["date"].max())


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

for column in ["pickup", "delivery", "equipment"]:
    print(f"\n{column}:")
    print("Unique values:", train[column].nunique())
    print(train[column].value_counts().head(10))


# ============================================================
# TRAIN VS VALIDATION CATEGORIES
# ============================================================

print("\n" + "=" * 60)
print("UNSEEN CATEGORIES IN VALIDATION")
print("=" * 60)

for column in ["pickup", "delivery", "equipment"]:
    train_values = set(train[column].dropna())
    validation_values = set(validation[column].dropna())

    unseen = validation_values - train_values

    print(f"{column}: {len(unseen)} unseen values")

    if unseen:
        print(sorted(unseen))


# ============================================================
# SIMPLE CORRELATIONS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION WITH POSTED RATE")
print("=" * 60)

numeric_columns = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate"
]

print(
    train[numeric_columns]
    .corr()["posted_rate"]
    .sort_values(ascending=False)
)
print("\n" + "=" * 60)
print("MONTHLY RATE ANALYSIS")
print("=" * 60)

train["month"] = train["date"].dt.month

monthly_rates = train.groupby("month")["posted_rate"].agg(
    ["count", "mean", "median"]
)

print(monthly_rates)


print("\n" + "=" * 60)
print("EQUIPMENT RATE ANALYSIS")
print("=" * 60)

equipment_rates = train.groupby("equipment")["posted_rate"].agg(
    ["count", "mean", "median"]
)

print(equipment_rates)


print("\n" + "=" * 60)
print("RATE PER MILE")
print("=" * 60)

train["rate_per_mile"] = train["posted_rate"] / train["distance"]

print(train["rate_per_mile"].describe())

print("\nAverage rate per mile by equipment:")
print(
    train.groupby("equipment")["rate_per_mile"]
    .agg(["mean", "median"])
) 