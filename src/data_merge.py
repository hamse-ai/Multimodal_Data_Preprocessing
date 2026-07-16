import pandas as pd

# Load datasets
profiles = pd.read_csv("data/raw/customer_social_profiles.csv")
transactions = pd.read_csv("data/raw/customer_transactions.csv")

# Convert IDs
profiles["customer_id"] = (
    profiles["customer_id_new"]
    .str.replace("A", "", regex=False)
    .astype(int)
)

# Merge
merged = pd.merge(
    profiles,
    transactions,
    left_on="customer_id",
    right_on="customer_id_legacy",
    how="inner"
)

print("Merged Shape:", merged.shape)

# Cleaning
print("\nMissing values:")
print(merged.isnull().sum())

merged = merged.drop_duplicates()

# Fill missing numeric values
numeric_cols = merged.select_dtypes(include=["number"]).columns
merged[numeric_cols] = merged[numeric_cols].fillna(
    merged[numeric_cols].median()
)

# Fill missing categorical values
categorical_cols = merged.select_dtypes(include=["object"]).columns
merged[categorical_cols] = merged[categorical_cols].fillna("Unknown")

# Feature Engineering
merged["purchase_date"] = pd.to_datetime(merged["purchase_date"])

merged["purchase_month"] = merged["purchase_date"].dt.month
merged["purchase_day"] = merged["purchase_date"].dt.day

merged["high_value_customer"] = (
    merged["purchase_amount"] > 300
).astype(int)

merged["engagement_level"] = pd.cut(
    merged["engagement_score"],
    bins=[0, 40, 70, 100],
    labels=["Low", "Medium", "High"]
)

merged.drop(
    columns=["customer_id_new", "customer_id_legacy"],
    inplace=True
)

print(merged.columns.tolist())

merged = merged[
    [
        "customer_id",
        "social_media_platform",
        "engagement_score",
        "purchase_interest_score",
        "review_sentiment",
        "transaction_id",
        "purchase_amount",
        "purchase_date",
        "purchase_month",
        "purchase_day",
        "customer_rating",
        "high_value_customer",
        "engagement_level",
        "product_category"
    ]
]

# Save
merged.to_csv(
    "data/processed/merged_customer_data.csv",
    index=False
)

print("\nSaved merged dataset.")
print(merged.head())
print(merged.info())