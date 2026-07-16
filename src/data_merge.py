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