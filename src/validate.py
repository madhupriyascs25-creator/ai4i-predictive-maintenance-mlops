import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/ai4i.csv")

print("=" * 50)
print("DATA VALIDATION REPORT")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

# Target column validation
assert "Machine failure" in df.columns, "Target column not found!"

print("\nValidation Successful!")