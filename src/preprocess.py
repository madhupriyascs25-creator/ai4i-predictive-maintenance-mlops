import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
df = pd.read_csv("data/raw/ai4i.csv")

# Remove unnecessary columns
drop_cols = ["UDI", "Product ID"]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Encode categorical feature
le = LabelEncoder()

if "Type" in df.columns:
    df["Type"] = le.fit_transform(df["Type"])

# Split features and target
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save processed data
os.makedirs("data/processed", exist_ok=True)

train_df = X_train.copy()
train_df["Machine failure"] = y_train

test_df = X_test.copy()
test_df["Machine failure"] = y_test

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Preprocessing completed successfully!")
print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)