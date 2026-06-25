import pandas as pd
from scipy.stats import ks_2samp
import json
import os

# Load datasets
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

drift_results = {}

for col in train.columns:
    # 1. Compatibility Check: Ensure the column actually exists in both datasets
    if col not in test.columns:
        print(f"Skipping '{col}': Not found in the test dataset.")
        continue
        
    # 2. Type Check: Strictly ensure the column is numeric (ignores objects, datetimes, categories)
    if pd.api.types.is_numeric_dtype(train[col]):
        
        # 3. Handle Missing Values: Drop NaNs so ks_2samp doesn't break or return NaN
        train_clean = train[col].dropna()
        test_clean = test[col].dropna()
        
        # 4. Edge Case Check: Ensure there is actually data left to compare
        if len(train_clean) == 0 or len(test_clean) == 0:
            print(f"Skipping '{col}': Insufficient data after removing NaNs.")
            continue
            
        # Run the KS test
        stat, p_value = ks_2samp(train_clean, test_clean)
        
        drift_results[col] = {
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05) # Cast explicitly to standard Python bool
        }
    else:
        print(f"Skipping '{col}': Non-numeric data type ({train[col].dtype}).")

# Ensure output directory exists
os.makedirs("reports", exist_ok=True)

# Save to JSON safely
with open("reports/drift_report.json", "w") as f:
    json.dump(drift_results, f, indent=4)

print("Drift report generated successfully!")