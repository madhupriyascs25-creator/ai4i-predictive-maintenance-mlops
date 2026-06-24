import json
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Load model
model = joblib.load("models/model.pkl")

# Load test data
test_df = pd.read_csv("data/processed/test.csv")

X_test = test_df.drop("Machine failure", axis=1)
y_test = test_df["Machine failure"]

# Predictions
y_pred = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
}

print("\nEvaluation Metrics")
print(metrics)

os.makedirs("reports", exist_ok=True)

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics saved to reports/metrics.json")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))