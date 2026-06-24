import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn

# --------------------------------------------------
# Create required directories
# --------------------------------------------------
os.makedirs("mlruns", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --------------------------------------------------
# Configure MLflow
# --------------------------------------------------
print("Current Working Directory:", os.getcwd())
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AI4I_Predictive_Maintenance")

# --------------------------------------------------
# Load processed datasets
# --------------------------------------------------
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# --------------------------------------------------
# Split features and target
# --------------------------------------------------
X_train = train_df.drop("Machine failure", axis=1)
y_train = train_df["Machine failure"]

X_test = test_df.drop("Machine failure", axis=1)
y_test = test_df["Machine failure"]

# --------------------------------------------------
# Start MLflow run
# --------------------------------------------------
with mlflow.start_run():

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n========== MODEL RESULTS ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("===================================\n")

    # Log parameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Save model locally
    joblib.dump(model, "models/model.pkl")

    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

print("Model saved successfully!")
print("Saved at: models/model.pkl")