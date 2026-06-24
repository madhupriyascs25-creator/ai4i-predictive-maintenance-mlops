# AI4I Predictive Maintenance MLOps Pipeline

## Project Overview

This project implements an end-to-end MLOps pipeline for predictive maintenance using the AI4I Predictive Maintenance Dataset.

The objective is to predict machine failure based on operational parameters and failure indicators while demonstrating industry-standard MLOps practices including data versioning, experiment tracking, containerization, CI/CD, monitoring, and drift detection.

---

## Dataset

AI4I Predictive Maintenance Dataset

Target Variable:

* Machine failure

Features:

* Type
* Air temperature [K]
* Process temperature [K]
* Rotational speed [rpm]
* Torque [Nm]
* Tool wear [min]
* TWF
* HDF
* PWF
* OSF
* RNF

---

## Project Structure

```text
.
├── data
│   ├── raw
│   └── processed
├── models
├── reports
├── src
│   ├── validate.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
├── .github
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

---

## MLOps Components Implemented

### Data Versioning

* DVC

### Data Validation

* Missing value checks
* Duplicate record checks
* Schema validation

### Data Processing

* Feature selection
* Label encoding
* Train-test split

### Model Training

* Random Forest Classifier

### Experiment Tracking

* MLflow with SQLite backend

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Model Serving

* FastAPI

### Containerization

* Docker

---

## Model Performance

Accuracy: 99.90%

Precision: 100%

Recall: 97.06%

F1 Score: 98.51%

---

## Run Project

### DVC Pipeline

```bash
dvc repro
```

### MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### FastAPI

```bash
uvicorn src.app:app --reload
```

### Docker

```bash
docker build -t ai4i-mlops .
docker run -p 8000:8000 ai4i-mlops
```

---

## Future Enhancements

* GitHub Actions CI/CD
* Prometheus Monitoring
* Evidently Drift Detection
* DevSecOps Security Scanning
* Model Registry
