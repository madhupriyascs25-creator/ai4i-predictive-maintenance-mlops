from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest
from fastapi import Response

import pandas as pd
import joblib

app = FastAPI(
    title="AI4I Predictive Maintenance API",
    version="1.0"
)

# Load trained model
model = joblib.load("models/model.pkl")

# Prometheus Counter
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

class MachineInput(BaseModel):
    Type: int
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    TWF: int
    HDF: int
    PWF: int
    OSF: int
    RNF: int


@app.get("/")
def home():
    return {
        "message": "AI4I Predictive Maintenance API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: MachineInput):

    REQUEST_COUNT.inc()
    print("Endpoint")

    df = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
        "TWF": data.TWF,
        "HDF": data.HDF,
        "PWF": data.PWF,
        "OSF": data.OSF,
        "RNF": data.RNF
    }])

    prediction = model.predict(df)[0]

    return {
        "machine_failure_prediction": int(prediction)
    }

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )