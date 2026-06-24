from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="AI4I Predictive Maintenance API",
    version="1.0"
)

# Load trained model
model = joblib.load("models/model.pkl")


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