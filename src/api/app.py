from fastapi import FastAPI,Request
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import time
from prometheus_fastapi_instrumentator import Instrumentator

# ---------------------------------------------------------
# Logging + Metrics (ADD THIS AT THE TOP OF app.py)
# 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logging.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logging.info(
        f"Completed request: {request.method} {request.url.path} "
        f"Status: {response.status_code} Time: {process_time:.2f} ms"
    )
    return response

Instrumentator().instrument(app).expose(app)
# Load ONLY the model pipeline
model = joblib.load("models/model.pkl")
# DO NOT load preprocessor.pkl (it causes double-transform)
# preprocessor = joblib.load("models/preprocessor.pkl")


class InputData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.post("/predict")
def predict(data: InputData):
    import pandas as pd

    # Build DataFrame with column names
    sample = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalach": data.thalach,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal
    }])


    # Fix classifier attribute
    clf = model.named_steps["model"]
    if not hasattr(clf, "multi_class"):
     clf.multi_class = "auto"
    # Predict using the pipeline (it already contains preprocessing)
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][pred]

    return {
        "prediction": int(pred),
        "confidence": float(prob)
    }
