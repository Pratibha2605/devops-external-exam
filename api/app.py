from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from utils import send_alert_email

app = FastAPI()

MODEL_PATH = "model.pkl"

@app.get("/")
def root():
    return {"message": "KT Practical Exam - FastAPI Server Running"}

@app.post("/predict")
def predict(payload: dict):
    try:
        # Validate schema
        required_keys = ["sqft", "bedrooms", "bathrooms"]

        for key in required_keys:
            if key not in payload:
                raise ValueError(f"Missing key: {key}")

        sqft = float(payload["sqft"])
        bedrooms = int(payload["bedrooms"])
        bathrooms = int(payload["bathrooms"])

        model = joblib.load(MODEL_PATH)

        features = np.array([[sqft, bedrooms, bathrooms]])
        prediction = model.predict(features)

        return {"prediction": float(prediction[0])}

    except Exception as e:
        send_alert_email(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
