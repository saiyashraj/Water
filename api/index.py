from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import joblib
import os

app = FastAPI()

# Load models
try:
    rf_model = joblib.load("models/rf_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    iso_model = joblib.load("models/iso_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print(f"Warning: Models not found. {e}")

@app.get("/")
async def root():
    try:
        with open("public/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except:
        return {"message": "AquaShield AI - Water Quality Prediction API", "status": "running"}

@app.get("/predict")
async def predict(
    ph: float,
    hardness: float,
    solids: float,
    chloramines: float,
    sulfate: float,
    conductivity: float,
    organic_carbon: float,
    trihalomethanes: float,
    turbidity: float
):
    try:
        # Prepare input
        features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        features_scaled = scaler.transform(features)
        
        # Predictions
        rf_pred = rf_model.predict(features_scaled)[0]
        xgb_pred = xgb_model.predict(features_scaled)[0]
        iso_pred = iso_model.predict(features_scaled)[0]
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + xgb_pred + iso_pred) / 3
        
        return {
            "potable": int(ensemble_pred > 0.5),
            "confidence": float(ensemble_pred),
            "rf_prediction": float(rf_pred),
            "xgb_prediction": float(xgb_pred),
            "iso_prediction": float(iso_pred)
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
