from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
import numpy as np
import joblib

# Global model variables
rf_model = None
xgb_model = None
iso_model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global rf_model, xgb_model, iso_model, scaler
    try:
        rf_model = joblib.load("models/rf_model.pkl")
        xgb_model = joblib.load("models/xgb_model.pkl")
        iso_model = joblib.load("models/iso_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Models not loaded: {e}")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    try:
        with open("public/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except:
        return {"status": "AquaShield AI running", "message": "Use /predict endpoint"}

@app.get("/health")
async def health():
    return {"status": "ok"}

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
        if scaler is None:
            return JSONResponse(status_code=500, content={"error": "Models not loaded"})
        
        # Prepare input
        features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        features_scaled = scaler.transform(features)
        
        # Predictions
        rf_pred = float(rf_model.predict(features_scaled)[0])
        xgb_pred = float(xgb_model.predict(features_scaled)[0])
        iso_pred = float(iso_model.predict(features_scaled)[0])
        
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
