from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import joblib
import os
from pathlib import Path

# Global model variables
rf_model = None
xgb_model = None
iso_model = None
scaler = None

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global rf_model, xgb_model, iso_model, scaler
    try:
        models_dir = BASE_DIR / "models"
        print(f"📁 Looking for models in: {models_dir}")
        print(f"📁 Directory exists: {models_dir.exists()}")
        
        rf_model = joblib.load(str(models_dir / "rf_model.pkl"))
        xgb_model = joblib.load(str(models_dir / "xgb_model.pkl"))
        iso_model = joblib.load(str(models_dir / "iso_model.pkl"))
        scaler = joblib.load(str(models_dir / "scaler.pkl"))
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print(f"❌ Files in models directory: {list((BASE_DIR / 'models').glob('*')) if (BASE_DIR / 'models').exists() else 'Directory not found'}")
    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    try:
        html_path = BASE_DIR / "public" / "index.html"
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return {"status": "AquaShield AI running", "error": str(e)}

@app.get("/health")
async def health():
    models_loaded = scaler is not None
    return {"status": "ok", "models_loaded": models_loaded}

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
            return JSONResponse(status_code=500, content={"error": "Models not initialized. Please wait for server to load."})
        
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
        print(f"❌ Prediction error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
