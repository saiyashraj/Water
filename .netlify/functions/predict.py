import json
import joblib
import numpy as np

def handler(event, context):
    try:
        # Parse query parameters
        params = event.get('queryStringParameters', {})
        
        # Load models
        rf_model = joblib.load("models/rf_model.pkl")
        xgb_model = joblib.load("models/xgb_model.pkl")
        iso_model = joblib.load("models/iso_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        
        # Prepare input
        features = np.array([[
            float(params['ph']),
            float(params['hardness']),
            float(params['solids']),
            float(params['chloramines']),
            float(params['sulfate']),
            float(params['conductivity']),
            float(params['organic_carbon']),
            float(params['trihalomethanes']),
            float(params['turbidity'])
        ]])
        
        features_scaled = scaler.transform(features)
        
        # Predictions
        rf_pred = float(rf_model.predict(features_scaled)[0])
        xgb_pred = float(xgb_model.predict(features_scaled)[0])
        iso_pred = float(iso_model.predict(features_scaled)[0])
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + xgb_pred + iso_pred) / 3
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "potable": int(ensemble_pred > 0.5),
                "confidence": ensemble_pred,
                "rf_prediction": rf_pred,
                "xgb_prediction": xgb_pred,
                "iso_prediction": iso_pred
            })
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
