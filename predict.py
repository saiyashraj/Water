import joblib
import numpy as np

# ----------------------------
# LOAD MODELS
# ----------------------------

rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------

def predict_water_quality(sample):

    """
    sample format:
    [
        ph,
        Hardness,
        Solids,
        Chloramines,
        Sulfate,
        Conductivity,
        Organic_carbon,
        Trihalomethanes,
        Turbidity
    ]
    """

    sample_scaled = scaler.transform([sample])

    # Random Forest probability
    rf_prob = rf_model.predict_proba(sample_scaled)[0][1]

    # XGBoost probability
    xgb_prob = xgb_model.predict_proba(sample_scaled)[0][1]

    # Average probability
    contamination_probability = ((rf_prob + xgb_prob) / 2) * 100

    # Final prediction
    status = "Unsafe" if contamination_probability > 50 else "Safe"

    # Anomaly detection
    anomaly = iso_model.predict(sample_scaled)[0]

    anomaly_status = (
        "Suspicious Water Pattern"
        if anomaly == -1
        else "Normal"
    )

    return {
        "contamination_probability": round(contamination_probability, 2),
        "water_status": status,
        "anomaly_status": anomaly_status
    }

# ----------------------------
# SAMPLE INPUT
# ----------------------------

sample_data = [
    6.8,
    180,
    15000,
    7.5,
    320,
    420,
    12,
    65,
    4.5
]

result = predict_water_quality(sample_data)

print("\nPrediction Result:")
print(result)