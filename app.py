import streamlit as st
import numpy as np
import joblib

# -----------------------
# LOAD MODELS
# -----------------------
rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------
# TITLE
# -----------------------
st.title("💧 AquaShield AI")
st.subheader("Smart Water Quality Prediction System")

st.markdown("Enter water parameters to predict contamination risk")

# -----------------------
# INPUT SLIDERS
# -----------------------

ph = st.slider("pH Level", 0.0, 14.0, 7.0)
hardness = st.slider("Hardness", 0.0, 400.0, 150.0)
solids = st.slider("Total Dissolved Solids (TDS)", 0.0, 50000.0, 15000.0)
chloramines = st.slider("Chloramines", 0.0, 15.0, 7.0)
sulfate = st.slider("Sulfate", 0.0, 500.0, 300.0)
conductivity = st.slider("Conductivity", 0.0, 1000.0, 400.0)
organic_carbon = st.slider("Organic Carbon", 0.0, 30.0, 10.0)
trihalomethanes = st.slider("Trihalomethanes", 0.0, 120.0, 60.0)
turbidity = st.slider("Turbidity", 0.0, 10.0, 4.0)

# -----------------------
# PREDICT BUTTON
# -----------------------
if st.button("🔍 Predict Water Quality"):

    input_data = np.array([[ph, hardness, solids, chloramines,
                            sulfate, conductivity,
                            organic_carbon, trihalomethanes,
                            turbidity]])

    # scale
    scaled = scaler.transform(input_data)

    # -----------------------
    # MODEL PREDICTIONS
    # -----------------------

    rf_prob = rf_model.predict_proba(scaled)[0][1]
    xgb_prob = xgb_model.predict_proba(scaled)[0][1]

    contamination_prob = (rf_prob + xgb_prob) / 2

    # anomaly detection
    anomaly = iso_model.predict(scaled)[0]
    anomaly_status = (
        "⚠ Suspicious Pattern"
        if anomaly == -1
        else "Normal"
    )

    # -----------------------
    # RULE-BASED OVERRIDES
    # -----------------------

    rule_violations = []

    if ph < 6.5 or ph > 8.5:
        rule_violations.append("Unsafe pH level")

    if turbidity > 5:
        rule_violations.append("High turbidity")

    if chloramines > 4:
        rule_violations.append("High chloramines")

    if sulfate > 250:
        rule_violations.append("High sulfate concentration")

    if solids > 19000:
        rule_violations.append("High Total Dissolved Solids (TDS)")

    # -----------------------
    # FINAL STATUS
    # -----------------------

    if len(rule_violations) > 0:
        status = "❌ Unsafe Water"

        # force minimum risk if rules fail
        contamination_prob = max(contamination_prob, 0.85)

    else:
        status = (
            "❌ Unsafe Water"
            if contamination_prob > 0.5
            else "✅ Safe Water"
        )

    # -----------------------
    # OUTPUT
    # -----------------------

    st.markdown("## 🧠 AI Prediction Result")

    st.write(f"### Status: {status}")

    st.write(
        f"### Contamination Probability: "
        f"{round(contamination_prob * 100, 2)}%"
    )

    st.write(f"### Anomaly Detection: {anomaly_status}")

    # progress bar
    st.progress(float(contamination_prob))

    # -----------------------
    # WARNINGS
    # -----------------------

    if rule_violations:

        st.error("🚨 Water Safety Rule Violations Detected")

        for violation in rule_violations:
            st.write(f"• {violation}")

    elif contamination_prob > 0.7:
        st.error("🚨 High Risk Water Detected!")

    elif contamination_prob > 0.4:
        st.warning("⚠ Moderate Risk Water")

    else:
        st.success("💧 Safe Water Quality")