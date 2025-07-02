# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Risk Predictor")
st.markdown("Use this intelligent tool to estimate the risk of heart disease based on health data.")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open("heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

st.header("📝 Enter Patient Details")

# Input layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 29, 77, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal (2)", "Asymptomatic (3)"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dL)", 100, 600, 230)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])

with col2:
    restecg = st.selectbox("Resting ECG", ["Normal (0)", "ST-T Abnormality (1)", "Left Ventricular Hypertrophy (2)"])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.radio("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST Segment", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
    ca = st.selectbox("Number of Major Vessels Colored (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"])

# Convert input to numeric
input_data = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": int(cp.split("(")[-1][0]),
    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs == "Yes" else 0,
    "restecg": int(restecg.split("(")[-1][0]),
    "thalach": thalach,
    "exang": 1 if exang == "Yes" else 0,
    "oldpeak": oldpeak,
    "slope": int(slope.split("(")[-1][0]),
    "ca": ca,
    "thal": int(thal.split("(")[-1][0])
}

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\n**Probability: {proba:.2f}**")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\n**Probability: {proba:.2f}**")

    st.markdown("---")
    st.caption("Note: This tool is for educational purposes and not a substitute for professional medical advice.")
