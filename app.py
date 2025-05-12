import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="CKD Prediction", layout="wide")
st.image("kidney.jpeg", width=100)

# Load the trained model and median values
model = pickle.load(open("xgb_ckd_model", "rb"))
median_values = pickle.load(open("median_values.pkl", "rb"))

# Define top features based on importance
selected_features = ['hemo', 'dm', 'sg', 'sc', 'htn', 'al', 'rc', 'bgr', 'sod', 'age']

# Feature descriptions (readable labels & tooltips)
feature_descriptions = {
    "age": ("Age", "Age of the patient (in years)"),
    "su": ("Sugar", "Urine sugar content"),
    "rbc": ("Red Blood Cells", "Red blood cell count"),
    "pc": ("Pus Cells", "Presence of pus cells"),
    "al": ("Albumin", "Albumin level in urine"),
    "pcc": ("Pus Cell Clumps", "Clumped pus cells presence"),
    "bgr": ("Blood Glucose Random", "Random blood glucose level"),
    "bu": ("Blood Urea", "Amount of urea in blood"),
    "sc": ("Serum Creatinine", "Kidney filtration indicator"),
    "sod": ("Sodium", "Sodium level in blood"),
    "pot": ("Potassium", "Potassium level in blood"),
    "hemo": ("Hemoglobin", "Hemoglobin concentration"),
    "wc": ("White Blood Cell Count", "WBC count per mm3"),
    "rc": ("Red Blood Cell Count", "RBC count per mm3"),
    "htn": ("Hypertension", "High blood pressure history (1/0)"),
    "dm": ("Diabetes Mellitus", "Diabetes history (1/0)"),
    "cad": ("Coronary Artery Disease", "CAD history (1/0)"),
    "appet": ("Appetite", "Normal or poor (1/0)"),
    "pe": ("Pedal Edema", "Swelling in legs (1/0)"),
    "ane": ("Anemia", "Anemic condition (1/0)")
}

# Streamlit App Layout
st.title("🩺 Chronic Kidney Disease (CKD) Predictor")
st.markdown("Enter your details below to check your CKD risk.")

# Input section
st.subheader("🔢 Patient Data Entry")
cols = st.columns(4)
input_data = {}

for i, feature in enumerate(selected_features):
    col = cols[i % 4]
    label, tooltip = feature_descriptions.get(feature, (feature, ""))
    val = col.number_input(f"{label}", help=tooltip, step=0.1, format="%.2f")
    input_data[feature] = val

# Predict button
if st.button("🔍 Predict"):
    # Ensure all expected features are present in the same order
    full_features = model.feature_names_in_  # automatically gets expected feature names

    # Fill missing features with median values
    for feat in full_features:
        if feat not in input_data:
            input_data[feat] = median_values[feat]

    # Reorder columns to match training order
    input_df = pd.DataFrame([input_data])[full_features]

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # Show result
    if prediction == 1:
        if prediction_proba >= 0.85:
            st.error(f"🛑 **High Risk of CKD!** (Confidence: {prediction_proba:.2%})")
        else:
            st.warning(f"⚠️ **Possible CKD.** (Confidence: {prediction_proba:.2%})")
    else:
        st.success(f"✅ **No CKD Detected.** (Confidence: {prediction_proba:.2%})")

# About section
st.markdown("---")
with st.expander("ℹ️ About this app"):
    st.markdown("""
    - **Model**: Trained using XGBoost on real patient data  
    - **Accuracy**: 98.75%  
    - **Input Fields**: Only top 10 most important features  
    - **Developer**: MD Inam and Team - AI & Data Science  
    - **Purpose**: Help early detection and awareness of CKD  
    """)
