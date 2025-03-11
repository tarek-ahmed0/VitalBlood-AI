import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Apply Styles
st.markdown(
    """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

        .header-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.5rem;
            font-weight: bold;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 1.1rem;
            color: #c40233;
        }
        .divider {
            border-top: 1px solid #170225;
            margin: 20px 0;
        }
        .result-box {
            font-family: 'Poppins', sans-serif;
            font-size: 1.2rem;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .has-anemia {
            background-color: #ffcccc;
            color: #c40233;
            border: 2px solid #c40233;
        }
        .no-anemia {
            background-color: #ccffcc;
            color: #006600;
            border: 2px solid #006600;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown(
    """
    <div class="header-title">VitalBlood AI. 🩸</div>
    <div class="header-subtitle">Smart anemia detection through advanced blood analysis.</div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True
)

# Load Model & Scaler with Error Handling
try:
    with open("anemia_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 Model file 'anemia_model.pkl' not found! Please upload the correct model file.")
    st.stop()

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 Scaler file 'scaler.pkl' not found! Please upload the correct scaler file.")
    st.stop()

# Define Features
feature_names = ["HB", "RBC", "PCV", "MCH", "MCHC"]
inputs = []

# Collect User Input
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.2f")
    inputs.append(value)

# Convert Inputs to NumPy Array
input_array = np.array(inputs).reshape(1, -1)

# Check for Missing Values
if np.isnan(input_array).any():
    st.warning("⚠️ Please fill in all values before predicting.")
else:
    # Scale the input
    try:
        input_scaled = scaler.transform(input_array)
    except ValueError as e:
        st.error(f"⚠️ Input scaling error: {e}")
        st.stop()

    # Prediction Button
    if st.button("🔍 Predict Anemia"):
        try:
            prediction = model.predict(input_scaled)
            result_html = (
                '<div class="result-box no-anemia">✅ No Anemia: No signs of anemia detected. Stay healthy! </div>'
                if prediction[0] == 0
                else '<div class="result-box has-anemia">🚨 Has Anemia: High likelihood of anemia detected. Please consult a doctor. </div>'
            )
            st.markdown(result_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"🚨 Prediction error: {e}")
