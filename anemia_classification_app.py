import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        .column-label {
            font-family: 'Poppins', sans-serif;
            font-weight: bold;
            font-size: 1.1rem;
            color: #6C63FF;
            margin-bottom: 10px;
        }
        .column-container {
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header-title">
        VitalBlood 🩸
    </div>
    <div class="header-subtitle">
        Smart anemia detection through advanced blood analysis.
    </div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True
)

try:
    with open("anemia_model.pkl", "rb") as f:
        model = pickle.load(f, encoding="latin1")
    if hasattr(model, 'monotonic_cst'):
        del model.monotonic_cst
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# Feature descriptions and normal ranges
feature_info = {
    "HB": {"desc": "Hemoglobin carries oxygen in the blood.", "range": "(Men: 13.8-17.2 g/dL, Women: 12.1-15.1 g/dL)"},
    "RBC": {"desc": "Red blood cells transport oxygen throughout the body.", "range": "(Men: 4.7-6.1 million/μL, Women: 4.2-5.4 million/μL)"},
    "PCV": {"desc": "Packed cell volume measures the proportion of red blood cells.", "range": "(Men: 40.7-50.3%, Women: 36.1-44.3%)"},
    "MCH": {"desc": "Mean corpuscular hemoglobin is the average amount of hemoglobin per red blood cell.", "range": "(27-33 pg/cell)"},
    "MCHC": {"desc": "Mean corpuscular hemoglobin concentration indicates hemoglobin concentration per red cell volume.", "range": "(32-36 g/dL)"}
}

inputs = []

for feature, details in feature_info.items():
    st.markdown(f"### {feature}")
    st.markdown(f"{details['desc']}")
    st.markdown(f"**Normal Range:** {details['range']}")
    value = st.number_input(f"Enter your {feature}:", min_value=0.0, format="%.2f")
    inputs.append(value)

input_array = np.array(inputs).reshape(1, -1)

if np.any(np.isnan(input_array)):
    st.warning("Please enter valid values for all fields.")
else:
    input_scaled = scaler.transform(input_array)

    has_anemia = """🚨 **Has Anemia**  
    _High likelihood of anemia. Please consult a doctor._"""
    no_anemia = """🩺 **No Anemia**  
    _No signs of anemia detected. Stay healthy!_"""

    if st.button("Predict Anemia"):
        try:
            prediction = model.predict(input_scaled)
            result = no_anemia if prediction[0] == 0 else has_anemia
            st.markdown(result, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
