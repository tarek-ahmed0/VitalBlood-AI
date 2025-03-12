import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

        .header-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2rem; /* Slightly smaller */
            font-weight: bold;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
            color: #c40233;
        }
        .divider {
            border-top: 1px solid #170225;
            margin: 15px 0;
        }
        .metric-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 3px;
        }
        .metric-desc {
            font-family: 'Poppins', sans-serif;
            font-size: 0.8rem;
            color: #bbbbbb;
            margin-bottom: 2px;
        }
        .metric-range {
            font-family: 'Poppins', sans-serif;
            font-size: 0.7rem;
            color: #999999;
            margin-bottom: 5px;
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
        Smart anemia detection through blood analysis.
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
    "HB": {"desc": "Oxygen-carrying protein.", "range": "(M: 13.8-17.2, W: 12.1-15.1 g/dL)"},
    "RBC": {"desc": "Oxygen transport cells.", "range": "(M: 4.7-6.1, W: 4.2-5.4 M/μL)"},
    "PCV": {"desc": "Blood cell percentage.", "range": "(M: 40.7-50.3%, W: 36.1-44.3%)"},
    "MCH": {"desc": "Hemoglobin per cell.", "range": "(27-33 pg/cell)"},
    "MCHC": {"desc": "Hemoglobin concentration.", "range": "(32-36 g/dL)"}
}

inputs = []

for feature, details in feature_info.items():
    st.markdown(f"<div class='metric-title'>{feature}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-desc'>{details['desc']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-range'>{details['range']}</div>", unsafe_allow_html=True)
    value = st.number_input("", min_value=0.0, format="%.2f", key=feature)  # Removed metric label
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
