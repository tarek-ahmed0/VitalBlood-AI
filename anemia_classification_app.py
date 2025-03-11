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
            color: #ffffff; /* White */
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 1.1rem;
            color: #c40233; /* Light Violet */
        }
        .icon {
            width: 50px;
            height: 50px;
        }
        .icon-small {
            width: 42px;
            height: 42px;
        }
        .divider {
            border-top: 1px solid #170225; /* Violet */
            margin: 20px 0;
        }
        .solid-border {
            border: 3px solid rgba(30, 10, 50);
            border-radius: 3px;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            background-color: rgba(23, 2, 37); /* Optional for better visibility */
            width: 100%;
            height: 100%;
            box-sizing: border-box;
        }
        .animated-border {
            background-color: rgba(255, 255, 255, 0);
            padding: 20px;
            border-radius: 3px;
            border: 3px solid transparent;
            border-image-slice: 1;
            animation: gradient-border 3s infinite;
            text-align: center;
            box-sizing: border-box;
            width: 100%;
            height: 100%;
        }
        @keyframes gradient-border {
            0% {
                border-image-source: linear-gradient(90deg, #ff00ff, #00ffff);
            }
            50% {
                border-image-source: linear-gradient(180deg, #00ffff, #ff00ff);
            }
            100% {
                border-image-source: linear-gradient(270deg, #ff00ff, #00ffff);
            }
        }
        .column-label {
            font-family: 'Poppins', sans-serif;
            font-weight: bold;
            font-size: 1.1rem;
            color: #6C63FF; /* Violet */
            margin-bottom: 10px;
        }
        .column-label2 {
            font-family: 'Poppins', sans-serif;
            font-weight: bold;
            font-size: 1.1rem;
            color: #ffffff; /* Violet */
            margin-bottom: 10px;
        }
        .column-container {
            margin-bottom: 20px; /* Add margin to bottom of each column */
        }
        /* Ensure equal padding and size for both columns */
        .col-image {
            padding: 20px;
            width: 100%;
            height: auto;
            object-fit: cover;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header section
st.markdown(
    """
    <div class="header-title">
        VitalBlood AI. 🩸
    </div>
    <div class="header-subtitle">
        Smart anemia detection through advanced blood analysis.
    </div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True
)


with open("anemia_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as scaler:
    scaler = pickle.load(scaler)

feature_names = ["HB", "RBC", "PCV", "MCH", "MCHC"]
inputs = []

for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.2f")
    inputs.append(value)

input_array = np.array(inputs).reshape(1, -1)
input_scaled = scaler.transform(input_array)

has_anemia = '<div class="result-box has-anemia">🚨 Has Anemia: High likelihood of anemia detected. Please consult a doctor. </div>'
no_anemia = '<div class="result-box no-anemia">✅ No Anemia: No signs of anemia detected. Stay healthy! </div>'

if st.button("Predict Anemia"):
    prediction = model.predict(input_scaled)
    result = no_anemia if prediction[0] == 0 else has_anemia
    st.markdown(result, unsafe_allow_html=True)
