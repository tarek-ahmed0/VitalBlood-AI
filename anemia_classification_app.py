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
            background-color: rgba(23, 2, 37);
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
            color: #6C63FF;
            margin-bottom: 10px;
        }
        .column-label2 {
            font-family: 'Poppins', sans-serif;
            font-weight: bold;
            font-size: 1.1rem;
            color: #ffffff;
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


feature_names = ["HB", "RBC", "PCV", "MCH", "MCHC"]
inputs = []

for feature in feature_names:
    value = st.number_input(f":green[{feature}]", min_value=0.0, format="%.2f")
    inputs.append(value)

input_array = np.array(inputs).reshape(1, -1)


if np.any(np.isnan(input_array)):
    st.warning("Please enter valid values for all fields.")

else:
    input_scaled = scaler.transform(input_array)

    has_anemia = """:red[Has Anemia] 🚨
    :gray[High likelihood of anemia. Please consult a doctor.]"""
    no_anemia = """No Anemia 🧬
    :gray[No signs of anemia detected. Stay healthy!]"""

    if st.button("Predict Anemia"):
        try:
            prediction = model.predict(input_scaled)
            result = no_anemia if prediction[0] == 0 else has_anemia
            st.markdown(result, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
