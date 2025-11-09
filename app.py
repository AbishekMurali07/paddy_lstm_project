"""
app.py
Main Streamlit app for Crop Yield Prediction using LSTM (Tamil Nadu - Paddy)
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.model import train_lstm_model
from src.dashboard import show_dashboard
from src.utils import preprocess_input
import os

# -------------------- App Configuration --------------------
st.set_page_config(
    page_title="🌾 Tamil Nadu Paddy Yield Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌾 Tamil Nadu Paddy Yield Prediction (LSTM Based)")
st.markdown("### A user-friendly dashboard with live prediction and interactive visualization")

# -------------------- Load or Train Model --------------------
DATA_PATH = "data/final_tn_dataset.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found! Please run `download_datasets.py` first to generate data.")
    st.stop()

@st.cache_resource
def load_model_and_data():
    model, scaler_X, scaler_y, X_test, y_test, df = train_lstm_model(DATA_PATH)
    return model, scaler_X, scaler_y, df

with st.spinner("🧠 Training LSTM model... please wait (takes ~1 min)"):
    model, scaler_X, scaler_y, df = load_model_and_data()

st.success("✅ Model trained successfully!")

# -------------------- Tabs --------------------
tab1, tab2 = st.tabs(["📈 Dashboard", "🔮 Predict Yield"])

# -------------------- Dashboard Tab --------------------
with tab1:
    show_dashboard(df)

# -------------------- Prediction Tab --------------------
with tab2:
    st.subheader("🔮 Predict Paddy Yield for a Given Set of Conditions")

    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.number_input("Cultivated Area (hectares)", 10000.0, 80000.0, 40000.0, 500.0)
        rainfall = st.slider("Annual Rainfall (mm)", 600.0, 1600.0, 1000.0)
        temp = st.slider("Average Temperature (°C)", 25.0, 35.0, 30.0)
    with col2:
        production = st.number_input("Production (tonnes)", 20000.0, 300000.0, 150000.0, 1000.0)
        nitrogen = st.slider("Nitrogen (ratio)", 0.1, 1.0, 0.4)
        phosphorus = st.slider("Phosphorus (ratio)", 0.1, 1.0, 0.3)
    with col3:
        potassium = st.slider("Potassium (ratio)", 0.1, 1.0, 0.3)
        district = st.selectbox("District", sorted(df["District"].unique()))

    if st.button("🚀 Predict Yield"):
        input_data = preprocess_input(area, production, rainfall, temp, nitrogen, phosphorus, potassium)
        scaled_input = scaler_X.transform(input_data)
        scaled_input = np.reshape(scaled_input, (scaled_input.shape[0], 1, scaled_input.shape[1]))
        prediction_scaled = model.predict(scaled_input)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        st.success(f"🌾 **Predicted Yield for {district}: {prediction[0][0]:.2f} tonnes/hectare**")

    st.markdown("---")
    st.info("ℹ️ Prediction is based on synthetic data for demonstration purposes.")
