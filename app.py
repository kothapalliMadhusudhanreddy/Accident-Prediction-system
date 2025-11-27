# app.py — Sleek Accident Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os

# -------------------------------
# Paths
folder_path = r"C:\Users\madhu\Downloads\model23"
model_path = os.path.join(folder_path, "model.joblib")
scaler_path = os.path.join(folder_path, "scaler.joblib")

# Load model & scaler
if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
    st.error(f"model.joblib or scaler.joblib not found in: {folder_path}")
    st.stop()

model = load(model_path)
scaler = load(scaler_path)

# -------------------------------
# Page config
st.set_page_config(page_title="Accident Prediction Dashboard",
                   page_icon="🚦",
                   layout="wide")

# -------------------------------
# Sidebar Inputs
st.sidebar.header("🚗 Traffic & Road Inputs")
lane_count = st.sidebar.number_input("Lane Count", 1, 6, 2)
speed_limit_kmph = st.sidebar.number_input("Speed Limit (kmph)", 0, 120, 60)
blackspot_score = st.sidebar.number_input("Blackspot Score", 0, 100, 15)
vehicle_count_per_hr = st.sidebar.number_input("Vehicle Count per Hour", 0, 5000, 300)
avg_speed_kmph = st.sidebar.slider("Average Speed (kmph)", 0, 200, 55)
hour_of_day = st.sidebar.number_input("Hour of Day (0–23)", 0, 23, 14)
is_peak_hour = st.sidebar.checkbox("Peak Hour?", True)
is_night = st.sidebar.checkbox("Night?", False)
weather_clear = st.sidebar.checkbox("Weather Clear?", True)
night_rain = st.sidebar.checkbox("Night & Rain?", False)

# -------------------------------
# Prepare Data
user_input = {
    'lane_count': lane_count,
    'speed_limit_kmph': speed_limit_kmph,
    'blackspot_score': blackspot_score,
    'vehicle_count_per_hr': vehicle_count_per_hr,
    'avg_speed_kmph': avg_speed_kmph,
    'hour_of_day': hour_of_day,
    'is_peak_hour': int(is_peak_hour),
    'is_night': int(is_night),
    'weather_clear': int(weather_clear),
    'night_rain': int(night_rain)
}

user_df = pd.DataFrame([user_input])

# Auto-fill missing features
expected_features = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else list(user_df.columns)
defaults = {
    'cycle_time_s': 60, 'day_of_week': 2, 'enforcement_level': 5, 'green_duration_s': 30,
    'red_duration_s': 30, 'yellow_duration_s': 5, 'has_signal': 1, 'road_highway': 0,
    'weather_rainy': 0, 'weather_foggy': 0, 'peak_highway': 0, 'night_fog': 0,
    'high_speed': 0, 'traffic_heavy': 0, 'is_weekend': 0
}

full_input = {col: user_df[col][0] if col in user_df.columns else defaults.get(col, 0)
              for col in expected_features}
full_df = pd.DataFrame([full_input], columns=expected_features)
full_df = full_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# -------------------------------
# Main Dashboard
st.markdown("<h1 style='text-align:center; color:#0f4c81;'>🚦 Accident Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Fast & Visual Accident Risk Assessment</p>", unsafe_allow_html=True)
st.markdown("---")

if st.button("Predict Accident Risk"):
    try:
        input_scaled = scaler.transform(full_df)
        prob = model.predict_proba(input_scaled)[:, 1]
        pred = (prob >= 0.44).astype(int)
        probability = prob[0] * 100

        # -------------------------------
        # Display Result Cards
        col1, col2 = st.columns(2)
        with col1:
            if pred[0] == 1:
                st.markdown(f"<div style='background-color:#ffcccc; padding:20px; border-radius:10px;'>"
                            f"<h2 style='text-align:center; color:#d60000;'>⚠️ No Accident occured</h2></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color:#ccffcc; padding:20px; border-radius:10px;'>"
                            f"<h2 style='text-align:center; color:#006400;'>✅ Accident occured</h2></div>", unsafe_allow_html=True)

       
        # -------------------------------
        # Expandable details
        with st.expander("Show Input Details"):
            st.dataframe(full_df.T.rename(columns={0:"Value"}))

    except Exception as e:
        st.error("")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: gray;'>Model powered by Gradient Boosting Classifier</p>", unsafe_allow_html=True)
