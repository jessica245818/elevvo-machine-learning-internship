"""Simple JSON/CSV frontend for the predictive-maintenance API."""

from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUIRED_COLUMNS = [
    "product_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

st.set_page_config(page_title="Predictive Maintenance", page_icon="🏭")
st.title("🏭 Predictive Maintenance")
st.caption("Task 10 · FastAPI + Task 9 low-false-alarm model")

tab_form, tab_file = st.tabs(["Single reading", "CSV upload"])

with tab_form:
    with st.form("sensor_form"):
        product_type = st.selectbox("Product type", ["L", "M", "H"])
        air = st.number_input("Air temperature (K)", 250.0, 350.0, 300.0)
        process = st.number_input(
            "Process temperature (K)", 250.0, 400.0, 310.0
        )
        speed = st.number_input(
            "Rotational speed (rpm)", 1.0, 10_000.0, 1_500.0
        )
        torque = st.number_input("Torque (Nm)", 0.0, 500.0, 40.0)
        wear = st.number_input("Tool wear (min)", 0.0, 1_000.0, 100.0)
        submitted = st.form_submit_button("Predict")
    if submitted:
        payload = {
            "product_type": product_type,
            "air_temperature_k": air,
            "process_temperature_k": process,
            "rotational_speed_rpm": speed,
            "torque_nm": torque,
            "tool_wear_min": wear,
        }
        try:
            response = requests.post(
                f"{API_URL}/predict", json=payload, timeout=15
            )
            response.raise_for_status()
            result = response.json()
            if result["machine_failure_alarm"]:
                st.error(
                    f"Maintenance required · probability "
                    f"{result['failure_probability']:.1%}"
                )
            else:
                st.success(
                    f"Normal operation · probability "
                    f"{result['failure_probability']:.1%}"
                )
            st.json(result)
        except requests.RequestException as error:
            st.error(f"API request failed: {error}")

with tab_file:
    uploaded = st.file_uploader("Upload sensor readings", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            st.dataframe(frame.head())
            if st.button("Predict uploaded rows"):
                records = frame[REQUIRED_COLUMNS].to_dict(orient="records")
                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch", json=records, timeout=30
                    )
                    response.raise_for_status()
                    results = pd.DataFrame(response.json())
                    st.dataframe(pd.concat([frame.reset_index(drop=True), results], axis=1))
                except requests.RequestException as error:
                    st.error(f"API request failed: {error}")
