import os
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_machine_failure_model_v1.joblib"
)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found at: {model_path}"
    )

model = joblib.load(model_path)

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Machine Failure Prediction")
st.write(
    """
    Enter the machine operating parameters below to predict
    whether the machine is likely to fail.
    """
)

# -----------------------------
# User Inputs
# -----------------------------
machine_type = st.selectbox(
    "Machine Type",
    ["H", "L", "M"]
)

air_temp = st.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=298.0,
    step=0.1
)

process_temp = st.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=500.0,
    value=308.0,
    step=0.1
)

rot_speed = st.number_input(
    "Rotational Speed (rpm)",
    min_value=0,
    max_value=3000,
    value=1500,
    step=1
)

torque = st.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.1
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=300,
    value=10,
    step=1
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Machine Failure"):

    input_df = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    prediction = model.predict(input_df)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠ Machine Failure Predicted")
    else:
        st.success("✅ No Machine Failure Predicted")

    if probability is not None:
        st.write(f"**Failure Probability:** {probability:.2%}")
