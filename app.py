import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
data = joblib.load("model.pkl")
model = data["model"]
features = data["features"]

# ─────────────────────────────
# UI CONFIG
# ─────────────────────────────
st.set_page_config(page_title="Liver Cirrhosis AI", layout="wide")

st.title("🧬 Liver Cirrhosis Stage Prediction AI")
st.write("Bitirme Projesi – Machine Learning Classification System")

# ─────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────
st.sidebar.header("Patient Data Input")

input_dict = {}

for col in features:
    if col in ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]:
        input_dict[col] = st.sidebar.selectbox(col, ["Y", "N", "M", "F", "D-penicillamine", "Placebo", "S"])
    else:
        input_dict[col] = st.sidebar.number_input(col, value=0.0)

# ─────────────────────────────
# PREDICT
# ─────────────────────────────
if st.button("🔍 Predict Stage"):

    df = pd.DataFrame([input_dict])

    # CRITICAL FIX
    df = df[features]

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]

    stages = {0:"Stage 1", 1:"Stage 2", 2:"Stage 3"}

    st.success(f"Prediction: {stages[pred]}")

    st.subheader("Prediction Confidence")

    st.write({
        "Stage 1": f"{prob[0]*100:.2f}%",
        "Stage 2": f"{prob[1]*100:.2f}%",
        "Stage 3": f"{prob[2]*100:.2f}%"
    })

    st.progress(float(max(prob)))
