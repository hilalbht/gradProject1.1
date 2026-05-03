import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────
# MODELİ YÜKLE
# ─────────────────────────────
model = joblib.load("model.pkl")

st.title("🧬 Liver Cirrhosis Stage Prediction (TEST)")

st.write("Basit test arayüzü — model çalışıyor mu kontrol et")

# ─────────────────────────────
# GİRİŞ ALANLARI (SADE TUTTUM)
# ─────────────────────────────

age = st.number_input("Age (years)", 0, 100, 50)
bilirubin = st.number_input("Bilirubin", 0.0, 50.0, 1.0)
alk_phos = st.number_input("Alk Phos", 0.0, 2000.0, 100.0)
sgot = st.number_input("SGOT", 0.0, 2000.0, 100.0)
albumin = st.number_input("Albumin", 0.0, 10.0, 3.5)

sex = st.selectbox("Sex", ["M", "F"])

# ─────────────────────────────
# DATAFRAME OLUŞTUR
# ─────────────────────────────

input_data = pd.DataFrame([{
    "Age": age,
    "Bilirubin": bilirubin,
    "Alk_Phos": alk_phos,
    "SGOT": sgot,
    "Albumin": albumin,
    "Sex": sex
}])

# ─────────────────────────────
# PREDICT
# ─────────────────────────────

if st.button("Predict Stage"):
    try:
        prediction = model.predict(input_data)[0]

        stage_map = {
            0: "Stage 1",
            1: "Stage 2",
            2: "Stage 3"
        }

        st.success(f"Prediction: {stage_map[prediction]}")

    except Exception as e:
        st.error(f"Hata oluştu: {e}")