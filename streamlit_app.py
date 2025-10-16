import streamlit as st
import numpy as np
import joblib
import os

st.title("Credit Card Fraud Detection App")

model_path = os.path.join('..','models','creditcard_fraud_model.joblib')
scaler_path = os.path.join('..','models','scaler_time_amount.joblib')
model = None
scaler = None
try:
    model = joblib.load(model_path)
except Exception as e:
    st.warning(f"Model not found at {model_path}. Please place your trained model there. Error: {e}")
try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.warning(f"Scaler not found at {scaler_path}. Please place your scaler there. Error: {e}")

time = st.number_input("Time (seconds)", value=0.0)
amount = st.number_input("Transaction Amount", value=0.0)

v_features = []
for i in range(1,29):
    v_features.append(st.number_input(f"V{i}", value=0.0))

features = [time] + v_features + [amount]

if st.button("Predict"):
    if model is None or scaler is None:
        st.error("Model or scaler missing. Put files into models/ and restart app.")
    else:
        input_array = np.array(features).reshape(1,-1)
        input_array[:, [0, -1]] = scaler.transform(input_array[:, [0, -1]])
        pred = model.predict(input_array)[0]
        if pred == 1:
            st.error("Fraudulent Transaction Detected!")
        else:
            st.success("Transaction is Legitimate")
