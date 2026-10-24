
import streamlit as st
import pandas as pd
import joblib

st.title("Starbucks Calories Predictor")
st.write("Estimate calories using nutrition information.")

saved = joblib.load("starbucks_model.pkl")
model = saved["model"]
columns = saved["columns"]

st.subheader("Enter drink nutrition details")

values = {}
for col in columns:
    values[col] = st.number_input(col, min_value=0.0, value=0.0)

if st.button("Predict Calories"):
    data = pd.DataFrame([values], columns=columns)
    prediction = model.predict(data)[0]
    st.success(f"Estimated Calories: {prediction:.0f}")
