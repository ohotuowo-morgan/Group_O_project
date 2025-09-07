import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved model and encoders
model = joblib.load("laptop_price_model.pkl")
encoders = joblib.load("label_encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Laptop Price Prediction App")
st.write("Enter the laptop specifications below to predict its price (in Euros).")

# --- User inputs ---
company = st.selectbox("Company", encoders['company'].classes_)
product = st.selectbox("Product", encoders['product'].classes_)
typename = st.selectbox("Type", encoders['typename'].classes_)
cpu = st.selectbox("CPU", encoders['cpu'].classes_)
ram = st.number_input("RAM (GB)", min_value=2, max_value=64, step=2)
memory = st.number_input("Memory (GB)", min_value=128, step=128)
gpu = st.selectbox("GPU", encoders['gpu'].classes_)
opsys = st.selectbox("Operating System", encoders['opsys'].classes_)
screenresolution = st.selectbox("Screen Resolution", encoders['screenresolution'].classes_)
price_per_inch = 0
memory_per_ram = memory / ram if ram != 0 else 0
is_high_end = 0

inches = st.slider("Screen Size (inches)", 10.0, 20.0, 15.6)
weight = st.number_input("Weight (kg)", min_value=0.8, max_value=5.0, value=2.0, step=0.1)

# --- Prepare input data ---
input_data = pd.DataFrame({
    'laptop_id': 0,
    'company': [encoders['company'].transform([company])[0]],
    'product': [encoders['product'].transform([product])[0]],
    'typename': [encoders['typename'].transform([typename])[0]],
    'inches': [inches],
    'screenresolution': [encoders['screenresolution'].transform([screenresolution])[0]],
    'cpu': [encoders['cpu'].transform([cpu])[0]],
    'ram': [ram],
    'memory': [memory],
    'gpu': [encoders['gpu'].transform([gpu])[0]],
    'opsys': [encoders['opsys'].transform([opsys])[0]],
    'weight': [weight],
    'memory_per_ram': [memory_per_ram],
})

# --- Prediction ---
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Laptop Price: €{prediction:.2f}")
