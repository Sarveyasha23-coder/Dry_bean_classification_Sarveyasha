import streamlit as st
import pickle
import numpy as np

import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🌱 Dry Bean Classification App")

features = []

for i in range(16):
    val = st.number_input(f"Feature {i+1}")
    features.append(val)

if st.button("Predict"):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    
    prediction = model.predict(features)
    
    st.success(f"Predicted Class: {prediction}")
