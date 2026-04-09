import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# Load model safely (IMPORTANT FIX)
model_path = Path(__file__).parent / "model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# UI
st.title("ML Prediction App")

st.write("Enter values below:")

# CHANGE THESE INPUTS based on your model
f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")

if st.button("Predict"):
    input_data = np.array([[f1, f2, f3]])
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")
