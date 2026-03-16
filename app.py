import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("xgboost_model.pkl")

st.title("XGBoost Prediction App")

st.write("Enter input values to make prediction")

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predict"):
    
    input_data = np.array([[feature1, feature2, feature3]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Prediction: {prediction[0]}")
