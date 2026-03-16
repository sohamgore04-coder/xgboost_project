import streamlit as st
import joblib
import pandas as pd
import os

# Load model safely
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "xgboost_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction App")

st.write(
"""
Enter the details of the house below and the model will estimate the **house price**.
This prediction is powered by a Machine Learning model built using XGBoost.
"""
)

st.divider()

# ----------- User Inputs -----------

area = st.number_input("Area (Square Feet)", min_value=300, max_value=10000, value=1200)

bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)

bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)

floors = st.slider("Number of Floors", 1, 3, 1)

age = st.slider("Age of the House (years)", 0, 50, 5)

location_score = st.slider(
    "Location Score (1 = poor location, 10 = prime location)", 
    1, 
    10, 
    5
)

st.divider()

# ----------- Prediction Button -----------

if st.button("Predict House Price"):

    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, floors, age, location_score]],
        columns=[
            "area",
            "bedrooms",
            "bathrooms",
            "floors",
            "age",
            "location_score"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ₹ {round(prediction,2):,}")
