import streamlit as st
import joblib
import pandas as pd
import os

# Load model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "xgboost_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠")

st.title("🏠 California House Price Prediction")
st.write("Enter the house details to estimate the **median house value**.")

st.markdown("---")

# Input fields
MedInc = st.number_input("Median Income", value=3.0)

HouseAge = st.number_input("House Age", value=20)

AveRooms = st.number_input("Average Rooms", value=5.0)

AveBedrms = st.number_input("Average Bedrooms", value=1.0)

Population = st.number_input("Population", value=1000)

AveOccup = st.number_input("Average Occupancy", value=3.0)

Latitude = st.number_input("Latitude", value=34.0)

Longitude = st.number_input("Longitude", value=-118.0)

st.markdown("---")

# Prediction
if st.button("Predict Price"):

    input_data = pd.DataFrame(
        [[
            MedInc,
            HouseAge,
            AveRooms,
            AveBedrms,
            Population,
            AveOccup,
            Latitude,
            Longitude
        ]],
        columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ${round(prediction * 100000,2):,}")
