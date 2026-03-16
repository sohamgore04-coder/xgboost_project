import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "xgboost_model.pkl")
model = joblib.load(model_path)

# -----------------------------
# Page Settings
# -----------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Title Section
# -----------------------------

st.title("🏠 House Price Prediction Dashboard")

st.write(
"""
This application predicts **house prices using Machine Learning**.  
Enter the property details and the model will estimate the price.
"""
)

st.markdown("---")

# -----------------------------
# Input Layout (Two Columns)
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    area = st.number_input(
        "📐 Area (Square Feet)",
        min_value=300,
        max_value=10000,
        value=1200
    )

    bedrooms = st.number_input(
        "🛏 Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "🚿 Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

with col2:

    floors = st.number_input(
        "🏢 Floors",
        min_value=1,
        max_value=5,
        value=1
    )

    age = st.number_input(
        "📅 House Age (Years)",
        min_value=0,
        max_value=100,
        value=5
    )

    location_score = st.number_input(
        "📍 Location Score (1–10)",
        min_value=1,
        max_value=10,
        value=5
    )

st.markdown("---")

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("🔍 Predict House Price"):

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

    st.success(f"💰 Estimated House Price: ₹ {round(prediction,2):,}")

    st.balloons()

# -----------------------------
# Feature Importance Chart
# -----------------------------

st.markdown("---")
st.subheader("📊 Feature Importance")

try:

    importance = model.feature_importances_

    features = [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "Age",
        "Location Score"
    ]

    fig, ax = plt.subplots()

    ax.barh(features, importance)

    ax.set_xlabel("Importance Score")
    ax.set_title("Which Features Affect Price Most")

    st.pyplot(fig)

except:
    st.info("Feature importance not available for this model.")

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.caption(
"""
Machine Learning Model: XGBoost  
App Framework: Streamlit
"""
)
