
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Landslide Susceptibility System",
    page_icon="⛰️",
    layout="wide"
)

# Load trained model and evaluation results
model = joblib.load("landslide_model.pkl")
results = joblib.load("evaluation_results.pkl")

# Title
st.title("⛰️ AI-Based Landslide Susceptibility Prediction")
st.write(
    "A Machine Learning dashboard that estimates "
    "landslide susceptibility using environmental features."
)

st.warning(
    "Educational demonstration only. This model estimates "
    "landslide susceptibility; it does not predict when a "
    "landslide will occur or provide emergency warnings."
)

st.divider()

# Input section
st.header("Enter Environmental Conditions")

st.sidebar.header("Environmental Inputs")

# Standardized environmental features
aspect = st.sidebar.slider(
    "Aspect (standardized)",
    -1.80, 1.83, 0.0, 0.01
)

river = st.sidebar.slider(
    "Distance from River (standardized)",
    -0.13, 8.04, 0.0, 0.01
)

elevation = st.sidebar.slider(
    "Elevation (standardized)",
    -1.21, 3.68, 0.0, 0.01
)

flow = st.sidebar.slider(
    "Flow Accumulation (standardized)",
    -0.58, 5.01, 0.0, 0.01
)

ndvi = st.sidebar.slider(
    "Vegetation Index (standardized)",
    -5.97, 1.79, 0.0, 0.01
)

rainfall = st.sidebar.slider(
    "Rainfall (standardized)",
    -2.33, 3.92, 0.0, 0.01
)

slope = st.sidebar.slider(
    "Slope (standardized)",
    -1.99, 3.73, 0.0, 0.01
)

# Land-use selection
land_cover_options = {
    "Tree cover": "LULC_10",
    "Shrubland": "LULC_20",
    "Grassland": "LULC_30",
    "Cropland": "LULC_40",
    "Built-up": "LULC_50",
    "Bare / sparse vegetation": "LULC_60",
    "Snow and ice": "LULC_70",
    "Permanent water bodies": "LULC_80",
    "Herbaceous wetland": "LULC_90",
    "Moss and lichen": "LULC_100"
}

land_cover = st.sidebar.selectbox(
    "Land Cover Type",
    list(land_cover_options.keys())
)

# Create input dictionary
input_values = {
    "Aspect_z": aspect,
    "Dist_River_z": river,
    "Elevation_z": elevation,
    "FlowAccumulation_z": flow,
    "NDVI_z": ndvi,
    "Rainfall_z": rainfall,
    "Slope_z": slope
}

# Add one-hot encoded land-cover features
selected_column = land_cover_options[land_cover]

for column in [
    "LULC_10", "LULC_20", "LULC_30",
    "LULC_40", "LULC_50", "LULC_60",
    "LULC_70", "LULC_80", "LULC_90",
    "LULC_100"
]:
    input_values[column] = 1 if column == selected_column else 0

# Arrange inputs in the exact training order
input_data = pd.DataFrame(
    [input_values],
    columns=results["feature_names"]
)

# Prediction
if st.button("Predict Landslide Susceptibility"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    classes = list(model.classes_)
    landslide_index = classes.index(1)
    landslide_probability = probabilities[landslide_index] * 100

    st.divider()
    st.header("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Landslide-susceptible")
    else:
        st.success("Prediction: Non-landslide class")

    st.metric(
        "Model-estimated landslide class probability",
        f"{landslide_probability:.2f}%"
    )

    st.progress(float(landslide_probability / 100))

    st.caption(
        "This probability is a model output, not a calibrated "
        "real-world probability or an emergency warning."
    )

# Model performance
st.divider()
st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Test Accuracy",
        f"{results['accuracy'] * 100:.2f}%"
    )

with col2:
    st.metric(
        "Training Algorithm",
        "Random Forest"
    )

# Feature importance
st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": results["feature_names"],
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)

# Confusion matrix
st.subheader("Confusion Matrix")

confusion = pd.DataFrame(
    results["confusion_matrix"],
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

st.dataframe(confusion)

# Classification report
st.subheader("Classification Report")

report = pd.DataFrame(results["report"]).transpose()
st.dataframe(report)

st.divider()

st.caption(
    "Developed as a Machine Learning mini project."
)
