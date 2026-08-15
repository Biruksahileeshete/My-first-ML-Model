# ========================================
# HOUSE PRICE PREDICTION APP
# Streamlit Web Application
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# HEADER
# ========================================

st.title("🏠 House Price Prediction App")
st.markdown("Predict house prices using machine learning!")

# ========================================
# SIDEBAR - USER INPUT
# ========================================

st.sidebar.header("🔧 Model Settings")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Linear Regression"]
)

# Model parameters
st.sidebar.subheader("Model Parameters")

if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, 10)
    max_depth = st.sidebar.slider("Max Depth", 2, 20, 10, 1)
else:
    # Linear Regression has no parameters
    st.sidebar.info("Linear Regression uses default parameters")

# Dataset size
dataset_size = st.sidebar.slider("Dataset Size", 100, 1000, 500, 100)
# ========================================
# MAIN AREA - INPUTS
# ========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏡 House Features")
    size = st.number_input("Size (sq ft)", 500, 5000, 2000, 100)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    age = st.slider("Age (years)", 0, 50, 10)

with col2:
    st.subheader("📍 Location")
    location_score = st.slider("Location Score", 1.0, 10.0, 7.0, 0.5)
    proximity = st.selectbox("Proximity to City", ["Downtown", "Suburb", "Rural"])
    
    # Convert proximity to score
    proximity_scores = {"Downtown": 9.0, "Suburb": 6.0, "Rural": 3.0}
    proximity_score = proximity_scores[proximity]

with col3:
    st.subheader("🔧 Additional Features")
    garage = st.checkbox("Has Garage")
    pool = st.checkbox("Has Pool")
    renovated = st.checkbox("Recently Renovated")
    
    # Add bonuses
    feature_bonus = (garage * 15000) + (pool * 25000) + (renovated * 20000)