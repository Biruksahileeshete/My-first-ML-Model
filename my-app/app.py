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
    # ========================================
# GENERATE AND TRAIN MODEL
# ========================================

@st.cache_data
def generate_dataset(n_samples=500):
    """Generate synthetic housing dataset"""
    np.random.seed(42)
    
    data = {
        'Size': np.random.randint(500, 5000, n_samples),
        'Bedrooms': np.random.randint(1, 6, n_samples),
        'Age': np.random.randint(0, 50, n_samples),
        'Location_Score': np.random.uniform(1, 10, n_samples),
        'Garage': np.random.choice([0, 1], n_samples),
        'Pool': np.random.choice([0, 1], n_samples),
        'Renovated': np.random.choice([0, 1], n_samples),
        'Price': np.zeros(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate price
    for i in range(n_samples):
        price = (
            50000 +
            df.loc[i, 'Size'] * 150 +
            df.loc[i, 'Bedrooms'] * 30000 -
            df.loc[i, 'Age'] * 500 +
            df.loc[i, 'Location_Score'] * 10000 +
            df.loc[i, 'Garage'] * 15000 +
            df.loc[i, 'Pool'] * 25000 +
            df.loc[i, 'Renovated'] * 20000 +
            np.random.normal(0, 20000)
        )
        df.loc[i, 'Price'] = max(50000, min(800000, price))
    
    return df

@st.cache_data
def train_model(X, y, model_type, params):
    """Train the selected model"""
    if model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            random_state=42
        )
    else:
        model = LinearRegression()
    
    model.fit(X, y)
    return model