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
# ========================================
# TRAIN BUTTON
# ========================================

if st.button("🔮 Predict Price", type="primary"):
    with st.spinner("Training model..."):
        
        # Generate dataset
        df = generate_dataset(dataset_size)
        
        # Prepare features
        features = ['Size', 'Bedrooms', 'Age', 'Location_Score', 'Garage', 'Pool', 'Renovated']
        X = df[features]
        y = df['Price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        params = {}
        if model_choice == "Random Forest":
            params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        
        model = train_model(X_train, y_train, model_choice, params)
        
        # Make prediction for user input
        user_features = np.array([[
            size, bedrooms, age, location_score, 
            int(garage), int(pool), int(renovated)
        ]])
        
        predicted_price = model.predict(user_features)[0]
        
        # Add feature bonus
        final_price = predicted_price + feature_bonus
         # ========================================
        # DISPLAY RESULTS
        # ========================================
        
        st.success("✅ Prediction Complete!")
        
        # Results columns
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.metric(
                label="🏠 Predicted House Price",
                value=f"${final_price:,.0f}",
                delta=f"${final_price - 300000:,.0f} vs average"
            )
            
            # Price category
            if final_price < 200000:
                st.info("🏡 Affordable House")
            elif final_price < 400000:
                st.success("🏠 Mid-Range House")
            elif final_price < 600000:
                st.warning("🏘️ Upscale House")
            else:
                st.error("🏰 Luxury House")
        
        with col2:
            st.metric("Model", model_choice)
            if model_choice == "Random Forest":
                st.metric("Trees", n_estimators)
                st.metric("Max Depth", max_depth)
        
        with col3:
            # Model performance
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("MAE", f"${mae:,.0f}")
        
        # ========================================
        # VISUALIZATIONS
        # ========================================
        
        st.subheader("📊 Price Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature contribution
            fig, ax = plt.subplots(figsize=(8, 4))
            
            contributions = {
                'Size': size * 150,
                'Bedrooms': bedrooms * 30000,
                'Age': -age * 500,
                'Location': location_score * 10000,
                'Garage': int(garage) * 15000,
                'Pool': int(pool) * 25000,
                'Renovated': int(renovated) * 20000
            }
            
            names = list(contributions.keys())
            values = list(contributions.values())
            
            colors = ['green' if v > 0 else 'red' for v in values]
            ax.barh(names, values, color=colors)
            ax.set_xlabel('Price Contribution ($)')
            ax.set_title('Feature Contributions')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with col2:
            # Actual vs Predicted scatter
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Sample predictions
            sample_indices = np.random.choice(len(X_test), 50, replace=False)
            sample_actual = y_test.iloc[sample_indices]
            sample_pred = y_pred[sample_indices]
            
            ax.scatter(sample_actual, sample_pred, alpha=0.6, color='blue')
            ax.plot([sample_actual.min(), sample_actual.max()], 
                   [sample_actual.min(), sample_actual.max()], 
                   'r--', linewidth=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Price ($)')
            ax.set_ylabel('Predicted Price ($)')
            ax.set_title('Actual vs Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        # ========================================
        # PRICE HISTORY CHART
        # ========================================
        
        st.subheader("📈 Price Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Size vs Price
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Get similar houses
            similar_houses = df[
                (df['Bedrooms'] == bedrooms) &
                (abs(df['Size'] - size) < 500)
            ]
            
            if len(similar_houses) > 0:
                ax.scatter(similar_houses['Size'], similar_houses['Price'], 
                          alpha=0.5, color='blue', label='Similar Houses')
                ax.scatter([size], [final_price], color='red', s=200, 
                          marker='*', label='Your House')
                ax.set_xlabel('Size (sq ft)')
                ax.set_ylabel('Price ($)')
                ax.set_title('Your House vs Similar Properties')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No similar houses found', 
                       ha='center', va='center', transform=ax.transAxes)
            
            st.pyplot(fig)
        
        with col2:
            # Price by Bedrooms
            fig, ax = plt.subplots(figsize=(8, 4))
            
            bedroom_prices = df.groupby('Bedrooms')['Price'].mean()
            ax.bar(bedroom_prices.index, bedroom_prices.values, 
                   color='lightblue', edgecolor='black')
            ax.axhline(y=final_price, color='red', linestyle='--', 
                      linewidth=2, label='Your Price')
            ax.set_xlabel('Number of Bedrooms')
            ax.set_ylabel('Average Price ($)')
            ax.set_title('Average Price by Bedrooms')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        # ========================================
        # FEATURE IMPORTANCE (Random Forest only)
        # ========================================
        
        if model_choice == "Random Forest" and hasattr(model, 'feature_importances_'):
            st.subheader("🔑 Feature Importance")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            importance = model.feature_importances_
            feature_names = ['Size', 'Bedrooms', 'Age', 'Location', 'Garage', 'Pool', 'Renovated']
            
            indices = np.argsort(importance)
            ax.barh([feature_names[i] for i in indices], importance[indices], color='teal')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

# ========================================
# SIDEBAR - INFORMATION
# ========================================

st.sidebar.markdown("---")
st.sidebar.subheader("📋 How to Use")
st.sidebar.markdown("""
1. Enter house features
2. Select model and parameters
3. Click **Predict Price**
4. View results and visualizations
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 About")
st.sidebar.markdown("""
- **Dataset:** Synthetic housing data
- **Models:** Random Forest & Linear Regression
- **Features:** Size, Bedrooms, Age, Location
""")

st.sidebar.markdown("---")
st.sidebar.caption("🏠 House Price Predictor v1.0")

# ========================================
# CUSTOM CSS
# ========================================

st.markdown("""
<style>
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        height: 3em;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)