# 🏠 House Price Prediction App

A Streamlit web application that predicts house prices using machine learning models. Built with Python, scikit-learn, and Streamlit.

## 📋 Features

- **Multiple ML Models**: Choose between Random Forest and Linear Regression
- **Interactive Parameters**: Adjust model hyperparameters in real-time
- **House Features Input**: Input size, bedrooms, age, and other property details
- **Location Analysis**: Factor in location score and proximity to city
- **Additional Features**: Toggle garage, pool, and renovation status
- **Visual Analytics**: View predictions and model performance metrics
- **Responsive UI**: Clean, professional interface with sidebar controls

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or navigate to the project directory**
```bash
cd my-app
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

- **streamlit** (1.28.0) - Web app framework
- **pandas** (2.0.3) - Data manipulation
- **numpy** (1.24.3) - Numerical computing
- **scikit-learn** (1.3.0) - Machine learning
- **matplotlib** (3.7.1) - Data visualization
- **seaborn** (0.12.2) - Statistical visualization

## 🎯 Usage

### Run the Application

```bash
streamlit run app.py
```

The app will start on `http://localhost:8501`

### Custom Port

To run on a different port:
```bash
streamlit run app.py --server.port 8502
```

### Input Parameters

**Sidebar Controls:**
- **Model Selection**: Choose between Random Forest or Linear Regression
- **Model Parameters**: 
  - Number of Trees (10-200)
  - Max Depth (2-20)
- **Dataset Size**: Adjust training data size (100-1000 samples)

**House Features:**
- Size (500-5000 sq ft)
- Bedrooms (1-6)
- Age (0-50 years)

**Location:**
- Location Score (1.0-10.0)
- Proximity to City (Downtown/Suburb/Rural)

**Additional Features:**
- Garage (checkbox)
- Pool (checkbox)
- Recently Renovated (checkbox)

**Predict Button:**
Click "🔮 Predict Price" to train the model and get predictions

## 🔧 How It Works

1. **Dataset Generation**: Creates synthetic housing data based on parameters
2. **Feature Engineering**: Combines user inputs with location and amenity bonuses
3. **Model Training**: Trains selected ML model on the dataset
4. **Price Prediction**: Predicts house price based on input features
5. **Performance Metrics**: Displays R² score and Mean Absolute Error (MAE)
6. **Visualization**: Shows price distribution and prediction results

## 📊 Price Calculation Formula

Base price is calculated using:
```
Price = Base(50,000) 
      + Size × 150
      + Bedrooms × 30,000
      - Age × 500
      + Location_Score × 10,000
      + Garage × 15,000
      + Pool × 25,000
      + Renovated × 20,000
      + Random Noise
```

## 📂 Project Structure

```
my-app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## 🎨 UI Components

- **Wide Layout**: Optimized for wider screens
- **Sidebar Navigation**: Organized model and dataset controls
- **Multi-Column Display**: Feature inputs organized by category
- **Real-time Updates**: Interactive widgets for instant feedback
- **Loading Spinner**: User feedback during model training

## 🔮 Future Enhancements

- [ ] Load real housing datasets (CSV/API)
- [ ] Add more ML models (XGBoost, Gradient Boosting)
- [ ] Feature importance visualization
- [ ] Model comparison analytics
- [ ] Data preprocessing tools
- [ ] Export predictions to CSV
- [ ] Historical prediction tracking
- [ ] Model persistence/loading

## 🐛 Troubleshooting

**Issue**: App won't start
- Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Port already in use
- Solution: Use a different port: `streamlit run app.py --server.port 8502`

**Issue**: Slow predictions
- Solution: Reduce dataset size from sidebar, or decrease number of trees in Random Forest

## 📝 Notes

- The synthetic dataset is generated fresh each time for demonstration
- Model training is cached for improved performance
- Random seed is fixed (42) for reproducible results

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

## 📄 License

This project is open source and available for educational purposes.

---

**Happy Predicting! 🎉**
