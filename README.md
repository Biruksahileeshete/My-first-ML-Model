# My-first-ML-Model

A beginner-friendly machine learning collection with regression and classification examples built in Python.

## Project Overview

This repository contains a collection of machine learning projects designed for beginners to learn and practice fundamental ML concepts. Each project demonstrates essential workflows including data generation, preprocessing, model training, evaluation, and prediction using Python and popular ML libraries.

### Included Projects

1. **Classification Basics** - Student Pass/Fail Prediction
   - Uses logistic regression to classify whether a student will pass or fail
   - Demonstrates binary classification fundamentals
   - Includes data exploration and visualization

2. **Scikit-learn** - Student Grade Prediction
   - Compares multiple regression models for predicting student grades
   - Demonstrates model selection and comparison techniques
   - Saves the best-performing model as `best_model.joblib`

3. **Decision Tree Classifier** - Student Grade Classification
   - Uses decision tree algorithms for grade classification
   - Explores tree-based classification methods
   - Educational example of non-linear classification

4. **Random Forest** - Customer Churn Prediction
   - Applies ensemble learning techniques to predict customer churn
   - Demonstrates the power of random forest algorithms
   - Real-world classification problem example

5. **Simple & Multiple Linear Regression** - House Price Prediction
   - Compares simple vs. multiple linear regression approaches
   - Demonstrates feature impact on housing prices
   - Shows regression fundamentals and multivariate analysis

6. **Streamlit App** (`my-app/`)
   - Interactive web application built with Streamlit
   - User-friendly interface for exploring ML models
   - Optional visualization and prediction dashboard

## Folder Structure

```text
My-first-ML-Model/
├── best_model.joblib                              # best trained model from scikit-learn project
├── README.md                                       # main project documentation
├── Classification basics/
│   ├── Readme.md                                  # classification project documentation
│   └── Student_Pass/
│       └── Student_Pass/
│           └── Pass_Fail_Prediction.py           # student pass/fail prediction script
├── Decision Tree Classifier/
│   ├── README.md                                  # decision tree documentation
│   └── Student_Grade.py                          # grade classification using decision trees
├── Random Forest/
│   ├── README.md                                  # random forest documentation
│   └── customer_churn_Prediction.py              # customer churn prediction using random forest
├── Scikit-learn/
│   ├── Readme.md                                  # regression comparison documentation
│   ├── first_model.py                            # model training and comparison
│   └── best_model.joblib                         # saved best model artifact
├── Simple & Multiple Linear Regression/
│   ├── Readme.md                                  # linear regression documentation
│   └── House_prediction.py                       # house price prediction script
└── my-app/
    ├── README.md                                  # app documentation
    ├── app.py                                     # Streamlit application
    └── requirements.txt                          # app dependencies
```

## What this project demonstrates

- **Core ML Concepts**: Dataset generation, train/test splitting, feature engineering, and model evaluation
- **Classification Algorithms**: Logistic Regression, Decision Trees, Random Forests
- **Regression Algorithms**: Simple Linear Regression, Multiple Linear Regression, Ridge, Lasso
- **Ensemble Methods**: Random Forest techniques for improved predictions
- **Model Comparison**: Techniques for comparing multiple models and selecting the best one
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, R², MAE, RMSE, Confusion Matrix
- **Data Visualization**: Using Matplotlib and Seaborn for exploratory data analysis
- **Model Persistence**: Saving and loading trained models with joblib
- **Interactive Prediction**: User input handling and real-time predictions
- **Web Applications**: Building interactive dashboards with Streamlit

## Setup

Recommended Python version: `3.10+`.

From the project root, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```powershell
pip install pandas numpy matplotlib scikit-learn seaborn joblib
```

## Run each project

### 1. Student Pass/Fail Classification

```powershell
python "Classification basics\Student_Pass\Student_Pass\Pass_Fail_Prediction.py"
```

### 2. Student Grade Prediction (Scikit-learn Regression)

```powershell
python "Scikit-learn\first_model.py"
```

### 3. Student Grade Classification (Decision Tree)

```powershell
python "Decision Tree Classifier\Student_Grade.py"
```

### 4. Customer Churn Prediction (Random Forest)

```powershell
python "Random Forest\customer_churn_Prediction.py"
```

### 5. House Price Prediction (Linear Regression)

```powershell
python "Simple & Multiple Linear Regression\House_prediction.py"
```

### 6. Streamlit Interactive App

First, install Streamlit if not already included:

```powershell
pip install streamlit
```

Then run the app:

```powershell
streamlit run "my-app\app.py"
```

This will open an interactive web interface in your browser.

## Project Details

- **Classification Basics**: Builds a logistic regression model for binary classification. Includes data exploration plots and performance metrics (accuracy, precision, recall).

- **Scikit-learn Regression**: Compares multiple regression models (Linear Regression, Ridge, Lasso, etc.) and identifies the best performer, which is saved as `best_model.joblib` for future use.

- **Decision Tree Classifier**: Demonstrates tree-based classification algorithms. Useful for understanding how decision trees make predictions based on feature splits.

- **Random Forest Churn Prediction**: Uses ensemble learning with multiple decision trees. Demonstrates how random forests improve prediction accuracy and handle complex patterns in data.

- **Linear Regression Housing**: Compares simple linear regression (single feature) vs. multiple linear regression (multiple features). Shows how additional features impact model performance and predictions.

- **Streamlit App**: Provides an interactive graphical interface for exploring models and making predictions without writing code. Great for visualization and real-time model interaction.

## Recommended workflow

1. **Activate the virtual environment**.
2. **Start with Simple & Multiple Linear Regression** to understand regression basics.
3. **Move to Classification Basics** to learn binary classification concepts.
4. **Explore Decision Tree Classifier** for tree-based approaches.
5. **Try Random Forest for Churn Prediction** to understand ensemble methods.
6. **Use Scikit-learn Project** to see model comparison and selection in action.
7. **Run the Streamlit App** to interact with models visually.

### Quick Start Commands

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Install all dependencies
pip install pandas numpy matplotlib scikit-learn seaborn joblib streamlit

# Run projects in recommended order
python "Simple & Multiple Linear Regression\House_prediction.py"
python "Classification basics\Student_Pass\Student_Pass\Pass_Fail_Prediction.py"
python "Decision Tree Classifier\Student_Grade.py"
python "Random Forest\customer_churn_Prediction.py"
python "Scikit-learn\first_model.py"
streamlit run "my-app\app.py"
```

## For More Details

Each project folder contains its own **README.md** with:
- Detailed project description and goals
- Dataset information and preprocessing steps
- Model architecture and hyperparameters
- Expected outputs and visualizations
- Troubleshooting tips and common issues

Navigate to each project folder to learn more about specific implementations.

## License

This repository is intended for educational use and can be freely modified.
