# My-first-ML-Model

A beginner-friendly machine learning collection with regression and classification examples built in Python.

## Project Overview

This repository contains several small ML projects that demonstrate model creation, training, evaluation, and prediction workflows using synthetic or generated datasets.

### Included projects

- `Classification basics/` - student pass/fail prediction using logistic regression.
- `scikit-learn/` - student grade prediction using multiple regression models and model comparison.
- `Simple & Multiple Linear Regression/` - house price prediction using simple and multiple linear regression.

## Folder Structure

```text
My-first-ML-Model/
├── .venv/                         # optional virtual environment folder
├── best_model.joblib              # trained model artifact from scikit-learn project
├── Classification basics/
│   ├── Readme.md                  # classification project documentation
│   └── Student_Pass/              # student pass/fail classification example
│       └── Student_Pass/
│           └── Fail_Prediction.py
├── scikit-learn/
│   ├── Readme.md                  # regression comparison project documentation
│   ├── first_model.py
│   └── best_model.joblib
└── Simple & Multiple Linear Regression/
    ├── Readme.md                  # house price regression project documentation
    └── House_prediction.py
```

## What this project demonstrates

- Synthetic dataset generation for regression and classification tasks
- Data preparation and train/test splitting
- Model training with scikit-learn
- Model evaluation with metrics such as accuracy, precision, recall, R², MAE, and RMSE
- Visualization of data and results using matplotlib and seaborn
- Saving and loading trained models
- Interactive prediction input from users

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

### 1. Student pass/fail classification

```powershell
python "Classification basics\Student_Pass\Student_Pass\Fail_Prediction.py"
```

### 2. Student grade prediction regression

```powershell
python "scikit-learn\first_model.py"
```

### 3. House price regression

```powershell
python "Simple & Multiple Linear Regression\House_prediction.py"
```

## Notes

- The `Classification basics` project builds a logistic regression model and includes data exploration plots.
- The `scikit-learn` project compares several regression models and saves the best performer as `best_model.joblib`.
- The `Simple & Multiple Linear Regression` project compares simple and multiple linear regression approaches for house pricing.

## Recommended workflow

1. Activate the virtual environment.
2. Run the desired project script.
3. Review printed evaluation metrics and plots.
4. Use interactive input prompts to test custom predictions.

## License

This repository is intended for educational use and can be freely modified.
