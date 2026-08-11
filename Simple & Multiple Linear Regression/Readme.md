# House Price Prediction with Linear Regression

This project demonstrates how to build and compare two linear regression models for predicting house prices:

- Simple Linear Regression using only `Size_sqft`
- Multiple Linear Regression using several features such as size, bedrooms, age, and location score

It is a beginner-friendly machine learning project focused on regression, data preparation, model evaluation, and visual interpretation.

## Project Goal

The objective is to predict a house price based on real-world-style property features. The project generates a synthetic housing dataset, trains linear regression models, compares their performance, and allows user-based price prediction inputs.

## Features

- Creates a realistic synthetic housing dataset
- Splits the data into training and testing sets
- Trains a simple regression model
- Trains a multiple regression model
- Compares model performance using R², MAE, and RMSE
- Displays charts for regression and prediction results
- Allows for manual prediction input based on selected model

## Dataset

The generated dataset includes:

- `Size_sqft`
- `Bedrooms`
- `Age`
- `Location_Score`
- `Price`

The `Price` variable is generated from a formula that combines:

- size of the house
- number of bedrooms
- age of the house
- neighborhood/location score
- random noise

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

## Project Structure

```text
My-first-ML-Model/
├── Simple & Multiple Linear Regression/
│   ├── House_prediction.py
│   └── Readme.md
├── scikit-learn/
│   ├── first_model.py
│   ├── Readme.md
│   └── best_model.joblib
```

## Requirements

Make sure Python 3.9+ is installed on your system.

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Run the Project

From the project root, run:

```bash
python "Simple & Multiple Linear Regression/House_prediction.py"
```

If your system uses the Python launcher:

```bash
py "Simple & Multiple Linear Regression/House_prediction.py"
```

## What Happens When You Run It

The script will:

1. create the house dataset
2. train the simple regression model
3. train the multiple regression model
4. compare the models using performance metrics
5. display visual plots
6. ask whether you want to make a custom price prediction

## Evaluation Metrics

The project uses:

- R² Score: higher is better
- MAE (Mean Absolute Error): lower is better
- RMSE (Root Mean Squared Error): lower is better

## Example Interpretation

The model estimates how much the house price changes with different features. For example:

- larger house size usually increases the price
- more bedrooms can increase the price
- older houses may reduce the estimated value
- better location scores usually raise the price

## Notes

This project is designed to help beginners understand:

- linear regression
- feature selection
- model evaluation
- prediction workflow
- simple data visualization in Python

It is a simple but useful introduction to regression modeling in machine learning.

## License

This project is for educational purposes and can be freely used and modified.
