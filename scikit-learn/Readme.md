# Student Grade Prediction Model

This project trains a machine learning model to predict a student's average grade using a realistic generated dataset. It uses scikit-learn and evaluates multiple regression models to find the strongest performer.

## Project Goal

The goal is to predict a student's final average grade based on factors such as:

- Study hours
- Attendance
- Sleep hours
- Previous GPA
- Extracurricular activities
- Part-time job status

## What the model does

The project:

- creates a synthetic student dataset
- prepares training and testing data
- trains several regression models
- compares their performance using metrics like R², MAE, and RMSE
- selects the best-performing model
- saves the best model to disk
- allows single-student prediction input

## Models used

The project compares:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Tuned Random Forest (with grid search)

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib

## Folder structure

```text
My-first-ML-Model/
├── scikit-learn/
│   ├── first_model.py
│   ├── Readme.md
│   └── best_model.joblib
```

## Requirements

Make sure you have Python 3.10+ installed.

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## Run the project

From the project root:

```bash
python scikit-learn/first_model.py
```

If you are using the project virtual environment directly:

```bash
.\.venv\Scripts\python.exe .\scikit-learn\first_model.py
```

## Output

When you run the script, it will:

1. generate the dataset
2. split it into training and test sets
3. train all models
4. print the comparison table
5. display model plots
6. save the best model as `best_model.joblib`

## Example metrics

The model evaluates performance using:

- R² Score: Higher is better
- MAE: Lower is better
- RMSE: Lower is better

## Notes

This is a beginner-friendly ML project designed to demonstrate:

- data preparation
- model training
- model evaluation
- prediction workflow

It is a good starting point for learning regression and model comparison in Python.

## License

This project is for educational purposes and is free to use and modify.
