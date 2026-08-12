# Student Pass/Fail Classification

A logistic regression classification project that predicts whether a student will pass or fail based on study and lifestyle features.

## Project Summary

This project generates a synthetic student dataset, explores the data, trains a logistic regression classifier, evaluates model performance, and allows interactive student outcome predictions.

## Key Features

- Synthetic dataset creation with realistic student variables
- Data exploration and visualizations using Matplotlib and Seaborn
- Feature scaling and train/test splitting
- Logistic regression model training and evaluation
- Confusion matrix, ROC curve, and probability distribution plots
- Interactive prediction for new student data
- Threshold comparison for different classification cutoffs

## Dataset Features

The generated dataset includes:

- `Study_Hours`
- `Attendance`
- `Previous_GPA`
- `Sleep_Hours`
- `Extracurricular`
- `Part_Time_Job`
- `Assignments_Completed`
- `Test_Scores`
- `Pass` (target label)

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

From the project root, run:

```powershell
python "Classification basics\Student_Pass\Student_Pass\Fail_Prediction.py"
```

If using the repository virtual environment:

```powershell
.venv\Scripts\activate
python "Classification basics\Student_Pass\Student_Pass\Fail_Prediction.py"
```

## What the Script Does

1. Generates a synthetic student dataset with multiple academic and lifestyle features.
2. Computes a pass/fail label using a weighted score formula.
3. Displays dataset statistics and class distribution.
4. Visualizes feature differences between pass and fail groups.
5. Prepares data with feature scaling and stratified train/test split.
6. Trains a logistic regression classifier.
7. Evaluates the model using accuracy, precision, recall, F1 score, and ROC-AUC.
8. Shows performance plots and a classification report.
9. Offers interactive predictions for new students.
10. Compares model performance across several probability thresholds.

## Notes

- The dataset is synthetic and intended for demonstration purposes.
- Because the generated dataset may be imbalanced, some metrics such as precision and recall can be affected.
- The script includes threshold comparison logic to help understand how different decision cutoffs impact classification performance.

## Recommended Use

Run the script to explore the generated student dataset and review model behavior. Try multiple inputs during the interactive prediction section to see how the model responds.
