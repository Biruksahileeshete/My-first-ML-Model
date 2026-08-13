# 🌳 Decision Tree Classifier - Student Pass/Fail Prediction

A comprehensive machine learning project implementing a Decision Tree Classifier to predict student academic outcomes (Pass/Fail) based on various performance and behavioral metrics.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Evaluation](#model-evaluation)
- [Advanced Features](#advanced-features)
- [Results](#results)
- [Requirements](#requirements)
- [License](#license)

## 🎯 Overview

This project demonstrates the complete machine learning workflow:
- **Data Generation**: Creates a synthetic student dataset with realistic correlations
- **Data Preparation**: Splits data into training and testing sets
- **Model Training**: Trains Decision Tree models with various hyperparameters
- **Model Evaluation**: Comprehensive evaluation using multiple metrics and visualizations
- **Predictions**: Makes predictions for individual students with confidence scores
- **Model Persistence**: Saves and loads trained models

## ✨ Features

### Core ML Features
- ✅ **Decision Tree Classification** - Classic tree-based classifier
- ✅ **Multiple Tree Depths** - Compares different model complexities
- ✅ **Cross-Validation** - 5-fold CV for robust evaluation
- ✅ **Hyperparameter Tuning** - GridSearchCV for optimal parameters
- ✅ **Feature Importance** - Identifies most influential factors

### Evaluation & Visualization
- 📊 **Confusion Matrix** - Heatmap with True/False positives/negatives
- 📈 **ROC Curve** - Plots ROC with AUC score
- 📉 **Precision-Recall Curve** - Additional performance metric
- 🌳 **Tree Visualization** - Beautiful decision tree diagram
- 📋 **Classification Report** - Precision, recall, F1-score
- 📊 **Feature Statistics** - Descriptive stats and class distribution

### Model Management
- 💾 **Save Models** - Export trained models to `.joblib` files
- 📂 **Load Models** - Import previously trained models
- 🔄 **Batch Predictions** - Support for single and multiple predictions
- ✔️ **Input Validation** - Validates all user inputs

### User Features
- 🔮 **Interactive Predictions** - Predict outcomes for custom students
- 💡 **Detailed Feedback** - Decision paths and confidence scores
- 🎨 **Formatted Output** - Clear, organized display with emojis
- ⚠️ **Error Handling** - Comprehensive error messages

## 📦 Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Quick Setup
```bash
# Navigate to project directory
cd "Decision Tree Classifier"

# Install dependencies
pip install -r requirements.txt

# Run the program
python Student_Grade.py
```

## 📁 Project Structure

```
Decision Tree Classifier/
├── Student_Grade.py          # Main program file
├── README.md                 # This file
├── best_model.joblib         # Pre-trained model (optional)
└── student_pass_fail_model.joblib  # Saved model output
```

## 🚀 Usage

### Running the Full Analysis

```python
python Student_Grade.py
```

The program will automatically:
1. Create a synthetic student dataset (500 students)
2. Display feature statistics and class distribution
3. Prepare data (80% train, 20% test with stratification)
4. Train models with different tree depths (3, 5, 8, unlimited)
5. Train final optimized model
6. Perform 5-fold cross-validation
7. Generate visualizations (confusion matrix, ROC, precision-recall)
8. Offer optional hyperparameter tuning
9. Ask to save the model
10. Enable interactive predictions

### Making Individual Predictions

When prompted during execution:
```
🔮 PREDICT STUDENT OUTCOME
==================================================
   Study Hours (2-15): 10
   Attendance (60-100): 85
   Previous GPA (1.5-4.0): 3.5
   Sleep Hours (4-10): 7
   Extracurricular (0=No, 1=Yes): 1
   Part Time Job (0=No, 1=Yes): 0
   Assignments Completed (5-10): 9
   Test Score (40-100): 78
```

### Using in Your Own Code

```python
from Student_Grade import DecisionTreePassFail

# Create instance
dt = DecisionTreePassFail()

# Create and prepare data
dt.create_dataset()
dt.prepare_data()

# Train model
dt.train_model(max_depth=5)

# Make predictions
dt.make_prediction()

# Save model
dt.save_model('my_model.joblib')
```

## 📊 Dataset

### Features (8 input variables)
| Feature | Range | Description |
|---------|-------|-------------|
| Study Hours | 2-15 | Weekly study hours |
| Attendance | 60-100 | Class attendance percentage |
| Previous GPA | 1.5-4.0 | Prior semester GPA |
| Sleep Hours | 4-10 | Average daily sleep hours |
| Extracurricular | 0-1 | Participation in activities |
| Part Time Job | 0-1 | Employment status |
| Assignments Completed | 5-10 | Number of assignments done |
| Test Scores | 40-100 | Average test performance |

### Target Variable
- **Pass** (0/1): Binary classification (0=Fail, 1=Pass)

### Dataset Stats
- **Total Samples**: 500 students
- **Training Samples**: 400 (80%)
- **Testing Samples**: 100 (20%)
- **Class Balance**: Stratified split to maintain class proportions

## 📈 Model Evaluation

The project uses multiple evaluation metrics:

### Metrics Reported
- **Accuracy**: Overall correctness rate
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (0-1 scale)
- **Confusion Matrix**: True/False Positives/Negatives

### Visualizations Generated
1. **Confusion Matrix Heatmap** - Shows prediction errors
2. **ROC Curve** - Trade-off between TPR and FPR
3. **Precision-Recall Curve** - Performance across thresholds
4. **Decision Tree Diagram** - Visual tree structure
5. **Feature Importance Chart** - Relative feature weights

## 🔧 Advanced Features

### Cross-Validation
```python
dt.cross_validation_evaluation(folds=5)
```
Provides robust performance estimate and detects overfitting.

### Hyperparameter Tuning
```python
dt.hyperparameter_tuning()
```
Tests combinations of:
- `max_depth`: [3, 5, 7, 10, 15]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

### Model Persistence
```python
# Save trained model
dt.save_model('my_model.joblib')

# Load saved model
dt.load_model('my_model.joblib')
```

### Input Validation
```python
errors = dt.validate_input(study_hours, attendance, gpa, sleep_hours, 
                          extracurricular, part_time_job, assignments, test_score)
if errors:
    print("Validation errors:", errors)
```

## 📊 Results

Typical performance metrics:
- **Accuracy**: ~85-92%
- **AUC-ROC**: ~0.90-0.95
- **Precision (Pass class)**: ~88-94%
- **Recall (Pass class)**: ~85-90%

*Note: Exact scores vary due to random data generation (seed=42)*

## 🛠️ Requirements

### System Requirements
- Python 3.7 or higher
- 500MB+ free disk space (for model files)
- Display capable of rendering matplotlib plots

### Python Dependencies
```
pandas>=1.0.0
numpy>=1.18.0
scikit-learn>=0.24.0
matplotlib>=3.0.0
seaborn>=0.11.0
joblib>=1.0.0
```

### Optional
- Jupyter Notebook (for interactive exploration)
- VS Code with Python extension

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Data preprocessing and preparation
- ✅ Train-test splitting with stratification
- ✅ Decision Tree algorithms
- ✅ Cross-validation techniques
- ✅ Hyperparameter optimization
- ✅ Model evaluation metrics
- ✅ Data visualization with matplotlib/seaborn
- ✅ Model persistence (serialization)
- ✅ Interactive user input handling
- ✅ Error handling and validation

## 🐛 Troubleshooting

### ModuleNotFoundError
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Matplotlib Display Issues
```python
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' depending on your system
```

### Memory Issues
Reduce dataset size in `create_dataset()`:
```python
n = 200  # Instead of 500
```

## 📝 Customization

### Changing Model Parameters
Edit `train_model()` parameters:
```python
dt.train_model(max_depth=7, min_samples_split=5)
```

### Modifying Feature Set
Edit feature list in `prepare_data()`:
```python
features = ['Study_Hours', 'Attendance', 'Previous_GPA', 'Sleep_Hours']
```

### Adjusting Train/Test Split
Modify in `prepare_data()`:
```python
train_test_split(X, y, test_size=0.25)  # 75/25 split
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Verify all dependencies are installed
4. Check Python version compatibility

## 📚 References

- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Cross-Validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [ROC Curves Explained](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)

## 📄 License

This project is open-source and available for educational purposes.

## ✍️ Author

Created as a machine learning learning project demonstrating Decision Tree classification with comprehensive model evaluation and visualization.

---

## 🎯 Quick Start Checklist

- [ ] Install Python 3.7+
- [ ] Install required dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn joblib`
- [ ] Run the program: `python Student_Grade.py`
- [ ] Follow interactive prompts
- [ ] Review generated visualizations
- [ ] Experiment with different parameters

---

**Last Updated**: August 2026  
**Version**: 2.0 (with Advanced Features)
