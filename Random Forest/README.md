# Customer Churn Prediction using Random Forest

## 📋 Project Overview

This project implements a **Customer Churn Prediction Model** using the **Random Forest Classifier** algorithm. It predicts whether customers are likely to churn (leave/cancel service) based on their behavioral, financial, and engagement patterns.

## 🎯 Key Features

- **Synthetic Dataset Generation**: Creates realistic customer data with 2000 customers and 18 features
- **Exploratory Data Analysis**: Visualizes patterns in customer behavior and churn distribution
- **Data Preprocessing**: Handles categorical encoding, scaling, and class imbalance
- **Random Forest Classification**: Trains an ensemble model with 100 decision trees
- **Hyperparameter Tuning**: Optional GridSearchCV for optimizing model performance
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, F1-score, and ROC-AUC metrics
- **Interactive Predictions**: Allows users to input customer data and predict churn probability
- **Feature Importance Analysis**: Identifies the most influential factors affecting churn

## 📊 Model Performance

The trained model achieves:
- **Accuracy**: 89.5%
- **Precision**: 88.0%
- **Recall**: 92.96%
- **F1-Score**: 90.41%
- **ROC-AUC**: 0.9633

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Installation Steps

1. **Clone/Navigate to the project directory**:
   ```bash
   cd "Random Forest"
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     .\.venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

## 🚀 Running the Project

Execute the script:
```bash
python customer_churn_Prediction.py
```

The program will guide you through the following steps:
1. **Create Dataset**: Generates synthetic customer data
2. **Explore Data**: Shows visualizations of churn patterns
3. **Prepare Data**: Handles encoding, scaling, and train/test split
4. **Train Model**: Trains the Random Forest classifier
5. **Evaluate Model**: Displays performance metrics and visualizations
6. **Optional Tuning**: Allows hyperparameter optimization
7. **Predictions**: Makes churn predictions for custom customer profiles

## 📁 Project Structure

```
customer_churn_Prediction.py    # Main script
README.md                        # This file
```

## 🎯 Key Features Influencing Churn

The model identifies these as the top factors:

1. **Days Since Last Login** (16.6%) - Inactive customers are at high risk
2. **Contract Length: Two-Year** (15.1%) - Long-term contracts reduce churn
3. **Engagement Score** (13.7%) - High engagement = lower churn risk
4. **Customer Service Rating** (12.3%) - Poor ratings increase churn
5. **Monthly Usage Hours** (8.7%) - Low usage = higher churn risk

## 💡 Use Cases

- **Customer Retention**: Identify high-risk customers for targeted retention campaigns
- **Resource Allocation**: Focus customer support on customers likely to churn
- **Pricing Strategy**: Offer incentives to high-risk customers
- **Service Improvement**: Understand which factors drive customer satisfaction

## 🔮 Interactive Prediction Example

When running the script, you can input customer details:
```
Monthly Usage Hours: 50
Number of Support Tickets: 2
Days Since Last Login: 45
Monthly Charges: 75
Total Revenue: 1500
Discount %: 10
Age: 35
Gender: Male
Income Level: Medium
Contract: Yearly
Payment Method: Credit Card
Has Referral: 1
Service Rating: 4
Newsletter Subscribed: 1
Mobile App User: 1
Engagement Score: 0.7
```

The model will predict whether this customer is likely to churn and provide risk assessment with recommendations.

## 📈 Visualization Outputs

The script generates the following plots:

1. **Churn Distribution**: Bar chart of churned vs. retained customers
2. **Support Tickets vs Churn**: Impact of support tickets on churn rate
3. **Contract Length vs Churn**: Churn rates by contract type
4. **Engagement Score Analysis**: Boxplot comparing engagement by churn status
5. **Usage Hours Analysis**: Boxplot comparing usage by churn status
6. **Login Activity Analysis**: Boxplot comparing days since login by churn status
7. **Confusion Matrix**: Classification performance visualization
8. **ROC Curve**: Trade-off between True Positive Rate and False Positive Rate
9. **Probability Distribution**: Separation of churn and retention probabilities
10. **Feature Importance**: Top features driving model decisions

## 🔧 Customization Options

You can modify these parameters in the code:

- `n_estimators`: Number of trees in the forest (default: 100)
- `max_depth`: Maximum tree depth (default: 10)
- `test_size`: Train/test split ratio (default: 0.2)
- `random_state`: Seed for reproducibility (default: 42)

## 📊 Data Features

The model uses 19 features across 4 categories:

**Usage Patterns**
- Monthly_Usage_Hours
- Num_Support_Tickets
- Days_Since_Last_Login

**Financial**
- Monthly_Charges
- Total_Revenue
- Discount_Percentage

**Demographics**
- Age
- Gender
- Income_Level

**Engagement**
- Has_Referral
- Customer_Service_Rating
- Newsletter_Subscribed
- Mobile_App_User
- Engagement_Score

**Service Information**
- Contract_Length
- Payment_Method

## ⚠️ Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
- **Solution**: Install scikit-learn: `pip install scikit-learn`

**Issue**: Matplotlib plots not showing
- **Solution**: Ensure you have a display environment or use Jupyter notebook

**Issue**: Memory errors with large datasets
- **Solution**: Reduce `n_estimators` or `max_depth` parameters

## 📚 References

- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Random Forest Algorithm](https://en.wikipedia.org/wiki/Random_forest)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

## 📝 License

This project is provided for educational purposes.

## 🤝 Contributing

Feel free to modify and enhance this project by:
- Adding new features to the dataset
- Trying different algorithms (XGBoost, LightGBM, etc.)
- Improving data visualization
- Optimizing hyperparameters further

## 📧 Support

For questions or improvements, review the code comments and scikit-learn documentation.

---

**Last Updated**: August 2026
**Python Version**: 3.7+
**Status**: ✅ Working
