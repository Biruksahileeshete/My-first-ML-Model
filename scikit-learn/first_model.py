# ========================================
# TRAIN MY FIRST ML MODEL
# Predict Student Grades with Scikit-Learn
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class FirstMLModel:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.df = None
    
    def create_dataset(self):
        """Create sample dataset"""
        print("\n📊 Creating sample dataset...")
        np.random.seed(42)
        n = 500  # 500 students
        
        data = {
            'Study_Hours': np.random.randint(2, 15, n),
            'Attendance': np.random.randint(60, 100, n),
            'Sleep_Hours': np.random.randint(4, 10, n),
            'Previous_GPA': np.round(np.random.uniform(2.0, 4.0, n), 2),
            'Extracurricular': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'Part_Time_Job': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'Grade_Math': np.random.randint(40, 100, n),
            'Grade_English': np.random.randint(50, 100, n),
            'Grade_Science': np.random.randint(45, 100, n),
        }
        
        self.df = pd.DataFrame(data)
        
        # Calculate target (what we want to predict)
        # Add some noise to make it realistic
        noise = np.random.normal(0, 5, n)
        self.df['Average_Grade'] = (
            0.3 * self.df['Study_Hours'] +
            0.3 * self.df['Attendance'] / 100 * 100 +
            0.2 * self.df['Previous_GPA'] * 20 +
            0.1 * self.df['Sleep_Hours'] * 5 +
            0.1 * self.df['Extracurricular'] * 10 +
            noise
        ).clip(0, 100)  # Keep grades between 0-100
        
        print(f"✅ Dataset created: {len(self.df)} students, {len(self.df.columns)} features")
        print("\nFirst 5 rows:")
        print(self.df.head())
        return self.df
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\n🔧 Preparing data for training...")
        
        # Features (X) - what we use to predict
        feature_cols = ['Study_Hours', 'Attendance', 'Sleep_Hours', 
                       'Previous_GPA', 'Extracurricular', 'Part_Time_Job']
        X = self.df[feature_cols]
        
        # Target (y) - what we want to predict
        y = self.df['Average_Grade']
        
        # Split data: 80% training, 20% testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✅ Data prepared!")
        print(f"   Training samples: {len(self.X_train)} ({len(self.X_train)/len(X)*100:.0f}%)")
        print(f"   Testing samples: {len(self.X_test)} ({len(self.X_test)/len(X)*100:.0f}%)")
        
        print("\n📊 Feature names:")
        for col in feature_cols:
            print(f"   - {col}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("\n" + "="*50)
        print("📈 TRAINING: Linear Regression")
        print("="*50)
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Evaluate
        score = model.score(self.X_test, self.y_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"✅ Model trained!")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        
        # Feature importance (coefficients)
        print("\n📊 Feature Importance (Coefficients):")
        for feature, coef in zip(self.X_train.columns, model.coef_):
            print(f"   {feature}: {coef:.2f}")
        
        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred
        }
        
        return model
    
    def train_decision_tree(self):
        """Train Decision Tree model"""
        print("\n" + "="*50)
        print("🌳 TRAINING: Decision Tree")
        print("="*50)
        
        model = DecisionTreeRegressor(max_depth=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Evaluate
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        print(f"✅ Model trained!")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        
        # Feature importance
        print("\n📊 Feature Importance:")
        for feature, importance in zip(self.X_train.columns, model.feature_importances_):
            print(f"   {feature}: {importance:.3f}")
        
        self.models['Decision Tree'] = model
        self.results['Decision Tree'] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred
        }
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("🌲 TRAINING: Random Forest")
        print("="*50)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Evaluate
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        print(f"✅ Model trained!")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        
        # Feature importance
        print("\n📊 Feature Importance:")
        for feature, importance in zip(self.X_train.columns, model.feature_importances_):
            print(f"   {feature}: {importance:.3f}")
        
        self.models['Random Forest'] = model
        self.results['Random Forest'] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred
        }
        
        return model
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*50)
        print("📊 MODEL COMPARISON")
        print("="*50)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'R² Score': [self.results[m]['r2'] for m in self.results],
            'MAE': [self.results[m]['mae'] for m in self.results],
            'RMSE': [self.results[m]['rmse'] for m in self.results]
        })
        
        print("\nComparison Table:")
        print(comparison.to_string(index=False))
        
        # Find best model
        best_model = comparison.loc[comparison['R² Score'].idxmax()]
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"   R² Score: {best_model['R² Score']:.4f}")
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')
        
        # R² Scores
        axes[0].bar(comparison['Model'], comparison['R² Score'], color=['blue', 'green', 'red'])
        axes[0].set_title('R² Scores (Higher is Better)')
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].bar(comparison['Model'], comparison['MAE'], color=['blue', 'green', 'red'])
        axes[1].set_title('MAE (Lower is Better)')
        axes[1].set_ylabel('MAE')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self):
        """Visualize predictions vs actual values"""
        print("\n📊 Visualizing predictions...")
        
        # Select best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['r2'])
        best_model = self.models[best_model_name]
        y_pred = self.results[best_model_name]['predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Best Model: {best_model_name}', fontsize=14, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 
                       'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Grade')
        axes[0, 0].set_ylabel('Predicted Grade')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals Plot
        residuals = self.y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Grade')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
        else:
            importance = abs(best_model.coef_)
        
        features = self.X_train.columns
        axes[1, 0].bar(features, importance)
        axes[1, 0].set_title('Feature Importance')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Importance')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        axes[1, 1].hist(self.y_test, bins=20, alpha=0.5, label='Actual', color='blue')
        axes[1, 1].hist(y_pred, bins=20, alpha=0.5, label='Predicted', color='red')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].set_xlabel('Grade')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def make_predictions(self):
        """Make predictions with the best model"""
        print("\n🔮 MAKE PREDICTIONS")
        print("="*50)
        
        # Get best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['r2'])
        best_model = self.models[best_model_name]
        
        print(f"Using model: {best_model_name}")
        print("\nEnter student information for prediction:")
        
        try:
            study_hours = float(input("   Study Hours (2-15): "))
            attendance = float(input("   Attendance (60-100): "))
            sleep_hours = float(input("   Sleep Hours (4-10): "))
            previous_gpa = float(input("   Previous GPA (2.0-4.0): "))
            extracurricular = int(input("   Extracurricular (0=No, 1=Yes): "))
            part_time_job = int(input("   Part Time Job (0=No, 1=Yes): "))
            
            # Create feature array
            features = np.array([[study_hours, attendance, sleep_hours, 
                                previous_gpa, extracurricular, part_time_job]])
            
            # Make prediction
            prediction = best_model.predict(features)[0]
            
            # Display results
            print("\n" + "="*50)
            print("📊 PREDICTION RESULTS")
            print("="*50)
            print(f"   Predicted Average Grade: {prediction:.2f}")
            
            # Categorize the prediction
            if prediction >= 80:
                grade = "Excellent 🌟"
            elif prediction >= 70:
                grade = "Good 👍"
            elif prediction >= 60:
                grade = "Average 📚"
            else:
                grade = "Needs Improvement 📈"
            
            print(f"   Performance Level: {grade}")
            
            # Show feature breakdown
            print("\n💡 Feature Breakdown:")
            print(f"   Study Hours: {study_hours} (Impact: {study_hours * 0.3:.1f} points)")
            print(f"   Attendance: {attendance}% (Impact: {attendance * 0.3/100 * 100:.1f} points)")
            print(f"   Previous GPA: {previous_gpa} (Impact: {previous_gpa * 0.2 * 20:.1f} points)")
            
        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")
    
    def run_full_training(self):
        """Run complete training pipeline"""
        print("\n🎯 WELCOME TO YOUR FIRST ML MODEL")
        print("="*50)
        
        # Create and prepare data
        self.create_dataset()
        self.prepare_data()
        
        # Train all models
        self.train_linear_regression()
        self.train_decision_tree()
        self.train_random_forest()
        
        # Evaluate and visualize
        self.compare_models()
        self.visualize_predictions()
        
        # Interactive predictions
        predict_more = input("\nMake predictions with best model? (Y/N): ").upper()
        while predict_more == 'Y':
            self.make_predictions()
            predict_more = input("\nMake another prediction? (Y/N): ").upper()
        
        print("\n✅ Training Complete! Your first ML model is ready!")


# ========================================
# RUN THE PROGRAM
# ========================================

if __name__ == "__main__":
    trainer = FirstMLModel()
    trainer.run_full_training()