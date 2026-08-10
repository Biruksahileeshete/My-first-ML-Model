# ========================================
# TRAIN MY FIRST ML MODEL
# Predict Student Grades with Scikit-Learn
# ========================================

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
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
        self.cv_scores = {}
        self.df = None
        self.feature_columns = [
            'Study_Hours',
            'Attendance',
            'Sleep_Hours',
            'Previous_GPA',
            'Extracurricular',
            'Part_Time_Job'
        ]
        self.best_model_name = None
        self.best_model = None

    def create_dataset(self):
        """Create a realistic student-performance dataset."""
        print("\n📊 Creating sample dataset...")
        np.random.seed(42)
        n = 500

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

        noise = np.random.normal(0, 5, n)
        self.df['Average_Grade'] = (
            0.30 * self.df['Study_Hours'] +
            0.30 * self.df['Attendance'] +
            0.20 * self.df['Previous_GPA'] * 20 +
            0.10 * self.df['Sleep_Hours'] * 5 +
            0.10 * self.df['Extracurricular'] * 10 +
            noise
        ).clip(0, 100)

        print(f"✅ Dataset created: {len(self.df)} students, {len(self.df.columns)} features")
        print("\nFirst 5 rows:")
        print(self.df.head())
        return self.df

    def prepare_data(self):
        """Prepare data for training and model evaluation."""
        print("\n🔧 Preparing data for training...")

        X = self.df[self.feature_columns]
        y = self.df['Average_Grade']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"✅ Data prepared!")
        print(f"   Training samples: {len(self.X_train)} ({len(self.X_train)/len(X)*100:.0f}%)")
        print(f"   Testing samples: {len(self.X_test)} ({len(self.X_test)/len(X)*100:.0f}%)")
        print("\n📊 Feature names:")
        for col in self.feature_columns:
            print(f"   - {col}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def _evaluate_model(self, model_name, model):
        """Fit, predict, and evaluate a model on the test set."""
        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

        print(f"✅ Model trained!")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")

        self.models[model_name] = model
        self.results[model_name] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred,
        }

        return r2, mae, rmse

    def _cross_validate(self, model_name, model):
        """Run cross-validation to measure generalization performance."""
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='r2', n_jobs=-1)
        self.cv_scores[model_name] = {
            'mean_r2': float(np.mean(scores)),
            'std_r2': float(np.std(scores)),
            'scores': scores,
        }
        print(f"   Cross-validation R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        return self.cv_scores[model_name]

    def train_linear_regression(self):
        """Train a linear regression model with normalization."""
        print("\n" + "=" * 50)
        print("📈 TRAINING: Linear Regression")
        print("=" * 50)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        model.fit(self.X_train, self.y_train)
        self._evaluate_model('Linear Regression', model)
        self._cross_validate('Linear Regression', model)

        print("\n📊 Feature Importance (Model Coefficients):")
        coefficients = model.named_steps['regressor'].coef_
        for feature, coef in zip(self.X_train.columns, coefficients):
            print(f"   {feature}: {coef:.2f}")

        return model

    def train_decision_tree(self):
        """Train a decision tree regressor."""
        print("\n" + "=" * 50)
        print("🌳 TRAINING: Decision Tree")
        print("=" * 50)

        model = DecisionTreeRegressor(max_depth=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        self._evaluate_model('Decision Tree', model)
        self._cross_validate('Decision Tree', model)

        print("\n📊 Feature Importance:")
        for feature, importance in zip(self.X_train.columns, model.feature_importances_):
            print(f"   {feature}: {importance:.3f}")

        return model

    def train_random_forest(self):
        """Train a random forest regressor with a strong baseline configuration."""
        print("\n" + "=" * 50)
        print("🌲 TRAINING: Random Forest")
        print("=" * 50)

        model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42)
        model.fit(self.X_train, self.y_train)
        self._evaluate_model('Random Forest', model)
        self._cross_validate('Random Forest', model)

        print("\n📊 Feature Importance:")
        for feature, importance in zip(self.X_train.columns, model.feature_importances_):
            print(f"   {feature}: {importance:.3f}")

        return model

    def tune_random_forest(self):
        """Use grid search to try stronger hyperparameter combinations."""
        print("\n" + "=" * 50)
        print("⚙️ TUNING: Random Forest")
        print("=" * 50)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10, None],
            'min_samples_leaf': [1, 2, 4],
        }

        search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            scoring='r2',
            cv=3,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(self.X_train, self.y_train)
        tuned_model = search.best_estimator_
        self._evaluate_model('Random Forest Tuned', tuned_model)
        self._cross_validate('Random Forest Tuned', tuned_model)
        print(f"\n🔍 Best hyperparameters: {search.best_params_}")
        return tuned_model

    def compare_models(self):
        """Compare all trained models and select the best one."""
        print("\n" + "=" * 50)
        print("📊 MODEL COMPARISON")
        print("=" * 50)

        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'R² Score': [self.results[m]['r2'] for m in self.results],
            'MAE': [self.results[m]['mae'] for m in self.results],
            'RMSE': [self.results[m]['rmse'] for m in self.results],
        }).sort_values('R² Score', ascending=False).reset_index(drop=True)

        print("\nComparison Table:")
        print(comparison.to_string(index=False))

        self.best_model_name = comparison.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   R² Score: {comparison.iloc[0]['R² Score']:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

        axes[0].bar(comparison['Model'], comparison['R² Score'], color=['blue', 'green', 'red', 'purple'])
        axes[0].set_title('R² Scores (Higher is Better)')
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=30)

        axes[1].bar(comparison['Model'], comparison['MAE'], color=['blue', 'green', 'red', 'purple'])
        axes[1].set_title('MAE (Lower is Better)')
        axes[1].set_ylabel('MAE')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=30)

        plt.tight_layout()
        plt.show()

        return comparison

    def save_best_model(self, file_path='best_model.joblib'):
        """Save the best model in the workspace for later use."""
        if self.best_model is None:
            raise ValueError('No trained model available. Train models first.')

        joblib.dump(self.best_model, file_path)
        print(f"✅ Best model saved to: {file_path}")
        return file_path

    def visualize_predictions(self):
        """Visualize the best model's predictions against actual outcomes."""
        if not self.results:
            raise ValueError('No results available. Train models first.')

        print("\n📊 Visualizing predictions...")
        best_model_name = self.best_model_name or max(self.results, key=lambda x: self.results[x]['r2'])
        best_model = self.models[best_model_name]
        y_pred = self.results[best_model_name]['predictions']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Best Model: {best_model_name}', fontsize=14, fontweight='bold')

        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Grade')
        axes[0, 0].set_ylabel('Predicted Grade')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        residuals = self.y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Grade')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)

        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
        else:
            importance = np.abs(best_model.named_steps['regressor'].coef_) if hasattr(best_model, 'named_steps') else np.abs(best_model.coef_)

        features = self.X_train.columns
        axes[1, 0].bar(features, importance)
        axes[1, 0].set_title('Feature Importance')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Importance')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(self.y_test, bins=20, alpha=0.5, label='Actual', color='blue')
        axes[1, 1].hist(y_pred, bins=20, alpha=0.5, label='Predicted', color='red')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].set_xlabel('Grade')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict_student(self, student_data):
        """Predict a grade for a single student using the best trained model."""
        if self.best_model is None:
            raise ValueError('No best model selected. Train models first.')

        features = pd.DataFrame([student_data], columns=self.feature_columns)
        prediction = self.best_model.predict(features)[0]
        return prediction

    def make_predictions(self):
        """Interactive prediction using the best model."""
        print("\n🔮 MAKE PREDICTIONS")
        print("=" * 50)

        if self.best_model is None:
            print("⚠️ No trained model is available yet. Train the models first.")
            return

        print(f"Using model: {self.best_model_name}")
        print("\nEnter student information for prediction:")

        try:
            student_data = {
                'Study_Hours': float(input("   Study Hours (2-15): ")),
                'Attendance': float(input("   Attendance (60-100): ")),
                'Sleep_Hours': float(input("   Sleep Hours (4-10): ")),
                'Previous_GPA': float(input("   Previous GPA (2.0-4.0): ")),
                'Extracurricular': int(input("   Extracurricular (0=No, 1=Yes): ")),
                'Part_Time_Job': int(input("   Part Time Job (0=No, 1=Yes): ")),
            }

            prediction = self.predict_student(student_data)

            print("\n" + "=" * 50)
            print("📊 PREDICTION RESULTS")
            print("=" * 50)
            print(f"   Predicted Average Grade: {prediction:.2f}")

            if prediction >= 80:
                grade = 'Excellent 🌟'
            elif prediction >= 70:
                grade = 'Good 👍'
            elif prediction >= 60:
                grade = 'Average 📚'
            else:
                grade = 'Needs Improvement 📈'

            print(f"   Performance Level: {grade}")

        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")

    def run_full_training(self, interactive=True):
        """Run the complete training workflow."""
        print("\n🎯 WELCOME TO YOUR FIRST ML MODEL")
        print("=" * 50)

        self.create_dataset()
        self.prepare_data()

        self.train_linear_regression()
        self.train_decision_tree()
        self.train_random_forest()
        self.tune_random_forest()

        self.compare_models()
        self.save_best_model()
        self.visualize_predictions()

        if interactive:
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
    trainer.run_full_training(interactive=True)