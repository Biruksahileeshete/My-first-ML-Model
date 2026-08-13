# ========================================
# DECISION TREE CLASSIFIER
# Student Pass/Fail Prediction
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, auc, roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import seaborn as sns

class DecisionTreePassFail:
    def __init__(self):
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def create_dataset(self):
        """Create student dataset"""
        print("\n📊 Creating student dataset...")
        np.random.seed(42)
        n = 500
        
        data = {
            'Study_Hours': np.random.randint(2, 15, n),
            'Attendance': np.random.randint(60, 100, n),
            'Previous_GPA': np.round(np.random.uniform(1.5, 4.0, n), 2),
            'Sleep_Hours': np.random.randint(4, 10, n),
            'Extracurricular': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'Part_Time_Job': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'Assignments_Completed': np.random.randint(5, 11, n),
            'Test_Scores': np.random.randint(40, 100, n)
        }
        
        self.df = pd.DataFrame(data)
        
        # Calculate pass/fail
        scores = []
        for i in range(n):
            score = (
                0.3 * self.df.loc[i, 'Study_Hours'] * 5 +
                0.25 * self.df.loc[i, 'Attendance'] * 0.6 +
                0.2 * self.df.loc[i, 'Previous_GPA'] * 20 +
                0.1 * self.df.loc[i, 'Test_Scores'] +
                0.1 * self.df.loc[i, 'Assignments_Completed'] * 5 +
                0.05 * self.df.loc[i, 'Extracurricular'] * 10
            )
            scores.append(score)
        
        scores = np.array(scores) + np.random.normal(0, 5, n)
        self.df['Pass'] = (scores >= 60).astype(int)
        
        print(f"✅ Dataset created: {len(self.df)} students")
        print(f"   Pass: {self.df['Pass'].sum()} ({self.df['Pass'].mean()*100:.1f}%)")
        print(f"   Fail: {len(self.df)-self.df['Pass'].sum()} ({(1-self.df['Pass'].mean())*100:.1f}%)")
        return self.df
    
    def prepare_data(self):
        """Prepare data for training"""
        features = ['Study_Hours', 'Attendance', 'Previous_GPA', 'Sleep_Hours',
                   'Extracurricular', 'Part_Time_Job', 'Assignments_Completed', 'Test_Scores']
        
        X = self.df[features]
        y = self.df['Pass']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✅ Data prepared: {len(self.X_train)} train, {len(self.X_test)} test")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, max_depth=None, min_samples_split=2):
        """Train Decision Tree model"""
        print(f"\n🌳 Training Decision Tree...")
        print(f"   Max Depth: {max_depth if max_depth else 'Unlimited'}")
        print(f"   Min Samples Split: {min_samples_split}")
        
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Evaluate
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n✅ Model trained!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   Tree Depth: {self.model.get_depth()}")
        print(f"   Number of Leaves: {self.model.get_n_leaves()}")
        
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Fail', 'Pass']))
        
        return self.model, y_pred, y_pred_proba
    
    def visualize_tree(self, feature_names=None, class_names=None):
        """Visualize the decision tree"""
        if self.model is None:
            print("❌ Train model first!")
            return
        
        if feature_names is None:
            feature_names = ['Study_Hours', 'Attendance', 'Previous_GPA', 'Sleep_Hours',
                           'Extracurricular', 'Part_Time_Job', 'Assignments_Completed', 'Test_Scores']
        
        if class_names is None:
            class_names = ['Fail', 'Pass']
        
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, feature_names=feature_names, 
                 class_names=class_names, filled=True, rounded=True,
                 fontsize=10, max_depth=4)
        plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self):
        """Show feature importance"""
        if self.model is None:
            print("❌ Train model first!")
            return
        
        feature_names = ['Study_Hours', 'Attendance', 'Previous_GPA', 'Sleep_Hours',
                        'Extracurricular', 'Part_Time_Job', 'Assignments_Completed', 'Test_Scores']
        
        importance = self.model.feature_importances_
        
        # Sort features by importance
        sorted_idx = np.argsort(importance)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), importance[sorted_idx], color='teal')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Decision Tree - Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        
        print("\n📊 Feature Importance:")
        for name, imp in zip(feature_names, importance):
            print(f"   {name}: {imp:.4f}")
    
    def make_prediction(self):
        """Make prediction for a single student"""
        print("\n🔮 PREDICT STUDENT OUTCOME")
        print("="*50)
        
        if self.model is None:
            print("❌ Train model first!")
            return
        
        try:
            study_hours = float(input("   Study Hours (2-15): "))
            attendance = float(input("   Attendance (60-100): "))
            previous_gpa = float(input("   Previous GPA (1.5-4.0): "))
            sleep_hours = float(input("   Sleep Hours (4-10): "))
            extracurricular = int(input("   Extracurricular (0=No, 1=Yes): "))
            part_time_job = int(input("   Part Time Job (0=No, 1=Yes): "))
            assignments = int(input("   Assignments Completed (5-10): "))
            test_score = float(input("   Test Score (40-100): "))
            
            # Validate input
            errors = self.validate_input(study_hours, attendance, previous_gpa, sleep_hours,
                                        extracurricular, part_time_job, assignments, test_score)
            
            if errors:
                print("\n❌ Input Validation Errors:")
                for error in errors:
                    print(f"   - {error}")
                return
            
            features = np.array([[study_hours, attendance, previous_gpa, sleep_hours,
                                extracurricular, part_time_job, assignments, test_score]])
            
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            print("\n" + "="*50)
            print("📊 PREDICTION RESULTS")
            print("="*50)
            
            print(f"\n   Decision Path:")
            path = self.model.decision_path(features)
            print(f"   Number of nodes in path: {path[0].getnnz()}")
            
            if prediction == 1:
                print(f"   ✅ Student will PASS")
                print(f"   Confidence: {probability[1]*100:.1f}%")
            else:
                print(f"   ❌ Student will FAIL")
                print(f"   Confidence: {probability[0]*100:.1f}%")
            
        except ValueError as e:
            print(f"❌ Invalid input! {e}")
    
    def show_feature_statistics(self):
        """Display statistics for each feature"""
        print("\n📈 FEATURE STATISTICS")
        print("="*70)
        
        feature_names = ['Study_Hours', 'Attendance', 'Previous_GPA', 'Sleep_Hours',
                        'Extracurricular', 'Part_Time_Job', 'Assignments_Completed', 'Test_Scores']
        
        stats_df = self.df[feature_names].describe().T
        stats_df['skew'] = self.df[feature_names].skew()
        print(stats_df)
        
        # Class distribution
        print("\n📊 CLASS DISTRIBUTION:")
        class_dist = self.df['Pass'].value_counts()
        print(f"   Pass (1): {class_dist[1]} ({class_dist[1]/len(self.df)*100:.1f}%)")
        print(f"   Fail (0): {class_dist[0]} ({class_dist[0]/len(self.df)*100:.1f}%)")
    
    def plot_confusion_matrix(self, y_pred):
        """Plot confusion matrix as heatmap"""
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n📊 CONFUSION MATRIX:")
        print(f"   True Negatives: {cm[0, 0]}")
        print(f"   False Positives: {cm[0, 1]}")
        print(f"   False Negatives: {cm[1, 0]}")
        print(f"   True Positives: {cm[1, 1]}")
    
    def plot_roc_curve(self, y_pred_proba):
        """Plot ROC curve and calculate AUC"""
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 ROC-AUC Score: {roc_auc:.4f}")
        return roc_auc
    
    def plot_precision_recall_curve(self, y_pred_proba):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.show()
    
    def cross_validation_evaluation(self, folds=5):
        """Perform cross-validation evaluation"""
        print(f"\n🔄 CROSS-VALIDATION EVALUATION ({folds}-fold)")
        print("="*50)
        
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=folds, scoring='accuracy')
        
        print(f"   Fold Scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Also test on test set
        test_accuracy = self.model.score(self.X_test, self.y_test)
        print(f"   Test Set Score: {test_accuracy:.4f}")
        
        return cv_scores
    
    def hyperparameter_tuning(self):
        """Perform grid search for best hyperparameters"""
        print("\n🔍 HYPERPARAMETER TUNING (GridSearchCV)")
        print("="*50)
        
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print("   Testing parameters...")
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n   Best Parameters: {grid_search.best_params_}")
        print(f"   Best CV Score: {grid_search.best_score_:.4f}")
        
        # Test on test set
        best_model = grid_search.best_estimator_
        test_accuracy = best_model.score(self.X_test, self.y_test)
        print(f"   Test Set Score: {test_accuracy:.4f}")
        
        return grid_search
    
    def save_model(self, filename="student_pass_fail_model.joblib"):
        """Save trained model to file"""
        if self.model is None:
            print("❌ No model to save! Train a model first.")
            return False
        
        try:
            joblib.dump(self.model, filename)
            print(f"✅ Model saved successfully: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filename="student_pass_fail_model.joblib"):
        """Load a previously trained model from file"""
        try:
            self.model = joblib.load(filename)
            print(f"✅ Model loaded successfully: {filename}")
            return True
        except FileNotFoundError:
            print(f"❌ Model file not found: {filename}")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def validate_input(self, study_hours, attendance, previous_gpa, sleep_hours, 
                      extracurricular, part_time_job, assignments, test_score):
        """Validate user input ranges"""
        errors = []
        
        if not (2 <= study_hours <= 15):
            errors.append("Study Hours must be between 2 and 15")
        if not (60 <= attendance <= 100):
            errors.append("Attendance must be between 60 and 100")
        if not (1.5 <= previous_gpa <= 4.0):
            errors.append("Previous GPA must be between 1.5 and 4.0")
        if not (4 <= sleep_hours <= 10):
            errors.append("Sleep Hours must be between 4 and 10")
        if extracurricular not in [0, 1]:
            errors.append("Extracurricular must be 0 or 1")
        if part_time_job not in [0, 1]:
            errors.append("Part Time Job must be 0 or 1")
        if not (5 <= assignments <= 10):
            errors.append("Assignments Completed must be between 5 and 10")
        if not (40 <= test_score <= 100):
            errors.append("Test Score must be between 40 and 100")
        
        return errors
    
    def run_full_analysis(self):
        """Run complete analysis"""
        print("\n🎯 WELCOME TO DECISION TREE CLASSIFIER")
        print("="*50)
        
        self.create_dataset()
        self.show_feature_statistics()
        self.prepare_data()
        
        # Try different depths
        print("\n📊 Comparing Different Tree Depths:")
        print("-"*50)
        
        for depth in [3, 5, 8, None]:
            self.train_model(max_depth=depth)
        
        # Train final model
        print("\n🌳 Training final model...")
        self.model, y_pred, y_pred_proba = self.train_model(max_depth=5)
        
        # Cross-validation
        self.cross_validation_evaluation(folds=5)
        
        # Visualizations
        print("\n📊 Generating visualizations...")
        self.plot_confusion_matrix(y_pred)
        self.plot_roc_curve(y_pred_proba)
        self.plot_precision_recall_curve(y_pred_proba)
        self.visualize_tree()
        self.feature_importance()
        
        # Hyperparameter tuning option
        tune = input("\n🔍 Perform hyperparameter tuning? (Y/N): ").upper()
        if tune == 'Y':
            self.hyperparameter_tuning()
        
        # Save model option
        save = input("\n💾 Save the model? (Y/N): ").upper()
        if save == 'Y':
            self.save_model()
        
        # Make predictions
        predict = input("\nMake a prediction? (Y/N): ").upper()
        if predict == 'Y':
            self.make_prediction()
        
        print("\n✅ Analysis Complete! 🌳")


# ========================================
# RUN THE PROGRAM
# ========================================

if __name__ == "__main__":
    dt = DecisionTreePassFail()
    dt.run_full_analysis()