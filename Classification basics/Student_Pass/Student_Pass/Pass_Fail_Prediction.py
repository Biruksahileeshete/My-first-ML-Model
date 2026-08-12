# ========================================
# STUDENT PASS/FAIL PREDICTION
# Logistic Regression Classification
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

class StudentPassFailPredictor:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
    
    def create_dataset(self):
        """Create realistic student dataset"""
        print("\n📊 Creating student dataset...")
        np.random.seed(42)
        n = 500  # 500 students
        
        # Generate features
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
        
        # Calculate pass/fail based on multiple factors
        # A student passes if weighted score >= 60
        scores = []
        for i in range(n):
            score = (
                0.3 * self.df.loc[i, 'Study_Hours'] * 5 +      # Study hours effect
                0.25 * self.df.loc[i, 'Attendance'] * 0.6 +     # Attendance effect
                0.2 * self.df.loc[i, 'Previous_GPA'] * 20 +     # Previous GPA effect
                0.1 * self.df.loc[i, 'Test_Scores'] +           # Test scores
                0.1 * self.df.loc[i, 'Assignments_Completed'] * 5 +  # Assignments
                0.05 * self.df.loc[i, 'Extracurricular'] * 10   # Extracurricular bonus
            )
            scores.append(score)
        
        # Add some noise to make it realistic
        scores = np.array(scores) + np.random.normal(0, 5, n)
        
        # Pass if score >= 60
        self.df['Pass'] = (scores >= 60).astype(int)
        
        # Display dataset info
        print(f"✅ Dataset created: {len(self.df)} students")
        print(f"\nClass Distribution:")
        print(f"   Pass: {self.df['Pass'].sum()} students ({self.df['Pass'].mean()*100:.1f}%)")
        print(f"   Fail: {len(self.df) - self.df['Pass'].sum()} students ({(1-self.df['Pass'].mean())*100:.1f}%)")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nFeature Statistics:")
        print(self.df.describe())
        
        return self.df
    
    def explore_data(self):
        """Visualize data relationships"""
        print("\n🔍 Exploring data relationships...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Student Data Exploration', fontsize=14, fontweight='bold')
        
        # 1. Study Hours vs Pass/Fail
        axes[0, 0].boxplot([self.df[self.df['Pass']==0]['Study_Hours'],
                           self.df[self.df['Pass']==1]['Study_Hours']])
        axes[0, 0].set_xticklabels(['Fail', 'Pass'])
        axes[0, 0].set_title('Study Hours by Result')
        axes[0, 0].set_ylabel('Study Hours')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Attendance vs Pass/Fail
        axes[0, 1].boxplot([self.df[self.df['Pass']==0]['Attendance'],
                           self.df[self.df['Pass']==1]['Attendance']])
        axes[0, 1].set_xticklabels(['Fail', 'Pass'])
        axes[0, 1].set_title('Attendance by Result')
        axes[0, 1].set_ylabel('Attendance %')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Previous GPA vs Pass/Fail
        axes[1, 0].boxplot([self.df[self.df['Pass']==0]['Previous_GPA'],
                           self.df[self.df['Pass']==1]['Previous_GPA']])
        axes[1, 0].set_xticklabels(['Fail', 'Pass'])
        axes[1, 0].set_title('Previous GPA by Result')
        axes[1, 0].set_ylabel('GPA')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Test Scores vs Pass/Fail
        axes[1, 1].boxplot([self.df[self.df['Pass']==0]['Test_Scores'],
                           self.df[self.df['Pass']==1]['Test_Scores']])
        axes[1, 1].set_xticklabels(['Fail', 'Pass'])
        axes[1, 1].set_title('Test Scores by Result')
        axes[1, 1].set_ylabel('Test Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print insights
        print("\n📊 Key Differences Between Pass and Fail Students:")
        for col in ['Study_Hours', 'Attendance', 'Previous_GPA', 'Test_Scores']:
            pass_mean = self.df[self.df['Pass']==1][col].mean()
            fail_mean = self.df[self.df['Pass']==0][col].mean()
            diff = pass_mean - fail_mean
            print(f"   {col}: Pass ({pass_mean:.1f}) vs Fail ({fail_mean:.1f}) - Difference: {diff:.1f}")
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\n🔧 Preparing data for training...")
        
        # Select features
        features = ['Study_Hours', 'Attendance', 'Previous_GPA', 'Sleep_Hours',
                   'Extracurricular', 'Part_Time_Job', 'Assignments_Completed', 'Test_Scores']
        
        X = self.df[features]
        y = self.df['Pass']
        
        # Split data: 80% training, 20% testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✅ Data prepared!")
        print(f"   Training samples: {len(self.X_train)} ({len(self.X_train)/len(X)*100:.0f}%)")
        print(f"   Testing samples: {len(self.X_test)} ({len(self.X_test)/len(X)*100:.0f}%)")
        print(f"   Features: {len(features)}")
        print(f"   Features names: {features}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("🎯 TRAINING LOGISTIC REGRESSION")
        print("="*50)
        
        # Create and train model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n✅ Model trained successfully!")
        print(f"\n📊 Performance Metrics:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        
        # Feature importance (coefficients)
        feature_names = ['Study_Hours', 'Attendance', 'Previous_GPA', 'Sleep_Hours',
                        'Extracurricular', 'Part_Time_Job', 'Assignments_Completed', 'Test_Scores']
        
        print(f"\n📊 Feature Importance (Coefficients):")
        for feature, coef in zip(feature_names, self.model.coef_[0]):
            impact = "Positive" if coef > 0 else "Negative"
            print(f"   {feature}: {coef:.3f} ({impact} impact)")
        
        # Store predictions for visualization
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        self.evaluate_model(y_pred, y_pred_proba)
        
        return self.model
    
    def evaluate_model(self, y_pred, y_pred_proba):
        """Evaluate and visualize model performance"""
        print("\n📊 Detailed Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Fail', 'Pass'], zero_division=0))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Analysis', fontsize=14, fontweight='bold')
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC (AUC = {roc_auc_score(self.y_test, y_pred_proba):.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Probability Distribution
        axes[1, 0].hist(self.y_pred_proba[self.y_test==0], bins=20, alpha=0.5, label='Fail', color='red')
        axes[1, 0].hist(self.y_pred_proba[self.y_test==1], bins=20, alpha=0.5, label='Pass', color='green')
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance (Coefficients)
        feature_names = ['Study_Hours', 'Attendance', 'Previous_GPA', 'Sleep_Hours',
                        'Extracurricular', 'Part_Time_Job', 'Assignments_Completed', 'Test_Scores']
        coeffs = self.model.coef_[0]
        colors = ['green' if c > 0 else 'red' for c in coeffs]
        axes[1, 1].barh(feature_names, coeffs, color=colors)
        axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_title('Feature Coefficients')
        axes[1, 1].set_xlabel('Coefficient')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def make_predictions(self):
        """Make predictions on new students"""
        print("\n🔮 PREDICT STUDENT OUTCOME")
        print("="*50)
        
        print("\nEnter student information:")
        
        try:
            study_hours = float(input("   Study Hours (2-15): "))
            attendance = float(input("   Attendance (60-100): "))
            previous_gpa = float(input("   Previous GPA (1.5-4.0): "))
            sleep_hours = float(input("   Sleep Hours (4-10): "))
            extracurricular = int(input("   Extracurricular (0=No, 1=Yes): "))
            part_time_job = int(input("   Part Time Job (0=No, 1=Yes): "))
            assignments = int(input("   Assignments Completed (5-10): "))
            test_score = float(input("   Test Score (40-100): "))
            
            # Create feature array
            features = np.array([[study_hours, attendance, previous_gpa, sleep_hours,
                                extracurricular, part_time_job, assignments, test_score]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            # Display results
            print("\n" + "="*50)
            print("📊 PREDICTION RESULTS")
            print("="*50)
            
            print(f"\n   Student Profile:")
            print(f"      Study Hours: {study_hours}")
            print(f"      Attendance: {attendance}%")
            print(f"      Previous GPA: {previous_gpa}")
            print(f"      Sleep Hours: {sleep_hours}")
            print(f"      Extracurricular: {'Yes' if extracurricular else 'No'}")
            print(f"      Part-Time Job: {'Yes' if part_time_job else 'No'}")
            print(f"      Assignments Completed: {assignments}")
            print(f"      Test Score: {test_score}")
            
            print(f"\n   🔮 Prediction:")
            if prediction == 1:
                print(f"      ✅ Student is likely to PASS")
                print(f"      Confidence: {probability*100:.1f}%")
                print(f"      💪 Strong student!")
            else:
                print(f"      ❌ Student is likely to FAIL")
                print(f"      Confidence: {(1-probability)*100:.1f}%")
                print(f"      📚 Needs improvement!")
            
            # Give recommendations
            print(f"\n💡 Recommendations:")
            if study_hours < 8:
                print(f"   📖 Increase study hours (currently {study_hours}, recommend 8+)")
            if attendance < 85:
                print(f"   📍 Improve attendance (currently {attendance}%, recommend 85%+)")
            if previous_gpa < 2.5:
                print(f"   📊 Focus on improving GPA (currently {previous_gpa})")
            if sleep_hours < 7:
                print(f"   😴 Get more sleep (currently {sleep_hours}, recommend 7-8 hours)")
            if assignments < 8:
                print(f"   📝 Complete more assignments (currently {assignments}, recommend 8+)")
            if test_score < 70:
                print(f"   📚 Practice more for tests (currently {test_score})")
            
        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")
    
    def compare_thresholds(self):
        """Compare different probability thresholds"""
        print("\n🎯 COMPARING DIFFERENT THRESHOLDS")
        print("="*50)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-"*60)
        
        results = []
        for threshold in thresholds:
            y_pred_threshold = (self.y_pred_proba >= threshold).astype(int)
            accuracy = accuracy_score(self.y_test, y_pred_threshold)
            precision = precision_score(self.y_test, y_pred_threshold, zero_division=0)
            recall = recall_score(self.y_test, y_pred_threshold, zero_division=0)
            f1 = f1_score(self.y_test, y_pred_threshold, zero_division=0)
            
            results.append((threshold, accuracy, precision, recall, f1))
            print(f"{threshold:.1f}{'':8} {accuracy:.4f}{'':7} {precision:.4f}{'':7} {recall:.4f}{'':7} {f1:.4f}")
        
        # Find best threshold
        best = max(results, key=lambda x: x[4])  # Max F1 score
        print(f"\n✅ Best threshold: {best[0]:.1f} (F1 Score: {best[4]:.4f})")
        
        return results
    
    def run_full_analysis(self):
        """Run complete analysis"""
        print("\n🎯 WELCOME TO STUDENT PASS/FAIL PREDICTOR")
        print("="*50)
        
        # Create and explore data
        self.create_dataset()
        self.explore_data()
        
        # Prepare and train
        self.prepare_data()
        self.train_model()
        
        # Compare thresholds
        self.compare_thresholds()
        
        # Interactive predictions
        print("\n" + "="*50)
        predict_choice = input("\nWant to predict a student outcome? (Y/N): ").upper()
        
        if predict_choice == 'Y':
            while True:
                self.make_predictions()
                again = input("\nPredict another student? (Y/N): ").upper()
                if again != 'Y':
                    break
        
        print("\n✅ Analysis Complete! 🎓")
# ========================================
# RUN THE PROGRAM
# ========================================

if __name__ == "__main__":
    predictor = StudentPassFailPredictor()
    predictor.run_full_analysis()