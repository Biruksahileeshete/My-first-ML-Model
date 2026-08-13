# ========================================
# DECISION TREE CLASSIFIER
# Student Pass/Fail Prediction
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

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
        
        # Evaluate
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n✅ Model trained!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   Tree Depth: {self.model.get_depth()}")
        print(f"   Number of Leaves: {self.model.get_n_leaves()}")
        
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Fail', 'Pass']))
        
        return self.model, y_pred
    
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
        
        try:
            study_hours = float(input("   Study Hours (2-15): "))
            attendance = float(input("   Attendance (60-100): "))
            previous_gpa = float(input("   Previous GPA (1.5-4.0): "))
            sleep_hours = float(input("   Sleep Hours (4-10): "))
            extracurricular = int(input("   Extracurricular (0=No, 1=Yes): "))
            part_time_job = int(input("   Part Time Job (0=No, 1=Yes): "))
            assignments = int(input("   Assignments Completed (5-10): "))
            test_score = float(input("   Test Score (40-100): "))
            
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
            
        except ValueError:
            print("❌ Invalid input!")
    
    def run_full_analysis(self):
        """Run complete analysis"""
        print("\n🎯 WELCOME TO DECISION TREE CLASSIFIER")
        print("="*50)
        
        self.create_dataset()
        self.prepare_data()
        
        # Try different depths
        print("\n📊 Comparing Different Tree Depths:")
        print("-"*50)
        
        for depth in [3, 5, 8, None]:
            self.train_model(max_depth=depth)
        
        # Train final model
        print("\n🌳 Training final model...")
        self.train_model(max_depth=5)
        
        # Visualize
        self.visualize_tree()
        self.feature_importance()
        
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