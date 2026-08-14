# ========================================
# CUSTOMER CHURN PREDICTION
# Using Random Forest Classifier
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV

class CustomerChurnPredictor:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def create_dataset(self):
        """Create realistic customer churn dataset"""
        print("\n📊 Creating customer dataset...")
        np.random.seed(42)
        n = 2000  # 2000 customers
        
        data = {
            'Customer_ID': [f'C{str(i).zfill(5)}' for i in range(1, n+1)],
            
            # Usage Patterns
            'Monthly_Usage_Hours': np.random.randint(10, 200, n),
            'Num_Support_Tickets': np.random.poisson(2, n),
            'Days_Since_Last_Login': np.random.randint(1, 365, n),
            
            # Financial
            'Monthly_Charges': np.round(np.random.uniform(20, 150, n), 2),
            'Total_Revenue': np.round(np.random.uniform(200, 5000, n), 2),
            'Discount_Percentage': np.round(np.random.uniform(0, 40, n), 2),
            
            # Demographics
            'Age': np.random.randint(18, 80, n),
            'Gender': np.random.choice(['Male', 'Female'], n, p=[0.48, 0.52]),
            'Income_Level': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.2, 0.5, 0.3]),
            
            # Service Related
            'Contract_Length': np.random.choice(['Monthly', 'Yearly', 'Two_Year'], n, p=[0.5, 0.3, 0.2]),
            'Payment_Method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer'], n),
            'Has_Referral': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'Customer_Service_Rating': np.random.randint(1, 6, n),
            
            # Engagement
            'Newsletter_Subscribed': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'Mobile_App_User': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'Engagement_Score': np.round(np.random.uniform(0, 1, n), 2)
        }
        
        self.df = pd.DataFrame(data)
        
        # Calculate churn probability based on multiple factors
        churn_scores = []
        for i in range(n):
            score = 0
            
            # High support tickets = higher churn
            if self.df.loc[i, 'Num_Support_Tickets'] > 3:
                score += 0.25
            elif self.df.loc[i, 'Num_Support_Tickets'] > 1:
                score += 0.1
            
            # Long time since login = higher churn
            days_since = self.df.loc[i, 'Days_Since_Last_Login']
            if days_since > 180:
                score += 0.3
            elif days_since > 90:
                score += 0.2
            elif days_since > 30:
                score += 0.1
            
            # Low engagement = higher churn
            if self.df.loc[i, 'Engagement_Score'] < 0.4:
                score += 0.2
            
            # Low usage = higher churn
            if self.df.loc[i, 'Monthly_Usage_Hours'] < 30:
                score += 0.15
            elif self.df.loc[i, 'Monthly_Usage_Hours'] > 100:
                score -= 0.1
            
            # Monthly contract = higher churn
            if self.df.loc[i, 'Contract_Length'] == 'Monthly':
                score += 0.2
            elif self.df.loc[i, 'Contract_Length'] == 'Two_Year':
                score -= 0.15
            
            # Poor service rating = higher churn
            if self.df.loc[i, 'Customer_Service_Rating'] <= 2:
                score += 0.15
            elif self.df.loc[i, 'Customer_Service_Rating'] >= 4:
                score -= 0.1
            
            # Add noise
            score += np.random.normal(0, 0.05)
            
            churn_scores.append(score)
        
        # Convert to binary (churn = 1 if score > 0.4)
        self.df['Churn'] = (np.array(churn_scores) > 0.4).astype(int)
        
        # Display dataset info
        print(f"✅ Dataset created: {len(self.df)} customers")
        print(f"\n📊 Churn Distribution:")
        print(f"   Churned: {self.df['Churn'].sum()} customers ({self.df['Churn'].mean()*100:.1f}%)")
        print(f"   Retained: {len(self.df)-self.df['Churn'].sum()} customers ({(1-self.df['Churn'].mean())*100:.1f}%)")
        
        print("\n📋 First 5 rows:")
        print(self.df.head())
        
        return self.df
    
    def explore_data(self):
        """Explore and visualize data patterns"""
        print("\n🔍 EXPLORING DATA PATTERNS")
        print("="*50)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Customer Churn Analysis', fontsize=16, fontweight='bold')
        
        # 1. Churn Distribution
        churn_counts = self.df['Churn'].value_counts()
        axes[0, 0].bar(['Retained', 'Churned'], churn_counts.values, color=['green', 'red'])
        axes[0, 0].set_title('Churn Distribution')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Support Tickets vs Churn
        churn_by_tickets = self.df.groupby('Num_Support_Tickets')['Churn'].mean()
        axes[0, 1].bar(churn_by_tickets.index, churn_by_tickets.values, color='orange')
        axes[0, 1].set_title('Support Tickets vs Churn Rate')
        axes[0, 1].set_xlabel('Number of Support Tickets')
        axes[0, 1].set_ylabel('Churn Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Contract Length vs Churn
        churn_by_contract = self.df.groupby('Contract_Length')['Churn'].mean()
        axes[0, 2].bar(churn_by_contract.index, churn_by_contract.values, 
                       color=['red', 'orange', 'green'])
        axes[0, 2].set_title('Contract Length vs Churn Rate')
        axes[0, 2].set_ylabel('Churn Rate')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Engagement Score vs Churn
        axes[1, 0].boxplot([self.df[self.df['Churn']==0]['Engagement_Score'],
                           self.df[self.df['Churn']==1]['Engagement_Score']],
                          labels=['Retained', 'Churned'])
        axes[1, 0].set_title('Engagement Score by Churn')
        axes[1, 0].set_ylabel('Engagement Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Monthly Usage vs Churn
        axes[1, 1].boxplot([self.df[self.df['Churn']==0]['Monthly_Usage_Hours'],
                           self.df[self.df['Churn']==1]['Monthly_Usage_Hours']],
                          labels=['Retained', 'Churned'])
        axes[1, 1].set_title('Monthly Usage by Churn')
        axes[1, 1].set_ylabel('Usage Hours')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Days Since Last Login vs Churn
        axes[1, 2].boxplot([self.df[self.df['Churn']==0]['Days_Since_Last_Login'],
                           self.df[self.df['Churn']==1]['Days_Since_Last_Login']],
                          labels=['Retained', 'Churned'])
        axes[1, 2].set_title('Days Since Last Login by Churn')
        axes[1, 2].set_ylabel('Days')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Key insights
        print("\n💡 Key Insights:")
        print(f"   • Customers with >3 support tickets: {self.df[self.df['Num_Support_Tickets']>3]['Churn'].mean()*100:.1f}% churn rate")
        print(f"   • Monthly contract customers: {self.df[self.df['Contract_Length']=='Monthly']['Churn'].mean()*100:.1f}% churn rate")
        print(f"   • Customers with engagement score <0.4: {self.df[self.df['Engagement_Score']<0.4]['Churn'].mean()*100:.1f}% churn rate")
        print(f"   • Customers with >90 days since login: {self.df[self.df['Days_Since_Last_Login']>90]['Churn'].mean()*100:.1f}% churn rate")
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\n🔧 Preparing data for training...")
        
        # Select features
        features = [
            'Monthly_Usage_Hours', 'Num_Support_Tickets', 'Days_Since_Last_Login',
            'Monthly_Charges', 'Total_Revenue', 'Discount_Percentage',
            'Age', 'Has_Referral', 'Customer_Service_Rating',
            'Newsletter_Subscribed', 'Mobile_App_User', 'Engagement_Score'
        ]
        
        # Encode categorical variables
        cat_features = ['Gender', 'Income_Level', 'Contract_Length', 'Payment_Method']
        df_encoded = pd.get_dummies(self.df, columns=cat_features, drop_first=True)
        
        # Update feature list
        self.feature_names = [col for col in df_encoded.columns if col != 'Churn' and col != 'Customer_ID']
        
        X = df_encoded[self.feature_names]
        y = self.df['Churn']
        
        # Handle class imbalance
        from sklearn.utils import resample
        if y.sum() < len(y) - y.sum():
            # Upsample minority class
            df_temp = pd.concat([X, y], axis=1)
            churn_majority = df_temp[df_temp['Churn']==0]
            churn_minority = df_temp[df_temp['Churn']==1]
            churn_minority_upsampled = resample(churn_minority,
                                               replace=True,
                                               n_samples=len(churn_majority),
                                               random_state=42)
            df_balanced = pd.concat([churn_majority, churn_minority_upsampled])
            X = df_balanced.drop('Churn', axis=1)
            y = df_balanced['Churn']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data prepared!")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Testing samples: {len(self.X_test)}")
        print(f"   Features: {len(self.feature_names)}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n🌲 TRAINING RANDOM FOREST")
        print("="*50)
        
        # Create model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        
        # Train
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Predict
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n✅ Model trained successfully!")
        print(f"\n📊 Performance Metrics:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, cv=5)
        print(f"\n📈 Cross-Validation Scores:")
        print(f"   Mean: {cv_scores.mean():.4f}")
        print(f"   Std: {cv_scores.std():.4f}")
        
        # Store predictions
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        # Feature importance
        self.show_feature_importance()
        
        # Visualize
        self.evaluate_model()
        
        return self.model
    
    def show_feature_importance(self):
        """Show feature importance"""
        print("\n📊 TOP 10 IMPORTANT FEATURES:")
        print("-"*50)
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        for i in range(min(10, len(self.feature_names))):
            print(f"{i+1}. {self.feature_names[indices[i]]}: {importance[indices[i]]:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        top_indices = indices[:10]
        plt.barh([self.feature_names[i] for i in top_indices], 
                importance[top_indices], color='teal')
        plt.xlabel('Feature Importance')
        plt.title('Random Forest - Top 10 Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self):
        """Evaluate and visualize model performance"""
        print("\n📊 Detailed Classification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=['Retained', 'Churned']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Random Forest Model Performance', fontsize=14, fontweight='bold')
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        axes[0, 1].plot(fpr, tpr, color='blue', linewidth=2,
                       label=f'ROC (AUC = {roc_auc_score(self.y_test, self.y_pred_proba):.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=1)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Probability Distribution
        axes[1, 0].hist(self.y_pred_proba[self.y_test==0], bins=20, alpha=0.5, label='Retained', color='green')
        axes[1, 0].hist(self.y_pred_proba[self.y_test==1], bins=20, alpha=0.5, label='Churned', color='red')
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:8]
        axes[1, 1].barh([self.feature_names[i] for i in indices], importance[indices], color='orange')
        axes[1, 1].set_title('Top 8 Features')
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def tune_hyperparameters(self):
        """Hyperparameter tuning using GridSearchCV"""
        print("\n🔧 HYPERPARAMETER TUNING")
        print("="*50)
        
        # Parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print("\n🔍 Searching for best parameters...")
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"\n✅ Best Parameters: {grid_search.best_params_}")
        print(f"   Best Score: {grid_search.best_score_:.4f}")
        
        # Update model
        self.model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\n📊 Model with tuned parameters:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        return self.model
    
    def predict_churn(self):
        """Predict churn for a new customer"""
        print("\n🔮 PREDICT CUSTOMER CHURN")
        print("="*50)
        
        try:
            print("\nEnter customer information:")
            
            # Usage
            monthly_usage = float(input("   Monthly Usage Hours (10-200): "))
            support_tickets = int(input("   Number of Support Tickets: "))
            days_since_login = int(input("   Days Since Last Login (1-365): "))
            
            # Financial
            monthly_charges = float(input("   Monthly Charges ($20-150): "))
            total_revenue = float(input("   Total Revenue ($200-5000): "))
            discount = float(input("   Discount % (0-40): "))
            
            # Demographics
            age = int(input("   Age (18-80): "))
            gender = input("   Gender (Male/Female): ")
            income = input("   Income Level (Low/Medium/High): ")
            
            # Service
            contract = input("   Contract (Monthly/Yearly/Two_Year): ")
            payment = input("   Payment Method (Credit Card/PayPal/Bank Transfer): ")
            referral = int(input("   Has Referral (0/1): "))
            service_rating = int(input("   Service Rating (1-5): "))
            
            # Engagement
            newsletter = int(input("   Newsletter Subscribed (0/1): "))
            mobile_app = int(input("   Mobile App User (0/1): "))
            engagement = float(input("   Engagement Score (0-1): "))
            
            # Create feature dictionary
            customer_data = {
                'Monthly_Usage_Hours': monthly_usage,
                'Num_Support_Tickets': support_tickets,
                'Days_Since_Last_Login': days_since_login,
                'Monthly_Charges': monthly_charges,
                'Total_Revenue': total_revenue,
                'Discount_Percentage': discount,
                'Age': age,
                'Has_Referral': referral,
                'Customer_Service_Rating': service_rating,
                'Newsletter_Subscribed': newsletter,
                'Mobile_App_User': mobile_app,
                'Engagement_Score': engagement,
                'Gender_Male': 1 if gender == 'Male' else 0,
                'Gender_Female': 1 if gender == 'Female' else 0,
                'Income_Level_Medium': 1 if income == 'Medium' else 0,
                'Income_Level_High': 1 if income == 'High' else 0,
                'Income_Level_Low': 1 if income == 'Low' else 0,
                'Contract_Length_Yearly': 1 if contract == 'Yearly' else 0,
                'Contract_Length_Two_Year': 1 if contract == 'Two_Year' else 0,
                'Contract_Length_Monthly': 1 if contract == 'Monthly' else 0,
                'Payment_Method_PayPal': 1 if payment == 'PayPal' else 0,
                'Payment_Method_Bank Transfer': 1 if payment == 'Bank Transfer' else 0,
                'Payment_Method_Credit Card': 1 if payment == 'Credit Card' else 0
            }
            
            # Create DataFrame
            customer_df = pd.DataFrame([customer_data])
            
            # Ensure all features exist
            for col in self.feature_names:
                if col not in customer_df.columns:
                    customer_df[col] = 0
            
            # Order columns
            customer_df = customer_df[self.feature_names]
            
            # Scale
            customer_scaled = self.scaler.transform(customer_df)
            
            # Predict
            prediction = self.model.predict(customer_scaled)[0]
            probability = self.model.predict_proba(customer_scaled)[0][1]
            
            # Display results
            print("\n" + "="*50)
            print("📊 PREDICTION RESULTS")
            print("="*50)
            
            if prediction == 1:
                print(f"\n   ⚠️ Customer is LIKELY TO CHURN")
                print(f"   Churn Probability: {probability*100:.1f}%")
                print("   🔴 High risk customer!")
            else:
                print(f"\n   ✅ Customer is LIKELY TO STAY")
                print(f"   Retention Probability: {(1-probability)*100:.1f}%")
                print("   🟢 Low risk customer!")
            
            # Risk factors
            print("\n📋 Risk Factors:")
            if monthly_usage < 30:
                print("   • Low usage (under 30 hours/month)")
            if support_tickets > 3:
                print(f"   • High support tickets ({support_tickets})")
            if days_since_login > 90:
                print(f"   • Long time since login ({days_since_login} days)")
            if engagement < 0.4:
                print(f"   • Low engagement score ({engagement})")
            if contract == 'Monthly':
                print("   • Monthly contract (higher churn risk)")
            if service_rating <= 2:
                print(f"   • Poor service rating ({service_rating})")
            
            # Recommendations
            print("\n💡 Recommendations:")
            if probability > 0.7:
                print("   1. Send personalized retention offer")
                print("   2. Schedule customer check-in call")
                print("   3. Offer loyalty rewards")
                print("   4. Investigate issues and improve service")
            elif probability > 0.4:
                print("   1. Send engagement email")
                print("   2. Offer product tips and best practices")
                print("   3. Check in with customer support")
            else:
                print("   1. Continue normal engagement")
                print("   2. Send periodic updates and offers")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    def churn_analysis_report(self):
        """Generate churn analysis report"""
        print("\n" + "="*60)
        print("📊 CUSTOMER CHURN ANALYSIS REPORT")
        print("="*60)
        
        # Churn statistics
        churn_rate = self.df['Churn'].mean() * 100
        print(f"\n📈 Churn Rate: {churn_rate:.1f}%")
        
        # Risk segments
        print("\n🎯 Customer Segments with Highest Churn:")
        segments = [
            ('Monthly Contract', self.df[self.df['Contract_Length']=='Monthly']['Churn'].mean()*100),
            ('Low Engagement', self.df[self.df['Engagement_Score']<0.4]['Churn'].mean()*100),
            ('>3 Support Tickets', self.df[self.df['Num_Support_Tickets']>3]['Churn'].mean()*100),
            ('>90 Days Since Login', self.df[self.df['Days_Since_Last_Login']>90]['Churn'].mean()*100)
        ]
        
        for segment, rate in sorted(segments, key=lambda x: x[1], reverse=True):
            print(f"   • {segment}: {rate:.1f}% churn rate")
        
        print("\n🏆 Retention Opportunities:")
        opportunities = [
            ('Upgrade to Yearly Contract', 'Lower churn rate'),
            ('Improve Engagement Score', 'Increase retention'),
            ('Reduce Support Tickets', 'Better customer service'),
            ('Re-engage Inactive Users', 'Win back lost customers')
        ]
        
        for strategy, benefit in opportunities:
            print(f"   • {strategy}: {benefit}")
    
    def run_full_analysis(self):
        """Run complete churn prediction pipeline"""
        print("\n🎯 WELCOME TO CUSTOMER CHURN PREDICTION")
        print("="*50)
        
        # Create and explore data
        self.create_dataset()
        self.explore_data()
        
        # Prepare data
        self.prepare_data()
        
        # Train Random Forest
        self.train_random_forest()
        
        # Optional hyperparameter tuning
        tune = input("\nPerform hyperparameter tuning? (Y/N): ").upper()
        if tune == 'Y':
            self.tune_hyperparameters()
        
        # Generate report
        self.churn_analysis_report()
        
        # Make predictions
        predict = input("\nPredict churn for a customer? (Y/N): ").upper()
        if predict == 'Y':
            while True:
                self.predict_churn()
                again = input("\nPredict another customer? (Y/N): ").upper()
                if again != 'Y':
                    break
        
        print("\n✅ Analysis Complete! 🌲")


# ========================================
# RUN THE PROGRAM
# ========================================

if __name__ == "__main__":
    predictor = CustomerChurnPredictor()
    predictor.run_full_analysis()