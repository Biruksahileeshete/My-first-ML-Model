# ========================================
# HOUSE PRICE PREDICTION
# Simple & Multiple Linear Regression
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class HousePricePredictor:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.simple_model = None
        self.multiple_model = None
        self.X_test_simple = None
        self.y_test_simple = None
        self.y_pred_simple = None
        self.X_test_multi = None
        self.y_test_multi = None
        self.y_pred_multi = None
    
    def create_dataset(self):
        """Create realistic house price dataset"""
        print("\n🏠 Creating house price dataset...")
        np.random.seed(42)
        n = 200  # 200 houses
        
        data = {
            'Size_sqft': np.random.randint(800, 5000, n),
            'Bedrooms': np.random.randint(1, 6, n),
            'Age': np.random.randint(0, 50, n),
            'Location_Score': np.random.uniform(1, 10, n),
            'Price': np.zeros(n)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate price with some randomness
        # Price = Base + (Size * 150) + (Bedrooms * 30000) - (Age * 500) + (Location * 10000) + noise
        for i in range(n):
            base = 50000
            size_effect = df.loc[i, 'Size_sqft'] * 150
            bedroom_effect = df.loc[i, 'Bedrooms'] * 30000
            age_effect = -df.loc[i, 'Age'] * 500
            location_effect = df.loc[i, 'Location_Score'] * 10000
            noise = np.random.normal(0, 20000)
            
            df.loc[i, 'Price'] = base + size_effect + bedroom_effect + age_effect + location_effect + noise
        
        # Keep prices positive and reasonable
        df['Price'] = df['Price'].clip(50000, 800000)
        self.df = df
        
        print(f"✅ Dataset created: {len(df)} houses")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData Statistics:")
        print(df.describe().round(2))
        return df
    
    def simple_regression(self):
        """Train simple linear regression with one feature"""
        print("\n" + "="*50)
        print("📈 SIMPLE LINEAR REGRESSION")
        print("Predicting Price using Size ONLY")
        print("="*50)
        
        # Select one feature
        X = self.df[['Size_sqft']]
        y = self.df['Price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n✅ Model trained!")
        print(f"   Equation: Price = {model.coef_[0]:.0f} × Size + {model.intercept_:.0f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: ${mae:,.0f}")
        print(f"   RMSE: ${rmse:,.0f}")
        
        self.simple_model = model
        
        # Store test data for visualization
        self.X_test_simple = X_test
        self.y_test_simple = y_test
        self.y_pred_simple = y_pred
        
        # Visualize
        self.plot_simple_regression(X_train, y_train, X_test, y_test, y_pred, model)
        
        return model
    
    def plot_simple_regression(self, X_train, y_train, X_test, y_test, y_pred, model):
        """Plot simple regression results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Simple Linear Regression - House Size vs Price', fontsize=14, fontweight='bold')
        
        # 1. Regression Line
        axes[0].scatter(X_train, y_train, alpha=0.5, label='Training Data', color='blue')
        axes[0].scatter(X_test, y_test, alpha=0.5, label='Testing Data', color='green')
        axes[0].plot(X_test, y_pred, color='red', linewidth=2, label='Prediction Line')
        axes[0].set_xlabel('Size (sq ft)')
        axes[0].set_ylabel('Price ($)')
        axes[0].set_title('Regression Line')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted
        axes[1].scatter(y_test, y_pred, alpha=0.6, color='purple')
        axes[1].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Price ($)')
        axes[1].set_ylabel('Predicted Price ($)')
        axes[1].set_title('Actual vs Predicted')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Interpretation:")
        print(f"   For every 1 sq ft increase, price increases by ${model.coef_[0]:.0f}")
        print(f"   A 100 sq ft increase adds ${model.coef_[0] * 100:,.0f} to price")
    
    def multiple_regression(self):
        """Train multiple linear regression with multiple features"""
        print("\n" + "="*50)
        print("📊 MULTIPLE LINEAR REGRESSION")
        print("Predicting Price using ALL features")
        print("="*50)
        
        # Select multiple features
        features = ['Size_sqft', 'Bedrooms', 'Age', 'Location_Score']
        X = self.df[features]
        y = self.df['Price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n✅ Model trained!")
        print(f"\n   R² Score: {r2:.4f}")
        print(f"   MAE: ${mae:,.0f}")
        print(f"   RMSE: ${rmse:,.0f}")
        
        # Show equation
        print(f"\n📝 Equation:")
        print(f"Price = {model.intercept_:.0f}")
        for feature, coef in zip(features, model.coef_):
            print(f"      + ({coef:.2f} × {feature})")
        
        # Feature importance
        print(f"\n📊 Feature Importance (Coefficients):")
        for feature, coef in zip(features, model.coef_):
            print(f"   {feature}: {coef:,.2f}")
        
        self.multiple_model = model
        self.X_test_multi = X_test
        self.y_test_multi = y_test
        self.y_pred_multi = y_pred
        
        self.plot_multiple_regression(y_test, y_pred, features)
        
        return model
    
    def plot_multiple_regression(self, y_test, y_pred, features):
        """Plot multiple regression results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Multiple Linear Regression - Results', fontsize=14, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='purple')
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='orange')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Price ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature Coefficients
        coeffs = self.multiple_model.coef_
        axes[1, 0].bar(features, coeffs, color='teal')
        axes[1, 0].set_title('Feature Coefficients')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Coefficient')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Comparison - Actual vs Predicted Distribution
        axes[1, 1].hist(y_test, bins=20, alpha=0.5, label='Actual', color='blue')
        axes[1, 1].hist(y_pred, bins=20, alpha=0.5, label='Predicted', color='red')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].set_xlabel('Price ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """Compare simple vs multiple regression"""
        if self.simple_model is None or self.multiple_model is None:
            print("\n⚠️ Both models must be trained before comparison.")
            return

        print("\n" + "="*50)
        print("📊 MODEL COMPARISON")
        print("="*50)
        
        # Simple model metrics
        y_pred_simple = self.simple_model.predict(self.X_test_simple)
        r2_simple = r2_score(self.y_test_simple, y_pred_simple)
        mae_simple = mean_absolute_error(self.y_test_simple, y_pred_simple)
        
        # Multiple model metrics
        y_pred_multi = self.multiple_model.predict(self.X_test_multi)
        r2_multi = r2_score(self.y_test_multi, y_pred_multi)
        mae_multi = mean_absolute_error(self.y_test_multi, y_pred_multi)
        
        print(f"\n{'Metric':<15} {'Simple':<15} {'Multiple':<15} {'Improvement':<15}")
        print("-" * 60)
        print(f"{'R² Score':<15} {r2_simple:.4f}{'':10} {r2_multi:.4f}{'':10} {(r2_multi-r2_simple):.4f}")
        print(f"{'MAE':<15} ${mae_simple:,.0f}{'':9} ${mae_multi:,.0f}{'':10} ${mae_simple-mae_multi:,.0f}")
        
        if r2_simple == 0:
            improvement = 0
        else:
            improvement = (r2_multi - r2_simple) / r2_simple * 100
        print(f"\n📈 Multiple Regression improved accuracy by {improvement:.1f}%")
        
        if r2_multi > r2_simple:
            print("✅ Multiple Regression is better - more features help!")
        else:
            print("ℹ️ Simple Regression is better - adding features didn't help")
    
    def make_predictions(self):
        """Make predictions on new houses"""
        if self.simple_model is None or self.multiple_model is None:
            print("\n⚠️ Train the models before making predictions.")
            return

        print("\n🔮 MAKE PREDICTIONS")
        print("="*50)
        
        print("\nChoose model:")
        print("1. Simple Regression (Size only)")
        print("2. Multiple Regression (All features)")
        
        choice = input("\nEnter choice (1/2): ")
        
        if choice == '1':
            print("\n📏 SIMPLE REGRESSION")
            size = float(input("   Size (sq ft): "))
            prediction = self.simple_model.predict([[size]])[0]
            self.display_prediction(size, prediction, model_type="Simple")
            
        elif choice == '2':
            print("\n🏠 MULTIPLE REGRESSION")
            size = float(input("   Size (sq ft): "))
            bedrooms = int(input("   Bedrooms: "))
            age = int(input("   Age (years): "))
            location = float(input("   Location Score (1-10): "))
            
            features = np.array([[size, bedrooms, age, location]])
            prediction = self.multiple_model.predict(features)[0]
            self.display_prediction(features, prediction, model_type="Multiple")
            
        else:
            print("❌ Invalid choice!")
    
    def display_prediction(self, features, price, model_type="Multiple"):
        """Display prediction results"""
        print("\n" + "="*50)
        print("📊 PREDICTION RESULTS")
        print("="*50)
        
        if model_type == "Simple":
            print(f"   House Size: {features} sq ft")
        else:
            print(f"   House Details:")
            print(f"      Size: {features[0][0]} sq ft")
            print(f"      Bedrooms: {int(features[0][1])}")
            print(f"      Age: {int(features[0][2])} years")
            print(f"      Location Score: {features[0][3]:.1f}")
        
        print(f"\n   💰 Predicted Price: ${price:,.0f}")
        
        # Add context
        if price < 200000:
            print("   🏡 This is an affordable house")
        elif price < 400000:
            print("   🏠 This is a mid-range house")
        elif price < 600000:
            print("   🏘️ This is an upscale house")
        else:
            print("   🏰 This is a luxury house")
    
    def run_full_analysis(self):
        """Run complete analysis"""
        print("\n🎯 WELCOME TO HOUSE PRICE PREDICTION")
        print("="*50)
        
        # Create dataset
        self.create_dataset()
        
        # Run both regressions
        self.simple_regression()
        self.multiple_regression()
        
        # Compare models
        self.compare_models()
        
        # Interactive predictions
        print("\n" + "="*50)
        predict_choice = input("\nWant to predict a house price? (Y/N): ").upper()
        
        if predict_choice == 'Y':
            self.make_predictions()
        
        print("\n✅ Analysis Complete! 🏠")


# ========================================
# RUN THE PROGRAM
# ========================================

if __name__ == "__main__":
    predictor = HousePricePredictor()
    predictor.run_full_analysis()