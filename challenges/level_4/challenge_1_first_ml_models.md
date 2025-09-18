# Level 4: Machine Learning Novice

## Challenge 1: Your First ML Models

Welcome to the exciting world of Machine Learning! Build and evaluate your first prediction models.

### Objective
Learn fundamental machine learning concepts by building classification and regression models.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            mean_squared_error, mean_absolute_error, r2_score,
                            classification_report, confusion_matrix)
from sklearn.datasets import load_iris, load_boston
import warnings
warnings.filterwarnings('ignore')

print("🚀 Welcome to Machine Learning!")
print("=====================================")

# Your tasks:
# 1. CLASSIFICATION CHALLENGE
print("\n=== CLASSIFICATION CHALLENGE ===")

# Load and explore the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Iris dataset overview:")
print(iris_df.head())
print(f"Shape: {iris_df.shape}")
print(f"Classes: {iris_df['species'].unique()}")

# Exploratory Data Analysis
print("\nClass distribution:")
print(iris_df['species'].value_counts())

# Feature correlation
print("\nFeature correlations:")
correlation_matrix = iris_df[iris.feature_names].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Iris Features Correlation Matrix')
plt.tight_layout()
plt.show()

# Prepare data for classification
X = iris_df[iris.feature_names]
y = iris_df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train multiple classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

classification_results = {}

print("\n🔬 Training Classification Models:")
for name, classifier in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for SVM and Logistic Regression, original for Random Forest
    if name in ['SVM', 'Logistic Regression']:
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
        y_pred_proba = classifier.predict_proba(X_test_scaled)
    else:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    classification_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# Cross-validation
print(f"\n🔄 Cross-Validation Results:")
for name, classifier in classifiers.items():
    if name in ['SVM', 'Logistic Regression']:
        cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5)
    else:
        cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
    
    print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 2. REGRESSION CHALLENGE
print("\n\n=== REGRESSION CHALLENGE ===")

# Create a regression dataset
np.random.seed(42)
n_samples = 500

# Simulate house price data
house_data = pd.DataFrame({
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'sqft_living': np.random.normal(2000, 800, n_samples),
    'sqft_lot': np.random.normal(7000, 3000, n_samples),
    'floors': np.random.choice([1, 1.5, 2, 3], n_samples, p=[0.4, 0.2, 0.3, 0.1]),
    'age': np.random.randint(0, 100, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'grade': np.random.randint(1, 11, n_samples)
})

# Create realistic price relationships
house_data['price'] = (
    house_data['sqft_living'] * 150 +  # $150 per sqft
    house_data['bedrooms'] * 10000 +   # $10k per bedroom
    house_data['bathrooms'] * 15000 +  # $15k per bathroom
    house_data['floors'] * 5000 +      # $5k per floor
    (100 - house_data['age']) * 1000 + # Depreciation
    house_data['condition'] * 8000 +   # Condition premium
    house_data['grade'] * 12000 +      # Quality premium
    np.random.normal(0, 50000, n_samples)  # Random noise
)

# Ensure positive prices
house_data['price'] = house_data['price'].clip(lower=50000)

print("House price dataset overview:")
print(house_data.describe())

# Regression analysis
X_reg = house_data.drop('price', axis=1)
y_reg = house_data['price']

# Split the data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Feature scaling for regression
scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# Build regression models
regressors = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

regression_results = {}

print("\n🏠 Training Regression Models:")
for name, regressor in regressors.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for Linear Regression and SVR, original for Random Forest
    if name in ['Linear Regression', 'SVR']:
        regressor.fit(X_reg_train_scaled, y_reg_train)
        y_reg_pred = regressor.predict(X_reg_test_scaled)
    else:
        regressor.fit(X_reg_train, y_reg_train)
        y_reg_pred = regressor.predict(X_reg_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)
    
    regression_results[name] = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2
    }
    
    print(f"  MSE: ${mse:,.2f}")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_reg_test, y_reg_pred, alpha=0.6)
    plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'Actual vs Predicted Prices - {name}')
    plt.tight_layout()
    plt.show()

# 3. HYPERPARAMETER TUNING
print("\n\n=== HYPERPARAMETER TUNING ===")

# Tune Random Forest for classification
print("🔧 Tuning Random Forest Classifier...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best cross-validation score: {rf_grid.best_score_:.4f}")

# Test the tuned model
best_rf = rf_grid.best_estimator_
y_pred_tuned = best_rf.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
print(f"Test accuracy with tuned model: {tuned_accuracy:.4f}")

# 4. FEATURE IMPORTANCE
print("\n\n=== FEATURE IMPORTANCE ===")

# Classification feature importance
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

feature_importance_clf = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("Classification Feature Importance:")
print(feature_importance_clf)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_clf, x='importance', y='feature')
plt.title('Feature Importance - Iris Classification')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# Regression feature importance
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_reg_train, y_reg_train)

feature_importance_reg = pd.DataFrame({
    'feature': X_reg.columns,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRegression Feature Importance:")
print(feature_importance_reg)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_reg, x='importance', y='feature')
plt.title('Feature Importance - House Price Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# 5. MODEL PERFORMANCE SUMMARY
print("\n\n=== MODEL PERFORMANCE SUMMARY ===")

print("📊 Classification Results:")
classification_df = pd.DataFrame(classification_results).T
print(classification_df.round(4))

print("\n🏠 Regression Results:")
regression_df = pd.DataFrame(regression_results).T
print(regression_df.round(2))

print("\n✅ Machine Learning Challenge Completed!")
print("\nWhat you've accomplished:")
print("• Built classification models for species prediction")
print("• Created regression models for price prediction")
print("• Evaluated models using appropriate metrics")
print("• Performed hyperparameter tuning")
print("• Analyzed feature importance")
print("• Compared multiple algorithms")
print("\n🎓 You're now a Machine Learning Novice!")
```

### Success Criteria
- Build both classification and regression models
- Properly split data into train/test sets
- Evaluate models using appropriate metrics
- Perform hyperparameter tuning
- Analyze and interpret feature importance
- Compare multiple algorithms effectively

### Learning Objectives
- Understand supervised learning fundamentals
- Master train/test split and cross-validation
- Learn model evaluation metrics
- Practice feature scaling and preprocessing
- Understand bias-variance tradeoff
- Develop model selection skills

---

*Tip: Always start simple, then increase complexity. A well-tuned simple model often beats a poorly tuned complex one!*