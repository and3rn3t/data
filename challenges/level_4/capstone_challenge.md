# Level 4: Machine Learning Novice

## Capstone Challenge: Complete ML Pipeline Project

Congratulations on mastering the individual ML concepts! Now let's put it all together in a comprehensive project that demonstrates your machine learning expertise.

### Objective

Build a complete end-to-end machine learning pipeline that incorporates all the skills from Level 4: data preparation, feature engineering, model evaluation, and hyperparameter optimization.

### Project: Customer Churn Prediction System

You'll build a system to predict customer churn for a telecommunications company - a real-world business problem that showcases practical ML skills.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

print("🎯 ML Capstone Project: Customer Churn Prediction")
print("=" * 55)

# STEP 1: CREATE REALISTIC CUSTOMER DATASET
print("\n📊 STEP 1: Data Generation and Exploration")
print("-" * 45)

np.random.seed(42)
n_customers = 3000

# Generate comprehensive customer data
print("Creating realistic customer dataset...")
customer_data = pd.DataFrame({
    # Demographics
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(40, 15, n_customers).clip(18, 80),
    'gender': np.random.choice(['M', 'F'], n_customers, p=[0.52, 0.48]),
    'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.84, 0.16]),

    # Service details
    'tenure_months': np.random.exponential(24, n_customers).clip(1, 72),
    'monthly_charges': np.random.normal(65, 20, n_customers).clip(20, 120),
    'total_charges': lambda x: x['tenure_months'] * x['monthly_charges'] +
                              np.random.normal(0, 50, n_customers),

    # Services
    'phone_service': np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1]),
    'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'],
                                     n_customers, p=[0.5, 0.4, 0.1]),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'],
                                       n_customers, p=[0.35, 0.45, 0.2]),
    'online_security': np.random.choice(['Yes', 'No', 'No internet service'],
                                      n_customers, p=[0.3, 0.5, 0.2]),
    'tech_support': np.random.choice(['Yes', 'No', 'No internet service'],
                                   n_customers, p=[0.35, 0.45, 0.2]),

    # Contract details
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'],
                               n_customers, p=[0.55, 0.21, 0.24]),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check',
                                      'Bank transfer (automatic)', 'Credit card (automatic)'],
                                     n_customers, p=[0.34, 0.19, 0.22, 0.25]),

    # Customer satisfaction metrics
    'customer_service_calls': np.random.poisson(2, n_customers),
    'late_payments': np.random.poisson(1, n_customers),
})

# Calculate total charges properly
customer_data['total_charges'] = (customer_data['tenure_months'] *
                                customer_data['monthly_charges'] +
                                np.random.normal(0, 50, n_customers)).clip(0)

# Create realistic churn based on business logic
print("Creating realistic churn patterns...")
churn_probability = np.zeros(n_customers)

# Contract type impact (month-to-month more likely to churn)
churn_probability += np.where(customer_data['contract'] == 'Month-to-month', 0.3, 0.1)

# Tenure impact (newer customers more likely to churn)
tenure_factor = (72 - customer_data['tenure_months']) / 72 * 0.2
churn_probability += tenure_factor

# Service issues impact
churn_probability += customer_data['customer_service_calls'] * 0.05
churn_probability += customer_data['late_payments'] * 0.08

# Monthly charges impact (high charges increase churn)
charge_factor = (customer_data['monthly_charges'] - 20) / 100 * 0.15
churn_probability += charge_factor

# Senior citizens slightly more likely to churn
churn_probability += customer_data['senior_citizen'] * 0.05

# Add some randomness
churn_probability += np.random.normal(0, 0.1, n_customers)

# Convert to binary churn
customer_data['churn'] = (churn_probability > 0.35).astype(int)

print(f"Dataset created with {n_customers:,} customers")
print(f"Churn rate: {customer_data['churn'].mean():.1%}")

# STEP 2: EXPLORATORY DATA ANALYSIS
print("\n🔍 STEP 2: Exploratory Data Analysis")
print("-" * 40)

print("Dataset shape:", customer_data.shape)
print("\nData types:")
print(customer_data.dtypes.value_counts())

print("\nChurn distribution:")
churn_counts = customer_data['churn'].value_counts()
print(churn_counts)
print(f"Churn rate: {churn_counts[1] / len(customer_data):.1%}")

# Key insights visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Customer Churn Analysis', fontsize=16, fontweight='bold')

# Churn by contract type
contract_churn = customer_data.groupby('contract')['churn'].mean()
axes[0,0].bar(contract_churn.index, contract_churn.values, color='skyblue')
axes[0,0].set_title('Churn Rate by Contract Type')
axes[0,0].set_ylabel('Churn Rate')
axes[0,0].tick_params(axis='x', rotation=45)

# Tenure vs Churn
axes[0,1].boxplot([customer_data[customer_data['churn']==0]['tenure_months'],
                   customer_data[customer_data['churn']==1]['tenure_months']],
                  labels=['Retained', 'Churned'])
axes[0,1].set_title('Tenure Distribution by Churn')
axes[0,1].set_ylabel('Tenure (Months)')

# Monthly charges vs Churn
axes[1,0].boxplot([customer_data[customer_data['churn']==0]['monthly_charges'],
                   customer_data[customer_data['churn']==1]['monthly_charges']],
                  labels=['Retained', 'Churned'])
axes[1,0].set_title('Monthly Charges by Churn')
axes[1,0].set_ylabel('Monthly Charges ($)')

# Customer service calls impact
service_churn = customer_data.groupby('customer_service_calls')['churn'].mean()
axes[1,1].plot(service_churn.index, service_churn.values, marker='o', color='red')
axes[1,1].set_title('Churn Rate vs Service Calls')
axes[1,1].set_xlabel('Customer Service Calls')
axes[1,1].set_ylabel('Churn Rate')

plt.tight_layout()
plt.show()

# STEP 3: FEATURE ENGINEERING
print("\n🔧 STEP 3: Feature Engineering")
print("-" * 35)

# Create a copy for feature engineering
fe_data = customer_data.copy()

print("Creating engineered features...")

# 1. Numerical feature engineering
fe_data['charges_per_month'] = fe_data['total_charges'] / fe_data['tenure_months']
fe_data['tenure_years'] = fe_data['tenure_months'] / 12
fe_data['is_new_customer'] = (fe_data['tenure_months'] <= 6).astype(int)
fe_data['high_value_customer'] = (fe_data['monthly_charges'] > fe_data['monthly_charges'].quantile(0.75)).astype(int)

# 2. Service complexity score
service_cols = ['phone_service', 'multiple_lines', 'internet_service', 'online_security', 'tech_support']
fe_data['service_complexity'] = fe_data[service_cols].apply(
    lambda row: sum([1 for val in row if val == 'Yes']), axis=1
)

# 3. Risk indicators
fe_data['payment_risk'] = (fe_data['payment_method'] == 'Electronic check').astype(int)
fe_data['contract_risk'] = (fe_data['contract'] == 'Month-to-month').astype(int)
fe_data['support_burden'] = fe_data['customer_service_calls'] + fe_data['late_payments']

# 4. Categorical encoding
print("Encoding categorical variables...")
categorical_cols = ['gender', 'contract', 'payment_method', 'internet_service']

# Label encoding for binary categories
label_encoder = LabelEncoder()
fe_data['gender_encoded'] = label_encoder.fit_transform(fe_data['gender'])

# One-hot encoding for multi-class categories
for col in ['contract', 'payment_method', 'internet_service']:
    dummies = pd.get_dummies(fe_data[col], prefix=col, drop_first=True)
    fe_data = pd.concat([fe_data, dummies], axis=1)

print(f"Features after engineering: {fe_data.shape[1]} columns")

# Select final feature set
feature_cols = [col for col in fe_data.columns if col not in
               ['customer_id', 'churn'] + categorical_cols +
               ['phone_service', 'multiple_lines', 'online_security', 'tech_support', 'paperless_billing']]

X = fe_data[feature_cols]
y = fe_data['churn']

print(f"Final feature set: {len(feature_cols)} features")
print("Features:", feature_cols)

# STEP 4: MODEL EVALUATION FRAMEWORK
print("\n🎯 STEP 4: Model Evaluation and Selection")
print("-" * 45)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)
}

# Evaluate models using cross-validation
print("\nModel Comparison (5-fold Cross-Validation):")
print("-" * 50)
results = {}

for name, model in models.items():
    if name == 'SVM':
        # Use scaled data for SVM
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    results[name] = cv_scores
    print(f"{name:20} | ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# STEP 5: HYPERPARAMETER OPTIMIZATION
print("\n🔧 STEP 5: Hyperparameter Optimization")
print("-" * 40)

# Select best performing model for optimization (Random Forest in this case)
print("Optimizing Random Forest hyperparameters...")

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

print(f"\nBest parameters: {rf_grid.best_params_}")
print(f"Best CV ROC-AUC: {rf_grid.best_score_:.4f}")

# STEP 6: FINAL MODEL EVALUATION
print("\n📈 STEP 6: Final Model Evaluation")
print("-" * 38)

# Get best model
best_model = rf_grid.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Comprehensive evaluation
print("Final Model Performance on Test Set:")
print("-" * 40)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance analysis
print("\nTop 10 Most Important Features:")
print("-" * 35)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"{i+1:2d}. {row['feature']:25} | {row['importance']:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Business insights
print("\n💼 Business Insights and Recommendations:")
print("-" * 45)
print("1. Contract type is a strong predictor - focus retention on month-to-month customers")
print("2. Customer service interactions are key - improve first-call resolution")
print("3. New customers (< 6 months) need special attention and onboarding")
print("4. High monthly charges increase churn risk - consider value proposition")
print("5. Payment method matters - electronic check users need engagement")

print(f"\n🎉 Congratulations! You've built a complete ML pipeline!")
print(f"Your model can identify {recall_score(y_test, y_pred):.1%} of churning customers")
print(f"with {precision_score(y_test, y_pred):.1%} precision.")
```

### Success Criteria

✅ **Complete Pipeline**: Built end-to-end ML system from data generation to business insights
✅ **Feature Engineering**: Created meaningful features that improve model performance
✅ **Model Comparison**: Evaluated multiple algorithms with proper cross-validation
✅ **Hyperparameter Tuning**: Optimized best model for maximum performance
✅ **Business Value**: Translated model results into actionable business recommendations

### Learning Objectives

- **Integration Skills**: Combine all Level 4 concepts in a cohesive project
- **Real-world Application**: Work with realistic business problem and dataset
- **End-to-end Thinking**: Consider the complete ML lifecycle
- **Business Communication**: Translate technical results to business insights

### Reflection Questions

1. **Data Quality**: What data quality issues did you encounter and how did you handle them?
2. **Feature Impact**: Which engineered features had the most impact on model performance?
3. **Model Selection**: Why did certain models perform better than others?
4. **Business Value**: How would you present these results to business stakeholders?
5. **Next Steps**: What would be your recommendations for deploying this model?

### Advanced Extensions

If you want to take this further:

1. **Model Interpretation**: Use SHAP or LIME to explain individual predictions
2. **Cost-Sensitive Learning**: Incorporate business costs of false positives/negatives
3. **Deployment Pipeline**: Create a simple web API for real-time predictions
4. **A/B Testing Framework**: Design experiments to validate model impact
5. **Monitoring System**: Plan how to detect model drift in production

---

**🎯 Congratulations on completing Level 4: Machine Learning Novice!**

You've mastered the fundamentals of machine learning and can now build robust, well-evaluated models that deliver real business value. You're ready to tackle more advanced ML challenges in Level 5!

_Pro Tip: This capstone project demonstrates the type of end-to-end thinking that makes great data scientists. Keep building projects that connect technical skills to business outcomes!_
