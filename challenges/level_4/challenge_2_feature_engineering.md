# Level 4: Machine Learning Novice

## Challenge 2: Feature Engineering Mastery

Master the art of feature engineering - the secret sauce that transforms raw data into powerful predictive features.

### Objective

Learn advanced feature engineering techniques including creation, selection, transformation, and encoding to dramatically improve model performance.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                 LabelEncoder, OneHotEncoder, PolynomialFeatures)
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression, RFE,
                                     SelectFromModel, chi2)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

print("🔧 Feature Engineering Mastery Challenge")
print("=" * 45)

# Create comprehensive dataset with various feature types
np.random.seed(42)
n_samples = 1000

print("📊 Creating Rich Dataset for Feature Engineering...")

# Generate customer dataset with mixed feature types
data = pd.DataFrame({
    # Numerical features
    'age': np.random.normal(35, 12, n_samples).clip(18, 80),
    'income': np.random.lognormal(10.5, 0.8, n_samples),
    'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
    'years_employed': np.random.exponential(5, n_samples).clip(0, 40),
    'debt_to_income': np.random.beta(2, 5, n_samples) * 0.8,

    # Categorical features
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'],
                                n_samples, p=[0.4, 0.35, 0.2, 0.05]),
    'job_category': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail', 'Other'],
                                   n_samples, p=[0.2, 0.15, 0.15, 0.25, 0.25]),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'],
                                     n_samples, p=[0.4, 0.45, 0.15]),

    # Date/Time features
    'account_created': pd.date_range('2020-01-01', '2024-01-01', periods=n_samples),

    # Boolean features
    'owns_home': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
    'has_savings': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),

    # Text-like features (simplified)
    'city_size': np.random.choice(['Small', 'Medium', 'Large'], n_samples, p=[0.3, 0.4, 0.3])
})

# Create target variable with realistic relationships
# Loan approval based on multiple factors
loan_score = (
    (data['credit_score'] - 300) / 550 * 0.4 +  # Credit score impact
    (data['income'] / data['income'].max()) * 0.25 +  # Income impact
    (1 - data['debt_to_income']) * 0.15 +  # Lower debt is better
    data['owns_home'].astype(int) * 0.1 +  # Home ownership
    data['has_savings'].astype(int) * 0.1 +  # Savings
    np.random.normal(0, 0.1, n_samples)  # Random noise
)

data['loan_approved'] = (loan_score > 0.6).astype(int)
data['loan_amount'] = np.where(
    data['loan_approved'] == 1,
    data['income'] * np.random.uniform(0.1, 0.4, n_samples),
    0
)

# Introduce some missing values for realistic scenario
missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
data.loc[missing_indices, 'years_employed'] = np.nan

missing_indices_2 = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
data.loc[missing_indices_2, 'income'] = np.nan

print(f"Dataset created: {data.shape}")
print(f"Target distribution - Approved: {data['loan_approved'].mean():.2%}")
print(f"Missing values: {data.isnull().sum().sum()}")
print("\nDataset Overview:")
print(data.head())

# CHALLENGE 1: HANDLING MISSING VALUES
print("\n" + "=" * 50)
print("🔍 CHALLENGE 1: HANDLING MISSING VALUES")
print("=" * 50)

print("\nMissing value analysis:")
missing_summary = pd.DataFrame({
    'Missing Count': data.isnull().sum(),
    'Missing Percentage': (data.isnull().sum() / len(data) * 100).round(2)
}).sort_values('Missing Count', ascending=False)

print(missing_summary[missing_summary['Missing Count'] > 0])

# Visualize missing patterns
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Value Pattern')
plt.tight_layout()
plt.show()

# Multiple imputation strategies
print("\n🔧 Applying Different Imputation Strategies:")

# Strategy 1: Simple imputation
data_simple = data.copy()
numeric_cols = data_simple.select_dtypes(include=[np.number]).columns
simple_imputer = SimpleImputer(strategy='median')
data_simple[numeric_cols] = simple_imputer.fit_transform(data_simple[numeric_cols])

# Strategy 2: KNN imputation
data_knn = data.copy()
knn_imputer = KNNImputer(n_neighbors=5)
data_knn[numeric_cols] = knn_imputer.fit_transform(data_knn[numeric_cols])

print("✅ Missing values handled with multiple strategies")

# CHALLENGE 2: CATEGORICAL ENCODING
print("\n" + "=" * 50)
print("📝 CHALLENGE 2: CATEGORICAL ENCODING")
print("=" * 50)

# Work with KNN-imputed data for consistency
data_encoded = data_knn.copy()

# One-Hot Encoding for nominal variables
nominal_cols = ['job_category', 'marital_status', 'city_size']
data_encoded = pd.get_dummies(data_encoded, columns=nominal_cols, prefix=nominal_cols)

# Ordinal encoding for education (has natural order)
education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
data_encoded['education_encoded'] = data_encoded['education'].map(education_mapping)

# Binary encoding for boolean features (already 0/1 but let's be explicit)
data_encoded['owns_home_encoded'] = data_encoded['owns_home'].astype(int)
data_encoded['has_savings_encoded'] = data_encoded['has_savings'].astype(int)

print("Categorical encoding results:")
print(f"Original features: {data.shape[1]}")
print(f"After encoding: {data_encoded.shape[1]}")
print(f"New feature columns:")
new_cols = [col for col in data_encoded.columns if col not in data.columns]
for col in new_cols[:10]:  # Show first 10
    print(f"  • {col}")

# CHALLENGE 3: FEATURE CREATION FROM DATES
print("\n" + "=" * 50)
print("📅 CHALLENGE 3: DATE/TIME FEATURE ENGINEERING")
print("=" * 50)

# Extract temporal features
data_encoded['account_age_days'] = (pd.Timestamp.now() - data_encoded['account_created']).dt.days
data_encoded['account_age_years'] = data_encoded['account_age_days'] / 365.25
data_encoded['account_created_year'] = data_encoded['account_created'].dt.year
data_encoded['account_created_month'] = data_encoded['account_created'].dt.month
data_encoded['account_created_quarter'] = data_encoded['account_created'].dt.quarter
data_encoded['is_weekend_signup'] = data_encoded['account_created'].dt.weekday.isin([5, 6]).astype(int)

# Cyclical encoding for monthly patterns
data_encoded['month_sin'] = np.sin(2 * np.pi * data_encoded['account_created_month'] / 12)
data_encoded['month_cos'] = np.cos(2 * np.pi * data_encoded['account_created_month'] / 12)

print("Date-based features created:")
date_features = ['account_age_days', 'account_age_years', 'account_created_year',
                'account_created_month', 'is_weekend_signup', 'month_sin', 'month_cos']
for feat in date_features:
    print(f"  • {feat}: {data_encoded[feat].dtype}")

# CHALLENGE 4: NUMERICAL FEATURE ENGINEERING
print("\n" + "=" * 50)
print("🔢 CHALLENGE 4: NUMERICAL FEATURE TRANSFORMATIONS")
print("=" * 50)

# Create interaction features
data_encoded['income_per_year_employed'] = data_encoded['income'] / (data_encoded['years_employed'] + 1)
data_encoded['credit_to_age_ratio'] = data_encoded['credit_score'] / data_encoded['age']
data_encoded['debt_income_interaction'] = data_encoded['debt_to_income'] * data_encoded['income']

# Binning continuous variables
data_encoded['age_group'] = pd.cut(data_encoded['age'],
                                  bins=[0, 25, 35, 50, 100],
                                  labels=['Young', 'Adult', 'Middle', 'Senior'])

data_encoded['income_quartile'] = pd.qcut(data_encoded['income'],
                                         q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Log transformations for skewed features
data_encoded['log_income'] = np.log1p(data_encoded['income'])
data_encoded['sqrt_years_employed'] = np.sqrt(data_encoded['years_employed'])

# Polynomial features (selected pairs to avoid explosion)
poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
key_numeric_features = ['age', 'credit_score', 'debt_to_income']
poly_array = poly_features.fit_transform(data_encoded[key_numeric_features])
poly_feature_names = poly_features.get_feature_names_out(key_numeric_features)

# Add polynomial features to dataset
poly_df = pd.DataFrame(poly_array, columns=poly_feature_names, index=data_encoded.index)
for col in poly_df.columns:
    if col not in key_numeric_features:  # Don't duplicate original features
        data_encoded[f'poly_{col}'] = poly_df[col]

print("Numerical feature engineering completed:")
print(f"  • Interaction features: 3")
print(f"  • Binned features: 2")
print(f"  • Transformed features: 3")
print(f"  • Polynomial features: {len([col for col in data_encoded.columns if col.startswith('poly_')])}")

# CHALLENGE 5: FEATURE SCALING
print("\n" + "=" * 50)
print("⚖️ CHALLENGE 5: FEATURE SCALING STRATEGIES")
print("=" * 50)

# Prepare final feature set for modeling
feature_cols = [col for col in data_encoded.columns if col not in
               ['loan_approved', 'loan_amount', 'account_created', 'education',
                'job_category', 'marital_status', 'city_size', 'owns_home', 'has_savings',
                'age_group', 'income_quartile']]

X = data_encoded[feature_cols]
y_classification = data_encoded['loan_approved']
y_regression = data_encoded['loan_amount']

# Split data
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_classification, test_size=0.3, random_state=42, stratify=y_classification
)

_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.3, random_state=42
)

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features: {len(feature_cols)}")

# Compare different scaling methods
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

scaling_results = {}

print("\n🔬 Testing Different Scaling Methods:")
for scaler_name, scaler in scalers.items():
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test with logistic regression (sensitive to scaling)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_class_train)
    accuracy = lr.score(X_test_scaled, y_class_test)

    scaling_results[scaler_name] = accuracy
    print(f"  {scaler_name}: {accuracy:.4f} accuracy")

# CHALLENGE 6: FEATURE SELECTION
print("\n" + "=" * 50)
print("🎯 CHALLENGE 6: FEATURE SELECTION METHODS")
print("=" * 50)

# Use StandardScaler for feature selection
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 1: Statistical feature selection
print("1️⃣ Statistical Feature Selection (SelectKBest):")
selector_stats = SelectKBest(score_func=f_classif, k=15)
X_train_stats = selector_stats.fit_transform(X_train_scaled, y_class_train)
X_test_stats = selector_stats.transform(X_test_scaled)

selected_features_stats = [feature_cols[i] for i in selector_stats.get_support(indices=True)]
print(f"   Selected {len(selected_features_stats)} features")

# Method 2: L1 regularization (Lasso)
print("\n2️⃣ L1 Regularization Feature Selection:")
lasso_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
X_train_lasso = lasso_selector.fit_transform(X_train_scaled, y_class_train)
X_test_lasso = lasso_selector.transform(X_test_scaled)

selected_features_lasso = [feature_cols[i] for i in lasso_selector.get_support(indices=True)]
print(f"   Selected {len(selected_features_lasso)} features")

# Method 3: Recursive Feature Elimination
print("\n3️⃣ Recursive Feature Elimination:")
rfe_selector = RFE(LogisticRegression(random_state=42, max_iter=1000), n_features_to_select=12)
X_train_rfe = rfe_selector.fit_transform(X_train_scaled, y_class_train)
X_test_rfe = rfe_selector.transform(X_test_scaled)

selected_features_rfe = [feature_cols[i] for i in rfe_selector.get_support(indices=True)]
print(f"   Selected {len(selected_features_rfe)} features")

# Method 4: Tree-based feature importance
print("\n4️⃣ Tree-based Feature Importance:")
rf_selector = SelectFromModel(RandomForestClassifier(random_state=42, n_estimators=100))
X_train_rf = rf_selector.fit_transform(X_train_scaled, y_class_train)
X_test_rf = rf_selector.transform(X_test_scaled)

selected_features_rf = [feature_cols[i] for i in rf_selector.get_support(indices=True)]
print(f"   Selected {len(selected_features_rf)} features")

# Compare feature selection methods
print("\n📊 Feature Selection Comparison:")
selection_results = {}
methods = {
    'All Features': (X_train_scaled, X_test_scaled),
    'Statistical': (X_train_stats, X_test_stats),
    'Lasso L1': (X_train_lasso, X_test_lasso),
    'RFE': (X_train_rfe, X_test_rfe),
    'Random Forest': (X_train_rf, X_test_rf)
}

for method_name, (X_tr, X_te) in methods.items():
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_tr, y_class_train)
    accuracy = lr.score(X_te, y_class_test)
    selection_results[method_name] = {
        'accuracy': accuracy,
        'n_features': X_tr.shape[1]
    }
    print(f"  {method_name}: {accuracy:.4f} accuracy with {X_tr.shape[1]} features")

# CHALLENGE 7: DIMENSIONALITY REDUCTION
print("\n" + "=" * 50)
print("📉 CHALLENGE 7: DIMENSIONALITY REDUCTION")
print("=" * 50)

# Principal Component Analysis
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA Results:")
print(f"  Original features: {X_train_scaled.shape[1]}")
print(f"  PCA components: {X_train_pca.shape[1]}")
print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA: Cumulative Explained Variance')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(1, min(11, len(pca.explained_variance_ratio_) + 1)),
        pca.explained_variance_ratio_[:10])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Individual Component Variance')
plt.tight_layout()
plt.show()

# Test PCA performance
lr_pca = LogisticRegression(random_state=42, max_iter=1000)
lr_pca.fit(X_train_pca, y_class_train)
pca_accuracy = lr_pca.score(X_test_pca, y_class_test)
print(f"  PCA Model Accuracy: {pca_accuracy:.4f}")

# CHALLENGE 8: FEATURE ENGINEERING PIPELINE EVALUATION
print("\n" + "=" * 50)
print("🏆 CHALLENGE 8: COMPREHENSIVE EVALUATION")
print("=" * 50)

# Create feature engineering summary
pipeline_results = {
    'Original Dataset': len(data.columns) - 2,  # Exclude targets
    'After Missing Value Handling': len(data_knn.columns) - 2,
    'After Categorical Encoding': len([c for c in data_encoded.columns
                                     if c not in ['loan_approved', 'loan_amount']]),
    'Final Feature Set': len(feature_cols)
}

print("📈 Feature Engineering Pipeline Summary:")
for stage, count in pipeline_results.items():
    print(f"  {stage}: {count} features")

# Model performance comparison
print(f"\n🎯 Model Performance Impact:")
comparison_data = []

# Baseline: minimal features
minimal_features = ['age', 'income', 'credit_score', 'debt_to_income']
X_minimal = data_encoded[minimal_features].fillna(0)
X_min_train, X_min_test = train_test_split(X_minimal, test_size=0.3, random_state=42)

scaler_min = StandardScaler()
X_min_train_scaled = scaler_min.fit_transform(X_min_train)
X_min_test_scaled = scaler_min.transform(X_min_test)

lr_minimal = LogisticRegression(random_state=42)
lr_minimal.fit(X_min_train_scaled, y_class_train)
minimal_accuracy = lr_minimal.score(X_min_test_scaled, y_class_test)

comparison_data.append({
    'Approach': 'Minimal Features (4)',
    'Accuracy': minimal_accuracy,
    'Features': len(minimal_features)
})

# Best engineered features
lr_best = LogisticRegression(random_state=42, max_iter=1000)
lr_best.fit(X_train_lasso, y_class_train)  # Use Lasso-selected features
best_accuracy = lr_best.score(X_test_lasso, y_class_test)

comparison_data.append({
    'Approach': f'Engineered Features ({len(selected_features_lasso)})',
    'Accuracy': best_accuracy,
    'Features': len(selected_features_lasso)
})

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df)

# Feature importance analysis
print(f"\n🔍 Top Engineered Features (Lasso Selected):")
feature_coefs = pd.DataFrame({
    'feature': selected_features_lasso,
    'coefficient': np.abs(lr_best.coef_[0])
}).sort_values('coefficient', ascending=False)

print(feature_coefs.head(10))

# Visualization of feature importance
plt.figure(figsize=(12, 8))
top_features = feature_coefs.head(15)
sns.barplot(data=top_features, y='feature', x='coefficient')
plt.title('Top 15 Most Important Engineered Features')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.show()

print("\n✅ Feature Engineering Challenge Completed!")
print("\nWhat you've mastered:")
print("• Multiple missing value imputation strategies")
print("• Categorical encoding techniques (one-hot, ordinal, binary)")
print("• Date/time feature extraction and cyclical encoding")
print("• Numerical transformations and interaction features")
print("• Feature scaling methods comparison")
print("• Statistical and model-based feature selection")
print("• Dimensionality reduction with PCA")
print("• End-to-end feature engineering pipeline")

improvement = ((best_accuracy - minimal_accuracy) / minimal_accuracy) * 100
print(f"\n🎉 Performance Improvement: {improvement:.1f}% boost from feature engineering!")
print("Feature engineering is truly the 'secret sauce' of machine learning! 🔥")
```

### Success Criteria

- Implement multiple missing value handling strategies
- Apply various categorical encoding techniques appropriately
- Create meaningful features from dates and numerical variables
- Compare and select optimal feature scaling methods
- Use statistical and model-based feature selection techniques
- Apply dimensionality reduction effectively
- Build comprehensive feature engineering pipelines
- Demonstrate significant model performance improvements

### Learning Objectives

- Master the full spectrum of feature engineering techniques
- Understand when and how to apply different encoding strategies
- Learn to create domain-specific interaction and transformation features
- Practice feature selection to avoid curse of dimensionality
- Develop intuition for feature quality and model impact
- Build reusable feature engineering pipelines

---

_Pro tip: Feature engineering is where domain knowledge meets data science - the best features often come from understanding the business problem deeply!_
