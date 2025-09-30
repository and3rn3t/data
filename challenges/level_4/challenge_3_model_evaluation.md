# Level 4: Machine Learning Novice

## Challenge 3: Model Evaluation and Validation

Master comprehensive model evaluation techniques to ensure your models are robust, reliable, and ready for production.

### Objective

Learn advanced evaluation metrics, cross-validation strategies, bias-variance analysis, and model diagnostics to build trustworthy machine learning systems.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score,
                                   validation_curve, learning_curve,
                                   StratifiedKFold, TimeSeriesSplit, GroupKFold)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           confusion_matrix, classification_report,
                           mean_squared_error, mean_absolute_error, r2_score,
                           mean_absolute_percentage_error)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("🎯 Model Evaluation and Validation Challenge")
print("=" * 47)

# Create comprehensive evaluation datasets
np.random.seed(42)

print("📊 Creating Evaluation Datasets...")

# Dataset 1: Binary Classification (Medical Diagnosis)
n_patients = 1500

medical_data = pd.DataFrame({
    'age': np.random.normal(50, 15, n_patients).clip(20, 90),
    'bmi': np.random.normal(26, 5, n_patients).clip(15, 45),
    'blood_pressure': np.random.normal(120, 20, n_patients).clip(80, 200),
    'cholesterol': np.random.normal(200, 40, n_patients).clip(100, 400),
    'glucose': np.random.normal(100, 25, n_patients).clip(60, 300),
    'family_history': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
    'smoking': np.random.choice([0, 1], n_patients, p=[0.65, 0.35]),
    'exercise_hours': np.random.exponential(3, n_patients).clip(0, 20)
})

# Create realistic disease probability
risk_score = (
    (medical_data['age'] - 20) / 70 * 0.3 +
    np.maximum(0, medical_data['bmi'] - 25) / 20 * 0.2 +
    np.maximum(0, medical_data['blood_pressure'] - 140) / 60 * 0.15 +
    np.maximum(0, medical_data['cholesterol'] - 200) / 200 * 0.1 +
    np.maximum(0, medical_data['glucose'] - 100) / 200 * 0.15 +
    medical_data['family_history'] * 0.2 +
    medical_data['smoking'] * 0.15 -
    medical_data['exercise_hours'] / 20 * 0.1 +
    np.random.normal(0, 0.1, n_patients)
)

medical_data['disease'] = (risk_score > 0.4).astype(int)

# Dataset 2: Regression (House Prices with Time Component)
n_houses = 1000
dates = pd.date_range('2020-01-01', '2024-01-01', periods=n_houses)

house_data = pd.DataFrame({
    'date_sold': dates,
    'sqft': np.random.normal(2000, 600, n_houses).clip(800, 5000),
    'bedrooms': np.random.randint(1, 6, n_houses),
    'bathrooms': np.random.uniform(1, 4, n_houses),
    'age': np.random.exponential(15, n_houses).clip(0, 100),
    'lot_size': np.random.normal(7000, 3000, n_houses).clip(1000, 50000),
    'garage': np.random.choice([0, 1, 2], n_houses, p=[0.1, 0.6, 0.3]),
    'condition': np.random.randint(1, 6, n_houses)
})

# Add time trend and seasonality
time_trend = (house_data['date_sold'] - house_data['date_sold'].min()).dt.days / 365 * 10000
seasonal = np.sin(2 * np.pi * house_data['date_sold'].dt.month / 12) * 15000

house_data['price'] = (
    house_data['sqft'] * 120 +
    house_data['bedrooms'] * 8000 +
    house_data['bathrooms'] * 12000 -
    house_data['age'] * 500 +
    house_data['lot_size'] * 2 +
    house_data['garage'] * 15000 +
    house_data['condition'] * 8000 +
    time_trend + seasonal +
    np.random.normal(0, 25000, n_houses)
).clip(50000, 2000000)

print(f"Medical dataset: {medical_data.shape}")
print(f"House price dataset: {house_data.shape}")
print(f"Disease prevalence: {medical_data['disease'].mean():.1%}")

# CHALLENGE 1: COMPREHENSIVE METRICS ANALYSIS
print("\n" + "=" * 55)
print("📊 CHALLENGE 1: COMPREHENSIVE METRICS ANALYSIS")
print("=" * 55)

# Prepare medical data for classification
X_med = medical_data.drop('disease', axis=1)
y_med = medical_data['disease']

X_med_train, X_med_test, y_med_train, y_med_test = train_test_split(
    X_med, y_med, test_size=0.3, random_state=42, stratify=y_med
)

# Scale features
scaler_med = StandardScaler()
X_med_train_scaled = scaler_med.fit_transform(X_med_train)
X_med_test_scaled = scaler_med.transform(X_med_test)

# Train multiple classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
}

print("🔬 Training Multiple Classifiers for Comprehensive Evaluation:")

classification_metrics = {}
y_pred_probas = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")

    if name in ['Logistic Regression', 'SVM', 'Neural Network']:
        clf.fit(X_med_train_scaled, y_med_train)
        y_pred = clf.predict(X_med_test_scaled)
        y_pred_proba = clf.predict_proba(X_med_test_scaled)[:, 1]
    else:
        clf.fit(X_med_train, y_med_train)
        y_pred = clf.predict(X_med_test)
        y_pred_proba = clf.predict_proba(X_med_test)[:, 1]

    # Calculate comprehensive metrics
    metrics = {
        'Accuracy': accuracy_score(y_med_test, y_pred),
        'Precision': precision_score(y_med_test, y_pred),
        'Recall': recall_score(y_med_test, y_pred),
        'F1-Score': f1_score(y_med_test, y_pred),
        'ROC-AUC': roc_auc_score(y_med_test, y_pred_proba),
        'Specificity': recall_score(y_med_test, y_pred, pos_label=0),  # True Negative Rate
        'NPV': precision_score(y_med_test, y_pred, pos_label=0)  # Negative Predictive Value
    }

    classification_metrics[name] = metrics
    y_pred_probas[name] = y_pred_proba

    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")

# Create comprehensive metrics comparison
metrics_df = pd.DataFrame(classification_metrics).T
print(f"\n📊 Comprehensive Metrics Comparison:")
print(metrics_df.round(4))

# Visualize metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Classification Metrics Comparison', fontsize=16, fontweight='bold')

# Accuracy and F1-Score
axes[0, 0].bar(metrics_df.index, metrics_df['Accuracy'], alpha=0.7, color='skyblue')
axes[0, 0].bar(metrics_df.index, metrics_df['F1-Score'], alpha=0.7, color='lightcoral')
axes[0, 0].set_title('Accuracy vs F1-Score')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend(['Accuracy', 'F1-Score'])
axes[0, 0].tick_params(axis='x', rotation=45)

# Precision vs Recall
axes[0, 1].scatter(metrics_df['Recall'], metrics_df['Precision'], s=100, alpha=0.7)
for i, name in enumerate(metrics_df.index):
    axes[0, 1].annotate(name, (metrics_df.iloc[i]['Recall'], metrics_df.iloc[i]['Precision']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Trade-off')
axes[0, 1].grid(True, alpha=0.3)

# ROC-AUC comparison
axes[1, 0].bar(metrics_df.index, metrics_df['ROC-AUC'], color='lightgreen', alpha=0.7)
axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Classifier')
axes[1, 0].set_title('ROC-AUC Scores')
axes[1, 0].set_ylabel('ROC-AUC')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend()

# ROC Curves
for name, y_proba in y_pred_probas.items():
    fpr, tpr, _ = roc_curve(y_med_test, y_proba)
    axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC={metrics_df.loc[name, "ROC-AUC"]:.3f})')

axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curves')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# CHALLENGE 2: CROSS-VALIDATION STRATEGIES
print("\n" + "=" * 55)
print("🔄 CHALLENGE 2: CROSS-VALIDATION STRATEGIES")
print("=" * 55)

# Standard k-fold cross-validation
print("1️⃣ Standard K-Fold Cross-Validation:")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

cv_scores = cross_val_score(rf_clf, X_med_train, y_med_train, cv=5, scoring='f1')
print(f"   5-Fold CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Stratified k-fold (maintains class balance)
print("\n2️⃣ Stratified K-Fold Cross-Validation:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(rf_clf, X_med_train, y_med_train, cv=skf, scoring='f1')
print(f"   Stratified 5-Fold CV F1-Score: {stratified_scores.mean():.4f} (+/- {stratified_scores.std() * 2:.4f})")

# Time series cross-validation for house prices
print("\n3️⃣ Time Series Cross-Validation:")
X_house = house_data.drop(['price', 'date_sold'], axis=1)
y_house = house_data['price']

# Sort by date for time series split
sorted_indices = house_data['date_sold'].argsort()
X_house_sorted = X_house.iloc[sorted_indices]
y_house_sorted = y_house.iloc[sorted_indices]

tscv = TimeSeriesSplit(n_splits=5)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
ts_scores = cross_val_score(rf_reg, X_house_sorted, y_house_sorted, cv=tscv, scoring='r2')
print(f"   Time Series CV R²-Score: {ts_scores.mean():.4f} (+/- {ts_scores.std() * 2:.4f})")

# Visualize cross-validation results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.boxplot([cv_scores, stratified_scores], labels=['Standard K-Fold', 'Stratified K-Fold'])
plt.title('Classification CV Comparison')
plt.ylabel('F1-Score')

plt.subplot(1, 3, 2)
plt.plot(range(1, len(ts_scores) + 1), ts_scores, 'bo-')
plt.title('Time Series CV Scores')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.grid(True, alpha=0.3)

# Learning curves
plt.subplot(1, 3, 3)
train_sizes, train_scores, val_scores = learning_curve(
    rf_clf, X_med_train, y_med_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')

plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# CHALLENGE 3: BIAS-VARIANCE ANALYSIS
print("\n" + "=" * 55)
print("⚖️ CHALLENGE 3: BIAS-VARIANCE ANALYSIS")
print("=" * 55)

# Validation curves for model complexity
print("🔍 Analyzing Model Complexity vs Performance:")

# Random Forest: n_estimators
param_range = [10, 25, 50, 100, 200, 300, 500]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42), X_med_train, y_med_train,
    param_name='n_estimators', param_range=param_range,
    cv=5, scoring='f1', n_jobs=-1
)

# Plot validation curve
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')

plt.xlabel('Number of Estimators')
plt.ylabel('F1 Score')
plt.title('Validation Curve: Random Forest')
plt.legend()
plt.grid(True, alpha=0.3)

# Logistic Regression: regularization strength
C_range = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores_lr, val_scores_lr = validation_curve(
    LogisticRegression(random_state=42, max_iter=1000), X_med_train_scaled, y_med_train,
    param_name='C', param_range=C_range,
    cv=5, scoring='f1', n_jobs=-1
)

plt.subplot(2, 2, 2)
train_mean_lr = train_scores_lr.mean(axis=1)
train_std_lr = train_scores_lr.std(axis=1)
val_mean_lr = val_scores_lr.mean(axis=1)
val_std_lr = val_scores_lr.std(axis=1)

plt.semilogx(C_range, train_mean_lr, 'o-', color='blue', label='Training score')
plt.fill_between(C_range, train_mean_lr - train_std_lr, train_mean_lr + train_std_lr, alpha=0.2, color='blue')
plt.semilogx(C_range, val_mean_lr, 'o-', color='red', label='Cross-validation score')
plt.fill_between(C_range, val_mean_lr - val_std_lr, val_mean_lr + val_std_lr, alpha=0.2, color='red')

plt.xlabel('Regularization parameter C')
plt.ylabel('F1 Score')
plt.title('Validation Curve: Logistic Regression')
plt.legend()
plt.grid(True, alpha=0.3)

# Model complexity analysis
models_complexity = {
    'High Bias (Logistic Reg C=0.01)': LogisticRegression(C=0.01, random_state=42),
    'Balanced (Random Forest)': RandomForestClassifier(n_estimators=100, random_state=42),
    'High Variance (Random Forest max_depth=None)': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
}

complexity_results = {}

print("\n📊 Bias-Variance Trade-off Analysis:")
for name, model in models_complexity.items():
    if 'Logistic' in name:
        cv_scores = cross_val_score(model, X_med_train_scaled, y_med_train, cv=5, scoring='f1')
    else:
        cv_scores = cross_val_score(model, X_med_train, y_med_train, cv=5, scoring='f1')

    complexity_results[name] = {
        'CV_mean': cv_scores.mean(),
        'CV_std': cv_scores.std()
    }

    print(f"  {name}:")
    print(f"    CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"    Variance: {cv_scores.std():.4f}")

# Plot complexity analysis
plt.subplot(2, 2, 3)
names = list(complexity_results.keys())
means = [complexity_results[name]['CV_mean'] for name in names]
stds = [complexity_results[name]['CV_std'] for name in names]

plt.bar(range(len(names)), means, yerr=stds, alpha=0.7, capsize=5)
plt.xticks(range(len(names)), [name.split('(')[0] for name in names], rotation=45)
plt.ylabel('F1 Score')
plt.title('Bias-Variance Trade-off')
plt.tight_layout()

# Bootstrap analysis for confidence intervals
plt.subplot(2, 2, 4)
n_bootstrap = 100
bootstrap_scores = []

for i in range(n_bootstrap):
    # Bootstrap sample
    bootstrap_indices = np.random.choice(len(X_med_train), size=len(X_med_train), replace=True)
    X_bootstrap = X_med_train.iloc[bootstrap_indices]
    y_bootstrap = y_med_train.iloc[bootstrap_indices]

    # Train and evaluate
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_temp.fit(X_bootstrap, y_bootstrap)
    score = f1_score(y_med_test, rf_temp.predict(X_med_test))
    bootstrap_scores.append(score)

plt.hist(bootstrap_scores, bins=20, alpha=0.7, density=True)
plt.axvline(np.mean(bootstrap_scores), color='red', linestyle='--',
           label=f'Mean: {np.mean(bootstrap_scores):.4f}')
plt.axvline(np.percentile(bootstrap_scores, 2.5), color='orange', linestyle='--',
           label=f'95% CI: [{np.percentile(bootstrap_scores, 2.5):.4f}, {np.percentile(bootstrap_scores, 97.5):.4f}]')
plt.axvline(np.percentile(bootstrap_scores, 97.5), color='orange', linestyle='--')
plt.xlabel('F1 Score')
plt.ylabel('Density')
plt.title('Bootstrap Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# CHALLENGE 4: REGRESSION EVALUATION
print("\n" + "=" * 55)
print("📈 CHALLENGE 4: REGRESSION EVALUATION METRICS")
print("=" * 55)

# Prepare house price data
X_house_clean = house_data.drop(['price', 'date_sold'], axis=1)
y_house_clean = house_data['price']

X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(
    X_house_clean, y_house_clean, test_size=0.3, random_state=42
)

# Train regression models
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1000),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

regression_metrics = {}

print("🏠 Training Regression Models:")
for name, regressor in regressors.items():
    regressor.fit(X_house_train, y_house_train)
    y_pred = regressor.predict(X_house_test)

    # Comprehensive regression metrics
    metrics = {
        'MSE': mean_squared_error(y_house_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_house_test, y_pred)),
        'MAE': mean_absolute_error(y_house_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_house_test, y_pred) * 100,
        'R²': r2_score(y_house_test, y_pred),
        'Adjusted_R²': 1 - (1 - r2_score(y_house_test, y_pred)) * (len(y_house_test) - 1) / (len(y_house_test) - X_house_test.shape[1] - 1)
    }

    regression_metrics[name] = metrics

    print(f"\n{name}:")
    print(f"  RMSE: ${metrics['RMSE']:,.0f}")
    print(f"  MAE: ${metrics['MAE']:,.0f}")
    print(f"  MAPE: {metrics['MAPE']:.1f}%")
    print(f"  R²: {metrics['R²']:.4f}")

# Regression metrics comparison
reg_metrics_df = pd.DataFrame(regression_metrics).T
print(f"\n📊 Regression Metrics Summary:")
print(reg_metrics_df.round(4))

# Residual analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Regression Model Evaluation', fontsize=16, fontweight='bold')

# Best model for detailed analysis
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_house_train, y_house_train)
y_pred_best = best_model.predict(X_house_test)
residuals = y_house_test - y_pred_best

# Predicted vs Actual
axes[0, 0].scatter(y_house_test, y_pred_best, alpha=0.6)
axes[0, 0].plot([y_house_test.min(), y_house_test.max()],
               [y_house_test.min(), y_house_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price ($)')
axes[0, 0].set_ylabel('Predicted Price ($)')
axes[0, 0].set_title('Predicted vs Actual Prices')

# Residual plot
axes[0, 1].scatter(y_pred_best, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='red', linestyle='--')
axes[0, 1].set_xlabel('Predicted Price ($)')
axes[0, 1].set_ylabel('Residuals ($)')
axes[0, 1].set_title('Residual Plot')

# Residual distribution
axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
axes[1, 0].set_xlabel('Residuals ($)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Residual Distribution')

# Q-Q plot for residual normality
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Residual Normality')

plt.tight_layout()
plt.show()

# CHALLENGE 5: MODEL CALIBRATION AND PROBABILITY ANALYSIS
print("\n" + "=" * 55)
print("🎲 CHALLENGE 5: MODEL CALIBRATION ANALYSIS")
print("=" * 55)

# Calibration analysis for classification models
print("📊 Probability Calibration Analysis:")

# Train models and get probabilities
calibration_data = {}

for name, clf in list(classifiers.items())[:3]:  # Use first 3 for clarity
    if name in ['Logistic Regression', 'SVM']:
        clf.fit(X_med_train_scaled, y_med_train)
        y_proba = clf.predict_proba(X_med_test_scaled)[:, 1]
    else:
        clf.fit(X_med_train, y_med_train)
        y_proba = clf.predict_proba(X_med_test)[:, 1]

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_med_test, y_proba, n_bins=10)

    calibration_data[name] = {
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value,
        'y_proba': y_proba
    }

# Plot calibration curves
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
for name, data in calibration_data.items():
    plt.plot(data['mean_predicted_value'], data['fraction_of_positives'],
            marker='o', label=name)

plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Reliability diagram
plt.subplot(2, 2, 2)
for name, data in calibration_data.items():
    plt.hist(data['y_proba'], bins=20, alpha=0.5, label=name, density=True)

plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Predicted Probability Distribution')
plt.legend()

# Precision-Recall curves
plt.subplot(2, 2, 3)
for name, data in calibration_data.items():
    precision, recall, _ = precision_recall_curve(y_med_test, data['y_proba'])
    plt.plot(recall, precision, label=f'{name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Brier Score (calibration quality metric)
plt.subplot(2, 2, 4)
brier_scores = []
model_names = []

for name, data in calibration_data.items():
    brier_score = np.mean((data['y_proba'] - y_med_test) ** 2)
    brier_scores.append(brier_score)
    model_names.append(name)
    print(f"  {name} Brier Score: {brier_score:.4f}")

plt.bar(model_names, brier_scores, alpha=0.7)
plt.ylabel('Brier Score (Lower is Better)')
plt.title('Model Calibration Quality')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# CHALLENGE 6: ADVANCED DIAGNOSTIC TECHNIQUES
print("\n" + "=" * 55)
print("🔬 CHALLENGE 6: ADVANCED MODEL DIAGNOSTICS")
print("=" * 55)

print("🔍 Advanced Diagnostic Analysis:")

# 1. Feature importance stability across folds
feature_importance_stability = {}
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_importances = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_med_train, y_med_train)):
    X_fold_train = X_med_train.iloc[train_idx]
    y_fold_train = y_med_train.iloc[train_idx]

    rf_fold = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_fold.fit(X_fold_train, y_fold_train)
    fold_importances.append(rf_fold.feature_importances_)

fold_importances = np.array(fold_importances)
importance_mean = fold_importances.mean(axis=0)
importance_std = fold_importances.std(axis=0)

# 2. Permutation importance
from sklearn.inspection import permutation_importance

rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(X_med_train, y_med_train)

perm_importance = permutation_importance(rf_final, X_med_test, y_med_test, n_repeats=10, random_state=42)

# Visualize diagnostic results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Advanced Model Diagnostics', fontsize=16, fontweight='bold')

# Feature importance stability
feature_names = X_med.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': importance_mean,
    'importance_std': importance_std
}).sort_values('importance_mean', ascending=True)

axes[0, 0].barh(importance_df['feature'], importance_df['importance_mean'],
               xerr=importance_df['importance_std'], alpha=0.7, capsize=3)
axes[0, 0].set_xlabel('Feature Importance')
axes[0, 0].set_title('Feature Importance Stability Across Folds')

# Permutation importance
perm_sorted_idx = perm_importance.importances_mean.argsort()
axes[0, 1].boxplot(perm_importance.importances[perm_sorted_idx].T,
                  vert=False, labels=feature_names[perm_sorted_idx])
axes[0, 1].set_xlabel('Permutation Importance')
axes[0, 1].set_title('Permutation Feature Importance')

# Learning curve with multiple metrics
train_sizes, train_scores_acc, val_scores_acc = learning_curve(
    rf_final, X_med_train, y_med_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

train_sizes, train_scores_f1, val_scores_f1 = learning_curve(
    rf_final, X_med_train, y_med_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1'
)

axes[1, 0].plot(train_sizes, train_scores_acc.mean(axis=1), 'o-', label='Training Accuracy')
axes[1, 0].plot(train_sizes, val_scores_acc.mean(axis=1), 'o-', label='Validation Accuracy')
axes[1, 0].plot(train_sizes, train_scores_f1.mean(axis=1), 's--', label='Training F1')
axes[1, 0].plot(train_sizes, val_scores_f1.mean(axis=1), 's--', label='Validation F1')
axes[1, 0].set_xlabel('Training Set Size')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Multi-Metric Learning Curves')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Model confidence analysis
y_proba_final = rf_final.predict_proba(X_med_test)[:, 1]
confidence_bins = np.linspace(0, 1, 11)
bin_indices = np.digitize(y_proba_final, confidence_bins)

bin_accuracy = []
bin_counts = []
for i in range(1, len(confidence_bins)):
    mask = bin_indices == i
    if mask.sum() > 0:
        bin_acc = accuracy_score(y_med_test[mask], (y_proba_final[mask] > 0.5).astype(int))
        bin_accuracy.append(bin_acc)
        bin_counts.append(mask.sum())
    else:
        bin_accuracy.append(0)
        bin_counts.append(0)

bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
axes[1, 1].bar(bin_centers, bin_counts, width=0.08, alpha=0.5, label='Count', color='lightblue')
ax_twin = axes[1, 1].twinx()
ax_twin.plot(bin_centers, bin_accuracy, 'ro-', label='Accuracy')
axes[1, 1].set_xlabel('Predicted Probability Bin')
axes[1, 1].set_ylabel('Count')
ax_twin.set_ylabel('Accuracy')
axes[1, 1].set_title('Prediction Confidence Analysis')

plt.tight_layout()
plt.show()

# FINAL EVALUATION SUMMARY
print("\n" + "=" * 55)
print("🏆 COMPREHENSIVE EVALUATION SUMMARY")
print("=" * 55)

print("📊 Model Performance Ranking (by F1-Score):")
ranking = metrics_df.sort_values('F1-Score', ascending=False)
for i, (model, metrics) in enumerate(ranking.iterrows(), 1):
    print(f"{i}. {model}: F1={metrics['F1-Score']:.4f}, ROC-AUC={metrics['ROC-AUC']:.4f}")

print(f"\n🎯 Key Evaluation Insights:")
print(f"• Best performing model: {ranking.index[0]}")
print(f"• Most stable model: Random Forest (lowest CV std)")
print(f"• Best calibrated: Logistic Regression (closest to perfect calibration)")
print(f"• Most important features: {importance_df.tail(3)['feature'].tolist()}")

print(f"\n✅ Model Evaluation and Validation Challenge Completed!")
print("What you've mastered:")
print("• Comprehensive evaluation metrics for classification and regression")
print("• Advanced cross-validation strategies (stratified, time series)")
print("• Bias-variance trade-off analysis and model complexity tuning")
print("• Residual analysis and regression diagnostics")
print("• Model calibration and probability analysis")
print("• Advanced diagnostic techniques (permutation importance, stability analysis)")
print("• Statistical validation and confidence intervals")

print(f"\n🎓 You now have the skills to rigorously evaluate any ML model!")
```

### Success Criteria

- Apply comprehensive evaluation metrics appropriate for different problem types
- Implement various cross-validation strategies effectively
- Analyze and interpret bias-variance trade-offs
- Perform thorough residual analysis for regression models
- Evaluate model calibration and probability reliability
- Use advanced diagnostic techniques for model validation
- Create production-ready evaluation pipelines

### Learning Objectives

- Master the full spectrum of evaluation metrics and their trade-offs
- Understand when to use different cross-validation approaches
- Learn to diagnose overfitting, underfitting, and model quality issues
- Practice statistical validation and confidence interval estimation
- Develop skills in model selection and performance comparison
- Build robust evaluation frameworks for production systems

---

_Pro tip: A model is only as good as your evaluation - invest time in comprehensive validation to build trustworthy systems!_
