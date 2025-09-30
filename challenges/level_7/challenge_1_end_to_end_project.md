# Level 7: Data Science Master

## Challenge 1: End-to-End Data Science Project

Master the complete data science workflow from problem definition to production deployment with a comprehensive real-world project combining all advanced techniques learned.

### Objective

Build a complete end-to-end data science solution that demonstrates mastery of the entire pipeline: data acquisition, preprocessing, feature engineering, modeling, evaluation, and deployment.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# Advanced libraries
import joblib
import json
from pathlib import Path

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

print("🏆 End-to-End Data Science Project: Customer Churn Prevention")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

print("🎯 Project Overview: Telecom Customer Churn Prediction")
print("\nBusiness Problem:")
print("• Predict customer churn to enable proactive retention")
print("• Identify key factors driving customer attrition")
print("• Provide actionable insights for business strategy")
print("• Build production-ready model with monitoring capabilities")

# PROJECT PHASE 1: DATA GENERATION AND EXPLORATION
print("\n" + "=" * 60)
print("📊 PHASE 1: DATA ACQUISITION & EXPLORATION")
print("=" * 60)

def generate_comprehensive_telecom_dataset(n_customers=10000):
    """Generate realistic telecom customer dataset with churn labels"""

    print(f"Generating comprehensive telecom dataset with {n_customers} customers...")

    np.random.seed(42)

    # Customer Demographics
    customer_data = {
        'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
        'age': np.random.normal(40, 15, n_customers).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_customers, p=[0.52, 0.48]),
        'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.84, 0.16]),
        'partner': np.random.choice(['Yes', 'No'], n_customers, p=[0.52, 0.48]),
        'dependents': np.random.choice(['Yes', 'No'], n_customers, p=[0.30, 0.70])
    }

    # Service Information
    tenure_months = np.random.exponential(24, n_customers)
    tenure_months = np.clip(tenure_months, 1, 72).astype(int)

    # Phone services
    phone_service = np.random.choice(['Yes', 'No'], n_customers, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == 'Yes',
        np.random.choice(['Yes', 'No'], n_customers, p=[0.42, 0.58]),
        'No phone service'
    )

    # Internet services
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22])

    # Additional services (conditional on internet)
    def conditional_service(base_service, yes_prob=0.5):
        return np.where(
            base_service != 'No',
            np.random.choice(['Yes', 'No'], n_customers, p=[yes_prob, 1-yes_prob]),
            'No internet service'
        )

    online_security = conditional_service(internet_service, 0.29)
    online_backup = conditional_service(internet_service, 0.34)
    device_protection = conditional_service(internet_service, 0.34)
    tech_support = conditional_service(internet_service, 0.29)
    streaming_tv = conditional_service(internet_service, 0.40)
    streaming_movies = conditional_service(internet_service, 0.40)

    # Contract and billing
    contract_types = ['Month-to-month', 'One year', 'Two year']
    contract_probs = [0.55, 0.21, 0.24]
    contract = np.random.choice(contract_types, n_customers, p=contract_probs)

    paperless_billing = np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41])

    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    payment_method = np.random.choice(payment_methods, n_customers, p=[0.34, 0.19, 0.22, 0.25])

    # Calculate monthly charges based on services
    base_charges = np.where(phone_service == 'Yes', 25, 0)
    base_charges += np.where(multiple_lines == 'Yes', 15, 0)

    # Internet charges
    internet_charges = np.where(internet_service == 'DSL', 35,
                               np.where(internet_service == 'Fiber optic', 55, 0))
    base_charges += internet_charges

    # Additional service charges
    additional_services = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    for service in additional_services:
        base_charges += np.where(service == 'Yes', np.random.normal(8, 2, n_customers), 0)

    # Add some noise and rounding
    monthly_charges = base_charges + np.random.normal(0, 5, n_customers)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

    # Total charges (tenure * monthly charges with some variation)
    total_charges = tenure_months * monthly_charges + np.random.normal(0, 50, n_customers)
    total_charges = np.maximum(total_charges, monthly_charges)  # At least one month

    # Calculate churn probability based on features
    churn_prob = 0.1  # Base probability

    # Factors that increase churn probability
    churn_prob += np.where(contract == 'Month-to-month', 0.4, 0)
    churn_prob += np.where(tenure_months < 12, 0.3, 0)
    churn_prob += np.where(monthly_charges > 80, 0.2, 0)
    churn_prob += np.where(payment_method == 'Electronic check', 0.15, 0)
    churn_prob += np.where(paperless_billing == 'Yes', 0.1, 0)
    churn_prob += np.where(tech_support == 'No', 0.1, 0)
    churn_prob += np.where(online_security == 'No', 0.1, 0)

    # Factors that decrease churn probability
    churn_prob -= np.where(contract == 'Two year', 0.25, 0)
    churn_prob -= np.where(tenure_months > 36, 0.2, 0)
    churn_prob -= np.where(dependents == 'Yes', 0.1, 0)
    churn_prob -= np.where(partner == 'Yes', 0.05, 0)

    # Ensure probability is between 0 and 1
    churn_prob = np.clip(churn_prob, 0.01, 0.95)

    # Generate churn labels
    churn = np.random.binomial(1, churn_prob, n_customers)

    # Combine all data
    dataset = pd.DataFrame({
        'customerID': customer_data['customer_id'],
        'gender': customer_data['gender'],
        'SeniorCitizen': customer_data['senior_citizen'],
        'Partner': customer_data['partner'],
        'Dependents': customer_data['dependents'],
        'tenure': tenure_months,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'Churn': churn
    })

    return dataset

# Generate the dataset
print("🏗️ Generating comprehensive telecom customer dataset...")
telecom_data = generate_comprehensive_telecom_dataset(10000)

print(f"Dataset generated successfully!")
print(f"Shape: {telecom_data.shape}")
print(f"Churn rate: {telecom_data['Churn'].mean():.1%}")

# Data Quality Assessment
print("\n📋 Data Quality Assessment:")

def comprehensive_data_quality_check(df):
    """Perform comprehensive data quality assessment"""

    print("🔍 Data Quality Report")
    print("-" * 30)

    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100

    if missing_values.sum() > 0:
        print(f"\n❌ Missing Values Found:")
        for col in missing_values[missing_values > 0].index:
            print(f"  {col}: {missing_values[col]} ({missing_percent[col]:.1f}%)")
    else:
        print(f"\n✅ No missing values found")

    # Data types
    print(f"\n📊 Data Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Duplicate rows: {duplicates}")

    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\n🏷️ Categorical Columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        unique_values = df[col].nunique()
        print(f"  {col}: {unique_values} unique values")
        if unique_values <= 10:
            print(f"    Values: {list(df[col].unique())}")

    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n🔢 Numerical Columns ({len(numerical_cols)}):")
    for col in numerical_cols:
        print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")

    return {
        'missing_values': missing_values,
        'duplicates': duplicates,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }

data_quality = comprehensive_data_quality_check(telecom_data)

# Exploratory Data Analysis
print("\n📈 Exploratory Data Analysis:")

def advanced_eda(df, target_col='Churn'):
    """Perform advanced exploratory data analysis"""

    print(f"🎯 Target Variable Analysis ({target_col}):")
    target_dist = df[target_col].value_counts(normalize=True)
    print(f"Class distribution:")
    for class_val, prop in target_dist.items():
        print(f"  {class_val}: {prop:.1%}")

    # Correlation analysis for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    if len(numerical_cols) > 0:
        print(f"\n📊 Correlation with Target:")
        correlations = df[numerical_cols + [target_col]].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
        print("Top 5 correlations:")
        for feature, corr in correlations.head().items():
            print(f"  {feature}: {corr:.3f}")

    # Categorical feature analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\n🏷️ Categorical Feature Analysis:")

    high_churn_features = []

    for col in categorical_cols:
        churn_by_category = df.groupby(col)[target_col].agg(['mean', 'count'])
        max_churn_rate = churn_by_category['mean'].max()
        min_churn_rate = churn_by_category['mean'].min()
        churn_range = max_churn_rate - min_churn_rate

        print(f"  {col}:")
        print(f"    Churn rate range: {min_churn_rate:.1%} - {max_churn_rate:.1%} (Δ{churn_range:.1%})")

        if churn_range > 0.2:  # High discrimination
            high_churn_features.append(col)
            print(f"    🎯 High discrimination feature!")

    print(f"\n🎯 High-discrimination features: {high_churn_features}")

    return {
        'target_distribution': target_dist,
        'correlations': correlations if len(numerical_cols) > 0 else None,
        'high_churn_features': high_churn_features
    }

eda_results = advanced_eda(telecom_data)

# PHASE 2: ADVANCED FEATURE ENGINEERING
print("\n" + "=" * 60)
print("🔧 PHASE 2: ADVANCED FEATURE ENGINEERING")
print("=" * 60)

def advanced_feature_engineering(df):
    """Create advanced features for better model performance"""

    print("🛠️ Creating advanced features...")

    # Create a copy for feature engineering
    df_features = df.copy()

    # 1. Tenure-based features
    df_features['tenure_groups'] = pd.cut(df_features['tenure'],
                                        bins=[0, 12, 24, 48, 72],
                                        labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])

    # 2. Charges-based features
    df_features['avg_monthly_charges'] = df_features['TotalCharges'] / (df_features['tenure'] + 1)
    df_features['charges_per_service'] = df_features['MonthlyCharges'] / (df_features[['PhoneService', 'InternetService']].apply(lambda x: (x != 'No').sum(), axis=1) + 1)

    # 3. Service adoption features
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_features['total_services'] = df_features[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df_features['service_adoption_rate'] = df_features['total_services'] / len(service_cols)

    # 4. Payment and billing features
    df_features['is_autopay'] = ((df_features['PaymentMethod'] == 'Bank transfer (automatic)') |
                                (df_features['PaymentMethod'] == 'Credit card (automatic)')).astype(int)

    # 5. Customer value features
    df_features['customer_lifetime_value'] = df_features['TotalCharges']
    df_features['clv_segment'] = pd.qcut(df_features['customer_lifetime_value'],
                                        q=4, labels=['Low', 'Medium', 'High', 'Premium'])

    # 6. Risk indicators
    df_features['high_risk'] = (
        (df_features['Contract'] == 'Month-to-month') &
        (df_features['tenure'] < 12) &
        (df_features['MonthlyCharges'] > df_features['MonthlyCharges'].median())
    ).astype(int)

    # 7. Interaction features
    df_features['fiber_no_security'] = (
        (df_features['InternetService'] == 'Fiber optic') &
        (df_features['OnlineSecurity'] == 'No')
    ).astype(int)

    df_features['senior_single'] = (
        (df_features['SeniorCitizen'] == 1) &
        (df_features['Partner'] == 'No')
    ).astype(int)

    print(f"✅ Feature engineering completed!")
    print(f"Original features: {df.shape[1]}")
    print(f"New features: {df_features.shape[1]}")
    print(f"Added features: {df_features.shape[1] - df.shape[1]}")

    # List new features
    original_cols = set(df.columns)
    new_cols = set(df_features.columns) - original_cols
    print(f"New features created: {list(new_cols)}")

    return df_features

# Apply feature engineering
engineered_data = advanced_feature_engineering(telecom_data)

# PHASE 3: DATA PREPROCESSING AND PIPELINE
print("\n" + "=" * 60)
print("🏭 PHASE 3: DATA PREPROCESSING PIPELINE")
print("=" * 60)

def create_preprocessing_pipeline(df, target_col='Churn'):
    """Create comprehensive preprocessing pipeline"""

    print("🔄 Building preprocessing pipeline...")

    # Separate features and target
    X = df.drop([target_col, 'customerID'], axis=1)
    y = df[target_col]

    # Handle categorical variables with proper encoding
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Categorical columns: {len(categorical_columns)}")
    print(f"Numerical columns: {len(numerical_columns)}")

    # Encode categorical variables
    X_processed = X.copy()
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le

    # Handle any remaining object columns from feature engineering
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le

    print(f"✅ Preprocessing completed!")
    print(f"Final feature matrix shape: {X_processed.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X_processed, y, label_encoders

# Apply preprocessing
X_processed, y, encoders = create_preprocessing_pipeline(engineered_data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training churn rate: {y_train.mean():.1%}")
print(f"Test churn rate: {y_test.mean():.1%}")

# PHASE 4: MODEL DEVELOPMENT AND EVALUATION
print("\n" + "=" * 60)
print("🤖 PHASE 4: MODEL DEVELOPMENT & EVALUATION")
print("=" * 60)

def comprehensive_model_evaluation(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models comprehensively"""

    print("🎯 Training multiple models for comparison...")

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Evaluation metrics
        auc_score = roc_auc_score(y_test, y_proba)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred)
        }

        print(f"  AUC Score: {auc_score:.4f}")
        print(f"  CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
    best_model = results[best_model_name]['model']

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"Best AUC Score: {results[best_model_name]['auc_score']:.4f}")

    return results, best_model, best_model_name

# Train and evaluate models
model_results, best_model, best_model_name = comprehensive_model_evaluation(
    X_train, X_test, y_train, y_test
)

# Feature Importance Analysis
print("\n📊 Feature Importance Analysis:")

def analyze_feature_importance(model, feature_names, top_k=15):
    """Analyze and display feature importance"""

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"Top {top_k} Most Important Features:")
        for i, (_, row) in enumerate(feature_importance_df.head(top_k).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")

        return feature_importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None

feature_importance = analyze_feature_importance(best_model, X_train.columns)

# PHASE 5: MODEL OPTIMIZATION
print("\n" + "=" * 60)
print("⚡ PHASE 5: MODEL OPTIMIZATION")
print("=" * 60)

def optimize_model(X_train, y_train, model_type='RandomForest'):
    """Optimize model using GridSearchCV"""

    print(f"🔍 Optimizing {model_type} model...")

    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"✅ Optimization completed!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

# Optimize the best model
if best_model_name == 'Random Forest':
    optimized_model = optimize_model(X_train, y_train, 'RandomForest')
elif best_model_name == 'Gradient Boosting':
    optimized_model = optimize_model(X_train, y_train, 'GradientBoosting')
else:
    optimized_model = best_model

# Evaluate optimized model
print("\n📊 Optimized Model Evaluation:")
optimized_pred = optimized_model.predict(X_test)
optimized_proba = optimized_model.predict_proba(X_test)[:, 1]
optimized_auc = roc_auc_score(y_test, optimized_proba)

print(f"Optimized Model AUC: {optimized_auc:.4f}")
print(f"Improvement: {optimized_auc - model_results[best_model_name]['auc_score']:.4f}")

# PHASE 6: MODEL DEPLOYMENT PREPARATION
print("\n" + "=" * 60)
print("🚀 PHASE 6: MODEL DEPLOYMENT PREPARATION")
print("=" * 60)

def prepare_model_for_deployment(model, encoders, feature_names, model_name="churn_model"):
    """Prepare model artifacts for deployment"""

    print("📦 Preparing model for deployment...")

    # Create deployment directory
    deployment_dir = Path("model_deployment")
    deployment_dir.mkdir(exist_ok=True)

    # Save model
    model_path = deployment_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    # Save encoders
    encoders_path = deployment_dir / f"{model_name}_encoders.joblib"
    joblib.dump(encoders, encoders_path)
    print(f"✅ Encoders saved to {encoders_path}")

    # Save feature names
    features_path = deployment_dir / f"{model_name}_features.json"
    with open(features_path, 'w') as f:
        json.dump(list(feature_names), f, indent=2)
    print(f"✅ Feature names saved to {features_path}")

    # Create model metadata
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'training_date': datetime.now().isoformat(),
        'feature_count': len(feature_names),
        'performance_metrics': {
            'auc_score': float(optimized_auc),
            'model_version': '1.0'
        }
    }

    metadata_path = deployment_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to {metadata_path}")

    return deployment_dir

# Prepare deployment artifacts
deployment_path = prepare_model_for_deployment(
    optimized_model, encoders, X_train.columns, "telecom_churn_model"
)

# Create prediction function
def create_prediction_function():
    """Create production-ready prediction function"""

    prediction_code = '''
def predict_churn(customer_data, model_path, encoders_path, features_path):
    """
    Production-ready churn prediction function

    Args:
        customer_data: Dictionary with customer features
        model_path: Path to trained model
        encoders_path: Path to label encoders
        features_path: Path to feature names

    Returns:
        Dictionary with prediction and probability
    """
    import joblib
    import json
    import pandas as pd
    import numpy as np

    # Load model artifacts
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)

    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    # Create DataFrame from input
    df = pd.DataFrame([customer_data])

    # Apply same preprocessing
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))

    # Ensure all features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features

    # Select and order features
    df = df[feature_names]

    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        'churn_prediction': int(prediction),
        'churn_probability': float(probability),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
    }

# Example usage:
if __name__ == "__main__":
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }

    result = predict_churn(
        sample_customer,
        'model_deployment/telecom_churn_model.joblib',
        'model_deployment/telecom_churn_model_encoders.joblib',
        'model_deployment/telecom_churn_model_features.json'
    )

    print(f"Prediction: {result}")
'''

    # Save prediction function
    with open(deployment_path / "predict_churn.py", 'w') as f:
        f.write(prediction_code)

    print(f"✅ Prediction function saved to {deployment_path}/predict_churn.py")

create_prediction_function()

# PHASE 7: COMPREHENSIVE VISUALIZATION AND REPORTING
print("\n" + "=" * 60)
print("📊 PHASE 7: COMPREHENSIVE VISUALIZATION & REPORTING")
print("=" * 60)

# Create comprehensive visualizations
fig, axes = plt.subplots(4, 4, figsize=(24, 20))
fig.suptitle('End-to-End Data Science Project: Telecom Churn Analysis', fontsize=16, fontweight='bold')

# Plot 1: Target distribution
ax = axes[0, 0]
churn_counts = telecom_data['Churn'].value_counts()
colors = ['lightblue', 'lightcoral']
wedges, texts, autotexts = ax.pie(churn_counts.values, labels=['No Churn', 'Churn'],
                                 autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Customer Churn Distribution')

# Plot 2: Feature importance
ax = axes[0, 1]
if feature_importance is not None:
    top_features = feature_importance.head(10)
    ax.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Feature Importance')
    ax.grid(axis='x', alpha=0.3)

# Plot 3: Model comparison
ax = axes[0, 2]
model_names = list(model_results.keys())
auc_scores = [model_results[name]['auc_score'] for name in model_names]
colors = ['gold' if name == best_model_name else 'lightblue' for name in model_names]

bars = ax.bar(model_names, auc_scores, color=colors, alpha=0.8)
ax.set_ylabel('AUC Score')
ax.set_title('Model Performance Comparison')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, auc_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Confusion matrix for best model
ax = axes[0, 3]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model_results[best_model_name]['y_pred'])
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title(f'Confusion Matrix - {best_model_name}')
tick_marks = np.arange(2)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(['No Churn', 'Churn'])
ax.set_yticklabels(['No Churn', 'Churn'])

# Add text annotations
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ax.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
           color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight='bold')

ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# Plot 5: Churn by contract type
ax = axes[1, 0]
churn_by_contract = telecom_data.groupby('Contract')['Churn'].agg(['mean', 'count'])
contract_types = churn_by_contract.index
churn_rates = churn_by_contract['mean']

bars = ax.bar(contract_types, churn_rates, color='lightcoral', alpha=0.7)
ax.set_ylabel('Churn Rate')
ax.set_title('Churn Rate by Contract Type')
ax.set_xticklabels(contract_types, rotation=45)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, rate in zip(bars, churn_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Monthly charges distribution
ax = axes[1, 1]
churn_charges = telecom_data[telecom_data['Churn'] == 1]['MonthlyCharges']
no_churn_charges = telecom_data[telecom_data['Churn'] == 0]['MonthlyCharges']

ax.hist(no_churn_charges, bins=30, alpha=0.7, label='No Churn', color='lightblue')
ax.hist(churn_charges, bins=30, alpha=0.7, label='Churn', color='lightcoral')
ax.set_xlabel('Monthly Charges ($)')
ax.set_ylabel('Frequency')
ax.set_title('Monthly Charges Distribution by Churn')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 7: Tenure vs Churn
ax = axes[1, 2]
tenure_churn = telecom_data.groupby('tenure')['Churn'].mean()
ax.plot(tenure_churn.index, tenure_churn.values, marker='o', alpha=0.7, linewidth=2)
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Churn Rate')
ax.set_title('Churn Rate by Tenure')
ax.grid(True, alpha=0.3)

# Plot 8: Service adoption vs churn
ax = axes[1, 3]
if 'total_services' in engineered_data.columns:
    service_churn = engineered_data.groupby('total_services')['Churn'].mean()
    ax.bar(service_churn.index, service_churn.values, color='lightgreen', alpha=0.7)
    ax.set_xlabel('Number of Additional Services')
    ax.set_ylabel('Churn Rate')
    ax.set_title('Churn Rate by Service Adoption')
    ax.grid(axis='y', alpha=0.3)

# Plot 9: ROC Curve
ax = axes[2, 0]
from sklearn.metrics import roc_curve
for name, results in model_results.items():
    fpr, tpr, _ = roc_curve(y_test, results['y_proba'])
    auc = results['auc_score']
    linestyle = '-' if name == best_model_name else '--'
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linestyle=linestyle, alpha=0.8)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 10: Precision-Recall Curve
ax = axes[2, 1]
from sklearn.metrics import precision_recall_curve
for name, results in model_results.items():
    precision, recall, _ = precision_recall_curve(y_test, results['y_proba'])
    linestyle = '-' if name == best_model_name else '--'
    ax.plot(recall, precision, label=f'{name}', linestyle=linestyle, alpha=0.8)

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 11: Feature correlation heatmap
ax = axes[2, 2]
if len(X_processed.select_dtypes(include=[np.number]).columns) > 0:
    # Select top correlated features for visualization
    corr_matrix = X_processed.select_dtypes(include=[np.number]).corr()

    # Select a subset for better visualization
    top_features = feature_importance['feature'].head(8).tolist() if feature_importance is not None else corr_matrix.columns[:8].tolist()
    top_features = [f for f in top_features if f in corr_matrix.columns]

    if len(top_features) > 1:
        subset_corr = corr_matrix.loc[top_features, top_features]
        im = ax.imshow(subset_corr, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(top_features)))
        ax.set_yticks(range(len(top_features)))
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.set_yticklabels(top_features)
        ax.set_title('Feature Correlation Heatmap')

        # Add correlation values
        for i in range(len(top_features)):
            for j in range(len(top_features)):
                ax.text(j, i, f'{subset_corr.iloc[i, j]:.2f}',
                       ha="center", va="center", fontsize=8)

# Plot 12: Customer segmentation by CLV
ax = axes[2, 3]
if 'clv_segment' in engineered_data.columns:
    clv_churn = engineered_data.groupby('clv_segment')['Churn'].agg(['mean', 'count'])
    segments = clv_churn.index
    churn_rates = clv_churn['mean']

    bars = ax.bar(segments, churn_rates, color='gold', alpha=0.7)
    ax.set_ylabel('Churn Rate')
    ax.set_title('Churn Rate by Customer Value Segment')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, churn_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

# Plot 13: Payment method analysis
ax = axes[3, 0]
payment_churn = telecom_data.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=True)
bars = ax.barh(range(len(payment_churn)), payment_churn.values, color='lightsteelblue', alpha=0.8)
ax.set_yticks(range(len(payment_churn)))
ax.set_yticklabels(payment_churn.index)
ax.set_xlabel('Churn Rate')
ax.set_title('Churn Rate by Payment Method')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, rate) in enumerate(zip(bars, payment_churn.values)):
    ax.text(rate + 0.01, i, f'{rate:.1%}', va='center', fontweight='bold')

# Plot 14: Internet service analysis
ax = axes[3, 1]
internet_churn = telecom_data.groupby('InternetService')['Churn'].mean()
bars = ax.bar(internet_churn.index, internet_churn.values, color='mediumseagreen', alpha=0.7)
ax.set_ylabel('Churn Rate')
ax.set_title('Churn Rate by Internet Service')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, rate in zip(bars, internet_churn.values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

# Plot 15: Model performance metrics
ax = axes[3, 2]
metrics = ['AUC', 'CV Mean', 'CV Std']
best_model_metrics = [
    model_results[best_model_name]['auc_score'],
    model_results[best_model_name]['cv_mean'],
    model_results[best_model_name]['cv_std']
]

bars = ax.bar(metrics, best_model_metrics, color=['gold', 'lightgreen', 'lightcoral'], alpha=0.7)
ax.set_ylabel('Score')
ax.set_title(f'Best Model Performance ({best_model_name})')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, score in zip(bars, best_model_metrics):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 16: Business impact projection
ax = axes[3, 3]
# Calculate potential business impact
total_customers = len(telecom_data)
churn_customers = telecom_data['Churn'].sum()
avg_monthly_revenue = telecom_data['MonthlyCharges'].mean()

# Assume model can prevent 30% of churn with 80% precision
prevented_churn = int(churn_customers * 0.3 * 0.8)
monthly_savings = prevented_churn * avg_monthly_revenue
annual_savings = monthly_savings * 12

impact_categories = ['Current\nChurn Cost', 'Potential\nSavings', 'Net\nBenefit']
impact_values = [
    churn_customers * avg_monthly_revenue * 12,  # Annual churn cost
    annual_savings,  # Annual savings from model
    annual_savings * 0.7  # Net benefit (accounting for intervention costs)
]

bars = ax.bar(impact_categories, [v/1000 for v in impact_values],
             color=['red', 'green', 'gold'], alpha=0.7)
ax.set_ylabel('Annual Value ($K)')
ax.set_title('Projected Business Impact')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, impact_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
           f'${value/1000:.0f}K', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# FINAL REPORTING
print("\n" + "=" * 60)
print("📋 PROJECT SUMMARY & BUSINESS RECOMMENDATIONS")
print("=" * 60)

print("🎯 PROJECT OUTCOMES:")
print(f"\n📊 Model Performance:")
print(f"   • Best Model: {best_model_name}")
print(f"   • AUC Score: {model_results[best_model_name]['auc_score']:.4f}")
print(f"   • Cross-validation: {model_results[best_model_name]['cv_mean']:.4f} (±{model_results[best_model_name]['cv_std']:.4f})")
print(f"   • Optimized AUC: {optimized_auc:.4f}")

print(f"\n🔍 Key Insights:")
if feature_importance is not None:
    top_3_features = feature_importance.head(3)
    print("   Top Risk Factors:")
    for i, (_, row) in enumerate(top_3_features.iterrows()):
        print(f"   {i+1}. {row['feature']} (Importance: {row['importance']:.3f})")

# Churn patterns
high_risk_contract = telecom_data.groupby('Contract')['Churn'].mean().idxmax()
high_risk_payment = telecom_data.groupby('PaymentMethod')['Churn'].mean().idxmax()

print(f"\n⚠️ High-Risk Segments:")
print(f"   • Contract Type: {high_risk_contract} ({telecom_data.groupby('Contract')['Churn'].mean().max():.1%} churn rate)")
print(f"   • Payment Method: {high_risk_payment} ({telecom_data.groupby('PaymentMethod')['Churn'].mean().max():.1%} churn rate)")
print(f"   • New Customers: {telecom_data[telecom_data['tenure'] < 12]['Churn'].mean():.1%} churn rate (tenure < 12 months)")

print(f"\n💰 Business Impact:")
total_customers = len(telecom_data)
current_churn_rate = telecom_data['Churn'].mean()
avg_monthly_revenue = telecom_data['MonthlyCharges'].mean()
annual_churn_cost = total_customers * current_churn_rate * avg_monthly_revenue * 12

print(f"   • Total Customers: {total_customers:,}")
print(f"   • Current Churn Rate: {current_churn_rate:.1%}")
print(f"   • Average Monthly Revenue per Customer: ${avg_monthly_revenue:.2f}")
print(f"   • Annual Cost of Churn: ${annual_churn_cost:,.0f}")

# Model impact projection
model_precision = 0.8  # Assumed precision for high-risk predictions
intervention_success_rate = 0.3  # Assumed 30% of interventions prevent churn
prevented_churn_annual = total_customers * current_churn_rate * model_precision * intervention_success_rate
annual_savings = prevented_churn_annual * avg_monthly_revenue * 12

print(f"   • Potential Annual Savings with Model: ${annual_savings:,.0f}")
print(f"   • ROI Potential: {(annual_savings / annual_churn_cost * 100):.1f}% of current churn cost")

print(f"\n🎯 BUSINESS RECOMMENDATIONS:")

print(f"\n1. 🎯 Immediate Actions:")
print("   • Target month-to-month customers with retention offers")
print("   • Implement automated alerts for customers with tenure < 12 months")
print("   • Review and improve electronic check payment experience")
print("   • Offer incentives for contract upgrades (1-year or 2-year)")

print(f"\n2. 📊 Retention Strategies:")
print("   • Proactive customer success outreach for high-risk segments")
print("   • Enhanced customer support for fiber optic customers without security")
print("   • Loyalty programs for long-term customers")
print("   • Personalized service bundles based on usage patterns")

print(f"\n3. 🔄 Operational Improvements:")
print("   • Implement real-time churn scoring in customer service systems")
print("   • A/B test different retention interventions")
print("   • Monitor model performance and retrain quarterly")
print("   • Create churn prevention workflows for high-risk customers")

print(f"\n4. 📈 Strategic Initiatives:")
print("   • Invest in customer onboarding improvements")
print("   • Develop predictive customer lifetime value models")
print("   • Create early warning systems for service quality issues")
print("   • Build customer feedback loops for continuous improvement")

print(f"\n🏆 MODEL DEPLOYMENT CHECKLIST:")
print("✅ Model trained and optimized")
print("✅ Performance validated on test set")
print("✅ Feature engineering pipeline documented")
print("✅ Model artifacts saved for deployment")
print("✅ Prediction function created")
print("✅ Business impact quantified")
print("✅ Monitoring strategy defined")

print(f"\n🎖️ NEXT STEPS:")
print("1. Deploy model to production environment")
print("2. Integrate with customer service systems")
print("3. Implement A/B testing framework for interventions")
print("4. Set up model monitoring and drift detection")
print("5. Plan quarterly model retraining schedule")
print("6. Create business dashboard for churn insights")

print("\n✅ End-to-End Data Science Project Challenge Completed!")
print("What you've mastered:")
print("• Complete data science project lifecycle")
print("• Advanced feature engineering techniques")
print("• Comprehensive model evaluation and selection")
print("• Model optimization and hyperparameter tuning")
print("• Production deployment preparation")
print("• Business impact analysis and ROI calculation")
print("• Strategic recommendations and implementation planning")

print(f"\n🚀 You've successfully completed a production-ready data science project!")
```

### Success Criteria

- Build complete end-to-end data science solution from problem to deployment
- Demonstrate advanced feature engineering and model optimization techniques
- Create production-ready model artifacts and prediction functions
- Provide comprehensive business analysis and strategic recommendations
- Implement proper model evaluation and performance monitoring frameworks
- Deliver actionable insights with quantified business impact

### Learning Objectives

- Master the complete data science project lifecycle
- Learn advanced feature engineering and preprocessing techniques
- Practice comprehensive model evaluation and selection methodologies
- Develop production deployment and monitoring capabilities
- Build business acumen for translating technical results to strategic value
- Create scalable and maintainable data science solutions

---

_Pro tip: Success in data science comes from understanding both the technical implementation and the business impact - always connect your models to real-world value creation!_
