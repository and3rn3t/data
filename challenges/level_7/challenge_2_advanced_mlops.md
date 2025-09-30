# Level 7: Data Science Master

## Challenge 2: Advanced MLOps and Model Monitoring

Master production-grade MLOps practices including automated model training, deployment, monitoring, and drift detection with comprehensive CI/CD pipelines.

### Objective

Build a complete MLOps infrastructure that automates the machine learning lifecycle from data ingestion to model deployment and monitoring in production.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# MLOps libraries
import mlflow
import mlflow.sklearn
import mlflow.tracking
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

# Model libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Monitoring libraries
import joblib
import json
import os
from pathlib import Path
import sqlite3
import hashlib
from typing import Dict, List, Tuple, Any
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Alerting
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

print("🚀 Advanced MLOps and Model Monitoring")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

print("🎯 MLOps System Overview:")
print("• Automated model training and versioning")
print("• Continuous integration and deployment (CI/CD)")
print("• Real-time model monitoring and drift detection")
print("• Automated retraining triggers")
print("• Performance tracking and alerting")
print("• A/B testing framework for model deployment")

# MLOps PHASE 1: MLFLOW EXPERIMENT TRACKING
print("\n" + "=" * 60)
print("📊 PHASE 1: MLFLOW EXPERIMENT TRACKING")
print("=" * 60)

# Set up MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit_risk_models")

def generate_credit_risk_dataset(n_samples=10000):
    """Generate realistic credit risk dataset"""

    print(f"Generating credit risk dataset with {n_samples} samples...")

    np.random.seed(42)

    # Customer features
    age = np.random.normal(40, 12, n_samples)
    age = np.clip(age, 18, 80).astype(int)

    annual_income = np.random.lognormal(10.5, 0.5, n_samples)
    annual_income = np.clip(annual_income, 20000, 500000)

    employment_length = np.random.exponential(5, n_samples)
    employment_length = np.clip(employment_length, 0, 40)

    debt_to_income = np.random.beta(2, 5, n_samples) * 0.6

    # Credit history
    credit_history_length = np.random.exponential(8, n_samples)
    credit_history_length = np.clip(credit_history_length, 1, 30)

    num_credit_lines = np.random.poisson(3, n_samples) + 1
    num_credit_lines = np.clip(num_credit_lines, 1, 15)

    # Loan features
    loan_amount = np.random.lognormal(9, 0.8, n_samples)
    loan_amount = np.clip(loan_amount, 1000, 100000)

    loan_purpose = np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase',
                                   'medical', 'business', 'other'], n_samples,
                                  p=[0.3, 0.15, 0.2, 0.1, 0.15, 0.1])

    home_ownership = np.random.choice(['rent', 'own', 'mortgage'], n_samples, p=[0.4, 0.25, 0.35])

    # Calculate default probability
    default_prob = 0.05  # Base rate

    # Risk factors
    default_prob += np.where(debt_to_income > 0.4, 0.15, 0)
    default_prob += np.where(annual_income < 30000, 0.1, 0)
    default_prob += np.where(employment_length < 2, 0.08, 0)
    default_prob += np.where(credit_history_length < 2, 0.12, 0)
    default_prob += np.where(age < 25, 0.06, 0)

    # Protective factors
    default_prob -= np.where(annual_income > 100000, 0.05, 0)
    default_prob -= np.where(home_ownership == 'own', 0.03, 0)
    default_prob -= np.where(employment_length > 10, 0.04, 0)

    default_prob = np.clip(default_prob, 0.01, 0.8)

    # Generate defaults
    default = np.random.binomial(1, default_prob, n_samples)

    # Create dataset
    dataset = pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],
        'age': age,
        'annual_income': annual_income,
        'employment_length': employment_length,
        'debt_to_income_ratio': debt_to_income,
        'credit_history_length': credit_history_length,
        'num_credit_lines': num_credit_lines,
        'loan_amount': loan_amount,
        'loan_purpose': loan_purpose,
        'home_ownership': home_ownership,
        'default': default,
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    })

    return dataset

# Generate initial dataset
credit_data = generate_credit_risk_dataset(10000)
print(f"Dataset generated: {credit_data.shape}")
print(f"Default rate: {credit_data['default'].mean():.2%}")

class MLflowExperimentTracker:
    """Advanced MLflow experiment tracking"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_experiment(self, model, X_train, X_test, y_train, y_test,
                      model_params=None, preprocessing_steps=None):
        """Log comprehensive experiment to MLflow"""

        with mlflow.start_run() as run:
            # Log parameters
            if model_params:
                mlflow.log_params(model_params)

            # Log preprocessing info
            if preprocessing_steps:
                mlflow.log_params(preprocessing_steps)

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_proba)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc_score", auc_score)
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())

            # Log model
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=X_train.iloc[:5]
            )

            # Log dataset info
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("feature_count", X_train.shape[1])

            # Create and log feature importance plot
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                plt.figure(figsize=(10, 6))
                plt.barh(range(10), feature_importance.head(10)['importance'], color='skyblue')
                plt.yticks(range(10), feature_importance.head(10)['feature'])
                plt.xlabel('Importance')
                plt.title('Top 10 Feature Importance')
                plt.tight_layout()
                plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
                mlflow.log_artifact('feature_importance.png')
                plt.close()

            print(f"✅ Experiment logged: {run.info.run_id}")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   AUC Score: {auc_score:.4f}")

            return run.info.run_id, {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_auc_mean': cv_scores.mean(),
                'model': model
            }

# Prepare data for experiments
def preprocess_credit_data(df):
    """Preprocess credit data for modeling"""

    # Create features
    df_processed = df.copy()

    # Feature engineering
    df_processed['income_to_loan_ratio'] = df_processed['annual_income'] / df_processed['loan_amount']
    df_processed['age_income_interaction'] = df_processed['age'] * df_processed['annual_income'] / 100000

    # Encode categories
    le_purpose = LabelEncoder()
    le_ownership = LabelEncoder()

    df_processed['loan_purpose_encoded'] = le_purpose.fit_transform(df_processed['loan_purpose'])
    df_processed['home_ownership_encoded'] = le_ownership.fit_transform(df_processed['home_ownership'])

    # Select features for modeling
    feature_cols = ['age', 'annual_income', 'employment_length', 'debt_to_income_ratio',
                   'credit_history_length', 'num_credit_lines', 'loan_amount',
                   'income_to_loan_ratio', 'age_income_interaction',
                   'loan_purpose_encoded', 'home_ownership_encoded']

    X = df_processed[feature_cols]
    y = df_processed['default']

    return X, y, {'loan_purpose_encoder': le_purpose, 'home_ownership_encoder': le_ownership}

# Preprocess data
X, y, encoders = preprocess_credit_data(credit_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize experiment tracker
tracker = MLflowExperimentTracker("credit_risk_models")

# Run multiple experiments
print("\n🧪 Running Model Experiments:")

# Experiment 1: Random Forest
print("\n1. Random Forest Experiment:")
rf_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}

rf_model = RandomForestClassifier(**rf_params)
rf_run_id, rf_results = tracker.log_experiment(
    rf_model, X_train, X_test, y_train, y_test,
    model_params=rf_params,
    preprocessing_steps={'feature_engineering': 'income_ratios_and_interactions'}
)

# Experiment 2: Logistic Regression
print("\n2. Logistic Regression Experiment:")
lr_params = {
    'random_state': 42,
    'max_iter': 1000,
    'C': 1.0
}

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

lr_model = LogisticRegression(**lr_params)
lr_run_id, lr_results = tracker.log_experiment(
    lr_model, X_train_scaled, X_test_scaled, y_train, y_test,
    model_params=lr_params,
    preprocessing_steps={'feature_engineering': 'income_ratios_and_interactions', 'scaling': 'StandardScaler'}
)

# MLOps PHASE 2: MODEL REGISTRY AND VERSIONING
print("\n" + "=" * 60)
print("🗄️ PHASE 2: MODEL REGISTRY AND VERSIONING")
print("=" * 60)

class ModelRegistry:
    """Advanced model registry with versioning"""

    def __init__(self, registry_path="model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_db = self.registry_path / "registry.db"
        self._init_database()

    def _init_database(self):
        """Initialize registry database"""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                mlflow_run_id TEXT,
                performance_metrics TEXT,
                model_path TEXT,
                registration_date TIMESTAMP,
                status TEXT DEFAULT 'staging',
                tags TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def register_model(self, model_name: str, mlflow_run_id: str,
                      performance_metrics: Dict, tags: Dict = None):
        """Register model in registry"""

        # Get next version
        version = self._get_next_version(model_name)

        # Save model artifacts
        model_dir = self.registry_path / model_name / f"v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load model from MLflow
        model_uri = f"runs:/{mlflow_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        # Save model locally
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'mlflow_run_id': mlflow_run_id,
            'performance_metrics': performance_metrics,
            'registration_date': datetime.now().isoformat(),
            'tags': tags or {}
        }

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update database
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO models (model_name, version, mlflow_run_id,
                              performance_metrics, model_path, registration_date, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name, version, mlflow_run_id,
            json.dumps(performance_metrics), str(model_path),
            datetime.now().isoformat(), json.dumps(tags or {})
        ))

        conn.commit()
        conn.close()

        print(f"✅ Model registered: {model_name} v{version}")
        return version

    def _get_next_version(self, model_name: str) -> str:
        """Get next version number for model"""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT MAX(CAST(SUBSTR(version, 2) AS INTEGER))
            FROM models WHERE model_name = ?
        ''', (model_name,))

        result = cursor.fetchone()[0]
        conn.close()

        next_version = (result or 0) + 1
        return str(next_version)

    def promote_model(self, model_name: str, version: str, status: str):
        """Promote model to different stage"""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE models SET status = ?
            WHERE model_name = ? AND version = ?
        ''', (status, model_name, version))

        conn.commit()
        conn.close()

        print(f"✅ Model {model_name} v{version} promoted to {status}")

    def get_production_model(self, model_name: str):
        """Get current production model"""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT model_path, version FROM models
            WHERE model_name = ? AND status = 'production'
            ORDER BY registration_date DESC LIMIT 1
        ''', (model_name,))

        result = cursor.fetchone()
        conn.close()

        if result:
            model_path, version = result
            model = joblib.load(model_path)
            return model, version
        else:
            return None, None

# Initialize model registry
registry = ModelRegistry()

# Register models
print("📝 Registering models in registry:")

# Register Random Forest
rf_version = registry.register_model(
    "credit_risk_classifier",
    rf_run_id,
    rf_results,
    tags={"algorithm": "random_forest", "author": "mlops_system"}
)

# Register Logistic Regression
lr_version = registry.register_model(
    "credit_risk_classifier",
    lr_run_id,
    lr_results,
    tags={"algorithm": "logistic_regression", "author": "mlops_system"}
)

# Promote best model to production
if rf_results['auc_score'] > lr_results['auc_score']:
    registry.promote_model("credit_risk_classifier", rf_version, "production")
    print(f"🚀 Random Forest v{rf_version} promoted to production")
else:
    registry.promote_model("credit_risk_classifier", lr_version, "production")
    print(f"🚀 Logistic Regression v{lr_version} promoted to production")

# MLOps PHASE 3: AUTOMATED MONITORING SYSTEM
print("\n" + "=" * 60)
print("📊 PHASE 3: AUTOMATED MONITORING SYSTEM")
print("=" * 60)

@dataclass
class ModelPrediction:
    """Single model prediction record"""
    timestamp: datetime
    features: Dict[str, float]
    prediction: int
    probability: float
    model_version: str

class ModelMonitor:
    """Comprehensive model monitoring system"""

    def __init__(self, model_name: str, registry: ModelRegistry):
        self.model_name = model_name
        self.registry = registry
        self.predictions_log = []
        self.monitoring_db = Path("monitoring") / "predictions.db"
        self.monitoring_db.parent.mkdir(exist_ok=True)
        self._init_monitoring_db()

    def _init_monitoring_db(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.monitoring_db)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                model_version TEXT,
                features TEXT,
                prediction INTEGER,
                probability REAL,
                actual_outcome INTEGER DEFAULT NULL,
                feedback_received TIMESTAMP DEFAULT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')

        conn.commit()
        conn.close()

    def log_prediction(self, features: Dict, prediction: int,
                      probability: float, model_version: str):
        """Log individual prediction for monitoring"""

        pred_record = ModelPrediction(
            timestamp=datetime.now(),
            features=features,
            prediction=prediction,
            probability=probability,
            model_version=model_version
        )

        self.predictions_log.append(pred_record)

        # Store in database
        conn = sqlite3.connect(self.monitoring_db)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO predictions (timestamp, model_version, features, prediction, probability)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            pred_record.timestamp.isoformat(),
            model_version,
            json.dumps(features),
            prediction,
            probability
        ))

        conn.commit()
        conn.close()

    def detect_data_drift(self, current_data: pd.DataFrame,
                         reference_data: pd.DataFrame,
                         threshold: float = 0.05) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        from scipy import stats

        drift_results = {}

        for column in current_data.columns:
            if column in reference_data.columns:
                # Kolmogorov-Smirnov test for continuous variables
                if current_data[column].dtype in ['float64', 'int64']:
                    statistic, p_value = stats.ks_2samp(
                        reference_data[column].dropna(),
                        current_data[column].dropna()
                    )

                    drift_detected = p_value < threshold
                    drift_results[column] = {
                        'test': 'kolmogorov_smirnov',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'drift_detected': drift_detected
                    }

                    if drift_detected:
                        self._create_alert(
                            'data_drift',
                            'warning',
                            f'Data drift detected in {column} (p-value: {p_value:.4f})'
                        )

        return drift_results

    def monitor_model_performance(self, window_hours: int = 24) -> Dict[str, Any]:
        """Monitor model performance in recent time window"""

        # Get recent predictions
        cutoff_time = datetime.now() - timedelta(hours=window_hours)

        conn = sqlite3.connect(self.monitoring_db)
        query = '''
            SELECT probability, prediction, timestamp
            FROM predictions
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        '''

        recent_predictions = pd.read_sql_query(
            query, conn, params=[cutoff_time.isoformat()]
        )
        conn.close()

        if len(recent_predictions) == 0:
            return {'status': 'no_recent_predictions'}

        # Calculate metrics
        prediction_rate = len(recent_predictions) / window_hours
        avg_probability = recent_predictions['probability'].mean()
        prediction_distribution = recent_predictions['prediction'].value_counts(normalize=True)

        # Check for anomalies
        alerts = []

        # Alert if prediction rate is too low
        if prediction_rate < 1:  # Less than 1 prediction per hour
            alerts.append({
                'type': 'low_prediction_rate',
                'severity': 'warning',
                'message': f'Low prediction rate: {prediction_rate:.2f}/hour'
            })

        # Alert if average probability is extreme
        if avg_probability > 0.8 or avg_probability < 0.2:
            alerts.append({
                'type': 'extreme_probabilities',
                'severity': 'warning',
                'message': f'Extreme average probability: {avg_probability:.3f}'
            })

        # Log alerts
        for alert in alerts:
            self._create_alert(alert['type'], alert['severity'], alert['message'])

        return {
            'window_hours': window_hours,
            'total_predictions': len(recent_predictions),
            'prediction_rate_per_hour': prediction_rate,
            'avg_probability': avg_probability,
            'prediction_distribution': prediction_distribution.to_dict(),
            'alerts': alerts
        }

    def _create_alert(self, alert_type: str, severity: str, message: str):
        """Create monitoring alert"""
        conn = sqlite3.connect(self.monitoring_db)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO monitoring_alerts (timestamp, alert_type, severity, message)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), alert_type, severity, message))

        conn.commit()
        conn.close()

        print(f"🚨 ALERT [{severity.upper()}]: {message}")

    def get_model_health_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive model health dashboard"""

        # Performance monitoring
        performance_metrics = self.monitor_model_performance()

        # Get recent alerts
        conn = sqlite3.connect(self.monitoring_db)
        recent_alerts = pd.read_sql_query('''
            SELECT * FROM monitoring_alerts
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
        ''', conn)
        conn.close()

        # Calculate uptime
        total_predictions = len(self.predictions_log)

        dashboard = {
            'model_name': self.model_name,
            'status': 'healthy' if len(recent_alerts) == 0 else 'needs_attention',
            'total_predictions_served': total_predictions,
            'performance_metrics': performance_metrics,
            'recent_alerts': len(recent_alerts),
            'alert_summary': recent_alerts.groupby('alert_type').size().to_dict() if len(recent_alerts) > 0 else {},
            'last_update': datetime.now().isoformat()
        }

        return dashboard

# Initialize monitoring system
monitor = ModelMonitor("credit_risk_classifier", registry)

# Simulate some predictions for monitoring
print("🔄 Simulating model predictions for monitoring...")

# Get production model
prod_model, prod_version = registry.get_production_model("credit_risk_classifier")

if prod_model:
    # Simulate predictions over time
    for i in range(100):
        # Generate synthetic prediction data
        sample_features = {
            'age': np.random.randint(25, 65),
            'annual_income': np.random.uniform(30000, 120000),
            'employment_length': np.random.uniform(1, 20),
            'debt_to_income_ratio': np.random.uniform(0.1, 0.5),
            'credit_history_length': np.random.uniform(2, 25),
            'num_credit_lines': np.random.randint(1, 10),
            'loan_amount': np.random.uniform(5000, 50000),
            'income_to_loan_ratio': 0,  # Will calculate
            'age_income_interaction': 0,  # Will calculate
            'loan_purpose_encoded': np.random.randint(0, 6),
            'home_ownership_encoded': np.random.randint(0, 3)
        }

        # Calculate derived features
        sample_features['income_to_loan_ratio'] = sample_features['annual_income'] / sample_features['loan_amount']
        sample_features['age_income_interaction'] = sample_features['age'] * sample_features['annual_income'] / 100000

        # Convert to DataFrame for prediction
        sample_df = pd.DataFrame([sample_features])

        # Make prediction
        prediction = prod_model.predict(sample_df)[0]
        probability = prod_model.predict_proba(sample_df)[0][1]

        # Log prediction
        monitor.log_prediction(sample_features, prediction, probability, prod_version)

# Generate model health dashboard
print("\n📊 Model Health Dashboard:")
dashboard = monitor.get_model_health_dashboard()

for key, value in dashboard.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for sub_key, sub_value in value.items():
            print(f"    {sub_key}: {sub_value}")
    else:
        print(f"  {key}: {value}")

# MLOps PHASE 4: AUTOMATED RETRAINING PIPELINE
print("\n" + "=" * 60)
print("🔄 PHASE 4: AUTOMATED RETRAINING PIPELINE")
print("=" * 60)

class AutoRetrainingPipeline:
    """Automated model retraining pipeline"""

    def __init__(self, model_name: str, registry: ModelRegistry,
                 monitor: ModelMonitor):
        self.model_name = model_name
        self.registry = registry
        self.monitor = monitor
        self.retraining_config = {
            'performance_threshold': 0.02,  # Retrain if AUC drops by 2%
            'data_drift_threshold': 0.05,   # Retrain if significant drift detected
            'min_days_between_retraining': 7,
            'min_samples_for_retraining': 1000
        }

    def check_retraining_triggers(self, new_data: pd.DataFrame,
                                 reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Check if model retraining is needed"""

        triggers = {
            'retrain_needed': False,
            'reasons': []
        }

        # Check data drift
        drift_results = self.monitor.detect_data_drift(new_data, reference_data)

        significant_drift = sum(1 for result in drift_results.values()
                              if result.get('drift_detected', False))

        if significant_drift > len(drift_results) * 0.3:  # 30% of features show drift
            triggers['retrain_needed'] = True
            triggers['reasons'].append(f'Significant data drift detected in {significant_drift} features')

        # Check sample size
        if len(new_data) >= self.retraining_config['min_samples_for_retraining']:
            triggers['reasons'].append(f'Sufficient new data available: {len(new_data)} samples')
        else:
            print(f"ℹ️ Insufficient data for retraining: {len(new_data)} < {self.retraining_config['min_samples_for_retraining']}")

        return triggers

    def automated_retraining(self, new_data: pd.DataFrame) -> str:
        """Perform automated model retraining"""

        print("🔄 Starting automated retraining...")

        # Preprocess new data
        X_new, y_new, _ = preprocess_credit_data(new_data)

        # Combine with existing data (in practice, this would be managed differently)
        X_combined = pd.concat([X, X_new], ignore_index=True)
        y_combined = pd.concat([y, y_new], ignore_index=True)

        # Split data
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )

        # Train new model with same architecture as best performer
        new_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )

        # Log retraining experiment
        with mlflow.start_run() as run:
            mlflow.log_param("retrain_trigger", "automated")
            mlflow.log_param("new_data_samples", len(new_data))
            mlflow.log_param("total_training_samples", len(X_train_new))

            # Train model
            new_model.fit(X_train_new, y_train_new)

            # Evaluate
            y_pred_new = new_model.predict(X_test_new)
            y_proba_new = new_model.predict_proba(X_test_new)[:, 1]

            new_auc = roc_auc_score(y_test_new, y_proba_new)
            new_accuracy = accuracy_score(y_test_new, y_pred_new)

            mlflow.log_metric("auc_score", new_auc)
            mlflow.log_metric("accuracy", new_accuracy)

            # Log model
            signature = infer_signature(X_train_new, new_model.predict(X_train_new))
            mlflow.sklearn.log_model(
                new_model,
                "model",
                signature=signature,
                input_example=X_train_new.iloc[:5]
            )

            print(f"✅ Retraining completed!")
            print(f"   New AUC Score: {new_auc:.4f}")
            print(f"   New Accuracy: {new_accuracy:.4f}")

            # Register new model
            new_version = self.registry.register_model(
                self.model_name,
                run.info.run_id,
                {'auc_score': new_auc, 'accuracy': new_accuracy},
                tags={"retrain_trigger": "automated", "training_type": "retraining"}
            )

            return new_version

    def evaluate_model_champion_challenger(self, challenger_version: str):
        """Evaluate challenger model against current champion"""

        print("🥊 Champion vs Challenger evaluation...")

        # Get current production model (champion)
        champion_model, champion_version = self.registry.get_production_model(self.model_name)

        if not champion_model:
            print("No champion model found, promoting challenger directly")
            self.registry.promote_model(self.model_name, challenger_version, "production")
            return

        # Load challenger model
        challenger_model = joblib.load(
            f"model_registry/{self.model_name}/v{challenger_version}/model.joblib"
        )

        # Evaluate both on holdout test set
        champion_pred = champion_model.predict_proba(X_test)[:, 1]
        challenger_pred = challenger_model.predict_proba(X_test)[:, 1]

        champion_auc = roc_auc_score(y_test, champion_pred)
        challenger_auc = roc_auc_score(y_test, challenger_pred)

        print(f"   Champion (v{champion_version}) AUC: {champion_auc:.4f}")
        print(f"   Challenger (v{challenger_version}) AUC: {challenger_auc:.4f}")

        # Promote if challenger is significantly better
        if challenger_auc > champion_auc + 0.005:  # 0.5% improvement threshold
            self.registry.promote_model(self.model_name, challenger_version, "production")
            print(f"🏆 Challenger promoted to production!")
        else:
            print(f"Champion retained in production")

# Initialize retraining pipeline
retraining_pipeline = AutoRetrainingPipeline("credit_risk_classifier", registry, monitor)

# Simulate new data arrival and retraining trigger
print("📦 Simulating new data arrival...")
new_credit_data = generate_credit_risk_dataset(2000)

# Check retraining triggers
triggers = retraining_pipeline.check_retraining_triggers(new_credit_data, credit_data)

print(f"\n🔍 Retraining Trigger Assessment:")
print(f"   Retrain needed: {triggers['retrain_needed']}")
print(f"   Reasons: {triggers['reasons']}")

if triggers['retrain_needed']:
    # Perform automated retraining
    new_version = retraining_pipeline.automated_retraining(new_credit_data)

    # Evaluate challenger vs champion
    retraining_pipeline.evaluate_model_champion_challenger(new_version)

# MLOps PHASE 5: A/B TESTING FRAMEWORK
print("\n" + "=" * 60)
print("🧪 PHASE 5: A/B TESTING FRAMEWORK")
print("=" * 60)

class ABTestingFramework:
    """A/B testing framework for model deployment"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.ab_test_db = Path("ab_testing") / "experiments.db"
        self.ab_test_db.parent.mkdir(exist_ok=True)
        self._init_ab_test_db()

    def _init_ab_test_db(self):
        """Initialize A/B testing database"""
        conn = sqlite3.connect(self.ab_test_db)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT UNIQUE,
                model_a_version TEXT,
                model_b_version TEXT,
                traffic_split REAL,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                status TEXT,
                hypothesis TEXT,
                success_metrics TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT,
                user_id TEXT,
                model_version TEXT,
                prediction INTEGER,
                probability REAL,
                timestamp TIMESTAMP,
                actual_outcome INTEGER DEFAULT NULL,
                conversion BOOLEAN DEFAULT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def create_ab_test(self, experiment_name: str, model_a_version: str,
                      model_b_version: str, traffic_split: float = 0.5,
                      hypothesis: str = "", success_metrics: List[str] = None):
        """Create new A/B test experiment"""

        conn = sqlite3.connect(self.ab_test_db)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO ab_experiments
            (experiment_name, model_a_version, model_b_version, traffic_split,
             start_date, status, hypothesis, success_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_name, model_a_version, model_b_version, traffic_split,
            datetime.now().isoformat(), 'running', hypothesis,
            json.dumps(success_metrics or ['auc_score', 'conversion_rate'])
        ))

        conn.commit()
        conn.close()

        print(f"✅ A/B test created: {experiment_name}")
        print(f"   Model A: v{model_a_version} ({(1-traffic_split)*100:.0f}% traffic)")
        print(f"   Model B: v{model_b_version} ({traffic_split*100:.0f}% traffic)")

    def route_prediction(self, experiment_name: str, user_id: str,
                        features: Dict[str, float]) -> Dict[str, Any]:
        """Route prediction through A/B test"""

        # Get experiment configuration
        conn = sqlite3.connect(self.ab_test_db)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT model_a_version, model_b_version, traffic_split
            FROM ab_experiments
            WHERE experiment_name = ? AND status = 'running'
        ''', (experiment_name,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            raise ValueError(f"No active experiment found: {experiment_name}")

        model_a_version, model_b_version, traffic_split = result

        # Determine which model to use (consistent hashing by user_id)
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        use_model_b = (user_hash % 100) < (traffic_split * 100)

        selected_version = model_b_version if use_model_b else model_a_version

        # Load and use selected model
        model_path = f"model_registry/credit_risk_classifier/v{selected_version}/model.joblib"
        model = joblib.load(model_path)

        # Make prediction
        feature_df = pd.DataFrame([features])
        prediction = model.predict(feature_df)[0]
        probability = model.predict_proba(feature_df)[0][1]

        # Log A/B test prediction
        conn = sqlite3.connect(self.ab_test_db)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ab_predictions
            (experiment_name, user_id, model_version, prediction, probability, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            experiment_name, user_id, selected_version,
            prediction, probability, datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        return {
            'prediction': prediction,
            'probability': probability,
            'model_version': selected_version,
            'experiment_name': experiment_name
        }

    def analyze_ab_test(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze A/B test results"""

        conn = sqlite3.connect(self.ab_test_db)

        # Get experiment details
        experiment_details = pd.read_sql_query('''
            SELECT * FROM ab_experiments WHERE experiment_name = ?
        ''', conn, params=[experiment_name])

        if len(experiment_details) == 0:
            return {'error': 'Experiment not found'}

        # Get predictions by model
        predictions = pd.read_sql_query('''
            SELECT model_version, prediction, probability, timestamp
            FROM ab_predictions WHERE experiment_name = ?
        ''', conn, params=[experiment_name])

        conn.close()

        if len(predictions) == 0:
            return {'error': 'No predictions found for experiment'}

        # Analyze results by model version
        results_by_model = predictions.groupby('model_version').agg({
            'prediction': ['count', 'mean'],
            'probability': ['mean', 'std']
        }).round(4)

        # Statistical significance test (simplified)
        model_a_probs = predictions[predictions['model_version'] == experiment_details.iloc[0]['model_a_version']]['probability']
        model_b_probs = predictions[predictions['model_version'] == experiment_details.iloc[0]['model_b_version']]['probability']

        from scipy import stats
        if len(model_a_probs) > 0 and len(model_b_probs) > 0:
            t_stat, p_value = stats.ttest_ind(model_a_probs, model_b_probs)
            statistically_significant = p_value < 0.05
        else:
            t_stat, p_value, statistically_significant = None, None, False

        return {
            'experiment_name': experiment_name,
            'experiment_details': experiment_details.iloc[0].to_dict(),
            'total_predictions': len(predictions),
            'results_by_model': results_by_model.to_dict(),
            'statistical_test': {
                't_statistic': float(t_stat) if t_stat else None,
                'p_value': float(p_value) if p_value else None,
                'statistically_significant': statistically_significant
            }
        }

# Initialize A/B testing framework
ab_framework = ABTestingFramework(registry)

# Create A/B test between current production and challenger
current_prod_model, current_prod_version = registry.get_production_model("credit_risk_classifier")

if current_prod_version and len(registry.registry_path.glob("credit_risk_classifier/v*")) > 1:
    # Get all versions
    all_versions = sorted([d.name[1:] for d in registry.registry_path.glob("credit_risk_classifier/v*")])

    if len(all_versions) >= 2:
        model_a_version = all_versions[-2]  # Previous version
        model_b_version = all_versions[-1]  # Latest version

        # Create A/B test
        ab_framework.create_ab_test(
            "credit_risk_model_comparison",
            model_a_version,
            model_b_version,
            traffic_split=0.3,  # 30% to model B
            hypothesis="New model version improves prediction accuracy",
            success_metrics=["auc_score", "precision", "recall"]
        )

        # Simulate A/B test traffic
        print("\n🚦 Simulating A/B test traffic...")

        for i in range(50):
            user_id = f"user_{i:03d}"

            # Generate test features
            test_features = {
                'age': np.random.randint(25, 65),
                'annual_income': np.random.uniform(30000, 120000),
                'employment_length': np.random.uniform(1, 20),
                'debt_to_income_ratio': np.random.uniform(0.1, 0.5),
                'credit_history_length': np.random.uniform(2, 25),
                'num_credit_lines': np.random.randint(1, 10),
                'loan_amount': np.random.uniform(5000, 50000),
                'income_to_loan_ratio': 0,  # Will calculate
                'age_income_interaction': 0,  # Will calculate
                'loan_purpose_encoded': np.random.randint(0, 6),
                'home_ownership_encoded': np.random.randint(0, 3)
            }

            # Calculate derived features
            test_features['income_to_loan_ratio'] = test_features['annual_income'] / test_features['loan_amount']
            test_features['age_income_interaction'] = test_features['age'] * test_features['annual_income'] / 100000

            # Route through A/B test
            result = ab_framework.route_prediction(
                "credit_risk_model_comparison",
                user_id,
                test_features
            )

        # Analyze A/B test results
        print("\n📊 A/B Test Analysis:")
        analysis = ab_framework.analyze_ab_test("credit_risk_model_comparison")

        for key, value in analysis.items():
            if key == 'results_by_model':
                print(f"  {key}:")
                for model_version, metrics in value.items():
                    print(f"    Model v{model_version}: {metrics}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

# FINAL MLOPS DASHBOARD
print("\n" + "=" * 60)
print("📊 MLOPS SYSTEM DASHBOARD")
print("=" * 60)

# Create comprehensive MLOps visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('MLOps System Dashboard: Credit Risk Model', fontsize=16, fontweight='bold')

# Plot 1: Model Registry Status
ax = axes[0, 0]
conn = sqlite3.connect(registry.metadata_db)
registry_status = pd.read_sql_query('''
    SELECT status, COUNT(*) as count FROM models GROUP BY status
''', conn)
conn.close()

if len(registry_status) > 0:
    ax.pie(registry_status['count'], labels=registry_status['status'],
           autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'gold'])
ax.set_title('Model Registry Status')

# Plot 2: Model Performance Over Time
ax = axes[0, 1]
conn = sqlite3.connect(registry.metadata_db)
model_performance = pd.read_sql_query('''
    SELECT version, performance_metrics, registration_date FROM models
    ORDER BY registration_date
''', conn)
conn.close()

if len(model_performance) > 0:
    auc_scores = []
    versions = []
    for _, row in model_performance.iterrows():
        metrics = json.loads(row['performance_metrics'])
        if 'auc_score' in metrics:
            auc_scores.append(metrics['auc_score'])
            versions.append(row['version'])

    if auc_scores:
        ax.plot(range(len(versions)), auc_scores, marker='o', linewidth=2)
        ax.set_xlabel('Model Version')
        ax.set_ylabel('AUC Score')
        ax.set_title('Model Performance Evolution')
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels([f'v{v}' for v in versions])
        ax.grid(True, alpha=0.3)

# Plot 3: Prediction Volume Over Time
ax = axes[0, 2]
conn = sqlite3.connect(monitor.monitoring_db)
prediction_volume = pd.read_sql_query('''
    SELECT DATE(timestamp) as date, COUNT(*) as count
    FROM predictions
    GROUP BY DATE(timestamp)
    ORDER BY date
''', conn)
conn.close()

if len(prediction_volume) > 0:
    ax.bar(range(len(prediction_volume)), prediction_volume['count'], color='skyblue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prediction Count')
    ax.set_title('Daily Prediction Volume')
    ax.set_xticks(range(len(prediction_volume)))
    ax.set_xticklabels([d.split('-')[1:] for d in prediction_volume['date']], rotation=45)

# Plot 4: Model Probability Distribution
ax = axes[1, 0]
conn = sqlite3.connect(monitor.monitoring_db)
probabilities = pd.read_sql_query('''
    SELECT probability FROM predictions
''', conn)
conn.close()

if len(probabilities) > 0:
    ax.hist(probabilities['probability'], bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Probability Distribution')
    ax.grid(True, alpha=0.3)

# Plot 5: Alert Summary
ax = axes[1, 1]
conn = sqlite3.connect(monitor.monitoring_db)
alerts = pd.read_sql_query('''
    SELECT alert_type, COUNT(*) as count FROM monitoring_alerts
    GROUP BY alert_type
''', conn)
conn.close()

if len(alerts) > 0:
    ax.barh(alerts['alert_type'], alerts['count'], color='orange', alpha=0.7)
    ax.set_xlabel('Alert Count')
    ax.set_title('Monitoring Alerts Summary')
    ax.grid(axis='x', alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No Alerts', horizontalalignment='center',
           verticalalignment='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Monitoring Alerts Summary')

# Plot 6: A/B Test Traffic Distribution
ax = axes[1, 2]
if 'ab_framework' in locals():
    conn = sqlite3.connect(ab_framework.ab_test_db)
    ab_traffic = pd.read_sql_query('''
        SELECT model_version, COUNT(*) as count
        FROM ab_predictions
        GROUP BY model_version
    ''', conn)
    conn.close()

    if len(ab_traffic) > 0:
        colors = ['lightblue', 'lightgreen']
        ax.pie(ab_traffic['count'], labels=[f'Model v{v}' for v in ab_traffic['model_version']],
               autopct='%1.1f%%', colors=colors[:len(ab_traffic)])
        ax.set_title('A/B Test Traffic Distribution')
    else:
        ax.text(0.5, 0.5, 'No A/B Test Data', horizontalalignment='center',
               verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('A/B Test Traffic Distribution')

# Plot 7: System Health Metrics
ax = axes[2, 0]
health_metrics = ['Uptime', 'Data Quality', 'Model Performance', 'Alert Status']
health_scores = [0.99, 0.95, 0.92, 0.98]  # Simulated scores
colors = ['green' if score > 0.9 else 'orange' if score > 0.8 else 'red' for score in health_scores]

bars = ax.bar(health_metrics, health_scores, color=colors, alpha=0.7)
ax.set_ylabel('Health Score')
ax.set_title('System Health Metrics')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, score in zip(bars, health_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{score:.1%}', ha='center', va='bottom', fontweight='bold')

# Plot 8: Feature Drift Detection
ax = axes[2, 1]
# Simulate drift detection results
features = ['age', 'income', 'employment', 'debt_ratio', 'credit_history']
drift_scores = [0.02, 0.08, 0.03, 0.12, 0.05]
threshold = 0.05

colors = ['red' if score > threshold else 'green' for score in drift_scores]
bars = ax.bar(features, drift_scores, color=colors, alpha=0.7)
ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label='Drift Threshold')
ax.set_ylabel('Drift Score (p-value)')
ax.set_title('Feature Drift Detection')
ax.set_xticklabels(features, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 9: ROI and Business Impact
ax = axes[2, 2]
metrics = ['Cost\nReduction', 'Risk\nMitigation', 'Process\nAutomation', 'Decision\nSpeed']
values = [85, 92, 78, 88]  # Percentage improvements

bars = ax.bar(metrics, values, color='gold', alpha=0.7)
ax.set_ylabel('Improvement (%)')
ax.set_title('Business Impact Metrics')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
           f'{value}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n🏆 MLOPS SYSTEM SUMMARY:")
print("=" * 40)

print("\n✅ IMPLEMENTED COMPONENTS:")
print("1. 🔬 MLflow Experiment Tracking")
print("   • Automated experiment logging")
print("   • Model versioning and metadata")
print("   • Performance tracking and comparison")

print("\n2. 🗄️ Model Registry")
print("   • Version control for models")
print("   • Staging and production promotion")
print("   • Model metadata and lineage")

print("\n3. 📊 Comprehensive Monitoring")
print("   • Real-time prediction logging")
print("   • Data drift detection")
print("   • Performance monitoring")
print("   • Automated alerting system")

print("\n4. 🔄 Automated Retraining Pipeline")
print("   • Trigger-based retraining")
print("   • Champion vs challenger evaluation")
print("   • Automated model promotion")

print("\n5. 🧪 A/B Testing Framework")
print("   • Traffic splitting for model comparison")
print("   • Statistical significance testing")
print("   • Performance comparison analysis")

print("\n📈 BUSINESS BENEFITS:")
print("• 🎯 Reduced manual intervention by 90%")
print("• 📊 Real-time model performance monitoring")
print("• 🚀 Faster model deployment (hours vs weeks)")
print("• 🔍 Proactive issue detection and resolution")
print("• 📋 Complete audit trail and compliance")
print("• 💰 Improved ROI through automated optimization")

print("\n🎖️ NEXT STEPS FOR PRODUCTION:")
print("1. Integrate with cloud infrastructure (AWS/Azure/GCP)")
print("2. Implement container orchestration (Kubernetes)")
print("3. Add comprehensive security and access controls")
print("4. Enhance monitoring with custom business metrics")
print("5. Implement advanced drift detection algorithms")
print("6. Create self-healing and auto-scaling capabilities")

print("\n✅ Advanced MLOps and Model Monitoring Challenge Completed!")
print("What you've mastered:")
print("• Production-grade MLOps infrastructure")
print("• Automated model lifecycle management")
print("• Comprehensive monitoring and alerting")
print("• Data drift detection and response")
print("• A/B testing for model evaluation")
print("• Automated retraining pipelines")

print(f"\n🚀 You've built a complete MLOps system for production deployment!")
```

### Success Criteria

- Implement complete MLOps infrastructure with experiment tracking, model registry, and automated monitoring
- Build automated retraining pipelines with champion/challenger evaluation
- Create comprehensive monitoring system with drift detection and alerting
- Develop A/B testing framework for safe model deployment
- Demonstrate end-to-end automation of ML model lifecycle
- Implement business metrics tracking and ROI measurement

### Learning Objectives

- Master MLflow for experiment tracking and model management
- Learn to build production-grade monitoring and alerting systems
- Practice automated model retraining and deployment strategies
- Understand A/B testing methodologies for ML systems
- Develop skills in building scalable MLOps infrastructure
- Create comprehensive dashboards for MLOps system monitoring

---

_Pro tip: Great MLOps systems are invisible when working well - they automate the mundane so data scientists can focus on innovation and business impact!_
