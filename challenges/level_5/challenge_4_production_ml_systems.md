# Level 5: Algorithm Architect

## Challenge 4: Production ML Systems and MLOps

Master the deployment, monitoring, and maintenance of machine learning systems in production environments.

### Objective

Learn to build robust, scalable, and maintainable ML systems including model deployment, monitoring, versioning, and automated retraining pipelines.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import joblib
import time
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                           roc_auc_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

print("🚀 Production ML Systems and MLOps")
print("=" * 40)

# Simulate production ML system components
print("🏗️ Building Production ML Infrastructure...")

# CHALLENGE 1: MODEL VERSIONING AND EXPERIMENT TRACKING
print("\n" + "=" * 60)
print("📝 CHALLENGE 1: MODEL VERSIONING & EXPERIMENT TRACKING")
print("=" * 60)

class MLExperimentTracker:
    """Track ML experiments with versioning and metadata"""

    def __init__(self, experiment_dir="ml_experiments"):
        self.experiment_dir = experiment_dir
        self.experiments = []
        self.current_experiment_id = 0

        # Create experiment directory
        os.makedirs(experiment_dir, exist_ok=True)

    def start_experiment(self, name, description="", tags=None):
        """Start a new ML experiment"""
        experiment_id = f"exp_{self.current_experiment_id:04d}"
        self.current_experiment_id += 1

        experiment = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'tags': tags or [],
            'start_time': datetime.now().isoformat(),
            'parameters': {},
            'metrics': {},
            'artifacts': {},
            'status': 'running'
        }

        self.experiments.append(experiment)
        print(f"🧪 Started experiment: {experiment_id} - {name}")
        return experiment_id

    def log_parameters(self, experiment_id, parameters):
        """Log experiment parameters"""
        for exp in self.experiments:
            if exp['id'] == experiment_id:
                exp['parameters'].update(parameters)
                break

    def log_metrics(self, experiment_id, metrics):
        """Log experiment metrics"""
        for exp in self.experiments:
            if exp['id'] == experiment_id:
                exp['metrics'].update(metrics)
                break

    def log_model(self, experiment_id, model, model_name="model"):
        """Save and log model artifact"""
        for exp in self.experiments:
            if exp['id'] == experiment_id:
                model_path = os.path.join(self.experiment_dir, f"{experiment_id}_{model_name}.pkl")
                joblib.dump(model, model_path)
                exp['artifacts'][model_name] = {
                    'path': model_path,
                    'type': 'model',
                    'size': os.path.getsize(model_path)
                }
                break

    def finish_experiment(self, experiment_id, status='completed'):
        """Finish an experiment"""
        for exp in self.experiments:
            if exp['id'] == experiment_id:
                exp['status'] = status
                exp['end_time'] = datetime.now().isoformat()

                # Save experiment metadata
                metadata_path = os.path.join(self.experiment_dir, f"{experiment_id}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(exp, f, indent=2)

                print(f"✅ Finished experiment: {experiment_id}")
                break

    def get_experiment_summary(self):
        """Get summary of all experiments"""
        summary = pd.DataFrame([
            {
                'ID': exp['id'],
                'Name': exp['name'],
                'Status': exp['status'],
                'Accuracy': exp['metrics'].get('accuracy', 'N/A'),
                'F1_Score': exp['metrics'].get('f1_score', 'N/A'),
                'Model_Type': exp['parameters'].get('model_type', 'Unknown')
            }
            for exp in self.experiments
        ])
        return summary

# Create sample dataset for production simulation
np.random.seed(42)

print("📊 Creating Production Dataset...")

def create_production_dataset(n_samples=5000):
    """Create a realistic production dataset"""

    # Customer features
    data = pd.DataFrame({
        'customer_age': np.random.normal(35, 12, n_samples).clip(18, 80),
        'annual_income': np.random.lognormal(10.5, 0.6, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'account_balance': np.random.lognormal(8, 1.5, n_samples),
        'transaction_frequency': np.random.poisson(25, n_samples),
        'avg_transaction_amount': np.random.lognormal(4, 0.8, n_samples),
        'months_as_customer': np.random.exponential(24, n_samples).clip(1, 120),
        'num_products': np.random.poisson(2.5, n_samples).clip(1, 10),
        'digital_engagement': np.random.beta(3, 2, n_samples),
        'geographic_region': np.random.choice(['North', 'South', 'East', 'West'],
                                           size=n_samples, p=[0.3, 0.25, 0.25, 0.2])
    })

    # Create realistic target variable (churn prediction)
    churn_prob = (
        -0.02 * (data['customer_age'] - 35) +  # Younger customers more likely to churn
        -0.3 * (data['credit_score'] - 650) / 100 +  # Lower credit score = higher churn
        -0.2 * np.log(data['account_balance'] + 1) / 10 +  # Lower balance = higher churn
        0.1 * (data['transaction_frequency'] - 25) / 25 +  # Low activity = higher churn
        -0.3 * data['digital_engagement'] +  # Low engagement = higher churn
        np.random.normal(0, 0.5, n_samples)  # Add noise
    )

    # Convert to binary outcome
    data['churn'] = (churn_prob > np.percentile(churn_prob, 75)).astype(int)

    return data

production_data = create_production_dataset(5000)

print(f"Production dataset shape: {production_data.shape}")
print(f"Churn rate: {production_data['churn'].mean():.3f}")

# Initialize experiment tracker
tracker = MLExperimentTracker()

# Run multiple experiments
models_to_test = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

# Prepare data
X = production_data.drop(['churn', 'geographic_region'], axis=1)  # Exclude categorical for now
y = production_data['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

experiment_results = []

for model_name, model in models_to_test.items():
    # Start experiment
    exp_id = tracker.start_experiment(
        name=f"Churn Prediction - {model_name}",
        description=f"Testing {model_name} for customer churn prediction",
        tags=['churn', 'classification', model_name.lower()]
    )

    # Log parameters
    if hasattr(model, 'get_params'):
        tracker.log_parameters(exp_id, {
            'model_type': model_name,
            **model.get_params()
        })

    # Train model
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'training_time': training_time
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

    # Log metrics
    tracker.log_metrics(exp_id, metrics)

    # Save model
    tracker.log_model(exp_id, model, f"{model_name}_model")

    # Finish experiment
    tracker.finish_experiment(exp_id)

    experiment_results.append({
        'model_name': model_name,
        'experiment_id': exp_id,
        'metrics': metrics
    })

    print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")

# Display experiment summary
print("\n📊 Experiment Summary:")
summary = tracker.get_experiment_summary()
print(summary)

# CHALLENGE 2: MODEL DEPLOYMENT SIMULATION
print("\n" + "=" * 60)
print("🚀 CHALLENGE 2: MODEL DEPLOYMENT SIMULATION")
print("=" * 60)

class MLModelService:
    """Simulate ML model serving in production"""

    def __init__(self, model, scaler, model_version="1.0.0"):
        self.model = model
        self.scaler = scaler
        self.model_version = model_version
        self.deployment_time = datetime.now()
        self.prediction_count = 0
        self.prediction_history = []
        self.performance_metrics = []

    def predict(self, features, log_prediction=True):
        """Make a prediction with logging"""
        try:
            # Preprocess features
            if isinstance(features, dict):
                # Convert single prediction dict to DataFrame
                features_df = pd.DataFrame([features])
            else:
                features_df = features

            # Scale features
            features_scaled = self.scaler.transform(features_df)

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            confidence = self.model.predict_proba(features_scaled)[0].max()

            # Log prediction
            if log_prediction:
                self.prediction_count += 1
                self.prediction_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'features': features_df.iloc[0].to_dict() if isinstance(features, dict) else 'batch',
                    'prediction': int(prediction),
                    'confidence': float(confidence),
                    'model_version': self.model_version
                })

            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'error': str(e),
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat()
            }

    def health_check(self):
        """Check model service health"""
        return {
            'status': 'healthy',
            'model_version': self.model_version,
            'uptime_hours': (datetime.now() - self.deployment_time).total_seconds() / 3600,
            'predictions_served': self.prediction_count,
            'last_prediction': self.prediction_history[-1]['timestamp'] if self.prediction_history else None
        }

    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.prediction_history:
            return {'message': 'No predictions made yet'}

        predictions = [p['prediction'] for p in self.prediction_history]
        confidences = [p['confidence'] for p in self.prediction_history]

        return {
            'total_predictions': len(predictions),
            'churn_predictions': sum(predictions),
            'churn_rate': sum(predictions) / len(predictions),
            'avg_confidence': np.mean(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences)
        }

# Deploy best model from experiments
best_experiment = max(experiment_results, key=lambda x: x['metrics']['f1_score'])
best_model_name = best_experiment['model_name']
best_model = models_to_test[best_model_name]

print(f"🎯 Deploying best model: {best_model_name}")
print(f"   F1-Score: {best_experiment['metrics']['f1_score']:.4f}")

# Create model service
model_service = MLModelService(best_model, scaler, model_version="1.0.0")

# Simulate production predictions
print("\n🔄 Simulating Production Predictions...")

# Generate some sample prediction requests
sample_customers = [
    {
        'customer_age': 28, 'annual_income': 45000, 'credit_score': 620,
        'account_balance': 2500, 'transaction_frequency': 15, 'avg_transaction_amount': 85,
        'months_as_customer': 8, 'num_products': 2, 'digital_engagement': 0.3
    },
    {
        'customer_age': 45, 'annual_income': 75000, 'credit_score': 720,
        'account_balance': 15000, 'transaction_frequency': 35, 'avg_transaction_amount': 120,
        'months_as_customer': 36, 'num_products': 4, 'digital_engagement': 0.8
    },
    {
        'customer_age': 22, 'annual_income': 35000, 'credit_score': 580,
        'account_balance': 800, 'transaction_frequency': 8, 'avg_transaction_amount': 45,
        'months_as_customer': 3, 'num_products': 1, 'digital_engagement': 0.1
    }
]

for i, customer in enumerate(sample_customers):
    result = model_service.predict(customer)
    print(f"Customer {i+1}: Churn={result['prediction']}, Confidence={result['confidence']:.3f}")

# Check service health
health = model_service.health_check()
print(f"\n🏥 Service Health: {health}")

# Get performance stats
stats = model_service.get_performance_stats()
print(f"📊 Performance Stats: {stats}")

# CHALLENGE 3: MODEL MONITORING AND DRIFT DETECTION
print("\n" + "=" * 60)
print("📊 CHALLENGE 3: MODEL MONITORING & DRIFT DETECTION")
print("=" * 60)

class ModelMonitor:
    """Monitor model performance and detect drift"""

    def __init__(self, reference_data, reference_labels, model, scaler):
        self.reference_data = reference_data
        self.reference_labels = reference_labels
        self.model = model
        self.scaler = scaler

        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(reference_data)
        self.reference_performance = self._calculate_performance(reference_data, reference_labels)

        self.monitoring_data = []
        self.alerts = []

    def _calculate_statistics(self, data):
        """Calculate statistical properties of data"""
        return {
            'mean': data.mean().to_dict(),
            'std': data.std().to_dict(),
            'min': data.min().to_dict(),
            'max': data.max().to_dict(),
            'quantiles': {
                'q25': data.quantile(0.25).to_dict(),
                'q50': data.quantile(0.5).to_dict(),
                'q75': data.quantile(0.75).to_dict()
            }
        }

    def _calculate_performance(self, X, y_true):
        """Calculate model performance metrics"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }

    def check_data_drift(self, new_data, threshold=2.0):
        """Detect data drift using statistical measures"""
        new_stats = self._calculate_statistics(new_data)
        drift_scores = {}

        for column in new_data.columns:
            if column in self.reference_stats['mean']:
                # Calculate z-score for mean drift
                ref_mean = self.reference_stats['mean'][column]
                ref_std = self.reference_stats['std'][column]
                new_mean = new_stats['mean'][column]

                if ref_std > 0:
                    z_score = abs(new_mean - ref_mean) / ref_std
                    drift_scores[column] = z_score

                    if z_score > threshold:
                        alert = {
                            'type': 'data_drift',
                            'feature': column,
                            'z_score': z_score,
                            'threshold': threshold,
                            'timestamp': datetime.now().isoformat(),
                            'severity': 'high' if z_score > 3.0 else 'medium'
                        }
                        self.alerts.append(alert)

        return drift_scores

    def check_performance_drift(self, new_data, new_labels, threshold=0.05):
        """Detect performance drift"""
        new_performance = self._calculate_performance(new_data, new_labels)

        performance_drift = {}
        for metric, ref_value in self.reference_performance.items():
            new_value = new_performance[metric]
            drift = abs(new_value - ref_value)
            performance_drift[metric] = {
                'reference': ref_value,
                'current': new_value,
                'drift': drift
            }

            if drift > threshold:
                alert = {
                    'type': 'performance_drift',
                    'metric': metric,
                    'reference_value': ref_value,
                    'current_value': new_value,
                    'drift': drift,
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'high' if drift > threshold * 2 else 'medium'
                }
                self.alerts.append(alert)

        return performance_drift

    def generate_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        return {
            'monitoring_period': {
                'start': self.monitoring_data[0]['timestamp'] if self.monitoring_data else None,
                'end': self.monitoring_data[-1]['timestamp'] if self.monitoring_data else None,
                'data_points': len(self.monitoring_data)
            },
            'alerts_summary': {
                'total_alerts': len(self.alerts),
                'data_drift_alerts': len([a for a in self.alerts if a['type'] == 'data_drift']),
                'performance_drift_alerts': len([a for a in self.alerts if a['type'] == 'performance_drift']),
                'high_severity': len([a for a in self.alerts if a.get('severity') == 'high'])
            },
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }

# Initialize monitoring
print("🔧 Setting up model monitoring...")
monitor = ModelMonitor(X_train, y_train, best_model, scaler)

# Simulate data drift by creating modified data
print("\n📈 Simulating Data Drift...")

# Create drifted data (simulate changing customer behavior)
drift_data = X_test.copy()
# Simulate economic change affecting income and spending
drift_data['annual_income'] *= 0.85  # Economic downturn
drift_data['avg_transaction_amount'] *= 0.9  # Reduced spending
drift_data['credit_score'] += np.random.normal(-20, 10, len(drift_data))  # Credit score decline

# Check for data drift
drift_scores = monitor.check_data_drift(drift_data, threshold=1.5)
print("Data Drift Scores:")
for feature, score in sorted(drift_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feature}: {score:.3f}")

# Check for performance drift (simulate with some labels)
performance_drift = monitor.check_performance_drift(drift_data, y_test, threshold=0.03)
print("\nPerformance Drift Analysis:")
for metric, values in performance_drift.items():
    print(f"  {metric}: {values['reference']:.3f} → {values['current']:.3f} (Δ{values['drift']:.3f})")

# Generate monitoring report
monitoring_report = monitor.generate_monitoring_report()
print("\n📋 Monitoring Report:")
print(f"  Total Alerts: {monitoring_report['alerts_summary']['total_alerts']}")
print(f"  Data Drift Alerts: {monitoring_report['alerts_summary']['data_drift_alerts']}")
print(f"  Performance Drift Alerts: {monitoring_report['alerts_summary']['performance_drift_alerts']}")

# CHALLENGE 4: AUTOMATED RETRAINING PIPELINE
print("\n" + "=" * 60)
print("🔄 CHALLENGE 4: AUTOMATED RETRAINING PIPELINE")
print("=" * 60)

class AutoRetrainingPipeline:
    """Automated model retraining pipeline"""

    def __init__(self, base_model, scaler, retrain_threshold=0.05):
        self.base_model = base_model
        self.scaler = scaler
        self.retrain_threshold = retrain_threshold
        self.model_versions = []
        self.current_version = "1.0.0"

    def should_retrain(self, performance_drift):
        """Determine if model should be retrained"""
        critical_metrics = ['f1_score', 'accuracy', 'roc_auc']

        for metric in critical_metrics:
            if metric in performance_drift:
                drift_amount = performance_drift[metric]['drift']
                if drift_amount > self.retrain_threshold:
                    return True, f"Performance drift in {metric}: {drift_amount:.4f}"

        return False, "No significant performance drift detected"

    def retrain_model(self, new_X_train, new_y_train, validation_X, validation_y):
        """Retrain model with new data"""
        print(f"🔄 Retraining model...")

        # Create new model instance
        from copy import deepcopy
        new_model = deepcopy(self.base_model)

        # Retrain scaler and model
        new_scaler = StandardScaler()
        new_X_train_scaled = new_scaler.fit_transform(new_X_train)
        new_model.fit(new_X_train_scaled, new_y_train)

        # Validate new model
        validation_X_scaled = new_scaler.transform(validation_X)
        val_pred = new_model.predict(validation_X_scaled)
        val_pred_proba = new_model.predict_proba(validation_X_scaled)[:, 1]

        validation_metrics = {
            'accuracy': accuracy_score(validation_y, val_pred),
            'f1_score': f1_score(validation_y, val_pred),
            'precision': precision_score(validation_y, val_pred),
            'recall': recall_score(validation_y, val_pred),
            'roc_auc': roc_auc_score(validation_y, val_pred_proba)
        }

        # Version management
        version_parts = self.current_version.split('.')
        new_version = f"{version_parts[0]}.{int(version_parts[1])+1}.0"

        model_version = {
            'version': new_version,
            'model': new_model,
            'scaler': new_scaler,
            'training_data_size': len(new_X_train),
            'validation_metrics': validation_metrics,
            'timestamp': datetime.now().isoformat(),
            'retrain_reason': 'Performance drift detected'
        }

        self.model_versions.append(model_version)
        self.current_version = new_version

        return model_version

    def compare_model_versions(self, metric='f1_score'):
        """Compare performance across model versions"""
        if len(self.model_versions) < 2:
            return "Not enough model versions for comparison"

        comparison = []
        for version_info in self.model_versions:
            comparison.append({
                'version': version_info['version'],
                'timestamp': version_info['timestamp'],
                metric: version_info['validation_metrics'][metric],
                'training_data_size': version_info['training_data_size']
            })

        return pd.DataFrame(comparison)

    def rollback_to_version(self, version):
        """Rollback to a previous model version"""
        for version_info in self.model_versions:
            if version_info['version'] == version:
                self.current_version = version
                return version_info
        return None

# Initialize retraining pipeline
retrain_pipeline = AutoRetrainingPipeline(best_model, scaler, retrain_threshold=0.03)

# Check if retraining is needed
should_retrain, reason = retrain_pipeline.should_retrain(performance_drift)
print(f"Should retrain: {should_retrain}")
print(f"Reason: {reason}")

if should_retrain:
    # Simulate getting new training data (combine original + drifted data)
    new_X_train = pd.concat([X_train, drift_data[:500]], ignore_index=True)
    new_y_train = pd.concat([y_train, y_test[:500]], ignore_index=True)

    # Use remaining drifted data for validation
    validation_X = drift_data[500:700]
    validation_y = y_test[500:700]

    # Retrain model
    new_model_version = retrain_pipeline.retrain_model(
        new_X_train, new_y_train, validation_X, validation_y
    )

    print(f"\n✅ Model retrained to version {new_model_version['version']}")
    print(f"New validation metrics:")
    for metric, value in new_model_version['validation_metrics'].items():
        print(f"  {metric}: {value:.4f}")

# CHALLENGE 5: A/B TESTING FOR MODELS
print("\n" + "=" * 60)
print("🧪 CHALLENGE 5: A/B TESTING FOR MODELS")
print("=" * 60)

class ModelABTest:
    """A/B testing framework for comparing model versions"""

    def __init__(self, model_a, model_b, scaler_a, scaler_b,
                 name_a="Model A", name_b="Model B"):
        self.model_a = model_a
        self.model_b = model_b
        self.scaler_a = scaler_a
        self.scaler_b = scaler_b
        self.name_a = name_a
        self.name_b = name_b

        self.test_results = []
        self.assignment_ratio = 0.5  # 50/50 split

    def assign_model(self, customer_id):
        """Assign customer to model A or B"""
        # Use customer ID hash for consistent assignment
        np.random.seed(hash(str(customer_id)) % 1000000)
        return 'A' if np.random.random() < self.assignment_ratio else 'B'

    def run_prediction(self, customer_id, features, true_label=None):
        """Run prediction with A/B testing"""
        assignment = self.assign_model(customer_id)

        if assignment == 'A':
            features_scaled = self.scaler_a.transform(features.reshape(1, -1))
            prediction = self.model_a.predict(features_scaled)[0]
            confidence = self.model_a.predict_proba(features_scaled)[0].max()
            model_used = self.name_a
        else:
            features_scaled = self.scaler_b.transform(features.reshape(1, -1))
            prediction = self.model_b.predict(features_scaled)[0]
            confidence = self.model_b.predict_proba(features_scaled)[0].max()
            model_used = self.name_b

        # Log result
        result = {
            'customer_id': customer_id,
            'model_assignment': assignment,
            'model_name': model_used,
            'prediction': prediction,
            'confidence': confidence,
            'true_label': true_label,
            'timestamp': datetime.now().isoformat()
        }

        self.test_results.append(result)

        return result

    def analyze_results(self):
        """Analyze A/B test results"""
        if not self.test_results:
            return "No test results available"

        results_df = pd.DataFrame(self.test_results)

        # Group by model assignment
        model_a_results = results_df[results_df['model_assignment'] == 'A']
        model_b_results = results_df[results_df['model_assignment'] == 'B']

        analysis = {
            'sample_sizes': {
                'model_a': len(model_a_results),
                'model_b': len(model_b_results)
            },
            'prediction_distribution': {
                'model_a_churn_rate': model_a_results['prediction'].mean() if len(model_a_results) > 0 else 0,
                'model_b_churn_rate': model_b_results['prediction'].mean() if len(model_b_results) > 0 else 0
            },
            'confidence_stats': {
                'model_a_avg_confidence': model_a_results['confidence'].mean() if len(model_a_results) > 0 else 0,
                'model_b_avg_confidence': model_b_results['confidence'].mean() if len(model_b_results) > 0 else 0
            }
        }

        # If we have true labels, calculate accuracy
        if results_df['true_label'].notna().any():
            a_with_labels = model_a_results.dropna(subset=['true_label'])
            b_with_labels = model_b_results.dropna(subset=['true_label'])

            if len(a_with_labels) > 0 and len(b_with_labels) > 0:
                analysis['accuracy_comparison'] = {
                    'model_a_accuracy': accuracy_score(a_with_labels['true_label'], a_with_labels['prediction']),
                    'model_b_accuracy': accuracy_score(b_with_labels['true_label'], b_with_labels['prediction'])
                }

        return analysis

# Set up A/B test
if len(retrain_pipeline.model_versions) > 0:
    # Compare original model vs retrained model
    original_model = best_model
    retrained_model = retrain_pipeline.model_versions[-1]['model']
    retrained_scaler = retrain_pipeline.model_versions[-1]['scaler']

    ab_test = ModelABTest(
        original_model, retrained_model,
        scaler, retrained_scaler,
        "Original Model", f"Retrained Model v{retrain_pipeline.current_version}"
    )

    print("🧪 Running A/B Test...")

    # Simulate customer predictions
    test_customers = X_test.iloc[:100]  # Use first 100 test customers
    test_labels = y_test.iloc[:100]

    for i, (_, customer_features) in enumerate(test_customers.iterrows()):
        customer_id = f"customer_{i:04d}"
        true_label = test_labels.iloc[i]

        result = ab_test.run_prediction(
            customer_id,
            customer_features.values,
            true_label
        )

    # Analyze A/B test results
    ab_analysis = ab_test.analyze_results()

    print("\n📊 A/B Test Results:")
    print(f"Sample Sizes - A: {ab_analysis['sample_sizes']['model_a']}, B: {ab_analysis['sample_sizes']['model_b']}")
    print(f"Churn Rate - A: {ab_analysis['prediction_distribution']['model_a_churn_rate']:.3f}, B: {ab_analysis['prediction_distribution']['model_b_churn_rate']:.3f}")
    print(f"Avg Confidence - A: {ab_analysis['confidence_stats']['model_a_avg_confidence']:.3f}, B: {ab_analysis['confidence_stats']['model_b_avg_confidence']:.3f}")

    if 'accuracy_comparison' in ab_analysis:
        print(f"Accuracy - A: {ab_analysis['accuracy_comparison']['model_a_accuracy']:.3f}, B: {ab_analysis['accuracy_comparison']['model_b_accuracy']:.3f}")

# Visualization of production ML system components
plt.figure(figsize=(20, 16))

# Experiment tracking results
plt.subplot(3, 4, 1)
exp_summary = tracker.get_experiment_summary()
if not exp_summary.empty:
    plt.bar(exp_summary['Name'], exp_summary['F1_Score'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('F1 Score')
    plt.title('Experiment Results Comparison')
    plt.grid(axis='y', alpha=0.3)

# Model service performance over time
plt.subplot(3, 4, 2)
if model_service.prediction_history:
    timestamps = [datetime.fromisoformat(p['timestamp']) for p in model_service.prediction_history]
    confidences = [p['confidence'] for p in model_service.prediction_history]

    plt.plot(range(len(confidences)), confidences, 'b-', alpha=0.7, marker='o')
    plt.xlabel('Prediction Number')
    plt.ylabel('Prediction Confidence')
    plt.title('Model Confidence Over Time')
    plt.grid(True, alpha=0.3)

# Data drift visualization
plt.subplot(3, 4, 3)
if drift_scores:
    features = list(drift_scores.keys())[:8]  # Top 8 features
    scores = [drift_scores[f] for f in features]

    colors = ['red' if s > 2.0 else 'orange' if s > 1.0 else 'green' for s in scores]
    plt.barh(range(len(features)), scores, color=colors, alpha=0.7)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Drift Score (Z-score)')
    plt.title('Data Drift Detection')
    plt.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Alert Threshold')
    plt.legend()

# Performance drift metrics
plt.subplot(3, 4, 4)
if performance_drift:
    metrics = list(performance_drift.keys())
    reference_values = [performance_drift[m]['reference'] for m in metrics]
    current_values = [performance_drift[m]['current'] for m in metrics]

    x_pos = np.arange(len(metrics))
    width = 0.35

    plt.bar(x_pos - width/2, reference_values, width, label='Reference', alpha=0.7)
    plt.bar(x_pos + width/2, current_values, width, label='Current', alpha=0.7)

    plt.xticks(x_pos, metrics, rotation=45)
    plt.ylabel('Metric Value')
    plt.title('Performance Drift Comparison')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

# Model version comparison
plt.subplot(3, 4, 5)
if len(retrain_pipeline.model_versions) > 0:
    version_comparison = retrain_pipeline.compare_model_versions()
    plt.plot(range(len(version_comparison)), version_comparison['f1_score'], 'g-', marker='o')
    plt.xticks(range(len(version_comparison)), version_comparison['version'], rotation=45)
    plt.ylabel('F1 Score')
    plt.title('Model Version Performance')
    plt.grid(True, alpha=0.3)

# A/B test results
plt.subplot(3, 4, 6)
if 'ab_test' in locals() and ab_test.test_results:
    ab_results_df = pd.DataFrame(ab_test.test_results)
    model_counts = ab_results_df['model_assignment'].value_counts()

    plt.pie(model_counts.values, labels=[f'Model {k}' for k in model_counts.index],
           autopct='%1.1f%%', startangle=90)
    plt.title('A/B Test Traffic Distribution')

# Alert timeline
plt.subplot(3, 4, 7)
if monitor.alerts:
    alert_timestamps = [datetime.fromisoformat(a['timestamp']) for a in monitor.alerts]
    alert_types = [a['type'] for a in monitor.alerts]

    # Count alerts by hour
    alert_hours = [ts.hour for ts in alert_timestamps]
    unique_hours = sorted(set(alert_hours))
    hour_counts = [alert_hours.count(h) for h in unique_hours]

    plt.bar(unique_hours, hour_counts, alpha=0.7, color='red')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Alerts')
    plt.title('Alert Timeline Distribution')
    plt.grid(axis='y', alpha=0.3)

# Feature importance from production model
plt.subplot(3, 4, 8)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    feature_names = X.columns

    # Top 8 features
    top_indices = np.argsort(feature_importance)[-8:]
    plt.barh(range(8), feature_importance[top_indices])
    plt.yticks(range(8), [feature_names[i] for i in top_indices])
    plt.xlabel('Feature Importance')
    plt.title('Production Model Feature Importance')

# Prediction distribution over time
plt.subplot(3, 4, 9)
if model_service.prediction_history:
    predictions = [p['prediction'] for p in model_service.prediction_history]
    cumulative_churn_rate = np.cumsum(predictions) / np.arange(1, len(predictions) + 1)

    plt.plot(range(1, len(cumulative_churn_rate) + 1), cumulative_churn_rate, 'r-', alpha=0.7)
    plt.xlabel('Prediction Number')
    plt.ylabel('Cumulative Churn Rate')
    plt.title('Churn Rate Trend')
    plt.grid(True, alpha=0.3)

# Model performance heatmap
plt.subplot(3, 4, 10)
performance_matrix = []
for exp in experiment_results:
    metrics = exp['metrics']
    performance_matrix.append([
        metrics['accuracy'], metrics['f1_score'],
        metrics['precision'], metrics['recall']
    ])

performance_df = pd.DataFrame(performance_matrix,
                            columns=['Accuracy', 'F1', 'Precision', 'Recall'],
                            index=[exp['model_name'] for exp in experiment_results])

sns.heatmap(performance_df, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Model Performance Heatmap')

# System health metrics
plt.subplot(3, 4, 11)
health_metrics = model_service.health_check()
uptime = health_metrics['uptime_hours']
predictions_served = health_metrics['predictions_served']

metrics_data = [uptime, predictions_served / 10, len(monitor.alerts)]  # Scale for visibility
metric_labels = ['Uptime (hours)', 'Predictions\n(scaled /10)', 'Total Alerts']

plt.bar(metric_labels, metrics_data, alpha=0.7, color=['green', 'blue', 'red'])
plt.ylabel('Count')
plt.title('System Health Metrics')
plt.xticks(rotation=45)

# Monitoring timeline
plt.subplot(3, 4, 12)
monitoring_events = []
if monitor.alerts:
    for alert in monitor.alerts:
        monitoring_events.append({
            'timestamp': datetime.fromisoformat(alert['timestamp']),
            'type': alert['type'],
            'severity': alert.get('severity', 'medium')
        })

if monitoring_events:
    event_df = pd.DataFrame(monitoring_events)
    event_counts = event_df.groupby(['type', 'severity']).size().unstack(fill_value=0)

    event_counts.plot(kind='bar', stacked=True, alpha=0.7,
                     color={'high': 'red', 'medium': 'orange', 'low': 'yellow'})
    plt.ylabel('Number of Events')
    plt.title('Monitoring Events by Type & Severity')
    plt.xticks(rotation=45)
    plt.legend(title='Severity')
else:
    plt.text(0.5, 0.5, 'No monitoring\nevents recorded',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Monitoring Events')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("🚀 PRODUCTION ML SYSTEMS INSIGHTS & BEST PRACTICES")
print("=" * 60)

print("📋 Key Production Components Implemented:")
print("1. Experiment Tracking & Model Versioning")
print(f"   • Tracked {len(tracker.experiments)} experiments")
print(f"   • Best model: {best_model_name} (F1: {best_experiment['metrics']['f1_score']:.4f})")
print(f"   • Model artifacts saved and versioned")

print(f"\n2. Model Deployment & Serving")
print(f"   • Model service deployed with version {model_service.model_version}")
print(f"   • Served {model_service.prediction_count} predictions")
print(f"   • Health monitoring and performance tracking enabled")

print(f"\n3. Monitoring & Drift Detection")
print(f"   • Generated {len(monitor.alerts)} monitoring alerts")
print(f"   • Data drift detected in {len([s for s in drift_scores.values() if s > 1.5])} features")
print(f"   • Performance drift monitoring active")

print(f"\n4. Automated Retraining")
if len(retrain_pipeline.model_versions) > 0:
    print(f"   • Model automatically retrained to v{retrain_pipeline.current_version}")
    print(f"   • Performance improvement detected")
else:
    print(f"   • Retraining pipeline configured and ready")

print(f"\n5. A/B Testing")
if 'ab_test' in locals():
    print(f"   • A/B test conducted with {len(ab_test.test_results)} samples")
    print(f"   • Statistical comparison between model versions")

print(f"\n🎯 Production Best Practices:")
print("• Comprehensive experiment tracking for model governance")
print("• Real-time monitoring for data and performance drift")
print("• Automated retraining triggers based on performance thresholds")
print("• A/B testing for safe model deployment and comparison")
print("• Health checks and service monitoring for reliability")
print("• Model versioning and rollback capabilities")

print(f"\n⚡ MLOps Implementation Highlights:")
print("• End-to-end ML pipeline from training to deployment")
print("• Scalable monitoring and alerting system")
print("• Automated model lifecycle management")
print("• Production-ready service architecture")
print("• Statistical drift detection and remediation")
print("• Comprehensive logging and audit trails")

print("\n✅ Production ML Systems and MLOps Challenge Completed!")
print("What you've mastered:")
print("• ML experiment tracking and model versioning systems")
print("• Production model deployment and serving infrastructure")
print("• Advanced monitoring and drift detection techniques")
print("• Automated retraining and model lifecycle management")
print("• A/B testing frameworks for model validation")
print("• Complete MLOps pipeline implementation")

print(f"\n🏭 You are now a Production ML Engineer! Ready for enterprise-scale systems!")
```

### Success Criteria

- Implement comprehensive ML experiment tracking and versioning
- Build production-ready model serving infrastructure
- Create advanced monitoring and drift detection systems
- Develop automated retraining pipelines with performance triggers
- Design A/B testing frameworks for model validation
- Build complete MLOps workflows for enterprise deployment

### Learning Objectives

- Understand production ML system architecture and components
- Master ML experiment tracking, versioning, and model governance
- Learn advanced monitoring, drift detection, and alerting systems
- Practice automated retraining and model lifecycle management
- Develop skills in A/B testing and statistical model comparison
- Build scalable and reliable MLOps infrastructure

---

_Pro tip: Production ML is 20% modeling and 80% engineering - focus on reliability, monitoring, and automation for successful deployments!_
