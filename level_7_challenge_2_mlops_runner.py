#!/usr/bin/env python3
"""
Level 7 Challenge 2: Advanced MLOps Challenge Runner

This script demonstrates advanced MLOps practices including automated model training,
deployment pipelines, model monitoring, and drift detection.
"""

import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# MLOps libraries
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("🚀 LEVEL 7 CHALLENGE 2: ADVANCED MLOPS")
print("=" * 55)

# PART 1: MLOps Infrastructure Setup
print("\n🏗️ PART 1: MLOPS INFRASTRUCTURE SETUP")
print("-" * 40)


@dataclass
class ModelMetrics:
    """Data class for model performance metrics"""

    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    timestamp: datetime


class ModelRegistry:
    """Production model registry for version management"""

    def __init__(self, registry_path="model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models_db = self.registry_path / "models.json"
        self._init_registry()

    def _init_registry(self):
        """Initialize the model registry database"""
        if not self.models_db.exists():
            initial_data = {
                "models": {},
                "production_model": None,
                "staging_models": [],
                "archived_models": [],
            }
            with open(self.models_db, "w") as f:
                json.dump(initial_data, f, indent=2)

    def register_model(
        self,
        model_name: str,
        version: str,
        metrics: ModelMetrics,
        model_path: str,
        stage: str = "staging",
    ):
        """Register a new model version"""

        with open(self.models_db, "r") as f:
            registry = json.load(f)

        model_info = {
            "version": version,
            "metrics": {
                "accuracy": metrics.accuracy,
                "roc_auc": metrics.roc_auc,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "timestamp": metrics.timestamp.isoformat(),
            },
            "model_path": model_path,
            "stage": stage,
            "registered_at": datetime.now().isoformat(),
        }

        if model_name not in registry["models"]:
            registry["models"][model_name] = {}

        registry["models"][model_name][version] = model_info

        if stage == "staging":
            registry["staging_models"].append({"name": model_name, "version": version})

        with open(self.models_db, "w") as f:
            json.dump(registry, f, indent=2)

        print(f"✅ Registered {model_name} v{version} in {stage}")
        return model_info

    def promote_to_production(self, model_name: str, version: str):
        """Promote a model to production stage"""

        with open(self.models_db, "r") as f:
            registry = json.load(f)

        # Archive current production model
        if registry["production_model"]:
            old_model = registry["production_model"]
            registry["archived_models"].append(old_model)

        # Set new production model
        registry["production_model"] = {"name": model_name, "version": version}
        registry["models"][model_name][version]["stage"] = "production"

        # Remove from staging
        registry["staging_models"] = [
            m
            for m in registry["staging_models"]
            if not (m["name"] == model_name and m["version"] == version)
        ]

        with open(self.models_db, "w") as f:
            json.dump(registry, f, indent=2)

        print(f"🚀 Promoted {model_name} v{version} to production!")

    def get_production_model(self):
        """Get current production model info"""
        with open(self.models_db, "r") as f:
            registry = json.load(f)
        return registry["production_model"]

    def list_models(self):
        """List all registered models"""
        with open(self.models_db, "r") as f:
            registry = json.load(f)
        return registry


class ModelMonitor:
    """Model performance monitoring and drift detection"""

    def __init__(self, monitor_db_path="model_monitoring.json"):
        self.monitor_db_path = Path(monitor_db_path)
        self._init_monitoring()

    def _init_monitoring(self):
        """Initialize monitoring database"""
        if not self.monitor_db_path.exists():
            initial_data = {
                "performance_history": [],
                "drift_alerts": [],
                "predictions_log": [],
            }
            with open(self.monitor_db_path, "w") as f:
                json.dump(initial_data, f, indent=2)

    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        features: Dict,
        prediction: Any,
        confidence: float,
    ):
        """Log individual predictions for monitoring"""

        with open(self.monitor_db_path, "r") as f:
            monitor_data = json.load(f)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "features": features,
            "prediction": str(prediction),
            "confidence": confidence,
        }

        monitor_data["predictions_log"].append(log_entry)

        # Keep only last 1000 predictions for performance
        if len(monitor_data["predictions_log"]) > 1000:
            monitor_data["predictions_log"] = monitor_data["predictions_log"][-1000:]

        with open(self.monitor_db_path, "w") as f:
            json.dump(monitor_data, f, indent=2)

    def log_performance(
        self, model_name: str, model_version: str, metrics: ModelMetrics
    ):
        """Log model performance metrics"""

        with open(self.monitor_db_path, "r") as f:
            monitor_data = json.load(f)

        perf_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "metrics": {
                "accuracy": metrics.accuracy,
                "roc_auc": metrics.roc_auc,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
            },
        }

        monitor_data["performance_history"].append(perf_entry)

        with open(self.monitor_db_path, "w") as f:
            json.dump(monitor_data, f, indent=2)

        print(f"📊 Logged performance for {model_name} v{model_version}")

    def detect_drift(
        self,
        current_features: pd.DataFrame,
        baseline_features: pd.DataFrame,
        threshold: float = 0.1,
    ):
        """Simple drift detection based on feature statistics"""

        drift_detected = False
        drift_features = []

        for column in current_features.columns:
            if column in baseline_features.columns:
                current_mean = current_features[column].mean()
                baseline_mean = baseline_features[column].mean()

                # Calculate relative drift
                if baseline_mean != 0:
                    drift_ratio = abs(current_mean - baseline_mean) / abs(baseline_mean)
                    if drift_ratio > threshold:
                        drift_detected = True
                        drift_features.append(
                            {
                                "feature": column,
                                "baseline_mean": baseline_mean,
                                "current_mean": current_mean,
                                "drift_ratio": drift_ratio,
                            }
                        )

        if drift_detected:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "drift_detected": True,
                "drift_features": drift_features,
                "threshold": threshold,
            }

            # Log alert
            with open(self.monitor_db_path, "r") as f:
                monitor_data = json.load(f)

            monitor_data["drift_alerts"].append(alert)

            with open(self.monitor_db_path, "w") as f:
                json.dump(monitor_data, f, indent=2)

            print(f"⚠️ Data drift detected in {len(drift_features)} features!")
            for feature in drift_features:
                print(
                    f"  • {feature['feature']}: {feature['drift_ratio']:.3f} drift ratio"
                )
        else:
            print("✅ No significant data drift detected")

        return drift_detected, drift_features


# Initialize MLOps components
print("🔧 Setting up MLOps infrastructure...")

model_registry = ModelRegistry()
model_monitor = ModelMonitor()

print("✅ Model Registry initialized")
print("✅ Model Monitor initialized")

# PART 2: Automated Model Training Pipeline
print("\n🤖 PART 2: AUTOMATED MODEL TRAINING PIPELINE")
print("-" * 40)


def create_synthetic_dataset(n_samples=1000, n_features=10, drift=False):
    """Create synthetic dataset with optional drift simulation"""

    np.random.seed(42 if not drift else 123)

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create target with some logical relationship
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.3 > 0).astype(int)

    # Add drift if requested
    if drift:
        X[:, 0] += 0.5  # Shift first feature
        X[:, 1] *= 1.2  # Scale second feature
        print("📊 Synthetic drift introduced to dataset")

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


def train_model_pipeline(data: pd.DataFrame, model_name: str):
    """Automated model training pipeline with MLflow tracking"""

    print(f"🏃 Training {model_name} pipeline...")

    # Set MLflow experiment
    mlflow.set_experiment("advanced_mlops_demo")

    with mlflow.start_run(
        run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):

        # Prepare data
        feature_columns = [col for col in data.columns if col != "target"]
        X = data[feature_columns]
        y = data["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        random_state=42,
                    ),
                ),
            ]
        )

        # Train model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Get detailed classification metrics
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted"
        )

        # Create metrics object
        metrics = ModelMetrics(
            accuracy=accuracy,
            roc_auc=roc_auc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            timestamp=datetime.now(),
        )

        # Log to MLflow
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        # Log model
        model_path = "model"
        mlflow.sklearn.log_model(pipeline, model_path)

        print(f"📈 Model Performance:")
        print(f"  • Accuracy: {accuracy:.3f}")
        print(f"  • ROC-AUC: {roc_auc:.3f}")
        print(f"  • Precision: {precision:.3f}")
        print(f"  • Recall: {recall:.3f}")
        print(f"  • F1-Score: {f1:.3f}")

        # Generate version
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Register model
        model_info = model_registry.register_model(
            model_name=model_name,
            version=version,
            metrics=metrics,
            model_path=model_path,
        )

        # Log performance to monitor
        model_monitor.log_performance(model_name, version, metrics)

        return pipeline, metrics, version


# Generate training data
print("📊 Generating training dataset...")
train_data = create_synthetic_dataset(n_samples=2000, n_features=8)

# Train initial model
trained_model, initial_metrics, model_version = train_model_pipeline(
    train_data, "customer_classifier"
)

# PART 3: Model Deployment Simulation
print("\n🚀 PART 3: MODEL DEPLOYMENT SIMULATION")
print("-" * 40)


class ModelServer:
    """Simulated model serving infrastructure"""

    def __init__(self, model_registry: ModelRegistry, model_monitor: ModelMonitor):
        self.model_registry = model_registry
        self.model_monitor = model_monitor
        self.loaded_model = None
        self.model_info = None

    def load_production_model(self):
        """Load the current production model"""

        prod_model = self.model_registry.get_production_model()
        if not prod_model:
            print("⚠️ No production model available")
            return False

        print(
            f"🔄 Loading production model: {prod_model['name']} v{prod_model['version']}"
        )

        # In real scenario, would load from MLflow or model storage
        # For demo, using the trained model
        self.loaded_model = trained_model
        self.model_info = prod_model

        print("✅ Production model loaded successfully")
        return True

    def predict(self, features: Dict) -> Dict:
        """Make prediction with monitoring"""

        if not self.loaded_model:
            raise ValueError("No model loaded")

        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])

        # Make prediction
        prediction = self.loaded_model.predict(feature_df)[0]
        confidence = self.loaded_model.predict_proba(feature_df).max()

        # Log prediction for monitoring
        self.model_monitor.log_prediction(
            model_name=self.model_info["name"],
            model_version=self.model_info["version"],
            features=features,
            prediction=prediction,
            confidence=confidence,
        )

        return {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "model_name": self.model_info["name"],
            "model_version": self.model_info["version"],
        }


# Initialize model server
model_server = ModelServer(model_registry, model_monitor)

# Promote model to production
print("🚀 Promoting model to production...")
model_registry.promote_to_production("customer_classifier", model_version)

# Load production model
model_server.load_production_model()

# Simulate predictions
print("\n📊 Simulating production predictions...")
for i in range(10):
    sample_features = {f"feature_{j}": float(np.random.randn()) for j in range(8)}

    result = model_server.predict(sample_features)
    print(
        f"Prediction {i+1}: Class {result['prediction']} (confidence: {result['confidence']:.3f})"
    )

# PART 4: Model Monitoring and Drift Detection
print("\n🔍 PART 4: MODEL MONITORING & DRIFT DETECTION")
print("-" * 40)

# Generate new data with drift
print("📊 Generating data with drift...")
baseline_data = train_data[[col for col in train_data.columns if col != "target"]]
drift_data = create_synthetic_dataset(n_samples=500, n_features=8, drift=True)
current_features = drift_data[[col for col in drift_data.columns if col != "target"]]

# Detect drift
print("\n🔍 Running drift detection...")
drift_detected, drift_features = model_monitor.detect_drift(
    current_features, baseline_data, threshold=0.1
)

# PART 5: Automated Retraining Pipeline
print("\n🔄 PART 5: AUTOMATED RETRAINING PIPELINE")
print("-" * 40)

if drift_detected:
    print("⚠️ Drift detected! Triggering automated retraining...")

    # Retrain model with new data
    combined_data = pd.concat([train_data, drift_data], ignore_index=True)

    new_model, new_metrics, new_version = train_model_pipeline(
        combined_data, "customer_classifier"
    )

    # Compare performance
    improvement = new_metrics.accuracy - initial_metrics.accuracy

    print(f"\n📊 Model Comparison:")
    print(f"  Original Accuracy: {initial_metrics.accuracy:.3f}")
    print(f"  New Accuracy: {new_metrics.accuracy:.3f}")
    print(f"  Improvement: {improvement:+.3f}")

    # Auto-promote if improvement is significant
    if improvement > 0.01:  # 1% improvement threshold
        print("✅ Significant improvement detected! Auto-promoting to production...")
        model_registry.promote_to_production("customer_classifier", new_version)
    else:
        print("⚠️ Improvement not significant. Keeping current production model.")

# PART 6: MLOps Dashboard Summary
print("\n📈 PART 6: MLOPS DASHBOARD SUMMARY")
print("-" * 40)

# Show registry status
registry_status = model_registry.list_models()
print(f"🏛️ Model Registry Status:")
print(f"  • Registered Models: {len(registry_status['models'])}")
print(f"  • Production Model: {registry_status['production_model']}")
print(f"  • Staging Models: {len(registry_status['staging_models'])}")
print(f"  • Archived Models: {len(registry_status['archived_models'])}")

# Show monitoring summary
with open(model_monitor.monitor_db_path, "r") as f:
    monitor_data = json.load(f)

print(f"\n📊 Monitoring Summary:")
print(f"  • Total Predictions: {len(monitor_data['predictions_log'])}")
print(f"  • Performance Logs: {len(monitor_data['performance_history'])}")
print(f"  • Drift Alerts: {len(monitor_data['drift_alerts'])}")

print("\n🏆 LEVEL 7 CHALLENGE 2 COMPLETED!")
print("=" * 45)

print("\n✅ ADVANCED MLOPS MASTERY DEMONSTRATED:")
print("  🏗️ Production MLOps infrastructure")
print("  🤖 Automated model training pipelines")
print("  🚀 Model deployment and serving")
print("  🔍 Comprehensive model monitoring")
print("  📊 Data drift detection systems")
print("  🔄 Automated retraining workflows")

print("\n🎓 PRODUCTION SKILLS LEARNED:")
print("  • Model registry and versioning")
print("  • Automated CI/CD for ML models")
print("  • Production model serving")
print("  • Performance monitoring and alerting")
print("  • Data drift detection and handling")
print("  • Automated model retraining")

print("\n🚀 NEXT CHALLENGE UNLOCKED:")
print("  Ready for Challenge 3: Real-time Analytics!")

print("\n🏅 Achievement Unlocked: MLOps Engineer!")
