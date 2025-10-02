"""
Level 5 Challenge 4: Production ML Systems and MLOps
Master deployment, monitoring, and maintenance of ML systems in production.
"""

import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class MLExperimentTracker:
    """Track ML experiments with model versioning and metadata"""

    def __init__(self, experiment_dir: str = "ml_experiments") -> None:
        self.experiment_dir = experiment_dir
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.current_experiment_id: Optional[str] = None

        # Create directories
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(f"{experiment_dir}/models", exist_ok=True)
        os.makedirs(f"{experiment_dir}/metrics", exist_ok=True)
        os.makedirs(f"{experiment_dir}/artifacts", exist_ok=True)

    def start_experiment(self, experiment_name: str, description: str = "") -> str:
        """Start a new experiment"""
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_experiment_id = experiment_id

        self.experiments[experiment_id] = {
            "name": experiment_name,
            "description": description,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "metrics": {},
            "model_path": None,
            "artifacts": [],
        }

        print(f"Started experiment: {experiment_id}")
        return experiment_id

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics for current experiment"""
        if self.current_experiment_id:
            self.experiments[self.current_experiment_id]["metrics"].update(metrics)

            # Save metrics to file
            metrics_file = (
                f"{self.experiment_dir}/metrics/{self.current_experiment_id}.json"
            )
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

    def save_model(self, model: Any, model_name: str = "model") -> str:
        """Save model for current experiment"""
        if self.current_experiment_id:
            model_path = f"{self.experiment_dir}/models/{self.current_experiment_id}_{model_name}.pkl"
            joblib.dump(model, model_path)
            self.experiments[self.current_experiment_id]["model_path"] = model_path
            print(f"Model saved: {model_path}")
            return model_path
        return ""

    def save_artifact(self, data: Any, artifact_name: str) -> str:
        """Save experiment artifact"""
        if self.current_experiment_id:
            artifact_path = f"{self.experiment_dir}/artifacts/{self.current_experiment_id}_{artifact_name}.pkl"
            joblib.dump(data, artifact_path)
            self.experiments[self.current_experiment_id]["artifacts"].append(
                artifact_path
            )
            return artifact_path
        return ""

    def end_experiment(self) -> None:
        """End current experiment"""
        if self.current_experiment_id:
            self.experiments[self.current_experiment_id][
                "end_time"
            ] = datetime.now().isoformat()
            self.experiments[self.current_experiment_id]["status"] = "completed"

            # Save experiment summary
            summary_file = (
                f"{self.experiment_dir}/{self.current_experiment_id}_summary.json"
            )
            with open(summary_file, "w") as f:
                json.dump(self.experiments[self.current_experiment_id], f, indent=2)

            print(f"Experiment completed: {self.current_experiment_id}")
            self.current_experiment_id = None

    def get_experiment_history(self) -> pd.DataFrame:
        """Get experiment history as DataFrame"""
        history_data = []
        for exp_id, exp_data in self.experiments.items():
            row = {
                "experiment_id": exp_id,
                "name": exp_data["name"],
                "status": exp_data["status"],
                "start_time": exp_data["start_time"],
            }
            row.update(exp_data.get("metrics", {}))
            history_data.append(row)

        return pd.DataFrame(history_data)


class ModelMonitor:
    """Monitor model performance and data drift in production"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.performance_history: list = []
        self.prediction_history: list = []
        self.drift_alerts: list = []

    def log_prediction(
        self,
        features: np.ndarray,
        prediction: Any,
        actual: Optional[Any] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log a prediction with features and optional actual value"""
        if timestamp is None:
            timestamp = datetime.now()

        log_entry = {
            "timestamp": timestamp,
            "features": (
                features.tolist() if isinstance(features, np.ndarray) else features
            ),
            "prediction": prediction,
            "actual": actual,
        }

        self.prediction_history.append(log_entry)

    def calculate_performance_metrics(self, window_hours: int = 24) -> Dict[str, float]:
        """Calculate performance metrics for recent predictions"""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)

        recent_predictions = [
            p
            for p in self.prediction_history
            if p["timestamp"] > cutoff_time and p["actual"] is not None
        ]

        if not recent_predictions:
            return {}

        y_true = [p["actual"] for p in recent_predictions]
        y_pred = [p["prediction"] for p in recent_predictions]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average="macro"),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "n_predictions": len(recent_predictions),
        }

        # Store performance history
        self.performance_history.append(
            {"timestamp": datetime.now(), "window_hours": window_hours, **metrics}
        )

        return metrics

    def detect_data_drift(
        self,
        reference_features: np.ndarray,
        current_features: np.ndarray,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        from scipy import stats

        drift_results: Dict[str, Any] = {
            "drift_detected": False,
            "feature_drifts": [],
            "overall_drift_score": 0.0,
        }

        n_features = reference_features.shape[1]
        drift_scores = []

        for i in range(n_features):
            # Kolmogorov-Smirnov test for distribution differences
            ks_stat, p_value = stats.ks_2samp(
                reference_features[:, i], current_features[:, i]
            )

            feature_drift = {
                "feature_index": i,
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "drift_detected": p_value < 0.05,
            }

            drift_results["feature_drifts"].append(feature_drift)
            drift_scores.append(ks_stat)

        # Overall drift score
        drift_results["overall_drift_score"] = np.mean(drift_scores)
        drift_results["drift_detected"] = (
            drift_results["overall_drift_score"] > threshold
        )

        if drift_results["drift_detected"]:
            alert = {
                "timestamp": datetime.now(),
                "type": "data_drift",
                "message": f"Data drift detected! Overall drift score: {drift_results['overall_drift_score']:.3f}",
                "details": drift_results,
            }
            self.drift_alerts.append(alert)
            print(f"⚠️  ALERT: {alert['message']}")

        return drift_results


class MLPipeline:
    """Production ML pipeline with preprocessing and prediction"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.preprocessor: Optional[StandardScaler] = None
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: Optional[list] = None
        self.model_version = "1.0.0"
        self.created_at = datetime.now()

    def fit(self, x_data: pd.DataFrame, y: pd.Series) -> None:
        """Fit the complete pipeline"""
        print(f"Training ML Pipeline: {self.model_name}")

        # Store feature names
        self.feature_names = x_data.columns.tolist()

        # Create preprocessor
        self.preprocessor = StandardScaler()
        x_scaled = self.preprocessor.fit_transform(x_data)

        # Train model (using RandomForest as default)
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, min_samples_leaf=1, max_features="sqrt"
        )
        self.model.fit(x_scaled, y)

        print(
            f"Pipeline training completed. Features: {len(self.feature_names) if self.feature_names else 0}"
        )

    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if (
            self.preprocessor is None
            or self.model is None
            or self.feature_names is None
        ):
            raise ValueError("Pipeline not trained. Call fit() first.")

        # Ensure feature order
        x_ordered = x_data[self.feature_names]

        # Preprocess and predict
        x_scaled = self.preprocessor.transform(x_ordered)
        predictions = self.model.predict(x_scaled)

        return np.asarray(predictions)

    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if (
            self.preprocessor is None
            or self.model is None
            or self.feature_names is None
        ):
            raise ValueError("Pipeline not trained. Call fit() first.")

        x_ordered = x_data[self.feature_names]
        x_scaled = self.preprocessor.transform(x_ordered)
        probabilities = self.model.predict_proba(x_scaled)

        return np.asarray(probabilities)

    def save_pipeline(self, file_path: str) -> None:
        """Save complete pipeline"""
        pipeline_data = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_names": self.feature_names,
            "created_at": self.created_at.isoformat(),
            "preprocessor": self.preprocessor,
            "model": self.model,
        }

        joblib.dump(pipeline_data, file_path)
        print(f"Pipeline saved: {file_path}")

    @classmethod
    def load_pipeline(cls, file_path: str) -> "MLPipeline":
        """Load complete pipeline"""
        pipeline_data = joblib.load(file_path)

        pipeline = cls(pipeline_data["model_name"])
        pipeline.model_version = pipeline_data["model_version"]
        pipeline.feature_names = pipeline_data["feature_names"]
        pipeline.created_at = datetime.fromisoformat(pipeline_data["created_at"])
        pipeline.preprocessor = pipeline_data["preprocessor"]
        pipeline.model = pipeline_data["model"]

        print(f"Pipeline loaded: {pipeline.model_name} v{pipeline.model_version}")
        return pipeline


class ModelRetrainer:
    """Automated model retraining system"""

    def __init__(self, pipeline: MLPipeline, monitor: ModelMonitor) -> None:
        self.pipeline = pipeline
        self.monitor = monitor
        self.retraining_history: list = []

    def should_retrain(
        self,
        min_accuracy: float = 0.8,
        min_samples: int = 100,
        drift_threshold: float = 0.1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Determine if model should be retrained"""

        # Check recent performance
        recent_metrics = self.monitor.calculate_performance_metrics(window_hours=24)

        retrain_reasons = []

        # Performance degradation check
        if recent_metrics.get("accuracy", 1.0) < min_accuracy:
            retrain_reasons.append(
                f"Accuracy below threshold: {recent_metrics['accuracy']:.3f} < {min_accuracy}"
            )

        # Sufficient data check
        if recent_metrics.get("n_predictions", 0) < min_samples:
            retrain_reasons.append(
                f"Insufficient samples: {recent_metrics.get('n_predictions', 0)} < {min_samples}"
            )

        # Data drift check
        if self.monitor.drift_alerts:
            recent_drift = self.monitor.drift_alerts[-1]
            if recent_drift["details"]["overall_drift_score"] > drift_threshold:
                retrain_reasons.append(
                    f"Data drift detected: {recent_drift['details']['overall_drift_score']:.3f}"
                )

        should_retrain = len(retrain_reasons) > 0

        decision_info = {
            "should_retrain": should_retrain,
            "reasons": retrain_reasons,
            "current_metrics": recent_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        return should_retrain, decision_info

    def retrain_model(self, new_x: pd.DataFrame, new_y: pd.Series) -> str:
        """Retrain the model with new data"""
        print("Starting model retraining...")

        # Create new pipeline version
        old_version = self.pipeline.model_version
        version_parts = old_version.split(".")
        new_minor = int(version_parts[1]) + 1
        new_version = f"{version_parts[0]}.{new_minor}.{version_parts[2]}"

        # Retrain
        self.pipeline.fit(new_x, new_y)
        self.pipeline.model_version = new_version
        self.pipeline.created_at = datetime.now()

        # Log retraining
        retrain_record = {
            "timestamp": datetime.now().isoformat(),
            "old_version": old_version,
            "new_version": new_version,
            "training_samples": len(new_x),
            "features": len(new_x.columns),
        }

        self.retraining_history.append(retrain_record)

        print(f"Model retrained: {old_version} -> {new_version}")
        return new_version


def create_production_dataset() -> pd.DataFrame:
    """Create dataset simulating production ML scenario"""
    print("Creating production ML dataset...")

    rng = np.random.default_rng(42)
    n_samples = 10000

    # E-commerce customer dataset
    data = pd.DataFrame(
        {
            "customer_age": rng.normal(35, 12, n_samples),
            "account_tenure_months": rng.exponential(24, n_samples),
            "total_purchases": rng.poisson(15, n_samples),
            "avg_order_value": rng.gamma(2, 50, n_samples),
            "days_since_last_order": rng.exponential(30, n_samples),
            "customer_service_calls": rng.poisson(3, n_samples),
            "email_open_rate": rng.beta(3, 5, n_samples),
            "mobile_app_usage_hours": rng.gamma(1.5, 2, n_samples),
            "loyalty_points": rng.pareto(1, n_samples) * 1000,
            "complaint_history": rng.poisson(1, n_samples),
        }
    )

    # Clip values
    data["customer_age"] = np.clip(data["customer_age"], 18, 80)
    data["account_tenure_months"] = np.clip(data["account_tenure_months"], 1, 120)
    data["days_since_last_order"] = np.clip(data["days_since_last_order"], 0, 365)
    data["email_open_rate"] = np.clip(data["email_open_rate"], 0, 1)

    # Create churn target
    churn_probability = (
        -data["total_purchases"] / 50
        + data["days_since_last_order"] / 100
        + data["complaint_history"] / 10
        + (1 - data["email_open_rate"])
        + rng.normal(0, 0.2, n_samples)
    )

    # Convert to binary
    data["will_churn"] = (
        churn_probability > np.percentile(churn_probability, 75)
    ).astype(int)

    print(f"Production dataset created: {data.shape}")
    print(f"Churn rate: {data['will_churn'].mean():.2%}")

    return data


def simulate_production_environment(
    pipeline: MLPipeline,
    monitor: ModelMonitor,
    test_data: pd.DataFrame,
    n_days: int = 30,
) -> None:
    """Simulate production environment with model serving"""
    print(f"Simulating {n_days} days of production environment...")

    rng = np.random.default_rng(42)
    predictions_per_day = 100
    feature_columns = [col for col in test_data.columns if col != "will_churn"]

    # Simulate daily predictions
    for day in range(n_days):
        print(f"Day {day + 1}: Processing {predictions_per_day} predictions")

        # Sample random requests
        daily_sample = test_data.sample(predictions_per_day, replace=True)

        for _, row in daily_sample.iterrows():
            features = row[feature_columns].values.reshape(1, -1)
            features_df = pd.DataFrame([row[feature_columns]], columns=feature_columns)

            # Make prediction
            prediction = pipeline.predict(features_df)[0]
            actual = row["will_churn"]

            # Simulate timestamp (spread across the day)
            timestamp = (
                datetime.now()
                - timedelta(days=n_days - day - 1)
                + timedelta(hours=rng.integers(0, 24))
            )

            # Log prediction
            monitor.log_prediction(features, prediction, actual, timestamp)

        # Introduce gradual data drift after day 20
        if day > 20:
            # Simulate feature drift by slightly shifting distributions
            drift_factor = (day - 20) * 0.02
            test_data.loc[:, "customer_age"] += rng.normal(
                0, drift_factor, len(test_data)
            )
            test_data.loc[:, "avg_order_value"] *= 1 + drift_factor

    print(f"Simulation complete: {len(monitor.prediction_history)} predictions logged")


def create_mlops_dashboard(monitor: ModelMonitor, retrainer: ModelRetrainer) -> None:
    """Create MLOps monitoring dashboard"""
    print("Creating MLOps monitoring dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("MLOps Production Dashboard", fontsize=16, fontweight="bold")

    # 1. Model Performance Over Time
    ax = axes[0, 0]
    if monitor.performance_history:
        perf_df = pd.DataFrame(monitor.performance_history)
        ax.plot(range(len(perf_df)), perf_df["accuracy"], "o-", label="Accuracy")
        ax.plot(range(len(perf_df)), perf_df["f1_score"], "s-", label="F1 Score")
        ax.set_xlabel("Time Window")
        ax.set_ylabel("Performance Score")
        ax.set_title("Model Performance Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No performance data available", ha="center", va="center")
        ax.set_title("Model Performance Over Time")

    # 2. Prediction Volume
    ax = axes[0, 1]
    if monitor.prediction_history:
        # Group predictions by day
        pred_df = pd.DataFrame(monitor.prediction_history)
        pred_df["date"] = pd.to_datetime(pred_df["timestamp"]).dt.date
        daily_counts = pred_df.groupby("date").size()

        ax.bar(range(len(daily_counts)), daily_counts.values)
        ax.set_xlabel("Days")
        ax.set_ylabel("Number of Predictions")
        ax.set_title("Daily Prediction Volume")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No prediction data available", ha="center", va="center")
        ax.set_title("Daily Prediction Volume")

    # 3. Data Drift Alerts
    ax = axes[1, 0]
    if monitor.drift_alerts:
        alert_times = [
            datetime.fromisoformat(alert["timestamp"]) for alert in monitor.drift_alerts
        ]
        drift_scores = [
            alert["details"]["overall_drift_score"] for alert in monitor.drift_alerts
        ]

        ax.scatter(range(len(alert_times)), drift_scores, c="red", s=100, alpha=0.7)
        ax.axhline(y=0.1, color="orange", linestyle="--", label="Drift Threshold")
        ax.set_xlabel("Alert Number")
        ax.set_ylabel("Drift Score")
        ax.set_title("Data Drift Alerts")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No drift alerts", ha="center", va="center")
        ax.set_title("Data Drift Alerts")

    # 4. Retraining History
    ax = axes[1, 1]
    if retrainer.retraining_history:
        versions = [r["new_version"] for r in retrainer.retraining_history]
        sample_counts = [r["training_samples"] for r in retrainer.retraining_history]

        bars = ax.bar(range(len(versions)), sample_counts)
        ax.set_xlabel("Retraining Event")
        ax.set_ylabel("Training Samples")
        ax.set_title("Model Retraining History")
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions, rotation=45)

        # Add value labels
        for bar, count in zip(bars, sample_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sample_counts) * 0.01,
                str(count),
                ha="center",
                va="bottom",
            )
    else:
        ax.text(0.5, 0.5, "No retraining events", ha="center", va="center")
        ax.set_title("Model Retraining History")

    plt.tight_layout()
    plt.show()


def main() -> Dict[str, Any]:
    """Main function to run production ML systems challenge"""
    print("=" * 70)
    print("LEVEL 5 CHALLENGE 4: PRODUCTION ML SYSTEMS & MLOPS")
    print("=" * 70)

    # Create production dataset
    data = create_production_dataset()

    # Split data
    feature_columns = [col for col in data.columns if col != "will_churn"]
    X = data[feature_columns]
    y = data["will_churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Initialize MLOps components
    print("\\n" + "=" * 50)
    print("INITIALIZING MLOPS COMPONENTS")
    print("=" * 50)

    tracker = MLExperimentTracker()
    monitor = ModelMonitor("churn_predictor")
    pipeline = MLPipeline("churn_predictor")

    # Run experiments
    print("\\n" + "=" * 50)
    print("EXPERIMENT TRACKING")
    print("=" * 50)

    # Experiment 1: Random Forest
    _ = tracker.start_experiment("rf_baseline", "Random Forest baseline model")
    pipeline.fit(X_train, y_train)

    # Evaluate
    train_pred = pipeline.predict(pd.DataFrame(X_train, columns=feature_columns))
    test_pred = pipeline.predict(pd.DataFrame(X_test, columns=feature_columns))

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_f1": f1_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred),
        "test_recall": recall_score(y_test, test_pred),
    }

    tracker.log_metrics(metrics)
    _ = tracker.save_model(pipeline)
    tracker.end_experiment()

    print(f"Baseline model metrics: {metrics}")

    # Initialize retrainer
    retrainer = ModelRetrainer(pipeline, monitor)

    # Simulate production environment
    print("\\n" + "=" * 50)
    print("PRODUCTION SIMULATION")
    print("=" * 50)

    # Use test data for production simulation
    test_data_with_target = pd.concat([X_test, y_test], axis=1)
    simulate_production_environment(pipeline, monitor, test_data_with_target, n_days=30)

    # Monitor and detect issues
    print("\\n" + "=" * 50)
    print("MONITORING & DRIFT DETECTION")
    print("=" * 50)

    # Calculate recent performance
    recent_perf = monitor.calculate_performance_metrics(window_hours=168)  # 1 week
    print(f"Recent performance: {recent_perf}")

    # Detect data drift
    reference_features = X_train.iloc[:1000].values  # Reference distribution
    current_features = X_test.iloc[:1000].values  # Current distribution

    drift_results = monitor.detect_data_drift(reference_features, current_features)
    print(
        f"Drift detection results: Drift detected = {drift_results['drift_detected']}"
    )

    # Check retraining decision
    should_retrain, decision_info = retrainer.should_retrain(min_accuracy=0.7)
    print(f"Should retrain: {should_retrain}")
    if should_retrain:
        print(f"Reasons: {decision_info['reasons']}")

        # Perform retraining with new data
        new_version = retrainer.retrain_model(X_test.iloc[:2000], y_test.iloc[:2000])
        print(f"Model retrained to version: {new_version}")

    # Create dashboard
    print("\\n" + "=" * 50)
    print("MLOPS DASHBOARD")
    print("=" * 50)

    create_mlops_dashboard(monitor, retrainer)

    # Summary
    print("\\n" + "=" * 70)
    print("CHALLENGE 4 COMPLETION SUMMARY")
    print("=" * 70)

    mlops_components = [
        "ML Experiment Tracking",
        "Model Versioning System",
        "Production Model Pipeline",
        "Real-time Prediction Logging",
        "Performance Monitoring",
        "Data Drift Detection",
        "Automated Retraining Decisions",
        "Model Deployment Pipeline",
        "MLOps Monitoring Dashboard",
        "Production Simulation Environment",
        "Kolmogorov-Smirnov Drift Tests",
        "Statistical Performance Tracking",
        "Model Artifact Management",
        "Pipeline State Persistence",
    ]

    print("MLOps components implemented:")
    for i, component in enumerate(mlops_components, 1):
        print(f"  {i}. {component}")

    experiment_history = tracker.get_experiment_history()
    print("\\nProduction Statistics:")
    print(f"  - Experiments tracked: {len(experiment_history)}")
    print(f"  - Predictions logged: {len(monitor.prediction_history)}")
    print(f"  - Performance windows: {len(monitor.performance_history)}")
    print(f"  - Drift alerts: {len(monitor.drift_alerts)}")
    print(f"  - Retraining events: {len(retrainer.retraining_history)}")
    print(f"  - Model version: {pipeline.model_version}")

    return {
        "tracker": tracker,
        "monitor": monitor,
        "pipeline": pipeline,
        "retrainer": retrainer,
        "experiment_history": experiment_history,
        "production_data": data,
    }


if __name__ == "__main__":
    main()

    print("\\n" + "=" * 70)
    print("CHALLENGE 4 STATUS: COMPLETE")
    print("=" * 70)
    print("Production ML systems and MLOps mastery achieved!")
    print("🚀 Level 5: Algorithm Architect - ALL CHALLENGES COMPLETE!")
