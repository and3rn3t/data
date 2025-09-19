"""
ML Experiment Tracking Integration

Provides integration with MLflow and Weights & Biases for experiment tracking,
model versioning, and reproducible machine learning workflows.
"""

import os
import json
from typing import Any, Dict, List, Optional, Union
import warnings
import tempfile
from pathlib import Path


class ExperimentTracker:
    """
    Integration class for ML experiment tracking tools.
    
    Supports MLflow and Weights & Biases with graceful fallbacks when
    libraries aren't available.
    """
    
    def __init__(self, project_name: str = "data-science-sandbox", experiment_name: str = "default"):
        """
        Initialize experiment tracker.
        
        Args:
            project_name: Name of the project/experiment group
            experiment_name: Name of the specific experiment
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        self.mlflow_available = self._check_mlflow()
        self.wandb_available = self._check_wandb()
        
        # Fallback tracking
        self.fallback_logs = []
        self.fallback_metrics = {}
        self.fallback_params = {}
        
        # Initialize available trackers
        self._init_trackers()
    
    def _check_mlflow(self) -> bool:
        """Check if MLflow is available."""
        try:
            import mlflow
            import mlflow.sklearn
            import mlflow.tensorflow
            import mlflow.pytorch
            self.mlflow = mlflow
            return True
        except ImportError:
            return False
    
    def _check_wandb(self) -> bool:
        """Check if Weights & Biases is available."""
        try:
            import wandb
            self.wandb = wandb
            return True
        except ImportError:
            return False
    
    def _init_trackers(self):
        """Initialize available experiment trackers."""
        if self.mlflow_available:
            try:
                # Set up MLflow tracking URI (local by default)
                tracking_dir = Path.home() / ".mlflow"
                tracking_dir.mkdir(exist_ok=True)
                
                self.mlflow.set_tracking_uri(f"file://{tracking_dir}")
                self.mlflow.set_experiment(self.experiment_name)
                
                print(f"✅ MLflow initialized - Tracking URI: {self.mlflow.get_tracking_uri()}")
            except Exception as e:
                print(f"⚠️ MLflow initialization failed: {e}")
                self.mlflow_available = False
        
        if self.wandb_available:
            try:
                # Initialize wandb in offline mode by default for sandbox
                os.environ.setdefault("WANDB_MODE", "offline")
                print("✅ W&B available - Use wandb.init() to start tracking")
            except Exception as e:
                print(f"⚠️ W&B initialization failed: {e}")
                self.wandb_available = False
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new experiment run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to associate with the run
            
        Returns:
            Run ID or identifier
        """
        run_id = f"run_{len(self.fallback_logs)}"
        
        if self.mlflow_available:
            self.mlflow.start_run(run_name=run_name, tags=tags)
            run_id = self.mlflow.active_run().info.run_id
            print(f"🚀 MLflow run started: {run_id}")
        
        if self.wandb_available:
            self.wandb.init(
                project=self.project_name,
                name=run_name,
                tags=list(tags.keys()) if tags else None,
                reinit=True
            )
            print(f"🚀 W&B run started: {self.wandb.run.name}")
        
        # Fallback tracking
        self.fallback_logs.append({
            "run_id": run_id,
            "run_name": run_name,
            "tags": tags or {},
            "metrics": {},
            "params": {},
            "artifacts": []
        })
        
        return run_id
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters for the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.mlflow_available:
            self.mlflow.log_params(params)
        
        if self.wandb_available and self.wandb.run:
            self.wandb.config.update(params)
        
        # Fallback
        if self.fallback_logs:
            self.fallback_logs[-1]["params"].update(params)
        self.fallback_params.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        if self.mlflow_available:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step)
        
        if self.wandb_available and self.wandb.run:
            self.wandb.log(metrics, step=step)
        
        # Fallback
        if self.fallback_logs:
            current_metrics = self.fallback_logs[-1]["metrics"]
            for key, value in metrics.items():
                if key not in current_metrics:
                    current_metrics[key] = []
                current_metrics[key].append({"value": value, "step": step})
        
        for key, value in metrics.items():
            if key not in self.fallback_metrics:
                self.fallback_metrics[key] = []
            self.fallback_metrics[key].append(value)
    
    def log_model(self, model: Any, model_name: str, framework: str = "sklearn"):
        """
        Log a trained model.
        
        Args:
            model: Trained model object
            model_name: Name to save the model under
            framework: Framework used ("sklearn", "tensorflow", "pytorch")
        """
        if self.mlflow_available:
            try:
                if framework == "sklearn":
                    self.mlflow.sklearn.log_model(model, model_name)
                elif framework == "tensorflow":
                    self.mlflow.tensorflow.log_model(model, model_name)
                elif framework == "pytorch":
                    self.mlflow.pytorch.log_model(model, model_name)
                else:
                    # Generic model logging
                    import joblib
                    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                        joblib.dump(model, f.name)
                        self.mlflow.log_artifact(f.name, f"models/{model_name}.joblib")
                        os.unlink(f.name)
                
                print(f"📊 Model '{model_name}' logged to MLflow")
            except Exception as e:
                print(f"⚠️ Model logging failed: {e}")
        
        if self.wandb_available and self.wandb.run:
            try:
                # Save model as W&B artifact
                import joblib
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                    joblib.dump(model, f.name)
                    
                    artifact = self.wandb.Artifact(
                        name=model_name,
                        type="model",
                        description=f"Trained {framework} model"
                    )
                    artifact.add_file(f.name)
                    self.wandb.log_artifact(artifact)
                    os.unlink(f.name)
                
                print(f"📊 Model '{model_name}' logged to W&B")
            except Exception as e:
                print(f"⚠️ W&B model logging failed: {e}")
        
        # Fallback
        if self.fallback_logs:
            self.fallback_logs[-1]["artifacts"].append({
                "name": model_name,
                "type": "model",
                "framework": framework,
                "logged_at": str(Path.cwd())
            })
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact file.
        
        Args:
            file_path: Path to the file to log
            artifact_path: Optional path within the artifact store
        """
        if self.mlflow_available:
            self.mlflow.log_artifact(file_path, artifact_path)
        
        if self.wandb_available and self.wandb.run:
            self.wandb.save(file_path)
        
        # Fallback
        if self.fallback_logs:
            self.fallback_logs[-1]["artifacts"].append({
                "file_path": file_path,
                "artifact_path": artifact_path,
                "type": "file"
            })
    
    def end_run(self):
        """End the current experiment run."""
        if self.mlflow_available:
            self.mlflow.end_run()
            print("🏁 MLflow run ended")
        
        if self.wandb_available and self.wandb.run:
            self.wandb.finish()
            print("🏁 W&B run finished")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary of tracked experiments.
        
        Returns:
            Dictionary with experiment summary
        """
        summary = {
            "total_runs": len(self.fallback_logs),
            "tracking_backends": [],
            "recent_metrics": self.fallback_metrics,
            "recent_params": self.fallback_params
        }
        
        if self.mlflow_available:
            summary["tracking_backends"].append("MLflow")
            try:
                experiment = self.mlflow.get_experiment_by_name(self.experiment_name)
                if experiment:
                    summary["mlflow_experiment_id"] = experiment.experiment_id
                    summary["mlflow_tracking_uri"] = self.mlflow.get_tracking_uri()
            except Exception:
                pass
        
        if self.wandb_available:
            summary["tracking_backends"].append("Weights & Biases")
        
        if not summary["tracking_backends"]:
            summary["tracking_backends"].append("Fallback (local)")
        
        return summary
    
    def create_comparison_demo(self) -> Dict[str, Any]:
        """
        Create a demonstration comparing different tracking tools.
        
        Returns:
            Dictionary with comparison and examples
        """
        comparison = {
            "mlflow": {
                "status": "✅ Available" if self.mlflow_available else "❌ Not installed",
                "install_cmd": "pip install mlflow",
                "strengths": [
                    "🎯 Model-centric workflow",
                    "🗂️ Built-in model registry",
                    "🚀 Easy model deployment",
                    "🔧 Framework-agnostic",
                    "📊 Comprehensive UI"
                ],
                "use_cases": [
                    "Model versioning and deployment",
                    "Comparing model performance",
                    "Reproducible ML pipelines",
                    "Team collaboration on ML projects"
                ]
            },
            "wandb": {
                "status": "✅ Available" if self.wandb_available else "❌ Not installed", 
                "install_cmd": "pip install wandb",
                "strengths": [
                    "📈 Beautiful visualization dashboards",
                    "🤝 Great for collaboration",
                    "🎨 Rich media logging (images, audio, 3D)",
                    "📱 Mobile app support",
                    "🧪 Experiment sweeps for hyperparameter optimization"
                ],
                "use_cases": [
                    "Research and experimentation",
                    "Hyperparameter sweeps",
                    "Team dashboards and reporting",
                    "Rich visualizations and media logging"
                ]
            }
        }
        
        # Add usage examples
        comparison["example_workflow"] = {
            "1_start_run": "tracker.start_run('my-experiment')",
            "2_log_params": "tracker.log_params({'lr': 0.01, 'epochs': 100})",
            "3_log_metrics": "tracker.log_metrics({'accuracy': 0.95, 'loss': 0.05})",
            "4_log_model": "tracker.log_model(model, 'my-model', 'sklearn')",
            "5_end_run": "tracker.end_run()"
        }
        
        return comparison
    
    def export_fallback_logs(self, output_file: str = "experiment_logs.json"):
        """
        Export fallback logs to a JSON file.
        
        Args:
            output_file: Path to save the logs
        """
        with open(output_file, 'w') as f:
            json.dump({
                "project_name": self.project_name,
                "experiment_name": self.experiment_name,
                "runs": self.fallback_logs,
                "summary_metrics": self.fallback_metrics,
                "summary_params": self.fallback_params
            }, f, indent=2)
        
        print(f"📁 Experiment logs exported to {output_file}")
        return output_file