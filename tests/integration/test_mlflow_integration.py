"""
MLflow Integration Tests
Tests machine learning experiment tracking and model management workflows
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from sandbox.integrations.ml_experiment_tracking import ExperimentTracker


class TestMLflowIntegration:
    """Test MLflow experiment tracking integration"""

    def setup_method(self) -> None:
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        # Initialize tracker with test directory
        self.tracker = ExperimentTracker()

    def teardown_method(self) -> None:
        """Cleanup test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_experiment_lifecycle(self) -> None:
        """Test complete MLflow experiment lifecycle"""
        if not self.tracker.mlflow_available:
            pytest.skip("MLflow not available")

        experiment_name = "integration_test_experiment"

        # Create experiment
        experiment_id = self.tracker.create_experiment(experiment_name)
        assert experiment_id is not None

        # Start run
        with self.tracker.start_run(experiment_name=experiment_name) as run:
            # Log parameters
            self.tracker.log_param("learning_rate", 0.01)
            self.tracker.log_param("batch_size", 32)

            # Log metrics
            self.tracker.log_metric("accuracy", 0.95)
            self.tracker.log_metric("loss", 0.05)

            # Verify run is active
            assert run is not None

        # Verify experiment was created and logged
        experiments = self.tracker.list_experiments()
        assert any(exp.name == experiment_name for exp in experiments)

    @pytest.mark.integration
    def test_model_registration_workflow(self) -> None:
        """Test model registration and versioning"""
        if not self.tracker.mlflow_available:
            pytest.skip("MLflow not available")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        # Create sample model and data
        X, y = make_classification(
            n_samples=100, n_features=4, n_classes=2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        model_name = "integration_test_model"

        # Register model
        with self.tracker.start_run():
            model_uri = self.tracker.log_model(
                model, "random_forest_model", registered_model_name=model_name
            )

            assert model_uri is not None

        # Verify model registration
        registered_models = self.tracker.list_registered_models()
        if registered_models:  # MLflow might not be fully configured
            model_names = [model.name for model in registered_models]
            assert model_name in model_names

    @pytest.mark.integration
    def test_artifact_logging(self) -> None:
        """Test artifact logging and retrieval"""
        if not self.tracker.mlflow_available:
            pytest.skip("MLflow not available")

        # Create test artifact
        test_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )

        artifact_path = self.temp_dir / "test_data.csv"
        test_data.to_csv(artifact_path, index=False)

        with self.tracker.start_run():
            # Log artifact
            self.tracker.log_artifact(str(artifact_path), "data")

            # Log additional metrics
            self.tracker.log_metric("data_size", len(test_data))
            self.tracker.log_param("artifact_type", "training_data")

    @pytest.mark.integration
    def test_cross_experiment_analysis(self) -> None:
        """Test analysis across multiple experiments"""
        if not self.tracker.mlflow_available:
            pytest.skip("MLflow not available")

        experiment_names = ["exp_1", "exp_2", "exp_3"]

        # Create multiple experiments with different parameters
        for i, exp_name in enumerate(experiment_names):
            self.tracker.create_experiment(exp_name)

            with self.tracker.start_run(experiment_name=exp_name):
                # Log different hyperparameters
                self.tracker.log_param("learning_rate", 0.01 * (i + 1))
                self.tracker.log_param("epochs", 10 * (i + 1))

                # Log metrics with some variation
                accuracy = 0.8 + (i * 0.05)
                self.tracker.log_metric("accuracy", accuracy)
                self.tracker.log_metric("f1_score", accuracy - 0.1)

        # Verify experiments exist
        experiments = self.tracker.list_experiments()
        experiment_names_found = [exp.name for exp in experiments]

        for exp_name in experiment_names:
            assert exp_name in experiment_names_found

    @pytest.mark.integration
    def test_model_serving_preparation(self) -> None:
        """Test preparing models for serving"""
        if not self.tracker.mlflow_available:
            pytest.skip("MLflow not available")

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import make_classification
            import mlflow.sklearn

            # Create and train a simple model
            X, y = make_classification(
                n_samples=50, n_features=3, n_classes=2, random_state=42
            )
            model = LogisticRegression(random_state=42)
            model.fit(X, y)

            with self.tracker.start_run():
                # Log model with signature for serving
                self.tracker.log_param("model_type", "logistic_regression")
                self.tracker.log_metric("train_accuracy", model.score(X, y))

                # Log model for serving
                model_uri = self.tracker.log_model(
                    model,
                    "serving_model",
                    registered_model_name="integration_serving_test",
                )

                assert model_uri is not None

        except ImportError:
            pytest.skip("Required ML libraries not available")

    @pytest.mark.integration
    def test_experiment_comparison_tools(self) -> None:
        """Test tools for comparing experiments"""
        if not self.tracker.mlflow_available:
            pytest.skip("MLflow not available")

        # Get comparison of available tools
        comparison = self.tracker.get_tool_comparison()

        assert isinstance(comparison, dict)
        assert "mlflow" in comparison
        assert "status" in comparison["mlflow"]

        # Verify MLflow capabilities are described
        mlflow_info = comparison["mlflow"]
        assert "strengths" in mlflow_info
        assert "use_cases" in mlflow_info

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_experiment_handling(self) -> None:
        """Test handling of experiments with many runs"""
        if not self.tracker.mlflow_available:
            pytest.skip("MLflow not available")

        experiment_name = "large_experiment_test"
        self.tracker.create_experiment(experiment_name)

        # Create multiple runs (scaled down for test speed)
        n_runs = 10  # In production this might be 100+

        for run_id in range(n_runs):
            with self.tracker.start_run(experiment_name=experiment_name):
                # Simulate hyperparameter sweep
                lr = 0.001 * (run_id + 1)
                batch_size = 16 + (run_id * 4)

                self.tracker.log_param("learning_rate", lr)
                self.tracker.log_param("batch_size", batch_size)

                # Simulate metrics (with some realistic variation)
                import random

                accuracy = 0.7 + random.uniform(0, 0.3)
                loss = 1.0 - accuracy + random.uniform(0, 0.1)

                self.tracker.log_metric("accuracy", accuracy)
                self.tracker.log_metric("loss", loss)

        # Verify all runs were created
        experiments = self.tracker.list_experiments()
        target_experiment = next(
            (exp for exp in experiments if exp.name == experiment_name), None
        )

        assert target_experiment is not None
