"""
Integration Test Configuration and Fixtures
Provides shared configuration and fixtures for integration tests
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Generator

import pytest


@pytest.fixture(scope="session")
def integration_test_dir() -> Generator[Path, None, None]:
    """Session-scoped temporary directory for integration tests"""
    temp_dir = Path(tempfile.mkdtemp(prefix="integration_test_"))
    yield temp_dir

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def clean_environment() -> Generator[None, None, None]:
    """Ensure clean environment for each test"""
    # Store original environment
    original_env = dict(os.environ)

    # Set test environment variables
    os.environ["PYTHONPATH"] = str(Path.cwd())
    os.environ["TEST_MODE"] = "true"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_data_dir(integration_test_dir: Path) -> Path:
    """Create mock data directory with test files"""
    data_dir = integration_test_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Create sample test files
    import pandas as pd

    # Sample sales data
    sales_data = pd.DataFrame(
        {
            "order_id": range(1, 101),
            "customer_id": range(1, 101),
            "product_name": [f"Product_{i%10}" for i in range(100)],
            "quantity": [i % 5 + 1 for i in range(100)],
            "price": [10.0 + (i % 20) for i in range(100)],
            "order_date": pd.date_range("2023-01-01", periods=100, freq="D"),
        }
    )
    sales_data.to_csv(data_dir / "sample_sales.csv", index=False)

    # Sample user data
    user_data = pd.DataFrame(
        {
            "user_id": range(1, 51),
            "username": [f"user_{i}" for i in range(50)],
            "level": [i % 7 + 1 for i in range(50)],
            "experience": [i * 10 for i in range(50)],
            "join_date": pd.date_range("2023-01-01", periods=50, freq="W"),
        }
    )
    user_data.to_csv(data_dir / "sample_users.csv", index=False)

    return data_dir


class IntegrationTestConfig:
    """Configuration class for integration tests"""

    # Test timeouts (in seconds)
    FAST_TIMEOUT = 5
    MEDIUM_TIMEOUT = 15
    SLOW_TIMEOUT = 60

    # Test data sizes
    SMALL_DATASET = 100
    MEDIUM_DATASET = 1000
    LARGE_DATASET = 10000

    # Database configuration
    TEST_DB_NAME = "test_sandbox.duckdb"

    # MLflow configuration
    MLFLOW_TRACKING_URI = "file:./test_mlruns"
    MLFLOW_ARTIFACT_LOCATION = "./test_mlartifacts"

    @classmethod
    def get_test_database_path(cls, test_dir: Path) -> str:
        """Get path for test database"""
        return str(test_dir / cls.TEST_DB_NAME)

    @classmethod
    def setup_mlflow_test_env(cls, test_dir: Path) -> None:
        """Setup MLflow test environment"""
        os.environ["MLFLOW_TRACKING_URI"] = f"file://{test_dir}/mlruns"
        os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = f"{test_dir}/mlartifacts"


def pytest_configure(config: Any) -> None:
    """Configure pytest for integration tests"""
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow integration tests")
    config.addinivalue_line("markers", "database: Database integration tests")
    config.addinivalue_line("markers", "mlflow: MLflow integration tests")
    config.addinivalue_line("markers", "pipeline: Data pipeline integration tests")
    config.addinivalue_line("markers", "workflow: End-to-end workflow tests")


# Integration test utilities
class IntegrationTestUtils:
    """Utility functions for integration tests"""

    @staticmethod
    def wait_for_condition(
        condition_func: Callable[[], bool], timeout: int = 10, interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true"""
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False

    @staticmethod
    def verify_file_exists(file_path: Path, timeout: int = 5) -> bool:
        """Verify file exists within timeout"""

        def check_file() -> bool:
            return file_path.exists()

        return IntegrationTestUtils.wait_for_condition(check_file, timeout)

    @staticmethod
    def create_test_progress_file(save_path: Path) -> dict:
        """Create a test progress file"""
        import json

        test_progress = {
            "player_name": "Integration Test Player",
            "current_level": 1,
            "experience_points": 0,
            "challenges_completed": [],
            "badges_earned": [],
            "level_progress": {f"level_{i}": {"unlocked": i == 1} for i in range(1, 8)},
        }

        with open(save_path, "w") as f:
            json.dump(test_progress, f, indent=2)

        return test_progress

    @staticmethod
    def cleanup_mlflow_artifacts(test_dir: Path) -> None:
        """Clean up MLflow artifacts after tests"""
        import shutil

        mlruns_dir = test_dir / "mlruns"
        mlartifacts_dir = test_dir / "mlartifacts"

        if mlruns_dir.exists():
            shutil.rmtree(mlruns_dir, ignore_errors=True)
        if mlartifacts_dir.exists():
            shutil.rmtree(mlartifacts_dir, ignore_errors=True)


# Pytest collection modification
def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Modify test collection for integration tests"""
    for item in items:
        # Add integration marker to all tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add slow marker to tests marked as slow
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
