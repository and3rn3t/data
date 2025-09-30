import json
import os
import tempfile
from typing import Any, Dict, Generator
from unittest.mock import patch

import pytest

from sandbox.core.game_engine import GameEngine


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def mock_progress_file(temp_dir: str) -> str:
    """Create a mock progress file for testing."""
    progress_data = {
        "player_name": "Test Player",
        "current_level": 1,
        "experience_points": 0,
        "badges_earned": [],
        "challenges_completed": [],
        "time_spent": 0,
        "created_at": "2024-01-01T00:00:00",
        "last_played": "2024-01-01T00:00:00",
        "level_progress": {
            str(i): {"unlocked": i == 1, "completed": False, "score": 0}
            for i in range(1, 7)
        },
    }

    progress_file = os.path.join(temp_dir, "progress.json")
    with open(progress_file, "w") as f:
        json.dump(progress_data, f)

    return progress_file


@pytest.fixture
def game_engine(mock_progress_file: str) -> GameEngine:
    """Create a GameEngine instance with mock data."""
    return GameEngine(save_file=mock_progress_file)


@pytest.fixture
def sample_datasets() -> Dict[str, Any]:
    """Create sample datasets for testing."""
    import pandas as pd

    datasets = {}

    # Simple dataset
    datasets["simple"] = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["New York", "London", "Tokyo"],
        }
    )

    # Dataset with missing values
    datasets["with_nulls"] = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [10, None, 30, 40],
            "category": ["A", "B", None, "A"],
        }
    )

    # Time series dataset
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    datasets["timeseries"] = pd.DataFrame(
        {"date": dates, "value": [i * 10 + i**2 for i in range(10)]}
    )

    return datasets


@pytest.fixture
def mock_jupyter() -> Generator[Any, None, None]:
    """Mock Jupyter Lab functionality for testing."""
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value.returncode = 0
        yield mock_subprocess
