# API Reference

This document provides detailed information about the Data Science Sandbox API and core components.

## Table of Contents

- [Core Classes](#core-classes)
- [GameEngine](#gameengine)
- [Dashboard](#dashboard)
- [Modern Data Tools](#modern-data-tools)
- [Data Validation](#data-validation)
- [ML Integrations](#ml-integrations)
- [Configuration](#configuration)

## Core Classes

### GameEngine

The main game logic and progress tracking system.

```python
from sandbox.core.game_engine import GameEngine

# Initialize game engine
game = GameEngine()

# Get player progress
progress = game.get_progress()
stats = game.get_stats()

# Complete a challenge
game.complete_challenge("level_1_data_loading")

# Check available challenges
challenges = game.get_available_challenges()
```

#### Methods

##### `get_progress() -> Dict[str, Any]`

Returns the current player progress including:

- `level`: Current level (1-7)
- `xp`: Experience points earned
- `challenges_completed`: List of completed challenge IDs
- `badges_earned`: List of earned badge IDs

##### `get_stats() -> Dict[str, Any]`

Returns player statistics:

- `level`: Current level
- `total_xp`: Total experience points
- `challenges_completed`: Number of completed challenges
- `badges_earned`: Number of badges earned
- `completion_percentage`: Overall completion percentage

##### `complete_challenge(challenge_id: str) -> bool`

Marks a challenge as completed and awards XP.

**Parameters:**

- `challenge_id`: Unique identifier for the challenge

**Returns:** `True` if challenge was successfully completed

##### `get_available_challenges() -> List[Dict[str, Any]]`

Returns list of challenges available at current level.

### Dashboard

Streamlit-based interactive dashboard for progress visualization.

```python
from sandbox.core.dashboard import Dashboard

# Initialize dashboard
dashboard = Dashboard()

# Run dashboard server
dashboard.run()
```

## Modern Data Tools

### ModernDataProcessor

High-performance data processing using DuckDB and Polars.

```python
from sandbox.integrations.modern_data_processing import ModernDataProcessor

# Initialize processor
processor = ModernDataProcessor()

# Process data with DuckDB
result = processor.duckdb_query(
    "SELECT * FROM df WHERE value > 100",
    data=your_dataframe
)

# Use Polars for fast operations
polars_df = processor.to_polars(pandas_df)
result = processor.polars_operations(polars_df)
```

#### Methods

##### `duckdb_query(query: str, data: pd.DataFrame) -> pd.DataFrame`

Execute SQL queries on pandas DataFrames using DuckDB.

**Parameters:**

- `query`: SQL query string
- `data`: Input pandas DataFrame

**Returns:** Query results as pandas DataFrame

##### `to_polars(df: pd.DataFrame) -> pl.DataFrame`

Convert pandas DataFrame to Polars DataFrame for faster operations.

##### `polars_operations(df: pl.DataFrame) -> pl.DataFrame`

Perform optimized operations using Polars backend.

### ExperimentTracker

MLflow integration for experiment tracking.

```python
from sandbox.integrations.ml_experiment_tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker()

# Start experiment
with tracker.start_experiment("my_experiment"):
    # Log parameters
    tracker.log_params({"learning_rate": 0.01, "epochs": 100})

    # Train model and log metrics
    model = train_model()
    tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})

    # Log model
    tracker.log_model(model, "my_model")
```

#### Methods

##### `start_experiment(name: str) -> MLflowExperiment`

Start a new MLflow experiment.

##### `log_params(params: Dict[str, Any]) -> None`

Log experiment parameters.

##### `log_metrics(metrics: Dict[str, float]) -> None`

Log experiment metrics.

##### `log_model(model: Any, name: str) -> None`

Log trained model artifact.

## Data Validation

### Schema Validation

Data quality validation using Pandera schemas.

```python
from sandbox.utils.data_validation import get_iris_schema, validate_completeness

# Validate data against schema
schema = get_iris_schema()
validated_df = schema.validate(iris_df)

# Check data completeness
completeness = validate_completeness(df, threshold=0.95)
for col, result in completeness.items():
    if not result['passes_threshold']:
        print(f"Column {col} has low completeness: {result['ratio']:.2%}")
```

#### Available Schemas

##### `get_iris_schema() -> pa.DataFrameSchema`

Returns validation schema for Iris dataset with:

- Numeric columns with range validation
- Species column with category validation

##### `get_sales_schema() -> pa.DataFrameSchema`

Returns validation schema for sales dataset with:

- Date validation
- Numeric constraints
- Business rule validation

#### Validation Functions

##### `validate_completeness(df: pd.DataFrame, threshold: float = 0.95) -> Dict`

Check data completeness for each column.

**Parameters:**

- `df`: DataFrame to validate
- `threshold`: Minimum completeness ratio (default: 0.95)

**Returns:** Dictionary with completeness results per column

##### `validate_data_types(df: pd.DataFrame, expected_types: Dict) -> Dict`

Validate column data types against expected types.

##### `validate_no_duplicates(df: pd.DataFrame, subset: List[str] = None) -> bool`

Check for duplicate rows in dataset.

## ML Integrations

### ModelExplainer

Model interpretability and explainability tools.

```python
from sandbox.integrations.model_explainability import ModelExplainer

# Initialize explainer
explainer = ModelExplainer()

# Explain model predictions
explanations = explainer.explain_predictions(
    model=trained_model,
    X_train=X_train,
    X_explain=X_test_sample,
    method="shap"  # or "lime", "auto"
)

# Generate model evaluation report
evaluation = explainer.evaluate_model(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    task_type="classification"
)
```

#### Methods

##### `explain_predictions(model, X_train, X_explain, method="auto") -> Dict`

Generate model explanations using SHAP, LIME, or simple methods.

##### `evaluate_model(model, X_test, y_test, task_type="auto") -> Dict`

Create comprehensive model evaluation report.

### HyperparameterOptimizer

Automated hyperparameter tuning using multiple backends.

```python
from sandbox.integrations.hyperparameter_tuning import HyperparameterOptimizer

# Initialize optimizer
optimizer = HyperparameterOptimizer()

# Define parameter space
param_space = {
    'n_estimators': (10, 100),
    'max_depth': [3, 5, 7, 10],
    'learning_rate': (0.01, 0.3)
}

# Optimize hyperparameters
best_params, study = optimizer.optimize_optuna(
    objective_func=your_objective_function,
    param_space=param_space,
    n_trials=100
)
```

## Configuration

### Environment Setup

```python
from sandbox.utils.modern_tools_config import initialize_all_tools
from sandbox.utils.logging_config import setup_logging

# Initialize all modern tools
tools = initialize_all_tools()

# Setup logging
setup_logging()
```

### Configuration Files

- **`config.py`**: Main application configuration
- **`pyproject.toml`**: Tool configurations (Black, Ruff, etc.)
- **`.pre-commit-config.yaml`**: Code quality automation
- **`requirements.txt`**: Core dependencies
- **`requirements-dev.txt`**: Development tools

## Error Handling

All API functions include comprehensive error handling:

```python
try:
    result = some_api_function()
except ValidationError as e:
    # Handle data validation errors
    print(f"Data validation failed: {e}")
except MLflowException as e:
    # Handle MLflow-specific errors
    print(f"MLflow error: {e}")
except Exception as e:
    # Handle general errors
    print(f"Unexpected error: {e}")
```

## Type Hints

The codebase uses comprehensive type hints for better IDE support:

```python
from typing import Dict, List, Optional, Union
import pandas as pd

def process_data(
    df: pd.DataFrame,
    columns: List[str],
    threshold: Optional[float] = None
) -> Dict[str, Union[pd.DataFrame, Dict[str, Any]]]:
    """Process data with type-safe parameters."""
    pass
```

## Testing

All API components include comprehensive tests:

```python
import pytest
from sandbox.core.game_engine import GameEngine

def test_game_engine_initialization():
    """Test game engine initialization."""
    game = GameEngine()
    assert game.get_progress()['level'] == 1
    assert game.get_progress()['xp'] == 0

def test_challenge_completion():
    """Test challenge completion logic."""
    game = GameEngine()
    result = game.complete_challenge("level_1_data_loading")
    assert result is True
    assert "level_1_data_loading" in game.get_progress()['challenges_completed']
```

Run tests with:

```bash
pytest tests/ -v --cov=sandbox --cov-report=html
```
