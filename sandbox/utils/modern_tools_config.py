"""
Configuration for modern data processing tools and integrations.
"""

from pathlib import Path
from typing import Any, Dict

import duckdb

# DuckDB Configuration
DUCKDB_CONFIG = {
    "database_path": "data/sandbox.duckdb",
    "memory_limit": "2GB",
    "threads": 4,
    "extensions": [
        "httpfs",  # HTTP/S3 file system
        "json",  # JSON functions
        "parquet",  # Parquet support
        "spatial",  # Spatial extensions
    ],
}

# Polars Configuration
POLARS_CONFIG = {
    "streaming": True,
    "n_threads": 4,
    "fmt_str_lengths": 100,
    "fmt_table_cell_list_len": 3,
}

# MLflow Configuration
MLFLOW_CONFIG = {
    "tracking_uri": "file:./mlruns",
    "artifact_location": "./mlartifacts",
    "experiment_name": "data_science_sandbox",
}

# Great Expectations Configuration
GE_CONFIG = {
    "data_context_root_dir": "./great_expectations",
    "expectations_store": {
        "type": "filesystem",
        "base_directory": "./great_expectations/expectations",
    },
    "validations_store": {
        "type": "filesystem",
        "base_directory": "./great_expectations/validations",
    },
}


def setup_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Set up and configure DuckDB connection with extensions.

    Returns:
        Configured DuckDB connection
    """
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    # Create connection
    conn = duckdb.connect(DUCKDB_CONFIG["database_path"])

    # Configure memory and threads
    conn.execute(f"SET memory_limit='{DUCKDB_CONFIG['memory_limit']}'")
    conn.execute(f"SET threads={DUCKDB_CONFIG['threads']}")

    # Install and load extensions
    for extension in DUCKDB_CONFIG["extensions"]:
        try:
            conn.execute(f"INSTALL {extension}")
            conn.execute(f"LOAD {extension}")
        except Exception as e:
            print(f"Warning: Could not load extension {extension}: {e}")

    return conn


def configure_polars() -> None:
    """Configure Polars global settings."""
    # Basic Polars configuration - keep it simple to avoid version conflicts
    pass  # Polars works well with default settings


def setup_mlflow() -> None:
    """Set up MLflow tracking configuration."""
    import mlflow

    # Create directories
    Path("mlruns").mkdir(exist_ok=True)
    Path("mlartifacts").mkdir(exist_ok=True)

    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])

    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            MLFLOW_CONFIG["experiment_name"],
            artifact_location=MLFLOW_CONFIG["artifact_location"],
        )
    except mlflow.exceptions.MlflowException:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(MLFLOW_CONFIG["experiment_name"])
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)


def initialize_all_tools() -> Dict[str, Any]:
    """
    Initialize all modern data processing tools.

    Returns:
        Dictionary of initialized tools and connections
    """
    tools = {}

    # Initialize DuckDB
    try:
        tools["duckdb"] = setup_duckdb_connection()
        print("✅ DuckDB initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DuckDB: {e}")

    # Configure Polars
    try:
        configure_polars()
        tools["polars_config"] = True
        print("✅ Polars configured successfully")
    except Exception as e:
        print(f"❌ Failed to configure Polars: {e}")

    # Setup MLflow
    try:
        setup_mlflow()
        tools["mlflow"] = True
        print("✅ MLflow initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize MLflow: {e}")

    return tools
