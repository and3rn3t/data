"""
Modern Data Processing Tools Integration

Provides access to high-performance data processing libraries like Polars, DuckDB,
and advanced PyArrow operations for the Data Science Sandbox.
"""

import warnings
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


class ModernDataProcessor:
    """
    Integration class for modern data processing tools.

    Provides easy access to Polars, DuckDB, and advanced PyArrow operations
    with fallbacks to pandas when libraries aren't available.
    """

    def __init__(self):
        """Initialize the processor with available libraries."""
        self.polars_available = self._check_polars()
        self.duckdb_available = self._check_duckdb()
        self.pyarrow_available = self._check_pyarrow()

    def _check_polars(self) -> bool:
        """Check if Polars is available."""
        try:
            import polars as pl

            self.pl = pl
            return True
        except ImportError:
            return False

    def _check_duckdb(self) -> bool:
        """Check if DuckDB is available."""
        try:
            import duckdb

            self.duckdb = duckdb
            return True
        except ImportError:
            return False

    def _check_pyarrow(self) -> bool:
        """Check if PyArrow is available."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            self.pa = pa
            self.pq = pq
            return True
        except ImportError:
            return False

    def create_sample_dataset(
        self, n_rows: int = 10000, dataset_type: str = "sales"
    ) -> Union[pd.DataFrame, Any]:
        """
        Create sample datasets using the fastest available library.

        Args:
            n_rows: Number of rows to generate
            dataset_type: Type of dataset ("sales", "ecommerce", "iot", "financial")

        Returns:
            DataFrame (Polars if available, otherwise pandas)
        """
        np.random.seed(42)

        if dataset_type == "sales":
            data = self._create_sales_data(n_rows)
        elif dataset_type == "ecommerce":
            data = self._create_ecommerce_data(n_rows)
        elif dataset_type == "iot":
            data = self._create_iot_data(n_rows)
        elif dataset_type == "financial":
            data = self._create_financial_data(n_rows)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        if self.polars_available:
            return self.pl.DataFrame(data)
        return pd.DataFrame(data)

    def _create_sales_data(self, n_rows: int) -> Dict[str, List]:
        """Create sales dataset."""
        return {
            "order_id": [f"ORD-{i:06d}" for i in range(n_rows)],
            "customer_id": np.random.choice(
                [f"CUST-{i:04d}" for i in range(1000)], n_rows
            ),
            "product_category": np.random.choice(
                ["Electronics", "Clothing", "Books", "Home", "Sports"],
                n_rows,
                p=[0.3, 0.25, 0.15, 0.2, 0.1],
            ),
            "sales_amount": np.random.gamma(2, 50, n_rows),
            "profit_margin": np.random.beta(2, 5, n_rows),
            "region": np.random.choice(["North", "South", "East", "West"], n_rows),
            "order_date": pd.date_range("2023-01-01", periods=n_rows, freq="H"),
            "shipping_cost": np.random.exponential(10, n_rows),
        }

    def _create_ecommerce_data(self, n_rows: int) -> Dict[str, List]:
        """Create e-commerce behavior dataset."""
        return {
            "session_id": [f"SESS-{i:08d}" for i in range(n_rows)],
            "user_id": np.random.choice([f"USER-{i:05d}" for i in range(5000)], n_rows),
            "page_views": np.random.poisson(5, n_rows),
            "time_on_site": np.random.exponential(300, n_rows),  # seconds
            "bounce_rate": np.random.beta(2, 8, n_rows),
            "conversion": np.random.choice([0, 1], n_rows, p=[0.97, 0.03]),
            "device_type": np.random.choice(
                ["mobile", "desktop", "tablet"], n_rows, p=[0.6, 0.3, 0.1]
            ),
            "traffic_source": np.random.choice(
                ["organic", "paid", "social", "direct"], n_rows
            ),
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="T"),
        }

    def _create_iot_data(self, n_rows: int) -> Dict[str, List]:
        """Create IoT sensor dataset."""
        base_temp = 25.0
        base_humidity = 60.0
        return {
            "sensor_id": np.random.choice(
                [f"SENSOR-{i:03d}" for i in range(50)], n_rows
            ),
            "temperature": base_temp + np.random.normal(0, 5, n_rows),
            "humidity": np.clip(
                base_humidity + np.random.normal(0, 15, n_rows), 0, 100
            ),
            "pressure": 1013.25 + np.random.normal(0, 10, n_rows),
            "light_intensity": np.random.exponential(500, n_rows),
            "battery_level": np.clip(100 - np.random.exponential(20, n_rows), 5, 100),
            "location": np.random.choice(["indoor", "outdoor"], n_rows),
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="5min"),
        }

    def _create_financial_data(self, n_rows: int) -> Dict[str, List]:
        """Create financial market dataset."""
        return {
            "symbol": np.random.choice(
                ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"], n_rows
            ),
            "price": 100 + np.random.normal(0, 20, n_rows),
            "volume": np.random.lognormal(10, 1, n_rows).astype(int),
            "high": 105 + np.random.normal(0, 20, n_rows),
            "low": 95 + np.random.normal(0, 20, n_rows),
            "market_cap": np.random.lognormal(15, 1, n_rows),
            "sector": np.random.choice(
                ["Tech", "Finance", "Healthcare", "Energy"], n_rows
            ),
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="1min"),
        }

    def query_with_sql(
        self, data: Union[pd.DataFrame, Any], query: str
    ) -> Union[pd.DataFrame, Any]:
        """
        Execute SQL queries on DataFrames using DuckDB.

        Args:
            data: DataFrame to query
            query: SQL query string

        Returns:
            Query results as DataFrame
        """
        if not self.duckdb_available:
            warnings.warn("DuckDB not available. Install with: pip install duckdb")
            return data.head()  # Return sample as fallback

        conn = self.duckdb.connect()

        if self.polars_available and hasattr(data, "to_pandas"):
            # Convert Polars to pandas for DuckDB
            pandas_data = data.to_pandas()
        else:
            pandas_data = data

        conn.register("df", pandas_data)
        result = conn.execute(query).fetchdf()
        conn.close()

        if self.polars_available:
            return self.pl.from_pandas(result)
        return result

    def optimize_datatypes(
        self, df: Union[pd.DataFrame, Any]
    ) -> Union[pd.DataFrame, Any]:
        """
        Optimize DataFrame memory usage by converting to appropriate data types.

        Args:
            df: DataFrame to optimize

        Returns:
            Optimized DataFrame
        """
        if self.polars_available and hasattr(df, "with_columns"):
            # Polars has built-in type optimization
            return df.with_columns(
                [
                    col.cast(self.pl.Float32) if col.dtype == self.pl.Float64 else col
                    for col in df.get_columns()
                    if col.dtype == self.pl.Float64
                ]
            )

        # Pandas optimization
        for col in df.columns:
            if df[col].dtype == "float64":
                df[col] = pd.to_numeric(df[col], downcast="float")
            elif df[col].dtype == "int64":
                df[col] = pd.to_numeric(df[col], downcast="integer")

        return df

    def performance_comparison_demo(self) -> Dict[str, str]:
        """
        Demonstrate performance differences between Pandas and Polars.

        Returns:
            Dictionary with comparison results
        """
        results = {}

        if self.polars_available:
            results["polars_status"] = (
                "✅ Available - High-performance DataFrame library"
            )
            results["polars_benefits"] = [
                "🚀 2-30x faster than pandas for many operations",
                "🧠 Better memory efficiency",
                "⚡ Lazy evaluation for query optimization",
                "🔧 Built-in parallel processing",
            ]
        else:
            results["polars_status"] = (
                "❌ Not installed - Install with: pip install polars"
            )

        if self.duckdb_available:
            results["duckdb_status"] = "✅ Available - Fast analytical database"
            results["duckdb_benefits"] = [
                "🗄️ SQL queries on DataFrame data",
                "📊 Optimized for analytical workloads",
                "🔄 Zero-copy integration with pandas/arrow",
                "📈 Columnar storage for better performance",
            ]
        else:
            results["duckdb_status"] = (
                "❌ Not installed - Install with: pip install duckdb"
            )

        if self.pyarrow_available:
            results["pyarrow_status"] = "✅ Available - Columnar in-memory analytics"
            results["pyarrow_benefits"] = [
                "🗂️ Efficient columnar data format",
                "💾 Fast parquet/feather file I/O",
                "🔗 Interoperability with many tools",
                "⚡ Zero-copy data sharing",
            ]
        else:
            results["pyarrow_status"] = (
                "❌ Not installed - Install with: pip install pyarrow"
            )

        return results

    def get_tool_recommendations(self) -> Dict[str, str]:
        """Get recommendations for when to use each tool."""
        return {
            "pandas": "📊 Traditional choice, great for small-medium datasets and data exploration",
            "polars": "🚀 Use for large datasets, performance-critical operations, or when memory efficiency matters",
            "duckdb": "🗄️ Perfect for SQL users, analytical queries, and when you need database-like operations on files",
            "pyarrow": "💾 Essential for working with parquet files, interoperability, and when you need columnar data processing",
        }
