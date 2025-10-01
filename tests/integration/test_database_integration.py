"""
Database Integration Tests
Tests DuckDB connections, data persistence, and database operations
"""

import os
import tempfile

import pandas as pd
import pytest

from sandbox.integrations.modern_data_processing import ModernDataProcessor


class TestDatabaseIntegration:
    """Test database connectivity and operations"""

    def setup_method(self) -> None:
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_sandbox.duckdb")
        self.processor = ModernDataProcessor()

    def teardown_method(self) -> None:
        """Cleanup test environment"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    @pytest.mark.integration
    def test_duckdb_connection_establishment(self) -> None:
        """Test basic DuckDB connection setup"""
        # Override config for test
        import duckdb

        conn = duckdb.connect(self.test_db_path)

        # Test basic functionality
        result = conn.execute("SELECT 42 as test_value").fetchone()
        assert result[0] == 42

        conn.close()

    @pytest.mark.integration
    def test_duckdb_dataframe_operations(self) -> None:
        """Test DuckDB operations with pandas DataFrames"""
        if not self.processor.duckdb_available:
            pytest.skip("DuckDB not available")

        # Create test data
        test_data = pd.DataFrame(
            {
                "id": range(1, 101),
                "value": range(100, 201),
                "category": ["A", "B", "C"] * 33 + ["A"],
            }
        )

        # Test SQL queries on DataFrame
        query = "SELECT category, COUNT(*) as count, AVG(value) as avg_value FROM df GROUP BY category"
        result = self.processor.query_with_sql(test_data, query)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Three categories
        assert "count" in result.columns
        assert "avg_value" in result.columns

    @pytest.mark.integration
    def test_data_persistence_workflow(self) -> None:
        """Test complete data persistence workflow"""
        if not self.processor.duckdb_available:
            pytest.skip("DuckDB not available")

        import duckdb

        conn = duckdb.connect(self.test_db_path)

        # Create and populate test table
        test_data = pd.DataFrame(
            {
                "user_id": range(1, 11),
                "score": [85, 92, 78, 95, 88, 76, 91, 83, 89, 94],
                "level": [1, 2, 1, 3, 2, 1, 3, 2, 2, 3],
            }
        )

        # Insert data
        conn.register("temp_df", test_data)
        conn.execute(
            """
            CREATE TABLE user_progress AS
            SELECT * FROM temp_df
        """
        )

        # Verify data persisted
        result = conn.execute("SELECT COUNT(*) FROM user_progress").fetchone()
        assert result[0] == 10

        # Test aggregation
        avg_score = conn.execute("SELECT AVG(score) FROM user_progress").fetchone()[0]
        assert abs(avg_score - test_data["score"].mean()) < 0.01

        conn.close()

    @pytest.mark.integration
    def test_database_error_handling(self) -> None:
        """Test database error handling and recovery"""
        if not self.processor.duckdb_available:
            pytest.skip("DuckDB not available")

        import duckdb

        conn = duckdb.connect(self.test_db_path)

        # Test invalid SQL handling
        with pytest.raises(Exception):
            conn.execute("SELECT * FROM non_existent_table")

        # Test recovery after error
        result = conn.execute("SELECT 1 as recovery_test").fetchone()
        assert result[0] == 1

        conn.close()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_dataset_performance(self) -> None:
        """Test performance with larger datasets"""
        if not self.processor.duckdb_available:
            pytest.skip("DuckDB not available")

        # Create larger test dataset
        large_data = self.processor.create_sample_dataset(
            n_rows=10000, dataset_type="sales"
        )

        # Test query performance
        import time

        start_time = time.time()

        query = """
        SELECT
            DATE_TRUNC('month', order_date) as month,
            COUNT(*) as orders,
            SUM(total_amount) as revenue
        FROM df
        GROUP BY month
        ORDER BY month
        """

        result = self.processor.query_with_sql(large_data, query)
        end_time = time.time()

        # Should complete within reasonable time
        assert (end_time - start_time) < 10  # 10 seconds max
        assert len(result) > 0
        assert "orders" in result.columns
        assert "revenue" in result.columns

    @pytest.mark.integration
    def test_concurrent_database_access(self) -> None:
        """Test concurrent database operations"""
        if not self.processor.duckdb_available:
            pytest.skip("DuckDB not available")

        import duckdb
        import threading
        import time

        results = []

        def database_worker(worker_id: int):
            """Worker function for concurrent access"""
            conn = duckdb.connect(self.test_db_path)

            # Each worker creates and queries its own table
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS worker_{worker_id} AS
                SELECT {worker_id} as worker_id, generate_series as id
                FROM generate_series(1, 100)
            """
            )

            count = conn.execute(f"SELECT COUNT(*) FROM worker_{worker_id}").fetchone()[
                0
            ]
            results.append(count)

            conn.close()

        # Start multiple workers
        threads = []
        for i in range(3):
            thread = threading.Thread(target=database_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Verify all workers completed successfully
        assert len(results) == 3
        assert all(count == 100 for count in results)
