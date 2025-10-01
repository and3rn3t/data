"""
Data Pipeline Integration Tests
Tests end-to-end data processing workflows and pipeline orchestration
"""

import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from sandbox.integrations.data_pipeline_builder import DataPipelineBuilder
from sandbox.integrations.modern_data_processing import ModernDataProcessor


class TestDataPipelineIntegration:
    """Test end-to-end data pipeline workflows"""

    def setup_method(self) -> None:
        """Setup test environment"""
        self.pipeline_builder = DataPipelineBuilder()
        self.processor = ModernDataProcessor()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        """Cleanup test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_complete_data_processing_workflow(self) -> None:
        """Test complete data processing from raw data to analysis"""
        # Create sample data
        raw_data = self.processor.create_sample_dataset(
            n_rows=1000, dataset_type="sales"
        )

        # Define transformation pipeline
        def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["revenue_category"] = pd.cut(
                df["total_amount"], bins=3, labels=["Low", "Medium", "High"]
            )
            return df

        def filter_recent_data(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            # Keep only recent 80% of data
            recent_cutoff = df["order_date"].quantile(0.2)
            return df[df["order_date"] >= recent_cutoff]

        transformations = [
            self.pipeline_builder._clean_missing_values,
            self.pipeline_builder._remove_duplicates,
            add_derived_fields,
            filter_recent_data,
        ]

        # Execute pipeline
        result = self.pipeline_builder.create_data_pipeline(
            pipeline_name="integration_test_pipeline",
            data_source=raw_data,
            transformations=transformations,
        )

        # Verify pipeline execution
        assert result["success"] is True
        assert "processed_data" in result
        assert len(result["transformations"]) == 4
        assert all(t["success"] for t in result["transformations"])

        # Verify data quality improvements
        processed_data = result["processed_data"]
        assert len(processed_data) <= len(raw_data)  # Some filtering occurred
        assert "revenue_category" in processed_data.columns
        assert processed_data.isnull().sum().sum() == 0  # No nulls after cleaning

    @pytest.mark.integration
    def test_data_validation_integration(self) -> None:
        """Test data validation integrated with pipeline"""
        # Create test data with known issues
        problematic_data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, None],  # Missing value
                "score": [95, 87, "invalid", 92, 78],  # Invalid type
                "category": ["A", "B", "A", "C", "B"],
            }
        )

        # Create validation suite
        validation_result = self.pipeline_builder.create_data_validation_suite(
            problematic_data, "integration_test_suite"
        )

        # Verify validation detected issues
        assert "expectations" in validation_result
        assert len(validation_result["expectations"]) > 0

        if "validation_results" in validation_result:
            # If using Great Expectations or Pandera, check for failures
            assert "success" in validation_result["validation_results"]

    @pytest.mark.integration
    def test_pipeline_with_file_io(self) -> None:
        """Test pipeline with file input/output operations"""
        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": range(1, 101),
                "metric_value": range(50, 150),
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            }
        )

        input_file = self.temp_dir / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        # Run pipeline with file input
        result = self.pipeline_builder.create_data_pipeline(
            pipeline_name="file_io_test",
            data_source=str(input_file),
            transformations=[
                self.pipeline_builder._clean_missing_values,
                self.pipeline_builder._standardize_column_names,
            ],
        )

        # Verify successful processing
        assert result["success"] is True
        processed_data = result["processed_data"]
        assert len(processed_data) == 100
        assert "user_id" in processed_data.columns

        # Test export functionality
        output_file = self.temp_dir / "pipeline_output.json"
        export_path = self.pipeline_builder.export_pipeline_history(str(output_file))

        assert Path(export_path).exists()
        assert len(self.pipeline_builder.pipeline_runs) >= 1

    @pytest.mark.integration
    def test_error_handling_in_pipeline(self) -> None:
        """Test error handling and recovery in data pipelines"""
        test_data = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": ["A", "B", "C", "D", "E"]}
        )

        def failing_transformation(df: pd.DataFrame) -> pd.DataFrame:
            """Transformation that will fail"""
            raise ValueError("Intentional failure for testing")

        def recovery_transformation(df: pd.DataFrame) -> pd.DataFrame:
            """Transformation that works"""
            return df.copy()

        transformations = [
            self.pipeline_builder._clean_missing_values,  # Should work
            failing_transformation,  # Should fail
            recovery_transformation,  # Should work
        ]

        result = self.pipeline_builder.create_data_pipeline(
            pipeline_name="error_handling_test",
            data_source=test_data,
            transformations=transformations,
        )

        # Pipeline should complete but report failures
        assert result["success"] is True  # Overall pipeline completes

        # Check individual transformation results
        transformation_results = result["transformations"]
        assert len(transformation_results) == 3
        assert transformation_results[0]["success"] is True  # Clean works
        assert transformation_results[1]["success"] is False  # Failing fails
        assert transformation_results[2]["success"] is True  # Recovery works

    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_monitoring(self) -> None:
        """Test pipeline performance monitoring and benchmarking"""
        # Create larger dataset for performance testing
        large_data = self.processor.create_sample_dataset(
            n_rows=5000, dataset_type="ecommerce"
        )

        # Run pipeline with timing
        import time

        start_time = time.time()

        result = self.pipeline_builder.create_data_pipeline(
            pipeline_name="performance_test",
            data_source=large_data,
            transformations=[
                self.pipeline_builder._clean_missing_values,
                self.pipeline_builder._remove_duplicates,
                self.pipeline_builder._standardize_column_names,
            ],
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify performance metrics
        assert result["success"] is True
        assert "duration_seconds" in result
        assert result["duration_seconds"] > 0
        assert execution_time < 30  # Should complete within 30 seconds

        # Check data quality report
        quality_report = self.pipeline_builder.create_data_quality_report(
            result["processed_data"]
        )
        assert "total_rows" in quality_report
        assert "total_columns" in quality_report
        assert quality_report["total_rows"] > 0

    @pytest.mark.integration
    def test_cross_tool_integration(self) -> None:
        """Test integration between different tools (DuckDB + Pipeline)"""
        if not self.processor.duckdb_available:
            pytest.skip("DuckDB not available")

        # Create test data
        sales_data = self.processor.create_sample_dataset(
            n_rows=1000, dataset_type="sales"
        )

        # Process with pipeline first
        pipeline_result = self.pipeline_builder.create_data_pipeline(
            pipeline_name="cross_tool_test",
            data_source=sales_data,
            transformations=[self.pipeline_builder._clean_missing_values],
        )

        processed_data = pipeline_result["processed_data"]

        # Then analyze with DuckDB
        analysis_query = """
        SELECT
            product_category,
            COUNT(*) as order_count,
            AVG(total_amount) as avg_amount,
            SUM(total_amount) as total_revenue
        FROM df
        GROUP BY product_category
        ORDER BY total_revenue DESC
        """

        analysis_result = self.processor.query_with_sql(processed_data, analysis_query)

        # Verify cross-tool integration
        assert isinstance(analysis_result, pd.DataFrame)
        assert len(analysis_result) > 0
        assert "product_category" in analysis_result.columns
        assert "total_revenue" in analysis_result.columns
        assert pipeline_result["success"] is True
