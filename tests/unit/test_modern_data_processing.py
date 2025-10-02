"""
Unit tests for ModernDataProcessor integration
"""

from unittest.mock import patch

import pandas as pd
import pytest

from sandbox.integrations.modern_data_processing import ModernDataProcessor


class TestModernDataProcessor:
    """Test suite for ModernDataProcessor class."""

    def test_initialization_without_optional_libraries(self):
        """Test initialization when optional libraries are not available."""
        with patch(
            "sandbox.integrations.modern_data_processing.ModernDataProcessor._check_polars",
            return_value=False,
        ), patch(
            "sandbox.integrations.modern_data_processing.ModernDataProcessor._check_duckdb",
            return_value=False,
        ), patch(
            "sandbox.integrations.modern_data_processing.ModernDataProcessor._check_pyarrow",
            return_value=False,
        ):

            processor = ModernDataProcessor()

            assert not processor.polars_available
            assert not processor.duckdb_available
            assert not processor.pyarrow_available

    def test_initialization_with_all_libraries(self):
        """Test initialization when all optional libraries are available."""
        with patch(
            "sandbox.integrations.modern_data_processing.ModernDataProcessor._check_polars",
            return_value=True,
        ), patch(
            "sandbox.integrations.modern_data_processing.ModernDataProcessor._check_duckdb",
            return_value=True,
        ), patch(
            "sandbox.integrations.modern_data_processing.ModernDataProcessor._check_pyarrow",
            return_value=True,
        ):

            processor = ModernDataProcessor()

            assert processor.polars_available
            assert processor.duckdb_available
            assert processor.pyarrow_available

    def test_check_polars_available(self):
        """Test Polars availability check when library is present."""
        processor = ModernDataProcessor()

        # This will depend on actual environment, but we test the method exists
        result = processor._check_polars()
        assert isinstance(result, bool)

    def test_check_polars_unavailable(self):
        """Test Polars availability check when library is missing."""
        with patch("builtins.__import__", side_effect=ImportError):
            processor = ModernDataProcessor()
            result = processor._check_polars()
            assert result is False

    def test_check_duckdb_available(self):
        """Test DuckDB availability check when library is present."""
        processor = ModernDataProcessor()

        # This will depend on actual environment
        result = processor._check_duckdb()
        assert isinstance(result, bool)

    def test_check_duckdb_unavailable(self):
        """Test DuckDB availability check when library is missing."""
        with patch("builtins.__import__", side_effect=ImportError):
            processor = ModernDataProcessor()
            result = processor._check_duckdb()
            assert result is False

    def test_create_sample_dataset_sales(self):
        """Test creating sample sales dataset."""
        processor = ModernDataProcessor()

        # Mock numpy random seed for reproducible results
        with patch("numpy.random.seed"):
            df = processor.create_sample_dataset(n_rows=100, dataset_type="sales")

            # Should return pandas DataFrame if polars not available
            assert isinstance(df, pd.DataFrame) or hasattr(df, "shape")

            if isinstance(df, pd.DataFrame):
                assert len(df) <= 100  # May be less due to processing

    def test_create_sample_dataset_ecommerce(self):
        """Test creating sample ecommerce dataset."""
        processor = ModernDataProcessor()

        with patch("numpy.random.seed"):
            df = processor.create_sample_dataset(n_rows=50, dataset_type="ecommerce")

            assert isinstance(df, pd.DataFrame) or hasattr(df, "shape")

    def test_create_sample_dataset_iot(self):
        """Test creating sample IoT dataset."""
        processor = ModernDataProcessor()

        with patch("numpy.random.seed"):
            df = processor.create_sample_dataset(n_rows=200, dataset_type="iot")

            assert isinstance(df, pd.DataFrame) or hasattr(df, "shape")

    def test_create_sample_dataset_financial(self):
        """Test creating sample financial dataset."""
        processor = ModernDataProcessor()

        with patch("numpy.random.seed"):
            df = processor.create_sample_dataset(n_rows=75, dataset_type="financial")

            assert isinstance(df, pd.DataFrame) or hasattr(df, "shape")

    def test_create_sample_dataset_invalid_type(self):
        """Test creating sample dataset with invalid type."""
        processor = ModernDataProcessor()

        with pytest.raises(ValueError):
            processor.create_sample_dataset(n_rows=100, dataset_type="invalid_type")

    def test_create_sample_dataset_zero_rows(self):
        """Test creating sample dataset with zero rows."""
        processor = ModernDataProcessor()

        df = processor.create_sample_dataset(n_rows=0, dataset_type="sales")

        # Should handle edge case gracefully
        assert isinstance(df, pd.DataFrame) or hasattr(df, "shape")

    @patch(
        "sandbox.integrations.modern_data_processing.ModernDataProcessor._check_polars"
    )
    def test_library_availability_caching(self, mock_check_polars):
        """Test that library availability is checked only once during initialization."""
        mock_check_polars.return_value = True

        processor = ModernDataProcessor()

        # Access the property multiple times
        _ = processor.polars_available
        _ = processor.polars_available
        _ = processor.polars_available

        # Should only be called once during initialization
        mock_check_polars.assert_called_once()

    def test_processor_methods_exist(self):
        """Test that all expected methods exist on the processor."""
        processor = ModernDataProcessor()

        expected_methods = [
            "_check_polars",
            "_check_duckdb",
            "_check_pyarrow",
            "create_sample_dataset",
        ]

        for method_name in expected_methods:
            assert hasattr(processor, method_name)
            assert callable(getattr(processor, method_name))

    def test_reproducible_random_seed(self):
        """Test that random seed makes data generation reproducible."""
        processor = ModernDataProcessor()

        # Create same dataset twice
        df1 = processor.create_sample_dataset(n_rows=10, dataset_type="sales")
        df2 = processor.create_sample_dataset(n_rows=10, dataset_type="sales")

        # Should be identical due to fixed seed
        if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
            pd.testing.assert_frame_equal(df1, df2)
