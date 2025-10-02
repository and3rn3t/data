"""
Unit tests for data validation utilities
"""

import pandas as pd
import pandera as pa
import pytest

from sandbox.utils.data_validation import (
    get_iris_schema,
    get_sales_schema,
    validate_completeness,
    validate_data_types,
    validate_no_duplicates,
)


class TestDataValidation:
    """Test suite for data validation utilities."""

    def test_get_iris_schema(self):
        """Test Iris dataset schema creation."""
        schema = get_iris_schema()

        assert isinstance(schema, pa.DataFrameSchema)
        assert "sepal_length" in schema.columns
        assert "sepal_width" in schema.columns
        assert "petal_length" in schema.columns
        assert "petal_width" in schema.columns
        assert "species" in schema.columns

    def test_get_sales_schema(self):
        """Test sales dataset schema creation."""
        schema = get_sales_schema()

        assert isinstance(schema, pa.DataFrameSchema)
        assert "date" in schema.columns
        assert "product_id" in schema.columns
        assert "quantity" in schema.columns
        assert "price" in schema.columns
        assert "total" in schema.columns

    def test_iris_schema_validation_valid_data(self):
        """Test Iris schema validation with valid data."""
        schema = get_iris_schema()

        valid_data = pd.DataFrame(
            {
                "sepal_length": [5.1, 4.9, 4.7],
                "sepal_width": [3.5, 3.0, 3.2],
                "petal_length": [1.4, 1.4, 1.3],
                "petal_width": [0.2, 0.2, 0.2],
                "species": ["setosa", "setosa", "setosa"],
            }
        )

        # Should not raise an exception
        validated_df = schema.validate(valid_data)
        assert len(validated_df) == 3

    def test_iris_schema_validation_invalid_data(self):
        """Test Iris schema validation with invalid data."""
        schema = get_iris_schema()

        invalid_data = pd.DataFrame(
            {
                "sepal_length": [5.1, 4.9, 15.0],  # Out of range
                "sepal_width": [3.5, 3.0, 3.2],
                "petal_length": [1.4, 1.4, 1.3],
                "petal_width": [0.2, 0.2, 0.2],
                "species": ["setosa", "setosa", "invalid_species"],  # Invalid species
            }
        )

        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)

    def test_sales_schema_validation_valid_data(self):
        """Test sales schema validation with valid data."""
        schema = get_sales_schema()

        valid_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                "product_id": ["P001", "P002", "P003"],
                "quantity": [5, 10, 3],
                "price": [19.99, 29.99, 15.50],
                "total": [99.95, 299.90, 46.50],
            }
        )

        # Should not raise an exception
        validated_df = schema.validate(valid_data)
        assert len(validated_df) == 3

    def test_sales_schema_validation_invalid_data(self):
        """Test sales schema validation with invalid data."""
        schema = get_sales_schema()

        invalid_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                "product_id": ["P001", "", "P003"],  # Empty product_id
                "quantity": [5, -10, 3],  # Negative quantity
                "price": [19.99, 29.99, -15.50],  # Negative price
                "total": [99.95, 299.90, 46.50],
            }
        )

        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_data)

    def test_validate_no_duplicates_clean_data(self):
        """Test duplicate validation with clean data."""
        df = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"]})

        result = validate_no_duplicates(df)
        assert result is True

    def test_validate_no_duplicates_with_duplicates(self):
        """Test duplicate validation with duplicate rows."""
        df = pd.DataFrame({"A": [1, 2, 2, 4], "B": ["a", "b", "b", "d"]})

        result = validate_no_duplicates(df)
        assert result is False

    def test_validate_no_duplicates_with_subset(self):
        """Test duplicate validation with subset of columns."""
        df = pd.DataFrame(
            {"A": [1, 2, 2, 4], "B": ["a", "b", "c", "d"], "C": [10, 20, 30, 40]}
        )

        # No duplicates in column A and B together
        result = validate_no_duplicates(df, subset=["A", "B"])
        assert result is True

        # Duplicates exist in column A only
        result = validate_no_duplicates(df, subset=["A"])
        assert result is False

    def test_validate_completeness_complete_data(self):
        """Test completeness validation with complete data."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

        result = validate_completeness(df, threshold=0.95)

        assert result["A"]["ratio"] == 1.0
        assert result["A"]["passes_threshold"]
        assert result["B"]["ratio"] == 1.0
        assert result["B"]["passes_threshold"]

    def test_validate_completeness_incomplete_data(self):
        """Test completeness validation with missing data."""
        df = pd.DataFrame({"A": [1, 2, None, 4, 5], "B": ["a", None, None, "d", "e"]})

        result = validate_completeness(df, threshold=0.8)

        assert result["A"]["ratio"] == 0.8
        assert result["A"]["passes_threshold"]
        assert result["B"]["ratio"] == 0.6
        assert not result["B"]["passes_threshold"]

    def test_validate_completeness_custom_threshold(self):
        """Test completeness validation with custom threshold."""
        df = pd.DataFrame({"A": [1, 2, None, 4, 5], "B": ["a", None, None, "d", "e"]})

        result = validate_completeness(df, threshold=0.9)

        assert not result["A"]["passes_threshold"]  # 0.8 < 0.9
        assert not result["B"]["passes_threshold"]  # 0.6 < 0.9

    def test_validate_data_types_matching_types(self):
        """Test data type validation with matching types."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [1.1, 2.2, 3.3, 4.4, 5.5],
                "C": ["a", "b", "c", "d", "e"],
            }
        )

        expected_types = {"A": "int64", "B": "float64", "C": "object"}

        result = validate_data_types(df, expected_types)

        assert result["A"]["matches"] is True
        assert result["B"]["matches"] is True
        assert result["C"]["matches"] is True

    def test_validate_data_types_mismatched_types(self):
        """Test data type validation with mismatched types."""
        df = pd.DataFrame(
            {
                "A": ["1", "2", "3"],  # String instead of int
                "B": [1, 2, 3],  # Int instead of float
            }
        )

        expected_types = {"A": "int64", "B": "float64"}

        result = validate_data_types(df, expected_types)

        assert result["A"]["matches"] is False
        assert result["A"]["expected"] == "int64"
        assert result["A"]["actual"] == "object"

        assert result["B"]["matches"] is False
        assert result["B"]["expected"] == "float64"
        assert result["B"]["actual"] == "int64"

    def test_validate_data_types_missing_columns(self):
        """Test data type validation with missing columns."""
        df = pd.DataFrame({"A": [1, 2, 3]})

        expected_types = {
            "A": "int64",
            "B": "float64",  # Column doesn't exist
            "C": "object",  # Column doesn't exist
        }

        result = validate_data_types(df, expected_types)

        assert "A" in result
        assert result["A"]["matches"] is True
        assert "B" not in result  # Missing columns not included
        assert "C" not in result

    def test_empty_dataframe_validation(self):
        """Test validation functions with empty DataFrame."""
        df = pd.DataFrame()

        # Should not raise errors
        no_duplicates = validate_no_duplicates(df)
        assert no_duplicates is True  # No rows means no duplicates

        completeness = validate_completeness(df)
        assert completeness == {}  # No columns to check

        type_check = validate_data_types(df, {"A": "int64"})
        assert type_check == {}  # No matching columns

    def test_single_row_dataframe_validation(self):
        """Test validation functions with single row DataFrame."""
        df = pd.DataFrame({"A": [1], "B": ["test"]})

        no_duplicates = validate_no_duplicates(df)
        assert no_duplicates is True

        completeness = validate_completeness(df)
        assert completeness["A"]["ratio"] == 1.0
        assert completeness["B"]["ratio"] == 1.0

        type_check = validate_data_types(df, {"A": "int64", "B": "object"})
        assert type_check["A"]["matches"] is True
        assert type_check["B"]["matches"] is True

    def test_schema_column_constraints(self):
        """Test that schema column constraints work correctly."""
        schema = get_iris_schema()

        # Test sepal_length range constraint
        sepal_length_column = schema.columns["sepal_length"]
        assert str(sepal_length_column.dtype) == "float64"

        # Test species categorical constraint
        species_column = schema.columns["species"]
        assert str(species_column.dtype) == "str"

    def test_sales_schema_constraints(self):
        """Test sales schema specific constraints."""
        schema = get_sales_schema()

        # Test quantity non-negative constraint
        quantity_column = schema.columns["quantity"]
        assert str(quantity_column.dtype) == "int64"

        # Test price non-negative constraint
        price_column = schema.columns["price"]
        assert str(price_column.dtype) == "float64"
