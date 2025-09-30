# Data Quality and Validation Configuration
# This file defines data validation rules and quality checks

from typing import Optional

import pandas as pd
import pandera as pa


# Data schema definitions for common datasets
def get_iris_schema():
    """Get schema for Iris dataset validation"""
    return pa.DataFrameSchema(
        {
            "sepal_length": pa.Column(float, pa.Check.between(0, 10)),
            "sepal_width": pa.Column(float, pa.Check.between(0, 10)),
            "petal_length": pa.Column(float, pa.Check.between(0, 10)),
            "petal_width": pa.Column(float, pa.Check.between(0, 10)),
            "species": pa.Column(
                str, pa.Check.isin(["setosa", "versicolor", "virginica"])
            ),
        }
    )


def get_sales_schema():
    """Get schema for sales dataset validation"""
    return pa.DataFrameSchema(
        {
            "date": pa.Column("datetime64[ns]"),
            "product_id": pa.Column(str, pa.Check.str_length(1, None)),
            "quantity": pa.Column(int, pa.Check.ge(0)),
            "price": pa.Column(float, pa.Check.ge(0)),
            "total": pa.Column(float, pa.Check.ge(0)),
        }
    )


# Data quality check functions
def validate_no_duplicates(df: pd.DataFrame, subset: Optional[list] = None) -> bool:
    """Check for duplicate rows in the dataset"""
    return not df.duplicated(subset=subset).any()


def validate_completeness(df: pd.DataFrame, threshold: float = 0.95) -> dict:
    """Check data completeness for each column"""
    completeness = {}
    for col in df.columns:
        complete_ratio = df[col].notna().sum() / len(df)
        completeness[col] = {
            "ratio": complete_ratio,
            "passes_threshold": complete_ratio >= threshold,
        }
    return completeness


def validate_data_types(df: pd.DataFrame, expected_types: dict) -> dict:
    """Validate that columns have expected data types"""
    type_check = {}
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = df[col].dtype
            type_check[col] = {
                "expected": expected_type,
                "actual": str(actual_type),
                "matches": str(actual_type) == expected_type
                or actual_type == expected_type,
            }
    return type_check
