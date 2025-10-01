"""
Test Level 1 Challenge Fixes
Verify that all Level 1 challenges work correctly after bug fixes
"""

from pathlib import Path

import numpy as np
import pandas as pd


def test_dataset_exists() -> None:
    """Test that sample dataset exists and has correct structure"""
    dataset_path = Path("data/datasets/sample_sales.csv")
    assert dataset_path.exists(), "Sample sales dataset not found"

    df = pd.read_csv(dataset_path)

    # Check required columns exist
    required_columns = [
        "sale_id",
        "date",
        "region",
        "category",
        "sales_rep",
        "quantity",
        "unit_price",
        "customer_age",
        "customer_satisfaction",
        "sales",
    ]

    for col in required_columns:
        assert col in df.columns, f"Required column '{col}' not found in dataset"

    # Check data types are reasonable
    assert df["sales"].dtype in ["float64", "int64"], "Sales column should be numeric"
    assert len(df) > 0, "Dataset should not be empty"


def test_challenge_1_code() -> None:
    """Test that Challenge 1 code executes without errors"""
    df = pd.read_csv("data/datasets/sample_sales.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Test basic operations from Challenge 1
    assert df.shape[0] > 0, "Dataset should have rows"
    assert df.shape[1] == 10, "Dataset should have 10 columns"

    # Test sales column operations
    total_sales = df["sales"].sum()
    assert total_sales > 0, "Total sales should be positive"

    avg_sales = df["sales"].mean()
    assert avg_sales > 0, "Average sales should be positive"

    # Test missing value check
    missing_count = df.isnull().sum().sum()
    assert isinstance(
        missing_count, (int, np.integer)
    ), "Missing value count should be integer"

    # Test data quality checks
    duplicates = df.duplicated().sum()
    assert isinstance(
        duplicates, (int, np.integer)
    ), "Duplicates count should be integer"

    negative_sales = (df["sales"] < 0).sum()
    assert isinstance(
        negative_sales, (int, np.integer)
    ), "Negative sales count should be integer"


def test_challenge_4_aggregations() -> None:
    """Test that Challenge 4 aggregation operations work correctly"""
    df = pd.read_csv("data/datasets/sample_sales.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Test monthly grouping
    df["month"] = df["date"].dt.to_period("M")
    monthly_sales = df.groupby("month")["sales"].sum()
    assert len(monthly_sales) > 0, "Monthly sales should have data"

    # Test category aggregation (using 'sales' not 'amount')
    category_stats = df.groupby("category")["sales"].agg(["mean", "std", "sum"])
    assert category_stats.shape[0] > 0, "Category stats should have data"
    assert category_stats.shape[1] == 3, "Should have 3 aggregation columns"

    # Test pivot table (using 'sales' not 'amount')
    pivot = pd.pivot_table(
        df,
        values="sales",
        index="category",
        columns="month",
        aggfunc="sum",
        fill_value=0,
    )
    assert pivot.shape[0] > 0, "Pivot table should have rows"
    assert pivot.shape[1] > 0, "Pivot table should have columns"

    # Test time-based resampling (using 'sales' not 'amount')
    df_indexed = df.set_index("date")
    monthly_resample = df_indexed.resample("M")["sales"].sum()
    assert len(monthly_resample) > 0, "Resampled data should exist"


def test_challenge_3_data_types() -> None:
    """Test that Challenge 3 data type operations work"""
    df = pd.read_csv("data/datasets/sample_sales.csv")

    # Test data type conversion
    df["date"] = pd.to_datetime(df["date"])
    assert df["date"].dtype == "datetime64[ns]", "Date should be datetime type"

    # Test filtering with correct column name
    avg_sales = df["sales"].mean()
    expensive_items = df[df["sales"] > avg_sales]
    assert len(expensive_items) > 0, "Should have some items above average sales"
    assert len(expensive_items) < len(df), "Not all items should be above average"

    # Test date extraction
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_name"] = df["date"].dt.day_name()

    assert "year" in df.columns, "Year extraction should work"
    assert "month" in df.columns, "Month extraction should work"
    assert "day_name" in df.columns, "Day name extraction should work"


if __name__ == "__main__":
    # Run tests directly
    test_dataset_exists()
    print("✅ Dataset structure test passed")

    test_challenge_1_code()
    print("✅ Challenge 1 code test passed")

    test_challenge_4_aggregations()
    print("✅ Challenge 4 aggregation test passed")

    test_challenge_3_data_types()
    print("✅ Challenge 3 data types test passed")

    print("\n🎉 All Level 1 fixes verified successfully!")
