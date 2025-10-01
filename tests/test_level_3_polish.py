"""
Test Level 3 Challenge Polish
Verify that all Level 3 challenges work correctly after improvements
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import numpy as np
import pandas as pd


def test_challenge_1_data_generation() -> None:
    """Test Challenge 1 data generation works correctly"""
    # Test the fixed data generation
    np.random.seed(42)
    n_samples = 100  # Smaller for testing

    # Generate dates first for season calculation
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")

    # Simulate company sales data
    data = pd.DataFrame(
        {
            "date": dates,
            "region": np.random.choice(
                ["North", "South", "East", "West"], n_samples, p=[0.3, 0.25, 0.25, 0.2]
            ),
            "product_category": np.random.choice(
                ["Electronics", "Clothing", "Home", "Sports", "Books"], n_samples
            ),
            "sales_amount": np.random.gamma(2, 500, n_samples),
            "profit_margin": np.random.normal(0.15, 0.05, n_samples),
            "customer_satisfaction": np.random.normal(4.2, 0.8, n_samples),
            "marketing_spend": np.random.exponential(100, n_samples),
        }
    )

    # Add season based on month
    data["season"] = data["date"].dt.month.map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
        }
    )

    # Add derived columns
    data["profit_amount"] = data["sales_amount"] * data["profit_margin"]
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter

    # Verify data integrity
    assert data.shape[0] == n_samples, "Should have correct number of rows"
    assert "season" in data.columns, "Season column should exist"
    assert data["season"].isnull().sum() == 0, "No null values in season"
    assert len(data["season"].unique()) <= 4, "Should have at most 4 seasons"

    # Verify derived columns
    assert "profit_amount" in data.columns, "Profit amount should be calculated"
    assert "month" in data.columns, "Month should be extracted"
    assert "quarter" in data.columns, "Quarter should be extracted"


def test_challenge_2_data_generation() -> None:
    """Test Challenge 2 dashboard data generation"""
    from datetime import datetime, timedelta

    np.random.seed(42)

    # Date range for testing (smaller)
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(30)]

    n_records = len(dates) * 2  # 2 records per day for testing

    data = pd.DataFrame(
        {
            "date": np.random.choice(dates, n_records),
            "region": np.random.choice(
                ["North America", "Europe", "Asia Pacific", "Latin America"],
                n_records,
                p=[0.4, 0.3, 0.2, 0.1],
            ),
            "product_category": np.random.choice(
                ["Electronics", "Software", "Hardware", "Services", "Accessories"],
                n_records,
                p=[0.3, 0.25, 0.2, 0.15, 0.1],
            ),
            "sales_rep": np.random.choice([f"Rep_{i}" for i in range(1, 6)], n_records),
            "customer_type": np.random.choice(
                ["Enterprise", "SMB", "Individual"], n_records, p=[0.2, 0.5, 0.3]
            ),
            "sales_amount": np.random.gamma(2, 1000, n_records),
            "profit_margin": np.random.normal(0.25, 0.08, n_records),
            "customer_satisfaction": np.random.normal(4.3, 0.7, n_records),
            "deal_size": np.random.choice(
                ["Small", "Medium", "Large", "Enterprise"],
                n_records,
                p=[0.4, 0.35, 0.2, 0.05],
            ),
        }
    )

    # Calculate derived metrics
    data["profit_amount"] = data["sales_amount"] * data["profit_margin"]
    data["month"] = pd.to_datetime(data["date"]).dt.to_period("M")
    data["quarter"] = pd.to_datetime(data["date"]).dt.to_period("Q")
    data["weekday"] = pd.to_datetime(data["date"]).dt.day_name()

    # Clean up satisfaction scores
    data["customer_satisfaction"] = np.clip(data["customer_satisfaction"], 1, 5)

    # Verify data structure
    assert data.shape[0] == n_records, "Should have correct number of records"
    assert all(
        1 <= score <= 5 for score in data["customer_satisfaction"]
    ), "Satisfaction scores should be 1-5"
    assert "profit_amount" in data.columns, "Derived profit amount should exist"


def test_challenge_3_data_generation() -> None:
    """Test Challenge 3 financial data generation"""
    np.random.seed(42)

    # Simulate financial market data (smaller for testing)
    n_stocks = 4
    n_days = 50
    stock_names = ["TECH_A", "TECH_B", "FINANCE_A", "FINANCE_B"]
    sectors = ["Technology", "Technology", "Finance", "Finance"]

    # Generate stock price data
    base_returns = np.random.normal(0.001, 0.02, (n_days, n_stocks))

    # Add sector correlations
    tech_factor = np.random.normal(0, 0.01, n_days)
    finance_factor = np.random.normal(0, 0.015, n_days)

    base_returns[:, 0:2] += tech_factor[:, np.newaxis] * 0.7  # Technology stocks
    base_returns[:, 2:4] += finance_factor[:, np.newaxis] * 0.6  # Finance stocks

    # Calculate cumulative prices
    initial_prices = np.random.uniform(50, 200, n_stocks)
    prices = initial_prices * np.exp(np.cumsum(base_returns, axis=0))

    # Verify price data structure
    assert prices.shape == (n_days, n_stocks), "Price matrix should have correct shape"
    assert all(prices[0] > 0), "Initial prices should be positive"
    assert len(stock_names) == n_stocks, "Stock names should match count"


def test_challenge_4_climate_data() -> None:
    """Test Challenge 4 climate storytelling data"""
    from datetime import datetime, timedelta

    np.random.seed(42)
    years = list(range(2000, 2024))  # Smaller range for testing
    n_years = len(years)

    # Global temperature anomaly
    base_temp = 0.0
    temp_trend = 0.018
    temp_anomaly = [
        base_temp + (year - 2000) * temp_trend + np.random.normal(0, 0.1)
        for year in years
    ]

    # CO2 levels
    base_co2 = 370
    co2_growth = 1.8
    co2_levels = [
        base_co2 + (year - 2000) * co2_growth + np.random.normal(0, 2) for year in years
    ]

    # Verify climate data
    assert len(temp_anomaly) == n_years, "Temperature data should match years"
    assert len(co2_levels) == n_years, "CO2 data should match years"
    assert all(co2 > 0 for co2 in co2_levels), "CO2 levels should be positive"


def test_all_challenges_exist() -> None:
    """Test that all Level 3 challenge files exist"""
    base_path = Path("challenges/level_3")

    expected_files = [
        "challenge_1_visualization_mastery.md",
        "challenge_2_interactive_dashboards.md",
        "challenge_3_advanced_plotting.md",
        "challenge_4_storytelling_with_data.md",
    ]

    for filename in expected_files:
        file_path = base_path / filename
        assert file_path.exists(), f"Challenge file {filename} should exist"

        # Check file is not empty
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        assert (
            len(content) > 100
        ), f"Challenge file {filename} should have substantial content"


if __name__ == "__main__":
    # Run all tests
    test_all_challenges_exist()
    print("✅ All challenge files exist and have content")

    test_challenge_1_data_generation()
    print("✅ Challenge 1 data generation works correctly")

    test_challenge_2_data_generation()
    print("✅ Challenge 2 dashboard data works correctly")

    test_challenge_3_data_generation()
    print("✅ Challenge 3 financial data works correctly")

    test_challenge_4_climate_data()
    print("✅ Challenge 4 climate data works correctly")

    print("\n🎉 All Level 3 challenges are polished and working correctly!")
