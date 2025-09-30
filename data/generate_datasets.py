"""
Generate sample datasets for the Data Science Sandbox
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def create_sample_sales_data() -> pd.DataFrame:
    """Create a sample sales dataset for learning exercises"""
    # Use modern NumPy random generator
    rng = np.random.default_rng(42)

    # Configuration
    n_records = 1000
    regions = ["North", "South", "East", "West", "Central"]
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
    sales_reps = [f"Rep_{i:02d}" for i in range(1, 21)]

    # Generate data
    # Generate random days first to avoid numpy int64 issues
    random_days = [int(day) for day in rng.integers(0, 365, n_records)]

    data = {
        "sale_id": range(1, n_records + 1),
        "date": [datetime.now() - timedelta(days=days) for days in random_days],
        "region": rng.choice(regions, n_records),
        "category": rng.choice(categories, n_records),
        "sales_rep": rng.choice(sales_reps, n_records),
        "quantity": rng.poisson(5, n_records) + 1,
        "unit_price": rng.uniform(10, 500, n_records).round(2),
        "customer_age": rng.normal(40, 15, n_records).astype(int).clip(18, 80),
        "customer_satisfaction": rng.choice(
            [1, 2, 3, 4, 5], n_records, p=[0.05, 0.1, 0.2, 0.4, 0.25]
        ),
    }

    # Calculate sales amount (convert to arrays for multiplication)
    data["sales"] = (np.array(data["quantity"]) * np.array(data["unit_price"])).round(2)

    # Create DataFrame first
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # Add some missing values for learning purposes
    missing_indices = rng.choice(n_records, size=int(n_records * 0.05), replace=False)
    for idx in missing_indices[: len(missing_indices) // 2]:
        df.loc[idx, "customer_satisfaction"] = np.nan
    for idx in missing_indices[len(missing_indices) // 2 :]:
        df.loc[idx, "customer_age"] = np.nan

    return df


def create_sample_datasets() -> None:
    """Create all sample datasets"""
    # Get the datasets directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, "datasets")

    # Create directory if it doesn't exist
    os.makedirs(datasets_dir, exist_ok=True)

    # Sales dataset
    sales_df = create_sample_sales_data()
    sales_df.to_csv(os.path.join(datasets_dir, "sample_sales.csv"), index=False)

    # Simple dataset for absolute beginners
    simple_df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "score": [85, 92, 78, 94, 87],
            "city": ["New York", "London", "Paris", "Tokyo", "Sydney"],
        }
    )
    simple_df.to_csv(os.path.join(datasets_dir, "simple_data.csv"), index=False)

    # Iris dataset (classic ML dataset)
    from sklearn.datasets import load_iris

    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target_names[iris.target]
    iris_df.to_csv(os.path.join(datasets_dir, "iris.csv"), index=False)

    print("✅ Sample datasets created successfully!")
    print(f"📁 Sales dataset: {sales_df.shape[0]} records, {sales_df.shape[1]} columns")
    print(
        f"📁 Simple dataset: {simple_df.shape[0]} records, {simple_df.shape[1]} columns"
    )
    print(f"📁 Iris dataset: {iris_df.shape[0]} records, {iris_df.shape[1]} columns")


if __name__ == "__main__":
    create_sample_datasets()
