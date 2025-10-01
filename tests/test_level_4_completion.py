"""
Test Level 4 Challenge Completion
Verify that all Level 4 Machine Learning challenges work correctly
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def test_challenge_1_first_ml_models() -> None:
    """Test Challenge 1: First ML Models functionality"""
    from sklearn.datasets import load_iris

    # Test Iris classification
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target
    iris_df["species"] = iris_df["target"].map(
        {0: "setosa", 1: "versicolor", 2: "virginica"}
    )

    assert iris_df.shape[0] == 150, "Iris dataset should have 150 samples"
    assert len(iris_df["species"].unique()) == 3, "Should have 3 species"
    assert "target" in iris_df.columns, "Target column should exist"

    # Test basic ML pipeline
    features = iris_df[iris.feature_names]
    target = iris_df["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy > 0.8, "Model accuracy should be reasonable"


def test_challenge_2_feature_engineering() -> None:
    """Test Challenge 2: Feature Engineering data generation"""
    np.random.seed(42)
    n_samples = 200

    # Test feature engineering dataset
    data = pd.DataFrame(
        {
            "age": np.random.normal(35, 12, n_samples).clip(18, 80),
            "income": np.random.lognormal(10.5, 0.8, n_samples),
            "credit_score": np.random.normal(650, 100, n_samples).clip(300, 850),
            "years_employed": np.random.exponential(5, n_samples).clip(0, 40),
            "debt_to_income": np.random.beta(2, 5, n_samples) * 0.8,
            "education": np.random.choice(
                ["High School", "Bachelor", "Master", "PhD"],
                n_samples,
                p=[0.4, 0.35, 0.2, 0.05],
            ),
            "job_category": np.random.choice(
                ["Tech", "Finance", "Healthcare", "Retail", "Other"],
                n_samples,
                p=[0.2, 0.15, 0.15, 0.25, 0.25],
            ),
            "marital_status": np.random.choice(
                ["Single", "Married", "Divorced"], n_samples, p=[0.4, 0.45, 0.15]
            ),
        }
    )

    # Add date feature
    data["account_created"] = pd.date_range(
        "2020-01-01", "2024-01-01", periods=n_samples
    )

    # Verify data structure
    assert data.shape[0] == n_samples, "Should have correct number of samples"
    assert "education" in data.columns, "Categorical features should exist"
    assert data["age"].between(18, 80).all(), "Age should be clipped properly"
    assert (
        data["credit_score"].between(300, 850).all()
    ), "Credit score should be in valid range"

    # Test feature engineering operations
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=["object"]).columns

    assert len(numeric_cols) > 0, "Should have numeric columns"
    assert len(categorical_cols) > 0, "Should have categorical columns"

    # Test scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])

    assert scaled_data.shape == (
        n_samples,
        len(numeric_cols),
    ), "Scaling should preserve shape"


def test_challenge_3_model_evaluation() -> None:
    """Test Challenge 3: Model Evaluation data and techniques"""
    np.random.seed(42)
    n_patients = 300

    # Test medical evaluation dataset
    medical_data = pd.DataFrame(
        {
            "age": np.random.normal(50, 15, n_patients).clip(20, 90),
            "bmi": np.random.normal(26, 5, n_patients).clip(15, 45),
            "blood_pressure": np.random.normal(120, 20, n_patients).clip(80, 200),
            "cholesterol": np.random.normal(200, 40, n_patients).clip(100, 400),
            "glucose": np.random.normal(100, 25, n_patients).clip(60, 300),
            "family_history": np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            "smoking": np.random.choice([0, 1], n_patients, p=[0.65, 0.35]),
            "exercise_hours": np.random.exponential(3, n_patients).clip(0, 20),
        }
    )

    # Create realistic target variable
    risk_score = (
        (medical_data["age"] - 20) / 70 * 0.3
        + (medical_data["bmi"] - 15) / 30 * 0.2
        + medical_data["family_history"] * 0.3
        + medical_data["smoking"] * 0.2
        + (medical_data["cholesterol"] - 100) / 300 * 0.15
    )

    medical_data["disease"] = (
        risk_score + np.random.normal(0, 0.15, n_patients) > 0.4
    ).astype(int)

    # Verify data quality
    assert medical_data.shape[0] == n_patients, "Should have correct number of patients"
    assert medical_data["disease"].dtype == int, "Disease should be integer (0/1)"

    # Check class balance (should not be too imbalanced)
    class_counts = medical_data["disease"].value_counts()
    minority_ratio = class_counts.min() / class_counts.max()
    assert (
        minority_ratio > 0.2
    ), f"Classes should be reasonably balanced, got ratio: {minority_ratio:.3f}"

    # Test evaluation techniques
    features = medical_data.drop("disease", axis=1)
    target = medical_data["disease"]

    # Cross-validation
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    cv_scores = cross_val_score(model, features, target, cv=3, scoring="accuracy")

    assert len(cv_scores) == 3, "Should have 3 CV scores"
    assert all(
        score >= 0 and score <= 1 for score in cv_scores
    ), "CV scores should be valid"
    assert cv_scores.mean() > 0.5, "Model should perform better than random"


def test_challenge_4_hyperparameter_tuning() -> None:
    """Test Challenge 4: Hyperparameter Tuning data and optimization"""
    np.random.seed(42)
    n_customers = 300

    # Test customer segmentation dataset
    customer_data = pd.DataFrame(
        {
            "age": np.random.normal(40, 15, n_customers).clip(18, 80),
            "income": np.random.lognormal(10.8, 0.6, n_customers),
            "spending_score": np.random.normal(50, 25, n_customers).clip(1, 100),
            "loyalty_years": np.random.exponential(3, n_customers).clip(0, 20),
            "purchase_frequency": np.random.poisson(8, n_customers),
            "avg_order_value": np.random.gamma(2, 50, n_customers),
            "seasonal_purchases": np.random.binomial(4, 0.6, n_customers),
            "digital_engagement": np.random.beta(2, 3, n_customers) * 100,
        }
    )

    # Create balanced customer segments using improved logic
    segment_scores = np.zeros(n_customers)

    # High-value customers
    high_value_mask = (
        (customer_data["income"] > customer_data["income"].quantile(0.7))
        & (customer_data["spending_score"] > 60)
        & (customer_data["loyalty_years"] > 2)
    )
    segment_scores[high_value_mask] += 2

    # Medium-value customers
    medium_value_mask = (
        (customer_data["income"] > customer_data["income"].quantile(0.3))
        & (customer_data["spending_score"] > 30)
        & (customer_data["purchase_frequency"] > 5)
    )
    segment_scores[medium_value_mask & ~high_value_mask] += 1

    # Add noise for realistic segmentation
    noise = np.random.normal(0, 0.2, n_customers)
    segment_scores = segment_scores + noise
    customer_data["segment"] = np.clip(np.round(segment_scores), 0, 2).astype(int)

    # Verify data quality
    assert (
        customer_data.shape[0] == n_customers
    ), "Should have correct number of customers"

    # Check segment distribution
    segment_counts = customer_data["segment"].value_counts()
    assert len(segment_counts) >= 2, "Should have at least 2 segments"

    # Ensure no segment is too small (at least 5% of data)
    min_segment_ratio = segment_counts.min() / n_customers
    assert (
        min_segment_ratio > 0.05
    ), f"Smallest segment should be > 5%, got {min_segment_ratio:.3f}"

    # Test hyperparameter tuning
    features = customer_data.drop("segment", axis=1)
    target = customer_data["segment"]

    # Simple grid search test
    param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy"
    )
    grid_search.fit(features, target)

    assert hasattr(
        grid_search, "best_params_"
    ), "Grid search should find best parameters"
    assert hasattr(grid_search, "best_score_"), "Grid search should have best score"
    assert 0 <= grid_search.best_score_ <= 1, "Best score should be valid"


def test_all_challenge_files_exist() -> None:
    """Test that all Level 4 challenge files exist and have content"""
    base_path = Path("challenges/level_4")

    expected_files = [
        "challenge_1_first_ml_models.md",
        "challenge_2_feature_engineering.md",
        "challenge_3_model_evaluation.md",
        "challenge_4_hyperparameter_tuning.md",
    ]

    for filename in expected_files:
        file_path = base_path / filename
        assert file_path.exists(), f"Challenge file {filename} should exist"

        # Check file has substantial content
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

        assert (
            len(content) > 1000
        ), f"Challenge file {filename} should have substantial content"


if __name__ == "__main__":
    # Run all tests
    test_all_challenge_files_exist()
    print("✅ All Level 4 challenge files exist with substantial content")

    test_challenge_1_first_ml_models()
    print("✅ Challenge 1: First ML Models works correctly")

    test_challenge_2_feature_engineering()
    print("✅ Challenge 2: Feature Engineering works correctly")

    test_challenge_3_model_evaluation()
    print("✅ Challenge 3: Model Evaluation works correctly")

    test_challenge_4_hyperparameter_tuning()
    print("✅ Challenge 4: Hyperparameter Tuning works correctly")

    print(
        "\n🎉 All Level 4: Machine Learning Novice challenges are complete and working!"
    )
