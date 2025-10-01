#!/usr/bin/env python3
"""
Level 7 Challenge 1: Modern Toolchain Challenge Runner

This script demonstrates the modern data science tools and techniques
covered in Level 7 Challenge 1.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

print("🚀 LEVEL 7 CHALLENGE 1: MODERN DATA SCIENCE TOOLCHAIN")
print("=" * 60)

# Test modern data processing libraries
print("\n📊 PART 1: MODERN DATA PROCESSING")
print("-" * 40)

# Test Polars (if available)
try:
    import polars as pl

    print("✅ Polars available - Lightning-fast DataFrames!")

    # Create sample data with Polars
    df_polars = pl.DataFrame(
        {
            "id": range(1000),
            "value": np.random.randn(1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )

    # Demonstrate Polars performance
    result = (
        df_polars.filter(pl.col("score") > 50)
        .group_by("category")
        .agg(
            [
                pl.col("value").mean().alias("avg_value"),
                pl.col("score").max().alias("max_score"),
                pl.count().alias("count"),
            ]
        )
        .sort("avg_value", descending=True)
    )

    print("🔥 Polars aggregation result:")
    print(result)

except ImportError:
    print("⚠️  Polars not available. Install with: pip install polars")
    # Fallback to pandas
    df_pandas = pd.DataFrame(
        {
            "id": range(1000),
            "value": np.random.randn(1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )

    result = (
        df_pandas[df_pandas["score"] > 50]
        .groupby("category")
        .agg({"value": "mean", "score": "max", "id": "count"})
        .rename(columns={"id": "count", "value": "avg_value", "score": "max_score"})
        .sort_values("avg_value", ascending=False)
    )

    print("📊 Pandas aggregation result (fallback):")
    print(result)

# Test DuckDB (if available)
try:
    import duckdb

    print("\n✅ DuckDB available - High-performance analytics!")

    conn = duckdb.connect(":memory:")

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=365, freq="D"),
            "sales": np.random.uniform(100, 1000, 365),
            "region": np.random.choice(["North", "South", "East", "West"], 365),
            "product": np.random.choice(["A", "B", "C", "D"], 365),
        }
    )

    # Register DataFrame with DuckDB
    conn.register("sales_data", sample_data)

    # Run SQL analytics
    query = """
    SELECT
        region,
        product,
        COUNT(*) as transactions,
        ROUND(AVG(sales), 2) as avg_sales,
        ROUND(SUM(sales), 2) as total_sales
    FROM sales_data
    WHERE sales > 500
    GROUP BY region, product
    ORDER BY total_sales DESC
    LIMIT 10
    """

    analytics_result = conn.execute(query).df()
    print("🗄️ DuckDB analytics result:")
    print(analytics_result)

except ImportError:
    print("⚠️  DuckDB not available. Install with: pip install duckdb")

print("\n🤖 PART 2: ML EXPERIMENT TRACKING")
print("-" * 40)

# Test MLflow (if available)
try:
    import mlflow
    import mlflow.sklearn

    print("✅ MLflow available - Experiment tracking ready!")

    # Set experiment
    mlflow.set_experiment("modern_toolchain_demo")

    with mlflow.start_run():
        # Create sample ML dataset
        X = np.random.randn(1000, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_samples", len(y_test))

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"📈 Model accuracy: {accuracy:.3f}")
        print("📊 Experiment logged to MLflow!")

except ImportError:
    print("⚠️  MLflow not available. Install with: pip install mlflow")

print("\n🔍 PART 3: MODEL EXPLAINABILITY")
print("-" * 40)

# Test SHAP (if available)
try:
    import shap

    print("✅ SHAP available - Model explanations ready!")

    # Create sample model and data for explanation
    X_sample = np.random.randn(100, 5)
    y_sample = (X_sample[:, 0] + X_sample[:, 1] - X_sample[:, 2] > 0).astype(int)

    model_sample = RandomForestClassifier(n_estimators=50, random_state=42)
    model_sample.fit(X_sample, y_sample)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model_sample)
    shap_values = explainer.shap_values(X_sample[:10])

    # Calculate feature importance
    if isinstance(shap_values, list):
        importance = np.abs(shap_values[1]).mean(0)  # For binary classification
    else:
        importance = np.abs(shap_values).mean(0)

    print("🎯 Feature importance (SHAP values):")
    try:
        for i, imp_val in enumerate(importance):
            print(f"  Feature {i}: {float(imp_val):.3f}")
    except Exception as e:
        print(f"  SHAP analysis completed (shape: {importance.shape})")

except ImportError:
    print("⚠️  SHAP not available. Install with: pip install shap")

# Test LIME (if available)
try:
    from lime import lime_tabular

    print("✅ LIME available - Local explanations ready!")

    # Create LIME explainer (using sample data from above)
    if "X_sample" in locals():
        explainer_lime = lime_tabular.LimeTabularExplainer(
            X_sample,
            mode="classification",
            feature_names=[f"feature_{i}" for i in range(X_sample.shape[1])],
        )

        # Explain a single prediction
        explanation = explainer_lime.explain_instance(
            X_sample[0], model_sample.predict_proba, num_features=X_sample.shape[1]
        )

        print("💡 LIME explanation for first sample:")
        for feature, weight in explanation.as_list():
            print(f"  {feature}: {weight:.3f}")

except ImportError:
    print("⚠️  LIME not available. Install with: pip install lime")

print("\n🎯 PART 4: HYPERPARAMETER OPTIMIZATION")
print("-" * 40)

# Test Optuna (if available)
try:
    import optuna

    print("✅ Optuna available - Advanced hyperparameter optimization!")

    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 10, 100)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

        # Create and train model
        model_opt = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )

        # Use sample data (create if not exists)
        X_opt = np.random.randn(500, 8)
        y_opt = (X_opt[:, 0] + X_opt[:, 1] - X_opt[:, 2] > 0).astype(int)

        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_opt, y_opt, test_size=0.3, random_state=42
        )

        model_opt.fit(X_train_opt, y_train_opt)
        y_pred_opt = model_opt.predict(X_val_opt)

        return accuracy_score(y_val_opt, y_pred_opt)

    # Run optimization (limited trials for demo)
    study = optuna.create_study(direction="maximize", study_name="rf_optimization")
    study.optimize(objective, n_trials=10, show_progress_bar=False)

    print(f"🏆 Best accuracy: {study.best_value:.3f}")
    print("🔧 Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

except ImportError:
    print("⚠️  Optuna not available. Install with: pip install optuna")

print("\n📊 PART 5: ADVANCED VISUALIZATION")
print("-" * 40)

try:
    import plotly.graph_objects as go
    import plotly.express as px

    print("✅ Plotly available - Interactive visualizations ready!")

    # Create sample visualization data
    viz_data = pd.DataFrame(
        {
            "x": np.random.randn(200),
            "y": np.random.randn(200),
            "category": np.random.choice(["A", "B", "C"], 200),
            "size": np.random.uniform(10, 50, 200),
        }
    )

    # Create interactive scatter plot
    fig = px.scatter(
        viz_data,
        x="x",
        y="y",
        color="category",
        size="size",
        title="Interactive Scatter Plot - Modern Visualization",
    )

    # Save as HTML (instead of showing)
    fig.write_html("modern_toolchain_viz.html")
    print("📈 Interactive visualization saved as 'modern_toolchain_viz.html'")

except ImportError:
    print("⚠️  Plotly not available. Install with: pip install plotly")

print("\n🏆 LEVEL 7 CHALLENGE 1 COMPLETED!")
print("=" * 50)

print("\n✅ MODERN TOOLCHAIN MASTERY DEMONSTRATED:")
print("  📊 High-performance data processing (Polars/DuckDB)")
print("  🤖 ML experiment tracking (MLflow)")
print("  🔍 Model explainability (SHAP/LIME)")
print("  🎯 Advanced hyperparameter optimization (Optuna)")
print("  📈 Interactive visualizations (Plotly)")

print("\n🎓 SKILLS LEARNED:")
print("  • Lightning-fast data manipulation with modern libraries")
print("  • Production-ready ML experiment tracking")
print("  • Model interpretability and transparency")
print("  • Automated hyperparameter optimization")
print("  • Interactive dashboard creation")

print("\n🚀 NEXT LEVEL UNLOCKED:")
print("  Ready for Challenge 2: Advanced MLOps!")

print(f"\n💻 Environment Status:")
libraries_status = {
    "polars": "polars" in globals(),
    "duckdb": "duckdb" in globals(),
    "mlflow": "mlflow" in globals(),
    "shap": "shap" in globals(),
    "optuna": "optuna" in globals(),
    "plotly": "go" in globals(),
}

for lib, available in libraries_status.items():
    status = "✅" if available else "⚠️ "
    print(f"  {status} {lib}")

print("\n🏅 Achievement Unlocked: Modern Toolchain Expert!")
