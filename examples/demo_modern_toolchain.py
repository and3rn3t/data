#!/usr/bin/env python3
"""
Modern Data Science Toolchain Demo

This script demonstrates the complete modern data science workflow
using all the newly integrated tools in the Data Science Sandbox.

Run with: python demo_modern_toolchain.py
"""

import warnings

warnings.filterwarnings("ignore")

print("🚀 MODERN DATA SCIENCE TOOLCHAIN DEMONSTRATION")
print("=" * 60)

try:
    # Import our integration modules
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score, train_test_split

    from sandbox.integrations import (
        DataPipelineBuilder,
        ExperimentTracker,
        HyperparameterOptimizer,
        ModelExplainer,
        ModernDataProcessor,
    )

    print("✅ All modules imported successfully!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Run: pip install -e .[all] to install all dependencies")
    exit(1)


def demo_data_processing():
    """Demonstrate high-performance data processing."""
    print("\n📊 STEP 1: High-Performance Data Processing")
    print("-" * 50)

    # Initialize modern data processor
    processor = ModernDataProcessor()

    # Show available tools
    print("🔍 Checking available tools:")
    status = processor.performance_comparison_demo()

    for tool, info in status.items():
        if isinstance(info, str):
            print(f"  • {tool.replace('_', ' ').title()}: {info}")

    return processor


def demo_ml_tracking():
    """Demonstrate ML experiment tracking."""
    print("\n🧪 STEP 2: ML Experiment Tracking")
    print("-" * 50)

    # Initialize experiment tracker
    tracker = ExperimentTracker()

    # Show tracking capabilities
    print("📋 Experiment tracking capabilities:")
    tracking_info = tracker.get_tracking_info()

    for capability, status in tracking_info.items():
        print(f"  • {capability.replace('_', ' ').title()}: {status}")

    return tracker


def demo_baseline_model(tracker, X_train, X_test, y_train, y_test):
    """Train and evaluate baseline model."""
    print("\n🌲 Training baseline Random Forest...")

    # Start experiment run
    tracker.start_run(
        run_name="baseline_random_forest",
        tags={"model_type": "random_forest", "dataset": "sales", "demo": True},
    )

    # Train baseline model
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, min_samples_leaf=1, max_features="sqrt"
    )
    rf_model.fit(X_train, y_train)

    # Evaluate model
    pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)

    return rf_model, accuracy, cv_scores


def demo_hyperparameter_optimization(X_train, y_train):
    """Demonstrate hyperparameter optimization."""
    print("\n🎯 STEP 3: Hyperparameter Optimization")
    print("-" * 50)

    optimizer = HyperparameterOptimizer()
    opt_tools = optimizer.get_tool_comparison()
    print("🎯 Optimization tools:")
    for tool, info in opt_tools.items():
        print(f"  • {tool.upper()}: {info['status']}")

    # Define optimization objective
    def rf_objective(params):
        """Objective function for Random Forest optimization."""
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"] if params["max_depth"] > 0 else None,
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
        )

        # Use cross-validation for robust estimation
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        return -cv_scores.mean()  # Negative because we minimize

    # Define parameter space
    param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 200},
        "max_depth": {"type": "int", "low": 1, "high": 20},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
    }

    # Run optimization
    print("⚡ Running hyperparameter optimization...")
    optimization_results = optimizer.optimize(rf_objective, param_space, n_trials=25)

    return optimization_results


def main():
    """Main demonstration workflow."""
    # Demo data processing
    processor = demo_data_processing()

    # Create sample dataset using the fastest available method
    print("\n🏭 Creating sample dataset (50K records)...")
    sales_data = processor.create_sample_dataset(n_rows=50000, dataset_type="sales")

    print(
        f"✅ Dataset created: {type(sales_data)} with {getattr(sales_data, 'shape', 'unknown shape')}"
    )

    # Demonstrate SQL queries on DataFrame
    print("\n🗃️ SQL Analysis on DataFrame:")
    sql_query = "SELECT region, AVG(sales_amount) as avg_sales FROM df GROUP BY region ORDER BY avg_sales DESC"

    try:
        result = processor.query_with_sql(sales_data, sql_query)
        print("  Top regions by average sales:")
        if hasattr(result, "head"):
            print(result.to_string(index=False))
        else:
            print(result)
    except Exception as e:
        print(f"  SQL query failed (fallback mode): {e}")

    # ============================================================================
    # STEP 2: Data Quality & Pipeline Engineering
    # ============================================================================

    print("\n🔧 STEP 2: Data Engineering & Quality")
    print("-" * 50)

    # Initialize data pipeline builder
    pipeline_builder = DataPipelineBuilder()

    # Convert to pandas for downstream processing
    if hasattr(sales_data, "to_pandas"):
        ml_data = sales_data.to_pandas()
    else:
        ml_data = sales_data.copy()

    # Generate data quality report
    print("📋 Generating data quality report...")
    quality_report = pipeline_builder.create_data_quality_report(ml_data)

    print(f"  • Data Quality Score: {quality_report['data_quality_score']:.1f}/100")
    print(
        f"  • Missing Data: {quality_report['dataset_overview']['missing_percentage']:.1f}%"
    )
    print(f"  • Duplicate Rows: {quality_report['dataset_overview']['duplicate_rows']}")

    # Create validation suite
    print("\n🛡️ Creating data validation suite...")
    validation_suite = pipeline_builder.create_data_validation_suite(ml_data)
    print(f"  • Method: {validation_suite['method']}")
    print(f"  • Expectations: {len(validation_suite['expectations'])}")

    # Build data pipeline
    print("\n⚙️ Building data processing pipeline...")
    pipeline_result = pipeline_builder.create_data_pipeline(
        pipeline_name="sales_data_preprocessing",
        data_source=ml_data,
        transformations=None,  # Use default transformations
    )

    print(
        f"  • Pipeline Status: {'✅ Success' if pipeline_result['success'] else '❌ Failed'}"
    )
    print(f"  • Original Shape: {pipeline_result['original_shape']}")
    print(f"  • Final Shape: {pipeline_result['final_shape']}")
    print(f"  • Duration: {pipeline_result.get('duration_seconds', 0):.2f} seconds")

    # Use processed data
    processed_data = pipeline_result["processed_data"]

    # ============================================================================
    # STEP 3: ML Experiment Tracking Setup
    # ============================================================================

    print("\n📊 STEP 3: ML Experiment Tracking")
    print("-" * 50)

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        project_name="modern-toolchain-demo", experiment_name="sales-prediction"
    )

    # Show tracking capabilities
    tracking_info = tracker.create_comparison_demo()
    print("🔍 Experiment tracking capabilities:")
    for tool, info in tracking_info.items():
        if isinstance(info, dict) and "status" in info:
            print(f"  • {tool.upper()}: {info['status']}")

    # Prepare ML dataset
    print("\n🎯 Preparing machine learning dataset...")

    # Create target variable (high profit prediction)
    processed_data["high_profit"] = (
        processed_data["profit_margin"] > processed_data["profit_margin"].median()
    ).astype(int)

    # Select features
    feature_cols = ["sales_amount", "shipping_cost"]
    X = processed_data[feature_cols]
    y = processed_data["high_profit"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  • Training set: {X_train.shape}")
    print(f"  • Test set: {X_test.shape}")

    # ============================================================================
    # STEP 4: Model Training with Experiment Tracking
    # ============================================================================

    print("\n🤖 STEP 4: Model Training & Tracking")
    print("-" * 50)

    # Start experiment run
    tracker.start_run(
        run_name="baseline_random_forest",
        tags={"model_type": "random_forest", "dataset": "sales", "demo": True},
    )

    # Train baseline model
    print("🌲 Training baseline Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, min_samples_leaf=1, max_features="sqrt"
    )
    rf_model.fit(X_train, y_train)

    # Log parameters
    tracker.log_params(
        {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
            "model_type": "RandomForest",
        }
    )

    # Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)

    # Log metrics
    tracker.log_metrics(
        {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
    )

    # Log model
    tracker.log_model(rf_model, "baseline_rf", framework="sklearn")

    print(f"  • Baseline Accuracy: {accuracy:.4f}")
    print(f"  • CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # End experiment run
    tracker.end_run()

    # ============================================================================
    # STEP 5: Model Explainability
    # ============================================================================

    print("\n🔍 STEP 5: Model Explainability")
    print("-" * 50)

    # Initialize model explainer
    explainer = ModelExplainer()

    # Show explainability tools
    explainer_tools = explainer.get_tool_comparison()
    print("🧠 Model explainability tools:")
    for tool, info in explainer_tools.items():
        print(f"  • {tool.upper()}: {info['status']}")

    # Generate model explanations
    print("\n🎯 Generating model explanations...")
    explanation_results = explainer.explain_prediction(
        model=rf_model,
        X_train=X_train,
        X_explain=X_test[:10],  # Explain first 10 predictions
        method="auto",
        feature_names=feature_cols,
    )

    print(f"  • Method used: {explanation_results['method_used']}")
    print("  • Feature Importance:")
    for feature, importance in explanation_results.get(
        "feature_importance", {}
    ).items():
        print(f"    - {feature}: {importance:.4f}")

    # Create evaluation report
    print("\n📊 Creating comprehensive evaluation report...")
    evaluation_report = explainer.create_model_evaluation_report(
        model=rf_model, X_test=X_test, y_test=y_test, task_type="classification"
    )

    print(f"  • Evaluation completed with {evaluation_report['task_type']} metrics")
    print(
        f"  • Visualizations: {'✅' if evaluation_report.get('yellowbrick_plots') else '📊 Basic'}"
    )

    # ============================================================================
    # STEP 6: Hyperparameter Optimization
    # ============================================================================

    print("\n⚡ STEP 6: Hyperparameter Optimization")
    print("-" * 50)

    # Initialize optimizer
    optimizer = HyperparameterOptimizer()

    # Show optimization tools
    opt_tools = optimizer.get_tool_comparison()
    print("🎯 Optimization tools:")
    for tool, info in opt_tools.items():
        print(f"  • {tool.upper()}: {info['status']}")

    # Define optimization objective
    def rf_objective(params):
        """Objective function for Random Forest optimization."""
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"] if params["max_depth"] > 0 else None,
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
        )

        # Use cross-validation for robust estimation
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        return -cv_scores.mean()  # Negative because we minimize

    # Define parameter space
    param_space = {
        "n_estimators": {"type": "int", "low": 50, "high": 200},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "min_samples_split": {"type": "int", "low": 2, "high": 10},
    }

    print("\n🚀 Running hyperparameter optimization...")
    print("   (This may take a minute...)")

    # Run optimization (limited trials for demo)
    optimization_results = optimizer.optimize_model(
        objective_function=rf_objective,
        param_space=param_space,
        n_trials=15,  # Reduced for demo speed
        method="auto",
        study_name="rf_optimization_demo",
    )

    print(f"  • Optimization method: {optimization_results['method']}")
    print("  • Best parameters found:")
    for param, value in optimization_results["best_params"].items():
        print(f"    - {param}: {value}")
    print(f"  • Best CV score: {-optimization_results['best_value']:.4f}")
    print(f"  • Total trials: {optimization_results['n_trials']}")

    # ============================================================================
    # STEP 7: Final Model with Optimized Parameters
    # ============================================================================

    print("\n🏆 STEP 7: Final Optimized Model")
    print("-" * 50)

    # Start new experiment for optimized model
    tracker.start_run(
        run_name="optimized_random_forest",
        tags={"model_type": "random_forest", "optimized": True, "demo": True},
    )

    # Train optimized model
    print("🌟 Training optimized model...")
    optimized_params = optimization_results["best_params"].copy()
    if (
        optimized_params.get("max_depth") is not None
        and optimized_params["max_depth"] <= 0
    ):
        optimized_params["max_depth"] = None

    # Add required hyperparameters if not present
    if "min_samples_leaf" not in optimized_params:
        optimized_params["min_samples_leaf"] = 1
    if "max_features" not in optimized_params:
        optimized_params["max_features"] = "sqrt"

    optimized_rf = RandomForestClassifier(**optimized_params, random_state=42)
    optimized_rf.fit(X_train, y_train)

    # Evaluate optimized model
    optimized_pred = optimized_rf.predict(X_test)
    optimized_accuracy = accuracy_score(y_test, optimized_pred)
    optimized_cv = cross_val_score(optimized_rf, X_train, y_train, cv=5)

    improvement = optimized_accuracy - accuracy

    # Log optimized results
    tracker.log_params(optimized_params)
    tracker.log_metrics(
        {
            "accuracy": optimized_accuracy,
            "cv_mean": optimized_cv.mean(),
            "cv_std": optimized_cv.std(),
            "improvement_over_baseline": improvement,
            "optimization_trials": optimization_results["n_trials"],
        }
    )

    # Log optimized model
    tracker.log_model(optimized_rf, "optimized_rf", framework="sklearn")

    print(f"  • Optimized Accuracy: {optimized_accuracy:.4f}")
    print(f"  • CV Score: {optimized_cv.mean():.4f} ± {optimized_cv.std():.4f}")
    print(f"  • Improvement: {improvement:.4f} ({improvement/accuracy*100:+.1f}%)")

    tracker.end_run()

    # ============================================================================
    # STEP 8: Summary & Recommendations
    # ============================================================================

    print("\n📋 STEP 8: Demo Summary & Recommendations")
    print("-" * 50)

    # Get experiment summary
    experiment_summary = tracker.get_experiment_summary()

    print("🎉 MODERN DATA SCIENCE TOOLCHAIN DEMO COMPLETED!")
    print(
        f"✅ Processed dataset: {pipeline_result['final_shape'][0]:,} rows, {pipeline_result['final_shape'][1]} columns"
    )
    print(f"✅ Data quality score: {quality_report['data_quality_score']:.1f}/100")
    print(f"✅ ML experiments tracked: {experiment_summary['total_runs']}")
    print(f"✅ Model explanations: {explanation_results['method_used']}")
    print(
        f"✅ Hyperparameter optimization: {optimization_results['method']} with {optimization_results['n_trials']} trials"
    )
    print(f"✅ Performance improvement: {improvement/accuracy*100:+.1f}%")

    print("\n🎯 Key Takeaways:")
    print("1. 🚀 Modern data processing tools significantly improve performance")
    print("2. 🔧 Data pipelines ensure reproducible and high-quality workflows")
    print("3. 📊 Experiment tracking is essential for ML project management")
    print("4. 🔍 Model explainability builds trust and provides insights")
    print("5. ⚡ Automated optimization can improve model performance")
    print("6. 🏭 These tools work seamlessly together in production environments")

    print("\n📚 Next Steps:")
    print(
        "• Try the Level 7 challenge: challenges/level_7/challenge_1_modern_toolchain.md"
    )
    print("• Explore individual tools with larger datasets")
    print("• Set up MLflow UI for experiment visualization: mlflow ui")
    print("• Experiment with different optimization algorithms")
    print("• Build your own data science projects using these tools")

    print("\n🏅 Congratulations! You've experienced the modern data science toolchain.")
    print("You're now ready to build production-ready ML systems! 🚀")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -e .[all]")
        import traceback

        traceback.print_exc()
