# Level 7: Modern Data Science Tools Master

## Challenge 1: Advanced Data Science Toolchain

Welcome to the cutting edge of data science! This challenge introduces you to modern, high-performance tools that are revolutionizing the field. You'll learn about faster data processing, automated experiment tracking, model explainability, and hyperparameter optimization.

### 🎯 Learning Objectives

By completing this challenge, you will:
- Master modern high-performance data processing with Polars and DuckDB
- Implement ML experiment tracking with MLflow and Weights & Biases
- Create model explanations using SHAP and LIME
- Optimize hyperparameters with Optuna and other advanced tools
- Build production-ready data science workflows

### 📚 Prerequisites

- Completed Level 5: Algorithm Architect
- Understanding of machine learning concepts
- Familiarity with pandas and scikit-learn

### 🛠️ Tools You'll Master

**Modern Data Processing:**
- 🚀 **Polars**: Lightning-fast DataFrame library (2-30x faster than pandas)
- 🗄️ **DuckDB**: High-performance analytical database
- 💾 **PyArrow**: Columnar in-memory analytics

**ML Operations:**
- 📊 **MLflow**: Experiment tracking and model management
- 📈 **Weights & Biases**: Advanced experiment tracking with visualizations
- 🎯 **Optuna**: State-of-the-art hyperparameter optimization

**Model Understanding:**
- 🔍 **SHAP**: Unified model explanations
- 💡 **LIME**: Local interpretable model explanations
- 📊 **Yellowbrick**: ML visualization library

### Instructions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import our new integration modules
from sandbox.integrations import (
    ModernDataProcessor, 
    ExperimentTracker, 
    ModelExplainer,
    HyperparameterOptimizer
)

print("🚀 MODERN DATA SCIENCE TOOLCHAIN CHALLENGE")
print("=" * 50)

# ================================
# PART 1: HIGH-PERFORMANCE DATA PROCESSING
# ================================

print("\n🚀 PART 1: Modern Data Processing")
print("-" * 40)

# Initialize the modern data processor
processor = ModernDataProcessor()

# Check available tools
print("📋 Available Tools:")
tool_status = processor.performance_comparison_demo()
for tool, status in tool_status.items():
    if "status" in status:
        print(f"  • {tool.upper()}: {status}")
    elif isinstance(status, list):
        print(f"  • {tool.upper()} Benefits:")
        for benefit in status[:3]:  # Show top 3 benefits
            print(f"    - {benefit}")

print("\n🎯 Task 1.1: Create High-Performance Datasets")

# Create different types of sample datasets
datasets = {}
dataset_types = ["sales", "ecommerce", "iot", "financial"]

for dataset_type in dataset_types:
    print(f"  Creating {dataset_type} dataset...")
    datasets[dataset_type] = processor.create_sample_dataset(
        n_rows=50000, 
        dataset_type=dataset_type
    )
    
    # Show basic info
    if hasattr(datasets[dataset_type], 'shape'):
        print(f"  ✅ {dataset_type}: {datasets[dataset_type].shape}")
    else:
        print(f"  ✅ {dataset_type}: {len(datasets[dataset_type])} rows")

print("\n🎯 Task 1.2: SQL Queries on DataFrames")

# Use DuckDB to query the sales dataset with SQL
sales_data = datasets["sales"]
print("\nSample SQL Analysis with DuckDB:")

sql_queries = [
    "SELECT region, AVG(sales_amount) as avg_sales FROM df GROUP BY region ORDER BY avg_sales DESC",
    "SELECT product_category, COUNT(*) as orders, SUM(sales_amount) as total_sales FROM df GROUP BY product_category",
    "SELECT region, product_category, AVG(profit_margin) as avg_margin FROM df GROUP BY region, product_category HAVING COUNT(*) > 100"
]

for i, query in enumerate(sql_queries, 1):
    try:
        result = processor.query_with_sql(sales_data, query)
        print(f"\n  Query {i} Results (top 5 rows):")
        if hasattr(result, 'head'):
            print(result.head().to_string())
        else:
            print(result[:5])
    except Exception as e:
        print(f"  Query {i} failed: {e}")

print("\n🎯 Task 1.3: Data Type Optimization")

# Optimize memory usage
optimized_data = processor.optimize_datatypes(sales_data)
print("✅ Data types optimized for better memory usage")

# ================================
# PART 2: ML EXPERIMENT TRACKING
# ================================

print("\n📊 PART 2: ML Experiment Tracking")
print("-" * 40)

# Initialize experiment tracker
tracker = ExperimentTracker(
    project_name="modern-data-science-challenge",
    experiment_name="random-forest-optimization"
)

# Show available tracking tools
tracking_comparison = tracker.create_comparison_demo()
print("📋 Available Experiment Tracking:")
for tool, info in tracking_comparison.items():
    if isinstance(info, dict) and "status" in info:
        print(f"  • {tool.upper()}: {info['status']}")

print("\n🎯 Task 2.1: Track Basic ML Experiment")

# Prepare data for ML
if hasattr(sales_data, 'to_pandas'):
    ml_data = sales_data.to_pandas()  # Convert from Polars if needed
else:
    ml_data = sales_data

# Create features for classification
ml_data['high_profit'] = (ml_data['profit_margin'] > ml_data['profit_margin'].median()).astype(int)

# Select features
feature_cols = ['sales_amount', 'shipping_cost']
X = ml_data[feature_cols]
y = ml_data['high_profit']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start experiment tracking
run_id = tracker.start_run(
    run_name="baseline_random_forest",
    tags={"model_type": "random_forest", "dataset": "sales"}
)

# Train baseline model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Log parameters
tracker.log_params({
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "random_state": 42
})

# Evaluate and log metrics
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)

tracker.log_metrics({
    "accuracy": accuracy,
    "cv_mean": cv_scores.mean(),
    "cv_std": cv_scores.std()
})

# Log the model
tracker.log_model(rf_model, "baseline_rf", framework="sklearn")

print(f"✅ Baseline Model - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# End the run
tracker.end_run()

# ================================
# PART 3: MODEL EXPLAINABILITY
# ================================

print("\n🔍 PART 3: Model Explainability")
print("-" * 40)

# Initialize model explainer
explainer = ModelExplainer()

# Show available tools
explainability_tools = explainer.get_tool_comparison()
print("📋 Available Explainability Tools:")
for tool, info in explainability_tools.items():
    print(f"  • {tool.upper()}: {info['status']}")

print("\n🎯 Task 3.1: Explain Model Predictions")

# Explain the trained model
explanation_results = explainer.explain_prediction(
    model=rf_model,
    X_train=X_train,
    X_explain=X_test[:10],  # Explain first 10 test instances
    method="auto",
    feature_names=feature_cols
)

print(f"✅ Explanations generated using: {explanation_results['method_used']}")
print(f"📊 Feature Importance:")
for feature, importance in explanation_results.get('feature_importance', {}).items():
    print(f"  • {feature}: {importance:.4f}")

print(f"💡 Interpretation: {explanation_results.get('explanations', {}).get('interpretation', 'N/A')}")

print("\n🎯 Task 3.2: Model Evaluation Report")

# Create comprehensive evaluation report
evaluation_report = explainer.create_model_evaluation_report(
    model=rf_model,
    X_test=X_test,
    y_test=y_test,
    task_type="classification"
)

print(f"✅ Model Evaluation Completed:")
print(f"  • Task Type: {evaluation_report['task_type']}")
print(f"  • Accuracy: {evaluation_report['accuracy']:.4f}")
print(f"  • Visualizations: {'✅' if evaluation_report.get('yellowbrick_plots') else '❌'}")

# ================================
# PART 4: HYPERPARAMETER OPTIMIZATION
# ================================

print("\n🎯 PART 4: Hyperparameter Optimization")
print("-" * 40)

# Initialize optimizer
optimizer = HyperparameterOptimizer()

# Show available tools
optimization_tools = optimizer.get_tool_comparison()
print("📋 Available Optimization Tools:")
for tool, info in optimization_tools.items():
    print(f"  • {tool.upper()}: {info['status']}")

print("\n🎯 Task 4.1: Optimize Random Forest")

# Define objective function
def rf_objective(params):
    """Objective function for Random Forest optimization."""
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=42
    )
    
    # Use cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    
    # Return negative score (since we minimize)
    return -cv_scores.mean()

# Define parameter space
param_space = {
    "n_estimators": {"type": "int", "low": 50, "high": 300},
    "max_depth": {"type": "int", "low": 3, "high": 15},
    "min_samples_split": {"type": "int", "low": 2, "high": 20}
}

print("🚀 Starting hyperparameter optimization...")

# Run optimization (reduced trials for demo)
optimization_results = optimizer.optimize_model(
    objective_function=rf_objective,
    param_space=param_space,
    n_trials=20,  # Reduced for demo speed
    method="auto",
    study_name="rf_optimization_demo"
)

print(f"✅ Optimization completed using: {optimization_results['method']}")
print(f"🎯 Best Parameters:")
for param, value in optimization_results['best_params'].items():
    print(f"  • {param}: {value}")
print(f"📈 Best CV Score: {-optimization_results['best_value']:.4f}")
print(f"🔄 Total Trials: {optimization_results['n_trials']}")

print("\n🎯 Task 4.2: Train Optimized Model")

# Start new experiment tracking for optimized model
run_id = tracker.start_run(
    run_name="optimized_random_forest",
    tags={"model_type": "random_forest", "optimized": True}
)

# Train optimized model
optimized_rf = RandomForestClassifier(**optimization_results['best_params'], random_state=42)
optimized_rf.fit(X_train, y_train)

# Evaluate optimized model
optimized_pred = optimized_rf.predict(X_test)
optimized_accuracy = accuracy_score(y_test, optimized_pred)
optimized_cv = cross_val_score(optimized_rf, X_train, y_train, cv=5)

# Log optimized results
tracker.log_params(optimization_results['best_params'])
tracker.log_metrics({
    "accuracy": optimized_accuracy,
    "cv_mean": optimized_cv.mean(),
    "cv_std": optimized_cv.std(),
    "improvement_over_baseline": optimized_accuracy - accuracy
})

tracker.log_model(optimized_rf, "optimized_rf", framework="sklearn")
tracker.end_run()

print(f"✅ Optimized Model Performance:")
print(f"  • Accuracy: {optimized_accuracy:.4f}")
print(f"  • CV Score: {optimized_cv.mean():.4f} ± {optimized_cv.std():.4f}")
print(f"  • Improvement: {optimized_accuracy - accuracy:.4f}")

# ================================
# PART 5: SUMMARY AND RECOMMENDATIONS
# ================================

print("\n📋 PART 5: Challenge Summary")
print("-" * 40)

# Get experiment summary
experiment_summary = tracker.get_experiment_summary()

print("🏆 CHALLENGE COMPLETED!")
print(f"✅ Datasets processed: {len(datasets)} different types")
print(f"✅ SQL queries executed: {len(sql_queries)}")
print(f"✅ ML experiments tracked: {experiment_summary['total_runs']}")
print(f"✅ Models explained: {explanation_results['method_used']}")
print(f"✅ Hyperparameter optimization: {optimization_results['method']}")

print("\n🎯 Key Takeaways:")
print("1. 🚀 Modern data processing tools can be 2-30x faster than pandas")
print("2. 📊 Experiment tracking is essential for reproducible ML")
print("3. 🔍 Model explainability builds trust and insight")
print("4. ⚡ Hyperparameter optimization can significantly improve performance")
print("5. 🔧 These tools integrate seamlessly with existing workflows")

print("\n📚 Tool Recommendations:")
tool_recommendations = processor.get_tool_recommendations()
for tool, recommendation in tool_recommendations.items():
    print(f"  • {tool.capitalize()}: {recommendation}")

print("\n🎓 Next Steps:")
print("1. Install and experiment with these tools in your own projects")
print("2. Set up MLflow or W&B for your next ML project")
print("3. Always explain your models, especially in production")
print("4. Use automated hyperparameter optimization for better results")
print("5. Consider Polars for large-scale data processing tasks")

print("\n🏅 CONGRATULATIONS!")
print("You've mastered the modern data science toolchain!")
print("You're now equipped with cutting-edge tools used in industry.")
```

### 🏆 Success Criteria

To complete this challenge successfully, you should:

1. **Data Processing Mastery** (25 points)
   - Successfully create datasets using ModernDataProcessor
   - Execute SQL queries on DataFrames using DuckDB
   - Optimize data types for memory efficiency

2. **Experiment Tracking** (25 points) 
   - Track ML experiments with parameters, metrics, and models
   - Compare baseline and optimized models
   - Generate experiment summaries

3. **Model Explainability** (25 points)
   - Generate model explanations using available tools
   - Create comprehensive evaluation reports
   - Interpret feature importance and model decisions

4. **Hyperparameter Optimization** (25 points)
   - Define appropriate parameter spaces
   - Run automated optimization
   - Achieve measurable performance improvements

### 💡 Bonus Challenges

**Expert Level Extensions:**
1. **Multi-objective Optimization**: Optimize for both accuracy and training time
2. **A/B Testing Framework**: Compare multiple models systematically  
3. **Production Pipeline**: Create an end-to-end ML pipeline with all tools
4. **Custom Visualizations**: Build custom dashboards for experiment tracking

### 🔧 Troubleshooting

**Common Issues:**
- **Missing Libraries**: The challenge gracefully handles missing libraries with fallbacks
- **Memory Issues**: Use smaller datasets (reduce n_rows) if memory is limited
- **Slow Optimization**: Reduce n_trials for faster completion during learning

**Installation Help:**
```bash
# Install missing libraries
pip install polars duckdb mlflow wandb shap lime yellowbrick optuna
```

### 📖 Additional Resources

**Documentation:**
- [Polars User Guide](https://pola-rs.github.io/polars/)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)

**Tutorials:**
- Modern Data Processing with Polars
- MLOps Best Practices
- Model Interpretability Guide
- Hyperparameter Optimization Strategies

---

*This challenge represents the cutting edge of data science tooling. Master these skills to build production-ready, explainable, and optimized machine learning systems!*