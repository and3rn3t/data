# 🚀 Data Science Sandbox - Modern Toolchain Setup Guide

Welcome to the enhanced Data Science Sandbox with cutting-edge tools! This guide will help you set up your environment with the latest data science technologies.

## 📋 Quick Installation Options

### Option 1: Basic Installation (Core Features Only)
```bash
git clone https://github.com/and3rn3t/data.git
cd data
pip install -e .
```

### Option 2: Modern Toolchain (Recommended)
```bash
git clone https://github.com/and3rn3t/data.git
cd data
pip install -e .[all]
```

### Option 3: Docker Setup (Everything Included)
```bash
git clone https://github.com/and3rn3t/data.git
cd data
docker-compose up -d
```

## 🛠️ Modular Installation

Install only the tools you need:

### Core Data Science (Always Required)
```bash
pip install -e .
```

### Modern Data Processing (2-30x Faster)
```bash
pip install -e .[modern]
# Includes: Polars, DuckDB, PyArrow
```

### ML Operations & Experiment Tracking  
```bash
pip install -e .[mlops]
# Includes: MLflow, Weights & Biases
```

### Hyperparameter Optimization
```bash
pip install -e .[optimization] 
# Includes: Optuna, Hyperopt
```

### Model Explainability
```bash
pip install -e .[explainability]
# Includes: SHAP, LIME
```

### Advanced Visualizations
```bash
pip install -e .[visualization]
# Includes: Yellowbrick, Bokeh
```

### Data Engineering & Pipelines
```bash
pip install -e .[data_engineering]
# Includes: Great Expectations, Pandera
```

### Workflow Orchestration
```bash
pip install -e .[workflows]
# Includes: Prefect
```

## 🎯 What's New in This Version

### 🚀 High-Performance Data Processing
- **Polars**: 2-30x faster than pandas for large datasets
- **DuckDB**: SQL queries directly on DataFrames
- **PyArrow**: Columnar data processing and Parquet support

### 📊 ML Experiment Tracking
- **MLflow**: Industry-standard experiment tracking
- **Weights & Biases**: Advanced visualizations and team collaboration

### 🔍 Model Explainability
- **SHAP**: Unified model explanations
- **LIME**: Local interpretable explanations
- **Yellowbrick**: ML visualization library

### ⚡ Automated Optimization
- **Optuna**: State-of-the-art hyperparameter optimization
- **Hyperopt**: Bayesian optimization
- **Scikit-Optimize**: Gaussian process optimization

### 🔧 Data Engineering
- **Great Expectations**: Data validation and testing
- **Pandera**: DataFrame schema validation
- **Prefect**: Modern workflow orchestration

## 🎮 New Learning Content

### Level 7: Modern Tools Master
Complete the advanced toolchain challenge to master:
- High-performance data processing
- Experiment tracking best practices  
- Model explainability techniques
- Automated hyperparameter tuning
- Production data pipelines

## 🚀 Launch Options

### Interactive Dashboard (Recommended)
```bash
python main.py --mode dashboard
```
Access at: http://localhost:8501

### Jupyter Lab Environment
```bash
python main.py --mode jupyter
```
Access at: http://localhost:8888

### Command Line Interface
```bash
python main.py --mode cli
```

### Docker Environment
```bash
docker-compose up -d
```
- Dashboard: http://localhost:8501
- Jupyter Lab: http://localhost:8889  
- MLflow UI: http://localhost:5001

## 🔧 Configuration

### Environment Variables
Create a `.env` file for custom configuration:

```env
# Application Settings
SANDBOX_MODE=development
LOG_LEVEL=INFO

# ML Experiment Tracking
MLFLOW_TRACKING_URI=file:///app/logs/mlruns
WANDB_MODE=offline

# Jupyter Configuration
JUPYTER_PORT=8888
JUPYTER_TOKEN=your_secure_token

# Dashboard Configuration  
STREAMLIT_PORT=8501
```

### Tool-Specific Setup

#### MLflow Setup
```bash
# Initialize MLflow tracking
mlflow server --host 0.0.0.0 --port 5000
```

#### Weights & Biases Setup
```bash
# Login to W&B (optional)
wandb login
```

#### Optuna Dashboard
```bash
# View optimization studies
optuna-dashboard sqlite:///optuna.db
```

## 📚 Usage Examples

### Modern Data Processing
```python
from sandbox.integrations import ModernDataProcessor

processor = ModernDataProcessor()
# Create high-performance dataset
data = processor.create_sample_dataset(n_rows=1000000, dataset_type="sales")

# SQL queries on DataFrames
result = processor.query_with_sql(data, "SELECT region, AVG(sales_amount) FROM df GROUP BY region")
```

### Experiment Tracking
```python
from sandbox.integrations import ExperimentTracker

tracker = ExperimentTracker()
tracker.start_run("my_experiment")
tracker.log_params({"learning_rate": 0.01})
tracker.log_metrics({"accuracy": 0.95})
tracker.log_model(model, "my_model")
tracker.end_run()
```

### Model Explainability
```python
from sandbox.integrations import ModelExplainer

explainer = ModelExplainer()
results = explainer.explain_prediction(model, X_train, X_test[:10])
print(f"Feature importance: {results['feature_importance']}")
```

### Hyperparameter Optimization
```python
from sandbox.integrations import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()
results = optimizer.optimize_model(
    objective_function=my_objective,
    param_space={"n_estimators": {"type": "int", "low": 50, "high": 500}},
    n_trials=100
)
```

## 🛟 Troubleshooting

### Common Issues

#### Missing Libraries
If you see import errors:
```bash
# Install missing dependencies
pip install polars duckdb mlflow wandb shap lime optuna
```

#### Memory Issues
For large datasets:
```python
# Use smaller samples during development
data = processor.create_sample_dataset(n_rows=10000)  # Instead of 1M
```

#### Docker Issues
```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Performance Tips

1. **Use Polars for Large Datasets**: 2-30x faster than pandas
2. **Enable DuckDB for SQL**: Faster analytical queries
3. **Optimize Data Types**: Use `processor.optimize_datatypes()`
4. **Cache Expensive Operations**: Save intermediate results
5. **Use Parallel Processing**: Most tools support multiprocessing

## 📖 Additional Resources

### Documentation
- [Modern Data Processing Guide](docs/modern_data_processing.md)
- [MLOps Best Practices](docs/mlops_guide.md)  
- [Model Explainability Tutorial](docs/explainability_guide.md)
- [Hyperparameter Optimization Strategies](docs/optimization_guide.md)

### External Resources
- [Polars User Guide](https://pola-rs.github.io/polars/)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)

## 🤝 Contributing

We welcome contributions! Areas of interest:
- New tool integrations
- Additional challenges and tutorials  
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

🎉 **Ready to explore the future of data science?** Start with the Level 7 challenge to master the modern toolchain!