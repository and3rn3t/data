from setuptools import find_packages, setup

# Handle README.md optionally for Docker builds
try:
    with open("README.md", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = (
        "An interactive data science learning platform with game-like progression"
    )

setup(
    name="data-science-sandbox",
    version="1.0.0",
    author="Data Science Sandbox Team",
    description="An interactive data science learning platform with game-like progression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "scikit-learn>=1.3.0",
        "jupyter>=1.0.0",
        "flask>=2.3.0",
        "streamlit>=1.28.0",
        # Modern data processing (optional but recommended)
        "polars>=0.20.0; extra == 'modern'",
        "duckdb>=0.9.0; extra == 'modern'",
        "pyarrow>=14.0.0; extra == 'modern'",
        # ML operations (optional but recommended)
        "mlflow>=2.8.0; extra == 'mlops'",
        "wandb>=0.16.0; extra == 'mlops'",
        "optuna>=3.4.0; extra == 'optimization'",
        # Model explainability (optional but recommended)
        "shap>=0.44.0; extra == 'explainability'",
        "lime>=0.2.0; extra == 'explainability'",
        "yellowbrick>=1.5.0; extra == 'visualization'",
        # Data engineering (optional)
        "great-expectations>=0.18.0; extra == 'data_engineering'",
        "pandera>=0.17.0; extra == 'data_engineering'",
        "prefect>=2.14.0; extra == 'workflows'",
    ],
    extras_require={
        "modern": ["polars>=0.20.0", "duckdb>=0.9.0", "pyarrow>=14.0.0"],
        "mlops": ["mlflow>=2.8.0", "wandb>=0.16.0"],
        "optimization": ["optuna>=3.4.0", "hyperopt>=0.2.7"],
        "explainability": ["shap>=0.44.0", "lime>=0.2.0"],
        "visualization": ["yellowbrick>=1.5.0", "bokeh>=3.2.0"],
        "data_engineering": ["great-expectations>=0.18.0", "pandera>=0.17.0"],
        "workflows": ["prefect>=2.14.0"],
        "all": [
            "polars>=0.20.0",
            "duckdb>=0.9.0",
            "pyarrow>=14.0.0",
            "mlflow>=2.8.0",
            "wandb>=0.16.0",
            "optuna>=3.4.0",
            "shap>=0.44.0",
            "lime>=0.2.0",
            "yellowbrick>=1.5.0",
            "great-expectations>=0.18.0",
            "pandera>=0.17.0",
        ],
    },
)
