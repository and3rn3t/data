# Complete Setup Guide

This guide will help you set up the Data Science Sandbox development environment from scratch.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start (Recommended)](#quick-start-recommended)
- [Manual Setup](#manual-setup)
- [Development Environment](#development-environment)
- [Docker Setup](#docker-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.8 or higher (3.10+ recommended)
- **Memory**: Minimum 8GB RAM (16GB recommended for large datasets)
- **Storage**: 5GB free space
- **Git**: For version control

### Required Software

1. **Python Installation**
   ```bash
   # Check Python version
   python --version
   ```
   
   If Python is not installed:
   - **Windows**: Download from [python.org](https://python.org) (NOT Windows Store)
   - **macOS**: Use Homebrew: `brew install python@3.11`
   - **Linux**: `sudo apt install python3.11 python3.11-pip python3.11-venv`

2. **Git Installation**
   ```bash
   # Verify Git installation
   git --version
   ```

3. **VS Code (Recommended)**
   - Download from [code.visualstudio.com](https://code.visualstudio.com)
   - Install recommended extensions (automatic prompt when opening project)

## Quick Start (Recommended)

### 1. Clone and Navigate

```bash
# Clone the repository
git clone https://github.com/and3rn3t/data.git
cd data
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development tools
pip install -r requirements-dev.txt
```

### 4. Initialize Modern Tools

```bash
# Generate sample data
python data/generate_datasets.py

# Initialize modern data science tools
python -c "
from sandbox.utils.modern_tools_config import initialize_all_tools
from sandbox.utils.logging_config import setup_logging
tools = initialize_all_tools()
setup_logging()
print('✅ Setup complete!')
"
```

### 5. Verify Installation

```bash
# Test the application
python main.py --help

# Run test suite
python -m pytest tests/ -v

# Check code quality
python -m ruff check sandbox/ --show-fixes
```

### 6. Launch Dashboard

```bash
# Start the interactive dashboard
python main.py --mode dashboard
```

Visit `http://localhost:8501` in your browser.

## Manual Setup

### Step-by-Step Installation

#### 1. Environment Preparation

```bash
# Create project directory
mkdir data-science-sandbox
cd data-science-sandbox

# Initialize git repository
git init
git remote add origin https://github.com/and3rn3t/data.git
git pull origin main
```

#### 2. Python Environment

```bash
# Create isolated Python environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

#### 3. Core Dependencies Installation

```bash
# Essential data science libraries
pip install pandas>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0
pip install seaborn>=0.12.0 plotly>=5.17.0 scikit-learn>=1.3.0

# Modern high-performance tools
pip install duckdb>=0.9.0 polars>=0.20.0 pyarrow>=14.0.0

# ML operations and experiment tracking  
pip install mlflow>=2.8.0 wandb>=0.16.0

# Model explainability
pip install shap>=0.44.0 lime>=0.2.0

# Data validation and quality
pip install pandera>=0.17.0 great-expectations>=0.18.0

# Interactive interfaces
pip install streamlit>=1.28.0 jupyterlab>=4.0.0
```

#### 4. Development Tools

```bash
# Code quality and formatting
pip install black>=23.0.0 isort>=5.12.0 ruff>=0.1.0

# Testing and coverage
pip install pytest>=7.4.0 pytest-cov>=4.1.0 pytest-mock>=3.11.0

# Type checking and security
pip install mypy>=1.5.0 bandit>=1.7.5 safety>=2.3.0

# Pre-commit hooks (optional on Windows)
pip install pre-commit>=3.4.0
```

#### 5. Configuration Files

```bash
# Generate configuration files
cat > pyproject.toml << EOF
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.ruff]
target-version = "py38"
line-length = 88
select = ["E", "W", "F", "I", "C", "B", "UP", "N"]
EOF
```

## Development Environment

### VS Code Setup

#### 1. Install Recommended Extensions

Open VS Code in the project directory and install these extensions:

- **Python Extension Pack** (`ms-python.python`)
- **Jupyter** (`ms-toolsai.jupyter`)
- **Black Formatter** (`ms-python.black-formatter`)
- **Ruff** (`charliermarsh.ruff`)
- **GitLens** (`eamodio.gitlens`)
- **Docker** (`ms-azuretools.vscode-docker`)

#### 2. Configure Workspace Settings

The project includes VS Code settings in `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true
}
```

#### 3. Use Integrated Tasks

Press `Ctrl+Shift+P` and type "Tasks: Run Task" to access:

- **Install Dependencies**: Install all required packages
- **Format Code**: Auto-format with Black
- **Run Tests**: Execute test suite with coverage
- **Lint Code**: Check code quality
- **Security Scan**: Run security analysis

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files

# Note: May have compatibility issues with Windows Store Python
# See docs/CODE_QUALITY.md for manual workflow
```

## Docker Setup

### Development Container

#### 1. Build Development Environment

```bash
# Build development container
docker-compose -f docker-compose.dev.yml build

# Start all services
docker-compose -f docker-compose.dev.yml up -d
```

#### 2. Available Services

- **Sandbox App**: `http://localhost:8501` (Streamlit dashboard)
- **Jupyter Lab**: `http://localhost:8888` (Interactive notebooks)
- **MLflow UI**: `http://localhost:5000` (Experiment tracking)
- **PostgreSQL**: `localhost:5432` (Database)

#### 3. Development Workflow

```bash
# Enter development container
docker-compose -f docker-compose.dev.yml exec sandbox-dev bash

# Run commands inside container
python main.py --mode dashboard

# View logs
docker-compose -f docker-compose.dev.yml logs -f sandbox-dev
```

### Production Deployment

```bash
# Build production image
docker build -t data-science-sandbox .

# Run production container
docker run -p 8501:8501 -v $(pwd)/data:/app/data data-science-sandbox
```

## Verification

### Health Checks

#### 1. Basic Functionality

```bash
# Test CLI interface
python main.py --mode cli --level 1

# Test data generation
python data/generate_datasets.py
ls -la data/datasets/

# Test modern tools
python -c "
from sandbox.utils.modern_tools_config import initialize_all_tools
tools = initialize_all_tools()
print('Available tools:', list(tools.keys()))
"
```

#### 2. Web Interfaces

```bash
# Start dashboard (should open browser automatically)
python main.py --mode dashboard

# Start Jupyter Lab
python main.py --mode jupyter
```

#### 3. Quality Assurance

```bash
# Run full test suite
python -m pytest tests/ -v --cov=sandbox --cov-report=html

# Code quality check
python -m ruff check sandbox/ main.py config.py

# Security scan
python -m bandit -r sandbox/ --skip B101,B601

# Dependency vulnerability check
python -m safety check --short-report
```

#### 4. Performance Test

```bash
# Test DuckDB performance
python -c "
import pandas as pd
from sandbox.integrations.modern_data_processing import ModernDataProcessor

processor = ModernDataProcessor()
df = pd.DataFrame({'x': range(1000000), 'y': range(1000000)})
result = processor.duckdb_query('SELECT COUNT(*) as total FROM df WHERE x > 500000', df)
print('DuckDB test passed:', result['total'].iloc[0] == 499999)
"
```

## Troubleshooting

### Common Issues

#### Python Version Conflicts

```bash
# Check Python version
python --version

# Use specific Python version
python3.11 -m venv venv
```

#### Dependency Installation Errors

```bash
# Clear pip cache
pip cache purge

# Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

#### Windows Store Python Issues

If using Windows Store Python and encountering pre-commit issues:

1. **Option 1**: Install Python from python.org
2. **Option 2**: Use manual code quality workflow (see `docs/CODE_QUALITY.md`)
3. **Option 3**: Use WSL2 or GitHub Codespaces

#### Module Import Errors

```bash
# Ensure you're in the project root
pwd

# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%cd%          # Windows
```

#### Docker Issues

```bash
# Check Docker installation
docker --version
docker-compose --version

# Restart Docker daemon
# Windows/macOS: Restart Docker Desktop
# Linux: sudo systemctl restart docker

# Clean Docker resources
docker system prune -a
```

#### Performance Issues

```bash
# Check memory usage
python -c "
import psutil
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB')
"

# Monitor resource usage during execution
htop  # Linux/macOS
# Task Manager on Windows
```

### Getting Help

#### 1. Check Documentation
- [API Reference](API.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Development Guide](DEVELOPMENT.md)

#### 2. Run Diagnostics

```bash
# System information
python -c "
import sys, platform, pandas as pd
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Pandas: {pd.__version__}')
"

# Check installed packages
pip list | grep -E '(pandas|numpy|streamlit|duckdb|polars)'
```

#### 3. Community Resources
- **GitHub Issues**: Report bugs or request features
- **Discussions**: Community Q&A
- **Wiki**: Detailed guides and tutorials

### Advanced Setup Options

#### Remote Development

```bash
# VS Code Remote Containers
# 1. Install "Remote - Containers" extension
# 2. Open project in VS Code
# 3. Ctrl+Shift+P → "Remote-Containers: Reopen in Container"
```

#### Cloud Development

```bash
# GitHub Codespaces
# 1. Go to repository on GitHub
# 2. Click "Code" → "Open with Codespaces"
# 3. Create new codespace
```

#### Conda Environment

```bash
# Create conda environment
conda create -n data-science-sandbox python=3.11
conda activate data-science-sandbox

# Install packages
conda install pandas numpy matplotlib scikit-learn jupyter
pip install streamlit duckdb polars mlflow
```

## Next Steps

After successful setup:

1. **Start Learning**: Launch dashboard and begin with Level 1
2. **Explore Notebooks**: Check out interactive tutorials in `notebooks/`
3. **Try Challenges**: Complete coding challenges in `challenges/`
4. **Experiment**: Use modern tools like DuckDB and Polars
5. **Contribute**: See development guide for contribution workflow

Your Data Science Sandbox is now ready for an amazing learning journey! 🚀