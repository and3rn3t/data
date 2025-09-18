#!/bin/bash

# Data Science Sandbox - Development Environment Setup Script

set -e

echo "🚀 Setting up Data Science Sandbox Development Environment"
echo "========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1)
print_success "Found $python_version"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install Python3 and pip3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Install development tools
print_status "Installing development tools..."
pip install black isort flake8 mypy pytest pytest-cov pytest-mock pre-commit bandit safety

# Install additional useful packages for development
pip install ipython jupyter-lab jupyter-book jupyter-contrib-nbextensions

# Generate sample datasets
print_status "Generating sample datasets..."
if [ -f "data/generate_datasets.py" ]; then
    python data/generate_datasets.py
    print_success "Sample datasets generated"
else
    print_warning "Sample data generator not found"
fi

# Setup git hooks (if .git exists)
if [ -d ".git" ]; then
    print_status "Setting up git hooks..."
    pre-commit install
    print_success "Git hooks installed"
else
    print_warning "Git repository not found, skipping git hooks setup"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p tests/unit tests/integration tests/fixtures
mkdir -p logs temp .pytest_cache
mkdir -p docs/api docs/guides docs/examples
mkdir -p data/processed data/raw data/external
mkdir -p notebooks/experimental notebooks/production
mkdir -p scripts/data_processing scripts/deployment scripts/maintenance

# Create basic test structure
if [ ! -f "tests/__init__.py" ]; then
    touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
    print_success "Test structure created"
fi

# Create environment file template
if [ ! -f ".env" ]; then
    print_status "Creating .env template..."
    cat > .env << 'EOF'
# Data Science Sandbox Environment Configuration

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
SANDBOX_MODE=development

# Database (if needed)
DATABASE_URL=sqlite:///sandbox.db

# API Keys (add your keys here)
# OPENAI_API_KEY=your_key_here
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_key

# Paths
DATA_DIR=./data
MODELS_DIR=./models
LOGS_DIR=./logs

# Jupyter Configuration
JUPYTER_PORT=8888
JUPYTER_TOKEN=sandbox_token

# Dashboard Configuration
STREAMLIT_PORT=8501
FLASK_PORT=5000

# Resource Limits
MAX_MEMORY_GB=8
MAX_WORKERS=4
EOF
    print_success ".env template created"
else
    print_warning ".env file already exists"
fi

# Create basic configuration files
if [ ! -f "pytest.ini" ]; then
    print_status "Creating pytest configuration..."
    cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=sandbox
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow-running tests
    api: API tests
EOF
    print_success "pytest configuration created"
fi

if [ ! -f ".flake8" ]; then
    print_status "Creating flake8 configuration..."
    cat > .flake8 << 'EOF'
[flake8]
max-line-length = 88
extend-ignore = 
    E203,  # whitespace before ':'
    E501,  # line too long (handled by black)
    W503,  # line break before binary operator
exclude = 
    .git,
    __pycache__,
    venv,
    .venv,
    build,
    dist,
    *.egg-info,
    .pytest_cache,
    .coverage,
    htmlcov
per-file-ignores =
    __init__.py:F401
EOF
    print_success "flake8 configuration created"
fi

if [ ! -f "pyproject.toml" ]; then
    print_status "Creating pyproject.toml..."
    cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.pytest_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"
EOF
    print_success "pyproject.toml created"
fi

# Update .gitignore
print_status "Updating .gitignore..."
if [ -f ".gitignore" ]; then
    # Add development-specific ignores if not already present
    grep -qxF "venv/" .gitignore || echo "venv/" >> .gitignore
    grep -qxF ".env" .gitignore || echo ".env" >> .gitignore
    grep -qxF "*.pyc" .gitignore || echo "*.pyc" >> .gitignore
    grep -qxF "__pycache__/" .gitignore || echo "__pycache__/" >> .gitignore
    grep -qxF ".pytest_cache/" .gitignore || echo ".pytest_cache/" >> .gitignore
    grep -qxF "htmlcov/" .gitignore || echo "htmlcov/" >> .gitignore
    grep -qxF ".coverage" .gitignore || echo ".coverage" >> .gitignore
    grep -qxF "*.egg-info/" .gitignore || echo "*.egg-info/" >> .gitignore
    grep -qxF "dist/" .gitignore || echo "dist/" >> .gitignore
    grep -qxF "build/" .gitignore || echo "build/" >> .gitignore
    grep -qxF ".mypy_cache/" .gitignore || echo ".mypy_cache/" >> .gitignore
    grep -qxF "logs/" .gitignore || echo "logs/" >> .gitignore
    grep -qxF "temp/" .gitignore || echo "temp/" >> .gitignore
    print_success ".gitignore updated"
fi

print_success "Development environment setup complete!"
print_status "You can now:"
echo "  • Open the project in VSCode for a fully configured experience"
echo "  • Run 'source venv/bin/activate' to activate the virtual environment"
echo "  • Run 'python main.py --help' to see available options"
echo "  • Run 'pytest' to execute tests"
echo "  • Run 'black .' to format code"
echo "  • Run 'flake8' to lint code"
echo ""
print_success "Happy coding! 🎉"