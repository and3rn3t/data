#!/bin/bash

# Data Science Sandbox - Testing Script

set -e

echo "🧪 Running Data Science Sandbox Tests"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Virtual environment: $VIRTUAL_ENV"
else
    print_status "Activating virtual environment..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        print_error "Virtual environment not found. Run ./scripts/setup.sh first."
        exit 1
    fi
fi

# Run code formatting check
print_status "Checking code formatting with Black..."
if black --check --diff .; then
    print_success "Code formatting is correct"
else
    print_error "Code formatting issues found. Run 'black .' to fix."
    exit 1
fi

# Run import sorting check
print_status "Checking import sorting with isort..."
if isort --check-only --diff .; then
    print_success "Import sorting is correct"
else
    print_error "Import sorting issues found. Run 'isort .' to fix."
    exit 1
fi

# Run linting
print_status "Running linting with flake8..."
if flake8 sandbox/ main.py config.py --max-line-length=88; then
    print_success "Linting passed"
else
    print_error "Linting issues found"
    exit 1
fi

# Run type checking
print_status "Running type checking with mypy..."
if mypy sandbox/ --ignore-missing-imports; then
    print_success "Type checking passed"
else
    print_error "Type checking issues found"
    exit 1
fi

# Run security checks
print_status "Running security checks with bandit..."
if bandit -r sandbox/ -f json -o bandit-report.json; then
    print_success "Security checks passed"
else
    print_error "Security issues found. Check bandit-report.json"
    exit 1
fi

# Run dependency vulnerability checks
print_status "Checking dependencies for vulnerabilities with safety..."
if safety check --json --output safety-report.json; then
    print_success "Dependency security checks passed"
else
    print_error "Vulnerable dependencies found. Check safety-report.json"
    exit 1
fi

# Run unit tests
print_status "Running unit tests..."
if pytest tests/unit/ -v --cov=sandbox --cov-report=term-missing; then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

# Run integration tests
print_status "Running integration tests..."
if pytest tests/integration/ -v; then
    print_success "Integration tests passed"
else
    print_error "Integration tests failed"
    exit 1
fi

# Generate coverage report
print_status "Generating coverage report..."
pytest --cov=sandbox --cov-report=html --cov-report=xml --cov-report=term

# Test basic functionality
print_status "Testing basic application functionality..."
if python -c "from sandbox.core.game_engine import GameEngine; engine = GameEngine(); print('✅ Core engine works')"; then
    print_success "Basic functionality test passed"
else
    print_error "Basic functionality test failed"
    exit 1
fi

print_success "All tests passed! 🎉"
print_status "Coverage report generated in htmlcov/"
print_status "Security reports: bandit-report.json, safety-report.json"