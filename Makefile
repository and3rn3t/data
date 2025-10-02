# Data Science Sandbox Makefile
# Provides common commands for development, testing, and deployment

.PHONY: help install install-dev test lint format type-check security clean docs serve
.DEFAULT_GOAL := help

# Python and virtual environment settings
PYTHON := python
PIP := pip
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

# Project settings
PROJECT_NAME := data-science-sandbox
SRC_DIRS := sandbox challenges
TEST_DIR := tests
DOCS_DIR := docs

help: ## Show this help message
	@echo "Data Science Sandbox Development Commands"
	@echo "========================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment Setup
install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[all]"
	$(PIP) install -r requirements-dev.txt
	pre-commit install

venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip setuptools wheel

# Code Quality
format: ## Format code with black and isort
	black $(SRC_DIRS) $(TEST_DIR) main.py config.py --line-length=88
	isort $(SRC_DIRS) $(TEST_DIR) main.py config.py --profile=black

lint: ## Lint code with ruff and flake8
	ruff check $(SRC_DIRS) main.py config.py --fix
	ruff format $(SRC_DIRS) main.py config.py
	flake8 $(SRC_DIRS) main.py config.py

type-check: ## Run type checking with mypy
	mypy $(SRC_DIRS) --ignore-missing-imports --show-error-codes

security: ## Run security checks
	bandit -r $(SRC_DIRS) -f json -o bandit-report.json
	safety check --json --output safety-report.json

quality: format lint type-check security ## Run all code quality checks

# Testing
test: ## Run tests with coverage
	pytest $(TEST_DIR) -v --cov=$(SRC_DIRS) --cov-report=html --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest $(TEST_DIR) -v -x

test-integration: ## Run integration tests
	pytest $(TEST_DIR)/integration -v

test-ui: ## Run UI tests
	python tests/ui/test_runner.py --smoke

# Data and Models
generate-data: ## Generate sample datasets
	$(PYTHON) data/generate_datasets.py

clean-data: ## Clean generated data files
	rm -rf data/processed/*
	rm -rf data/external/*

clean-models: ## Clean model artifacts
	rm -rf mlruns/
	rm -rf mlartifacts/
	rm -rf model_registry/
	rm -rf wandb/

# Application
serve: ## Start the Streamlit dashboard
	streamlit run streamlit_app.py

cli: ## Start CLI mode
	$(PYTHON) main.py --mode cli

jupyter: ## Launch Jupyter Lab
	$(PYTHON) main.py --mode jupyter

dashboard: ## Start web dashboard
	$(PYTHON) main.py --mode dashboard

# Documentation
docs: ## Generate documentation
	sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html

docs-serve: ## Serve documentation locally
	cd $(DOCS_DIR)/_build/html && python -m http.server 8000

# Cleanup
clean: ## Clean build artifacts and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

clean-all: clean clean-data clean-models ## Clean everything

# Pre-commit hooks
pre-commit: ## Run pre-commit hooks manually
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Docker
docker-build: ## Build Docker image
	docker build -t $(PROJECT_NAME) .

docker-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t $(PROJECT_NAME):dev .

docker-run: ## Run Docker container
	docker run -p 8501:8501 $(PROJECT_NAME)

# MLOps
mlflow-ui: ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

tensorboard: ## Start TensorBoard (if logs exist)
	tensorboard --logdir=logs --host=0.0.0.0 --port=6006

# Profiling
profile: ## Profile application performance
	python -m cProfile -o profile.prof main.py --mode cli
	python -c "import pstats; pstats.Stats('profile.prof').sort_stats('tottime').print_stats(20)"

memory-profile: ## Profile memory usage
	mprof run main.py --mode cli
	mprof plot

# Utilities
check-deps: ## Check for dependency updates
	pip list --outdated

update-deps: ## Update dependencies (use with caution)
	pip-review --local --interactive

freeze: ## Generate requirements.txt from current environment
	pip freeze > requirements-frozen.txt

# Git hooks
setup-git: ## Setup git hooks and config
	pre-commit install
	git config core.autocrlf false
	git config pull.rebase true

# Backup
backup-config: ## Backup important config files
	mkdir -p backups/config
	cp pyproject.toml backups/config/
	cp .flake8 backups/config/
	cp .pre-commit-config.yaml backups/config/
	cp pytest.ini backups/config/
	echo "Config files backed up to backups/config/"
