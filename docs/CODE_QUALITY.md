# Manual Code Quality Workflow

Since pre-commit hooks may have compatibility issues on Windows, here's how to run code quality checks manually:

## Before Committing Changes

### 1. Format Code

```powershell
# Format with Black
python -m black --line-length=88 sandbox/ main.py config.py

# Sort imports with isort
python -m isort --profile=black sandbox/ main.py config.py
```

### 2. Check Code Quality

```powershell
# Run Ruff linter (fast, modern)
python -m ruff check sandbox/ main.py config.py --fix

# Run flake8 (traditional linting)
python -m flake8 sandbox/ main.py config.py --max-line-length=88

# Type checking with mypy
python -m mypy sandbox/ --ignore-missing-imports
```

### 3. Security Scans

```powershell
# Check for security issues
python -m bandit -r sandbox/ --skip B101,B601

# Check for vulnerable dependencies
python -m safety check --short-report
```

### 4. Run Tests

```powershell
# Run test suite with coverage
python -m pytest tests/ -v --cov=sandbox --cov-report=html
```

## VS Code Tasks

You can also use the predefined VS Code tasks:

- `Ctrl+Shift+P` → "Tasks: Run Task"
- Select from available tasks like "Format Code", "Lint Code", etc.

## Automated CI/CD

The GitHub Actions workflows will automatically run these checks on:

- Push to main branch
- Pull requests
- Weekly security scans

## Pre-commit Setup (Advanced)

If you want to set up pre-commit hooks (may require non-Windows Store Python):

```powershell
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Windows Store Python Issues

If you're using Windows Store Python and encountering pre-commit issues:

1. **Option 1**: Use manual workflow above
2. **Option 2**: Install Python from python.org instead of Windows Store
3. **Option 3**: Use GitHub Codespaces or WSL2 for development

The automated GitHub Actions will ensure code quality regardless of local setup.
