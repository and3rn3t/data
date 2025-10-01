# UI Testing for Data Science Sandbox

This directory contains comprehensive UI tests for the Streamlit dashboard.

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r ../../../requirements-dev.txt
   playwright install
   ```

2. **Verify setup:**

   ```bash
   python test_runner.py --check
   ```

3. **Run smoke tests:**

   ```bash
   python test_runner.py --smoke
   ```

## Test Files

- `conftest.py` - Test configuration and fixtures
- `test_dashboard.py` - Main dashboard functionality
- `test_challenges.py` - Challenge-related features
- `test_progress.py` - Progress tracking and analytics
- `test_setup_validation.py` - Environment validation
- `test_runner.py` - Test execution utilities

## Usage

```bash
# Run all UI tests
python test_runner.py --full

# Generate comprehensive report
python test_runner.py --report

# Run specific test file
python test_runner.py --file test_dashboard.py
```

## Integration

Use VS Code tasks or run via pytest:

```bash
pytest tests/ui/ -m ui -v --html=reports/ui_report.html
```

See [UI_TESTING.md](../../../docs/UI_TESTING.md) for complete documentation.
