# UI Testing Guide

## Overview

This document provides comprehensive guidance for UI testing in the Data Science Sandbox project. The UI testing framework ensures that the Streamlit dashboard remains functional and user-friendly as the application evolves.

## Testing Framework

### Technology Stack

- **Playwright**: Modern browser automation framework
- **pytest-asyncio**: Async testing support for Python
- **pytest-html**: HTML test reporting
- **Streamlit**: The web framework being tested

### Test Structure

```text
tests/ui/
├── __init__.py           # Package initialization
├── conftest.py           # Test configuration and fixtures
├── test_dashboard.py     # Main dashboard functionality tests
├── test_challenges.py    # Challenge-related UI tests
├── test_progress.py      # Progress tracking tests
├── test_runner.py        # Test execution utilities
├── reports/              # Generated test reports
└── screenshots/          # Test screenshots for debugging
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Install UI testing packages
pip install -r requirements-dev.txt

# Install Playwright browsers
playwright install
```

### 2. Environment Setup

The testing framework automatically:

- Starts a Streamlit test server on port 8502
- Configures browser automation
- Sets up screenshot capture for debugging
- Manages test data and fixtures

### 3. Verify Installation

```bash
# Check if everything is properly configured
python tests/ui/test_runner.py --check
```

## Running Tests

### Quick Smoke Tests

Run essential tests to verify basic functionality:

```bash
# Via test runner
python tests/ui/test_runner.py --smoke

# Via pytest directly
pytest tests/ui/ -m smoke -v
```

### Full Test Suite

Run complete UI test suite:

```bash
# Via test runner
python tests/ui/test_runner.py --full

# Via pytest directly
pytest tests/ui/ -m ui -v
```

### Specific Test Categories

```bash
# Run dashboard tests only
python tests/ui/test_runner.py --file test_dashboard.py

# Run tests with specific marker
python tests/ui/test_runner.py --marker slow
```

### VS Code Integration

Use the built-in tasks in VS Code:

1. **Ctrl+Shift+P** → "Tasks: Run Task"
2. Select from available UI test tasks:
   - Install UI Test Dependencies
   - Install Playwright Browsers
   - Check UI Test Environment
   - Run UI Smoke Tests
   - Run Full UI Tests
   - Generate UI Test Report

## Test Categories

### 1. Smoke Tests (`@pytest.mark.smoke`)

Quick tests that verify basic functionality:

- Dashboard loads successfully
- Main navigation works
- Key elements are visible

### 2. UI Tests (`@pytest.mark.ui`)

Comprehensive interface testing:

- Navigation functionality
- Form interactions
- Button clicks
- Visual elements
- Responsive design

### 3. Slow Tests (`@pytest.mark.slow`)

Complex interaction flows:

- Multi-step user journeys
- Challenge completion flows
- Progress tracking accuracy

## Test Implementation

### Basic Test Structure

```python
import pytest
from playwright.async_api import Page, expect
from tests.ui.conftest import StreamlitTestUtils

class TestFeature:
    @pytest.mark.ui
    async def test_feature_functionality(self, page: Page, test_config: dict) -> None:
        """Test feature functionality"""
        # Wait for app to load
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Perform test actions
        element = page.locator("button")
        await expect(element).to_be_visible()

        # Take screenshot for debugging
        await StreamlitTestUtils.take_screenshot(page, "feature_test", test_config)
```

### Available Utility Functions

```python
# StreamlitTestUtils class provides:
await StreamlitTestUtils.wait_for_streamlit_ready(page)
await StreamlitTestUtils.click_sidebar_button(page, "Challenges")
await StreamlitTestUtils.get_metric_value(page, "Experience")
await StreamlitTestUtils.take_screenshot(page, "test_name", config)
await StreamlitTestUtils.wait_for_element(page, ".my-selector")
```

## Continuous Integration

### GitHub Actions

The project includes automated UI testing in CI/CD:

```yaml
# .github/workflows/ui-tests.yml
- Matrix testing across Python versions and browsers
- Accessibility testing with axe-core
- Performance testing with Lighthouse
- Automated report generation
```

### Test Reports

Reports are automatically generated and include:

- Test execution results
- Screenshots of failures
- Performance metrics
- Accessibility audit results

## Debugging

### Screenshots

Screenshots are automatically captured:

- On test failures
- At key test points
- For visual verification

Location: `tests/ui/screenshots/`

### Browser Developer Tools

For interactive debugging:

```python
# Add this line in your test to pause execution
await page.pause()
```

### Verbose Logging

Enable detailed logging:

```bash
pytest tests/ui/ -v --tb=long --capture=no
```

## Best Practices

### 1. Test Independence

- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 2. Reliable Selectors

```python
# Prefer data-testid attributes
page.locator('[data-testid="stSidebar"]')

# Use semantic selectors
page.locator('button:has-text("Start Challenge")')

# Avoid fragile CSS selectors
```

### 3. Wait Strategies

```python
# Wait for elements to be ready
await page.wait_for_selector('[data-testid="stApp"]')

# Wait for network to be idle
await page.wait_for_load_state("networkidle")

# Custom wait conditions
await expect(element).to_be_visible()
```

### 4. Error Handling

```python
# Use continue-on-error for non-critical tests
try:
    await element.click()
except Exception as e:
    logger.warning(f"Optional action failed: {e}")
```

## Troubleshooting

### Common Issues

1. **Streamlit Server Not Starting**

   ```bash
   # Check if port is available
   netstat -an | grep 8502

   # Manual server start
   streamlit run streamlit_app.py --server.port 8502
   ```

2. **Browser Installation Issues**

   ```bash
   # Reinstall browsers
   playwright install --force

   # Check browser status
   playwright install --dry-run
   ```

3. **Test Timeouts**

   ```python
   # Increase timeout for slow operations
   await page.wait_for_selector(selector, timeout=30000)
   ```

4. **Import Errors**

   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements-dev.txt

   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

### Debug Mode

Run tests with debug information:

```bash
# Enable debug mode
PYTEST_DEBUG=1 pytest tests/ui/ -v --tb=long

# Run single test with debugging
pytest tests/ui/test_dashboard.py::TestDashboardUI::test_dashboard_loads -v -s
```

## Integration with Development Workflow

### Pre-commit Hooks

Add UI tests to pre-commit pipeline:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: ui-smoke-tests
      name: UI Smoke Tests
      entry: python tests/ui/test_runner.py --smoke
      language: python
      pass_filenames: false
```

### Development Guidelines

1. **Feature Development**

   - Write UI tests alongside new features
   - Update tests when modifying UI components
   - Run smoke tests before committing

2. **Code Reviews**

   - Include UI test coverage in reviews
   - Verify test screenshots in CI artifacts
   - Check accessibility compliance

3. **Release Process**
   - Run full UI test suite before releases
   - Generate comprehensive test reports
   - Validate cross-browser compatibility

## Future Enhancements

### Planned Improvements

1. **Visual Regression Testing**

   - Implement screenshot comparison
   - Detect unintended visual changes
   - Automated visual baseline management

2. **Mobile Testing**

   - Extend responsive design testing
   - Add mobile device emulation
   - Touch interaction testing

3. **Performance Monitoring**

   - Continuous performance tracking
   - Load time regression detection
   - Memory usage monitoring

4. **User Journey Testing**
   - Complete user workflow testing
   - Multi-session testing
   - User persona-based testing

### Contributing

To add new UI tests:

1. Create test files in `tests/ui/`
2. Use appropriate markers (`@pytest.mark.ui`, etc.)
3. Follow naming conventions (`test_*.py`)
4. Include documentation and screenshots
5. Update this guide as needed

### Resources

- [Playwright Documentation](https://playwright.dev/python/)
- [pytest-asyncio Guide](https://pytest-asyncio.readthedocs.io/)
- [Streamlit Testing Best Practices](https://docs.streamlit.io/library/advanced-features/testing)
- [Web Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
