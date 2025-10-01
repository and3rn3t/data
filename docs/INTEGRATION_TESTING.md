# Integration Testing Documentation

## Overview

This document describes the comprehensive integration testing framework for the Data Science Sandbox project. Integration tests verify that different components work together correctly, including database connections, data pipelines, ML experiment tracking, and end-to-end user workflows.

## Test Structure

```
tests/integration/
├── __init__.py                          # Integration test package
├── conftest.py                          # Shared fixtures and configuration
├── test_runner.py                       # Integration test runner utility
├── test_database_integration.py         # Database connection and operation tests
├── test_data_pipeline_integration.py    # Data pipeline workflow tests
├── test_mlflow_integration.py          # ML experiment tracking tests
└── test_end_to_end_workflows.py        # Complete user workflow tests
```

## Test Categories

### 1. Database Integration Tests (`test_database_integration.py`)

Tests DuckDB database connectivity and operations:

- **Connection establishment**: Basic database setup and connectivity
- **DataFrame operations**: SQL queries on pandas DataFrames
- **Data persistence**: Writing and reading data to/from database
- **Error handling**: Database error recovery and resilience
- **Performance**: Large dataset handling and query optimization
- **Concurrency**: Multiple simultaneous database connections

**Key Test Methods:**

- `test_duckdb_connection_establishment()` - Basic connection setup
- `test_duckdb_dataframe_operations()` - SQL operations on DataFrames
- `test_data_persistence_workflow()` - Complete persistence workflow
- `test_large_dataset_performance()` - Performance with large datasets

### 2. Data Pipeline Integration Tests (`test_data_pipeline_integration.py`)

Tests end-to-end data processing workflows:

- **Complete workflow**: Raw data → processing → analysis
- **Data validation**: Schema validation and quality checks
- **File I/O operations**: Reading from and writing to files
- **Error handling**: Pipeline resilience and recovery
- **Performance monitoring**: Execution time and resource usage
- **Cross-tool integration**: DuckDB + Pipeline Builder workflows

**Key Test Methods:**

- `test_complete_data_processing_workflow()` - Full data processing pipeline
- `test_data_validation_integration()` - Data quality validation
- `test_pipeline_with_file_io()` - File input/output operations
- `test_cross_tool_integration()` - Integration between different tools

### 3. MLflow Integration Tests (`test_mlflow_integration.py`)

Tests machine learning experiment tracking workflows:

- **Experiment lifecycle**: Create → log → track → compare experiments
- **Model registration**: Model versioning and registry operations
- **Artifact logging**: Storing and retrieving ML artifacts
- **Cross-experiment analysis**: Comparing multiple experiments
- **Model serving preparation**: Preparing models for deployment
- **Large experiment handling**: Managing experiments with many runs

**Key Test Methods:**

- `test_experiment_lifecycle()` - Complete MLflow experiment workflow
- `test_model_registration_workflow()` - Model registry operations
- `test_artifact_logging()` - Artifact storage and retrieval
- `test_cross_experiment_analysis()` - Multi-experiment comparisons

### 4. End-to-End Workflow Tests (`test_end_to_end_workflows.py`)

Tests complete user workflows and system integration:

- **New user onboarding**: First-time user experience
- **Level progression**: Advancing through challenge levels
- **Badge earning**: Achievement system functionality
- **Progress persistence**: Save/load user progress
- **Error recovery**: System resilience to failures
- **Data analysis integration**: Challenges with data processing
- **Full system integration**: All components working together
- **Concurrent users**: Multiple simultaneous user sessions

**Key Test Methods:**

- `test_new_user_onboarding_workflow()` - New user experience
- `test_level_progression_workflow()` - Multi-level progression
- `test_progress_persistence_workflow()` - Progress save/load
- `test_full_system_integration()` - Complete system workflow

## Running Integration Tests

### Prerequisites

Ensure all required dependencies are installed:

```bash
# Install core requirements
pip install pandas pytest

# Install optional integrations (tests will be skipped if missing)
pip install duckdb mlflow polars great-expectations pandera
```

### Test Execution Commands

#### Check Requirements

```bash
python tests/integration/test_runner.py --check
```

#### Run All Integration Tests

```bash
python tests/integration/test_runner.py --all
```

#### Run Specific Test Categories

```bash
# Database tests only
python tests/integration/test_runner.py --database

# Data pipeline tests only
python tests/integration/test_runner.py --pipeline

# MLflow tests only
python tests/integration/test_runner.py --mlflow

# End-to-end workflow tests only
python tests/integration/test_runner.py --workflow
```

#### Run by Performance

```bash
# Fast tests only (< 10 seconds each)
python tests/integration/test_runner.py --fast

# Slow tests only (comprehensive testing)
python tests/integration/test_runner.py --slow
```

#### Generate Reports

```bash
# Generate comprehensive test report with coverage
python tests/integration/test_runner.py --report
```

#### Smoke Tests

```bash
# Quick validation that integration components work
python tests/integration/test_runner.py --smoke
```

### Using pytest directly

```bash
# Run all integration tests
pytest tests/integration/ -m integration -v

# Run specific test file
pytest tests/integration/test_database_integration.py -v

# Run tests with specific markers
pytest tests/integration/ -m "integration and not slow" -v

# Run with coverage reporting
pytest tests/integration/ --cov=sandbox --cov-report=html
```

## Test Markers

Integration tests use pytest markers for organization:

- `@pytest.mark.integration` - All integration tests
- `@pytest.mark.slow` - Tests that take longer to execute
- `@pytest.mark.database` - Database-specific tests
- `@pytest.mark.mlflow` - MLflow-specific tests
- `@pytest.mark.pipeline` - Data pipeline tests
- `@pytest.mark.workflow` - End-to-end workflow tests

## Test Configuration

### Environment Variables

Tests use these environment variables when available:

- `TEST_MODE=true` - Enables test mode
- `MLFLOW_TRACKING_URI` - MLflow tracking server URL
- `PYTHONPATH` - Python module search path

### Test Fixtures

Shared fixtures provided in `conftest.py`:

- `integration_test_dir` - Temporary directory for test files
- `clean_environment` - Clean environment variables
- `mock_data_dir` - Directory with sample test data

### Test Configuration Class

`IntegrationTestConfig` provides:

- Timeout values for different test types
- Dataset size configurations
- Database and MLflow settings
- Helper methods for test setup

## Expected Test Behavior

### Graceful Degradation

Tests are designed to gracefully handle missing optional dependencies:

```python
if not self.processor.duckdb_available:
    pytest.skip("DuckDB not available")
```

### Temporary Resources

All tests use temporary directories and clean up automatically:

- Test databases are created in temporary locations
- MLflow artifacts are stored in test-specific directories
- All temporary files are cleaned up after tests complete

### Error Scenarios

Tests verify both success and failure scenarios:

- Database connection failures
- Invalid data processing operations
- MLflow connectivity issues
- Game engine error recovery

## Performance Expectations

### Fast Tests (< 10 seconds each)

- Basic connectivity tests
- Small dataset operations
- Configuration validation

### Slow Tests (10-60 seconds each)

- Large dataset processing
- Complex pipeline operations
- Multi-experiment MLflow workflows
- Full system integration scenarios

## Troubleshooting

### Common Issues

1. **Missing Dependencies**

   - Install optional packages: `pip install duckdb mlflow`
   - Tests will skip gracefully if dependencies are missing

2. **Permission Errors**

   - Ensure write permissions in test directory
   - Check temporary directory access

3. **Timeout Issues**

   - Increase timeout values in `IntegrationTestConfig`
   - Run with `--slow` flag for comprehensive testing

4. **Database Lock Issues**
   - Ensure proper cleanup in test teardown
   - Check for concurrent test execution conflicts

### Debug Mode

Run tests with verbose output and no capture:

```bash
pytest tests/integration/ -v -s --tb=long
```

### Test Reports

Generated reports include:

- **HTML Report**: `tests/integration/reports/integration_test_report.html`
- **JUnit XML**: `tests/integration/reports/integration_test_results.xml`
- **Coverage Report**: `tests/integration/reports/integration_coverage/`

## Integration with CI/CD

### GitHub Actions Integration

```yaml
- name: Run Integration Tests
  run: |
    python tests/integration/test_runner.py --check
    python tests/integration/test_runner.py --fast

- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: integration-test-reports
    path: tests/integration/reports/
```

### Pre-commit Hooks

Add integration test validation:

```yaml
- repo: local
  hooks:
    - id: integration-smoke-tests
      name: Integration Smoke Tests
      entry: python tests/integration/test_runner.py --smoke
      language: system
      pass_filenames: false
```

## Best Practices

### Writing Integration Tests

1. **Use appropriate markers** for test categorization
2. **Clean up resources** in teardown methods
3. **Handle missing dependencies** gracefully
4. **Test both success and failure** scenarios
5. **Use realistic test data** sizes
6. **Verify end-to-end workflows** not just individual components

### Test Organization

1. **Group related tests** in appropriate test files
2. **Use descriptive test names** that explain the scenario
3. **Document complex test scenarios** with docstrings
4. **Share common setup** through fixtures
5. **Separate fast and slow tests** with markers

### Maintenance

1. **Update tests** when adding new integrations
2. **Monitor test performance** and optimize slow tests
3. **Review test coverage** regularly
4. **Update documentation** when test structure changes
5. **Validate tests** work in different environments
