# GitHub Copilot Instructions

## Project Overview

This is a data science sandbox project designed for learning and experimentation with data analysis, machine learning, and modern data science tools. The project includes a gamified learning system with progressive challenges across multiple levels.

## Project Structure

- `sandbox/` - Core application modules
- `data/` - Dataset storage and generation scripts
- `notebooks/` - Jupyter notebooks for interactive analysis
- `challenges/` - Progressive learning challenges (Levels 1-7)
- `tests/` - Unit tests and test configurations
- `scripts/` - Deployment and maintenance scripts

## Code Style and Standards

### Python Style Guidelines

- Follow PEP 8 conventions
- Use Black formatter with 88 character line length
- Sort imports using isort with Black profile
- Include type hints where appropriate (mypy compatible)
- Write docstrings for all public functions and classes

### File Organization

- Keep related functionality in appropriate modules under `sandbox/`
- Use `__init__.py` files to define package interfaces
- Place utility functions in `sandbox/utils/`
- Core game mechanics go in `sandbox/core/`
- Data processing integrations in `sandbox/integrations/`

## Coding Preferences

### Data Science Code

- Use pandas for data manipulation
- Prefer matplotlib/seaborn for visualization
- Use scikit-learn for machine learning tasks
- Include proper error handling for file I/O operations
- Add logging for debugging and monitoring

### Architecture Patterns

- Follow object-oriented principles for core classes
- Use dependency injection where appropriate
- Implement proper separation of concerns
- Create reusable components in the utils module

### Testing

- Write unit tests for all core functionality
- Use pytest framework
- Include coverage reporting
- Test edge cases and error conditions
- Mock external dependencies

### Documentation

- Include inline comments for complex logic
- Write clear docstrings with parameter and return type information
- Update README files when adding new features
- Document configuration options

## Specific Guidelines

### When Working with Data

- Always validate input data before processing
- Handle missing values explicitly
- Use appropriate data types (categories for categorical data)
- Include data validation and cleaning steps
- Document data assumptions and transformations

### Machine Learning Code

- Split data properly (train/validation/test)
- Use cross-validation for model evaluation
- Include feature engineering documentation
- Save and version models appropriately
- Track experiments and hyperparameters

### Dashboard and Visualization

- Create interactive plots when beneficial
- Use consistent color schemes and styling
- Include proper axis labels and titles
- Ensure plots are accessible and readable
- Document visualization choices

### Configuration Management

- Use `config.py` for application settings
- Keep sensitive information in environment variables
- Document configuration options clearly
- Use type hints for configuration classes

## Development Workflow

### Code Quality Checks

The project includes automated code quality tools:

- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting
- `mypy` for type checking
- `pytest` for testing

### Available Tasks

Use the predefined VS Code tasks for common operations:

- "Install Dependencies" - Install required packages
- "Format Code" - Auto-format with Black
- "Sort Imports" - Organize imports with isort
- "Lint Code" - Check code quality with flake8
- "Type Check" - Validate types with mypy
- "Run Tests" - Execute test suite with coverage
- "Generate Sample Data" - Create example datasets

### Branch and Commit Strategy

- Make small, focused commits
- Write clear commit messages
- Test code before committing
- Run code quality checks before pushing

## Error Handling and Logging

- Use proper exception handling with specific exception types
- Include informative error messages
- Add logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Handle file not found and permission errors gracefully

## Performance Considerations

- Use vectorized operations with pandas/numpy
- Consider memory usage with large datasets
- Profile code when performance is critical
- Use appropriate data structures for the task

## Security Guidelines

- Don't hardcode sensitive information
- Validate user inputs
- Use secure file handling practices
- Keep dependencies updated

## When Suggesting Code Changes

- Consider the existing code style and patterns
- Suggest improvements that align with the project structure
- Include appropriate tests for new functionality
- Update documentation when adding features
- Consider backward compatibility when modifying existing APIs

## Challenge-Specific Guidelines

When working on challenge files or related code:

- Maintain progressive difficulty across levels
- Include clear learning objectives
- Provide hints and explanations in comments
- Create reproducible examples
- Test all code examples thoroughly

## Integration Guidelines

When adding new integrations (in `sandbox/integrations/`):

- Create modular, reusable components
- Include proper configuration management
- Add comprehensive error handling
- Document integration requirements
- Provide usage examples
