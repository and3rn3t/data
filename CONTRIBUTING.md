# Contributing to Data Science Sandbox

Thank you for your interest in contributing to the Data Science Sandbox! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contribution Types](#contribution-types)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

### Our Commitment

We are committed to providing a welcoming and inspiring community for all contributors, regardless of experience level, gender identity, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, nationality, or other protected characteristics.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Getting Started

### Prerequisites

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/data.git
   cd data
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Configure Git**
   ```bash
   # Configure your identity
   git config user.name "Your Name"
   git config user.email "your.email@example.com"

   # Add upstream remote
   git remote add upstream https://github.com/and3rn3t/data.git
   ```

## Development Workflow

### 1. Create Feature Branch

```bash
# Update your fork
git checkout main
git pull upstream main
git push origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow our [coding standards](#coding-standards) and ensure your changes:

- Have clear, focused commits
- Include appropriate tests
- Update documentation if needed
- Follow the established project structure

### 3. Test Your Changes

```bash
# Run test suite
python -m pytest tests/ -v --cov=sandbox

# Check code quality
python -m ruff check sandbox/ --fix
python -m black sandbox/ main.py config.py

# Security scan
python -m bandit -r sandbox/ --skip B101,B601

# Type checking
python -m mypy sandbox/ --ignore-missing-imports
```

### 4. Commit and Push

```bash
# Add your changes
git add .

# Commit with descriptive message
git commit -m "feat: add new challenge validation system

- Implement schema-based challenge validation
- Add unit tests for validation logic
- Update documentation with examples"

# Push to your fork
git push origin feature/your-feature-name
```

### 5. Create Pull Request

1. Go to GitHub and create a pull request from your fork
2. Fill out the pull request template completely
3. Wait for review and address feedback
4. Celebrate when your PR is merged! 🎉

## Contribution Types

### 🐛 Bug Fixes

**What qualifies:**
- Fixing broken functionality
- Correcting logical errors
- Resolving performance issues

**Guidelines:**
- Include steps to reproduce the bug
- Add regression tests
- Reference the issue number in your PR

**Example:**
```bash
git commit -m "fix: resolve dashboard crash on empty datasets

- Add null check in progress calculation
- Include test case for edge case
- Fixes #123"
```

### ✨ New Features

**What qualifies:**
- New challenges or levels
- Additional data science tools integration
- Dashboard enhancements
- New utility functions

**Guidelines:**
- Discuss major features in an issue first
- Follow existing patterns and conventions
- Include comprehensive tests
- Update documentation

**Example:**
```bash
git commit -m "feat: add Level 7 deep learning challenges

- Implement TensorFlow integration
- Add 5 new deep learning challenges
- Include neural network visualization tools
- Update progress tracking for new level"
```

### 📚 Documentation

**What qualifies:**
- API documentation improvements
- Tutorial additions
- Setup guide enhancements
- Code comments and docstrings

**Guidelines:**
- Use clear, concise language
- Include practical examples
- Follow existing documentation style
- Test all code examples

### 🎨 UI/UX Improvements

**What qualifies:**
- Dashboard design improvements
- Better data visualizations
- Enhanced user experience
- Accessibility improvements

**Guidelines:**
- Maintain iOS-inspired design language
- Ensure responsive design
- Test across different browsers
- Consider accessibility standards

### ⚡ Performance Optimizations

**What qualifies:**
- Algorithm improvements
- Memory usage reductions
- Database query optimizations
- Caching implementations

**Guidelines:**
- Include performance benchmarks
- Document the improvement
- Ensure correctness is maintained
- Add performance tests if applicable

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some specific guidelines:

#### Code Formatting
```python
# Use Black with 88-character line length
# This is automatically enforced by pre-commit hooks

# Good
def process_user_data(
    dataframe: pd.DataFrame, 
    validation_rules: Dict[str, Any]
) -> pd.DataFrame:
    """Process user data with validation."""
    pass

# Bad - not following Black formatting
def process_user_data(dataframe,validation_rules):
    pass
```

#### Type Hints
```python
# Always include type hints for public functions
from typing import Dict, List, Optional, Union
import pandas as pd

def analyze_dataset(
    df: pd.DataFrame,
    columns: List[str],
    threshold: Optional[float] = None
) -> Dict[str, Union[float, int]]:
    """Analyze dataset with specified parameters."""
    pass
```

#### Documentation
```python
def complete_challenge(self, challenge_id: str) -> bool:
    """Complete a challenge and update user progress.
    
    Args:
        challenge_id: Unique identifier for the challenge
        
    Returns:
        True if challenge was successfully completed
        
    Raises:
        ValidationError: If challenge_id is invalid
        ProgressError: If challenge is already completed
        
    Example:
        >>> game = GameEngine()
        >>> success = game.complete_challenge("level_1_data_loading")
        >>> assert success is True
    """
    pass
```

#### Error Handling
```python
# Use specific exception types
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value provided: {e}")
    raise
except FileNotFoundError as e:
    logger.warning(f"File not found, using defaults: {e}")
    result = get_default_value()
```

### Project Structure Guidelines

#### File Organization
```
sandbox/
├── core/           # Core game logic
│   ├── __init__.py
│   ├── game_engine.py
│   └── dashboard.py
├── integrations/   # External tool integrations
│   ├── __init__.py
│   ├── ml_tracking.py
│   └── data_processing.py
├── utils/          # Utility functions
│   ├── __init__.py
│   ├── validation.py
│   └── logging.py
└── levels/         # Level-specific content
    ├── __init__.py
    └── level_1.py
```

#### Import Standards
```python
# Standard library imports first
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

# Third-party imports second
import pandas as pd
import numpy as np
import streamlit as st

# Local imports last
from sandbox.core.game_engine import GameEngine
from sandbox.utils.validation import validate_data
```

### Configuration Management

#### Environment Variables
```python
# Use environment variables for sensitive data
import os
from typing import Optional

DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
```

#### Configuration Files
```python
# config.py - Centralized configuration
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AppConfig:
    """Application configuration settings."""
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = "data"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            debug=os.getenv("DEBUG", "False").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            data_dir=os.getenv("DATA_DIR", "data")
        )
```

## Testing Guidelines

### Test Structure

```python
# tests/test_game_engine.py
import pytest
from unittest.mock import patch, MagicMock

from sandbox.core.game_engine import GameEngine


class TestGameEngine:
    """Test suite for GameEngine class."""
    
    @pytest.fixture
    def game_engine(self):
        """Create GameEngine instance for testing."""
        return GameEngine()
    
    def test_initialization(self, game_engine):
        """Test proper initialization of GameEngine."""
        assert game_engine.get_progress()['level'] == 1
        assert game_engine.get_progress()['xp'] == 0
        assert len(game_engine.get_progress()['challenges_completed']) == 0
    
    def test_challenge_completion(self, game_engine):
        """Test challenge completion functionality."""
        # Given
        challenge_id = "level_1_data_loading"
        initial_xp = game_engine.get_progress()['xp']
        
        # When
        result = game_engine.complete_challenge(challenge_id)
        
        # Then
        assert result is True
        assert challenge_id in game_engine.get_progress()['challenges_completed']
        assert game_engine.get_progress()['xp'] > initial_xp
    
    def test_invalid_challenge_completion(self, game_engine):
        """Test handling of invalid challenge IDs."""
        # Given
        invalid_challenge_id = "non_existent_challenge"
        
        # When/Then
        with pytest.raises(ValueError, match="Invalid challenge ID"):
            game_engine.complete_challenge(invalid_challenge_id)
    
    @patch('sandbox.core.game_engine.save_progress')
    def test_progress_persistence(self, mock_save, game_engine):
        """Test that progress is properly saved."""
        # Given
        challenge_id = "level_1_data_loading"
        
        # When
        game_engine.complete_challenge(challenge_id)
        
        # Then
        mock_save.assert_called_once()


# Integration tests
class TestGameEngineIntegration:
    """Integration tests for GameEngine."""
    
    def test_full_level_completion(self):
        """Test completing an entire level."""
        game = GameEngine()
        level_1_challenges = [
            "level_1_data_loading",
            "level_1_basic_analysis",
            "level_1_visualization"
        ]
        
        for challenge in level_1_challenges:
            game.complete_challenge(challenge)
        
        # Should advance to level 2
        assert game.get_progress()['level'] == 2
```

### Test Categories

#### Unit Tests
- Test individual functions in isolation
- Mock external dependencies
- Fast execution (< 1 second per test)
- High coverage of edge cases

#### Integration Tests
- Test component interactions
- Use real dependencies where practical
- Moderate execution time (< 30 seconds per test)
- Focus on critical user workflows

#### Performance Tests
```python
def test_large_dataset_performance():
    """Test performance with large datasets."""
    import time
    
    # Create large dataset
    large_df = pd.DataFrame({
        'col1': range(1000000),
        'col2': np.random.randn(1000000)
    })
    
    # Measure processing time
    start_time = time.time()
    result = process_large_dataset(large_df)
    execution_time = time.time() - start_time
    
    # Should complete within reasonable time
    assert execution_time < 10.0  # seconds
    assert len(result) == 1000000
```

### Testing Best Practices

1. **Follow AAA Pattern**: Arrange, Act, Assert
2. **Use Descriptive Names**: Test names should explain what they test
3. **One Assertion Per Test**: Focus on one specific behavior
4. **Use Fixtures**: Share test setup code efficiently
5. **Mock External Dependencies**: Keep tests fast and reliable

## Documentation Guidelines

### Code Documentation

#### Docstring Style
```python
def analyze_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns.
    
    This function computes pairwise correlation of columns, excluding
    NA/null values. Non-numeric columns are automatically excluded.
    
    Args:
        df: Input DataFrame with numeric columns
        method: Method of correlation:
            - 'pearson': Standard correlation coefficient
            - 'kendall': Kendall Tau correlation coefficient  
            - 'spearman': Spearman rank correlation
        min_periods: Minimum number of observations required per pair
            of columns to have a valid result. Optional parameter.
    
    Returns:
        Correlation matrix as DataFrame with correlation coefficients
        between -1 and 1, where:
        - 1 indicates perfect positive correlation
        - 0 indicates no linear correlation
        - -1 indicates perfect negative correlation
    
    Raises:
        ValueError: If method is not supported
        TypeError: If DataFrame contains no numeric columns
    
    Example:
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4, 5],
        ...     'B': [2, 4, 6, 8, 10],
        ...     'C': [1, 3, 2, 4, 5]
        ... })
        >>> corr_matrix = analyze_correlation_matrix(df)
        >>> print(corr_matrix.loc['A', 'B'])
        1.0
    
    Note:
        For small datasets (< 30 observations), correlation coefficients
        may be unstable and should be interpreted with caution.
    """
```

#### Markdown Documentation

##### Structure
```markdown
# Title (H1 - Only one per document)

Brief introduction paragraph explaining the purpose and scope.

## Main Section (H2)

Content organized in logical sections.

### Subsection (H3)

More detailed content within sections.

#### Details (H4)

Specific implementation details or examples.
```

##### Code Examples
````markdown
## Usage Example

Here's how to use the feature:

```python
from sandbox.core.game_engine import GameEngine

# Initialize game engine
game = GameEngine()

# Complete a challenge
success = game.complete_challenge("level_1_data_loading")
print(f"Challenge completed: {success}")
```

Expected output:
```
Challenge completed: True
```
````

### API Documentation

Use consistent formatting for API documentation:

```markdown
### `method_name(param1: type, param2: type = default) -> return_type`

Brief description of what the method does.

**Parameters:**
- `param1` (type): Description of parameter
- `param2` (type, optional): Description with default value

**Returns:**
- `return_type`: Description of return value

**Raises:**
- `ExceptionType`: When this exception is raised

**Example:**
```python
result = obj.method_name("value", param2=42)
```
```

## Pull Request Process

### Before Submitting

1. **Update Documentation**: Ensure all changes are documented
2. **Add Tests**: Include tests for new functionality
3. **Run Quality Checks**: Ensure all checks pass
4. **Update Changelog**: Add entry to CHANGELOG.md if applicable

### PR Template

When creating a pull request, use this template:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist:
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Documentation Review**: Ensure documentation is updated
4. **Testing Verification**: Verify tests cover new functionality

### After Merge

1. **Update Local Repository**:
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

2. **Delete Feature Branch**:
   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

## Recognition

### Contributors

All contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Annual contributor appreciation posts

### Contribution Levels

- **Contributor**: Made at least one merged PR
- **Regular Contributor**: 5+ merged PRs or significant documentation
- **Core Contributor**: 15+ merged PRs and ongoing involvement
- **Maintainer**: Trusted with repository management responsibilities

Thank you for contributing to Data Science Sandbox! Your efforts help create a better learning experience for everyone. 🚀