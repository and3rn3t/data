# 🎮 Data Science Sandbox - Developer Guide

## Welcome to the Development Environment!

This comprehensive guide will help you set up, develop, and contribute to the Data Science Sandbox project.

## 📋 Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Git** for version control
- **VSCode** (recommended) with the extensions listed in `.vscode/extensions.json`
- **Virtual environment support** (venv, conda, or similar)

## 🚀 Quick Setup

### 1. Automated Setup (Recommended)

```bash
# Clone and navigate to the project
git clone https://github.com/and3rn3t/data.git
cd data

# Run the setup script (creates venv, installs deps, configures environment)
./scripts/setup.sh

# Activate virtual environment
source venv/bin/activate

# Test the installation
python main.py --help
```

### 2. Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install black isort flake8 mypy pytest pytest-cov

# Generate sample datasets
python data/generate_datasets.py

# Run tests to verify setup
pytest
```

## 🛠️ Development Workflow

### VSCode Integration

Open the project in VSCode for the best development experience:

1. **Extensions**: Install recommended extensions when prompted
2. **Python Interpreter**: Select the virtual environment interpreter
3. **Debugging**: Use F5 to start debugging with pre-configured launch configs
4. **Tasks**: Use Ctrl+Shift+P → "Tasks: Run Task" for common operations

### Available Scripts

| Script | Purpose | Usage |
|--------|---------|--------|
| `./scripts/setup.sh` | Initial environment setup | `./scripts/setup.sh` |
| `./scripts/test.sh` | Run comprehensive tests | `./scripts/test.sh` |
| `./scripts/deploy.sh` | Deploy application | `./scripts/deploy.sh local` |
| `./scripts/maintenance.sh` | Maintenance operations | `./scripts/maintenance.sh clean` |

### Code Quality Tools

#### Formatting
```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Combined formatting (recommended)
black . && isort .
```

#### Linting
```bash
# Check code style
flake8 sandbox/ main.py config.py

# Type checking
mypy sandbox/ --ignore-missing-imports
```

#### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sandbox --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

#### Security & Dependencies
```bash
# Security scan
bandit -r sandbox/

# Check for vulnerable dependencies
safety check
```

## 📁 Project Structure

```
data-science-sandbox/
├── 🎮 main.py                 # Main application entry point
├── ⚙️ config.py               # Configuration and game settings
├── 📦 requirements.txt        # Python dependencies
├── 🔧 setup.py               # Package setup
│
├── 🧩 sandbox/               # Core application modules
│   ├── 🎯 core/             # Game engine and dashboard
│   │   ├── game_engine.py   # Progress tracking, levels, achievements
│   │   └── dashboard.py     # Streamlit web interface
│   ├── 🏆 levels/           # Level-specific content and logic
│   ├── 🏅 achievements/     # Badge and achievement system
│   └── 🛠️ utils/            # Utility functions and helpers
│
├── 🎯 challenges/           # Coding challenges by level
│   ├── level_1/            # 🥇 Data Explorer challenges
│   ├── level_2/            # 🥈 Analytics Apprentice challenges
│   ├── level_3/            # 🥉 Visualization Virtuoso challenges
│   ├── level_4/            # 🏆 Machine Learning Novice challenges
│   ├── level_5/            # 🎖️ Algorithm Architect challenges
│   └── level_6/            # 🏅 Data Science Master challenges
│
├── 📚 notebooks/           # Interactive learning materials
│   ├── beginner/          # Level 1-2 notebooks
│   ├── intermediate/      # Level 3-4 notebooks
│   └── advanced/          # Level 5-6 notebooks
│
├── 📊 data/               # Datasets and resources
│   ├── datasets/          # Sample datasets for practice
│   ├── samples/           # Example outputs and solutions
│   └── generate_datasets.py  # Dataset generation script
│
├── 📖 docs/               # Documentation
│   ├── api/              # API reference documentation
│   ├── guides/           # User and developer guides
│   ├── examples/         # Example usage and tutorials
│   └── deployment/       # Deployment guides
│
├── 🧪 tests/             # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── fixtures/         # Test data and fixtures
│   └── conftest.py       # Pytest configuration and fixtures
│
├── 🛠️ scripts/           # Development and deployment scripts
│   ├── setup.sh          # Environment setup
│   ├── test.sh           # Testing automation
│   ├── deploy.sh         # Deployment automation
│   └── maintenance.sh    # Maintenance tasks
│
└── 🎛️ .vscode/           # VSCode configuration
    ├── settings.json     # Editor settings
    ├── launch.json       # Debug configurations
    ├── tasks.json        # Task definitions
    └── extensions.json   # Recommended extensions
```

## 🧩 Core Components

### GameEngine (`sandbox/core/game_engine.py`)
The central component that manages:
- 👤 User progress and statistics
- 🏆 Level progression and unlocking
- 🏅 Badge earning and tracking
- 💾 Progress persistence
- 📊 Statistics calculation

```python
# Example usage
from sandbox.core.game_engine import GameEngine

engine = GameEngine()
engine.complete_challenge("level_1_first_steps", score=95)
engine.add_experience(100, "Completed first challenge")
stats = engine.get_stats()
```

### Dashboard (`sandbox/core/dashboard.py`)
Streamlit-based web interface providing:
- 📊 Progress visualization
- 🎯 Challenge navigation
- 🏆 Achievement tracking
- ⚙️ Settings management

### Challenge System
Markdown-based challenges with:
- 📝 Clear instructions and objectives
- 💻 Complete, runnable code examples
- ✅ Success criteria and learning objectives
- 🎯 Progressive difficulty scaling

## 🎨 Adding New Features

### Creating New Challenges

1. **Create Challenge File**:
```bash
# Create new challenge file
touch challenges/level_X/challenge_Y_name.md
```

2. **Follow Challenge Template**:
```markdown
# Level X: [Level Name]

## Challenge Y: [Challenge Title]

Brief motivating introduction.

### Objective
Clear, specific learning objective.

### Instructions
```python
# Complete, runnable code with:
# 1. All necessary imports
# 2. Data setup
# 3. Step-by-step tasks
# 4. Expected outputs
```

### Success Criteria
- Measurable outcomes
- Skills demonstrated

### Learning Objectives
- Concepts mastered
- Skills developed

---
*Pro tip: Practical advice*
```

3. **Test the Challenge**:
```bash
# Test the code in the challenge
cd challenges/level_X/
python -c "exec(open('challenge_Y_name.md').read().split('```python')[1].split('```')[0])"
```

### Adding New Levels

1. **Update Configuration**:
```python
# In config.py
LEVELS = {
    # ... existing levels ...
    7: {
        "name": "New Level Name", 
        "description": "Level description"
    }
}
```

2. **Create Level Directory**:
```bash
mkdir challenges/level_7
mkdir notebooks/level_7
```

3. **Update GameEngine**: Modify level progression logic if needed

### Extending the Dashboard

1. **Add New Page**:
```python
# In sandbox/core/dashboard.py
def show_new_feature(self):
    """New dashboard feature."""
    st.header("🆕 New Feature")
    # Implementation here
```

2. **Update Navigation**:
```python
# Add to create_sidebar method
pages = ['Dashboard', 'Levels', 'Challenges', 'Badges', 'New Feature']
```

## 🧪 Testing Guidelines

### Test Structure
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **Fixtures**: Reusable test data in `tests/fixtures/`

### Writing Tests
```python
# tests/unit/test_new_feature.py
import pytest
from sandbox.core.game_engine import GameEngine

class TestNewFeature:
    def test_new_functionality(self, game_engine):
        """Test description."""
        # Arrange
        initial_state = game_engine.get_stats()
        
        # Act
        result = game_engine.new_method(test_data)
        
        # Assert
        assert result is not None
        assert game_engine.get_stats()['metric'] > initial_state['metric']
```

### Running Tests
```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_game_engine.py

# With coverage report
pytest --cov=sandbox --cov-report=html

# Specific test method
pytest tests/unit/test_game_engine.py::TestGameEngine::test_add_experience
```

## 📊 Data Management

### Adding New Datasets

1. **Create Dataset**:
```python
# data/generate_datasets.py
def create_new_dataset():
    """Generate new sample dataset."""
    data = pd.DataFrame({
        # Dataset structure
    })
    data.to_csv('data/datasets/new_dataset.csv', index=False)
```

2. **Update Generation Script**:
```python
# Add to main generation function
def generate_all_datasets():
    # ... existing datasets ...
    create_new_dataset()
```

3. **Document Dataset**:
```markdown
# data/datasets/README.md
## new_dataset.csv
- **Purpose**: Description
- **Columns**: Column descriptions
- **Size**: Number of rows/columns
- **Usage**: Which challenges use this data
```

## 🚀 Deployment Options

### Local Development
```bash
# CLI mode
python main.py --mode cli

# Dashboard mode
python main.py --mode dashboard

# Jupyter mode
python main.py --mode jupyter
```

### Docker Deployment
```bash
# Build image
docker build -t data-science-sandbox .

# Run container
docker run -p 8501:8501 -p 8888:8888 data-science-sandbox
```

### Production Deployment
```bash
# Using the deployment script
./scripts/deploy.sh heroku    # Deploy to Heroku
./scripts/deploy.sh aws       # Deploy to AWS (with setup)
```

## 🤝 Contributing Guidelines

### Pull Request Process
1. **Fork & Clone**: Fork the repo and create a feature branch
2. **Develop**: Make changes following code standards
3. **Test**: Ensure all tests pass and add new tests
4. **Document**: Update documentation as needed
5. **Submit**: Create PR with clear description

### Code Review Checklist
- ✅ Follows code style (Black, isort, flake8)
- ✅ Includes comprehensive tests
- ✅ Updates documentation
- ✅ Maintains backward compatibility
- ✅ Includes appropriate error handling

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-amazing-feature

# Make changes and commit
git add .
git commit -m "Add new amazing feature"

# Run tests before pushing
./scripts/test.sh

# Push and create PR
git push origin feature/new-amazing-feature
```

## 🐛 Debugging Tips

### Common Issues

1. **Import Errors**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Check Python path
python -c "import sys; print(sys.path)"
```

2. **Test Failures**:
```bash
# Run with verbose output
pytest -v

# Run specific failing test
pytest tests/unit/test_failing.py::test_method -s
```

3. **Dashboard Issues**:
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with debugging
streamlit run --logger.level debug main.py -- --mode dashboard
```

### Performance Profiling
```python
# Profile specific functions
import cProfile
cProfile.run('your_function_call()', 'profile_output.prof')

# Memory profiling
from memory_profiler import profile
@profile
def your_function():
    # Function code
```

## 📚 Additional Resources

- 📖 [API Documentation](docs/api/)
- 🎯 [Challenge Creation Guide](docs/guides/creating-challenges.md)
- 🚀 [Deployment Guide](docs/deployment/)
- 🧪 [Testing Best Practices](docs/guides/testing.md)
- 🎨 [UI/UX Guidelines](docs/guides/ui-guidelines.md)

## 💬 Getting Help

- 🐛 **Issues**: Report bugs via GitHub Issues
- 💡 **Features**: Request features via GitHub Discussions
- 📧 **Contact**: Reach out to maintainers
- 📖 **Documentation**: Check docs/ directory first

---

**Happy developing! 🎉** Remember, this is an educational platform - every change should enhance the learning experience!