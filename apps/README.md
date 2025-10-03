# Applications

This directory contains the main application entry points:

## Streamlit Applications

- `streamlit_app.py` - Original Streamlit dashboard implementation
- `streamlit_app_modern.py` - Modern iOS HIG-compliant dashboard with enhanced UI/UX

## Running the Applications

### Modern Dashboard (Recommended)

```bash
python -m streamlit run apps/streamlit_app_modern.py --server.port=8502
```

### Original Dashboard

```bash
python -m streamlit run apps/streamlit_app.py --server.port=8501
```

## Features

The modern dashboard includes:

- iOS Human Interface Guidelines compliance
- Enhanced gamification features
- Professional data visualization
- Responsive design
- Improved user experience

Use the task runner for convenience:

```bash
# Using VS Code tasks
Ctrl+Shift+P -> "Tasks: Run Task" -> "Start Modern iOS Dashboard"
```
