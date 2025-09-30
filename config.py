"""
Configuration settings for the Data Science Sandbox
"""

import os

# Base configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
CHALLENGES_DIR = os.path.join(BASE_DIR, "challenges")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

# Game Configuration
LEVELS = {
    1: {
        "name": "Data Explorer",
        "description": "Learn basic data manipulation and visualization",
    },
    2: {
        "name": "Analytics Apprentice",
        "description": "Master statistical analysis and data cleaning",
    },
    3: {
        "name": "Visualization Virtuoso",
        "description": "Create stunning data visualizations",
    },
    4: {"name": "Machine Learning Novice", "description": "Build your first ML models"},
    5: {
        "name": "Algorithm Architect",
        "description": "Advanced ML algorithms and optimization",
    },
    6: {
        "name": "Data Science Master",
        "description": "Complex projects and real-world challenges",
    },
    7: {
        "name": "Modern Toolchain Master",
        "description": "Master cutting-edge tools, MLOps, and ethical AI",
    },
}

# Achievement badges
BADGES = {
    "first_steps": {
        "name": "First Steps",
        "description": "Complete your first challenge",
    },
    "data_cleaner": {"name": "Data Cleaner", "description": "Clean a messy dataset"},
    "viz_master": {
        "name": "Visualization Master",
        "description": "Create 5 different chart types",
    },
    "stats_guru": {
        "name": "Statistics Guru",
        "description": "Complete statistical analysis challenges",
    },
    "ml_rookie": {
        "name": "ML Rookie",
        "description": "Build your first machine learning model",
    },
    "model_optimizer": {
        "name": "Model Optimizer",
        "description": "Improve model performance by 10%",
    },
    "data_storyteller": {
        "name": "Data Storyteller",
        "description": "Create a complete data story",
    },
    "problem_solver": {
        "name": "Problem Solver",
        "description": "Complete all challenges in a level",
    },
}

# Challenge categories
CATEGORIES = {
    "data_exploration": "Data Exploration & Understanding",
    "data_cleaning": "Data Cleaning & Preprocessing",
    "visualization": "Data Visualization",
    "statistics": "Statistical Analysis",
    "machine_learning": "Machine Learning",
    "deep_learning": "Deep Learning",
    "time_series": "Time Series Analysis",
    "nlp": "Natural Language Processing",
    "computer_vision": "Computer Vision",
    "real_world": "Real-World Projects",
}
