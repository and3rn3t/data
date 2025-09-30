#!/usr/bin/env python3
"""
Streamlit app entry point for Data Science Sandbox Dashboard
This file is launched by streamlit run to avoid ScriptRunContext issues
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sandbox.core.dashboard import Dashboard
from sandbox.core.game_engine import GameEngine

# Initialize game engine
game = GameEngine()

# Create and run dashboard
dashboard = Dashboard(game)
dashboard.run()
