#!/usr/bin/env python3
"""
Modern iOS HIG-Compliant Streamlit App Entry Point
Professional redesign following Apple's Human Interface Guidelines
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sandbox.core.game_engine import GameEngine
from sandbox.ui.modern_ios_dashboard import ModernIOSDashboard

# Initialize game engine
game = GameEngine()

# Create and run modern dashboard
dashboard = ModernIOSDashboard(game)
dashboard.run()
