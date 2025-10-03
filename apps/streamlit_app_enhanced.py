#!/usr/bin/env python3
"""
Enhanced Streamlit App with Gamification Features
Integrates achievement system, auto-validation, and analytics
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sandbox.ui.enhanced_gamification_dashboard import create_gamification_dashboard

# Configure Streamlit page
st.set_page_config(
    page_title="Data Science Sandbox - Enhanced",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create enhanced gamification dashboard
if __name__ == "__main__":
    create_gamification_dashboard()
