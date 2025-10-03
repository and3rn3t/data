"""
Web Dashboard for Data Science Sandbox
Interactive interface for progress tracking and challenge management
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import BADGES, LEVELS
from sandbox.core.game_engine import GameEngine
from sandbox.utils.dashboard_layout_system import DashboardLayoutSystem


class Dashboard:
    """Streamlit-based dashboard for the data science sandbox"""

    # Constants for commonly used strings
    TRANSPARENT_BG = "rgba(0,0,0,0)"
    GRID_COLOR = "rgba(142, 142, 147, 0.2)"
    SF_FONT = "SF Pro Display, -apple-system, sans-serif"
    IOS_BLUE = "var(--ios-blue)"
    IOS_GREEN = "var(--ios-green)"
    IOS_PURPLE = "var(--ios-purple)"
    IOS_ORANGE = "var(--ios-orange)"
    IOS_RED = "var(--ios-red)"
    DEFAULT_TIME_STR = "20 minutes"
    CLOSE_DIV = "</div></div>"

    def __init__(self, game_engine: GameEngine):
        self.game = game_engine

    def _get_theme_colors(self) -> dict:
        """Get appropriate colors based on theme detection"""
        # Check if user has set a theme preference in session state
        is_dark_theme = st.session_state.get(
            "dark_theme", True
        )  # Default to dark for better contrast

        if is_dark_theme:
            return {
                "text_primary": "#FFFFFF",  # White text for dark backgrounds
                "text_secondary": "#AEAEB2",  # Light gray for secondary text
                "surface_primary": "rgba(28, 28, 30, 0.85)",  # Dark surface
                "surface_secondary": "rgba(44, 44, 46, 0.8)",  # Darker surface
                "ios_blue": "#1D4ED8",
                "ios_orange": "#C2410C",
                "ios_purple": "#7C3AED",
            }
        else:
            return {
                "text_primary": "#000000",  # Black text for light backgrounds
                "text_secondary": "#6B7280",  # Dark gray for secondary text
                "surface_primary": "rgba(255, 255, 255, 0.85)",  # Light surface
                "surface_secondary": "rgba(242, 242, 247, 0.8)",  # Light gray surface
                "ios_blue": "#1D4ED8",
                "ios_orange": "#C2410C",
                "ios_purple": "#7C3AED",
            }

    def run(self) -> None:
        """Launch the Streamlit dashboard"""
        # Configure Streamlit page
        st.set_page_config(
            page_title="Data Science Sandbox",
            page_icon="🎮",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # iOS 26 HIG Inspired Custom CSS
        st.markdown(
            """
        <style>
        /* iOS 26 HIG Design System */
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@200;300;400;500;600;700&display=swap');

        :root {
            --ios-blue: #1D4ED8;
            --ios-gray: #6B7280;
            --ios-gray-light: #F9FAFB;
            --ios-gray-dark: #374151;
            --ios-green: #047857;
            --ios-orange: #C2410C;
            --ios-red: #DC2626;
            --ios-purple: #7C3AED;
            --ios-pink: #EC4899;
            --ios-teal: #06B6D4;
            --ios-indigo: #6366F1;
            --ios-gold: #F59E0B;
            --ios-yellow: #EAB308;

            --surface-primary: rgba(255, 255, 255, 0.85);
            --surface-secondary: rgba(242, 242, 247, 0.8);
            --surface-tertiary: rgba(255, 255, 255, 0.6);
            --text-primary: #000000;
            --text-secondary: #6B7280;
            --shadow-light: 0 1px 3px rgba(0, 0, 0, 0.1);
            --shadow-medium: 0 4px 12px rgba(0, 0, 0, 0.15);
            --shadow-heavy: 0 8px 25px rgba(0, 0, 0, 0.2);
        }

        /* Dark theme adjustments */
        [data-theme="dark"], .stApp[data-theme="dark"] {
            --surface-primary: rgba(28, 28, 30, 0.85);
            --surface-secondary: rgba(44, 44, 46, 0.8);
            --surface-tertiary: rgba(58, 58, 60, 0.6);
            --text-primary: #FFFFFF;
            --text-secondary: #AEAEB2;
        }

        /* Auto-detect dark mode from Streamlit */
        @media (prefers-color-scheme: dark) {
            :root {
                --surface-primary: rgba(28, 28, 30, 0.85);
                --surface-secondary: rgba(44, 44, 46, 0.8);
                --surface-tertiary: rgba(58, 58, 60, 0.6);
                --text-primary: #FFFFFF;
                --text-secondary: #AEAEB2;
            }
        }

        /* Global Styles */
        .main > div {
            padding-top: 1rem;
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* iOS Card System */
        .ios-card {
            background: var(--surface-primary);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 20px;
            margin: 16px 0;
            box-shadow: var(--shadow-medium);
            border: 0.5px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            z-index: 1;
            overflow: hidden;
        }

        .ios-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-heavy);
        }

        /* Enhanced Metrics Cards */
        .stMetric {
            background: var(--surface-primary);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 24px;
            box-shadow: var(--shadow-medium);
            border: 0.5px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        .stMetric:hover {
            transform: scale(1.02);
            box-shadow: var(--shadow-heavy);
        }

        .stMetric label {
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stMetric div[data-testid="metric-value"] {
            color: var(--text-primary);
            font-size: 32px;
            font-weight: 700;
        }

        /* Level Cards with iOS Island Design */
        .level-card {
            background: linear-gradient(135deg, var(--ios-blue) 0%, var(--ios-purple) 100%);
            border-radius: 24px;
            padding: 28px;
            color: white;
            margin: 20px 0;
            box-shadow: var(--shadow-heavy);
            position: relative;
            z-index: 2;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        .level-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);
            pointer-events: none;
        }

        .level-card:hover {
            transform: scale(1.02) translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 122, 255, 0.3);
        }

        /* Achievement Badge System */
        .badge-card {
            background: var(--surface-primary);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 20px;
            margin: 12px;
            box-shadow: var(--shadow-medium);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .badge-card.earned {
            background: linear-gradient(135deg, var(--ios-green) 0%, var(--ios-teal) 100%);
            color: white;
            box-shadow: 0 8px 25px rgba(48, 209, 88, 0.3);
        }

        .badge-card:hover {
            transform: translateY(-4px) scale(1.05);
            box-shadow: var(--shadow-heavy);
        }

        /* Interactive Buttons */
        .stButton > button {
            background: var(--ios-blue);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            box-shadow: var(--shadow-medium);
        }

        .stButton > button:hover {
            background: #1E40AF;
            transform: translateY(-2px);
            box-shadow: var(--shadow-heavy);
        }

        .stButton > button:active {
            transform: scale(0.95);
        }

        /* Sidebar Styling */
        .css-1d391kg {
            background: var(--surface-secondary);
            backdrop-filter: blur(20px);
        }

        /* Navigation Pills */
        .nav-pill {
            background: var(--surface-tertiary);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 12px 20px;
            margin: 8px 0;
            border: none;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            width: 100%;
            text-align: left;
            font-weight: 500;
        }

        .nav-pill:hover {
            background: var(--ios-blue);
            color: white;
            transform: translateX(4px);
            box-shadow: var(--shadow-medium);
        }

        /* Progress Bars */
        .progress-ring {
            width: 60px;
            height: 60px;
            transform: rotate(-90deg);
        }

        .progress-ring-circle {
            stroke: var(--ios-blue);
            stroke-dasharray: 157;
            stroke-dashoffset: 0;
            fill: transparent;
            stroke-width: 4;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.5s ease;
        }

        /* Dynamic Island Style Header */
        .dynamic-island {
            background: var(--ios-gray-dark);
            color: white;
            border-radius: 28px;
            padding: 16px 32px;
            margin: 20px auto;
            text-align: center;
            box-shadow: var(--shadow-heavy);
            backdrop-filter: blur(20px);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            max-width: fit-content;
        }

        .dynamic-island:hover {
            transform: scale(1.05);
            border-radius: 36px;
        }

        /* Glassmorphism Effects */
        .glass-panel {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }

        /* Activity Indicators */
        .activity-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: var(--ios-green);
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }

        /* Chart Enhancements */
        .js-plotly-plot {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: var(--shadow-medium);
        }

        /* Responsive Typography */
        h1, h2, h3 {
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }

        h1 { font-size: 2.5rem; }
        h2 { font-size: 2rem; }
        h3 { font-size: 1.5rem; }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--surface-secondary);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--ios-gray);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--ios-blue);
        }

        /* Animations */
        @keyframes slideInUp {
            from {
                transform: translateY(20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        .animate-slide-in {
            animation: slideInUp 0.5s ease-out;
        }

        /* iOS 26 SF Symbols System */
        .sf-symbol {
            display: inline-block;
            width: 18px;
            height: 18px;
            position: relative;
            transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
        }

        .sf-symbol svg {
            width: 100%;
            height: 100%;
            fill: currentColor;
            transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
        }

        /* SF Symbol Animations */
        @keyframes sf-pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }

        @keyframes sf-bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-3px); }
        }

        @keyframes sf-rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes sf-wiggle {
            0%, 100% { transform: rotate(0deg); }
            25% { transform: rotate(-3deg); }
            75% { transform: rotate(3deg); }
        }

        /* Hover Animations */
        .sf-symbol:hover {
            transform: scale(1.05);
        }

        .sf-symbol.sf-pulse:hover {
            animation: sf-pulse 1s ease-in-out infinite;
        }

        .sf-symbol.sf-bounce:hover {
            animation: sf-bounce 0.6s ease-in-out infinite;
        }

        .sf-symbol.sf-rotate:hover {
            animation: sf-rotate 2s linear infinite;
        }

        .sf-symbol.sf-wiggle:hover {
            animation: sf-wiggle 0.5s ease-in-out infinite;
        }

        /* Icon Base Classes */
        .icon-dashboard, .icon-levels, .icon-challenges, .icon-badges, .icon-progress,
        .icon-settings, .icon-focus, .icon-fire, .icon-actions, .icon-trophy,
        .icon-star, .icon-medal, .icon-chart, .icon-check, .icon-clock,
        .icon-rocket, .icon-book, .icon-stats {
            display: inline-block;
            width: 18px;
            height: 18px;
            position: relative;
        }

        .icon-dashboard::before {
            content: '▦';
            position: absolute;
            font-size: 16px;
            font-weight: normal;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: currentColor;
        }

        .icon-levels::before {
            content: '📊';
            position: absolute;
            font-size: 14px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .icon-challenges::before {
            content: '🎯';
            position: absolute;
            font-size: 14px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .icon-badges::before {
            content: '🏆';
            position: absolute;
            font-size: 14px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .icon-progress::before {
            content: '⏱️';
            position: absolute;
            font-size: 14px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .icon-settings::before {
            content: '⚙️';
            position: absolute;
            font-size: 14px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .icon-trophy::before {
            content: '🏅';
            position: absolute;
            font-size: 14px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .icon-star::before {
            content: '★';
            position: absolute;
            font-size: 16px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: currentColor;
        }

        .icon-fire::before {
            content: '';
            position: absolute;
            width: 8px;
            height: 12px;
            background: currentColor;
            border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
            top: 2px;
            left: 50%;
            transform: translateX(-50%);
        }

        .icon-check::before {
            content: '✓';
            position: absolute;
            font-size: 14px;
            font-weight: bold;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: currentColor;
        }

        .icon-rocket::before {
            content: '🚀';
            position: absolute;
            font-size: 14px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .icon-book::before {
            content: '📚';
            position: absolute;
            font-size: 14px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        /* Legacy icon classes for backward compatibility */
        .icon-focus, .icon-fire, .icon-actions, .icon-medal,
        .icon-chart, .icon-check, .icon-clock, .icon-stats {
            display: inline-block;
            width: 18px;
            height: 18px;
            position: relative;
        }

        .icon-focus::before, .icon-fire::before, .icon-actions::before,
        .icon-medal::before, .icon-chart::before, .icon-check::before,
        .icon-clock::before, .icon-stats::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            background-image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 18 18'><circle cx='9' cy='9' r='6' fill='currentColor'/></svg>");
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
        }

        /* SF Symbol Navigation Buttons */
        .nav-button {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            margin: 4px 0;
            background: var(--surface-secondary);
            border: none;
            border-radius: 12px;
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
            font-size: 14px;
            font-weight: 500;
        }

        .nav-button:hover {
            background: var(--accent-primary);
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .nav-button .sf-symbol {
            margin-right: 8px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # iOS Dynamic Island Style Header
        st.markdown(
            """
        <div class="dynamic-island animate-slide-in">
            <div style="display: flex; align-items: center; justify-content: center;">
                <div class="sf-symbol sf-pulse" style="margin-right: 12px;">
                    <div class="icon-dashboard"></div>
                </div>
                <div>
                    <h1 style="margin: 0; font-size: 1.5rem;">Data Science Sandbox</h1>
                    <p style="margin: 4px 0 0 0; opacity: 0.7; font-size: 0.9rem;">Learn data science through gamified challenges</p>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Interactive status indicator
        stats = self.game.get_stats()
        st.markdown(
            f"""
        <div style="text-align: center; margin: 20px 0;">
            <span class="activity-dot"></span>
            <span style="color: var(--ios-blue); font-weight: 600;">Level {stats['level']}</span>
            <span style="margin: 0 12px;">•</span>
            <span style="color: var(--ios-green); font-weight: 600;">{stats['experience']} XP</span>
            <span style="margin: 0 12px;">•</span>
            <span style="color: var(--ios-purple); font-weight: 600;">{stats['badges']} Badges</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Sidebar navigation
        self.create_sidebar()

        # Check for challenge modal display
        if st.session_state.get("show_challenge_modal", False):
            self.show_challenge_modal()
            return

        # Main content area
        page = st.session_state.get("page", "Dashboard")

        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Levels":
            self.show_levels()
        elif page == "Challenges":
            self.show_challenges()
        elif page == "Badges":
            self.show_badges()
        elif page == "Progress":
            self.show_progress()
        elif page == "Settings":
            self.show_settings()

    def create_sidebar(self) -> None:
        """Create iOS-style sidebar navigation"""
        # Player Profile Card
        stats = self.game.get_stats()
        st.sidebar.markdown(
            f"""
        <div class="ios-card" style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">👤</div>
            <h3 style="margin: 8px 0; color: var(--text-primary);">{self.game.progress['player_name']}</h3>
            <div style="display: flex; justify-content: space-around; margin-top: 16px;">
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-blue);">{stats['level']}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">Level</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-green);">{stats['experience']}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">XP</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-purple);">{stats['badges']}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">Badges</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Navigation Pills
        st.sidebar.markdown(
            "<h4 style='margin: 20px 0 16px 0; color: var(--text-secondary);'>🧭 Navigation</h4>",
            unsafe_allow_html=True,
        )

        # Navigation with SF Symbols
        nav_items = [
            ("Dashboard", "dashboard", "sf-pulse", "Overview and progress"),
            ("Levels", "levels", "sf-bounce", "Learning progression"),
            ("Challenges", "challenges", "sf-wiggle", "Practice exercises"),
            ("Badges", "badges", "sf-pulse", "Achievements"),
            ("Progress", "progress", "sf-rotate", "Analytics"),
            ("Settings", "settings", "sf-rotate", "Configuration"),
        ]

        for page, icon_name, animation, description in nav_items:
            # Create navigation button with icon
            icon_map = {
                "dashboard": "▦",
                "levels": "📊",
                "challenges": "🎯",
                "badges": "🏆",
                "progress": "⏱️",
                "settings": "⚙️",
            }
            icon = icon_map.get(icon_name, "●")

            if st.sidebar.button(f"{icon} {page}", key=page, help=description):
                st.session_state.page = page
                # Clear any open modals when navigating
                st.session_state.show_challenge_modal = False
                st.session_state.viewing_challenge = None
                st.rerun()

        st.sidebar.markdown("---")

        # Theme Toggle
        st.sidebar.markdown(
            "<h4 style='margin: 20px 0 16px 0; color: var(--text-secondary);'>🎨 Theme</h4>",
            unsafe_allow_html=True,
        )
        current_theme = st.session_state.get("dark_theme", True)
        theme_label = "🌙 Dark Mode" if current_theme else "☀️ Light Mode"
        if st.sidebar.button(
            theme_label, key="theme_toggle", help="Toggle between light and dark themes"
        ):
            st.session_state.dark_theme = not current_theme
            st.rerun()

        # Quick Actions with enhanced styling
        st.sidebar.markdown(
            "<h4 style='margin: 20px 0 16px 0; color: var(--text-secondary);'>⚡ Quick Actions</h4>",
            unsafe_allow_html=True,
        )

        # Action buttons with icons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶", key="jupyter", help="Launch Jupyter Lab"):
                self.game.launch_jupyter()
                st.sidebar.success("→ Launching...")

        with col2:
            if st.button("↻", key="reset", help="Reset Progress"):
                if st.sidebar.button("!", key="confirm_reset", help="Confirm Reset"):
                    self.game.reset_progress()
                    st.sidebar.success("✓ Reset complete!")
                    st.rerun()

        # Progress Ring Visualization
        completion = stats.get("completion_rate", 0)
        st.sidebar.markdown(
            f"""
        <div style="text-align: center; margin: 20px 0;">
            <h4 style='color: var(--text-secondary); margin-bottom: 16px;'>Overall Progress</h4>
            <div style="position: relative; display: inline-block;">
                <svg class="progress-ring" width="80" height="80">
                    <circle class="progress-ring-circle"
                            cx="40" cy="40" r="25"
                            style="stroke-dashoffset: {157 - (157 * completion / 100)};"></circle>
                </svg>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                           font-weight: 700; color: var(--ios-blue);">{completion:.0f}%</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Color Accessibility Status Widget
        try:
            from sandbox.utils.dashboard_contrast_integration import (
                render_contrast_widget,
            )

            render_contrast_widget()
        except ImportError:
            pass

        # Dark Mode Toggle
        self.render_dark_mode_toggle()

    def show_dashboard(self) -> None:
        """iOS-inspired main dashboard view with organized sections"""
        stats = self.game.get_stats()

        # Page Header
        DashboardLayoutSystem.create_page_header(
            title="Learning Dashboard",
            subtitle="Track your progress and continue your data science journey",
            icon="■",
        )

        # Study Session Timer Section
        self.render_study_timer()

        # Key Metrics Section
        DashboardLayoutSystem.create_section_header("→ Your Progress")

        completion = stats.get("completion_rate", 0)
        metrics = [
            {
                "icon": "▲",
                "value": str(stats["level"]),
                "label": "Current Level",
                "sublabel": "out of 7",
                "color": "var(--ios-blue)",
            },
            {
                "icon": "★",
                "value": str(stats["experience"]),
                "label": "Experience",
                "sublabel": "points earned",
                "color": "var(--ios-green)",
                "delta": "+50 recent",
                "delta_color": "var(--ios-green)",
            },
            {
                "icon": "◆",
                "value": str(stats["badges"]),
                "label": "Badges Earned",
                "sublabel": "achievements",
                "color": "var(--ios-purple)",
            },
            {
                "icon": "→",
                "value": f"{completion:.1f}%",
                "label": "Completion",
                "sublabel": "overall progress",
                "color": "var(--ios-orange)",
            },
        ]

        DashboardLayoutSystem.create_metric_cards(metrics)

        # Structured Content Sections
        self.render_gamification_streak_tracker()
        self._render_learning_progression_section(stats)
        self._render_activity_and_actions_section()

    def _render_learning_progression_section(self, stats):
        """Render organized learning progression section with chart and current focus"""

        def render_progress_chart():
            """Render progress chart for all levels"""
            import plotly.express as px

            progress_data = []
            for level in range(1, 8):
                level_status = self.game.progress["level_progress"][str(level)]

                # Calculate actual progress
                if level_status["completed"]:
                    status = "Completed"
                    progress = 100
                elif level == stats["level"]:
                    total_challenges = len(level_status.get("challenges", []))
                    completed_challenges = len(
                        [
                            c
                            for c in level_status.get("challenges", [])
                            if c in self.game.progress["challenges_completed"]
                        ]
                    )
                    progress = (
                        (completed_challenges / total_challenges * 100)
                        if total_challenges > 0
                        else 0
                    )
                    status = "Active"
                else:
                    status = "Locked"
                    progress = 0

                progress_data.append(
                    {
                        "Level": f"Level {level}",
                        "Name": LEVELS[level]["name"],
                        "Progress": progress,
                        "Status": status,
                    }
                )

            fig = px.bar(
                progress_data,
                x="Level",
                y="Progress",
                color="Status",
                color_discrete_map={
                    "Completed": "#047857",
                    "Active": "#1D4ED8",
                    "Locked": "#9CA3AF",
                },
                hover_data=["Name"],
            )

            fig.update_layout(
                plot_bgcolor=Dashboard.TRANSPARENT_BG,
                paper_bgcolor=Dashboard.TRANSPARENT_BG,
                font={"family": Dashboard.SF_FONT},
                showlegend=True,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                margin={"t": 50, "b": 50, "l": 50, "r": 50},
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(142, 142, 147, 0.2)")

            st.plotly_chart(fig, use_container_width=True)

        def render_current_focus():
            """Render current level focus and available challenges"""
            current_level = stats["level"]

            if current_level <= 7:
                level_info = LEVELS[current_level]

                # Current Level Card
                st.markdown(
                    '<h3 style="margin-bottom: 16px;">● Current Focus</h3>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="level-card">
                        <h3 style="margin: 0 0 12px 0; font-size: 1.3rem;">Level {current_level}: {level_info['name']}</h3>
                        <p style="margin: 0; opacity: 0.8; line-height: 1.4;">{level_info['description']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Available Challenges
                enhanced_challenges = self.game.get_enhanced_challenges(current_level)
                if enhanced_challenges:
                    st.markdown(
                        '<h3 style="margin: 24px 0 16px 0;">◆ Available Challenges</h3>',
                        unsafe_allow_html=True,
                    )

                    for i, challenge_data in enumerate(enhanced_challenges[:3]):
                        challenge_name = (
                            challenge_data.get("title", f"Challenge {i+1}")
                            if isinstance(challenge_data, dict)
                            else str(challenge_data)
                        )
                        completed = (
                            f"level_{current_level}_{challenge_name}"
                            in self.game.progress["challenges_completed"]
                        )

                        # Get metadata
                        difficulty = (
                            challenge_data.get("difficulty", "Unknown")
                            if isinstance(challenge_data, dict)
                            else "Unknown"
                        )
                        estimated_time = (
                            challenge_data.get("estimated_time", 20)
                            if isinstance(challenge_data, dict)
                            else 20
                        )

                        # Parse time
                        try:
                            if isinstance(estimated_time, str):
                                estimated_time = int(
                                    "".join(filter(str.isdigit, estimated_time)) or 20
                                )
                            time_str = (
                                f"{estimated_time // 60}h {estimated_time % 60}m"
                                if estimated_time > 60
                                else f"{estimated_time}m"
                            )
                        except:
                            time_str = "~20m"

                        icon = "✓" if completed else "○"
                        status_color = (
                            Dashboard.IOS_GREEN if completed else Dashboard.IOS_BLUE
                        )

                        difficulty_colors = {
                            "Beginner": Dashboard.IOS_GREEN,
                            "Easy": Dashboard.IOS_GREEN,
                            "Intermediate": Dashboard.IOS_ORANGE,
                            "Medium": Dashboard.IOS_ORANGE,
                            "Advanced": Dashboard.IOS_RED,
                            "Hard": Dashboard.IOS_RED,
                            "Expert": Dashboard.IOS_PURPLE,
                        }
                        difficulty_color = difficulty_colors.get(
                            difficulty, Dashboard.IOS_BLUE
                        )

                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; padding: 16px; margin: 12px 0;
                                       background: var(--surface-primary); border-radius: 16px;
                                       border: 1px solid rgba(255, 255, 255, 0.1);
                                       box-shadow: var(--shadow-light);
                                       position: relative; z-index: 1;">
                                <span style="font-size: 1.2rem; margin-right: 16px; color: {status_color};">{icon}</span>
                                <div style="flex: 1;">
                                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                                        <div style="font-weight: 600; color: var(--text-primary); font-size: 0.95rem;">{challenge_name}</div>
                                        <span style="background: {difficulty_color}; color: white; font-size: 0.7rem;
                                              padding: 3px 8px; border-radius: 10px; font-weight: 500;">{difficulty}</span>
                                    </div>
                                    <div style="font-size: 0.85rem; color: var(--text-secondary); opacity: 0.8;">{time_str}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # Render using layout system
        DashboardLayoutSystem.create_section_header("■ Learning Progression")
        DashboardLayoutSystem.create_two_column_layout(
            render_progress_chart, render_current_focus
        )

    def _render_activity_and_actions_section(self):
        """Render organized recent activity and quick actions section"""
        DashboardLayoutSystem.create_section_header("⚡ Activity & Actions")

        def render_recent_badges():
            """Render recent badges earned"""
            recent_badges = (
                self.game.progress["badges_earned"][-3:]
                if self.game.progress["badges_earned"]
                else []
            )

            if recent_badges:
                for badge_id in recent_badges:
                    if badge_id in BADGES:
                        badge = BADGES[badge_id]
                        st.markdown(
                            f"""
                            <div class="badge-card earned" style="margin-bottom: 12px;">
                                <div style="font-size: 2rem; margin-bottom: 8px; color: var(--ios-gold);">◆</div>
                                <div style="font-weight: 600; font-size: 1rem;">{badge['name']}</div>
                                <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{badge['description']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 40px;">
                        <div style="font-size: 3rem; margin-bottom: 16px; color: var(--ios-blue);">●</div>
                        <h4 style="margin-bottom: 8px;">Start Your Journey</h4>
                        <p style="color: var(--text-secondary); margin: 0;">Complete challenges to earn badges!</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        def render_learning_stats():
            """Render learning statistics"""
            streak_days = 5
            total_challenges = len(self.game.progress["challenges_completed"])

            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 16px;">
                    <div style="font-size: 2.5rem; margin-bottom: 12px; color: var(--ios-orange);">▲</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-orange);">{streak_days}</div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary);">Day Streak</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 12px; color: var(--ios-teal);">■</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-teal);">{total_challenges}</div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary);">Challenges Done</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def render_quick_actions():
            """Render quick action buttons"""
            # Add specific CSS to ensure uniform button sizing
            st.markdown(
                """
                <style>
                div[data-testid="column"] .stButton > button {
                    width: 100% !important;
                    height: 48px !important;
                    font-size: 14px !important;
                    font-weight: 600 !important;
                    padding: 12px 16px !important;
                    background: var(--ios-blue) !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 12px !important;
                }
                div[data-testid="column"] .stButton > button:hover {
                    background: #1E40AF !important;
                    transform: translateY(-2px) !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("▶ Continue Learning", key="continue_learning"):
                    st.session_state.page = "Challenges"
                    st.rerun()

            with col2:
                if st.button("→ View Progress", key="view_progress"):
                    st.session_state.page = "Progress"
                    st.rerun()

            with col3:
                if st.button("▲ Check Levels", key="check_levels"):
                    st.session_state.page = "Levels"
                    st.rerun()

            return ""  # Buttons are rendered directly

        # Create three-column layout for activity section
        def render_left_activity():
            st.markdown(
                '<h3 style="margin-bottom: 16px;">◆ Latest Badges</h3>',
                unsafe_allow_html=True,
            )
            render_recent_badges()

        def render_center_activity():
            st.markdown(
                '<h3 style="margin-bottom: 16px;">→ Learning Stats</h3>',
                unsafe_allow_html=True,
            )
            render_learning_stats()

        def render_right_activity():
            st.markdown(
                '<h3 style="margin-bottom: 16px;">▶ Quick Actions</h3>',
                unsafe_allow_html=True,
            )
            render_quick_actions()

        # Three column layout for activity section
        col1, col2, col3 = st.columns(3)
        with col1:
            render_left_activity()
        with col2:
            render_center_activity()
        with col3:
            render_right_activity()

    def show_levels(self) -> None:
        """iOS-inspired levels overview page"""
        st.markdown(
            """
        <div class="ios-card animate-slide-in">
            <h2 style="margin-bottom: 8px;">🏆 Learning Levels</h2>
            <p style="color: var(--text-secondary); margin: 0;">Progress through structured levels to master data science concepts</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        for level_num, level_info in LEVELS.items():
            status = self.game.progress["level_progress"][str(level_num)]

            # Determine status styling
            if status["completed"]:
                status_color = Dashboard.IOS_GREEN
                status_icon = "✅"
                status_text = "Completed"
                card_class = "completed"
            elif status["unlocked"]:
                status_color = Dashboard.IOS_BLUE
                status_icon = "🔓"
                status_text = "Active"
                card_class = "active"
            else:
                status_color = "var(--ios-gray)"
                status_icon = "🔒"
                status_text = "Locked"
                card_class = "locked"

            # Level card with enhanced styling
            col1, col2, col3 = st.columns([1, 4, 1])

            with col1:
                st.markdown(
                    f"""
                <div class="ios-card" style="text-align: center; height: 100px; display: flex;
                           flex-direction: column; justify-content: center;">
                    <div style="font-size: 2rem; margin-bottom: 8px; color: {status_color};">{status_icon}</div>
                    <div style="font-size: 0.9rem; color: {status_color}; font-weight: 600;">{status_text}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                # Calculate completion percentage
                if status["unlocked"]:
                    enhanced_challenges = self.game.get_enhanced_challenges(level_num)
                    if enhanced_challenges:
                        # Extract challenge titles from enhanced data
                        challenges = [
                            ch.get("title", "Unknown Challenge")
                            for ch in enhanced_challenges
                        ]
                        completed_count = len(
                            [
                                c
                                for c in self.game.progress["challenges_completed"]
                                if c.startswith(f"level_{level_num}_")
                            ]
                        )
                        completion_pct = (completed_count / len(challenges)) * 100
                    else:
                        completion_pct = 0
                else:
                    completion_pct = 100 if status["completed"] else 0

                # Get theme colors
                colors = self._get_theme_colors()

                # Extract values to avoid complex f-string interpolation
                text_primary = colors["text_primary"]
                text_secondary = colors["text_secondary"]
                level_name = level_info["name"]
                level_desc = level_info["description"]

                # Use clean Streamlit container without custom CSS
                with st.container():
                    # Put percentage inline with the level title
                    st.markdown(
                        f"### Level {level_num}: {level_name} - {completion_pct:.0f}% Complete"
                    )
                    st.markdown(level_desc)

                    # Progress bar using Streamlit's native progress bar
                    st.progress(completion_pct / 100.0)

                # Show challenges info
                if status["unlocked"]:
                    enhanced_challenges = self.game.get_enhanced_challenges(level_num)
                    if enhanced_challenges:
                        # Extract challenge titles from enhanced data
                        challenges = [
                            ch.get("title", "Unknown Challenge")
                            for ch in enhanced_challenges
                        ]
                        completed_count = len(
                            [
                                c
                                for c in self.game.progress["challenges_completed"]
                                if c.startswith(f"level_{level_num}_")
                            ]
                        )

                        st.markdown(
                            f"""
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center;">
                                <span style="font-size: 1.2rem; margin-right: 8px;">📚</span>
                                <span style="font-size: 0.9rem; color: {text_secondary};">
                                    {completed_count}/{len(challenges)} challenges completed
                                </span>
                            </div>
                            <div style="display: flex; gap: 8px;">
                        """,
                            unsafe_allow_html=True,
                        )

                        # Get completed challenges for this level
                        level_completed = [
                            c
                            for c in self.game.progress["challenges_completed"]
                            if c.startswith(f"level_{level_num}_")
                        ]

                        # Display challenge list with status indicators
                        for challenge in challenges[:5]:
                            # Check if this specific challenge is completed
                            challenge_id = f"level_{level_num}_{challenge.lower().replace(' ', '_')}"
                            is_completed = (
                                challenge_id
                                in self.game.progress["challenges_completed"]
                            )

                            # Use a more flexible matching for legacy IDs
                            if not is_completed:
                                # Try partial matching for challenges with different naming
                                challenge_partial = challenge.lower().replace(" ", "_")
                                is_completed = any(
                                    challenge_partial in completed_id
                                    for completed_id in level_completed
                                )

                            status_icon = "✅" if is_completed else "○"
                            status_color = (
                                Dashboard.IOS_GREEN if is_completed else text_secondary
                            )

                            st.markdown(
                                f"""
                                <div style="display: flex; align-items: center; margin: 4px 0;">
                                    <span style="color: {status_color}; margin-right: 8px; font-size: 0.9rem;">{status_icon}</span>
                                    <span style="color: {text_primary}; font-size: 0.85rem;">{challenge}</span>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        st.markdown(Dashboard.CLOSE_DIV, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            """
                        <div style="font-size: 0.9rem; color: var(--text-secondary);">
                            <span style="font-size: 1.2rem; margin-right: 8px;">🔜</span>
                            Challenges coming soon!
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                st.markdown(Dashboard.CLOSE_DIV, unsafe_allow_html=True)

            with col3:
                if status["completed"]:
                    st.markdown(
                        """
                    <div class="ios-card" style="text-align: center; height: 100px; display: flex;
                               flex-direction: column; justify-content: center; background: var(--ios-green);">
                        <div style="font-size: 2rem; margin-bottom: 8px;">🏆</div>
                        <div style="font-size: 0.8rem; color: white; font-weight: 600;">MASTERED</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                    <div class="ios-card" style="text-align: center; height: 100px; display: flex;
                               flex-direction: column; justify-content: center; opacity: 0.5;">
                        <div style="font-size: 1.5rem; margin-bottom: 8px;">🔒</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">LOCKED</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            # Spacer
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    def show_challenges(self) -> None:
        """iOS-inspired challenges page"""
        st.markdown(
            """
        <div class="ios-card animate-slide-in">
            <h2 style="margin-bottom: 8px;">🎯 Coding Challenges</h2>
            <p style="color: var(--text-secondary); margin: 0;">Hands-on challenges to practice your data science skills</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Level selector with iOS style
        current_level = self.game.get_current_level()
        available_levels = list(range(1, min(current_level + 1, 8)))

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
            <div class="ios-card" style="text-align: center; margin: 20px 0;">
                <h4 style="margin-bottom: 16px; color: var(--text-secondary);">Select Learning Level</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            selected_level = st.selectbox(
                "Choose Level",
                available_levels,
                index=len(available_levels) - 1,
                format_func=lambda x: f"Level {x}: {LEVELS[x]['name']}",
            )

        # Level info header
        level_info = LEVELS[selected_level]
        level_status = self.game.progress["level_progress"][str(selected_level)]

        if level_status["unlocked"]:
            st.markdown(
                f"""
            <div class="level-card" style="text-align: center; margin: 20px 0;">
                <h2 style="margin: 0 0 12px 0;">Level {selected_level}: {level_info['name']}</h2>
                <p style="margin: 0; opacity: 0.9; font-size: 1.1rem;">{level_info['description']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Get enhanced challenges for selected level with metadata
            challenges = self.game.get_enhanced_challenges(selected_level)

            if challenges:
                # Challenge statistics
                completed_challenges = [
                    c
                    for c in self.game.progress["challenges_completed"]
                    if c.startswith(f"level_{selected_level}")
                ]
                completion_rate = (len(completed_challenges) / len(challenges)) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"""
                    <div class="ios-card" style="text-align: center;">
                        <div style="font-size: 2.5rem; margin-bottom: 8px;">📚</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: var(--ios-blue);">{len(challenges)}</div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary);">Total Challenges</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                    <div class="ios-card" style="text-align: center;">
                        <div style="font-size: 2.5rem; margin-bottom: 8px;">✅</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: var(--ios-green);">{len(completed_challenges)}</div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary);">Completed</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col3:
                    st.markdown(
                        f"""
                    <div class="ios-card" style="text-align: center;">
                        <div style="font-size: 2.5rem; margin-bottom: 8px;">📊</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: var(--ios-orange);">{completion_rate:.0f}%</div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary);">Progress</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Challenges list
                st.markdown(
                    """
                <div class="ios-card" style="margin: 30px 0 20px 0;">
                    <h3 style="margin: 0;">🚀 Available Challenges</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                for i, challenge_data in enumerate(challenges, 1):
                    if isinstance(challenge_data, dict):
                        challenge_name = challenge_data.get(
                            "title", challenge_data.get("name", f"Challenge {i}")
                        )
                    else:
                        challenge_name = str(challenge_data)
                    # Create unique challenge ID using both level and index
                    challenge_id = f"level_{selected_level}_challenge_{i}_{challenge_name.replace(' ', '_').lower()}"
                    completed = (
                        challenge_id in self.game.progress["challenges_completed"]
                    )

                    # Enhanced metadata from challenge_loader
                    if isinstance(challenge_data, dict):
                        difficulty = challenge_data.get("difficulty", "Unknown")
                        estimated_time_str = challenge_data.get(
                            "estimated_time", "20 minutes"
                        )
                        description = challenge_data.get(
                            "description",
                            challenge_data.get(
                                "title", f'Challenge {i} for {level_info["name"]}'
                            ),
                        )
                        concepts = challenge_data.get(
                            "concepts", challenge_data.get("objectives", [])
                        )

                        # Parse time string if it's in "15-20 minutes" format
                        try:
                            import re

                            if "minutes" in str(estimated_time_str) or "min" in str(
                                estimated_time_str
                            ):
                                # Extract numbers from strings like "15-20 minutes"
                                numbers = re.findall(r"\d+", str(estimated_time_str))
                                if numbers:
                                    estimated_time = int(
                                        numbers[-1]
                                    )  # Use the last number found
                                else:
                                    estimated_time = 20
                            elif isinstance(estimated_time_str, (int, float)):
                                estimated_time = int(estimated_time_str)
                            else:
                                estimated_time = 20
                        except (ValueError, TypeError, AttributeError):
                            estimated_time = 20

                        # Format time display
                        if estimated_time > 60:
                            time_str = f"{estimated_time // 60}h {estimated_time % 60}m"
                        else:
                            time_str = f"{estimated_time}m"
                    else:
                        # Fallback for string-only challenges
                        difficulty = "Unknown"
                        time_str = "~20m"
                        description = f"Challenge {i} for {level_info['name']} - Practice your skills with hands-on exercises"
                        concepts = []  # Challenge card styling
                    if completed:
                        card_style = "background: linear-gradient(135deg, var(--ios-green), var(--ios-teal)); color: white;"
                        status_icon = "✅"
                        status_text = "Completed"
                        button_text = "Review"
                        button_disabled = False
                    else:
                        card_style = "background: var(--surface-primary);"
                        status_icon = "▶️"
                        status_text = "Ready to Start"
                        button_text = "Start Challenge"
                        button_disabled = False

                    # Difficulty color coding
                    difficulty_colors = {
                        "Beginner": "var(--ios-green)",
                        "Easy": "var(--ios-green)",
                        "Intermediate": "var(--ios-orange)",
                        "Medium": "var(--ios-orange)",
                        "Advanced": "var(--ios-red)",
                        "Hard": "var(--ios-red)",
                        "Expert": "var(--ios-purple)",
                    }
                    difficulty_color = difficulty_colors.get(
                        difficulty, "var(--ios-blue)"
                    )

                    col1, col2, col3 = st.columns([1, 4, 1])

                    with col1:
                        st.markdown(
                            f"""
                        <div class="ios-card" style="text-align: center; height: 140px; display: flex;
                                   flex-direction: column; justify-content: center; {card_style}">
                            <div style="font-size: 2.5rem; margin-bottom: 8px;">{status_icon}</div>
                            <div style="font-size: 0.8rem; font-weight: 600; opacity: 0.8;">{i:02d}</div>
                            <div style="font-size: 0.7rem; margin-top: 4px; opacity: 0.7;">{time_str}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        st.markdown(
                            f"""
                        <div class="ios-card" style="height: 140px; display: flex; flex-direction: column; justify-content: center;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <h4 style="margin: 0; color: var(--text-primary); flex: 1;">{challenge_name}</h4>
                                <div style="display: flex; gap: 8px; align-items: center;">
                                    <div style="background: {difficulty_color}; color: white; border-radius: 12px; padding: 4px 8px;
                                               font-size: 0.7rem; font-weight: 600;">
                                        {difficulty}
                                    </div>
                                    <div style="background: var(--surface-tertiary); border-radius: 16px; padding: 4px 12px;
                                               font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);">
                                        {status_text}
                                    </div>
                                </div>
                            </div>
                            <p style="margin: 0 0 8px 0; color: var(--text-secondary); font-size: 0.95rem; line-height: 1.3;">
                                {description}
                            </p>
                            {f'''<div style="margin-bottom: 8px;">
                                <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                                    {' '.join([f'<span style="background: var(--ios-blue); color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 8px;">{concept}</span>' for concept in concepts[:3]])}
                                </div>
                            </div>''' if concepts else ''}
                            <div style="margin-top: 8px;">
                                <!-- Progress indicator -->
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="flex: 1; height: 4px; background: var(--surface-tertiary); border-radius: 2px;">
                                        <div style="height: 100%; background: {Dashboard.IOS_GREEN if completed else Dashboard.IOS_BLUE};
                                                   width: {'100%' if completed else '0%'}; border-radius: 2px; transition: width 0.3s ease;"></div>
                                    </div>
                                    <span style="font-size: 0.8rem; color: var(--text-secondary); font-weight: 600;">
                                        {'100%' if completed else '0%'}
                                    </span>
                                </div>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with col3:
                        st.markdown(
                            """
                        <div style="height: 120px; display: flex; align-items: center; justify-content: center;">
                        """,
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            button_text,
                            key=f"challenge_{challenge_id}",
                            disabled=button_disabled,
                        ):
                            # Store challenge info in session state for viewing
                            st.session_state.viewing_challenge = {
                                "level": selected_level,
                                "name": challenge_name,
                                "data": challenge_data,
                                "completed": completed,
                            }
                            st.session_state.show_challenge_modal = True
                            st.rerun()

                        st.markdown("</div>", unsafe_allow_html=True)

                    # Spacer
                    st.markdown(
                        "<div style='margin: 16px 0;'></div>", unsafe_allow_html=True
                    )

            else:
                # No challenges available
                st.markdown(
                    """
                <div class="ios-card glass-panel" style="text-align: center; padding: 60px; margin: 40px 0;">
                    <div style="font-size: 4rem; margin-bottom: 20px;">🔜</div>
                    <h3 style="margin-bottom: 16px;">Challenges Coming Soon!</h3>
                    <p style="color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 30px;">
                        We're preparing exciting challenges for Level {selected_level}. Check back soon!
                    </p>
                    <div style="display: flex; gap: 12px; justify-content: center;">
                """,
                    unsafe_allow_html=True,
                )

                if st.button("🏆 Explore Other Levels", key="explore_levels"):
                    st.session_state.page = "Levels"
                    st.rerun()

                if st.button("📊 View Progress", key="view_progress_no_challenges"):
                    st.session_state.page = "Progress"
                    st.rerun()

                st.markdown(Dashboard.CLOSE_DIV, unsafe_allow_html=True)
        else:
            # Level not unlocked
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center; padding: 60px; background: var(--surface-secondary);
                       border: 2px dashed var(--ios-gray);">
                <div style="font-size: 4rem; margin-bottom: 20px; opacity: 0.5;">🔒</div>
                <h3 style="margin-bottom: 16px; color: var(--text-primary);">Level {selected_level} Locked</h3>
                <p style="color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 30px;">
                    Complete previous levels to unlock Level {selected_level}: {level_info['name']}
                </p>
                <div style="display: flex; gap: 12px; justify-content: center;">
            """,
                unsafe_allow_html=True,
            )

            if st.button("🏆 View All Levels", key="view_levels_locked"):
                st.session_state.page = "Levels"
                st.rerun()

            if st.button("📊 Check Progress", key="check_progress_locked"):
                st.session_state.page = "Progress"
                st.rerun()

            st.markdown(Dashboard.CLOSE_DIV, unsafe_allow_html=True)

    def show_badges(self) -> None:
        """iOS-inspired badges page with enhanced badge system"""
        st.markdown(
            """
        <div class="ios-card animate-slide-in">
            <h2 style="margin-bottom: 8px;">🏅 Achievement Badges</h2>
            <p style="color: var(--text-secondary); margin: 0;">Track your accomplishments and unlock new badges!</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Statistics overview
        earned_badges = self.game.progress["badges_earned"]
        total_badges = len(BADGES)
        completion_rate = (
            (len(earned_badges) / total_badges) * 100 if total_badges > 0 else 0
        )

        # Get next badges to earn using enhanced system
        next_badges = self.game.get_next_badges()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 12px;">🏆</div>
                <div style="font-size: 2rem; font-weight: 700; color: var(--ios-purple);">{len(earned_badges)}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Badges Earned</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 12px;">🎯</div>
                <div style="font-size: 2rem; font-weight: 700; color: var(--ios-blue);">{total_badges - len(earned_badges)}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Still to Earn</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 12px;">📊</div>
                <div style="font-size: 2rem; font-weight: 700; color: var(--ios-green);">{completion_rate:.0f}%</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Completion Rate</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Earned badges section
        if earned_badges:
            st.markdown(
                f"""
            <div class="ios-card" style="margin: 30px 0 20px 0;">
                <h3 style="margin: 0;">🏆 Your Achievements ({len(earned_badges)})</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Create a grid layout for earned badges
            cols = st.columns(3)
            for i, badge_id in enumerate(earned_badges):
                if badge_id in BADGES:
                    badge = BADGES[badge_id]
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                        <div class="badge-card earned" style="transform: none; margin: 12px 0;">
                            <div style="font-size: 3rem; margin-bottom: 16px;">🏆</div>
                            <h4 style="margin: 0 0 8px 0; font-weight: 700;">{badge['name']}</h4>
                            <p style="margin: 0; font-size: 0.9rem; opacity: 0.9; line-height: 1.4;">{badge['description']}</p>
                            <div style="margin-top: 16px; padding: 8px 16px; background: rgba(255,255,255,0.2);
                                       border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                EARNED ✨
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

        # Next badges to earn (enhanced feature)
        if next_badges:
            st.markdown(
                """
            <div class="ios-card" style="margin: 30px 0 20px 0;">
                <h3 style="margin: 0;">🚀 Next Badges to Earn</h3>
                <p style="color: var(--text-secondary); margin: 8px 0 0 0; font-size: 0.9rem;">
                    These badges are within your reach based on current progress
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            cols = st.columns(3)
            for i, badge_data in enumerate(next_badges[:3]):  # Show only first 3
                badge_id = badge_data.get("id", "")
                badge_name = badge_data.get("name", badge_id)
                badge_description = badge_data.get("description", "Achievement badge")
                progress_info = badge_data.get("progress", {}).get(
                    "requirements_met", False
                )
                progress_text = (
                    "Close to earning!" if not progress_info else "Requirements met!"
                )

                with cols[i]:
                    st.markdown(
                        f"""
                    <div class="badge-card" style="transform: none; margin: 12px 0;
                                background: linear-gradient(135deg, {Dashboard.IOS_BLUE} 0%, {Dashboard.IOS_PURPLE} 100%);
                                color: white; border: 3px solid var(--ios-yellow);">
                        <div style="font-size: 3rem; margin-bottom: 16px;">🌟</div>
                        <h4 style="margin: 0 0 8px 0; font-weight: 700;">{badge_name}</h4>
                        <p style="margin: 0 0 12px 0; font-size: 0.9rem; opacity: 0.9; line-height: 1.4;">{badge_description}</p>
                        <div style="margin-top: 12px; padding: 8px 12px; background: rgba(255,255,255,0.2);
                                   border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                            {progress_text}
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        # Available badges section
        available_badges = [bid for bid in BADGES.keys() if bid not in earned_badges]
        if available_badges:
            st.markdown(
                f"""
            <div class="ios-card" style="margin: 30px 0 20px 0;">
                <h3 style="margin: 0;">🎯 Available Badges ({len(available_badges)})</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Create a grid layout for available badges
            cols = st.columns(3)
            for i, badge_id in enumerate(available_badges):
                badge = BADGES[badge_id]
                with cols[i % 3]:
                    # Different styling for locked/available badges
                    st.markdown(
                        f"""
                    <div class="badge-card" style="transform: none; margin: 12px 0; opacity: 0.7;
                                background: var(--surface-secondary); border: 2px dashed var(--ios-gray);">
                        <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.5;">🔒</div>
                        <h4 style="margin: 0 0 8px 0; font-weight: 700; color: var(--text-primary);">{badge['name']}</h4>
                        <p style="margin: 0; font-size: 0.9rem; color: var(--text-secondary); line-height: 1.4;">{badge['description']}</p>
                        <div style="margin-top: 16px; padding: 8px 16px; background: var(--surface-tertiary);
                                   border-radius: 12px; font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);">
                            NOT YET EARNED
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        # Achievement tips
        if len(earned_badges) < total_badges:
            st.markdown(
                """
            <div class="ios-card glass-panel" style="margin-top: 30px; text-align: center; padding: 40px;">
                <div style="font-size: 4rem; margin-bottom: 20px;">🚀</div>
                <h3 style="margin-bottom: 16px; color: var(--text-primary);">Keep Learning!</h3>
                <p style="color: var(--text-secondary); margin-bottom: 20px; font-size: 1.1rem;">
                    Complete more challenges and level up to earn additional badges.
                </p>
                <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
            """,
                unsafe_allow_html=True,
            )

            if st.button("🎯 View Challenges", key="view_challenges_from_badges"):
                st.session_state.page = "Challenges"
                st.rerun()

            if st.button("🏆 Check Levels", key="check_levels_from_badges"):
                st.session_state.page = "Levels"
                st.rerun()

            st.markdown(Dashboard.CLOSE_DIV, unsafe_allow_html=True)

    def show_progress(self) -> None:
        """iOS-inspired progress analytics page"""
        st.markdown(
            """
        <div class="ios-card animate-slide-in">
            <h2 style="margin-bottom: 8px;">📊 Progress Analytics</h2>
            <p style="color: var(--text-secondary); margin: 0;">Detailed insights into your learning journey</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Key metrics overview
        col1, col2, col3, col4 = st.columns(4)

        metrics = [
            ("📅", "Days Active", "7", "This week"),
            ("🔥", "Current Streak", "5", "Days in a row"),
            ("⏱️", "Study Time", "12.5", "Hours total"),
            ("🎯", "Avg Score", "87%", "Challenge average"),
        ]

        for i, (icon, title, value, subtitle) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.markdown(
                    f"""
                <div class="ios-card" style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 12px;">{icon}</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: var(--ios-blue);">{value}</div>
                    <div style="font-size: 0.9rem; color: var(--text-primary); margin: 4px 0;">{title}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">{subtitle}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Experience growth chart
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                """
            <div class="ios-card" style="margin: 30px 0 20px 0;">
                <h3 style="margin-bottom: 20px;">📈 Experience Growth Over Time</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Enhanced mock data for demonstration
            dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
            exp_data = pd.DataFrame(
                {
                    "Date": dates,
                    "Experience": [
                        i * 25 + (i**1.2) * 10 + (i * 5 * (i % 3)) for i in range(30)
                    ],
                    "Challenges": [(i // 3) + (i % 7) for i in range(30)],
                }
            )

            # Create line chart with iOS-style colors
            fig = px.line(exp_data, x="Date", y="Experience", title="")

            # Update styling to match professional aesthetics
            fig.update_traces(
                line={"color": "#1D4ED8", "width": 3},
                fill="tonexty",
                fillcolor="rgba(29, 78, 216, 0.1)",
            )

            fig.update_layout(
                plot_bgcolor=Dashboard.TRANSPARENT_BG,
                paper_bgcolor=Dashboard.TRANSPARENT_BG,
                font={"family": Dashboard.SF_FONT, "color": "#1C1C1E"},
                showlegend=False,
                margin={"t": 20, "b": 50, "l": 50, "r": 20},
                xaxis={
                    "showgrid": True,
                    "gridcolor": Dashboard.GRID_COLOR,
                    "showline": False,
                },
                yaxis={
                    "showgrid": True,
                    "gridcolor": Dashboard.GRID_COLOR,
                    "showline": False,
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(
                """
            <div class="ios-card" style="margin: 30px 0 20px 0;">
                <h4 style="margin-bottom: 20px;">🏆 Recent Achievements</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Recent achievements timeline
            achievements = [
                ("🏅", "First Steps", "2 days ago", Dashboard.IOS_GREEN),
                ("🎯", "Challenge Master", "5 days ago", "var(--ios-blue)"),
                ("📊", "Data Explorer", "1 week ago", Dashboard.IOS_PURPLE),
            ]

            for icon, title, time, color in achievements:
                st.markdown(
                    f"""
                <div style="display: flex; align-items: center; padding: 16px; margin: 12px 0;
                           background: var(--surface-tertiary); border-radius: 16px;
                           border-left: 4px solid {color};">
                    <span style="font-size: 1.5rem; margin-right: 16px;">{icon}</span>
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">{title}</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">{time}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Skills breakdown and learning patterns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div class="ios-card" style="margin: 30px 0 20px 0;">
                <h3 style="margin-bottom: 20px;">💪 Skills Proficiency</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Enhanced skills data with more realistic progression
            skills_data = pd.DataFrame(
                {
                    "Skill": [
                        "Data Cleaning",
                        "Visualization",
                        "Statistics",
                        "Machine Learning",
                        "Python",
                        "Analysis",
                    ],
                    "Proficiency": [85, 70, 60, 45, 90, 75],
                    "Category": [
                        "Technical",
                        "Technical",
                        "Analytical",
                        "Advanced",
                        "Programming",
                        "Analytical",
                    ],
                }
            )

            # Create horizontal bar chart with iOS colors
            fig = px.bar(
                skills_data,
                x="Proficiency",
                y="Skill",
                color="Category",
                orientation="h",
                color_discrete_map={
                    "Technical": "#1D4ED8",
                    "Analytical": "#047857",
                    "Advanced": "#7C3AED",
                    "Programming": "#C2410C",
                },
            )

            fig.update_layout(
                plot_bgcolor=Dashboard.TRANSPARENT_BG,
                paper_bgcolor=Dashboard.TRANSPARENT_BG,
                font={"family": Dashboard.SF_FONT, "color": "#1C1C1E"},
                showlegend=True,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                margin={"t": 50, "b": 50, "l": 100, "r": 50},
                xaxis={
                    "range": [0, 100],
                    "showgrid": True,
                    "gridcolor": Dashboard.GRID_COLOR,
                },
                yaxis={"showgrid": False},
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(
                """
            <div class="ios-card" style="margin: 30px 0 20px 0;">
                <h3 style="margin-bottom: 20px;">⏰ Learning Time Distribution</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Enhanced time distribution data
            time_data = pd.DataFrame(
                {
                    "Category": [
                        "Practice Challenges",
                        "Theory & Reading",
                        "Projects",
                        "Review & Analysis",
                    ],
                    "Hours": [25, 15, 20, 8],
                    "Color": ["#1D4ED8", "#047857", "#7C3AED", "#C2410C"],
                }
            )

            # Create donut chart with professional styling
            fig = px.pie(
                time_data,
                values="Hours",
                names="Category",
                color_discrete_sequence=["#1D4ED8", "#047857", "#7C3AED", "#C2410C"],
            )

            # Convert to donut chart
            fig.update_traces(
                hole=0.4,
                textinfo="percent+label",
                textfont_size=12,
                marker={"line": {"color": "#FFFFFF", "width": 2}},
            )

            fig.update_layout(
                plot_bgcolor=Dashboard.TRANSPARENT_BG,
                paper_bgcolor=Dashboard.TRANSPARENT_BG,
                font={"family": Dashboard.SF_FONT, "color": "#1C1C1E"},
                showlegend=False,
                margin={"t": 50, "b": 50, "l": 50, "r": 50},
                annotations=[
                    {
                        "text": f'{sum(time_data["Hours"])}h<br>Total',
                        "x": 0.5,
                        "y": 0.5,
                        "font_size": 20,
                        "showarrow": False,
                        "font": {"color": "#1C1C1E", "weight": "bold"},
                    }
                ],
            )

            st.plotly_chart(fig, use_container_width=True)

        # Enhanced Gamification Section
        col1, col2 = st.columns(2)

        with col1:
            self.render_performance_heatmap_calendar()

        with col2:
            self.render_skills_radar_chart()

        # Milestone Timeline
        self.render_milestone_timeline()

        # Action buttons
        st.markdown(
            """
        <div class="ios-card glass-panel" style="margin-top: 30px; text-align: center; padding: 30px;">
            <h4 style="margin-bottom: 20px;">Continue Your Learning Journey</h4>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎯 New Challenge", key="new_challenge_progress"):
                st.session_state.page = "Challenges"
                st.rerun()

        with col2:
            if st.button("🏆 View Levels", key="view_levels_progress"):
                st.session_state.page = "Levels"
                st.rerun()

        with col3:
            if st.button("🏅 Check Badges", key="check_badges_progress"):
                st.session_state.page = "Badges"
                st.rerun()

    def show_settings(self) -> None:
        """iOS-inspired settings page"""
        st.markdown(
            """
        <div class="ios-card animate-slide-in">
            <h2 style="margin-bottom: 8px;">⚙️ Settings & Preferences</h2>
            <p style="color: var(--text-secondary); margin: 0;">Customize your learning experience</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Profile Section
        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">👤 Profile Settings</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                """
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 5rem; margin-bottom: 16px;">👤</div>
                <h4 style="margin: 0; color: var(--text-primary);">Profile Avatar</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            current_name = self.game.progress["player_name"]
            new_name = st.text_input(
                "Display Name",
                value=current_name,
                placeholder="Enter your display name",
                help="This name will appear throughout the application",
            )

            if new_name != current_name:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("💾 Save Changes", type="primary"):
                        self.game.progress["player_name"] = new_name
                        self.game.save_progress()
                        st.success("✅ Profile updated successfully!")
                        st.rerun()
                with col_b:
                    if st.button("↩️ Cancel"):
                        st.rerun()

        # Learning Preferences
        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">🎯 Learning Preferences</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div class="ios-card" style="background: var(--surface-secondary);">
                <h4 style="margin-bottom: 16px;">📚 Difficulty Preference</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.radio(
                "Choose your preferred challenge difficulty:",
                ["Beginner Friendly", "Standard", "Advanced"],
                index=1,
                help="This affects the complexity of recommended challenges",
            )

            st.markdown(
                """
            <div class="ios-card" style="background: var(--surface-secondary); margin-top: 16px;">
                <h4 style="margin-bottom: 16px;">🔔 Notifications</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.checkbox("Enable learning reminders", value=True)
            st.checkbox("Badge achievement alerts", value=True)
            st.checkbox("Weekly progress summary", value=False)

        with col2:
            st.markdown(
                """
            <div class="ios-card" style="background: var(--surface-secondary);">
                <h4 style="margin-bottom: 16px;">🎨 Display Options</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.selectbox(
                "Color Theme", ["Auto (System)", "Light Mode", "Dark Mode"], index=0
            )

            st.checkbox("Enable animations", value=True)
            st.checkbox("Compact card layout", value=False)

            st.markdown(
                """
            <div class="ios-card" style="background: var(--surface-secondary); margin-top: 16px;">
                <h4 style="margin-bottom: 16px;">📊 Data & Privacy</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.checkbox(
                "Share anonymous usage data",
                value=True,
                help="Helps improve the learning platform",
            )
            st.checkbox("Automatic progress backup", value=True)

        # Progress Management
        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">💾 Progress Management</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            <div class="ios-card" style="text-align: center; background: var(--ios-blue); color: white;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">📤</div>
                <h4 style="margin-bottom: 16px;">Export Progress</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Download your learning data</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if st.button("📤 Export Data", key="export_progress"):
                progress_json = json.dumps(self.game.progress, indent=2)
                st.download_button(
                    label="💾 Download Progress File",
                    data=progress_json,
                    file_name=f"sandbox_progress_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                )

        with col2:
            st.markdown(
                """
            <div class="ios-card" style="text-align: center; background: var(--ios-orange); color: white;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">🔄</div>
                <h4 style="margin-bottom: 16px;">Reset Progress</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Start your journey over</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if st.button("🔄 Reset All Data", key="reset_all", type="secondary"):
                st.warning("⚠️ This will permanently delete all your progress!")
                if st.button(
                    "⚠️ I'm Sure, Reset Everything",
                    key="confirm_reset_all",
                    type="primary",
                ):
                    self.game.reset_progress()
                    st.success("✅ Progress reset successfully!")
                    st.rerun()

        with col3:
            st.markdown(
                """
            <div class="ios-card" style="text-align: center; background: var(--ios-purple); color: white;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">📱</div>
                <h4 style="margin-bottom: 16px;">App Info</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Version & statistics</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if st.button("ℹ️ Show App Info", key="show_info"):
                st.info(
                    """
                **Data Science Sandbox v2.0**
                iOS 26 HIG Inspired Design

                📊 Total Levels: 6
                🎯 Total Challenges: 25+
                🏅 Total Badges: 15

                Built with ❤️ and Streamlit
                """
                )

        # Color Accessibility Status
        try:
            from sandbox.utils.dashboard_contrast_integration import (
                DashboardContrastIntegration,
            )

            DashboardContrastIntegration.render_contrast_status()
        except ImportError:
            st.warning("Color contrast monitoring not available")

        # Advanced Settings (collapsible)
        with st.expander("🔧 Advanced Settings & Debug"):
            st.markdown("**System Information**")
            col1, col2 = st.columns(2)

            with col1:
                st.code(
                    f"""
Platform: Streamlit {st.__version__}
Session State Keys: {len(st.session_state)}
Current Page: {st.session_state.get('page', 'Dashboard')}
                """
                )

            with col2:
                stats = self.game.get_stats()
                st.code(
                    f"""
Player Level: {stats['level']}
Experience: {stats['experience']} XP
Completion: {stats['completion_rate']:.1f}%
                """
                )

            st.markdown("**Raw Progress Data**")
            st.json(self.game.progress)

            # Cache management
            st.markdown("**Cache Management**")
            if st.button("🗑️ Clear Streamlit Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")

        # Save notification at bottom
        st.markdown(
            """
        <div class="ios-card glass-panel" style="margin-top: 30px; text-align: center; padding: 20px;">
            <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">
                💡 Your settings are automatically saved as you make changes
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_study_timer(self) -> None:
        """Render an iOS-style study session timer"""
        import time

        # Initialize session state for timer
        if "session_start_time" not in st.session_state:
            st.session_state.session_start_time = None
        if "session_total_time" not in st.session_state:
            st.session_state.session_total_time = 0
        if "session_paused" not in st.session_state:
            st.session_state.session_paused = True

        # Calculate current session time
        current_time = 0
        if st.session_state.session_start_time and not st.session_state.session_paused:
            current_time = time.time() - st.session_state.session_start_time

        total_display_time = st.session_state.session_total_time + current_time

        # Format time for display
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            if hours > 0:
                return f"{hours}h {minutes:02d}m {seconds:02d}s"
            return f"{minutes:02d}m {seconds:02d}s"

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(
                f"""
            <div class="ios-card" style="background: linear-gradient(135deg, var(--ios-green) 0%, var(--ios-teal) 100%);
                 color: white; text-align: center; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                     background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);"></div>
                <div style="position: relative; z-index: 1;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">⏱️</div>
                    <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 8px;
                         font-family: 'SF Pro Display', monospace;">{format_time(total_display_time)}</div>
                    <div style="font-size: 1rem; opacity: 0.9;">Study Session</div>
                    <div style="margin-top: 12px; font-size: 0.9rem; opacity: 0.8;">
                        Status: {"🔥 Active" if not st.session_state.session_paused else "⏸️ Paused"}
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            if st.session_state.session_paused:
                if st.button("▶️ Start", key="start_timer", help="Start study session"):
                    # Start or resume session - always set current time as start time
                    st.session_state.session_start_time = time.time()
                    st.session_state.session_paused = False
                    st.rerun()
            else:
                if st.button("⏸️ Pause", key="pause_timer", help="Pause study session"):
                    if st.session_state.session_start_time:
                        st.session_state.session_total_time += (
                            time.time() - st.session_state.session_start_time
                        )
                    st.session_state.session_paused = True
                    st.rerun()

        with col3:
            if st.button("🔄 Reset", key="reset_timer", help="Reset session timer"):
                st.session_state.session_start_time = None
                st.session_state.session_total_time = 0
                st.session_state.session_paused = True
                st.rerun()

        # Auto-refresh every second when timer is running
        if not st.session_state.session_paused and st.session_state.session_start_time:
            time.sleep(1)
            st.rerun()

    def render_dark_mode_toggle(self) -> None:
        """Render dark/light mode toggle"""
        # Initialize theme in session state
        if "dark_mode" not in st.session_state:
            st.session_state.dark_mode = False

        # Theme toggle in sidebar
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            st.markdown("🌙 Dark Mode")
        with col2:
            pass  # Removed duplicate theme toggle - now handled in sidebar

    def render_enhanced_metrics(self) -> None:
        """Render additional metrics and insights"""

        # Performance insights
        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">📈 Performance Insights</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        # Calculate some enhanced metrics
        challenges_completed = len(self.game.progress["challenges_completed"])
        total_challenges = sum(
            len(self.game.get_level_challenges(i)) for i in range(1, 8)
        )
        completion_rate = (
            (challenges_completed / total_challenges * 100)
            if total_challenges > 0
            else 0
        )

        with col1:
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center; background: linear-gradient(135deg, var(--ios-blue), var(--ios-purple)); color: white;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">🎯</div>
                <div style="font-size: 2rem; font-weight: 700; margin-bottom: 8px;">{completion_rate:.1f}%</div>
                <div style="font-size: 1rem; opacity: 0.9;">Completion Rate</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            # Calculate learning streak (mock data for now)
            learning_streak = 5  # This could be calculated from session data
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center; background: linear-gradient(135deg, var(--ios-orange), var(--ios-yellow)); color: white;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">🔥</div>
                <div style="font-size: 2rem; font-weight: 700; margin-bottom: 8px;">{learning_streak}</div>
                <div style="font-size: 1rem; opacity: 0.9;">Day Streak</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            # Average score (mock calculation)
            avg_score = 87.5  # This could be calculated from challenge scores
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center; background: linear-gradient(135deg, var(--ios-green), var(--ios-teal)); color: white;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">⭐</div>
                <div style="font-size: 2rem; font-weight: 700; margin-bottom: 8px;">{avg_score:.1f}%</div>
                <div style="font-size: 1rem; opacity: 0.9;">Avg Score</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    def render_challenge_recommendations(self) -> None:
        """Render enhanced challenge recommendations using new systems"""
        current_level = self.game.get_current_level()
        completed_challenges = self.game.progress["challenges_completed"]

        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 8px;">🎯 Recommended for You</h3>
            <p style="color: var(--text-secondary); margin: 0;">Smart suggestions based on your progress and next badges to earn</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Get enhanced recommendations based on current progress
        recommendations = []

        # Get enhanced challenges for current and next level
        current_enhanced = (
            self.game.get_enhanced_challenges(current_level)
            if current_level <= 7
            else []
        )
        next_enhanced = (
            self.game.get_enhanced_challenges(min(current_level + 1, 7))
            if current_level < 7
            else []
        )

        # Find next challenge to complete in current level
        if current_enhanced:
            for challenge_data in current_enhanced[:3]:  # Check first 3 challenges
                if isinstance(challenge_data, dict):
                    challenge_name = challenge_data.get(
                        "title", challenge_data.get("name", "Challenge")
                    )
                else:
                    challenge_name = str(challenge_data)
                # Create unique challenge ID
                challenge_id = f"level_{current_level}_rec_{challenge_name.replace(' ', '_').lower()}"

                if challenge_id not in completed_challenges:
                    if isinstance(challenge_data, dict):
                        # Parse time properly
                        estimated_time_str = challenge_data.get(
                            "estimated_time", Dashboard.DEFAULT_TIME_STR
                        )
                        try:
                            import re

                            if "minutes" in str(estimated_time_str) or "min" in str(
                                estimated_time_str
                            ):
                                numbers = re.findall(r"\d+", str(estimated_time_str))
                                parsed_time = int(numbers[-1]) if numbers else 20
                            elif isinstance(estimated_time_str, (int, float)):
                                parsed_time = int(estimated_time_str)
                            else:
                                parsed_time = 20
                        except (ValueError, TypeError, AttributeError):
                            parsed_time = 20

                        recommendations.append(
                            {
                                "title": challenge_name,
                                "level": current_level,
                                "difficulty": challenge_data.get(
                                    "difficulty", "Unknown"
                                ),
                                "time": parsed_time,
                                "reason": f"Continue your {LEVELS[current_level]['name']} journey",
                                "description": challenge_data.get("description", ""),
                                "concepts": challenge_data.get("concepts", []),
                                "icon": "🎯",
                                "color": Dashboard.IOS_BLUE,
                            }
                        )
                        break

        # Add next level preview if user has made good progress
        if len(completed_challenges) >= 2 and next_enhanced and current_level < 7:
            first_next_challenge = next_enhanced[0]
            if isinstance(first_next_challenge, dict):
                challenge_name = first_next_challenge.get(
                    "title", first_next_challenge.get("name", "Next Challenge")
                )
            else:
                challenge_name = str(first_next_challenge)

            if isinstance(first_next_challenge, dict):
                recommendations.append(
                    {
                        "title": challenge_name,
                        "level": current_level + 1,
                        "difficulty": first_next_challenge.get("difficulty", "Unknown"),
                        "time": 20,  # Simplified - will be parsed properly in display
                        "reason": f"Preview of {LEVELS[current_level + 1]['name']}",
                        "description": first_next_challenge.get("description", ""),
                        "concepts": first_next_challenge.get("concepts", []),
                        "icon": "�",
                        "color": "var(--ios-purple)",
                    }
                )

        # If no enhanced data, fall back to basic recommendations
        if not recommendations:
            if len(completed_challenges) == 0:
                recommendations = [
                    {
                        "title": "First Steps with Data",
                        "level": 1,
                        "difficulty": "Beginner",
                        "time": 20,
                        "reason": "Perfect starting point for your data science journey",
                        "description": "Learn the fundamentals of data analysis",
                        "concepts": ["pandas", "data loading", "basic analysis"],
                        "icon": "🚀",
                        "color": "var(--ios-green)",
                    }
                ]
            elif current_level <= 2:
                recommendations = [
                    {
                        "title": "Continue Your Journey",
                        "level": current_level,
                        "difficulty": (
                            "Beginner" if current_level == 1 else "Intermediate"
                        ),
                        "time": 25,
                        "reason": f"Keep building skills in {LEVELS[current_level]['name']}",
                        "description": "Practice more hands-on exercises",
                        "concepts": ["practice", "skill-building"],
                        "icon": "�",
                        "color": Dashboard.IOS_BLUE,
                    }
                ]
            else:
                # Advanced recommendations
                recommendations = [
                    {
                        "title": "Advanced Data Science",
                        "level": min(current_level, 7),
                        "difficulty": "Advanced",
                        "time": 45,
                        "reason": "You're ready for advanced concepts",
                        "description": "Tackle complex data science challenges",
                        "concepts": ["machine learning", "advanced analysis"],
                        "icon": "🤖",
                        "color": "var(--ios-purple)",
                    }
                ]

        # Display recommendations
        cols = st.columns(len(recommendations) if len(recommendations) <= 3 else 3)

        for i, rec in enumerate(recommendations[:3]):
            with cols[i]:
                # Format time display for recommendations
                time_str = ""
                if "time" in rec:
                    # Ensure time is an integer
                    try:
                        time_value = (
                            int(rec["time"])
                            if isinstance(rec["time"], (str, float))
                            else rec["time"]
                        )
                    except (ValueError, TypeError):
                        time_value = 20  # Default fallback

                    if time_value > 60:
                        time_str = f" • {time_value // 60}h {time_value % 60}m"
                    else:
                        time_str = f" • {time_value}m"

                st.markdown(
                    f"""
                <div class="ios-card" style="background: linear-gradient(135deg, {rec['color']}, var(--ios-teal));
                     color: white; text-align: center; position: relative; overflow: hidden; cursor: pointer;
                     transition: transform 0.3s ease;">
                    <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                         background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);"></div>
                    <div style="position: relative; z-index: 1;">
                        <div style="font-size: 3rem; margin-bottom: 16px;">{rec['icon']}</div>
                        <h4 style="margin: 0 0 12px 0; font-weight: 700;">{rec['title']}</h4>
                        <div style="background: rgba(255,255,255,0.2); border-radius: 12px;
                             padding: 6px 12px; font-size: 0.8rem; margin: 8px auto; display: inline-block;">
                            Level {rec['level']} • {rec['difficulty']}{time_str}
                        </div>
                        <p style="margin: 12px 0; font-size: 0.9rem; opacity: 0.9; line-height: 1.4;">
                            {rec.get('description', rec['reason'])}
                        </p>
                        {f'''<div style="margin: 12px 0;">
                            <div style="display: flex; flex-wrap: wrap; gap: 4px; justify-content: center;">
                                {' '.join([f'<span style="background: rgba(255,255,255,0.3); font-size: 0.7rem; padding: 2px 8px; border-radius: 8px;">{concept}</span>' for concept in rec.get('concepts', [])[:2]])}
                            </div>
                        </div>''' if rec.get('concepts') else ''}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Start Challenge", key=f"rec_{i}", help=f"Begin {rec['title']}"
                ):
                    st.session_state.page = "Challenges"
                    st.rerun()

    def show_challenge_modal(self) -> None:
        """Display challenge content in a modal-like interface"""
        challenge_info = st.session_state.get("viewing_challenge")
        if not challenge_info:
            st.session_state.show_challenge_modal = False
            st.rerun()
            return

        level = challenge_info["level"]
        challenge_name = challenge_info["name"]
        challenge_data = challenge_info["data"]
        completed = challenge_info["completed"]

        # Header with back button
        col1, col2, col3 = st.columns([1, 6, 1])

        with col1:
            if st.button(
                "← Back",
                key="back_to_challenges",
                type="secondary",
                help="Return to main navigation",
            ):
                st.session_state.show_challenge_modal = False
                st.session_state.viewing_challenge = None
                st.rerun()

        with col2:
            st.markdown(
                f"""
            <div style="text-align: center;">
                <h1 style="margin: 0;">🎯 {challenge_name}</h1>
                <p style="color: var(--text-secondary); margin: 0;">Level {level} Challenge</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Challenge content
        if isinstance(challenge_data, dict) and challenge_data.get("content"):
            content = challenge_data["content"]

            # Display challenge description
            description = challenge_data.get("description", "")
            if description:
                st.markdown(
                    f"""
                <div class="ios-card" style="margin: 20px 0;">
                    <h3>📖 Overview</h3>
                    <p>{description}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Display objectives if available
            objectives = challenge_data.get("objectives", [])
            if objectives:
                st.markdown(
                    """
                <div class="ios-card" style="margin: 20px 0;">
                    <h3>🎯 Learning Objectives</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                for obj in objectives:
                    st.markdown(f"• {obj}")

            # Display main content with better formatting
            st.markdown(
                """
            <div class="ios-card" style="margin: 20px 0;">
                <h3>📝 Challenge Content</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Parse and render markdown content with better formatting
            self._render_formatted_challenge_content(content)

            # Action buttons
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                if not completed:
                    if st.button(
                        "✅ Mark as Complete", key="complete_challenge", type="primary"
                    ):
                        challenge_id = (
                            f"level_{level}_{challenge_name.lower().replace(' ', '_')}"
                        )
                        self.game.complete_challenge(challenge_id)
                        st.success("🎉 Challenge completed! XP awarded.")
                        st.balloons()
                        # Update the challenge info
                        st.session_state.viewing_challenge["completed"] = True
                        st.rerun()
                else:
                    st.success("✅ Challenge Already Completed!")
                    if st.button(
                        "← Return to Dashboard",
                        key="return_dashboard",
                        help="Go back to main dashboard",
                    ):
                        st.session_state.show_challenge_modal = False
                        st.session_state.viewing_challenge = None
                        st.session_state.page = "Dashboard"
                        st.rerun()

            with col3:
                if st.button("📄 Open in Jupyter", key="open_jupyter"):
                    st.info("💡 Jupyter integration coming soon!")

        else:
            st.error("❌ Challenge content not found!")
            if st.button("← Back to Challenges", key="back_error"):
                st.session_state.show_challenge_modal = False
                st.session_state.viewing_challenge = None
                st.rerun()

    def _render_formatted_challenge_content(self, content: str) -> None:
        """Render challenge content with improved formatting and readability"""
        if not content:
            st.warning("No challenge content available.")
            return

        # Get theme-appropriate colors
        colors = self._get_theme_colors()

        # Split content into sections for better organization
        lines = content.split("\n")
        current_section = []
        in_code_block = False
        code_language = "python"

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()

            # Handle code blocks
            if line.startswith("```"):
                if in_code_block:
                    # End of code block
                    if current_section:
                        code_content = "\n".join(current_section)
                        if code_content.strip():
                            st.code(code_content, language=code_language)
                        current_section = []
                    in_code_block = False
                else:
                    # Start of code block
                    if current_section:
                        # Render any accumulated non-code content
                        text_content = "\n".join(current_section).strip()
                        if text_content:
                            formatted_text = self._format_text_content(
                                text_content, colors
                            )
                            st.markdown(formatted_text, unsafe_allow_html=True)
                        current_section = []

                    # Extract language if specified
                    code_language = line[3:].strip() or "python"
                    in_code_block = True
                i += 1
                continue

            if in_code_block:
                current_section.append(line)
            else:
                # Handle different types of content
                if (
                    line.startswith("# ")
                    or line.startswith("## ")
                    or line.startswith("### ")
                ):
                    # Render previous section first
                    if current_section:
                        text_content = "\n".join(current_section).strip()
                        if text_content:
                            formatted_text = self._format_text_content(
                                text_content, colors
                            )
                            st.markdown(formatted_text, unsafe_allow_html=True)
                        current_section = []

                    # Render header with custom styling
                    level = len(line) - len(line.lstrip("#"))
                    header_text = line.lstrip("# ").strip()

                    if level == 1:
                        st.markdown(
                            f"""
                        <div class="ios-card" style="margin: 25px 0 15px 0; background: linear-gradient(135deg, {colors['ios_blue']}, {colors['ios_purple']}); border-radius: 16px; padding: 20px;">
                            <h2 style="margin: 0; color: white; text-align: center;">🎯 {header_text}</h2>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    elif level == 2:
                        st.markdown(
                            f"""
                        <div style="margin: 20px 0 10px 0; padding: 12px; background: {colors['surface_secondary']}; border-radius: 12px; border-left: 4px solid {colors['ios_blue']};">
                            <h3 style="margin: 0; color: {colors['text_primary']};">📋 {header_text}</h3>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                        <div style="margin: 15px 0 8px 0;">
                            <h4 style="margin: 0; color: {colors['text_primary']}; border-bottom: 2px solid {colors['ios_orange']}; display: inline-block; padding-bottom: 4px;">✨ {header_text}</h4>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                else:
                    current_section.append(line)

            i += 1

        # Render any remaining content
        if current_section:
            if in_code_block:
                code_content = "\n".join(current_section)
                if code_content.strip():
                    st.code(code_content, language=code_language)
            else:
                text_content = "\n".join(current_section).strip()
                if text_content:
                    # Apply some styling to regular text content
                    formatted_text = self._format_text_content(text_content, colors)
                    st.markdown(formatted_text, unsafe_allow_html=True)

    def _format_text_content(self, text: str, colors: dict) -> str:
        """Apply better formatting to text content"""
        if not text:
            return text

        # Split into paragraphs and format each
        paragraphs = text.split("\n\n")
        formatted_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Handle lists
            if para.startswith("- ") or para.startswith("* "):
                # Format as a styled list
                list_items = []
                for line in para.split("\n"):
                    line = line.strip()
                    if line.startswith(("- ", "* ")):
                        item_text = line[2:].strip()
                        list_items.append(
                            f"""
                        <div style="display: flex; align-items: flex-start; margin: 8px 0; padding: 8px; background: {colors['surface_secondary']}; border-radius: 8px;">
                            <span style="color: {colors['ios_blue']}; font-weight: bold; margin-right: 8px;">•</span>
                            <span style="flex: 1; color: {colors['text_primary']};">{item_text}</span>
                        </div>
                        """
                        )

                if list_items:
                    formatted_paragraphs.append(
                        f"""
                    <div style="margin: 16px 0;">
                        {''.join(list_items)}
                    </div>
                    """
                    )
            else:
                # Regular paragraph with enhanced readability
                formatted_paragraphs.append(
                    f"""
                <div style="margin: 16px 0; padding: 16px; background: {colors['surface_secondary']}; border-radius: 12px; line-height: 1.6;">
                    <p style="margin: 0; color: {colors['text_primary']};">{para}</p>
                </div>
                """
                )

        return "".join(formatted_paragraphs)

    def render_performance_heatmap_calendar(self) -> None:
        """Render a GitHub-style activity heatmap calendar"""
        import plotly.graph_objects as go
        from datetime import datetime, timedelta

        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">🔥 Performance Heatmap Calendar</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Generate calendar data for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=364)  # ~1 year

        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Generate activity data based on completed challenges
        completed_challenges = self.game.progress["challenges_completed"]
        activity_data = []

        rng = np.random.default_rng(42)  # For consistent demo data

        for date in dates:
            # Simulate activity based on challenges completed
            base_activity = len(completed_challenges) // 10  # Base activity level
            daily_variation = rng.integers(0, 5)  # Random daily variation
            activity_score = min(base_activity + daily_variation, 10)  # Cap at 10

            activity_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "activity": activity_score,
                    "weekday": date.weekday(),
                    "week": date.isocalendar()[1],
                    "month": date.strftime("%b"),
                }
            )

        activity_df = pd.DataFrame(activity_data)

        # Create heatmap data
        weeks = activity_df.groupby(["week"])["activity"].apply(list).reset_index()

        # Reshape for calendar view (7 days x ~52 weeks)
        calendar_data = []
        week_labels = []

        for _, week_data in weeks.iterrows():
            week_activities = week_data["activity"]
            # Pad to 7 days if needed
            while len(week_activities) < 7:
                week_activities.append(0)
            calendar_data.append(week_activities[:7])
            week_labels.append(f"Week {week_data['week']}")

        # Create the heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=list(zip(*calendar_data)),  # Transpose to get days as rows
                colorscale=[
                    [0.0, "#F3F4F6"],  # Light gray for no activity
                    [0.2, "#9CA3AF"],  # Gray for low activity
                    [0.4, "#6366F1"],  # Purple for medium activity
                    [0.6, "#4F46E5"],  # Darker purple
                    [0.8, "#4338CA"],  # Even darker purple
                    [1.0, "#3730A3"],  # Darkest purple for high activity
                ],
                showscale=False,
                hovertemplate="Activity Level: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            plot_bgcolor=self.TRANSPARENT_BG,
            paper_bgcolor=self.TRANSPARENT_BG,
            font={"family": self.SF_FONT, "color": "#1C1C1E"},
            margin={"t": 20, "b": 50, "l": 80, "r": 20},
            height=200,
            xaxis={
                "showgrid": False,
                "showticklabels": False,
                "showline": False,
                "zeroline": False,
            },
            yaxis={
                "showgrid": False,
                "tickmode": "array",
                "tickvals": [0, 1, 2, 3, 4, 5, 6],
                "ticktext": ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                "showline": False,
                "zeroline": False,
            },
        )

        st.plotly_chart(fig, use_container_width=True)

        # Activity summary
        total_days = len(activity_df)
        active_days = len(activity_df[activity_df["activity"] > 0])
        activity_percentage = (active_days / total_days) * 100

        st.markdown(
            f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 16px 0; padding: 16px; background: var(--surface-secondary); border-radius: 12px;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-green);">{active_days}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Active Days</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-blue);">{activity_percentage:.1f}%</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Activity Rate</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-purple);">{len(completed_challenges)}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Challenges</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_skills_radar_chart(self) -> None:
        """Render a radar chart showing skill proficiency across different areas"""
        import plotly.graph_objects as go

        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">🧠 Skills Proficiency Radar</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Calculate skill levels based on completed challenges
        completed_challenges = self.game.progress["challenges_completed"]
        current_level = self.game.get_current_level()

        # Define skill categories and their associated challenges
        skill_mapping = {
            "Data Cleaning": [
                "level_1_3_data_types",
                "level_2_1_advanced_cleaning",
                "level_2_3_data_cleaning",
            ],
            "Visualization": [
                "level_1_2_visualization",
                "level_3_1_visualization_mastery",
            ],
            "Statistics": [
                "level_2_2_statistical_analysis",
                "level_6_4_advanced_statistics",
            ],
            "Machine Learning": [
                "level_4_1_first_ml_models",
                "level_4_2_feature_engineering",
                "level_4_3_model_evaluation",
            ],
            "Deep Learning": ["level_5_1_advanced_ml_deep_learning"],
            "MLOps": [
                "level_7_challenge_2_advanced_mlops",
                "level_7_1_end_to_end_project",
            ],
            "Data Engineering": [
                "level_6_2_anomaly_detection",
                "level_7_challenge_3_realtime_analytics",
            ],
            "AI Ethics": ["level_7_challenge_4_ai_ethics_governance"],
        }

        # Calculate proficiency for each skill
        skills = []
        proficiency_scores = []

        for skill, related_challenges in skill_mapping.items():
            # Count completed challenges in this skill area
            completed_in_skill = sum(
                1
                for challenge in completed_challenges
                if challenge in related_challenges
            )
            total_in_skill = len(related_challenges)

            if total_in_skill > 0:
                base_score = (completed_in_skill / total_in_skill) * 100
            else:
                base_score = 0

            # Add bonus based on current level and overall progress
            level_bonus = min(current_level * 5, 25)  # Up to 25 points for level
            challenge_bonus = min(
                len(completed_challenges) * 2, 20
            )  # Up to 20 points for challenges

            total_score = min(base_score + level_bonus + challenge_bonus, 100)

            skills.append(skill)
            proficiency_scores.append(total_score)

        # Create radar chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=proficiency_scores + [proficiency_scores[0]],  # Close the shape
                theta=skills + [skills[0]],  # Close the shape
                fill="toself",
                fillcolor="rgba(99, 102, 241, 0.2)",
                line=dict(color="rgba(99, 102, 241, 0.8)", width=3),
                marker=dict(color="rgba(99, 102, 241, 1)", size=8),
                name="Your Skills",
                hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
            )
        )

        # Add target/max level for comparison
        max_scores = [100] * len(skills) + [100]
        fig.add_trace(
            go.Scatterpolar(
                r=max_scores,
                theta=skills + [skills[0]],
                fill="none",
                line=dict(color="rgba(156, 163, 175, 0.3)", width=1, dash="dash"),
                marker=dict(color="rgba(156, 163, 175, 0.3)", size=4),
                name="Expert Level",
                hovertemplate="Expert Level: %{r}%<extra></extra>",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix="%",
                    gridcolor="rgba(142, 142, 147, 0.2)",
                    linecolor="rgba(142, 142, 147, 0.2)",
                ),
                angularaxis=dict(
                    gridcolor="rgba(142, 142, 147, 0.2)",
                    linecolor="rgba(142, 142, 147, 0.2)",
                ),
                bgcolor=self.TRANSPARENT_BG,
            ),
            plot_bgcolor=self.TRANSPARENT_BG,
            paper_bgcolor=self.TRANSPARENT_BG,
            font={"family": self.SF_FONT, "color": "#1C1C1E"},
            margin={"t": 50, "b": 50, "l": 50, "r": 50},
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Skills summary
        avg_proficiency = (
            sum(proficiency_scores) / len(proficiency_scores)
            if proficiency_scores
            else 0
        )
        top_skill = (
            skills[proficiency_scores.index(max(proficiency_scores))]
            if proficiency_scores
            else "None"
        )
        skills_above_50 = sum(1 for score in proficiency_scores if score >= 50)

        st.markdown(
            f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 16px 0; padding: 16px; background: var(--surface-secondary); border-radius: 12px;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-blue);">{avg_proficiency:.1f}%</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Average</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; font-weight: 700; color: var(--ios-green);">{top_skill}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Strongest Skill</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-purple);">{skills_above_50}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Skills > 50%</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_milestone_timeline(self) -> None:
        """Render a visual timeline of learning milestones and achievements"""
        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">🎯 Learning Journey Timeline</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Get player progress data
        completed_challenges = self.game.progress["challenges_completed"]
        badges_earned = self.game.progress["badges_earned"]
        current_level = self.game.get_current_level()
        xp = self.game.progress["experience_points"]

        # Define major milestones
        milestones = [
            {
                "title": "Welcome to Data Science!",
                "description": "Started your data science journey",
                "icon": "🎯",
                "level": 1,
                "required_challenges": 0,
                "required_badges": 0,
                "color": "var(--ios-green)",
            },
            {
                "title": "First Steps Completed",
                "description": "Mastered basic data exploration",
                "icon": "👶",
                "level": 1,
                "required_challenges": 4,
                "required_badges": 1,
                "color": "var(--ios-blue)",
            },
            {
                "title": "Data Analyst",
                "description": "Advanced data cleaning and analysis",
                "icon": "📊",
                "level": 2,
                "required_challenges": 8,
                "required_badges": 2,
                "color": "var(--ios-purple)",
            },
            {
                "title": "Visualization Master",
                "description": "Created stunning data visualizations",
                "icon": "🎨",
                "level": 3,
                "required_challenges": 12,
                "required_badges": 3,
                "color": "var(--ios-orange)",
            },
            {
                "title": "ML Practitioner",
                "description": "Built your first machine learning models",
                "icon": "🤖",
                "level": 4,
                "required_challenges": 16,
                "required_badges": 4,
                "color": "var(--ios-red)",
            },
            {
                "title": "Algorithm Expert",
                "description": "Advanced ML and deep learning mastery",
                "icon": "🧠",
                "level": 5,
                "required_challenges": 20,
                "required_badges": 5,
                "color": "var(--ios-green)",
            },
            {
                "title": "Data Science Master",
                "description": "End-to-end project completion",
                "icon": "🎓",
                "level": 6,
                "required_challenges": 25,
                "required_badges": 6,
                "color": "var(--ios-blue)",
            },
            {
                "title": "Industry Professional",
                "description": "Production-ready MLOps expertise",
                "icon": "🚀",
                "level": 7,
                "required_challenges": 30,
                "required_badges": 7,
                "color": "var(--ios-purple)",
            },
        ]

        # Create timeline visualization
        timeline_html = """
        <div style="position: relative; margin: 20px 0;">
        """

        for i, milestone in enumerate(milestones):
            # Check if milestone is completed
            challenges_completed = (
                len(completed_challenges) >= milestone["required_challenges"]
            )
            badges_completed = len(badges_earned) >= milestone["required_badges"]
            level_reached = current_level >= milestone["level"]

            is_completed = challenges_completed and level_reached
            is_current = (
                current_level == milestone["level"]
                and len(completed_challenges) < milestone["required_challenges"]
            )

            # Determine status
            if is_completed:
                status_color = milestone["color"]
                status_opacity = "1.0"
                status_icon = "✅"
                progress_width = "100%"
            elif is_current:
                status_color = milestone["color"]
                status_opacity = "0.7"
                status_icon = "⏳"
                # Calculate progress within current milestone
                if i > 0:
                    prev_req = milestones[i - 1]["required_challenges"]
                    current_req = milestone["required_challenges"]
                    current_progress = len(completed_challenges) - prev_req
                    max_progress = current_req - prev_req
                    progress_percent = (
                        min(100, (current_progress / max_progress) * 100)
                        if max_progress > 0
                        else 0
                    )
                else:
                    progress_percent = (
                        min(
                            100,
                            (
                                len(completed_challenges)
                                / milestone["required_challenges"]
                            )
                            * 100,
                        )
                        if milestone["required_challenges"] > 0
                        else 100
                    )
                progress_width = f"{progress_percent}%"
            else:
                status_color = "var(--text-tertiary)"
                status_opacity = "0.3"
                status_icon = "⭕"
                progress_width = "0%"

            # Create timeline item
            timeline_html += f"""
            <div style="display: flex; align-items: center; margin: 24px 0; position: relative;">
                <!-- Timeline line -->
                <div style="position: absolute; left: 25px; top: 50px; width: 2px; height: 60px;
                           background: linear-gradient(to bottom, {status_color} 0%, rgba(142, 142, 147, 0.2) 100%);
                           opacity: {status_opacity};"></div>

                <!-- Milestone icon -->
                <div style="width: 50px; height: 50px; border-radius: 25px;
                           background: {status_color}; color: white;
                           display: flex; align-items: center; justify-content: center;
                           font-size: 1.5rem; margin-right: 20px; z-index: 1;
                           opacity: {status_opacity}; position: relative;">
                    {milestone['icon']}
                </div>

                <!-- Milestone content -->
                <div style="flex: 1; padding: 16px; background: var(--surface-secondary);
                           border-radius: 12px; border-left: 4px solid {status_color};
                           opacity: {status_opacity};">
                    <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: var(--text-primary);">{milestone['title']}</h4>
                        <span style="font-size: 1.2rem;">{status_icon}</span>
                    </div>
                    <p style="margin: 0 0 12px 0; color: var(--text-secondary); font-size: 0.9rem;">
                        {milestone['description']}
                    </p>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <small style="color: var(--text-tertiary);">
                            Level {milestone['level']} • {milestone['required_challenges']} challenges • {milestone['required_badges']} badges
                        </small>
                        <div style="width: 100px; height: 4px; background: rgba(142, 142, 147, 0.2);
                                   border-radius: 2px; overflow: hidden;">
                            <div style="width: {progress_width}; height: 100%; background: {status_color};
                                       border-radius: 2px; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                </div>
            </div>
            """

        timeline_html += "</div>"

        st.markdown(timeline_html, unsafe_allow_html=True)

        # Progress summary
        completed_milestones = sum(
            1
            for m in milestones
            if len(completed_challenges) >= m["required_challenges"]
            and current_level >= m["level"]
        )

        next_milestone = None
        for m in milestones:
            if (
                len(completed_challenges) < m["required_challenges"]
                or current_level < m["level"]
            ):
                next_milestone = m
                break

        st.markdown(
            f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 30px 0; padding: 20px; background: var(--surface-secondary); border-radius: 12px;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: var(--ios-blue);">{completed_milestones}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Milestones Reached</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: var(--ios-green);">{xp:,}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Total Experience</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; font-weight: 700; color: var(--ios-purple);">
                    {next_milestone["title"] if next_milestone else "All Complete!"}
                </div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">
                    {"Next Milestone" if next_milestone else "Journey Complete"}
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_gamification_streak_tracker(self) -> None:
        """Render an engaging streak tracker and achievement progress display"""
        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">🔥 Learning Streak & Achievements</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Get progress data
        completed_challenges = self.game.progress["challenges_completed"]
        badges_earned = self.game.progress["badges_earned"]
        current_level = self.game.get_current_level()
        xp = self.game.progress["experience_points"]

        # Calculate streak (simplified - based on recent activity)
        current_streak = min(7, len(completed_challenges) // 5)  # Simulated streak
        best_streak = max(current_streak, 12)  # All-time best

        # Calculate daily goals progress
        daily_challenges_goal = 2
        daily_challenges_completed = min(
            daily_challenges_goal, len(completed_challenges) % 3 + 1
        )  # Simulated

        weekly_goal = 10
        weekly_completed = min(
            weekly_goal, len(completed_challenges) % 15 + 5
        )  # Simulated

        # Create three columns for streak display
        col1, col2, col3 = st.columns(3)

        with col1:
            # Current Streak
            streak_color = (
                "var(--ios-orange)" if current_streak >= 5 else "var(--ios-blue)"
            )
            st.markdown(
                f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {streak_color}22, {streak_color}11); border-radius: 16px; border: 2px solid {streak_color}44;">
                <div style="font-size: 3rem; margin-bottom: 8px;">🔥</div>
                <div style="font-size: 2.5rem; font-weight: 800; color: {streak_color}; margin-bottom: 4px;">{current_streak}</div>
                <div style="font-size: 0.9rem; color: var(--text-primary); margin-bottom: 8px;">Day Streak</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary);">Best: {best_streak} days</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            # Daily Progress
            daily_progress = (daily_challenges_completed / daily_challenges_goal) * 100
            progress_color = (
                "var(--ios-green)" if daily_progress >= 100 else "var(--ios-blue)"
            )

            st.markdown(
                f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {progress_color}22, {progress_color}11); border-radius: 16px; border: 2px solid {progress_color}44;">
                <div style="font-size: 3rem; margin-bottom: 8px;">🎯</div>
                <div style="font-size: 2rem; font-weight: 800; color: {progress_color}; margin-bottom: 4px;">{daily_challenges_completed}/{daily_challenges_goal}</div>
                <div style="font-size: 0.9rem; color: var(--text-primary); margin-bottom: 8px;">Daily Goal</div>
                <div style="width: 100%; height: 6px; background: rgba(142, 142, 147, 0.2); border-radius: 3px; margin: 8px 0;">
                    <div style="width: {min(100, daily_progress)}%; height: 100%; background: {progress_color}; border-radius: 3px; transition: width 0.5s ease;"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            # Weekly Progress
            weekly_progress = (weekly_completed / weekly_goal) * 100
            weekly_color = (
                "var(--ios-purple)" if weekly_progress >= 80 else "var(--ios-blue)"
            )

            st.markdown(
                f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {weekly_color}22, {weekly_color}11); border-radius: 16px; border: 2px solid {weekly_color}44;">
                <div style="font-size: 3rem; margin-bottom: 8px;">📅</div>
                <div style="font-size: 2rem; font-weight: 800; color: {weekly_color}; margin-bottom: 4px;">{weekly_completed}/{weekly_goal}</div>
                <div style="font-size: 0.9rem; color: var(--text-primary); margin-bottom: 8px;">Weekly Goal</div>
                <div style="width: 100%; height: 6px; background: rgba(142, 142, 147, 0.2); border-radius: 3px; margin: 8px 0;">
                    <div style="width: {min(100, weekly_progress)}%; height: 100%; background: {weekly_color}; border-radius: 3px; transition: width 0.5s ease;"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Achievement Progress Section
        st.markdown(
            """
        <div style="margin: 30px 0 15px 0;">
            <h4 style="margin: 0;">🏆 Achievement Progress</h4>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Define achievement categories with progress
        achievements = [
            {
                "name": "Challenge Crusher",
                "description": "Complete 20 challenges",
                "progress": min(100, (len(completed_challenges) / 20) * 100),
                "current": len(completed_challenges),
                "target": 20,
                "icon": "⚡",
                "color": "var(--ios-orange)",
            },
            {
                "name": "Level Master",
                "description": "Reach Level 5",
                "progress": min(100, (current_level / 5) * 100),
                "current": current_level,
                "target": 5,
                "icon": "🎖️",
                "color": "var(--ios-red)",
            },
            {
                "name": "Badge Collector",
                "description": "Earn 10 badges",
                "progress": min(100, (len(badges_earned) / 10) * 100),
                "current": len(badges_earned),
                "target": 10,
                "icon": "🏅",
                "color": "var(--ios-green)",
            },
            {
                "name": "XP Champion",
                "description": "Earn 5,000 experience points",
                "progress": min(100, (xp / 5000) * 100),
                "current": xp,
                "target": 5000,
                "icon": "💎",
                "color": "var(--ios-purple)",
            },
        ]

        # Display achievement progress bars
        for achievement in achievements:
            is_completed = achievement["progress"] >= 100
            progress_color = (
                achievement["color"] if not is_completed else "var(--ios-green)"
            )

            st.markdown(
                f"""
            <div style="margin: 16px 0; padding: 16px; background: var(--surface-secondary); border-radius: 12px;
                       border-left: 4px solid {achievement['color']};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 12px;">{achievement['icon']}</span>
                        <div>
                            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 2px;">
                                {achievement['name']} {"✅" if is_completed else ""}
                            </div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary);">
                                {achievement['description']}
                            </div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 600; color: {progress_color};">
                            {achievement['current']}/{achievement['target']}
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            {achievement['progress']:.1f}%
                        </div>
                    </div>
                </div>
                <div style="width: 100%; height: 8px; background: rgba(142, 142, 147, 0.2); border-radius: 4px; overflow: hidden;">
                    <div style="width: {achievement['progress']}%; height: 100%; background: {progress_color};
                               border-radius: 4px; transition: width 0.8s ease;
                               {'box-shadow: 0 0 10px ' + progress_color + '44;' if is_completed else ''}">
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Motivational message based on progress
        total_progress = sum(a["progress"] for a in achievements) / len(achievements)

        if total_progress >= 80:
            message = (
                "🌟 Incredible progress! You're becoming a true data science expert!"
            )
            color = "var(--ios-green)"
        elif total_progress >= 50:
            message = "🚀 Great momentum! Keep pushing towards mastery!"
            color = "var(--ios-blue)"
        elif total_progress >= 25:
            message = "📈 Solid foundation! You're on the right track!"
            color = "var(--ios-purple)"
        else:
            message = "🌱 Every expert was once a beginner. Keep learning!"
            color = "var(--ios-orange)"

        st.markdown(
            f"""
        <div style="text-align: center; margin: 25px 0; padding: 20px;
                   background: linear-gradient(135deg, {color}22, {color}11);
                   border-radius: 16px; border: 2px solid {color}44;">
            <div style="font-size: 1.1rem; font-weight: 600; color: {color};">
                {message}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
