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
                st.rerun()

        st.sidebar.markdown("---")

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
                    challenges = self.game.get_level_challenges(level_num)
                    if challenges:
                        completed_count = len(
                            [
                                c
                                for c in self.game.progress["challenges_completed"]
                                if c.startswith(f"level_{level_num}")
                            ]
                        )
                        completion_pct = (completed_count / len(challenges)) * 100
                    else:
                        completion_pct = 0
                else:
                    completion_pct = 100 if status["completed"] else 0

                st.markdown(
                    f"""
                <div class="ios-card level-{card_class}" style="position: relative;">
                    <div style="position: relative; z-index: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                            <h3 style="margin: 0; color: var(--text-primary);">Level {level_num}: {level_info['name']}</h3>
                            <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 4px 12px;">
                                <span style="font-size: 0.8rem; font-weight: 600; color: {status_color};">{completion_pct:.0f}%</span>
                            </div>
                        </div>
                        <p style="margin: 0 0 16px 0; color: var(--text-secondary); line-height: 1.5;">{level_info['description']}</p>

                        <!-- Progress bar -->
                        <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 6px; margin: 16px 0;">
                            <div style="background: {status_color}; height: 100%; border-radius: 10px; width: {completion_pct}%;
                                       transition: width 0.3s ease;"></div>
                        </div>
                """,
                    unsafe_allow_html=True,
                )

                # Show challenges info
                if status["unlocked"]:
                    challenges = self.game.get_level_challenges(level_num)
                    if challenges:
                        completed_count = len(
                            [
                                c
                                for c in self.game.progress["challenges_completed"]
                                if c.startswith(f"level_{level_num}")
                            ]
                        )

                        st.markdown(
                            f"""
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center;">
                                <span style="font-size: 1.2rem; margin-right: 8px;">📚</span>
                                <span style="font-size: 0.9rem; color: var(--text-secondary);">
                                    {completed_count}/{len(challenges)} challenges completed
                                </span>
                            </div>
                            <div style="display: flex; gap: 8px;">
                        """,
                            unsafe_allow_html=True,
                        )

                        # Show first few challenge indicators
                        for _i, challenge in enumerate(challenges[:5]):
                            completed = (
                                f"level_{level_num}_{challenge}"
                                in self.game.progress["challenges_completed"]
                            )
                            dot_color = (
                                Dashboard.IOS_GREEN
                                if completed
                                else "rgba(255,255,255,0.3)"
                            )
                            st.markdown(
                                f"""
                                <div style="width: 8px; height: 8px; background: {dot_color};
                                           border-radius: 50%; display: inline-block;"></div>
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
                if status["unlocked"] and not status["completed"]:
                    if st.button(
                        "▶️", key=f"start_{level_num}", help=f"Start Level {level_num}"
                    ):
                        st.session_state.page = "Challenges"
                        st.success(f"🚀 Starting Level {level_num}!")
                        st.rerun()
                elif status["completed"]:
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
                    challenge_id = f"level_{selected_level}_{challenge_name}"
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
                            if completed:
                                st.success(f"📖 Reviewing challenge: {challenge_name}")
                            else:
                                st.success(f"🚀 Started challenge: {challenge_name}")
                                # Here you would typically open the challenge or mark it as started

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

        # Weekly activity heatmap
        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">📊 Weekly Activity Heatmap</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Create mock weekly activity data
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
        activity_data = []
        rng = np.random.default_rng(42)

        for week in weeks:
            for day in days:
                activity_data.append(
                    {"Week": week, "Day": day, "Activity": rng.integers(0, 10)}
                )

        activity_df = pd.DataFrame(activity_data)

        # Pivot for heatmap
        heatmap_data = activity_df.pivot(index="Week", columns="Day", values="Activity")

        fig = px.imshow(
            heatmap_data, color_continuous_scale=["#F9FAFB", "#1D4ED8"], aspect="auto"
        )

        fig.update_layout(
            plot_bgcolor=Dashboard.TRANSPARENT_BG,
            paper_bgcolor=Dashboard.TRANSPARENT_BG,
            font={"family": Dashboard.SF_FONT, "color": "#1C1C1E"},
            margin={"t": 20, "b": 50, "l": 80, "r": 50},
        )

        st.plotly_chart(fig, use_container_width=True)

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
            if st.checkbox(
                "Enable dark mode",
                value=st.session_state.dark_mode,
                key="theme_toggle",
                label_visibility="hidden",
            ):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()

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
                challenge_id = f"level_{current_level}_{challenge_name}"

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
