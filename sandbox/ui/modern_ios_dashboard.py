"""
Modern iOS HIG-Compliant Dashboard for Data Science Sandbox
Complete redesign following iOS Human Interface Guidelines
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import BADGES, LEVELS
from sandbox.core.game_engine import GameEngine
from sandbox.ui.ios_design_system import IOSDesignSystem
from sandbox.utils.dashboard_layout_system import DashboardLayoutSystem


class ModernIOSDashboard:
    """
    iOS HIG-compliant dashboard with professional spacing, typography, and components
    Follows iOS 16+ Human Interface Guidelines for exceptional user experience
    """

    def __init__(self, game_engine: GameEngine):
        self.game = game_engine
        self.colors = self._get_adaptive_colors()

    def _get_adaptive_colors(self) -> Dict[str, str]:
        """Get adaptive colors that work in both light and dark modes"""
        return {
            "primary": "var(--system-blue)",
            "success": "var(--system-green)",
            "warning": "var(--system-orange)",
            "error": "var(--system-red)",
            "background": "var(--system-grouped-background)",
            "surface": "var(--secondary-system-grouped-background)",
            "text_primary": "var(--label)",
            "text_secondary": "var(--secondary-label)",
            "text_tertiary": "var(--tertiary-label)",
            "separator": "var(--separator)",
        }

    def configure_streamlit(self) -> None:
        """Configure Streamlit with iOS-optimized settings"""
        st.set_page_config(
            page_title="Data Science Sandbox",
            page_icon="🎓",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                "Get Help": None,
                "Report a bug": None,
                "About": "# Data Science Sandbox\nA professional learning platform following iOS Human Interface Guidelines",
            },
        )

    def inject_ios_styles(self) -> None:
        """Inject complete iOS HIG design system"""
        st.markdown(IOSDesignSystem.get_css(), unsafe_allow_html=True)

        # Additional custom styles for this specific app
        st.markdown(
            """
        <style>
        /* App-specific styles */
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: var(--spacing-md);
            padding-top: var(--spacing-xs);
        }

        .dashboard-header {
            text-align: center;
            margin: var(--spacing-sm) 0 var(--spacing-lg) 0;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: var(--spacing-md);
            margin: var(--spacing-xl) 0;
        }

        .progress-section {
            margin: var(--spacing-xxl) 0;
        }

        .action-buttons-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: var(--spacing-md);
            margin: var(--spacing-xl) 0;
        }

        .level-progress-container {
            background-color: var(--secondary-system-grouped-background);
            border-radius: var(--corner-radius-large);
            padding: var(--spacing-lg);
            border: 0.5px solid var(--separator);
        }

        .achievement-badge {
            display: inline-flex;
            align-items: center;
            gap: var(--spacing-xs);
            padding: var(--spacing-sm) var(--spacing-md);
            background-color: var(--system-blue);
            color: white;
            border-radius: var(--corner-radius-medium);
            font-weight: 600;
            font-size: 14px;
            line-height: 1.2;
        }

        .streak-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, var(--system-orange), var(--system-red));
            border-radius: 30px;
            color: white;
            font-weight: 700;
            font-size: 18px;
        }

        /* Override Streamlit components to match iOS style */
        .stTabs [data-baseweb="tab-list"] {
            gap: var(--spacing-xs);
            background-color: transparent;
            border-bottom: 0.5px solid var(--separator);
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: var(--spacing-sm) var(--spacing-md);
            background-color: transparent;
            color: var(--secondary-label);
            border: none;
            border-radius: var(--corner-radius-medium) var(--corner-radius-medium) 0 0;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--system-blue);
            color: white;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--system-background);
            border-right: 0.5px solid var(--separator);
            padding: var(--spacing-md);
        }

        .sidebar-section {
            margin: var(--spacing-lg) 0;
        }

        .sidebar-nav-item {
            display: flex;
            align-items: center;
            padding: var(--spacing-md);
            margin: var(--spacing-xs) 0;
            border-radius: var(--corner-radius-medium);
            color: var(--label);
            text-decoration: none;
            transition: all 0.2s ease-in-out;
            cursor: pointer;
            min-height: 44px;
        }

        .sidebar-nav-item:hover {
            background-color: var(--tertiary-system-background);
        }

        .sidebar-nav-item.active {
            background-color: var(--system-blue);
            color: white;
        }

        .sidebar-nav-icon {
            width: 20px;
            height: 20px;
            margin-right: var(--spacing-md);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Additional Streamlit overrides to remove default spacing */
        .main .block-container {
            padding-top: 0 !important;
            padding-bottom: var(--spacing-lg) !important;
        }

        .css-1y4p8pa {
            padding-top: 0 !important;
        }

        .css-12oz5g7 {
            padding-top: 0 !important;
        }

        /* Remove default Streamlit header space */
        .css-1rs6os .css-17eq0hr {
            padding-top: 0 !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def create_modern_sidebar(self) -> None:
        """Create iOS-style sidebar navigation"""
        with st.sidebar:
            # App Header
            st.markdown(
                """
            <div style="text-align: center; padding: 24px 0; border-bottom: 0.5px solid rgba(60, 60, 67, 0.29); margin-bottom: 24px;">
                <div class="title2" style="margin-bottom: 4px;">🎓</div>
                <div class="headline">Data Science Sandbox</div>
                <div class="subheadline">Professional Learning Platform</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # User Progress Summary
            stats = self.game.get_stats()
            st.markdown(
                '<div style="font-size: 13px; line-height: 18px; font-weight: 400; letter-spacing: -0.08px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; margin-bottom: 8px; padding: 0 24px;">PROGRESS OVERVIEW</div>',
                unsafe_allow_html=True,
            )

            progress_metrics_html = f"""
            <div style="padding: 24px; background: #F2F2F7; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; padding: 8px 0;">
                    <div style="display: flex; flex-direction: column; align-items: center;">
                        <div style="font-size: 22px; font-weight: 600; color: #007AFF; margin-bottom: 4px;">{stats['level']}</div>
                        <div style="font-size: 12px; font-weight: 500; color: rgba(60, 60, 67, 0.6);">Level</div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center;">
                        <div style="font-size: 22px; font-weight: 600; color: #34C759; margin-bottom: 4px;">{stats['experience']}</div>
                        <div style="font-size: 12px; font-weight: 500; color: rgba(60, 60, 67, 0.6);">XP</div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center;">
                        <div style="font-size: 22px; font-weight: 600; color: #AF52DE; margin-bottom: 4px;">{stats['badges']}</div>
                        <div style="font-size: 12px; font-weight: 500; color: rgba(60, 60, 67, 0.6);">Badges</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(progress_metrics_html, unsafe_allow_html=True)

            # Navigation Menu
            st.markdown(
                '<div style="font-size: 13px; line-height: 18px; font-weight: 400; letter-spacing: -0.08px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; margin-bottom: 8px; padding: 0 24px;">NAVIGATION</div>',
                unsafe_allow_html=True,
            )

            nav_items = [
                {"icon": "📊", "title": "Dashboard", "page": "Dashboard"},
                {"icon": "🎯", "title": "Levels", "page": "Levels"},
                {"icon": "🏆", "title": "Challenges", "page": "Challenges"},
                {"icon": "🏅", "title": "Badges", "page": "Badges"},
                {"icon": "📈", "title": "Progress", "page": "Progress"},
                {"icon": "⚙️", "title": "Settings", "page": "Settings"},
            ]

            for item in nav_items:
                if st.button(
                    f"{item['icon']} {item['title']}",
                    key=f"nav_{item['page']}",
                    use_container_width=True,
                ):
                    st.session_state.page = item["page"]
                    st.rerun()

    def show_dashboard_home(self) -> None:
        """Main dashboard view with iOS HIG design"""
        # Header Section
        st.markdown(
            """
        <div class="dashboard-header">
            <h1 class="large-title">Welcome Back</h1>
            <p class="subheadline">Continue your data science journey with professionally designed challenges</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Key Metrics Section
        self._render_key_metrics()

        # Progress Section
        self._render_progress_overview()

        # Recent Activity & Quick Actions
        self._render_activity_section()

        # Current Focus Area
        self._render_current_focus()

    def _render_key_metrics(self) -> None:
        """Render key performance metrics with iOS cards"""
        st.markdown(
            '<div style="font-size: 13px; line-height: 18px; font-weight: 400; letter-spacing: -0.08px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; margin-bottom: 8px; padding: 0 24px;">OVERVIEW</div>',
            unsafe_allow_html=True,
        )

        stats = self.game.get_stats()
        completion_rate = stats.get("completion_rate", 0)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div style="background: #F2F2F7; border-radius: 20px; padding: 24px; text-align: center;">
                    <div style="font-size: 32px; font-weight: 600; color: #007AFF; margin-bottom: 8px;">{stats["level"]}</div>
                    <div style="font-size: 13px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; letter-spacing: 0.5px;">Current Level</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: #F2F2F7; border-radius: 20px; padding: 24px; text-align: center;">
                    <div style="font-size: 32px; font-weight: 600; color: #34C759; margin-bottom: 8px;">{stats["experience"]}</div>
                    <div style="font-size: 13px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; letter-spacing: 0.5px;">Experience Points</div>
                    <div style="font-size: 12px; color: #34C759; margin-top: 4px;">+50 recent</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div style="background: #F2F2F7; border-radius: 20px; padding: 24px; text-align: center;">
                    <div style="font-size: 32px; font-weight: 600; color: #FF9500; margin-bottom: 8px;">{completion_rate:.1f}%</div>
                    <div style="font-size: 13px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; letter-spacing: 0.5px;">Completion Rate</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div style="background: #F2F2F7; border-radius: 20px; padding: 24px; text-align: center;">
                    <div style="font-size: 32px; font-weight: 600; color: #AF52DE; margin-bottom: 8px;">{stats["badges"]}</div>
                    <div style="font-size: 13px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; letter-spacing: 0.5px;">Badges Earned</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    def _render_progress_overview(self) -> None:
        """Render progress overview with modern charts"""
        st.markdown(
            '<div style="font-size: 13px; line-height: 18px; font-weight: 400; letter-spacing: -0.08px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; margin-bottom: 8px; padding: 0 24px;">LEARNING PROGRESS</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # Progress Chart
            progress_data = self._get_progress_data()

            fig = px.bar(
                progress_data,
                x="Level",
                y="Completion",
                color="Status",
                color_discrete_map={
                    "Completed": "#34C759",  # system-green
                    "In Progress": "#FF9500",  # system-orange
                    "Locked": "#8E8E93",  # system-gray
                },
                title="Level Completion Progress",
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="SF Pro Display, -apple-system, sans-serif"),
                title_font=dict(size=20, color="var(--label)"),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            fig.update_xaxes(
                gridcolor="var(--separator)",
                title_text="",
                tickfont=dict(color="var(--secondary-label)"),
            )
            fig.update_yaxes(
                gridcolor="var(--separator)",
                title_text="Completion %",
                tickfont=dict(color="var(--secondary-label)"),
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Streak & Achievement Summary
            st.markdown(
                """
            <div style="padding: 24px; background: #F2F2F7; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 17px; font-weight: 600; margin-bottom: 24px; text-align: center; color: #000000;">Current Streak</div>
                <div style="display: flex; justify-content: center; margin-bottom: 24px;">
                    <div style="font-size: 32px; font-weight: bold; color: #FF9500;">🔥7</div>
                </div>
                <div style="font-size: 15px; text-align: center; color: rgba(60, 60, 67, 0.6);">
                    Days of consistent learning
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Recent Achievements - iOS system background
            st.markdown(
                """
                <div style="padding: 24px; background: #F2F2F7; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 16px;">
                    <div style="font-size: 17px; font-weight: 600; margin-bottom: 16px; color: #000000;">Recent Achievements</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <div style="display: inline-flex; align-items: center; background: rgba(0, 122, 255, 0.1); color: #007AFF; padding: 6px 12px; border-radius: 16px; font-size: 14px; font-weight: 500; margin-right: 8px;">
                            🏆 First Steps
                        </div>
                        <div style="display: inline-flex; align-items: center; background: rgba(0, 122, 255, 0.1); color: #007AFF; padding: 6px 12px; border-radius: 16px; font-size: 14px; font-weight: 500; margin-right: 8px;">
                            📊 Data Explorer
                        </div>
                        <div style="display: inline-flex; align-items: center; background: rgba(0, 122, 255, 0.1); color: #007AFF; padding: 6px 12px; border-radius: 16px; font-size: 14px; font-weight: 500; margin-right: 8px;">
                            🎯 Challenge Master
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_activity_section(self) -> None:
        """Render recent activity and quick actions"""
        st.markdown(
            '<div style="font-size: 13px; line-height: 18px; font-weight: 400; letter-spacing: -0.08px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; margin-bottom: 8px; padding: 0 24px;">QUICK ACTIONS</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "📚 Continue Learning",
                key="continue_learning",
                use_container_width=True,
            ):
                st.session_state.page = "Challenges"
                st.rerun()

        with col2:
            if st.button(
                "🏆 View Achievements",
                key="view_achievements",
                use_container_width=True,
            ):
                st.session_state.page = "Badges"
                st.rerun()

        with col3:
            if st.button(
                "📊 Detailed Progress",
                key="detailed_progress",
                use_container_width=True,
            ):
                st.session_state.page = "Progress"
                st.rerun()

    def _render_current_focus(self) -> None:
        """Render current learning focus area"""
        st.markdown(
            '<div style="font-size: 13px; line-height: 18px; font-weight: 400; letter-spacing: -0.08px; color: rgba(60, 60, 67, 0.6); text-transform: uppercase; margin-bottom: 8px; padding: 0 24px;">CURRENT FOCUS</div>',
            unsafe_allow_html=True,
        )

        # Current Focus - iOS system background
        st.markdown(
            """
            <div style="padding: 24px; background: #F2F2F7; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 24px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 24px;">
                    <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #007AFF, #AF52DE); border-radius: 20px; display: flex; align-items: center; justify-content: center; margin-right: 24px; color: white; font-size: 24px;">
                        🎯
                    </div>
                    <div style="flex: 1;">
                        <div style="font-size: 20px; font-weight: 600; margin-bottom: 4px; color: #000000;">Level 7: Modern Toolchain Master</div>
                        <div style="font-size: 15px; color: rgba(60, 60, 67, 0.6);">Master modern data science tools and workflows</div>
                    </div>
                </div>
                <div style="margin-top: 16px;">
                    <div style="font-size: 15px; margin-bottom: 8px; color: rgba(60, 60, 67, 0.6); font-weight: 500;">
                        Progress towards next level
                    </div>
                    <div style="width: 100%; height: 4px; background-color: #E5E5EA; border-radius: 2px; overflow: hidden;">
                        <div style="height: 100%; background-color: #007AFF; border-radius: 2px; width: 70%; transition: width 0.3s ease-in-out;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                        <div style="font-size: 12px; font-weight: 500; color: rgba(60, 60, 67, 0.6);">70.0% Complete</div>
                        <div style="font-size: 12px; font-weight: 500; color: rgba(60, 60, 67, 0.6);">2 challenges remaining</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def show_levels_view(self) -> None:
        """Show levels overview with modern design"""
        st.markdown(
            """
        <div class="dashboard-header">
            <h1 class="large-title">Learning Levels</h1>
            <p class="subheadline">Master each level to unlock new challenges and earn badges</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Create level cards
        for level_num in range(1, 8):
            self._render_level_card(level_num)

    def _render_level_card(self, level_num: int) -> None:
        """Render individual level card"""
        level_info = LEVELS.get(level_num, {})
        current_level = self.game.get_current_level()
        is_unlocked = level_num <= current_level
        is_completed = self._is_level_completed(level_num)

        # Determine status and styling
        if is_completed:
            status_color = "#34C759"
            status_icon = "✅"
            status_text = "COMPLETED"
        elif is_unlocked:
            status_color = "#FF9500"
            status_icon = "🎯"
            status_text = "IN PROGRESS"
        else:
            status_color = "#8E8E93"
            status_icon = "🔒"
            status_text = "LOCKED"

        level_card_html = f"""
        <div class="ios-card" style="margin: 24px 0;">
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="width: 80px; height: 80px; background: {status_color}; border-radius: 20px;
                           display: flex; align-items: center; justify-content: center; margin-right: 24px;
                           color: white; font-size: 32px;">
                    {status_icon}
                </div>
                <div style="flex: 1;">
                    <div class="title2">Level {level_num}: {level_info.get('name', 'Unknown')}</div>
                    <div class="subheadline" style="margin: 4px 0;">{level_info.get('description', '')}</div>
                    <div class="caption1" style="color: {status_color}; font-weight: 600;">{status_text}</div>
                </div>
            </div>
        """

        if is_unlocked:
            # Add progress details
            challenges = level_info.get("challenges", [])
            completed_challenges = self._get_completed_challenges_for_level(level_num)
            progress = len(completed_challenges) / len(challenges) if challenges else 0

            level_card_html += f"""
            <div style="margin-top: 24px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <div class="body">Progress</div>
                    <div class="body">{len(completed_challenges)}/{len(challenges)} challenges</div>
                </div>
            """

            # Add action button
            if not is_completed:
                level_card_html += f"""
                <div style="margin-top: 24px;">
                    <button class="ios-button-primary" style="width: 100%;" onclick="continueLevel({level_num})">
                        Continue Level {level_num}
                    </button>
                </div>
                """

        level_card_html += "</div>"
        st.markdown(level_card_html, unsafe_allow_html=True)

        # Add JavaScript for button interaction
        if (
            is_unlocked
            and not is_completed
            and st.button(f"Continue Level {level_num}", key=f"level_{level_num}")
        ):
            st.session_state.page = "Challenges"
            st.session_state.selected_level = level_num
            st.rerun()

    def run(self) -> None:
        """Main entry point for the modern dashboard"""
        # Configure Streamlit
        self.configure_streamlit()

        # Inject iOS styles
        self.inject_ios_styles()

        # Create sidebar
        self.create_modern_sidebar()

        # Main content area
        with st.container():
            st.markdown('<div class="main-content">', unsafe_allow_html=True)

            # Route to appropriate view
            current_page = st.session_state.get("page", "Dashboard")

            if current_page == "Dashboard":
                self.show_dashboard_home()
            elif current_page == "Levels":
                self.show_levels_view()
            elif current_page == "Challenges":
                self.show_challenges_view()
            elif current_page == "Badges":
                self.show_badges_view()
            elif current_page == "Progress":
                self.show_progress_view()
            elif current_page == "Settings":
                self.show_settings_view()

            st.markdown("</div>", unsafe_allow_html=True)

    def show_challenges_view(self) -> None:
        """Show challenges with modern design - placeholder"""
        st.markdown(
            """
        <div style="padding: 24px 0; text-align: center;">
            <h1 style="font-size: 34px; font-weight: 700; margin-bottom: 8px; color: #000000;">Challenges</h1>
            <p style="font-size: 15px; color: rgba(60, 60, 67, 0.6); margin-bottom: 32px;">Complete challenges to earn experience and unlock new levels</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="padding: 48px 24px; background: #F2F2F7; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; margin: 24px 0;">
                <div style="font-size: 48px; margin-bottom: 16px;">🎯</div>
                <div style="font-size: 22px; font-weight: 600; margin-bottom: 8px; color: #000000;">Challenges Coming Soon</div>
                <div style="font-size: 15px; color: rgba(60, 60, 67, 0.6);">Interactive challenges with real-time feedback and progress tracking are being developed.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def show_badges_view(self) -> None:
        """Show badges with modern design - placeholder"""
        st.markdown(
            """
        <div style="padding: 24px 0; text-align: center;">
            <h1 style="font-size: 34px; font-weight: 700; margin-bottom: 8px; color: #000000;">Achievements</h1>
            <p style="font-size: 15px; color: rgba(60, 60, 67, 0.6); margin-bottom: 32px;">Your earned badges and achievements</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="padding: 48px 24px; background: #F2F2F7; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; margin: 24px 0;">
                <div style="font-size: 48px; margin-bottom: 16px;">🏅</div>
                <div style="font-size: 22px; font-weight: 600; margin-bottom: 8px; color: #000000;">Achievements Gallery</div>
                <div style="font-size: 15px; color: rgba(60, 60, 67, 0.6);">Your earned badges and achievements will be displayed here with detailed unlock conditions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def show_progress_view(self) -> None:
        """Show detailed progress with modern design - placeholder"""
        st.markdown(
            """
        <div style="padding: 24px 0; text-align: center;">
            <h1 style="font-size: 34px; font-weight: 700; margin-bottom: 8px; color: #000000;">Progress Analytics</h1>
            <p style="font-size: 15px; color: rgba(60, 60, 67, 0.6); margin-bottom: 32px;">Detailed insights into your learning journey</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="padding: 48px 24px; background: #F2F2F7; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; margin: 24px 0;">
                <div style="font-size: 48px; margin-bottom: 16px;">📈</div>
                <div style="font-size: 22px; font-weight: 600; margin-bottom: 8px; color: #000000;">Learning Analytics</div>
                <div style="font-size: 15px; color: rgba(60, 60, 67, 0.6);">Detailed insights into your learning patterns, time spent, and skill development progress.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def show_settings_view(self) -> None:
        """Show settings with modern design - placeholder"""
        st.markdown(
            """
        <div style="padding: 24px 0; text-align: center;">
            <h1 style="font-size: 34px; font-weight: 700; margin-bottom: 8px; color: #000000;">Settings</h1>
            <p style="font-size: 15px; color: rgba(60, 60, 67, 0.6); margin-bottom: 32px;">Customize your learning experience</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="padding: 48px 24px; background: #F2F2F7; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; margin: 24px 0;">
                <div style="font-size: 48px; margin-bottom: 16px;">⚙️</div>
                <div style="font-size: 22px; font-weight: 600; margin-bottom: 8px; color: #000000;">Configuration</div>
                <div style="font-size: 15px; color: rgba(60, 60, 67, 0.6);">Preferences for notifications, difficulty levels, learning paths, and personalization settings.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Helper methods
    def _get_progress_data(self) -> pd.DataFrame:
        """Get progress data for visualization"""
        data = []
        for level in range(1, 8):
            completion = self._calculate_level_progress(level) * 100
            if completion >= 100:
                status = "Completed"
            elif completion > 0:
                status = "In Progress"
            else:
                status = "Locked"

            data.append(
                {"Level": f"Level {level}", "Completion": completion, "Status": status}
            )

        return pd.DataFrame(data)

    def _calculate_level_progress(self, level: int) -> float:
        """Calculate progress for a specific level"""
        # Simple calculation - can be made more sophisticated
        current_level = self.game.get_current_level()
        if level < current_level:
            return 1.0
        elif level == current_level:
            # Calculate based on completed challenges
            return min(0.7, level * 0.1)  # Placeholder logic
        else:
            return 0.0

    def _get_recent_achievements(self) -> List[Dict[str, str]]:
        """Get recent achievements"""
        return [
            {"icon": "🏆", "name": "First Steps"},
            {"icon": "📊", "name": "Data Explorer"},
            {"icon": "🎯", "name": "Challenge Master"},
        ]

    def _get_remaining_challenges(self, level: int) -> int:
        """Get number of remaining challenges for level"""
        level_info = LEVELS.get(level, {})
        total_challenges = len(level_info.get("challenges", []))
        completed = len(self._get_completed_challenges_for_level(level))
        return max(0, total_challenges - completed)

    def _is_level_completed(self, level: int) -> bool:
        """Check if a level is completed"""
        return level < self.game.get_current_level()

    def _get_completed_challenges_for_level(self, level: int) -> List[str]:
        """Get completed challenges for a specific level"""
        # Placeholder - would integrate with actual game engine
        return self.game.progress.get("challenges_completed", [])
