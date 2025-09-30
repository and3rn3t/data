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


class Dashboard:
    """Streamlit-based dashboard for the data science sandbox"""

    def __init__(self, game_engine: GameEngine):
        self.game = game_engine

    def run(self):
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
            --ios-blue: #007AFF;
            --ios-gray: #8E8E93;
            --ios-gray-light: #F2F2F7;
            --ios-gray-dark: #1C1C1E;
            --ios-green: #30D158;
            --ios-orange: #FF9500;
            --ios-red: #FF3B30;
            --ios-purple: #AF52DE;
            --ios-pink: #FF2D92;
            --ios-teal: #40C8E0;
            --ios-indigo: #5856D6;
            --ios-yellow: #FFD60A;
            
            --surface-primary: rgba(255, 255, 255, 0.85);
            --surface-secondary: rgba(242, 242, 247, 0.8);
            --surface-tertiary: rgba(255, 255, 255, 0.6);
            --text-primary: #000000;
            --text-secondary: #8E8E93;
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
            margin: 12px 0;
            box-shadow: var(--shadow-medium);
            border: 0.5px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
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
            margin: 16px 0;
            box-shadow: var(--shadow-heavy);
            position: relative;
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
            background: #0056CC;
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
        </style>
        """,
            unsafe_allow_html=True,
        )

        # iOS Dynamic Island Style Header
        st.markdown(
            """
        <div class="dynamic-island animate-slide-in">
            <h1 style="margin: 0; font-size: 1.5rem;">🎮 Data Science Sandbox</h1>
            <p style="margin: 4px 0 0 0; opacity: 0.7; font-size: 0.9rem;">Learn data science through gamified challenges</p>
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

    def create_sidebar(self):
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

        # Navigation with emoji icons
        nav_items = [
            ("Dashboard", "📊", "Overview and progress"),
            ("Levels", "🏆", "Learning progression"),
            ("Challenges", "🎯", "Practice exercises"),
            ("Badges", "🏅", "Achievements"),
            ("Progress", "📈", "Analytics"),
            ("Settings", "⚙️", "Configuration"),
        ]

        for page, icon, description in nav_items:
            is_current = st.session_state.get("page", "Dashboard") == page
            button_style = (
                """
                background: linear-gradient(135deg, var(--ios-blue), var(--ios-purple));
                color: white;
                transform: translateX(4px);
                box-shadow: var(--shadow-medium);
            """
                if is_current
                else ""
            )

            if st.sidebar.button(f"{icon} {page}", key=page, help=description):
                st.session_state.page = page
                st.rerun()

        st.sidebar.markdown("---")

        # Quick Actions with enhanced styling
        st.sidebar.markdown(
            "<h4 style='margin: 20px 0 16px 0; color: var(--text-secondary);'>🚀 Quick Actions</h4>",
            unsafe_allow_html=True,
        )

        # Action buttons with icons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("📊", key="jupyter", help="Launch Jupyter Lab"):
                self.game.launch_jupyter()
                st.sidebar.success("🚀 Launching...")

        with col2:
            if st.button("🔄", key="reset", help="Reset Progress"):
                if st.sidebar.button("⚠️", key="confirm_reset", help="Confirm Reset"):
                    self.game.reset_progress()
                    st.sidebar.success("✅ Reset complete!")
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

    def show_dashboard(self):
        """iOS-inspired main dashboard view"""
        stats = self.game.get_stats()

        # Enhanced Key Metrics with iOS Cards
        st.markdown(
            "<h2 style='margin: 30px 0 20px 0;'>📊 Learning Dashboard</h2>",
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">🏆</div>
                <div style="font-size: 2rem; font-weight: 700; color: var(--ios-blue);">{stats['level']}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 4px;">Current Level</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">out of 6</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">⭐</div>
                <div style="font-size: 2rem; font-weight: 700; color: var(--ios-green);">{stats['experience']}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 4px;">Experience</div>
                <div style="font-size: 0.8rem; color: var(--ios-green); margin-top: 4px;">+50 recent</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">🏅</div>
                <div style="font-size: 2rem; font-weight: 700; color: var(--ios-purple);">{stats['badges']}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 4px;">Badges Earned</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">achievements</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            completion = stats.get("completion_rate", 0)
            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">📈</div>
                <div style="font-size: 2rem; font-weight: 700; color: var(--ios-orange);">{completion:.1f}%</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 4px;">Completion</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">overall progress</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Interactive Progress Section
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                """
            <div class="ios-card">
                <h3 style="margin-bottom: 20px;">📊 Learning Progression</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Enhanced progress chart with iOS styling
            progress_data = []
            for level in range(1, 7):
                level_status = self.game.progress["level_progress"][str(level)]
                progress_data.append(
                    {
                        "Level": f"Level {level}",
                        "Name": LEVELS[level]["name"],
                        "Status": (
                            "Completed"
                            if level_status["completed"]
                            else "Active" if level_status["unlocked"] else "Locked"
                        ),
                        "Progress": (
                            100
                            if level_status["completed"]
                            else 65 if level_status["unlocked"] else 0
                        ),
                    }
                )

            df = pd.DataFrame(progress_data)
            fig = px.bar(
                df,
                x="Level",
                y="Progress",
                color="Status",
                color_discrete_map={
                    "Completed": "#30D158",
                    "Active": "#007AFF",
                    "Locked": "#8E8E93",
                },
                hover_data=["Name"],
            )

            # iOS-style chart customization
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="SF Pro Display, -apple-system, sans-serif"),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                margin=dict(t=50, b=50, l=50, r=50),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(142, 142, 147, 0.2)")

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(
                """
            <div class="ios-card">
                <h3 style="margin-bottom: 16px;">🎯 Current Focus</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            current_level = stats["level"]
            if current_level <= 6:
                level_info = LEVELS[current_level]
                st.markdown(
                    f"""
                <div class="level-card">
                    <div style="position: relative; z-index: 1;">
                        <h3 style="margin: 0 0 12px 0; font-size: 1.3rem;">Level {current_level}</h3>
                        <h4 style="margin: 0 0 16px 0; opacity: 0.9;">{level_info['name']}</h4>
                        <p style="margin: 0; opacity: 0.8; line-height: 1.4;">{level_info['description']}</p>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Interactive challenge preview
                challenges = self.game.get_level_challenges(current_level)
                if challenges:
                    st.markdown(
                        """
                    <div class="ios-card" style="margin-top: 16px;">
                        <h4 style="margin-bottom: 16px;">🔥 Available Challenges</h4>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    for i, challenge in enumerate(challenges[:3]):  # Show first 3
                        completed = (
                            f"level_{current_level}_{challenge}"
                            in self.game.progress["challenges_completed"]
                        )
                        icon = "✅" if completed else "⏳"
                        status_color = (
                            "var(--ios-green)" if completed else "var(--ios-blue)"
                        )

                        st.markdown(
                            f"""
                        <div style="display: flex; align-items: center; padding: 12px; margin: 8px 0; 
                                   background: var(--surface-tertiary); border-radius: 12px;">
                            <span style="font-size: 1.2rem; margin-right: 12px; color: {status_color};">{icon}</span>
                            <div style="flex: 1;">
                                <div style="font-weight: 600; color: var(--text-primary);">{challenge}</div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary);">Challenge {i+1}</div>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

        # Enhanced Recent Activity Section
        st.markdown(
            """
        <div class="ios-card animate-slide-in" style="margin-top: 30px;">
            <h3 style="margin-bottom: 20px;">📈 Recent Activity</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            # Recent badges
            recent_badges = (
                self.game.progress["badges_earned"][-3:]
                if self.game.progress["badges_earned"]
                else []
            )
            if recent_badges:
                st.markdown(
                    "<h4 style='color: var(--text-secondary);'>🏆 Latest Badges</h4>",
                    unsafe_allow_html=True,
                )
                for badge_id in recent_badges:
                    if badge_id in BADGES:
                        badge = BADGES[badge_id]
                        st.markdown(
                            f"""
                        <div class="badge-card earned">
                            <div style="font-size: 2rem; margin-bottom: 8px;">🏆</div>
                            <div style="font-weight: 600; font-size: 1rem;">{badge['name']}</div>
                            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 4px;">{badge['description']}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    """
                <div class="ios-card" style="text-align: center; padding: 40px;">
                    <div style="font-size: 3rem; margin-bottom: 16px;">🎯</div>
                    <h4 style="margin-bottom: 8px;">Start Your Journey</h4>
                    <p style="color: var(--text-secondary); margin: 0;">Complete challenges to earn badges!</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        with col2:
            # Learning streaks and stats
            st.markdown(
                "<h4 style='color: var(--text-secondary);'>🔥 Learning Stats</h4>",
                unsafe_allow_html=True,
            )

            # Mock streak data
            streak_days = 5
            total_challenges = len(self.game.progress["challenges_completed"])

            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">🔥</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-orange);">{streak_days}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Day Streak</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="ios-card" style="text-align: center; margin-top: 12px;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">📚</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--ios-teal);">{total_challenges}</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Challenges Done</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            # Quick actions
            st.markdown(
                "<h4 style='color: var(--text-secondary);'>⚡ Quick Actions</h4>",
                unsafe_allow_html=True,
            )

            if st.button("🚀 Continue Learning", type="primary"):
                st.session_state.page = "Challenges"
                st.rerun()

            if st.button("📊 View Progress", key="view_progress"):
                st.session_state.page = "Progress"
                st.rerun()

            if st.button("🏆 Check Levels", key="check_levels"):
                st.session_state.page = "Levels"
                st.rerun()

    def show_levels(self):
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
                status_color = "var(--ios-green)"
                status_icon = "✅"
                status_text = "Completed"
                card_class = "completed"
            elif status["unlocked"]:
                status_color = "var(--ios-blue)"
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
                        for i, challenge in enumerate(challenges[:5]):
                            completed = (
                                f"level_{level_num}_{challenge}"
                                in self.game.progress["challenges_completed"]
                            )
                            dot_color = (
                                "var(--ios-green)"
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

                        st.markdown("</div></div>", unsafe_allow_html=True)
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

                st.markdown("</div></div>", unsafe_allow_html=True)

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

    def show_challenges(self):
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
        available_levels = list(range(1, min(current_level + 1, 7)))

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

            # Get challenges for selected level
            challenges = self.game.get_level_challenges(selected_level)

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

                for i, challenge in enumerate(challenges, 1):
                    challenge_id = f"level_{selected_level}_{challenge}"
                    completed = (
                        challenge_id in self.game.progress["challenges_completed"]
                    )

                    # Challenge card styling
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

                    col1, col2, col3 = st.columns([1, 4, 1])

                    with col1:
                        st.markdown(
                            f"""
                        <div class="ios-card" style="text-align: center; height: 120px; display: flex; 
                                   flex-direction: column; justify-content: center; {card_style}">
                            <div style="font-size: 2.5rem; margin-bottom: 8px;">{status_icon}</div>
                            <div style="font-size: 0.8rem; font-weight: 600; opacity: 0.8;">{i:02d}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        st.markdown(
                            f"""
                        <div class="ios-card" style="height: 120px; display: flex; flex-direction: column; justify-content: center;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <h4 style="margin: 0; color: var(--text-primary); flex: 1;">{challenge}</h4>
                                <div style="background: var(--surface-tertiary); border-radius: 16px; padding: 4px 12px; 
                                           font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);">
                                    {status_text}
                                </div>
                            </div>
                            <p style="margin: 0; color: var(--text-secondary); font-size: 0.95rem;">
                                Challenge {i} for {level_info['name']} - Practice your skills with hands-on exercises
                            </p>
                            <div style="margin-top: 12px;">
                                <!-- Progress indicator -->
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="flex: 1; height: 4px; background: var(--surface-tertiary); border-radius: 2px;">
                                        <div style="height: 100%; background: {'var(--ios-green)' if completed else 'var(--ios-blue)'}; 
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
                                st.success(f"📖 Reviewing challenge: {challenge}")
                            else:
                                st.success(f"🚀 Started challenge: {challenge}")
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

                st.markdown("</div></div>", unsafe_allow_html=True)
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

            st.markdown("</div></div>", unsafe_allow_html=True)

    def show_badges(self):
        """iOS-inspired badges page"""
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

            st.markdown("</div></div>", unsafe_allow_html=True)

    def show_progress(self):
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
        stats = self.game.get_stats()
        completed_challenges = len(self.game.progress["challenges_completed"])
        earned_badges = len(self.game.progress["badges_earned"])

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

            # Update styling to match iOS aesthetics
            fig.update_traces(
                line=dict(color="#007AFF", width=3),
                fill="tonexty",
                fillcolor="rgba(0, 122, 255, 0.1)",
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="SF Pro Display, -apple-system, sans-serif", color="#1C1C1E"
                ),
                showlegend=False,
                margin=dict(t=20, b=50, l=50, r=20),
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(142, 142, 147, 0.2)", showline=False
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(142, 142, 147, 0.2)", showline=False
                ),
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
                ("🏅", "First Steps", "2 days ago", "var(--ios-green)"),
                ("🎯", "Challenge Master", "5 days ago", "var(--ios-blue)"),
                ("📊", "Data Explorer", "1 week ago", "var(--ios-purple)"),
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
                    "Technical": "#007AFF",
                    "Analytical": "#30D158",
                    "Advanced": "#AF52DE",
                    "Programming": "#FF9500",
                },
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="SF Pro Display, -apple-system, sans-serif", color="#1C1C1E"
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                margin=dict(t=50, b=50, l=100, r=50),
                xaxis=dict(
                    range=[0, 100], showgrid=True, gridcolor="rgba(142, 142, 147, 0.2)"
                ),
                yaxis=dict(showgrid=False),
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
                    "Color": ["#007AFF", "#30D158", "#AF52DE", "#FF9500"],
                }
            )

            # Create donut chart with iOS styling
            fig = px.pie(
                time_data,
                values="Hours",
                names="Category",
                color_discrete_sequence=["#007AFF", "#30D158", "#AF52DE", "#FF9500"],
            )

            # Convert to donut chart
            fig.update_traces(
                hole=0.4,
                textinfo="percent+label",
                textfont_size=12,
                marker=dict(line=dict(color="#FFFFFF", width=2)),
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="SF Pro Display, -apple-system, sans-serif", color="#1C1C1E"
                ),
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50),
                annotations=[
                    dict(
                        text=f'{sum(time_data["Hours"])}h<br>Total',
                        x=0.5,
                        y=0.5,
                        font_size=20,
                        showarrow=False,
                        font=dict(color="#1C1C1E", weight="bold"),
                    )
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

        for week in weeks:
            for day in days:
                activity_data.append(
                    {"Week": week, "Day": day, "Activity": np.random.randint(0, 10)}
                )

        activity_df = pd.DataFrame(activity_data)

        # Pivot for heatmap
        heatmap_data = activity_df.pivot(index="Week", columns="Day", values="Activity")

        fig = px.imshow(
            heatmap_data, color_continuous_scale=["#F2F2F7", "#007AFF"], aspect="auto"
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="SF Pro Display, -apple-system, sans-serif", color="#1C1C1E"
            ),
            margin=dict(t=20, b=50, l=80, r=50),
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

    def show_settings(self):
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

            difficulty = st.radio(
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

            notifications = st.checkbox("Enable learning reminders", value=True)
            badge_notifications = st.checkbox("Badge achievement alerts", value=True)
            weekly_summary = st.checkbox("Weekly progress summary", value=False)

        with col2:
            st.markdown(
                """
            <div class="ios-card" style="background: var(--surface-secondary);">
                <h4 style="margin-bottom: 16px;">🎨 Display Options</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            theme = st.selectbox(
                "Color Theme", ["Auto (System)", "Light Mode", "Dark Mode"], index=0
            )

            animations = st.checkbox("Enable animations", value=True)
            compact_view = st.checkbox("Compact card layout", value=False)

            st.markdown(
                """
            <div class="ios-card" style="background: var(--surface-secondary); margin-top: 16px;">
                <h4 style="margin-bottom: 16px;">📊 Data & Privacy</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            analytics = st.checkbox(
                "Share anonymous usage data",
                value=True,
                help="Helps improve the learning platform",
            )
            progress_backup = st.checkbox("Automatic progress backup", value=True)

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
