"""
Web Dashboard for Data Science Sandbox
Interactive interface for progress tracking and challenge management
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sandbox.core.game_engine import GameEngine
from config import LEVELS, BADGES, CATEGORIES

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
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .level-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.title("🎮 Data Science Sandbox")
        st.markdown("*Learn data science through gamified challenges*")
        
        # Sidebar navigation
        self.create_sidebar()
        
        # Main content area
        page = st.session_state.get('page', 'Dashboard')
        
        if page == 'Dashboard':
            self.show_dashboard()
        elif page == 'Levels':
            self.show_levels()
        elif page == 'Challenges':
            self.show_challenges()
        elif page == 'Badges':
            self.show_badges()
        elif page == 'Progress':
            self.show_progress()
        elif page == 'Settings':
            self.show_settings()
    
    def create_sidebar(self):
        """Create sidebar navigation"""
        st.sidebar.title("🎯 Navigation")
        
        # Player info
        stats = self.game.get_stats()
        st.sidebar.markdown(f"""
        **👤 {self.game.progress['player_name']}**  
        🏆 Level {stats['level']}/6  
        ⭐ {stats['experience']} XP  
        🏅 {stats['badges']} Badges  
        """)
        
        # Navigation buttons
        pages = ['Dashboard', 'Levels', 'Challenges', 'Badges', 'Progress', 'Settings']
        for page in pages:
            if st.sidebar.button(f"📚 {page}", key=page):
                st.session_state.page = page
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Quick actions
        st.sidebar.subheader("🚀 Quick Actions")
        if st.sidebar.button("📊 Launch Jupyter Lab"):
            self.game.launch_jupyter()
            st.sidebar.success("Jupyter Lab launching...")
        
        if st.sidebar.button("🔄 Reset Progress"):
            if st.sidebar.button("⚠️ Confirm Reset"):
                self.game.reset_progress()
                st.sidebar.success("Progress reset!")
                st.rerun()
    
    def show_dashboard(self):
        """Main dashboard view"""
        stats = self.game.get_stats()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Level", f"{stats['level']}/6", delta=None)
        with col2:
            st.metric("Experience Points", stats['experience'], delta="+50")
        with col3:
            st.metric("Badges Earned", stats['badges'], delta=None)
        with col4:
            st.metric("Completion", f"{stats['completion_rate']:.1f}%", delta=None)
        
        # Progress visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Learning Progress")
            
            # Create progress chart
            progress_data = []
            for level in range(1, 7):
                level_status = self.game.progress["level_progress"][str(level)]
                progress_data.append({
                    'Level': f"Level {level}",
                    'Name': LEVELS[level]['name'],
                    'Status': 'Completed' if level_status['completed'] else 
                             'Unlocked' if level_status['unlocked'] else 'Locked',
                    'Progress': 100 if level_status['completed'] else 
                              50 if level_status['unlocked'] else 0
                })
            
            df = pd.DataFrame(progress_data)
            fig = px.bar(df, x='Level', y='Progress', color='Status',
                        title="Level Progression",
                        color_discrete_map={'Completed': '#28a745', 
                                          'Unlocked': '#ffc107', 
                                          'Locked': '#6c757d'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Current Focus")
            current_level = stats['level']
            if current_level <= 6:
                level_info = LEVELS[current_level]
                st.markdown(f"""
                <div class="level-card">
                    <h3>Level {current_level}: {level_info['name']}</h3>
                    <p>{level_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show available challenges
                challenges = self.game.get_level_challenges(current_level)
                if challenges:
                    st.write("**Available Challenges:**")
                    for challenge in challenges[:3]:  # Show first 3
                        completed = f"level_{current_level}_{challenge}" in self.game.progress["challenges_completed"]
                        status = "✅" if completed else "⏳"
                        st.write(f"{status} {challenge}")
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        if self.game.progress["badges_earned"]:
            st.write("🏆 **Latest Badges:**")
            for badge_id in self.game.progress["badges_earned"][-3:]:
                if badge_id in BADGES:
                    badge = BADGES[badge_id]
                    st.write(f"• {badge['name']}: {badge['description']}")
        else:
            st.write("Complete your first challenge to start earning badges! 🎯")
    
    def show_levels(self):
        """Levels overview page"""
        st.header("🏆 Learning Levels")
        st.markdown("Progress through structured levels to master data science concepts.")
        
        for level_num, level_info in LEVELS.items():
            status = self.game.progress["level_progress"][str(level_num)]
            
            # Level card
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if status["completed"]:
                    st.success(f"✅ Level {level_num}")
                elif status["unlocked"]:
                    st.warning(f"🔓 Level {level_num}")
                else:
                    st.error(f"🔒 Level {level_num}")
            
            with col2:
                st.subheader(level_info['name'])
                st.write(level_info['description'])
                
                # Show challenges for this level
                challenges = self.game.get_level_challenges(level_num)
                if challenges and status["unlocked"]:
                    completed_count = len([c for c in self.game.progress["challenges_completed"] 
                                         if c.startswith(f"level_{level_num}")])
                    st.write(f"📚 {completed_count}/{len(challenges)} challenges completed")
            
            with col3:
                if status["unlocked"] and not status["completed"]:
                    if st.button(f"Start Level {level_num}", key=f"start_{level_num}"):
                        st.success(f"Starting Level {level_num}!")
            
            st.markdown("---")
    
    def show_challenges(self):
        """Challenges page"""
        st.header("🎯 Coding Challenges")
        st.markdown("Hands-on challenges to practice your data science skills.")
        
        # Filter by level
        current_level = self.game.get_current_level()
        selected_level = st.selectbox("Select Level", 
                                    range(1, min(current_level + 1, 7)), 
                                    index=current_level-1)
        
        # Show challenges for selected level
        if self.game.progress["level_progress"][str(selected_level)]["unlocked"]:
            challenges = self.game.get_level_challenges(selected_level)
            
            if challenges:
                st.subheader(f"Level {selected_level}: {LEVELS[selected_level]['name']}")
                
                for i, challenge in enumerate(challenges, 1):
                    challenge_id = f"level_{selected_level}_{challenge}"
                    completed = challenge_id in self.game.progress["challenges_completed"]
                    
                    col1, col2, col3 = st.columns([1, 4, 1])
                    
                    with col1:
                        st.write("✅" if completed else f"{i}.")
                    
                    with col2:
                        st.write(f"**{challenge}**")
                        st.write(f"Challenge {i} for {LEVELS[selected_level]['name']}")
                    
                    with col3:
                        if not completed:
                            if st.button("Start", key=f"start_challenge_{challenge_id}"):
                                st.success(f"Challenge {challenge} marked as started!")
                        else:
                            st.success("Done!")
            else:
                st.info(f"No challenges available yet for Level {selected_level}. Check back soon!")
        else:
            st.warning(f"Level {selected_level} is not unlocked yet. Complete previous levels to unlock.")
    
    def show_badges(self):
        """Badges page"""
        st.header("🏅 Achievement Badges")
        st.markdown("Track your accomplishments and unlock new badges!")
        
        # Earned badges
        earned_badges = self.game.progress["badges_earned"]
        if earned_badges:
            st.subheader(f"🏆 Earned Badges ({len(earned_badges)})")
            
            cols = st.columns(3)
            for i, badge_id in enumerate(earned_badges):
                if badge_id in BADGES:
                    badge = BADGES[badge_id]
                    with cols[i % 3]:
                        st.success(f"🏆 **{badge['name']}**")
                        st.write(badge['description'])
        
        # Available badges
        available_badges = [bid for bid in BADGES.keys() if bid not in earned_badges]
        if available_badges:
            st.subheader(f"🎯 Available Badges ({len(available_badges)})")
            
            cols = st.columns(3)
            for i, badge_id in enumerate(available_badges):
                badge = BADGES[badge_id]
                with cols[i % 3]:
                    st.info(f"⚪ **{badge['name']}**")
                    st.write(badge['description'])
    
    def show_progress(self):
        """Progress analytics page"""
        st.header("📊 Progress Analytics")
        
        # Experience over time chart
        st.subheader("Experience Growth")
        
        # Mock data for demonstration (in real app, track over time)
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        exp_data = pd.DataFrame({
            'Date': dates,
            'Experience': [i * 25 + (i**1.2) * 10 for i in range(30)]
        })
        
        fig = px.line(exp_data, x='Date', y='Experience', 
                     title="Experience Points Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Skills Breakdown")
            # Mock skills data
            skills_data = pd.DataFrame({
                'Skill': ['Data Cleaning', 'Visualization', 'Statistics', 'Machine Learning'],
                'Proficiency': [85, 70, 60, 45]
            })
            
            fig = px.bar(skills_data, x='Skill', y='Proficiency', 
                        title="Skill Proficiency Levels")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Time Spent by Category")
            # Mock time data
            time_data = pd.DataFrame({
                'Category': ['Tutorials', 'Challenges', 'Projects', 'Research'],
                'Hours': [15, 25, 10, 8]
            })
            
            fig = px.pie(time_data, values='Hours', names='Category',
                        title="Learning Time Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_settings(self):
        """Settings page"""
        st.header("⚙️ Settings")
        
        # Player name
        new_name = st.text_input("Player Name", 
                                value=self.game.progress["player_name"])
        if st.button("Update Name"):
            self.game.progress["player_name"] = new_name
            self.game.save_progress()
            st.success("Name updated!")
        
        # Progress management
        st.subheader("Progress Management")
        if st.button("🔄 Reset All Progress", type="secondary"):
            if st.button("⚠️ Confirm Reset", type="primary"):
                self.game.reset_progress()
                st.success("Progress reset successfully!")
                st.rerun()
        
        # Export progress
        if st.button("📤 Export Progress"):
            progress_json = json.dumps(self.game.progress, indent=2)
            st.download_button(
                label="Download Progress File",
                data=progress_json,
                file_name="sandbox_progress.json",
                mime="application/json"
            )
        
        # Debug info
        with st.expander("🔧 Debug Information"):
            st.json(self.game.progress)