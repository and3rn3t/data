"""
Enhanced Gamification Dashboard for Streamlit
Interactive dashboard showcasing enhanced badge system and analytics
"""

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import our enhanced systems
try:
    from sandbox.achievements.enhanced_badge_system import EnhancedBadgeManager
    from sandbox.analytics.progress_analytics import LearningAnalytics, SkillRadarChart
    from sandbox.core.enhanced_game_engine import EnhancedGameEngine

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    st.warning("⚠️ Enhanced gamification modules not yet integrated. Showing demo data.")


def create_gamification_dashboard():
    """Main dashboard for enhanced gamification features"""

    st.set_page_config(
        page_title="Enhanced Learning Dashboard", page_icon="🎮", layout="wide"
    )

    # Header
    st.markdown(
        """
    # 🎮 Enhanced Learning Dashboard
    *Gamified data science learning with real-time analytics*
    """
    )

    # Initialize systems
    if IMPORTS_AVAILABLE:
        enhanced_engine = EnhancedGameEngine()
        badge_manager = EnhancedBadgeManager()
        skill_radar = SkillRadarChart()
        analytics = LearningAnalytics()
    else:
        enhanced_engine = None
        badge_manager = None
        skill_radar = None
        analytics = None

    # Sidebar - User Selection
    with st.sidebar:
        st.markdown("## 👤 User Profile")
        user_id = st.selectbox(
            "Select User", ["alex_data_explorer", "sam_analyst", "demo_user"]
        )

        st.markdown("## 🎯 Quick Actions")
        if st.button("🚀 Start New Challenge"):
            st.info("Challenge started! Code validation ready.")

        if st.button("📊 Generate Report"):
            st.success("Analytics report generated!")

        st.markdown("## 🔄 Auto-Refresh")
        auto_refresh = st.checkbox("Enable auto-refresh", value=True)
        if auto_refresh:
            st.rerun()

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏠 Overview",
            "🏆 Enhanced Badges",
            "📈 Skill Analytics",
            "🎯 Challenge Validation",
            "🔮 Recommendations",
        ]
    )

    with tab1:
        create_overview_dashboard(enhanced_engine, user_id)

    with tab2:
        create_enhanced_badges_section(badge_manager, user_id)

    with tab3:
        create_skill_analytics_section(skill_radar, analytics, user_id)

    with tab4:
        create_challenge_validation_section()

    with tab5:
        create_recommendations_section(analytics, user_id)


def create_overview_dashboard(enhanced_engine, user_id):
    """Create the main overview dashboard"""

    st.markdown("## 🌟 Welcome Back, Data Scientist!")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔥 Current Streak", "12 days", delta="2 days")

    with col2:
        st.metric("🏆 Badges Earned", "23", delta="3 new")

    with col3:
        st.metric("📊 Skills Mastered", "8/12", delta="1 new")

    with col4:
        st.metric("⚡ XP Points", "2,847", delta="156")

    st.markdown("---")

    # Progress overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 Learning Progress")

        # Mock progress data
        progress_data = {
            "Week": ["W1", "W2", "W3", "W4", "W5", "W6"],
            "Challenges": [3, 5, 4, 7, 6, 8],
            "XP Earned": [150, 280, 220, 410, 350, 480],
            "Skills": [1, 2, 2, 4, 3, 4],
        }

        df = pd.DataFrame(progress_data)

        # Challenges completed chart
        st.bar_chart(df.set_index("Week")["Challenges"])

        # XP progress
        st.line_chart(df.set_index("Week")["XP Earned"])

    with col2:
        st.markdown("### 🎯 This Week's Goals")

        # Weekly goals progress
        st.markdown("**Challenges Completed**")
        st.progress(0.6, text="3/5 challenges")

        st.markdown("**Skills Improved**")
        st.progress(0.8, text="4/5 skills")

        st.markdown("**Badges Earned**")
        st.progress(0.5, text="1/2 badges")

        st.markdown("### 🚀 Quick Start")

        recommended_challenges = [
            "🐼 Pandas Data Cleaning",
            "📊 Plotly Visualization",
            "🤖 ML Model Validation",
        ]

        for challenge in recommended_challenges:
            if st.button(challenge, key=f"quick_{challenge}"):
                st.success(f"Starting {challenge}...")


def create_enhanced_badges_section(badge_manager, user_id):
    """Create the enhanced badges section"""

    st.markdown("## 🏆 Enhanced Achievement System")

    # Badge categories
    badge_categories = {
        "🌟 Skill Badges": {
            "Python Master": {
                "earned": True,
                "progress": 100,
                "description": "Master Python fundamentals",
            },
            "Pandas Expert": {
                "earned": True,
                "progress": 100,
                "description": "Excel at data manipulation",
            },
            "Visualization Guru": {
                "earned": False,
                "progress": 75,
                "description": "Create stunning visualizations",
            },
            "ML Specialist": {
                "earned": False,
                "progress": 45,
                "description": "Master machine learning concepts",
            },
        },
        "📈 Progress Badges": {
            "Speed Demon": {
                "earned": True,
                "progress": 100,
                "description": "Complete challenges quickly",
            },
            "Completionist": {
                "earned": False,
                "progress": 80,
                "description": "Complete all challenges in a level",
            },
            "Streak Master": {
                "earned": True,
                "progress": 100,
                "description": "Maintain 7-day learning streak",
            },
            "Perfectionist": {
                "earned": False,
                "progress": 60,
                "description": "Achieve 95%+ average score",
            },
        },
        "🎖️ Special Badges": {
            "Bug Hunter": {
                "earned": True,
                "progress": 100,
                "description": "Find and fix code bugs",
            },
            "Innovation Award": {
                "earned": False,
                "progress": 30,
                "description": "Create unique solutions",
            },
            "Mentor": {
                "earned": False,
                "progress": 0,
                "description": "Help other learners",
            },
            "Community Star": {
                "earned": False,
                "progress": 20,
                "description": "Active community participation",
            },
        },
        "🏅 Achievement Badges": {
            "First Success": {
                "earned": True,
                "progress": 100,
                "description": "Complete your first challenge",
            },
            "Level Up": {
                "earned": True,
                "progress": 100,
                "description": "Complete an entire skill level",
            },
            "Data Ninja": {
                "earned": False,
                "progress": 85,
                "description": "Master advanced techniques",
            },
            "Analytics Master": {
                "earned": False,
                "progress": 55,
                "description": "Complete analytics specialization",
            },
        },
    }

    # Display badge categories
    for category, badges in badge_categories.items():
        st.markdown(f"### {category}")

        cols = st.columns(2)
        for i, (badge_name, badge_info) in enumerate(badges.items()):
            col = cols[i % 2]

            with col:
                earned = badge_info["earned"]
                progress = badge_info["progress"]

                # Badge display
                if earned:
                    st.success(f"✅ **{badge_name}** - {badge_info['description']}")
                else:
                    st.info(f"⏳ **{badge_name}** - {progress}% complete")
                    st.progress(progress / 100)

        st.markdown("---")

    # Badge statistics
    st.markdown("### 📊 Badge Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_earned = sum(
            1
            for cat in badge_categories.values()
            for badge in cat.values()
            if badge["earned"]
        )
        total_badges = sum(len(cat) for cat in badge_categories.values())
        st.metric("Badges Earned", f"{total_earned}/{total_badges}")

    with col2:
        avg_progress = (
            sum(
                badge["progress"]
                for cat in badge_categories.values()
                for badge in cat.values()
            )
            / total_badges
        )
        st.metric("Average Progress", f"{avg_progress:.1f}%")

    with col3:
        recent_badges = ["Python Master", "Speed Demon", "Bug Hunter"]
        st.metric("Recent Achievements", len(recent_badges))


def create_skill_analytics_section(skill_radar, analytics, user_id):
    """Create the skill analytics section"""

    st.markdown("## 📈 Advanced Skill Analytics")

    # Mock skill data
    user_skills = {
        "Data Manipulation": 85,
        "Data Analysis": 72,
        "Data Visualization": 90,
        "Machine Learning": 58,
        "Programming": 88,
        "Database & SQL": 65,
    }

    comparison_data = {
        "Average Learner": {
            "Data Manipulation": 60,
            "Data Analysis": 55,
            "Data Visualization": 50,
            "Machine Learning": 45,
            "Programming": 65,
            "Database & SQL": 55,
        },
        "Target Level": {
            "Data Manipulation": 95,
            "Data Analysis": 90,
            "Data Visualization": 95,
            "Machine Learning": 85,
            "Programming": 90,
            "Database & SQL": 80,
        },
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎯 Skill Radar Chart")

        # Create radar chart
        categories = list(user_skills.keys())
        values_user = list(user_skills.values())
        values_avg = [comparison_data["Average Learner"][cat] for cat in categories]
        values_target = [comparison_data["Target Level"][cat] for cat in categories]

        fig = go.Figure()

        # User skills
        fig.add_trace(
            go.Scatterpolar(
                r=values_user,
                theta=categories,
                fill="toself",
                name="Your Skills",
                fillcolor="rgba(69, 183, 209, 0.3)",
                line={"color": "rgb(69, 183, 209)", "width": 3},
            )
        )

        # Average learner
        fig.add_trace(
            go.Scatterpolar(
                r=values_avg,
                theta=categories,
                name="Average Learner",
                line={"color": "gray", "dash": "dash"},
            )
        )

        # Target level
        fig.add_trace(
            go.Scatterpolar(
                r=values_target,
                theta=categories,
                name="Target Level",
                line={"color": "green", "dash": "dot"},
            )
        )

        fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 100]}},
            showlegend=True,
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🎖️ Skill Achievements")

        for skill, level in user_skills.items():
            if level >= 80:
                st.success(f"🌟 **{skill}**: Expert ({level}%)")
            elif level >= 60:
                st.info(f"📈 **{skill}**: Proficient ({level}%)")
            else:
                st.warning(f"📚 **{skill}**: Learning ({level}%)")

        st.markdown("### 📊 Learning Insights")

        st.markdown(
            """
        **🔥 Your Strengths:**
        - Data Visualization (90%)
        - Programming (88%)
        - Data Manipulation (85%)

        **📈 Areas to Improve:**
        - Machine Learning (58%)
        - Database & SQL (65%)

        **🎯 Next Steps:**
        - Focus on ML fundamentals
        - Practice SQL queries
        - Maintain visualization excellence
        """
        )

    # Learning patterns
    st.markdown("### 🧠 Learning Pattern Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Learning Style**")
        st.info("🎨 Visual Learner")
        st.caption("You excel with charts and interactive content")

    with col2:
        st.markdown("**Peak Performance**")
        st.info("🌅 Morning (9-11 AM)")
        st.caption("Schedule learning sessions in the morning")

    with col3:
        st.markdown("**Learning Velocity**")
        st.info("🚀 Accelerating")
        st.caption("Your progress is increasing over time")


def create_challenge_validation_section():
    """Create the challenge validation section"""

    st.markdown("## 🎯 Real-time Challenge Validation")

    st.markdown(
        """
    Experience instant feedback on your code with our enhanced validation system!
    """
    )

    # Challenge selector
    challenge_options = {
        "Data Cleaning Challenge": "Clean and analyze customer data",
        "Visualization Challenge": "Create interactive sales dashboard",
        "ML Model Challenge": "Build predictive model for customer churn",
    }

    selected_challenge = st.selectbox(
        "Select Challenge", list(challenge_options.keys())
    )
    st.info(f"📋 **Challenge**: {challenge_options[selected_challenge]}")

    # Code editor
    st.markdown("### 💻 Code Editor")

    sample_code = """
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('customer_data.csv')

# Your solution here
def analyze_customer_data(df):
    # Clean the data
    df_clean = df.dropna()

    # Calculate summary statistics
    summary = df_clean.describe()

    return summary

# Execute the analysis
result = analyze_customer_data(df)
print(result)
"""

    user_code = st.text_area("Enter your solution:", value=sample_code, height=300)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔍 Validate Code", type="primary"):
            with st.spinner("Running validation..."):
                # Simulate validation
                validation_result = {
                    "success": True,
                    "score": 87,
                    "feedback": [
                        "✅ Code executes successfully",
                        "✅ Proper data cleaning approach",
                        "⚠️ Consider handling edge cases",
                        "💡 Add error handling for robustness",
                    ],
                    "skills_demonstrated": [
                        "pandas",
                        "data_cleaning",
                        "function_definition",
                    ],
                    "execution_time": 1.2,
                    "suggestions": [
                        "Use specific column selection instead of dropna()",
                        "Add docstring to your function",
                    ],
                }

                # Display results
                if validation_result["success"]:
                    st.success(
                        f"🎉 Challenge Completed! Score: {validation_result['score']}/100"
                    )
                else:
                    st.error("❌ Validation failed. Please check your code.")

                # Detailed feedback
                st.markdown("#### 📋 Detailed Feedback")
                for feedback in validation_result["feedback"]:
                    st.write(feedback)

                # Skills and suggestions
                col_skills, col_suggestions = st.columns(2)

                with col_skills:
                    st.markdown("**🎯 Skills Demonstrated:**")
                    for skill in validation_result["skills_demonstrated"]:
                        st.write(f"• {skill.replace('_', ' ').title()}")

                with col_suggestions:
                    st.markdown("**💡 Suggestions:**")
                    for suggestion in validation_result["suggestions"]:
                        st.write(f"• {suggestion}")

    with col2:
        if st.button("💡 Get Hint"):
            st.info(
                """
            **💡 Hint Level 1:**

            Think about what happens when you use `dropna()` without parameters.
            Consider which columns actually need to be checked for missing values.

            **Next steps:**
            1. Examine the data structure first
            2. Identify which columns are critical
            3. Apply targeted cleaning methods
            """
            )

    # Validation history
    st.markdown("### 📊 Validation History")

    validation_history = pd.DataFrame(
        {
            "Attempt": [1, 2, 3, 4],
            "Score": [65, 72, 81, 87],
            "Time": ["2 min", "3 min", "2.5 min", "1.2 min"],
            "Skills": [
                "pandas",
                "pandas, functions",
                "pandas, functions, error_handling",
                "pandas, data_cleaning, function_definition",
            ],
        }
    )

    st.dataframe(validation_history, use_container_width=True)


def create_recommendations_section(analytics, user_id):
    """Create the personalized recommendations section"""

    st.markdown("## 🔮 Personalized Learning Recommendations")

    # Mock recommendations
    recommendations = [
        {
            "type": "skill_improvement",
            "title": "Master Machine Learning Fundamentals",
            "description": "Focus on supervised learning algorithms to strengthen this foundational skill",
            "priority": "high",
            "estimated_time": "4-6 hours",
            "skills": ["machine_learning", "scikit_learn"],
            "difficulty": "intermediate",
        },
        {
            "type": "learning_path",
            "title": "Advanced Analytics Specialization",
            "description": "Dive deeper into statistical analysis and predictive modeling",
            "priority": "medium",
            "estimated_time": "15-20 hours",
            "skills": ["statistics", "modeling", "hypothesis_testing"],
            "difficulty": "advanced",
        },
        {
            "type": "skill_exploration",
            "title": "Explore Deep Learning",
            "description": "Broaden your skillset with neural networks and deep learning",
            "priority": "low",
            "estimated_time": "8-10 hours",
            "skills": ["deep_learning", "tensorflow", "pytorch"],
            "difficulty": "advanced",
        },
    ]

    for i, rec in enumerate(recommendations):
        with st.expander(f"🎯 {rec['title']}", expanded=(i == 0)):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(rec["description"])

                # Skills tags
                st.markdown("**Skills involved:**")
                skill_cols = st.columns(len(rec["skills"]))
                for j, skill in enumerate(rec["skills"]):
                    skill_cols[j].markdown(f"`{skill}`")

            with col2:
                # Recommendation metadata
                priority_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                st.markdown(
                    f"**Priority:** {priority_colors[rec['priority']]} {rec['priority'].title()}"
                )
                st.markdown(f"**Time:** ⏱️ {rec['estimated_time']}")
                st.markdown(f"**Level:** 🎚️ {rec['difficulty'].title()}")

                if st.button(f"Start Path {i+1}", key=f"start_{i}"):
                    st.success(f"Started: {rec['title']}")

    # Learning path visualization
    st.markdown("### 🛤️ Recommended Learning Path")

    path_data = {
        "Step": ["Current", "Next 2 weeks", "Month 1", "Month 2", "Month 3"],
        "Focus": [
            "Data Visualization",
            "ML Fundamentals",
            "Advanced ML",
            "Deep Learning",
            "Specialization",
        ],
        "Confidence": [90, 70, 50, 30, 20],
    }

    df_path = pd.DataFrame(path_data)
    st.line_chart(df_path.set_index("Step")["Confidence"])

    # Achievement predictions
    st.markdown("### 🏆 Achievement Predictions")

    predicted_achievements = [
        {"badge": "ML Specialist", "probability": 85, "timeframe": "2 weeks"},
        {"badge": "Advanced Analyst", "probability": 70, "timeframe": "1 month"},
        {"badge": "Data Scientist", "probability": 45, "timeframe": "3 months"},
    ]

    for achievement in predicted_achievements:
        st.markdown(
            f"""
        **🎖️ {achievement['badge']}**
        - Probability: {achievement['probability']}%
        - Estimated time: {achievement['timeframe']}
        """
        )
        st.progress(achievement["probability"] / 100)


if __name__ == "__main__":
    create_gamification_dashboard()
