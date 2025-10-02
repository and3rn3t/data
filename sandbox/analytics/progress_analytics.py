"""
Enhanced Progress Analytics & Visualization System
Real-time skill tracking, radar charts, and learning analytics
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SkillRadarChart:
    """
    Create interactive skill radar charts for learning progress
    """

    def __init__(self) -> None:
        self.skill_categories = {
            "data_manipulation": {
                "name": "Data Manipulation",
                "skills": ["pandas", "numpy", "data_cleaning", "data_transformation"],
                "color": "#FF6B6B",
            },
            "analysis": {
                "name": "Data Analysis",
                "skills": [
                    "statistical_analysis",
                    "exploratory_data_analysis",
                    "hypothesis_testing",
                    "correlation_analysis",
                ],
                "color": "#4ECDC4",
            },
            "visualization": {
                "name": "Data Visualization",
                "skills": [
                    "matplotlib",
                    "plotly",
                    "dashboard_creation",
                    "storytelling",
                ],
                "color": "#45B7D1",
            },
            "machine_learning": {
                "name": "Machine Learning",
                "skills": [
                    "supervised_learning",
                    "unsupervised_learning",
                    "model_evaluation",
                    "feature_engineering",
                ],
                "color": "#96CEB4",
            },
            "programming": {
                "name": "Programming",
                "skills": [
                    "python_fundamentals",
                    "functions",
                    "error_handling",
                    "code_optimization",
                ],
                "color": "#FFEAA7",
            },
            "database": {
                "name": "Database & SQL",
                "skills": [
                    "sql_queries",
                    "database_design",
                    "data_warehousing",
                    "etl_processes",
                ],
                "color": "#DDA0DD",
            },
        }

    def create_skill_radar(
        self, user_skills: Dict[str, int], comparison_data: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create interactive radar chart for user skills

        Args:
            user_skills: Dict mapping skill names to proficiency levels (0-100)
            comparison_data: Optional comparison data (peer average, target levels)

        Returns:
            Plotly figure object for the radar chart
        """
        # Prepare data for radar chart
        categories = []
        values = []
        colors = []

        for _category_key, category_info in self.skill_categories.items():
            category_name = category_info["name"]
            category_skills = category_info["skills"]

            # Calculate average proficiency for this category
            category_scores = [user_skills.get(skill, 0) for skill in category_skills]
            avg_score = np.mean(category_scores) if category_scores else 0

            categories.append(category_name)
            values.append(avg_score)
            colors.append(category_info["color"])

        # Create radar chart
        fig = go.Figure()

        # Add user data
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name="Your Skills",
                fillcolor="rgba(69, 183, 209, 0.3)",
                line={"color": "rgb(69, 183, 209)", "width": 3},
                marker={"size": 8},
            )
        )

        # Add comparison data if provided
        if comparison_data:
            for comparison_name, comparison_skills in comparison_data.items():
                comp_values = []
                for _category_key, category_info in self.skill_categories.items():
                    category_skills = category_info["skills"]
                    category_scores = [
                        comparison_skills.get(skill, 0) for skill in category_skills
                    ]
                    avg_score = np.mean(category_scores) if category_scores else 0
                    comp_values.append(avg_score)

                fig.add_trace(
                    go.Scatterpolar(
                        r=comp_values,
                        theta=categories,
                        fill="tonext" if comparison_name == "Target" else None,
                        name=comparison_name,
                        line={
                            "dash": "dash" if comparison_name == "Target" else "solid"
                        },
                    )
                )

        # Update layout
        fig.update_layout(
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": [0, 100],
                    "tickfont": {"size": 10},
                    "gridcolor": "lightgray",
                },
                "angularaxis": {
                    "tickfont": {"size": 12, "color": "darkslategray"},
                    "rotation": 90,
                    "direction": "clockwise",
                },
            },
            title={
                "text": "Skill Proficiency Radar",
                "x": 0.5,
                "font": {"size": 20, "color": "darkslategray"},
            },
            showlegend=True,
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02,
            },
            width=600,
            height=500,
        )

        return fig

    def create_skill_progress_timeline(self, skill_history: List[Dict]) -> go.Figure:
        """
        Create timeline visualization of skill development
        """
        df = pd.DataFrame(skill_history)
        df["date"] = pd.to_datetime(df["date"])

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Overall Progress",
                "Skills Acquired",
                "Challenge Completion Rate",
                "Learning Velocity",
            ),
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        # Overall progress line
        if "overall_score" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["overall_score"],
                    name="Overall Score",
                    line={"color": "blue"},
                ),
                row=1,
                col=1,
            )

        # Skills acquired bar chart
        if "skills_count" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df["date"],
                    y=df["skills_count"],
                    name="Skills Count",
                    marker_color="lightblue",
                ),
                row=1,
                col=2,
            )

        # Challenge completion rate
        if "completion_rate" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["completion_rate"],
                    name="Completion Rate",
                    mode="lines+markers",
                ),
                row=2,
                col=1,
            )

        # Learning velocity (skills per week)
        if len(df) > 1:
            df["learning_velocity"] = (
                df["skills_count"].diff() / df["date"].diff().dt.days * 7
            )
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["learning_velocity"],
                    name="Learning Velocity",
                    mode="lines+markers",
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            height=600, showlegend=True, title_text="Learning Progress Dashboard"
        )

        return fig


class LearningAnalytics:
    """
    Advanced analytics for learning patterns and recommendations
    """

    def __init__(self) -> None:
        self.learning_patterns: Dict[str, Any] = {}
        self.recommendation_engine = RecommendationEngine()

    def analyze_learning_patterns(self, user_data: Dict) -> Dict[str, Any]:
        """
        Analyze user learning patterns and provide insights
        """
        analysis = {
            "learning_style": self._identify_learning_style(user_data),
            "peak_performance_times": self._analyze_performance_times(user_data),
            "skill_progression_rate": self._calculate_progression_rates(user_data),
            "challenge_preferences": self._analyze_challenge_preferences(user_data),
            "areas_for_improvement": self._identify_improvement_areas(user_data),
            "strengths": self._identify_strengths(user_data),
            "next_recommendations": self._generate_recommendations(user_data),
        }

        return analysis

    def _identify_learning_style(self, user_data: Dict) -> Dict[str, Any]:
        """Identify user's learning style based on behavior patterns"""
        # Analyze completion patterns, time spent, hint usage, etc.
        total_challenges = len(user_data.get("completed_challenges", []))
        hint_usage = user_data.get("total_hints_used", 0)
        avg_time_per_challenge = user_data.get("avg_completion_time", 0)

        style_indicators = {
            "analytical": 0,
            "visual": 0,
            "hands_on": 0,
            "collaborative": 0,
        }

        # Analytical learners take more time, use fewer hints
        if avg_time_per_challenge > 600 and hint_usage / max(total_challenges, 1) < 2:
            style_indicators["analytical"] += 3

        # Visual learners prefer dashboard challenges
        dashboard_challenges = len(
            [
                c
                for c in user_data.get("completed_challenges", [])
                if "visualization" in c.get("skills", [])
            ]
        )
        if dashboard_challenges / max(total_challenges, 1) > 0.3:
            style_indicators["visual"] += 2

        # Hands-on learners complete challenges quickly
        if avg_time_per_challenge < 300:
            style_indicators["hands_on"] += 2

        primary_style = max(style_indicators, key=lambda x: style_indicators[x])

        return {
            "primary_style": primary_style,
            "confidence": style_indicators[primary_style] / 10,
            "description": self._get_learning_style_description(primary_style),
        }

    def _analyze_performance_times(self, user_data: Dict) -> List[Dict]:
        """Analyze when user performs best"""
        completion_times = user_data.get("completion_timestamps", [])
        if not completion_times:
            return []

        # Group by hour of day
        hourly_performance: Dict[int, List[int]] = {}
        for timestamp in completion_times:
            hour = datetime.fromisoformat(timestamp).hour
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(1)  # Could track score here

        # Find peak hours
        peak_hours = sorted(
            hourly_performance.items(), key=lambda x: len(x[1]), reverse=True
        )[:3]

        return [
            {
                "hour": hour,
                "performance_level": len(sessions),
                "recommendation": f"Consider scheduling learning sessions around {hour}:00",
            }
            for hour, sessions in peak_hours
        ]

    def _calculate_progression_rates(self, user_data: Dict) -> Dict[str, float]:
        """Calculate skill progression rates"""
        skill_history = user_data.get("skill_progress", {})
        rates = {}

        for skill, history in skill_history.items():
            if len(history) > 1:
                # Calculate rate of improvement over time
                times = [datetime.fromisoformat(entry["date"]) for entry in history]
                scores = [entry["score"] for entry in history]

                if len(scores) > 1:
                    # Simple linear regression for rate
                    time_diffs = [(t - times[0]).days for t in times]
                    score_diffs = [s - scores[0] for s in scores]

                    if sum(time_diffs) > 0:
                        rate = sum(score_diffs) / sum(time_diffs)  # points per day
                        rates[skill] = rate

        return rates

    def _analyze_challenge_preferences(self, user_data: Dict) -> Dict[str, Any]:
        """Analyze what types of challenges user prefers"""
        completed = user_data.get("completed_challenges", [])
        if not completed:
            return {}

        difficulty_prefs: Dict[str, int] = {}
        skill_prefs: Dict[str, int] = {}

        for challenge in completed:
            # Difficulty preferences
            difficulty = challenge.get("difficulty", "unknown")
            difficulty_prefs[difficulty] = difficulty_prefs.get(difficulty, 0) + 1

            # Skill area preferences
            for skill in challenge.get("skills", []):
                skill_prefs[skill] = skill_prefs.get(skill, 0) + 1

        return {
            "preferred_difficulty": (
                max(difficulty_prefs, key=lambda x: difficulty_prefs[x])
                if difficulty_prefs
                else "beginner"
            ),
            "favorite_skills": sorted(
                skill_prefs.items(), key=lambda x: x[1], reverse=True
            )[:3],
            "challenge_variety": len({c.get("category") for c in completed}),
        }

    def _identify_improvement_areas(self, user_data: Dict) -> List[Dict]:
        """Identify areas where user needs improvement"""
        current_skills = user_data.get("current_skills", {})
        improvement_areas = []

        # Skills below 50% proficiency
        for skill, proficiency in current_skills.items():
            if proficiency < 50:
                improvement_areas.append(
                    {
                        "skill": skill,
                        "current_level": proficiency,
                        "priority": "high" if proficiency < 30 else "medium",
                        "suggested_actions": self._get_improvement_suggestions(skill),
                    }
                )

        return improvement_areas

    def _identify_strengths(self, user_data: Dict) -> List[Dict]:
        """Identify user's strongest skills"""
        current_skills = user_data.get("current_skills", {})
        strengths = []

        # Skills above 80% proficiency
        for skill, proficiency in current_skills.items():
            if proficiency >= 80:
                strengths.append(
                    {
                        "skill": skill,
                        "proficiency": proficiency,
                        "level": "expert" if proficiency >= 90 else "advanced",
                        "opportunities": self._get_strength_opportunities(skill),
                    }
                )

        return strengths

    def _generate_recommendations(self, user_data: Dict) -> List[Dict]:
        """Generate personalized learning recommendations"""
        return self.recommendation_engine.get_recommendations(user_data)

    def _get_learning_style_description(self, style: str) -> str:
        """Get description for learning style"""
        descriptions = {
            "analytical": "You prefer to understand concepts deeply before applying them. "
            "You benefit from detailed explanations and systematic approaches.",
            "visual": "You learn best through visualizations and graphical representations. "
            "Charts, diagrams, and interactive dashboards enhance your understanding.",
            "hands_on": "You prefer learning by doing. Quick iteration and immediate "
            "feedback help you grasp concepts effectively.",
            "collaborative": "You benefit from discussion and peer interaction. "
            "Consider joining study groups or forums.",
        }
        return descriptions.get(style, "Mixed learning style with diverse preferences.")

    def _get_improvement_suggestions(self, skill: str) -> List[str]:
        """Get specific improvement suggestions for a skill"""
        suggestions = {
            "pandas": [
                "Practice data filtering and selection",
                "Master groupby operations",
                "Learn merge and join operations",
            ],
            "visualization": [
                "Create different chart types",
                "Practice color theory and design",
                "Learn interactive plotting",
            ],
        }
        return suggestions.get(
            skill,
            ["Practice with relevant challenges", "Review documentation and examples"],
        )

    def _get_strength_opportunities(self, skill: str) -> List[str]:
        """Get opportunities to leverage strengths"""
        opportunities = {
            "pandas": [
                "Mentor other learners",
                "Create advanced tutorials",
                "Contribute to open source projects",
            ],
            "visualization": [
                "Design dashboard templates",
                "Create data storytelling examples",
                "Lead visualization workshops",
            ],
        }
        return opportunities.get(
            skill, ["Share knowledge with community", "Take on advanced projects"]
        )


class RecommendationEngine:
    """
    Personalized learning path recommendations
    """

    def __init__(self) -> None:
        self.recommendation_strategies = [
            self._skill_gap_recommendations,
            self._learning_path_recommendations,
            self._difficulty_progression_recommendations,
            self._variety_recommendations,
        ]

    def get_recommendations(
        self, user_data: Dict, max_recommendations: int = 5
    ) -> List[Dict]:
        """
        Generate personalized recommendations for the user
        """
        all_recommendations = []

        # Apply each strategy
        for strategy in self.recommendation_strategies:
            recommendations = strategy(user_data)
            all_recommendations.extend(recommendations)

        # Score and rank recommendations
        scored_recommendations = self._score_recommendations(
            all_recommendations, user_data
        )

        # Return top recommendations
        return sorted(scored_recommendations, key=lambda x: x["score"], reverse=True)[
            :max_recommendations
        ]

    def _skill_gap_recommendations(self, user_data: Dict) -> List[Dict]:
        """Recommend challenges to fill skill gaps"""
        recommendations = []
        current_skills = user_data.get("current_skills", {})

        # Find skills with low proficiency
        for skill, proficiency in current_skills.items():
            if proficiency < 60:  # Skill gap threshold
                recommendations.append(
                    {
                        "type": "skill_improvement",
                        "title": f"Improve {skill.replace('_', ' ').title()}",
                        "description": f"Focus on {skill} to strengthen this foundational skill",
                        "target_skill": skill,
                        "urgency": "high" if proficiency < 30 else "medium",
                        "estimated_time": "2-3 hours",
                    }
                )

        return recommendations

    def _learning_path_recommendations(self, user_data: Dict) -> List[Dict]:
        """Recommend structured learning paths"""
        recommendations = []
        completed_challenges = len(user_data.get("completed_challenges", []))

        # Beginner path
        if completed_challenges < 10:
            recommendations.append(
                {
                    "type": "learning_path",
                    "title": "Data Science Fundamentals Path",
                    "description": "Master the basics of data manipulation and analysis",
                    "path_id": "fundamentals",
                    "estimated_time": "10-15 hours",
                    "difficulty": "beginner",
                }
            )

        # Intermediate paths
        elif completed_challenges < 25:
            recommendations.append(
                {
                    "type": "learning_path",
                    "title": "Advanced Analytics Path",
                    "description": "Dive deeper into statistical analysis and modeling",
                    "path_id": "advanced_analytics",
                    "estimated_time": "15-20 hours",
                    "difficulty": "intermediate",
                }
            )

        return recommendations

    def _difficulty_progression_recommendations(self, user_data: Dict) -> List[Dict]:
        """Recommend appropriate difficulty progression"""
        recommendations = []
        recent_performance = user_data.get("recent_scores", [])

        if recent_performance:
            avg_score = np.mean(recent_performance)

            if avg_score > 85:  # Ready for harder challenges
                recommendations.append(
                    {
                        "type": "difficulty_progression",
                        "title": "Challenge Yourself Further",
                        "description": "You're excelling! Try more advanced challenges",
                        "suggested_difficulty": "advanced",
                        "confidence": avg_score / 100,
                    }
                )
            elif avg_score < 60:  # Needs reinforcement
                recommendations.append(
                    {
                        "type": "difficulty_progression",
                        "title": "Reinforce Fundamentals",
                        "description": "Strengthen your foundation before advancing",
                        "suggested_difficulty": "beginner",
                        "focus": "accuracy",
                    }
                )

        return recommendations

    def _variety_recommendations(self, user_data: Dict) -> List[Dict]:
        """Recommend variety in learning"""
        recommendations = []
        completed_skills = set()

        for challenge in user_data.get("completed_challenges", []):
            completed_skills.update(challenge.get("skills", []))

        # Suggest unexplored areas
        all_skills = {"pandas", "numpy", "visualization", "sql", "machine_learning"}
        unexplored = all_skills - completed_skills

        for skill in list(unexplored)[:2]:  # Limit to 2 suggestions
            recommendations.append(
                {
                    "type": "skill_exploration",
                    "title": f"Explore {skill.replace('_', ' ').title()}",
                    "description": f"Broaden your skillset with {skill} challenges",
                    "target_skill": skill,
                    "novelty": True,
                }
            )

        return recommendations

    def _score_recommendations(
        self, recommendations: List[Dict], user_data: Dict
    ) -> List[Dict]:
        """Score recommendations based on user context"""
        for rec in recommendations:
            score = 50  # Base score

            # Boost urgent recommendations
            if rec.get("urgency") == "high":
                score += 20
            elif rec.get("urgency") == "medium":
                score += 10

            # Boost based on user preferences
            learning_style = user_data.get("learning_style", {})
            if (
                learning_style.get("primary_style") == "analytical"
                and rec.get("type") == "learning_path"
            ):
                score += 15

            # Boost novel recommendations for advanced users
            if (
                rec.get("novelty")
                and len(user_data.get("completed_challenges", [])) > 20
            ):
                score += 10

            rec["score"] = min(100, score)

        return recommendations
