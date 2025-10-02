"""
Enhanced GameEngine Integration
Connects enhanced badge system, auto-validation, and analytics
"""

from typing import Any, Dict, List

from sandbox.achievements.challenge_validator import ChallengeValidator
from sandbox.achievements.enhanced_badge_system import EnhancedBadgeManager
from sandbox.analytics.progress_analytics import LearningAnalytics, SkillRadarChart
from sandbox.core.game_engine import GameEngine


class EnhancedGameEngine(GameEngine):
    """
    Extended GameEngine with enhanced gamification features
    """

    def __init__(self) -> None:
        super().__init__()
        self.enhanced_badge_manager = EnhancedBadgeManager()
        self.challenge_validator = ChallengeValidator()
        self.learning_analytics = LearningAnalytics()
        self.skill_radar = SkillRadarChart()

    def complete_challenge_with_validation(
        self, challenge_id: str, user_code: str, user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Complete challenge with automatic validation and enhanced rewards
        """
        # Validate the solution
        validation_result = self.challenge_validator.validate_challenge(
            challenge_id, user_code
        )

        result = {
            "challenge_id": challenge_id,
            "validation": validation_result,
            "rewards": [],
            "badges_earned": [],
            "skill_updates": {},
            "analytics": {},
            "success": False,
        }

        # Only proceed if validation passed
        if validation_result["success"]:
            # Complete the challenge in the base engine
            # Complete the base challenge (mock implementation)
            base_result = {"success": True, "score": validation_result["score"]}
            result["success"] = base_result

            if base_result:
                # Award enhanced badges
                mock_progress: Dict[str, List] = {
                    "challenges_completed": [],
                    "badges_earned": [],
                }
                challenge_data = {
                    "id": challenge_id,
                    "completion_time": validation_result.get("execution_time", 0),
                    "score": validation_result.get("score", 0),
                    "skills": validation_result.get("skills_demonstrated", []),
                }
                badges_earned = self.enhanced_badge_manager.check_and_award_badges(
                    mock_progress, challenge_data
                )
                result["badges_earned"] = badges_earned

                # Update skill proficiencies
                skill_updates = self._update_skill_proficiencies(
                    user_id, validation_result
                )
                result["skill_updates"] = skill_updates

                # Calculate XP rewards
                badge_dicts = [
                    {"xp_reward": 50} for _ in badges_earned
                ]  # Mock badge rewards
                xp_reward = self._calculate_xp_reward(validation_result, badge_dicts)
                result["rewards"] = [{"type": "xp", "amount": xp_reward}]

                # Generate analytics insights
                analytics = self._generate_challenge_analytics(
                    user_id, challenge_id, validation_result
                )
                result["analytics"] = analytics

        return result

    def get_enhanced_progress(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Get comprehensive progress including analytics and visualizations
        """
        # Get base progress (mock implementation)
        base_progress = {"total_xp": 100, "level": 1, "badges": []}

        # Get enhanced badge information
        badge_info = self.enhanced_badge_manager.get_user_badge_summary(user_id)

        # Get learning analytics
        user_data = self._compile_user_data(user_id)
        analytics = self.learning_analytics.analyze_learning_patterns(user_data)

        # Generate skill radar data
        current_skills = self._get_current_skills(user_id)
        radar_data = {
            "user_skills": current_skills,
            "comparison_data": {
                "Average Learner": self._get_average_skills(),
                "Target Level": self._get_target_skills(),
            },
        }

        return {
            "base_progress": base_progress,
            "enhanced_badges": badge_info,
            "learning_analytics": analytics,
            "skill_radar_data": radar_data,
            "recommendations": analytics.get("next_recommendations", []),
            "achievements_summary": self._get_achievements_summary(user_id),
        }

    def get_personalized_dashboard(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Generate personalized dashboard data
        """
        user_data = self._compile_user_data(user_id)

        # Generate dashboard components
        dashboard = {
            "welcome_message": self._generate_welcome_message(user_data),
            "current_streak": self._get_current_streak(user_id),
            "weekly_goals": self._get_weekly_goals(user_id),
            "featured_challenges": self._get_featured_challenges(user_data),
            "skill_spotlight": self._get_skill_spotlight(user_data),
            "peer_comparisons": self._get_peer_comparisons(user_data),
            "learning_path_progress": self._get_learning_path_progress(user_id),
            "quick_stats": self._get_quick_stats(user_data),
        }

        return dashboard

    def _update_skill_proficiencies(
        self, user_id: str, validation_result: Dict
    ) -> Dict[str, int]:
        """Update user skill proficiencies based on challenge performance"""
        skills_demonstrated = validation_result.get("skills_demonstrated", [])
        score = validation_result.get("score", 0)

        skill_updates = {}
        current_skills = self._get_current_skills(user_id)

        # Calculate skill point increases
        for skill in skills_demonstrated:
            current_level = current_skills.get(skill, 0)

            # Higher score = more skill points
            skill_increase = max(1, int(score / 20))  # 1-5 points based on score

            # Diminishing returns for higher levels
            if current_level > 80:
                skill_increase = max(1, skill_increase // 2)

            new_level = min(100, current_level + skill_increase)
            skill_updates[skill] = new_level

            # Save to user profile
            self._save_skill_progress(user_id, skill, new_level)

        return skill_updates

    def _calculate_xp_reward(
        self, validation_result: Dict, badges_earned: List[Dict]
    ) -> int:
        """Calculate XP reward based on performance and achievements"""
        base_xp = validation_result.get("score", 0)  # Base XP equals score

        # Bonus XP for badges
        badge_bonus = sum(badge.get("xp_reward", 0) for badge in badges_earned)

        # Bonus for code quality
        quality_bonus = 0
        skills_count = len(validation_result.get("skills_demonstrated", []))
        if skills_count >= 3:
            quality_bonus += 20

        # Penalty for hints/errors
        errors = len(validation_result.get("errors", []))
        error_penalty = min(errors * 5, 30)

        total_xp = max(0, base_xp + badge_bonus + quality_bonus - error_penalty)
        return int(total_xp)

    def _generate_challenge_analytics(
        self, user_id: str, _challenge_id: str, validation_result: Dict
    ) -> Dict[str, Any]:
        """Generate analytics for the completed challenge"""
        return {
            "completion_time": validation_result.get("execution_time", 0),
            "code_quality_score": len(validation_result.get("skills_demonstrated", [])),
            "error_count": len(validation_result.get("errors", [])),
            "improvement_suggestions": validation_result.get("suggestions", []),
            "difficulty_assessment": self._assess_difficulty_match(validation_result),
            "learning_momentum": self._calculate_learning_momentum(user_id),
        }

    def _compile_user_data(self, user_id: str) -> Dict[str, Any]:
        """Compile comprehensive user data for analytics"""
        # This would typically load from database/files
        # For now, return mock structure
        return {
            "user_id": user_id,
            "completed_challenges": self._get_completed_challenges(user_id),
            "current_skills": self._get_current_skills(user_id),
            "completion_timestamps": self._get_completion_timestamps(user_id),
            "recent_scores": self._get_recent_scores(user_id),
            "total_hints_used": self._get_total_hints_used(user_id),
            "avg_completion_time": self._get_avg_completion_time(user_id),
            "skill_progress": self._get_skill_progress_history(user_id),
        }

    def _generate_welcome_message(self, user_data: Dict) -> str:
        """Generate personalized welcome message"""
        completed_count = len(user_data.get("completed_challenges", []))

        if completed_count == 0:
            return "🌟 Welcome to your data science journey! Ready to start your first challenge?"
        elif completed_count < 5:
            return f"🚀 Great start! You've completed {completed_count} challenges. Keep the momentum going!"
        elif completed_count < 15:
            return f"💪 You're making excellent progress! {completed_count} challenges completed and counting!"
        else:
            return f"🏆 Data science champion! {completed_count} challenges mastered. You're on fire!"

    def _get_current_streak(self, _user_id: str) -> Dict[str, Any]:
        """Get user's current learning streak"""
        # Mock implementation - would calculate from actual data
        return {"days": 7, "message": "7-day streak! 🔥", "next_milestone": 14}

    def _get_weekly_goals(self, _user_id: str) -> Dict[str, Any]:
        """Get user's weekly learning goals"""
        return {
            "challenges_completed": {"current": 3, "target": 5},
            "skills_improved": {"current": 2, "target": 3},
            "badges_earned": {"current": 1, "target": 2},
        }

    def _get_featured_challenges(self, user_data: Dict) -> List[Dict]:
        """Get featured challenges based on user preferences"""
        # Use recommendation engine
        recommendations = (
            self.learning_analytics.recommendation_engine.get_recommendations(
                user_data, max_recommendations=3
            )
        )

        featured = []
        for rec in recommendations:
            featured.append(
                {
                    "id": f"challenge_{rec.get('target_skill', 'general')}",
                    "title": rec.get("title", "Featured Challenge"),
                    "difficulty": rec.get("suggested_difficulty", "intermediate"),
                    "estimated_time": rec.get("estimated_time", "30 minutes"),
                    "skills": [rec.get("target_skill", "general")],
                    "reason": rec.get("description", "Recommended for you"),
                }
            )

        return featured

    def _get_skill_spotlight(self, user_data: Dict) -> Dict[str, Any]:
        """Get spotlight skill based on recent progress"""
        current_skills = user_data.get("current_skills", {})
        if not current_skills:
            return {
                "skill": "pandas",
                "level": 0,
                "message": "Start with data manipulation!",
            }

        # Find skill with highest proficiency
        top_skill = max(current_skills, key=current_skills.get)
        return {
            "skill": top_skill,
            "level": current_skills[top_skill],
            "message": f"Your strongest skill! Keep excelling in {top_skill.replace('_', ' ')}",
        }

    def _get_peer_comparisons(self, _user_data: Dict) -> Dict[str, Any]:
        """Get peer comparison data"""
        return {
            "percentile": 75,
            "message": "You're performing better than 75% of learners!",
            "areas_ahead": ["data_manipulation", "visualization"],
            "areas_to_catch_up": ["machine_learning"],
        }

    def _get_learning_path_progress(self, _user_id: str) -> Dict[str, Any]:
        """Get learning path progress"""
        return {
            "current_path": "Data Science Fundamentals",
            "progress": 60,
            "next_milestone": "Advanced Analytics",
            "completed_modules": ["Data Manipulation", "Basic Visualization"],
            "upcoming_modules": ["Statistical Analysis", "Machine Learning Basics"],
        }

    def _get_quick_stats(self, user_data: Dict) -> Dict[str, Any]:
        """Get quick statistics for dashboard"""
        return {
            "total_challenges": len(user_data.get("completed_challenges", [])),
            "total_skills": len(user_data.get("current_skills", {})),
            "average_score": 85,
            "time_saved": "2.5 hours",  # Time saved through efficiency
        }

    # Helper methods for data retrieval (would be implemented with actual storage)
    def _get_completed_challenges(self, _user_id: str) -> List[Dict]:
        """Get list of completed challenges"""
        return []  # Mock implementation

    def _get_current_skills(self, _user_id: str) -> Dict[str, int]:
        """Get current skill proficiencies"""
        return {"pandas": 75, "numpy": 60, "visualization": 80}  # Mock

    def _get_completion_timestamps(self, _user_id: str) -> List[str]:
        """Get challenge completion timestamps"""
        return []  # Mock implementation

    def _get_recent_scores(self, _user_id: str) -> List[int]:
        """Get recent challenge scores"""
        return [85, 90, 78, 92, 88]  # Mock implementation

    def _get_total_hints_used(self, _user_id: str) -> int:
        """Get total hints used by user"""
        return 5  # Mock implementation

    def _get_avg_completion_time(self, _user_id: str) -> int:
        """Get average completion time in seconds"""
        return 450  # Mock implementation

    def _get_skill_progress_history(self, _user_id: str) -> Dict[str, List[Dict]]:
        """Get historical skill progress"""
        return {}  # Mock implementation

    def _save_skill_progress(self, _user_id: str, _skill: str, _level: int) -> None:
        """Save skill progress to persistent storage"""
        pass  # Mock implementation

    def _assess_difficulty_match(self, validation_result: Dict) -> str:
        """Assess if challenge difficulty matched user skill level"""
        score = validation_result.get("score", 0)
        if score >= 90:
            return "too_easy"
        elif score >= 70:
            return "appropriate"
        else:
            return "too_difficult"

    def _calculate_learning_momentum(self, user_id: str) -> str:
        """Calculate user's learning momentum"""
        recent_scores = self._get_recent_scores(user_id)
        if len(recent_scores) >= 3:
            if all(s >= 80 for s in recent_scores[-3:]):
                return "accelerating"
            elif all(s < 70 for s in recent_scores[-3:]):
                return "struggling"
        return "steady"

    def _get_average_skills(self) -> Dict[str, int]:
        """Get average skill levels across all users"""
        return {"pandas": 50, "numpy": 45, "visualization": 55}  # Mock

    def _get_target_skills(self) -> Dict[str, int]:
        """Get target skill levels for competency"""
        return {"pandas": 85, "numpy": 80, "visualization": 90}  # Mock

    def _get_achievements_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user achievements"""
        badge_summary = self.enhanced_badge_manager.get_user_badge_summary(user_id)
        return {
            "total_badges": badge_summary.get("total_badges", 0),
            "recent_achievements": badge_summary.get("recent_badges", []),
            "next_badge_progress": badge_summary.get("progress_towards_next", {}),
            "achievement_rate": "Above Average",  # Mock calculation
        }
