"""
Enhanced Achievement System for Data Science Sandbox
Comprehensive badge categories and auto-validation logic
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BASE_DIR


class EnhancedBadgeManager:
    """
    Enhanced badge system with comprehensive categories:
    - Skill badges (Python Master, Visualization Expert)
    - Progress badges (Speed Demon, Completionist)
    - Special badges (Bug Hunter, Innovation Award)
    """

    def __init__(self) -> None:
        self.badge_categories = {
            "skill": self._get_skill_badges(),
            "progress": self._get_progress_badges(),
            "special": self._get_special_badges(),
            "achievement": self._get_achievement_badges(),
        }

        # Load badge progress tracking
        self.badge_progress_file = Path(BASE_DIR) / "badge_progress.json"
        self.badge_progress = self._load_badge_progress()

    def _get_skill_badges(self) -> Dict[str, Dict]:
        """Skill-based badges for technical proficiency"""
        return {
            "python_master": {
                "name": "🐍 Python Master",
                "description": "Master Python fundamentals across 10 challenges",
                "category": "skill",
                "requirement": "complete_challenges_with_skill",
                "skill": "python",
                "count": 10,
                "xp_reward": 200,
            },
            "pandas_expert": {
                "name": "🐼 Pandas Expert",
                "description": "Excel at data manipulation with pandas",
                "category": "skill",
                "requirement": "complete_challenges_with_skill",
                "skill": "pandas",
                "count": 8,
                "xp_reward": 150,
            },
            "visualization_expert": {
                "name": "📊 Visualization Expert",
                "description": "Create beautiful and informative visualizations",
                "category": "skill",
                "requirement": "complete_challenges_with_skill",
                "skill": "visualization",
                "count": 6,
                "xp_reward": 150,
            },
            "ml_architect": {
                "name": "🤖 ML Architect",
                "description": "Design and build advanced machine learning systems",
                "category": "skill",
                "requirement": "complete_challenges_with_skill",
                "skill": "machine_learning",
                "count": 8,
                "xp_reward": 250,
            },
            "stats_wizard": {
                "name": "🔢 Statistics Wizard",
                "description": "Master statistical analysis and hypothesis testing",
                "category": "skill",
                "requirement": "complete_challenges_with_skill",
                "skill": "statistics",
                "count": 5,
                "xp_reward": 175,
            },
        }

    def _get_progress_badges(self) -> Dict[str, Dict]:
        """Progress and engagement badges"""
        return {
            "speed_demon": {
                "name": "⚡ Speed Demon",
                "description": "Complete 3 challenges in under 30 minutes total",
                "category": "progress",
                "requirement": "fast_completion",
                "time_limit": 1800,  # 30 minutes
                "count": 3,
                "xp_reward": 100,
            },
            "completionist": {
                "name": "✅ Completionist",
                "description": "Complete all challenges in 3 different levels",
                "category": "progress",
                "requirement": "complete_levels",
                "levels": 3,
                "xp_reward": 300,
            },
            "consistent_learner": {
                "name": "📅 Consistent Learner",
                "description": "Complete challenges on 7 different days",
                "category": "progress",
                "requirement": "learning_streak",
                "days": 7,
                "xp_reward": 200,
            },
            "night_owl": {
                "name": "🦉 Night Owl",
                "description": "Complete 5 challenges after 10 PM",
                "category": "progress",
                "requirement": "time_based",
                "hour_after": 22,
                "count": 5,
                "xp_reward": 75,
            },
            "early_bird": {
                "name": "🐦 Early Bird",
                "description": "Complete 5 challenges before 8 AM",
                "category": "progress",
                "requirement": "time_based",
                "hour_before": 8,
                "count": 5,
                "xp_reward": 75,
            },
        }

    def _get_special_badges(self) -> Dict[str, Dict]:
        """Special achievement badges"""
        return {
            "bug_hunter": {
                "name": "🐛 Bug Hunter",
                "description": "Find and fix data quality issues",
                "category": "special",
                "requirement": "data_quality_fixes",
                "count": 3,
                "xp_reward": 150,
            },
            "innovation_award": {
                "name": "💡 Innovation Award",
                "description": "Use creative approaches in challenges",
                "category": "special",
                "requirement": "creative_solutions",
                "count": 2,
                "xp_reward": 200,
            },
            "perfectionist": {
                "name": "💯 Perfectionist",
                "description": "Score 100% on 5 challenges",
                "category": "special",
                "requirement": "perfect_scores",
                "count": 5,
                "xp_reward": 175,
            },
            "explorer": {
                "name": "🗺️ Explorer",
                "description": "Try challenges from all 7 levels",
                "category": "special",
                "requirement": "level_exploration",
                "levels": 7,
                "xp_reward": 250,
            },
        }

    def _get_achievement_badges(self) -> Dict[str, Dict]:
        """Traditional achievement milestones"""
        return {
            "first_steps": {
                "name": "👶 First Steps",
                "description": "Complete your first challenge",
                "category": "achievement",
                "requirement": "challenge_count",
                "count": 1,
                "xp_reward": 50,
            },
            "getting_started": {
                "name": "🚀 Getting Started",
                "description": "Complete 5 challenges",
                "category": "achievement",
                "requirement": "challenge_count",
                "count": 5,
                "xp_reward": 100,
            },
            "data_apprentice": {
                "name": "🎓 Data Apprentice",
                "description": "Complete 15 challenges",
                "category": "achievement",
                "requirement": "challenge_count",
                "count": 15,
                "xp_reward": 200,
            },
            "data_professional": {
                "name": "💼 Data Professional",
                "description": "Complete 30 challenges",
                "category": "achievement",
                "requirement": "challenge_count",
                "count": 30,
                "xp_reward": 400,
            },
            "data_master": {
                "name": "🏆 Data Master",
                "description": "Complete 50 challenges",
                "category": "achievement",
                "requirement": "challenge_count",
                "count": 50,
                "xp_reward": 500,
            },
        }

    def _load_badge_progress(self) -> Dict:
        """Load badge progress tracking data"""
        try:
            if self.badge_progress_file.exists():
                with open(self.badge_progress_file) as f:
                    return dict(json.load(f))
        except Exception:
            pass

        return {
            "completion_times": {},
            "completion_dates": {},
            "skill_counts": {
                "python": 0,
                "pandas": 0,
                "visualization": 0,
                "machine_learning": 0,
                "statistics": 0,
            },
            "perfect_scores": [],
            "data_quality_fixes": 0,
            "creative_solutions": 0,
        }

    def _save_badge_progress(self) -> None:
        """Save badge progress tracking data"""
        try:
            self.badge_progress_file.parent.mkdir(exist_ok=True)
            with open(self.badge_progress_file, "w") as f:
                json.dump(self.badge_progress, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save badge progress: {e}")

    def check_and_award_badges(
        self, progress: Dict, challenge_data: Optional[Dict] = None
    ) -> List[str]:
        """
        Enhanced badge checking with detailed progress tracking

        Args:
            progress: Current player progress
            challenge_data: Optional data about the just-completed challenge
        """
        newly_earned = []

        # Update progress tracking if challenge was completed
        if challenge_data:
            self._update_challenge_progress(challenge_data)

        # Check all badge categories
        for _category_name, badges in self.badge_categories.items():
            for badge_id, badge_info in badges.items():
                if badge_id not in progress.get("badges_earned", []):
                    if self._check_badge_requirement(badge_info, progress):
                        newly_earned.append(badge_id)

        # Save updated progress
        self._save_badge_progress()

        return newly_earned

    def _update_challenge_progress(self, challenge_data: Dict) -> None:
        """Update progress tracking with completed challenge data"""
        challenge_id = challenge_data.get("id", "")
        completion_time = challenge_data.get("completion_time", 0)
        score = challenge_data.get("score", 0)
        skills_used = challenge_data.get("skills", [])

        # Track completion times
        self.badge_progress["completion_times"][challenge_id] = completion_time

        # Track completion dates
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.badge_progress["completion_dates"]:
            self.badge_progress["completion_dates"][today] = []
        self.badge_progress["completion_dates"][today].append(challenge_id)

        # Track skill usage
        for skill in skills_used:
            if skill in self.badge_progress["skill_counts"]:
                self.badge_progress["skill_counts"][skill] += 1

        # Track perfect scores
        if score >= 100:
            self.badge_progress["perfect_scores"].append(challenge_id)

    def _check_badge_requirement(self, badge_info: Dict, progress: Dict) -> bool:
        """Check if badge requirements are met"""
        requirement = badge_info.get("requirement")

        if requirement == "challenge_count":
            return bool(
                len(progress.get("challenges_completed", [])) >= badge_info["count"]
            )

        elif requirement == "complete_challenges_with_skill":
            skill = badge_info["skill"]
            return bool(
                self.badge_progress["skill_counts"].get(skill, 0) >= badge_info["count"]
            )

        elif requirement == "fast_completion":
            recent_times = list(self.badge_progress["completion_times"].values())[
                -badge_info["count"] :
            ]
            return bool(
                len(recent_times) >= badge_info["count"]
                and sum(recent_times) <= badge_info["time_limit"]
            )

        elif requirement == "complete_levels":
            # Check if challenges completed across multiple levels
            completed = progress.get("challenges_completed", [])
            levels_with_completions = set()
            for challenge in completed:
                if challenge.startswith("level_"):
                    level = challenge.split("_")[1]
                    levels_with_completions.add(level)
            return bool(len(levels_with_completions) >= badge_info["levels"])

        elif requirement == "learning_streak":
            return bool(
                len(self.badge_progress["completion_dates"]) >= badge_info["days"]
            )

        elif requirement == "time_based":
            # Track completions by hour of day
            hour_after = badge_info.get("hour_after")
            hour_before = badge_info.get("hour_before")
            target_count = badge_info.get("count", 1)

            time_based_completions = 0
            for completion_list in self.badge_progress["completion_dates"].values():
                # For now, assume random distribution - would need actual timestamps
                if hour_after and len(completion_list) > 0:
                    time_based_completions += len(completion_list) // 6  # Approximate
                elif hour_before and len(completion_list) > 0:
                    time_based_completions += len(completion_list) // 8  # Approximate

            return bool(time_based_completions >= target_count)

        elif requirement == "perfect_scores":
            return bool(
                len(self.badge_progress["perfect_scores"]) >= badge_info["count"]
            )

        elif requirement == "data_quality_fixes":
            return bool(
                self.badge_progress.get("data_quality_fixes", 0) >= badge_info["count"]
            )

        elif requirement == "creative_solutions":
            return bool(
                self.badge_progress.get("creative_solutions", 0) >= badge_info["count"]
            )

        elif requirement == "level_exploration":
            completed = progress.get("challenges_completed", [])
            levels_tried = set()
            for challenge in completed:
                if challenge.startswith("level_"):
                    level = challenge.split("_")[1]
                    levels_tried.add(level)
            return bool(len(levels_tried) >= badge_info["levels"])

        return False

    def get_all_badges(self) -> Dict[str, Dict]:
        """Get all available badges organized by category"""
        all_badges = {}
        for _category_name, badges in self.badge_categories.items():
            all_badges.update(badges)
        return all_badges

    def get_badges_by_category(self, category: str) -> Dict[str, Dict]:
        """Get badges for a specific category"""
        return self.badge_categories.get(category, {})

    def get_badge_progress_summary(self, progress: Dict) -> Dict:
        """Get comprehensive badge progress summary"""
        all_badges = self.get_all_badges()
        earned = progress.get("badges_earned", [])

        summary: Dict[str, Any] = {
            "total_badges": len(all_badges),
            "earned_badges": len(earned),
            "completion_rate": len(earned) / len(all_badges) * 100 if all_badges else 0,
            "by_category": {},
        }

        for category_name, badges in self.badge_categories.items():
            category_earned = sum(1 for badge_id in badges.keys() if badge_id in earned)
            summary["by_category"][category_name] = {
                "total": len(badges),
                "earned": category_earned,
                "completion_rate": category_earned / len(badges) * 100 if badges else 0,
            }

        return summary

    def get_next_badges(self, progress: Dict, limit: int = 5) -> List[Dict]:
        """Get the next badges that can be earned with progress indicators"""
        all_badges = self.get_all_badges()
        earned = progress.get("badges_earned", [])
        next_badges = []

        for badge_id, badge_info in all_badges.items():
            if badge_id not in earned:
                # Calculate progress toward this badge
                progress_pct = self._calculate_badge_progress(badge_info, progress)

                badge_with_progress = badge_info.copy()
                badge_with_progress["id"] = badge_id
                badge_with_progress["progress_percentage"] = progress_pct

                next_badges.append(badge_with_progress)

        # Sort by progress percentage (closest to completion first)
        next_badges.sort(key=lambda x: x["progress_percentage"], reverse=True)

        return next_badges[:limit]

    def _calculate_badge_progress(self, badge_info: Dict, progress: Dict) -> float:
        """Calculate percentage progress toward a badge"""
        requirement = badge_info.get("requirement")

        if requirement == "challenge_count":
            current = len(progress.get("challenges_completed", []))
            target = badge_info["count"]
            return float(min(current / target * 100, 100))

        elif requirement == "complete_challenges_with_skill":
            skill = badge_info["skill"]
            current = self.badge_progress["skill_counts"].get(skill, 0)
            target = badge_info["count"]
            return float(min(current / target * 100, 100))

        elif requirement == "perfect_scores":
            current = len(self.badge_progress["perfect_scores"])
            target = badge_info["count"]
            return float(min(current / target * 100, 100))

        # Add more progress calculations as needed
        return 0

    def get_user_badge_summary(self, _user_id: str) -> Dict[str, Any]:
        """Get comprehensive badge summary for a user"""
        # Mock progress data for demo purposes
        mock_progress = {
            "badges_earned": [],
            "challenges_completed": [],
            "total_xp": 0,
        }

        return self.get_badge_progress_summary(mock_progress)
