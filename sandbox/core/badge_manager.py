"""
Enhanced Badge System
Automatic badge earning and management
"""

from typing import Any, Dict, List

from config import BADGES


class BadgeManager:
    """Manages badge earning logic and notifications"""

    def __init__(self) -> None:
        self.badge_triggers = {
            "first_steps": self._check_first_steps,
            "problem_solver": self._check_problem_solver,
            "data_explorer": self._check_data_explorer,
            "visualization_master": self._check_visualization_master,
            "ml_enthusiast": self._check_ml_enthusiast,
            "code_ninja": self._check_code_ninja,
            "persistent_learner": self._check_persistent_learner,
            "challenge_crusher": self._check_challenge_crusher,
            "level_master": self._check_level_master,
            "perfectionist": self._check_perfectionist,
            "speed_demon": self._check_speed_demon,
            "data_scientist": self._check_data_scientist,
            "expert": self._check_expert,
            "mentor": self._check_mentor,
            "legend": self._check_legend,
        }

    def check_and_award_badges(self, progress: Dict) -> List[str]:
        """Check all badge conditions and return newly earned badges"""
        newly_earned = []

        for badge_id, check_function in self.badge_triggers.items():
            if badge_id not in progress.get("badges_earned", []):
                if check_function(progress):
                    newly_earned.append(badge_id)

        return newly_earned

    def _check_first_steps(self, progress: Dict) -> bool:
        """Complete your first challenge"""
        return len(progress.get("challenges_completed", [])) >= 1

    def _check_problem_solver(self, progress: Dict) -> bool:
        """Complete 5 challenges"""
        return len(progress.get("challenges_completed", [])) >= 5

    def _check_data_explorer(self, progress: Dict) -> bool:
        """Complete Level 1"""
        level_progress = progress.get("level_progress", {})
        return bool(level_progress.get("1", {}).get("completed", False))

    def _check_visualization_master(self, progress: Dict) -> bool:
        """Complete all visualization challenges (Level 3)"""
        level_progress = progress.get("level_progress", {})
        return bool(level_progress.get("3", {}).get("completed", False))

    def _check_ml_enthusiast(self, progress: Dict) -> bool:
        """Complete your first ML challenge (any Level 4+ challenge)"""
        completed = progress.get("challenges_completed", [])
        return any(
            challenge.startswith("level_4")
            or challenge.startswith("level_5")
            or challenge.startswith("level_6")
            or challenge.startswith("level_7")
            for challenge in completed
        )

    def _check_code_ninja(self, progress: Dict) -> bool:
        """Complete 10 challenges"""
        return len(progress.get("challenges_completed", [])) >= 10

    def _check_persistent_learner(self, progress: Dict) -> bool:
        """Complete challenges on 3 different days (simplified: 3 challenges)"""
        return len(progress.get("challenges_completed", [])) >= 3

    def _check_challenge_crusher(self, progress: Dict) -> bool:
        """Complete all challenges in any single level"""
        level_progress = progress.get("level_progress", {})
        return any(level.get("completed", False) for level in level_progress.values())

    def _check_level_master(self, progress: Dict) -> bool:
        """Complete 3 levels"""
        level_progress = progress.get("level_progress", {})
        completed_levels = sum(
            1 for level in level_progress.values() if level.get("completed", False)
        )
        return completed_levels >= 3

    def _check_perfectionist(self, progress: Dict) -> bool:
        """Complete 15 challenges"""
        return len(progress.get("challenges_completed", [])) >= 15

    def _check_speed_demon(self, progress: Dict) -> bool:
        """Complete 5 challenges in a row (simplified: 5 challenges)"""
        return len(progress.get("challenges_completed", [])) >= 5

    def _check_data_scientist(self, progress: Dict) -> bool:
        """Complete 20 challenges"""
        return len(progress.get("challenges_completed", [])) >= 20

    def _check_expert(self, progress: Dict) -> bool:
        """Complete 5 levels"""
        level_progress = progress.get("level_progress", {})
        completed_levels = sum(
            1 for level in level_progress.values() if level.get("completed", False)
        )
        return completed_levels >= 5

    def _check_mentor(self, progress: Dict) -> bool:
        """Complete 25 challenges"""
        return len(progress.get("challenges_completed", [])) >= 25

    def _check_legend(self, progress: Dict) -> bool:
        """Complete all 7 levels"""
        level_progress = progress.get("level_progress", {})
        completed_levels = sum(
            1 for level in level_progress.values() if level.get("completed", False)
        )
        return completed_levels >= 7

    def get_badge_info(self, badge_id: str) -> Dict:
        """Get detailed information about a badge"""
        if badge_id in BADGES:
            badge_data = BADGES[badge_id].copy()
            # Note: trigger_function is available but not included in return data
            return badge_data
        return {}

    def get_progress_towards_badge(self, badge_id: str, progress: Dict) -> Dict:
        """Get progress information towards earning a specific badge"""
        if badge_id not in self.badge_triggers:
            return {"error": "Unknown badge"}

        # This would normally contain more sophisticated progress tracking
        # For now, we'll provide basic information
        return {
            "badge_id": badge_id,
            "earned": badge_id in progress.get("badges_earned", []),
            "requirements_met": self.badge_triggers[badge_id](progress),
        }

    def get_next_badges(self, progress: Dict) -> List[Dict]:
        """Get the next badges that can be earned"""
        next_badges = []

        for badge_id in self.badge_triggers:
            if badge_id not in progress.get("badges_earned", []):
                badge_info = self.get_badge_info(badge_id)
                progress_info = self.get_progress_towards_badge(badge_id, progress)

                if badge_info:
                    next_badges.append(
                        {**badge_info, "id": badge_id, "progress": progress_info}
                    )

        return next_badges[:5]  # Return top 5 next badges
