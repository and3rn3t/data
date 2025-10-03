"""
Core game engine for the Data Science Sandbox
Handles progress tracking, level management, and achievements
"""

import json
import os
import subprocess
import webbrowser
from datetime import datetime
from typing import Any, Dict, List

from config import BADGES, BASE_DIR, LEVELS


class GameEngine:
    """Main game engine that manages progress, levels, and achievements"""

    def __init__(self, save_file: str = "progress.json"):
        self.save_file = os.path.join(BASE_DIR, save_file)
        self.progress = self.load_progress()

        # Initialize enhanced systems
        from .badge_manager import BadgeManager
        from .challenge_loader import ChallengeLoader

        self.badge_manager = BadgeManager()
        self.challenge_loader = ChallengeLoader()

    def load_progress(self) -> Dict[str, Any]:
        """Load progress from save file or create new profile"""
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file) as f:
                    data: Dict[str, Any] = json.load(f)
                    return data
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        # Create new progress profile
        return {
            "player_name": "Data Scientist",
            "current_level": 1,
            "experience_points": 0,
            "badges_earned": [],
            "challenges_completed": [],
            "time_spent": 0,
            "created_at": datetime.now().isoformat(),
            "last_played": datetime.now().isoformat(),
            "level_progress": {
                str(i): {"unlocked": i == 1, "completed": False, "score": 0}
                for i in range(1, 8)
            },
        }

    def save_progress(self) -> None:
        """Save current progress to file"""
        self.progress["last_played"] = datetime.now().isoformat()
        with open(self.save_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def reset_progress(self) -> None:
        """Reset all progress"""
        if os.path.exists(self.save_file):
            os.remove(self.save_file)
        self.progress = self.load_progress()
        self.save_progress()

    def get_current_level(self) -> int:
        """Get current level"""
        return int(self.progress["current_level"])

    def set_current_level(self, level: int) -> None:
        """Set current level (for testing/admin)"""
        if 1 <= level <= 7:
            self.progress["current_level"] = level
            # Unlock all levels up to this one
            for i in range(1, level + 1):
                self.progress["level_progress"][str(i)]["unlocked"] = True
            self.save_progress()

    def unlock_next_level(self) -> int:
        """Unlock next level when current is completed"""
        current: int = self.progress["current_level"]
        if current < 7:
            next_level: int = current + 1
            self.progress["level_progress"][str(next_level)]["unlocked"] = True
            self.progress["current_level"] = next_level
            self.save_progress()
            print(f"🎉 Level {next_level} unlocked!")
            return next_level
        return current

    def add_experience(self, points: int, reason: str = "") -> None:
        """Add experience points"""
        self.progress["experience_points"] += points
        print(f"+{points} XP! {reason}")
        self.save_progress()

    def complete_challenge(self, challenge_id: str) -> bool:
        """Mark a challenge as completed and check for badges"""
        # Validate challenge exists
        if not self._is_valid_challenge(challenge_id):
            return False

        if challenge_id not in self.progress["challenges_completed"]:
            self.progress["challenges_completed"].append(challenge_id)
            self.add_experience(100, f"Completed {challenge_id}")

            # Check for newly earned badges
            newly_earned = self.badge_manager.check_and_award_badges(self.progress)
            for badge_id in newly_earned:
                self.earn_badge(badge_id)

            self.save_progress()
            self.check_level_completion()
            return True
        return False  # Challenge was already completed

    def earn_badge(self, badge_id: str) -> None:
        """Earn a new badge"""
        if badge_id not in self.progress["badges_earned"] and badge_id in BADGES:
            self.progress["badges_earned"].append(badge_id)
            badge = BADGES[badge_id]
            print(f"🏆 New Badge Unlocked: {badge['name']} - {badge['description']}")
            self.add_experience(50, f"Earned {badge['name']} badge")

    def check_level_completion(self) -> None:
        """Check if current level is completed and unlock next"""
        current_level = self.get_current_level()
        level_challenges = self.get_level_challenges(current_level)
        completed_challenges = [
            c
            for c in self.progress["challenges_completed"]
            if c.startswith(f"level_{current_level}")
        ]

        if len(completed_challenges) >= len(level_challenges) * 0.8:  # 80% completion
            self.progress["level_progress"][str(current_level)]["completed"] = True
            if current_level < 7:  # Can unlock next level
                self.unlock_next_level()
                print(
                    f"🎊 Level {current_level} Complete! Level {current_level + 1} Unlocked!"
                )
                self.earn_badge("problem_solver")

    def _normalize_challenge_name(self, challenge_name: str) -> str:
        """Normalize challenge name for matching."""
        normalized_name = challenge_name.lower().replace(" ", "_")
        # Remove leading number patterns like "1_", "2_", etc.
        if (
            len(normalized_name) > 2
            and normalized_name[1] == "_"
            and normalized_name[0].isdigit()
        ):
            normalized_name = normalized_name[2:]  # Remove "1_" from "1_first_steps"
        return normalized_name

    def _get_possible_challenge_ids(
        self, level: int, normalized_name: str
    ) -> List[str]:
        """Get possible challenge ID formats."""
        return [
            f"level_{level}_{normalized_name}",
            f"level_{level}_challenge_{normalized_name}",
            f"level_{level}_1_{normalized_name}",
            f"level_{level}_2_{normalized_name}",
            f"level_{level}_3_{normalized_name}",
            f"level_{level}_4_{normalized_name}",
        ]

    def _check_direct_match(self, possible_ids: List[str], completed_id: str) -> bool:
        """Check for direct ID matches."""
        return completed_id in possible_ids

    def _check_partial_match(
        self, level: int, completed_id: str, normalized_name: str
    ) -> bool:
        """Check for partial name matches."""
        completed_name_part = completed_id.replace(f"level_{level}_", "")

        # Remove common prefixes from completed name
        if completed_name_part.startswith("challenge_"):
            completed_name_part = completed_name_part[10:]  # Remove "challenge_"

        # Check if the core challenge name matches
        return (
            normalized_name in completed_name_part
            or completed_name_part in normalized_name
            or normalized_name == completed_name_part
        )

    def _get_difficulty_level(self, level: int) -> str:
        """Get difficulty level based on challenge level."""
        if level <= 2:
            return "beginner"
        if level <= 4:
            return "intermediate"
        return "advanced"

    def _is_challenge_completed(self, level: int, challenge_name: str) -> bool:
        """Check if a specific challenge is completed using flexible matching"""
        completed = self.progress["challenges_completed"]
        normalized_name = self._normalize_challenge_name(challenge_name)
        possible_ids = self._get_possible_challenge_ids(level, normalized_name)

        # Check each completed challenge
        for completed_id in completed:
            if not completed_id.startswith(f"level_{level}_"):
                continue

            # Direct ID match
            if self._check_direct_match(possible_ids, completed_id):
                return True

            # Partial match
            if self._check_partial_match(level, completed_id, normalized_name):
                return True

        return False

    def get_level_challenges_with_completion(self, level: int) -> List[Dict[str, Any]]:
        """Get level challenges with completion status"""
        challenges = self.get_level_challenges(level)
        result = []

        for i, challenge in enumerate(challenges, 1):
            is_completed = self._is_challenge_completed(level, challenge)
            result.append(
                {
                    "name": challenge,
                    "number": i,
                    "completed": is_completed,
                    "id": f"level_{level}_{i}_{challenge.lower().replace(' ', '_')}",
                }
            )

        return result

    def get_level_completion_stats(self, level: int) -> Dict[str, Any]:
        """Get completion statistics for a level"""
        challenges = self.get_level_challenges_with_completion(level)
        completed_count = sum(1 for c in challenges if c["completed"])

        return {
            "total_challenges": len(challenges),
            "completed_challenges": completed_count,
            "completion_rate": (
                (completed_count / len(challenges) * 100) if challenges else 0
            ),
            "is_level_complete": completed_count >= len(challenges) * 0.8,
        }

    def get_level_challenges(self, level: int) -> List[str]:
        """Get list of challenges for a specific level"""
        challenges_dir = os.path.join(BASE_DIR, "challenges", f"level_{level}")
        challenges = []

        if os.path.exists(challenges_dir):
            for file in os.listdir(challenges_dir):
                if file.endswith(".md") and file.startswith("challenge_"):
                    # Extract challenge name from filename
                    name = file.replace("challenge_", "").replace(".md", "")
                    # Convert underscores to spaces and title case
                    formatted_name = name.replace("_", " ").title()
                    challenges.append(formatted_name)

        return sorted(challenges)

    def get_enhanced_challenges(self, level: int) -> List[Dict[str, Any]]:
        """Get enhanced challenge information with metadata"""
        return self.challenge_loader.list_challenges(level)

    def get_challenge_details(self, level: int, challenge_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific challenge"""
        challenge_data = self.challenge_loader.load_challenge(level, challenge_name)
        if challenge_data:
            progress_info = self.challenge_loader.get_challenge_progress(
                level, challenge_name, self.progress["challenges_completed"]
            )
            challenge_data.update(progress_info)
        return challenge_data or {}

    def get_next_badges(self) -> List[Dict[str, Any]]:
        """Get badges that can be earned next"""
        return self.badge_manager.get_next_badges(self.progress)

    def get_stats(self) -> Dict[str, Any]:
        """Get player statistics"""
        return {
            "level": self.get_current_level(),
            "experience": self.progress["experience_points"],
            "badges": len(self.progress["badges_earned"]),
            "challenges_completed": len(self.progress["challenges_completed"]),
            "total_levels": 7,
            "completion_rate": len(self.progress["challenges_completed"])
            / max(1, self.count_total_challenges())
            * 100,
        }

    def count_total_challenges(self) -> int:
        """Count total number of challenges across all levels"""
        total = 0
        for level in range(1, 8):
            total += len(self.get_level_challenges(level))
        return max(1, total)  # Avoid division by zero

    def get_available_challenges(self) -> List[Dict[str, Any]]:
        """Get list of challenges available to the player"""
        current_level = self.get_current_level()
        all_challenges = []

        # Get challenges from current level and all unlocked levels
        for level in range(1, current_level + 1):
            challenges_dir = os.path.join(BASE_DIR, "challenges", f"level_{level}")

            if os.path.exists(challenges_dir):
                for file in os.listdir(challenges_dir):
                    if file.endswith(".md") and file.startswith("challenge_"):
                        # Extract challenge name from filename (e.g., challenge_1_first_steps.md)
                        base_name = file.replace("challenge_", "").replace(".md", "")
                        challenge_id = f"level_{level}_{base_name}"

                        # Only include challenges not yet completed
                        if challenge_id not in self.progress["challenges_completed"]:
                            # Format display name
                            display_name = base_name.replace("_", " ").title()

                            challenge_info = {
                                "id": challenge_id,
                                "name": display_name,
                                "level": level,
                                "difficulty": self._get_difficulty_level(level),
                            }
                            all_challenges.append(challenge_info)

        return sorted(all_challenges, key=lambda x: (x["level"], x["id"]))

    def _is_valid_challenge(self, challenge_id: str) -> bool:
        """Check if a challenge ID is valid"""
        # Extract level from challenge_id (format: level_X_number_challenge_name)
        try:
            parts = challenge_id.split("_")
            if len(parts) < 4 or parts[0] != "level":
                return False

            level = int(parts[1])

            # Check if level exists and challenge file exists
            if level in LEVELS:
                challenges_dir = os.path.join(BASE_DIR, "challenges", f"level_{level}")

                # Reconstruct the original filename from the challenge_id
                # level_1_1_first_steps -> challenge_1_first_steps.md
                challenge_number = parts[2]
                challenge_name = "_".join(parts[3:])
                challenge_file = f"challenge_{challenge_number}_{challenge_name}.md"

                return os.path.exists(os.path.join(challenges_dir, challenge_file))
            return False
        except (ValueError, IndexError):
            return False

    def start_cli_mode(self) -> None:
        """Start CLI interface"""
        print("\n🎮 Data Science Sandbox - CLI Mode")
        print("=" * 50)

        while True:
            stats = self.get_stats()
            print(
                f"\n👤 {self.progress['player_name']} | Level {stats['level']} | {stats['experience']} XP"
            )

            self.show_main_menu()

            try:
                choice = input("Select option (1-6, 'q' to quit): ").strip().lower()

                if choice == "q":
                    print("👋 Thanks for playing! Keep learning!")
                    break
                elif choice == "1":
                    self.show_stats()
                elif choice == "2":
                    self.show_levels()
                elif choice == "3":
                    self.show_badges()
                elif choice == "4":
                    self.list_challenges()
                elif choice == "5":
                    self.show_help()
                elif choice == "6":
                    self.launch_jupyter()
                else:
                    print("❌ Invalid option. Please try again.")
            except KeyboardInterrupt:
                print("\n👋 Thanks for playing! Keep learning!")
                break

    def show_main_menu(self) -> None:
        """Display main menu options"""
        print("\n📚 Main Menu:")
        print("1. View Stats & Progress")
        print("2. Browse Levels")
        print("3. View Badges")
        print("4. List Challenges")
        print("5. Help & Documentation")
        print("6. Launch Jupyter Lab")

    def show_stats(self) -> None:
        """Display player statistics"""
        stats = self.get_stats()
        print("\n📊 Player Statistics")
        print("=" * 30)
        print(f"Current Level: {stats['level']}/{stats['total_levels']}")
        print(f"Experience Points: {stats['experience']}")
        print(f"Badges Earned: {stats['badges']}")
        print(f"Challenges Completed: {stats['challenges_completed']}")
        print(f"Overall Progress: {stats['completion_rate']:.1f}%")

    def show_levels(self) -> None:
        """Display level information"""
        print("\n🏆 Levels & Progression")
        print("=" * 40)
        for level_num, level_info in LEVELS.items():
            status = self.progress["level_progress"][str(level_num)]
            icon = "🔓" if status["unlocked"] else "🔒"
            completion = "✅" if status["completed"] else "⏳"
            print(f"{icon} {completion} Level {level_num}: {level_info['name']}")
            print(f"    {level_info['description']}")

    def show_badges(self) -> None:
        """Display earned and available badges"""
        print("\n🏆 Your Badges")
        print("=" * 40)
        for badge_id in self.progress["badges_earned"]:
            if badge_id in BADGES:
                badge = BADGES[badge_id]
                print(f"🏆 {badge['name']}: {badge['description']}")

        print("\n🎯 Available Badges:")
        for badge_id, badge in BADGES.items():
            if badge_id not in self.progress["badges_earned"]:
                print(f"⚪ {badge['name']}: {badge['description']}")

    def list_challenges(self) -> None:
        """List available challenges"""
        print("\n🎯 Available Challenges")
        print("=" * 35)
        current_level = self.get_current_level()

        for level in range(1, min(current_level + 1, 8)):
            if self.progress["level_progress"][str(level)]["unlocked"]:
                challenges = self.get_level_challenges(level)
                print(f"\n📚 Level {level} - {LEVELS[level]['name']}:")
                for challenge in challenges:
                    completed = (
                        f"level_{level}_{challenge}"
                        in self.progress["challenges_completed"]
                    )
                    status = "✅" if completed else "⏳"
                    print(f"  {status} {challenge}")

    def show_help(self) -> None:
        """Display help information"""
        print("\n📖 Help & Documentation")
        print("=" * 40)
        print("🎯 How to Play:")
        print("• Progress through levels by completing challenges")
        print("• Earn XP and badges for achievements")
        print("• Use Jupyter notebooks for interactive learning")
        print("• Each level builds on previous skills")
        print("\n📚 Resources:")
        print("• Challenge files: /challenges/level_X/")
        print("• Notebooks: /notebooks/")
        print("• Documentation: /docs/")
        print("\n💡 Tips:")
        print("• Read challenge instructions carefully")
        print("• Experiment with the code examples")
        print("• Don't hesitate to explore beyond the basics!")

    def launch_jupyter(self) -> None:
        """Launch Jupyter Lab environment"""
        notebooks_dir = os.path.join(BASE_DIR, "notebooks")

        try:
            print("🚀 Launching Jupyter Lab...")
            # Try to launch Jupyter Lab
            subprocess.run(
                ["jupyter", "lab", "--notebook-dir", notebooks_dir],
                check=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Jupyter Lab failed to start: {e}")
            try:
                print("🔄 Trying to open in browser manually...")
                # Try to open the Jupyter Lab URL in browser
                webbrowser.open("http://localhost:8888")
                print(
                    "📚 Jupyter Lab should open in your browser at: http://localhost:8888"
                )
            except Exception as fallback_error:
                print(f"❌ Could not start Jupyter Lab: {fallback_error}")

    def launch_dashboard(self) -> None:
        """Launch Streamlit dashboard with enhanced features"""
        try:
            print("🚀 Launching Data Science Sandbox Dashboard...")
            print(
                "💡 Dashboard includes enhanced challenge metadata and smart badge recommendations"
            )
            subprocess.run(
                [
                    "python",
                    "-m",
                    "streamlit",
                    "run",
                    os.path.join(BASE_DIR, "apps", "streamlit_app.py"),
                ],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Dashboard failed to start: {e}")
            print("💡 Make sure Streamlit is installed: pip install streamlit")
            print("💡 Try running: python -m streamlit run apps/streamlit_app.py")

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress data"""
        return self.progress.copy()
