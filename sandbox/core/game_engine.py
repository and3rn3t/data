"""
Core game engine for the D        self.progress = {
            "player_name": "Data Scientist",
            "current_level": 1,
            "experience_points": 0,
            "badges_earned": [],
            "challenges_completed": [],
                  for level in range(1, 8):
            if self.progress["level_progress"][str(level)]["unlocked"]:
                challenges = self.get_level_challenges(level)
                print(f"\n📚 Level {level} - {LEVELS[level]['name']}:")
                for challenge in challenges:
                    completed = (
                        f"level_{level}_{challenge}"
                        in self.progress["challenges_completed"]
                    )
                    status = "✅" if completed else "⏳"
                    print(f"  {status} {challenge}")nt": 0,
            "created_at": datetime.now().isoformat(),
            "last_played": datetime.now().isoformat(),
            "level_progress": {
                str(i): {"unlocked": i == 1, "completed": False, "score": 0}
                for i in range(1, 8)
            },
        }ndbox
Handles progress tracking, level management, and achievements
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from config import BADGES, BASE_DIR, LEVELS


class GameEngine:
    """Main game engine that manages progress, levels, and achievements"""

    def __init__(self, save_file: str = "progress.json"):
        self.save_file = os.path.join(BASE_DIR, save_file)
        self.progress = self.load_progress()

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
        current = self.progress["current_level"]
        if current < 7:
            next_level = current + 1
            self.progress["level_progress"][str(next_level)]["unlocked"] = True
            self.progress["current_level"] = next_level
            self.save_progress()
            print(f"🎉 Level {next_level} unlocked!")
            return next_level
        return current

    def add_experience(self, points: int, reason: str = "") -> None:
        """Add experience points"""
        self.progress["experience_points"] += points
        print(f"🎉 +{points} XP! {reason}")
        self.save_progress()

    def complete_challenge(self, challenge_id: str, score: int = 100) -> None:
        """Mark a challenge as completed"""
        if challenge_id not in self.progress["challenges_completed"]:
            self.progress["challenges_completed"].append(challenge_id)
            self.add_experience(score, f"Completed {challenge_id}")

            # Check for level completion
            self.check_level_completion()
            self.save_progress()

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

    def get_level_challenges(self, level: int) -> List[str]:
        """Get list of challenges for a specific level"""
        challenges_dir = os.path.join(BASE_DIR, "challenges", f"level_{level}")
        if os.path.exists(challenges_dir):
            challenge_files = [
                f
                for f in os.listdir(challenges_dir)
                if f.endswith(".md") or f.endswith(".py") or f.endswith(".ipynb")
            ]
            # Return clean challenge names without file extensions
            return [
                os.path.splitext(f)[0]
                .replace("challenge_", "")
                .replace("_", " ")
                .title()
                for f in sorted(challenge_files)
            ]
        return []

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

    def start_cli_mode(self) -> None:
        """Start CLI interface"""
        print("\n🎮 Data Science Sandbox - CLI Mode")
        print("=" * 50)
        while True:
            self.show_main_menu()
            choice = input("\nSelect option (1-6, 'q' to quit): ").strip().lower()

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

    def show_main_menu(self) -> None:
        """Display main menu"""
        stats = self.get_stats()
        print(
            f"\n👤 {self.progress['player_name']} | Level {stats['level']} | {stats['experience']} XP"
        )
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
        print(f"Current Level: {stats['level']}/6")
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
        """Display earned badges"""
        print(f"\n🏅 Badges Earned ({len(self.progress['badges_earned'])})")
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
        print("\n❓ Data Science Sandbox Help")
        print("=" * 40)
        print("Welcome to your data science learning journey!")
        print("\n🎯 How to Play:")
        print("• Complete challenges to earn XP and badges")
        print("• Unlock new levels by completing 80% of current level")
        print("• Use Jupyter notebooks for hands-on practice")
        print("• Track your progress with the dashboard")
        print("\n📁 Directory Structure:")
        print("• /notebooks - Interactive learning materials")
        print("• /challenges - Coding challenges by level")
        print("• /data - Sample datasets and resources")
        print("• /docs - Documentation and guides")

    def launch_jupyter(self) -> None:
        """Launch Jupyter Lab environment"""
        try:
            import subprocess
            import sys

            print("🚀 Launching Jupyter Lab...")
            # Use python -m jupyterlab to ensure we use the installed module
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyterlab",
                    "--notebook-dir",
                    BASE_DIR,
                    "--no-browser",  # Don't auto-open browser on Windows
                ],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Jupyter Lab failed to start: {e}")
            print("🔄 Trying to open in browser manually...")
            # Fallback: try to start without check and print URL
            try:
                import subprocess
                import sys

                subprocess.Popen(
                    [sys.executable, "-m", "jupyterlab", "--notebook-dir", BASE_DIR]
                )
                print(
                    "📚 Jupyter Lab should open in your browser at: http://localhost:8888"
                )
            except Exception as fallback_error:
                print(f"❌ Could not start Jupyter Lab: {fallback_error}")
                print(
                    "🔄 Alternative: Open notebooks manually in your preferred environment"
                )

    def launch_dashboard(self) -> None:
        """Launch Streamlit dashboard properly"""
        try:
            import subprocess
            import sys

            print("🚀 Launching Data Science Sandbox Dashboard...")
            # Launch streamlit using the dedicated streamlit_app.py file
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    os.path.join(BASE_DIR, "streamlit_app.py"),
                    "--server.port",
                    "8501",
                    "--server.headless",
                    "true",
                ],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Dashboard failed to start: {e}")
            print("🔄 Alternative: Run 'streamlit run streamlit_app.py' manually")
            # Fallback: try to start in background
            try:
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        os.path.join(BASE_DIR, "streamlit_app.py"),
                        "--server.port",
                        "8501",
                    ]
                )
                print(
                    "📊 Dashboard should open in your browser at: http://localhost:8501"
                )
            except Exception as fallback_error:
                print(f"❌ Could not start Dashboard: {fallback_error}")
                print("🔄 Try installing streamlit: pip install streamlit")
