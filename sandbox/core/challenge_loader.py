"""
Challenge Loading and Management System
Handles loading, parsing, and managing challenge files
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BASE_DIR


class ChallengeLoader:
    """Loads and manages challenge files from the filesystem"""

    def __init__(self) -> None:
        self.challenges_dir = Path(BASE_DIR) / "challenges"
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_challenge(
        self, level: int, challenge_name: str
    ) -> Optional[Dict[str, Any]]:
        """Load a specific challenge file"""
        challenge_key = f"level_{level}_{challenge_name}"

        if challenge_key in self._cache:
            return self._cache[challenge_key]

        level_dir = self.challenges_dir / f"level_{level}"

        # Find matching challenge file
        for file_path in level_dir.glob("*.md"):
            if challenge_name.replace(" ", "_").lower() in file_path.stem.lower():
                content = self._parse_challenge_file(file_path)
                self._cache[challenge_key] = content
                return content

        return None

    def _parse_challenge_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a challenge markdown file"""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Extract metadata from the file
            lines = content.split("\n")
            title = ""
            description = ""
            objectives = []

            # Parse the markdown structure
            current_section = None
            for line in lines:
                line = line.strip()

                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("## ") and "objective" in line.lower():
                    current_section = "objectives"
                elif line.startswith("## ") and (
                    "description" in line.lower() or "overview" in line.lower()
                ):
                    current_section = "description"
                elif line.startswith("- ") and current_section == "objectives":
                    objectives.append(line[2:].strip())
                elif (
                    current_section == "description"
                    and line
                    and not line.startswith("#")
                ):
                    description += line + " "

            return {
                "title": title or file_path.stem.replace("_", " ").title(),
                "description": (
                    description.strip()[:200] + "..."
                    if len(description.strip()) > 200
                    else description.strip()
                ),
                "objectives": objectives,
                "file_path": str(file_path),
                "content": content,
                "difficulty": self._estimate_difficulty(content),
                "estimated_time": self._estimate_time(content),
            }
        except Exception as e:
            return {
                "title": f"Challenge: {file_path.stem}",
                "description": "Challenge description not available",
                "objectives": [],
                "file_path": str(file_path),
                "content": "",
                "difficulty": "Medium",
                "estimated_time": "30 minutes",
                "error": str(e),
            }

    def _estimate_difficulty(self, content: str) -> str:
        """Estimate challenge difficulty based on content"""
        content_lower = content.lower()

        # Check for advanced concepts
        advanced_keywords = [
            "machine learning",
            "deep learning",
            "optimization",
            "algorithm",
            "complex",
        ]
        intermediate_keywords = ["visualization", "analysis", "statistics", "dataframe"]
        beginner_keywords = ["pandas", "basic", "introduction", "first"]

        advanced_count = sum(
            1 for keyword in advanced_keywords if keyword in content_lower
        )
        intermediate_count = sum(
            1 for keyword in intermediate_keywords if keyword in content_lower
        )
        beginner_count = sum(
            1 for keyword in beginner_keywords if keyword in content_lower
        )

        if advanced_count > 2:
            return "Advanced"
        elif intermediate_count > 2:
            return "Intermediate"
        elif beginner_count > 1:
            return "Beginner"
        else:
            return "Intermediate"

    def _estimate_time(self, content: str) -> str:
        """Estimate completion time based on content length"""
        word_count = len(content.split())

        if word_count < 500:
            return "15-20 minutes"
        elif word_count < 1000:
            return "30-45 minutes"
        elif word_count < 2000:
            return "1-2 hours"
        else:
            return "2-3 hours"

    def list_challenges(self, level: int) -> List[Dict[str, Any]]:
        """List all challenges for a specific level"""
        level_dir = self.challenges_dir / f"level_{level}"

        if not level_dir.exists():
            return []

        challenges = []
        for file_path in sorted(level_dir.glob("challenge_*.md")):
            challenge_name = (
                file_path.stem.replace("challenge_", "").replace("_", " ").title()
            )
            challenge_data = self.load_challenge(level, challenge_name)
            if challenge_data:
                challenges.append(challenge_data)

        return challenges

    def get_challenge_progress(
        self, level: int, challenge_name: str, completed_challenges: List[str]
    ) -> Dict[str, Any]:
        """Get progress information for a specific challenge"""
        challenge_id = f"level_{level}_{challenge_name.lower().replace(' ', '_')}"
        is_completed = any(
            challenge_id in completed for completed in completed_challenges
        )

        return {
            "id": challenge_id,
            "completed": is_completed,
            "progress": 100 if is_completed else 0,
            "status": "Completed" if is_completed else "Available",
        }

    def search_challenges(
        self, query: str, level: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for challenges matching a query"""
        results = []

        levels_to_search = [level] if level else range(1, 8)

        for search_level in levels_to_search:
            challenges = self.list_challenges(search_level)
            for challenge in challenges:
                if (
                    query.lower() in challenge["title"].lower()
                    or query.lower() in challenge["description"].lower()
                ):
                    challenge["level"] = search_level
                    results.append(challenge)

        return results

    def clear_cache(self) -> None:
        """Clear the challenge cache"""
        self._cache.clear()
