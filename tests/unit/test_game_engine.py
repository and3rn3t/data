"""
Unit tests for GameEngine core functionality
"""

import os
from typing import Any
from unittest.mock import patch

from sandbox.core.game_engine import GameEngine


class TestGameEngine:
    """Test suite for GameEngine class."""

    def test_initialization_with_existing_progress(self, game_engine: Any) -> None:
        """Test GameEngine initialization with existing progress file."""
        assert game_engine.progress["player_name"] == "Test Player"
        assert game_engine.progress["current_level"] == 1
        assert game_engine.progress["experience_points"] == 0

    def test_initialization_with_new_progress(self, temp_dir: Any) -> None:
        """Test GameEngine initialization without existing progress file."""
        new_save_file = os.path.join(temp_dir, "new_progress.json")
        engine = GameEngine(save_file=new_save_file)

        assert engine.progress["player_name"] == "Data Scientist"
        assert engine.progress["current_level"] == 1
        assert engine.progress["experience_points"] == 0
        assert len(engine.progress["level_progress"]) == 7

    def test_save_progress(self, game_engine: Any) -> None:
        """Test saving progress to file."""
        original_exp = game_engine.progress["experience_points"]
        game_engine.add_experience(50, "Test completion")

        # Reload engine to verify save
        new_engine = GameEngine(save_file=game_engine.save_file)
        assert new_engine.progress["experience_points"] == original_exp + 50

    def test_add_experience(self, game_engine: Any) -> None:
        """Test adding experience points."""
        initial_exp = game_engine.progress["experience_points"]
        game_engine.add_experience(100, "Completed challenge")

        assert game_engine.progress["experience_points"] == initial_exp + 100

    def test_complete_challenge(self, game_engine: Any) -> None:
        """Test completing a challenge."""
        challenge_id = "level_1_first_steps"
        initial_challenges = len(game_engine.progress["challenges_completed"])

        game_engine.complete_challenge(challenge_id, 95)

        assert (
            len(game_engine.progress["challenges_completed"]) == initial_challenges + 1
        )
        assert challenge_id in game_engine.progress["challenges_completed"]

    def test_earn_badge(self, game_engine: Any) -> None:
        """Test earning a badge."""
        badge_id = "first_steps"
        initial_badges = len(game_engine.progress["badges_earned"])

        game_engine.earn_badge(badge_id)

        assert len(game_engine.progress["badges_earned"]) == initial_badges + 1
        assert badge_id in game_engine.progress["badges_earned"]

    def test_unlock_next_level(self, game_engine: Any) -> None:
        """Test unlocking next level."""
        # Should be at level 1 initially
        assert game_engine.get_current_level() == 1

        next_level = game_engine.unlock_next_level()

        assert next_level == 2
        assert game_engine.progress["level_progress"]["2"]["unlocked"] is True
        assert game_engine.progress["current_level"] == 2

    def test_unlock_next_level_at_max(self, game_engine: Any) -> None:
        """Test unlocking next level when already at maximum."""
        game_engine.set_current_level(7)  # Set to max level

        next_level = game_engine.unlock_next_level()

        assert next_level == 7  # Should remain at 7
        assert game_engine.progress["current_level"] == 7

    def test_get_stats(self, game_engine: Any) -> None:
        """Test getting player statistics."""
        # Add some data
        game_engine.add_experience(150)
        game_engine.earn_badge("first_steps")
        game_engine.complete_challenge("level_1_test")

        stats = game_engine.get_stats()

        assert stats["level"] == 1
        assert stats["experience"] == 300  # 150 + 50 (badge) + 100 (challenge)
        assert stats["badges"] == 1
        assert stats["challenges_completed"] == 1
        assert "completion_rate" in stats

    def test_reset_progress(self, game_engine: Any) -> None:
        """Test resetting all progress."""
        # Add some progress first
        game_engine.add_experience(100)
        game_engine.earn_badge("test_badge")
        game_engine.complete_challenge("test_challenge")

        game_engine.reset_progress()

        assert game_engine.progress["experience_points"] == 0
        assert len(game_engine.progress["badges_earned"]) == 0
        assert len(game_engine.progress["challenges_completed"]) == 0
        assert game_engine.progress["current_level"] == 1

    def test_get_level_challenges(self, game_engine: Any) -> None:
        """Test getting challenges for a level."""
        challenges = game_engine.get_level_challenges(1)

        # Should return some challenges (mocked or real)
        assert isinstance(challenges, list)

    @patch("subprocess.run")
    def test_launch_jupyter_success(
        self, mock_subprocess: Any, game_engine: Any
    ) -> None:
        """Test successful Jupyter Lab launch."""
        mock_subprocess.return_value.returncode = 0

        # Should not raise exception
        game_engine.launch_jupyter()
        mock_subprocess.assert_called_once()

    @patch("subprocess.run")
    def test_launch_jupyter_failure(
        self, mock_subprocess: Any, game_engine: Any, capsys: Any
    ) -> None:
        """Test Jupyter Lab launch failure."""
        mock_subprocess.side_effect = FileNotFoundError()

        game_engine.launch_jupyter()

        captured = capsys.readouterr()
        assert "Jupyter Lab failed to start" in captured.out

    def test_count_total_challenges(self, game_engine: Any) -> None:
        """Test counting total challenges across all levels."""
        total = game_engine.count_total_challenges()

        assert total >= 1  # Should have at least 1 to avoid division by zero
        assert isinstance(total, int)

    def test_set_current_level(self, game_engine: Any) -> None:
        """Test setting current level directly."""
        game_engine.set_current_level(3)

        assert game_engine.progress["current_level"] == 3

    def test_invalid_level_setting(self, game_engine: Any) -> None:
        """Test setting invalid level values."""
        # Test setting level too high
        game_engine.set_current_level(10)
        assert game_engine.progress["current_level"] <= 7

        # Test setting level too low
        game_engine.set_current_level(0)
        assert game_engine.progress["current_level"] >= 1
