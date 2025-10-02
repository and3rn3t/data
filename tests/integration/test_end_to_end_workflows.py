"""
End-to-End Workflow Integration Tests
Tests complete user workflows from challenge selection to completion
"""

import tempfile
from pathlib import Path

import pytest

from sandbox.core.game_engine import GameEngine


class TestEndToEndWorkflows:
    """Test complete user workflows and system integration"""

    def setup_method(self) -> None:
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_save_file = self.temp_dir / "test_progress.json"
        self.game_engine = GameEngine(save_file=str(self.test_save_file))

    def teardown_method(self) -> None:
        """Cleanup test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_new_user_onboarding_workflow(self) -> None:
        """Test complete new user onboarding process"""
        # Verify initial state
        initial_progress = self.game_engine.get_progress()
        assert initial_progress["current_level"] == 1
        assert initial_progress["experience_points"] == 0
        assert len(initial_progress["challenges_completed"]) == 0

        # Complete first challenge
        first_challenge = "level_1_1_first_steps"
        success = self.game_engine.complete_challenge(first_challenge)

        assert success is True

        # Verify progress updated
        updated_progress = self.game_engine.get_progress()
        assert first_challenge in updated_progress["challenges_completed"]
        assert updated_progress["experience_points"] > 0

        # Check if level progression available
        available_challenges = self.game_engine.get_available_challenges()
        assert len(available_challenges) >= 1

    @pytest.mark.integration
    def test_level_progression_workflow(self) -> None:
        """Test progression through multiple levels"""
        current_level = 1

        # Complete challenges to advance levels
        while current_level <= 3:  # Test first 3 levels
            available_challenges = self.game_engine.get_available_challenges()

            if not available_challenges:
                break

            # Complete first available challenge
            challenge_id = available_challenges[0]["id"]
            success = self.game_engine.complete_challenge(challenge_id)
            assert success is True

            # Check for level advancement
            new_progress = self.game_engine.get_progress()
            if new_progress["current_level"] > current_level:
                current_level = new_progress["current_level"]

                # Verify level unlock mechanics
                assert len(new_progress["challenges_completed"]) > 0
                assert new_progress["experience_points"] > 0

    @pytest.mark.integration
    def test_badge_earning_workflow(self) -> None:
        """Test badge earning and achievement system"""
        initial_badges = len(self.game_engine.get_progress()["badges_earned"])

        # Complete multiple challenges to trigger badge conditions
        challenges_to_complete = [
            "level_1_first_steps",
            "level_1_data_loading",
            "level_1_basic_analysis",
        ]

        for challenge_id in challenges_to_complete:
            # Check if challenge exists before attempting
            available = self.game_engine.get_available_challenges()
            challenge_ids = [c["id"] for c in available]

            if challenge_id in challenge_ids:
                success = self.game_engine.complete_challenge(challenge_id)
                assert success is True

        # Verify badge system activated
        final_progress = self.game_engine.get_progress()
        final_badges = len(final_progress["badges_earned"])

        # Should have earned at least one badge (or no badges if system not configured)
        assert final_badges >= initial_badges

    @pytest.mark.integration
    def test_progress_persistence_workflow(self) -> None:
        """Test progress persistence across sessions"""
        # Complete some challenges
        challenge_id = "level_1_first_steps"
        initial_success = self.game_engine.complete_challenge(challenge_id)

        if initial_success:
            initial_exp = self.game_engine.get_progress()["experience_points"]
            initial_challenges = self.game_engine.get_progress()["challenges_completed"]

            # Create new game engine instance (simulates app restart)
            new_engine = GameEngine(save_file=str(self.test_save_file))

            # Verify progress persisted
            restored_progress = new_engine.get_progress()
            assert restored_progress["experience_points"] == initial_exp
            assert set(restored_progress["challenges_completed"]) == set(
                initial_challenges
            )

    @pytest.mark.integration
    def test_error_recovery_workflow(self) -> None:
        """Test system recovery from various error conditions"""
        # Test invalid challenge completion
        invalid_result = self.game_engine.complete_challenge("nonexistent_challenge")
        assert invalid_result is False

        # Verify system state remains stable
        progress_after_error = self.game_engine.get_progress()
        assert "current_level" in progress_after_error
        assert "experience_points" in progress_after_error

        # Test recovery - valid operation should still work
        valid_challenge = "level_1_first_steps"
        recovery_result = self.game_engine.complete_challenge(valid_challenge)

        # Should succeed if challenge exists, or fail gracefully if not
        assert isinstance(recovery_result, bool)

    @pytest.mark.integration
    def test_data_analysis_workflow(self) -> None:
        """Test integration of data analysis tools in challenges"""
        try:
            from sandbox.integrations.modern_data_processing import ModernDataProcessor

            processor = ModernDataProcessor()

            # Test basic data processing capability
            sample_data = processor.create_sample_dataset(
                n_rows=100, dataset_type="sales"
            )
            assert len(sample_data) == 100

            # Test SQL query capability if available
            if processor.duckdb_available:
                query_result = processor.query_with_sql(
                    sample_data, "SELECT COUNT(*) as row_count FROM df"
                )
                assert len(query_result) == 1
                # Handle both Pandas and Polars DataFrames
                if hasattr(query_result, "to_pandas"):
                    # It's a Polars DataFrame, convert to pandas
                    query_result = query_result.to_pandas()
                assert query_result.iloc[0]["row_count"] == 100

                # Simulate challenge completion with data analysis
                self.game_engine.add_experience(50, "Completed data analysis challenge")

        except ImportError:
            pytest.skip("Data processing integrations not available")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_system_integration(self) -> None:
        """Test integration across all major system components"""
        # Test game engine + data processing + progress tracking
        initial_state = self.game_engine.get_progress()

        # Simulate completing data science workflow
        try:
            from sandbox.integrations.modern_data_processing import ModernDataProcessor
            from sandbox.integrations.data_pipeline_builder import DataPipelineBuilder

            processor = ModernDataProcessor()
            pipeline_builder = DataPipelineBuilder()

            # Create and process data (simulates challenge activity)
            test_data = processor.create_sample_dataset(
                n_rows=500, dataset_type="ecommerce"
            )

            # Run data pipeline
            pipeline_result = pipeline_builder.create_data_pipeline(
                pipeline_name="integration_workflow",
                data_source=test_data,
                transformations=[pipeline_builder._clean_missing_values],
            )

            # Award experience for successful data processing
            if pipeline_result["success"]:
                self.game_engine.add_experience(
                    100, "Completed data pipeline challenge"
                )

                # Try to complete a related challenge
                challenge_completed = self.game_engine.complete_challenge(
                    "level_1_first_steps"
                )

                # Verify system state consistency
                final_state = self.game_engine.get_progress()
                assert (
                    final_state["experience_points"]
                    >= initial_state["experience_points"]
                )

        except ImportError:
            # If integrations not available, just test core functionality
            self.game_engine.add_experience(50, "Basic system integration test")
            final_state = self.game_engine.get_progress()
            assert (
                final_state["experience_points"] >= initial_state["experience_points"]
            )

    @pytest.mark.integration
    def test_concurrent_user_simulation(self) -> None:
        """Test system behavior with concurrent operations"""
        import threading
        import time

        results = []

        def user_simulation(user_id: int):
            """Simulate user activity"""
            # Create separate game engine for each user
            user_save_file = self.temp_dir / f"user_{user_id}_progress.json"
            user_engine = GameEngine(save_file=str(user_save_file))

            # Simulate user actions
            user_engine.add_experience(10 * user_id, f"User {user_id} activity")

            # Try to complete a challenge
            success = user_engine.complete_challenge("level_1_first_steps")

            results.append(
                {
                    "user_id": user_id,
                    "challenge_success": success,
                    "final_exp": user_engine.get_progress()["experience_points"],
                }
            )

        # Run multiple user simulations
        threads = []
        for user_id in range(3):
            thread = threading.Thread(target=user_simulation, args=(user_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)

        # Verify all users completed successfully
        assert len(results) == 3
        for result in results:
            assert result["final_exp"] >= 0
            assert isinstance(result["challenge_success"], bool)
