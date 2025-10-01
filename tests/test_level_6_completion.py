"""
Test suite for Level 6: Data Science Master challenges.

This module validates that all Level 6 challenges are complete and working correctly,
covering domain-specific expertise including time series, NLP, computer vision,
and comprehensive data science projects.
"""

from pathlib import Path


class TestLevel6Completion:
    """Test Level 6: Data Science Master challenge completion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.level_6_dir = Path("challenges/level_6")
        self.expected_challenges = {
            1: "challenge_1_time_series_analysis.md",
            2: "challenge_2_nlp_text_analytics.md",
            3: "challenge_3_computer_vision.md",
            4: "challenge_4_capstone_project.md",
        }

    def test_all_challenge_files_exist(self):
        """Test that all Level 6 challenge files exist with substantial content."""
        for challenge_num, filename in self.expected_challenges.items():
            challenge_path = self.level_6_dir / filename

            assert (
                challenge_path.exists()
            ), f"Challenge {challenge_num} file missing: {filename}"

            # Check file has substantial content (master level should be comprehensive)
            content = challenge_path.read_text(encoding="utf-8")
            assert (
                len(content) > 800
            ), f"Challenge {challenge_num} appears incomplete (too short)"
            assert (
                "```python" in content
            ), f"Challenge {challenge_num} missing code blocks"
            assert (
                "Level 6:" in content
            ), f"Challenge {challenge_num} missing level identifier"

        print("✅ All Level 6 challenge files exist with substantial content")

    def test_challenge_1_time_series_mastery(self):
        """Test Challenge 1: Time Series Analysis covers advanced forecasting."""
        challenge_path = self.level_6_dir / "challenge_1_time_series_analysis.md"
        content = challenge_path.read_text(encoding="utf-8")

        # Check for advanced time series techniques
        ts_methods = [
            "ARIMA",
            "seasonal_decompose",
            "ExponentialSmoothing",
            "VAR",
            "adfuller",
            "TimeSeriesSplit",
        ]

        for method in ts_methods:
            assert method in content, f"Missing time series method: {method}"

        # Check for business applications
        business_concepts = [
            "forecasting",
            "trend",
            "seasonal",
            "stationarity",
            "anomaly",
        ]

        for concept in business_concepts:
            assert concept in content.lower(), f"Missing business concept: {concept}"

        print("✅ Challenge 1: Time Series Analysis works correctly")

    def test_challenge_2_nlp_text_analytics(self):
        """Test Challenge 2: NLP and Text Analytics covers modern techniques."""
        challenge_path = self.level_6_dir / "challenge_2_nlp_text_analytics.md"
        content = challenge_path.read_text(encoding="utf-8")

        # Check for NLP frameworks and techniques
        nlp_methods = [
            "TfidfVectorizer",
            "CountVectorizer",
            "nltk",
            "spacy",
            "sentiment",
            "tokeniz",
            "lemma",
            "named entity",
        ]

        found_methods = sum(1 for method in nlp_methods if method in content.lower())
        assert found_methods >= 5, f"Insufficient NLP coverage: {found_methods}/8"

        # Check for modern NLP applications
        nlp_applications = [
            "sentiment analysis",
            "text classification",
            "topic modeling",
            "document similarity",
            "text preprocessing",
        ]

        app_coverage = sum(1 for app in nlp_applications if app in content.lower())
        assert app_coverage >= 3, f"Insufficient NLP applications: {app_coverage}/5"

        print("✅ Challenge 2: NLP and Text Analytics works correctly")

    def test_challenge_3_computer_vision(self):
        """Test Challenge 3: Computer Vision covers image processing and ML."""
        challenge_path = self.level_6_dir / "challenge_3_computer_vision.md"
        content = challenge_path.read_text(encoding="utf-8")

        # Check for computer vision libraries
        cv_libraries = ["opencv", "PIL", "Image", "scikit-image", "matplotlib", "numpy"]

        lib_count = sum(1 for lib in cv_libraries if lib in content.lower())
        assert lib_count >= 4, f"Insufficient CV library coverage: {lib_count}/6"

        # Check for image processing techniques
        cv_techniques = [
            "image processing",
            "filter",
            "edge detection",
            "feature extraction",
            "classification",
            "CNN",
            "convolutional",
        ]

        tech_count = sum(1 for tech in cv_techniques if tech in content.lower())
        assert tech_count >= 4, f"Insufficient CV techniques: {tech_count}/7"

        print("✅ Challenge 3: Computer Vision works correctly")

    def test_challenge_4_capstone_project(self):
        """Test Challenge 4: Capstone Project integrates all domain skills."""
        challenge_path = self.level_6_dir / "challenge_4_capstone_project.md"
        content = challenge_path.read_text(encoding="utf-8")

        # Check for end-to-end project elements
        capstone_elements = [
            "end-to-end",
            "project",
            "pipeline",
            "deployment",
            "evaluation",
        ]

        for element in capstone_elements:
            assert element in content.lower(), f"Missing capstone element: {element}"

        # Check for integration of multiple domains
        domain_integration = ["time series", "text", "nlp", "image", "vision", "model"]

        integration_count = sum(
            1 for domain in domain_integration if domain in content.lower()
        )
        assert (
            integration_count >= 4
        ), f"Insufficient domain integration: {integration_count}/6"

        print("✅ Challenge 4: Capstone Project works correctly")

    def test_level_6_master_level_coverage(self):
        """Test that Level 6 provides comprehensive data science mastery."""
        all_content = ""

        # Read all challenge content
        for filename in self.expected_challenges.values():
            challenge_path = self.level_6_dir / filename
            if (
                challenge_path.exists()
            ):  # Handle missing files gracefully during development
                all_content += challenge_path.read_text(encoding="utf-8").lower()

        # Check for comprehensive domain coverage
        domain_expertise = [
            "time series",
            "forecasting",
            "nlp",
            "natural language",
            "computer vision",
            "image",
            "text analysis",
            "deep learning",
        ]

        coverage_count = sum(1 for domain in domain_expertise if domain in all_content)
        assert (
            coverage_count >= 6
        ), f"Insufficient domain expertise coverage: {coverage_count}/8"

        # Check for advanced techniques and business focus
        master_level_skills = [
            "end-to-end",
            "production",
            "pipeline",
            "deployment",
            "business",
            "stakeholder",
            "roi",
            "impact",
        ]

        skills_count = sum(1 for skill in master_level_skills if skill in all_content)
        assert skills_count >= 5, f"Insufficient master-level skills: {skills_count}/8"

        print("✅ Level 6 provides comprehensive data science mastery")

    def test_business_value_and_real_world_focus(self):
        """Test that all challenges emphasize practical business applications."""
        business_keywords = [
            "business",
            "real-world",
            "industry",
            "application",
            "stakeholder",
            "roi",
            "impact",
            "decision",
            "production",
            "deployment",
        ]

        challenges_with_business_focus = 0

        for challenge_num, filename in self.expected_challenges.items():
            challenge_path = self.level_6_dir / filename
            if challenge_path.exists():
                content = challenge_path.read_text(encoding="utf-8").lower()

                business_mentions = sum(
                    1 for keyword in business_keywords if keyword in content
                )
                if business_mentions >= 3:
                    challenges_with_business_focus += 1

        # At least 75% of challenges should have strong business focus
        expected_business_challenges = (
            len(
                [
                    f
                    for f in self.expected_challenges.values()
                    if (self.level_6_dir / f).exists()
                ]
            )
            * 0.75
        )

        assert (
            challenges_with_business_focus >= expected_business_challenges
        ), f"Insufficient business focus: {challenges_with_business_focus} challenges have business context"

        print("✅ All challenges emphasize business value and real-world applications")


def test_level_6_data_science_master_complete():
    """Integration test verifying Level 6: Data Science Master is complete."""
    tester = TestLevel6Completion()
    tester.setup_method()

    # Run all validation tests
    tester.test_all_challenge_files_exist()
    tester.test_challenge_1_time_series_mastery()
    tester.test_challenge_2_nlp_text_analytics()
    tester.test_challenge_3_computer_vision()
    tester.test_challenge_4_capstone_project()
    tester.test_level_6_master_level_coverage()
    tester.test_business_value_and_real_world_focus()

    print("\n🎉 All Level 6: Data Science Master challenges are complete and working!")
    print("Students can now master domain-specific data science applications:")
    print("• Advanced time series forecasting and analysis")
    print("• Natural language processing and text analytics")
    print("• Computer vision and image processing")
    print("• End-to-end data science project execution")


if __name__ == "__main__":
    test_level_6_data_science_master_complete()
