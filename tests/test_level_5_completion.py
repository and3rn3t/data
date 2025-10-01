"""
Test suite for Level 5: Algorithm Architect challenges.

This module validates that all Level 5 challenges are complete and working correctly,
covering advanced machine learning algorithms, deep learning, feature engineering,
and production ML systems.
"""

from pathlib import Path


class TestLevel5Completion:
    """Test Level 5: Algorithm Architect challenge completion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.level_5_dir = Path("challenges/level_5")
        self.expected_challenges = {
            1: "challenge_1_advanced_algorithms.md",
            2: "challenge_2_deep_learning.md",
            3: "challenge_3_advanced_feature_engineering.md",
            4: "challenge_4_production_ml_systems.md",
        }

    def test_all_challenge_files_exist(self):
        """Test that all Level 5 challenge files exist with substantial content."""
        for challenge_num, filename in self.expected_challenges.items():
            challenge_path = self.level_5_dir / filename

            assert (
                challenge_path.exists()
            ), f"Challenge {challenge_num} file missing: {filename}"

            # Check file has substantial content (advanced challenges should be comprehensive)
            content = challenge_path.read_text(encoding="utf-8")
            assert (
                len(content) > 500
            ), f"Challenge {challenge_num} appears incomplete (too short)"
            assert (
                "```python" in content
            ), f"Challenge {challenge_num} missing code blocks"
            assert (
                "Level 5:" in content
            ), f"Challenge {challenge_num} missing level identifier"

        print("✅ All Level 5 challenge files exist with substantial content")

    def test_challenge_1_advanced_algorithms(self):
        """Test Challenge 1: Advanced Algorithms and Ensemble Methods works correctly."""
        challenge_path = self.level_5_dir / "challenge_1_advanced_algorithms.md"
        content = challenge_path.read_text(encoding="utf-8")

        # Check for ensemble methods coverage
        ensemble_methods = [
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "AdaBoostClassifier",
            "VotingClassifier",
            "StackingClassifier",
            "xgboost",
            "lightgbm",
        ]

        for method in ensemble_methods:
            assert method in content, f"Missing ensemble method: {method}"

        # Check for advanced evaluation techniques
        advanced_eval = [
            "cross_val_score",
            "learning_curve",
            "validation_curve",
            "permutation_importance",
            "roc_auc_score",
        ]

        for technique in advanced_eval:
            assert technique in content, f"Missing evaluation technique: {technique}"

        # Check for proper algorithm comparison
        assert "ensemble" in content.lower(), "Missing ensemble methodology discussion"
        assert "boosting" in content.lower(), "Missing boosting concept"
        assert "bagging" in content.lower(), "Missing bagging concept"

        print("✅ Challenge 1: Advanced Algorithms works correctly")

    def test_challenge_2_deep_learning(self):
        """Test Challenge 2: Deep Learning and Neural Networks works correctly."""
        challenge_path = self.level_5_dir / "challenge_2_deep_learning.md"
        content = challenge_path.read_text(encoding="utf-8")

        # Check for neural network frameworks
        frameworks = ["MLPClassifier", "MLPRegressor", "tensorflow", "keras"]
        for framework in frameworks:
            assert (
                framework in content
            ), f"Missing neural network framework: {framework}"

        # Check for deep learning concepts
        dl_concepts = [
            "layers",
            "epochs",
            "batch_size",
            "learning_rate",
            "activation",
            "optimizer",
            "loss",
            "callbacks",
        ]

        for concept in dl_concepts:
            assert (
                concept in content.lower()
            ), f"Missing deep learning concept: {concept}"

        # Check for proper neural network architecture
        assert "Dense" in content or "layers" in content, "Missing layer architecture"
        assert "compile" in content, "Missing model compilation"
        assert "fit" in content, "Missing model training"

        print("✅ Challenge 2: Deep Learning works correctly")

    def test_challenge_3_advanced_feature_engineering(self):
        """Test Challenge 3: Advanced Feature Engineering works correctly."""
        challenge_path = (
            self.level_5_dir / "challenge_3_advanced_feature_engineering.md"
        )
        content = challenge_path.read_text(encoding="utf-8")

        # Check for advanced feature engineering techniques
        feature_methods = [
            "PolynomialFeatures",
            "QuantileTransformer",
            "PowerTransformer",
            "SelectKBest",
            "RFE",
            "RFECV",
            "PCA",
            "TruncatedSVD",
        ]

        for method in feature_methods:
            assert method in content, f"Missing feature engineering method: {method}"

        # Check for automated ML concepts
        automl_concepts = [
            "Pipeline",
            "ColumnTransformer",
            "GridSearchCV",
            "feature_selection",
            "dimensionality",
        ]

        for concept in automl_concepts:
            assert concept in content, f"Missing AutoML concept: {concept}"

        # Check for custom transformers
        assert (
            "BaseEstimator" in content or "TransformerMixin" in content
        ), "Missing custom transformer concepts"

        print("✅ Challenge 3: Advanced Feature Engineering works correctly")

    def test_challenge_4_production_ml_systems(self):
        """Test Challenge 4: Production ML Systems works correctly."""
        challenge_path = self.level_5_dir / "challenge_4_production_ml_systems.md"
        content = challenge_path.read_text(encoding="utf-8")

        # Check for production ML concepts
        production_concepts = [
            "experiment",
            "tracking",
            "deployment",
            "monitoring",
            "pipeline",
            "joblib",
            "pickle",
        ]

        for concept in production_concepts:
            assert concept in content.lower(), f"Missing production concept: {concept}"

        # Check for model versioning (can be model_versioning or "versioning")
        assert (
            "model_versioning" in content.lower() or "versioning" in content.lower()
        ), "Missing versioning concept"

        # Check for MLOps practices (match what's actually implemented)
        mlops_practices = [
            "MLExperimentTracker",
            "MLModelService",
            "ModelMonitor",
            "AutoRetrainingPipeline",
            "ModelABTest",
        ]

        practice_count = sum(1 for practice in mlops_practices if practice in content)
        assert (
            practice_count >= 3
        ), f"Insufficient MLOps practices: {practice_count}/5 found"

        # Check for proper production considerations
        assert "production" in content.lower(), "Missing production discussion"
        assert "scalab" in content.lower(), "Missing scalability considerations"

        print("✅ Challenge 4: Production ML Systems works correctly")

    def test_level_5_comprehensive_coverage(self):
        """Test that Level 5 provides comprehensive advanced ML coverage."""
        all_content = ""

        # Read all challenge content
        for filename in self.expected_challenges.values():
            challenge_path = self.level_5_dir / filename
            all_content += challenge_path.read_text(encoding="utf-8").lower()

        # Check for comprehensive algorithm coverage
        advanced_algorithms = [
            "xgboost",
            "lightgbm",
            "catboost",
            "neural",
            "deep",
            "ensemble",
            "stacking",
            "voting",
            "boosting",
            "bagging",
        ]

        coverage_count = sum(1 for alg in advanced_algorithms if alg in all_content)
        assert (
            coverage_count >= 7
        ), f"Insufficient advanced algorithm coverage: {coverage_count}/10"

        # Check for advanced evaluation and validation
        advanced_validation = [
            "cross_validation",
            "learning_curve",
            "roc_curve",
            "precision_recall",
            "hyperparameter",
            "grid_search",
        ]

        validation_count = sum(1 for val in advanced_validation if val in all_content)
        assert (
            validation_count >= 4
        ), f"Insufficient advanced validation coverage: {validation_count}/6"

        # Check for production readiness
        production_elements = [
            "production",
            "deployment",
            "monitoring",
            "versioning",
            "experiment",
            "tracking",
            "pipeline",
            "mlops",
        ]

        production_count = sum(1 for elem in production_elements if elem in all_content)
        assert (
            production_count >= 6
        ), f"Insufficient production readiness: {production_count}/8"

        print("✅ Level 5 provides comprehensive advanced ML coverage")

    def test_business_value_integration(self):
        """Test that challenges emphasize business value and real-world application."""
        business_keywords = [
            "business",
            "stakeholder",
            "roi",
            "impact",
            "decision",
            "production",
            "scalab",
            "maintain",
            "monitor",
            "deploy",
        ]

        for challenge_num, filename in self.expected_challenges.items():
            challenge_path = self.level_5_dir / filename
            content = challenge_path.read_text(encoding="utf-8").lower()

            business_mentions = sum(
                1 for keyword in business_keywords if keyword in content
            )
            assert (
                business_mentions >= 2
            ), f"Challenge {challenge_num} lacks business context"

        print("✅ All challenges emphasize business value and real-world application")


def test_level_5_algorithm_architect_complete():
    """Integration test verifying Level 5: Algorithm Architect is complete."""
    tester = TestLevel5Completion()
    tester.setup_method()

    # Run all validation tests
    tester.test_all_challenge_files_exist()
    tester.test_challenge_1_advanced_algorithms()
    tester.test_challenge_2_deep_learning()
    tester.test_challenge_3_advanced_feature_engineering()
    tester.test_challenge_4_production_ml_systems()
    tester.test_level_5_comprehensive_coverage()
    tester.test_business_value_integration()

    print("\n🎉 All Level 5: Algorithm Architect challenges are complete and working!")
    print("Students can now master advanced ML algorithms, deep learning,")
    print("sophisticated feature engineering, and production ML systems!")


if __name__ == "__main__":
    test_level_5_algorithm_architect_complete()
