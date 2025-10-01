"""
Integration Test Runner
Provides utilities for running and managing integration tests
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest


class IntegrationTestRunner:
    """Manages execution of integration tests for the Data Science Sandbox"""

    def __init__(self, test_dir: Optional[Path] = None):
        """Initialize the integration test runner"""
        self.test_dir = test_dir or Path(__file__).parent
        self.results: Dict = {}

    def run_database_integration_tests(self) -> int:
        """Run database integration tests"""
        print("🗄️ Running Database Integration Tests...")

        args = [
            str(self.test_dir / "test_database_integration.py"),
            "-m",
            "database or integration",
            "--tb=short",
            "-v",
            "--disable-warnings",
        ]

        return pytest.main(args)

    def run_pipeline_integration_tests(self) -> int:
        """Run data pipeline integration tests"""
        print("🔄 Running Data Pipeline Integration Tests...")

        args = [
            str(self.test_dir / "test_data_pipeline_integration.py"),
            "-m",
            "pipeline or integration",
            "--tb=short",
            "-v",
            "--disable-warnings",
        ]

        return pytest.main(args)

    def run_mlflow_integration_tests(self) -> int:
        """Run MLflow integration tests"""
        print("📊 Running MLflow Integration Tests...")

        args = [
            str(self.test_dir / "test_mlflow_integration.py"),
            "-m",
            "mlflow or integration",
            "--tb=short",
            "-v",
            "--disable-warnings",
        ]

        return pytest.main(args)

    def run_workflow_integration_tests(self) -> int:
        """Run end-to-end workflow tests"""
        print("🔄 Running End-to-End Workflow Tests...")

        args = [
            str(self.test_dir / "test_end_to_end_workflows.py"),
            "-m",
            "workflow or integration",
            "--tb=short",
            "-v",
            "--disable-warnings",
        ]

        return pytest.main(args)

    def run_all_integration_tests(self) -> int:
        """Run complete integration test suite"""
        print("🧪 Running Complete Integration Test Suite...")

        args = [
            str(self.test_dir),
            "-m",
            "integration",
            "--tb=short",
            "-v",
            "--disable-warnings",
            f"--html={self.test_dir / 'reports' / 'integration_test_report.html'}",
            "--self-contained-html",
        ]

        return pytest.main(args)

    def run_fast_integration_tests(self) -> int:
        """Run fast integration tests (excluding slow tests)"""
        print("⚡ Running Fast Integration Tests...")

        args = [
            str(self.test_dir),
            "-m",
            "integration and not slow",
            "--tb=short",
            "-v",
            "--disable-warnings",
        ]

        return pytest.main(args)

    def run_slow_integration_tests(self) -> int:
        """Run slow integration tests only"""
        print("🐌 Running Slow Integration Tests...")

        args = [
            str(self.test_dir),
            "-m",
            "integration and slow",
            "--tb=short",
            "-v",
            "--disable-warnings",
        ]

        return pytest.main(args)

    def check_integration_requirements(self) -> bool:
        """Check if integration test requirements are met"""
        print("🔍 Checking Integration Test Requirements...")

        requirements = {
            "pandas": "Data processing",
            "pytest": "Test framework",
            "duckdb": "Database operations (optional)",
            "mlflow": "ML experiment tracking (optional)",
        }

        missing_requirements = []
        optional_missing = []

        for package, description in requirements.items():
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package}: {description}")
            except ImportError:
                if "optional" in description:
                    optional_missing.append(f"{package}: {description}")
                else:
                    missing_requirements.append(f"{package}: {description}")
                print(f"❌ {package}: {description}")

        if missing_requirements:
            print(f"\n❌ Missing required packages:")
            for req in missing_requirements:
                print(f"   - {req}")
            return False

        if optional_missing:
            print(f"\n⚠️ Missing optional packages (some tests will be skipped):")
            for req in optional_missing:
                print(f"   - {req}")

        print("\n✅ Integration test environment ready")
        return True

    def generate_integration_report(self) -> None:
        """Generate comprehensive integration test report"""
        print("📊 Generating Integration Test Report...")

        # Create reports directory
        reports_dir = self.test_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        # Run tests with comprehensive reporting
        args = [
            str(self.test_dir),
            "-m",
            "integration",
            "--tb=short",
            "-v",
            f"--html={reports_dir / 'integration_test_report.html'}",
            "--self-contained-html",
            f"--junitxml={reports_dir / 'integration_test_results.xml'}",
            "--cov=sandbox",
            f"--cov-report=html:{reports_dir / 'integration_coverage'}",
            "--cov-report=term",
        ]

        result = pytest.main(args)

        if result == 0:
            print(
                f"✅ Integration test report generated: {reports_dir / 'integration_test_report.html'}"
            )
        else:
            print(
                f"⚠️ Integration tests completed with issues. Report: {reports_dir / 'integration_test_report.html'}"
            )

    def run_integration_smoke_tests(self) -> int:
        """Run quick smoke tests for integration components"""
        print("💨 Running Integration Smoke Tests...")

        # Test basic imports and availability
        smoke_test_code = '''
import pytest

def test_core_imports():
    """Test that core modules can be imported"""
    from sandbox.core.game_engine import GameEngine
    from sandbox.integrations.modern_data_processing import ModernDataProcessor
    assert GameEngine is not None
    assert ModernDataProcessor is not None

def test_basic_functionality():
    """Test basic functionality works"""
    from sandbox.integrations.modern_data_processing import ModernDataProcessor
    processor = ModernDataProcessor()

    # Test data creation
    data = processor.create_sample_dataset(n_rows=10, dataset_type="sales")
    assert len(data) == 10
'''

        # Write smoke test
        smoke_test_file = self.test_dir / "test_integration_smoke.py"
        with open(smoke_test_file, "w") as f:
            f.write(smoke_test_code)

        try:
            args = [
                str(smoke_test_file),
                "--tb=short",
                "-v",
                "--disable-warnings",
            ]

            result = pytest.main(args)
            return result

        finally:
            # Cleanup smoke test file
            if smoke_test_file.exists():
                smoke_test_file.unlink()


def main() -> None:
    """Main entry point for integration test runner"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Science Sandbox Integration Test Runner"
    )
    parser.add_argument(
        "--database", action="store_true", help="Run database integration tests"
    )
    parser.add_argument(
        "--pipeline", action="store_true", help="Run pipeline integration tests"
    )
    parser.add_argument(
        "--mlflow", action="store_true", help="Run MLflow integration tests"
    )
    parser.add_argument(
        "--workflow", action="store_true", help="Run workflow integration tests"
    )
    parser.add_argument("--all", action="store_true", help="Run all integration tests")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only")
    parser.add_argument("--slow", action="store_true", help="Run slow tests only")
    parser.add_argument("--check", action="store_true", help="Check requirements")
    parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive report"
    )
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")

    args = parser.parse_args()

    runner = IntegrationTestRunner()

    if args.check:
        success = runner.check_integration_requirements()
        sys.exit(0 if success else 1)

    if args.smoke:
        sys.exit(runner.run_integration_smoke_tests())
    elif args.database:
        sys.exit(runner.run_database_integration_tests())
    elif args.pipeline:
        sys.exit(runner.run_pipeline_integration_tests())
    elif args.mlflow:
        sys.exit(runner.run_mlflow_integration_tests())
    elif args.workflow:
        sys.exit(runner.run_workflow_integration_tests())
    elif args.all:
        sys.exit(runner.run_all_integration_tests())
    elif args.fast:
        sys.exit(runner.run_fast_integration_tests())
    elif args.slow:
        sys.exit(runner.run_slow_integration_tests())
    elif args.report:
        runner.generate_integration_report()
    else:
        # Default: run fast integration tests
        print("No specific test type specified. Running fast integration tests...")
        sys.exit(runner.run_fast_integration_tests())


if __name__ == "__main__":
    main()
