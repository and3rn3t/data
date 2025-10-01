"""
UI Test Runner
Provides utilities for running and managing UI tests
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UITestRunner:
    """Manages execution of UI tests for the Data Science Sandbox"""

    def __init__(self, test_dir: Optional[Path] = None):
        """Initialize the UI test runner"""
        self.test_dir = test_dir or Path(__file__).parent
        self.results: Dict = {}

    def run_smoke_tests(self) -> int:
        """Run quick smoke tests to verify basic functionality"""
        logger.info("🚀 Running UI Smoke Tests...")

        args = [
            str(self.test_dir),
            "-m",
            "smoke",
            "--tb=short",
            "-v",
            "--disable-warnings",
        ]

        return pytest.main(args)

    def run_full_ui_tests(self) -> int:
        """Run complete UI test suite"""
        logger.info("🧪 Running Full UI Test Suite...")

        args = [
            str(self.test_dir),
            "-m",
            "ui",
            "--tb=short",
            "-v",
            "--disable-warnings",
            f"--html={self.test_dir / 'reports' / 'ui_test_report.html'}",
            "--self-contained-html",
        ]

        return pytest.main(args)

    def run_specific_test_file(self, test_file: str) -> int:
        """Run tests from a specific file"""
        logger.info(f"🎯 Running tests from {test_file}...")

        test_path = self.test_dir / test_file
        if not test_path.exists():
            logger.error(f"Test file not found: {test_path}")
            return 1

        args = [str(test_path), "--tb=short", "-v", "--disable-warnings"]

        return pytest.main(args)

    def run_tests_by_marker(self, marker: str) -> int:
        """Run tests with specific marker"""
        logger.info(f"🏷️ Running tests marked as '{marker}'...")

        args = [
            str(self.test_dir),
            "-m",
            marker,
            "--tb=short",
            "-v",
            "--disable-warnings",
        ]

        return pytest.main(args)

    def check_test_environment(self) -> bool:
        """Check if test environment is properly configured"""
        logger.info("🔍 Checking UI test environment...")

        required_packages = ["playwright", "pytest", "pytest-asyncio", "pytest-html"]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.info(
                "Install with: pip install playwright pytest-asyncio pytest-html"
            )
            logger.info("Then run: playwright install")
            return False

        # Check if Playwright browsers are installed
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                browser.close()
            logger.info("✅ Playwright browser available")
        except Exception as e:
            logger.error(f"Playwright browser not available: {e}")
            logger.info("Run: playwright install")
            return False

        logger.info("✅ UI test environment ready")
        return True

    def generate_test_report(self) -> None:
        """Generate comprehensive test report"""
        logger.info("📊 Generating UI test report...")

        # Create reports directory
        reports_dir = self.test_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        # Run tests with coverage and reporting
        args = [
            str(self.test_dir),
            "-m",
            "ui",
            "--tb=short",
            "-v",
            f"--html={reports_dir / 'ui_test_report.html'}",
            "--self-contained-html",
            f"--junitxml={reports_dir / 'ui_test_results.xml'}",
        ]

        result = pytest.main(args)

        if result == 0:
            logger.info(
                f"✅ Test report generated: {reports_dir / 'ui_test_report.html'}"
            )
        else:
            logger.warning(
                f"⚠️ Tests completed with issues. Report: {reports_dir / 'ui_test_report.html'}"
            )


def main() -> None:
    """Main entry point for UI test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Data Science Sandbox UI Test Runner")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--file", type=str, help="Run specific test file")
    parser.add_argument("--marker", type=str, help="Run tests with specific marker")
    parser.add_argument("--check", action="store_true", help="Check test environment")
    parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive report"
    )

    args = parser.parse_args()

    runner = UITestRunner()

    if args.check:
        success = runner.check_test_environment()
        sys.exit(0 if success else 1)

    if args.smoke:
        sys.exit(runner.run_smoke_tests())
    elif args.full:
        sys.exit(runner.run_full_ui_tests())
    elif args.file:
        sys.exit(runner.run_specific_test_file(args.file))
    elif args.marker:
        sys.exit(runner.run_tests_by_marker(args.marker))
    elif args.report:
        runner.generate_test_report()
    else:
        # Default: run smoke tests
        print("No specific test type specified. Running smoke tests...")
        sys.exit(runner.run_smoke_tests())


if __name__ == "__main__":
    main()
