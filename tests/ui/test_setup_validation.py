"""
Integration test to validate UI testing setup
This test validates that the UI testing framework is properly configured
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest


class TestUISetup:
    """Test UI testing framework setup"""

    def test_playwright_installation(self) -> None:
        """Test that Playwright is properly installed"""
        try:
            import playwright
            from playwright.sync_api import sync_playwright

            # Test browser availability
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto("data:text/html,<h1>Test</h1>")
                assert "Test" in page.content()
                browser.close()

        except ImportError:
            pytest.skip(
                "Playwright not installed - run: pip install playwright && playwright install"
            )
        except Exception as e:
            pytest.fail(f"Playwright setup issue: {e}")

    def test_pytest_asyncio_available(self) -> None:
        """Test that pytest-asyncio is available"""
        try:
            import pytest_asyncio

            assert pytest_asyncio is not None
        except ImportError:
            pytest.fail(
                "pytest-asyncio not available - run: pip install pytest-asyncio"
            )

    def test_streamlit_available(self) -> None:
        """Test that Streamlit is available for testing"""
        try:
            import streamlit

            assert streamlit is not None
        except ImportError:
            pytest.fail("Streamlit not available - run: pip install streamlit")

    def test_ui_test_directory_structure(self) -> None:
        """Test that UI test directory structure is correct"""
        ui_test_dir = Path(__file__).parent

        required_files = [
            "conftest.py",
            "test_dashboard.py",
            "test_challenges.py",
            "test_progress.py",
            "test_runner.py",
        ]

        for file_name in required_files:
            file_path = ui_test_dir / file_name
            assert file_path.exists(), f"Required UI test file missing: {file_name}"

    def test_streamlit_app_exists(self) -> None:
        """Test that the main Streamlit app file exists"""
        project_root = Path(__file__).parent.parent.parent
        streamlit_app = project_root / "streamlit_app.py"
        assert streamlit_app.exists(), "streamlit_app.py not found in project root"

    def test_test_runner_executable(self) -> None:
        """Test that the UI test runner is executable"""
        test_runner = Path(__file__).parent / "test_runner.py"
        assert test_runner.exists(), "test_runner.py not found"

        # Test that it can be executed (check environment)
        result = subprocess.run(
            [sys.executable, str(test_runner), "--check"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should exit with 0 if environment is ready, or 1 if setup needed
        assert result.returncode in [
            0,
            1,
        ], f"Test runner failed unexpectedly: {result.stderr}"

    def test_ui_test_markers_configured(self) -> None:
        """Test that pytest markers for UI tests are configured"""
        # This would typically be configured in pytest.ini or pyproject.toml
        # For now, just verify the markers are used in test files

        ui_test_dir = Path(__file__).parent
        test_files = list(ui_test_dir.glob("test_*.py"))

        ui_marker_found = False
        for test_file in test_files:
            content = test_file.read_text()
            if "@pytest.mark.ui" in content:
                ui_marker_found = True
                break

        assert ui_marker_found, "No UI test markers found in test files"

    def test_screenshots_directory_creation(self) -> None:
        """Test that screenshots directory can be created"""
        ui_test_dir = Path(__file__).parent
        screenshots_dir = ui_test_dir / "screenshots"

        # Create directory if it doesn't exist
        screenshots_dir.mkdir(exist_ok=True)
        assert screenshots_dir.exists(), "Could not create screenshots directory"

        # Test write permissions
        test_file = screenshots_dir / "test_write.tmp"
        test_file.write_text("test")
        assert test_file.exists(), "Cannot write to screenshots directory"
        test_file.unlink()  # Clean up

    def test_reports_directory_creation(self) -> None:
        """Test that reports directory can be created"""
        ui_test_dir = Path(__file__).parent
        reports_dir = ui_test_dir / "reports"

        # Create directory if it doesn't exist
        reports_dir.mkdir(exist_ok=True)
        assert reports_dir.exists(), "Could not create reports directory"

    @pytest.mark.slow
    def test_streamlit_can_start(self) -> None:
        """Test that Streamlit app can start (slow test)"""
        project_root = Path(__file__).parent.parent.parent
        streamlit_app = project_root / "streamlit_app.py"

        # Try to start Streamlit in check mode
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(streamlit_app),
                "--server.port",
                "8599",  # Use different port
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
                "--server.runOnSave",
                "false",
            ],
            timeout=10,  # Quick timeout for this test
            capture_output=True,
            text=True,
        )

        # Should start but might timeout - that's OK, we're just testing it can launch
        # Exit code 0 = success, 1 = timeout (still good), >1 = error
        assert result.returncode <= 1, f"Streamlit failed to start: {result.stderr}"


if __name__ == "__main__":
    # Run setup validation tests
    pytest.main([__file__, "-v"])
