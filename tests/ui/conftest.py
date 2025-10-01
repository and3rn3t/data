"""
Test configuration and fixtures for UI testing
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, Optional

import pytest
import pytest_asyncio
from playwright.async_api import Browser, BrowserContext, Page, async_playwright


class StreamlitTestServer:
    """Manages Streamlit test server lifecycle"""

    def __init__(self, app_path: str, port: int = 8502):
        self.app_path = app_path
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"

    def start(self) -> None:
        """Start the Streamlit server"""
        if self.process:
            return

        # Start Streamlit with test configuration
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            self.app_path,
            "--server.port",
            str(self.port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
            "--server.runOnSave",
            "false",
            "--server.address",
            "localhost",
        ]

        # Set environment for testing
        env = os.environ.copy()
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        env["STREAMLIT_SERVER_HEADLESS"] = "true"

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for server to be ready
        max_wait = 30  # seconds
        wait_time = 0
        while wait_time < max_wait:
            try:
                import requests  # type: ignore

                response = requests.get(f"{self.base_url}/healthz", timeout=2)
                if response.status_code == 200:
                    break
            except (requests.exceptions.RequestException, ImportError):
                pass

            time.sleep(1)
            wait_time += 1

        if wait_time >= max_wait:
            self.stop()
            raise RuntimeError(f"Streamlit server failed to start after {max_wait}s")

    def stop(self) -> None:
        """Stop the Streamlit server"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def is_running(self) -> bool:
        """Check if server is running"""
        return self.process is not None and self.process.poll() is None


@pytest.fixture(scope="session")
def streamlit_server() -> Generator[StreamlitTestServer, None, None]:
    """Session-scoped Streamlit server fixture"""
    app_path = Path(__file__).parent.parent.parent / "streamlit_app.py"
    server = StreamlitTestServer(str(app_path))

    try:
        server.start()
        yield server
    finally:
        server.stop()


@pytest_asyncio.fixture(scope="session")
async def browser() -> AsyncGenerator[Browser, None]:
    """Session-scoped browser fixture"""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
            ],
        )
        yield browser
        await browser.close()


@pytest_asyncio.fixture
async def context(browser: Browser) -> AsyncGenerator[BrowserContext, None]:
    """Browser context fixture with custom configuration"""
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        timezone_id="America/New_York",
    )
    yield context
    await context.close()


@pytest_asyncio.fixture
async def page(
    context: BrowserContext, streamlit_server: StreamlitTestServer
) -> AsyncGenerator[Page, None]:
    """Page fixture with Streamlit app loaded"""
    page = await context.new_page()

    # Navigate to Streamlit app
    await page.goto(streamlit_server.base_url)

    # Wait for Streamlit to fully load
    await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

    yield page
    await page.close()


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration fixture"""
    return {
        "timeout": 10000,  # Default timeout in ms
        "slow_timeout": 30000,  # Timeout for slow operations
        "screenshot_dir": Path(__file__).parent / "screenshots",
        "test_data_dir": Path(__file__).parent / "test_data",
    }


@pytest.fixture(autouse=True)
def setup_test_environment(test_config: Dict[str, Any]) -> Generator[None, None, None]:
    """Setup test environment for each test"""
    # Create screenshot directory
    test_config["screenshot_dir"].mkdir(exist_ok=True)

    yield

    # Cleanup after test if needed


# Utility functions for UI testing
class StreamlitTestUtils:
    """Utility class for Streamlit UI testing"""

    @staticmethod
    async def wait_for_streamlit_ready(page: Page) -> None:
        """Wait for Streamlit app to be fully loaded"""
        await page.wait_for_selector('[data-testid="stApp"]')
        await page.wait_for_load_state("networkidle")

    @staticmethod
    async def click_sidebar_button(page: Page, button_text: str) -> None:
        """Click a button in the sidebar"""
        await page.click(f'[data-testid="stSidebar"] >> text="{button_text}"')

    @staticmethod
    async def get_metric_value(page: Page, metric_label: str) -> str:
        """Get value from a metric widget"""
        metric_selector = (
            f'[data-testid="metric-container"] >> text="{metric_label}" >> ..'
        )
        result = await page.text_content(
            f"{metric_selector} >> [data-testid='metric-value']"
        )
        return result or ""

    @staticmethod
    async def take_screenshot(page: Page, name: str, config: Dict[str, Any]) -> Path:
        """Take screenshot for debugging"""
        screenshot_path = config["screenshot_dir"] / f"{name}.png"
        await page.screenshot(path=str(screenshot_path))
        return Path(screenshot_path)

    @staticmethod
    async def wait_for_element(page: Page, selector: str) -> None:
        """Wait for element to be visible"""
        await page.wait_for_selector(selector, timeout=10000)

    @staticmethod
    async def check_element_visible(page: Page, selector: str) -> bool:
        """Check if element is visible"""
        try:
            await page.wait_for_selector(selector, timeout=1000)
            return True
        except Exception:
            return False


# Pytest configuration
def pytest_configure(config: Any) -> None:
    """Configure pytest for UI testing"""
    config.addinivalue_line("markers", "ui: UI/E2E tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "smoke: Quick smoke tests")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Modify test collection for UI tests"""
    for item in items:
        # Add ui marker to all tests in ui directory
        if "ui" in str(item.fspath):
            item.add_marker(pytest.mark.ui)
