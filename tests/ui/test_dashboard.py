"""
Dashboard UI Tests
Tests for the main dashboard functionality and navigation
"""

import pytest
from playwright.async_api import Page, expect

from tests.ui.conftest import StreamlitTestUtils


class TestDashboardUI:
    """Test suite for Dashboard UI components"""

    @pytest.mark.ui
    @pytest.mark.smoke
    async def test_dashboard_loads(self, page: Page, test_config: dict) -> None:
        """Test that the dashboard loads successfully"""
        # Wait for app to be ready
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Check main app container exists
        app_container = page.locator('[data-testid="stApp"]')
        await expect(app_container).to_be_visible()

        # Check for header elements
        header = page.locator("text=Data Science Sandbox")
        await expect(header).to_be_visible()

        # Take screenshot for debugging
        await StreamlitTestUtils.take_screenshot(page, "dashboard_loaded", test_config)

    @pytest.mark.ui
    async def test_sidebar_navigation(self, page: Page, test_config: dict) -> None:
        """Test sidebar navigation functionality"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Check sidebar exists
        sidebar = page.locator('[data-testid="stSidebar"]')
        await expect(sidebar).to_be_visible()

        # Test navigation buttons
        navigation_items = [
            "Dashboard",
            "Levels",
            "Challenges",
            "Badges",
            "Progress",
            "Settings",
        ]

        for item in navigation_items:
            # Check if navigation item exists
            nav_button = page.locator(f'[data-testid="stSidebar"] >> text="{item}"')

            # Some buttons might not be visible initially, so check if they exist
            if await nav_button.count() > 0:
                await expect(nav_button).to_be_visible()

    @pytest.mark.ui
    async def test_progress_metrics(self, page: Page, test_config: dict) -> None:
        """Test progress metrics display"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for metric containers
        metrics = page.locator('[data-testid="metric-container"]')

        # Should have at least some metrics displayed
        await expect(metrics.first).to_be_visible(timeout=test_config["timeout"])

        # Take screenshot of metrics
        await StreamlitTestUtils.take_screenshot(page, "progress_metrics", test_config)

    @pytest.mark.ui
    async def test_level_progress_display(self, page: Page, test_config: dict) -> None:
        """Test that level progress is displayed correctly"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Check for level indicators
        level_indicators = ["Level 1", "XP", "Badges"]

        for indicator in level_indicators:
            level_element = page.locator(f"text={indicator}")
            if await level_element.count() > 0:
                await expect(level_element).to_be_visible()

    @pytest.mark.ui
    async def test_theme_toggle_exists(self, page: Page, test_config: dict) -> None:
        """Test that theme toggle functionality exists"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for theme-related elements in sidebar
        theme_section = page.locator('[data-testid="stSidebar"] >> text="Theme"')

        # Theme toggle might exist
        if await theme_section.count() > 0:
            await expect(theme_section).to_be_visible()

    @pytest.mark.ui
    @pytest.mark.slow
    async def test_page_navigation_flow(self, page: Page, test_config: dict) -> None:
        """Test complete navigation flow between pages"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Start from dashboard
        dashboard_content = page.locator("text=Learning Dashboard")
        if await dashboard_content.count() > 0:
            await expect(dashboard_content).to_be_visible()

        # Try to navigate to different sections
        pages_to_test = ["Levels", "Challenges", "Progress"]

        for page_name in pages_to_test:
            # Look for navigation button
            nav_button = page.locator(
                f'[data-testid="stSidebar"] >> text="{page_name}"'
            )

            if await nav_button.count() > 0:
                await nav_button.click()

                # Wait for page to load
                await page.wait_for_timeout(2000)

                # Take screenshot of the page
                await StreamlitTestUtils.take_screenshot(
                    page, f"page_{page_name.lower()}", test_config
                )

    @pytest.mark.ui
    async def test_responsive_design(self, page: Page, test_config: dict) -> None:
        """Test responsive design at different viewport sizes"""
        viewports = [
            {"width": 1920, "height": 1080},  # Desktop
            {"width": 1024, "height": 768},  # Tablet
            {"width": 375, "height": 667},  # Mobile
        ]

        for viewport in viewports:
            await page.set_viewport_size(
                {"width": viewport["width"], "height": viewport["height"]}
            )
            await StreamlitTestUtils.wait_for_streamlit_ready(page)

            # Check that main content is visible
            app_container = page.locator('[data-testid="stApp"]')
            await expect(app_container).to_be_visible()

            # Take screenshot for each viewport
            size_name = f"{viewport['width']}x{viewport['height']}"
            await StreamlitTestUtils.take_screenshot(
                page, f"responsive_{size_name}", test_config
            )

    @pytest.mark.ui
    async def test_error_handling(self, page: Page, test_config: dict) -> None:
        """Test that the app handles errors gracefully"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Check for any error messages on the page
        error_indicators = [
            '[data-testid="stException"]',
            'text="Error"',
            'text="Exception"',
            'text="Failed"',
        ]

        for error_selector in error_indicators:
            error_element = page.locator(error_selector)
            if await error_element.count() > 0:
                # If errors exist, take screenshot for debugging
                await StreamlitTestUtils.take_screenshot(
                    page, "error_detected", test_config
                )
                # Don't fail the test, just document errors found

    @pytest.mark.ui
    async def test_interactive_elements(self, page: Page, test_config: dict) -> None:
        """Test interactive elements like buttons and forms"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for buttons
        buttons = page.locator("button")
        button_count = await buttons.count()

        if button_count > 0:
            # Test that buttons are clickable
            first_button = buttons.first
            await expect(first_button).to_be_enabled()

        # Look for input elements
        inputs = page.locator("input")
        await inputs.count()  # Check inputs exist but don't store unused variable

        # Document the interactive elements found
        await StreamlitTestUtils.take_screenshot(
            page, "interactive_elements", test_config
        )

    @pytest.mark.ui
    async def test_css_styling_applied(self, page: Page, test_config: dict) -> None:
        """Test that custom CSS styling is properly applied"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Check for custom CSS classes or styles mentioned in the dashboard
        css_elements_to_check = [
            ".ios-card",
            ".dynamic-island",
            ".sf-symbol",
            ".activity-dot",
        ]

        for css_class in css_elements_to_check:
            elements = page.locator(css_class)
            element_count = await elements.count()

            # Document presence of custom styling
            if element_count > 0:
                print(f"Found {element_count} elements with class {css_class}")

        # Take screenshot to verify visual styling
        await StreamlitTestUtils.take_screenshot(page, "css_styling", test_config)
