"""
Simple UI Test Demo
A basic test to validate the UI testing framework without server complexities
"""

import pytest
from playwright.async_api import async_playwright


class TestBasicUI:
    """Basic UI test without Streamlit dependency"""

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_playwright_basic_functionality(self) -> None:
        """Test basic Playwright functionality"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to a simple page
            await page.goto("data:text/html,<h1>Test Page</h1><p>UI Testing Works!</p>")

            # Check content
            title = page.locator("h1")
            assert await title.text_content() == "Test Page"

            paragraph = page.locator("p")
            assert await paragraph.text_content() == "UI Testing Works!"

            await browser.close()

    @pytest.mark.ui
    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_browser_automation_working(self) -> None:
        """Smoke test: Verify browser automation is functional"""
        async with async_playwright() as p:
            # Test that we can launch different browsers
            for browser_type in [p.chromium]:  # Just test chromium for now
                browser = await browser_type.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()

                # Navigate to a test page
                await page.goto("data:text/html,<div>Browser automation works!</div>")

                # Verify content
                content = await page.text_content("body")
                assert content and "Browser automation works!" in content

                await browser.close()


if __name__ == "__main__":
    # Run this test directly
    pytest.main([__file__, "-v"])
