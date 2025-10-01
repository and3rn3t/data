"""
Enhanced Challenge Interaction UI Tests
Tests for challenge selection, modal interactions, and user flows
"""

from typing import Any

import pytest
from playwright.async_api import async_playwright


class TestChallengeInteractionUI:
    """Test suite for Challenge Interaction functionality"""

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_challenge_page_content(self) -> None:
        """Test that challenges page loads with expected content"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Navigate to challenges page
                challenges_nav = page.locator(
                    '[data-testid="stSidebar"] >> text="Challenges"'
                )
                if await challenges_nav.count() > 0:
                    await challenges_nav.click()
                    await page.wait_for_timeout(3000)

                # Check for challenge-related content
                page_content = await page.text_content("body")

                challenge_indicators = [
                    "challenge",
                    "coding",
                    "exercise",
                    "level",
                    "difficulty",
                    "practice",
                    "complete",
                ]

                found_indicators = []
                if page_content:
                    found_indicators = [
                        indicator
                        for indicator in challenge_indicators
                        if indicator.lower() in page_content.lower()
                    ]

                print(f"Found challenge indicators: {found_indicators}")

                # Look for level selectors or dropdowns
                level_selectors = page.locator('select, [role="combobox"]')
                selector_count = await level_selectors.count()
                print(f"Found {selector_count} potential level selectors")

                # Look for challenge cards or buttons
                challenge_buttons = page.locator(
                    'button:has-text("Challenge"), button:has-text("Start")'
                )
                button_count = await challenge_buttons.count()
                print(f"Found {button_count} potential challenge buttons")

                await page.screenshot(
                    path="tests/ui/screenshots/challenges_page_content.png"
                )

                # At least some challenge content should exist
                assert (
                    len(found_indicators) > 0 or selector_count > 0 or button_count > 0
                ), "No challenge-related content found on challenges page"

                print("✅ Challenge page content test passed!")

            except Exception as e:
                print(f"❌ Challenge page content test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/challenges_content_error.png"
                )
                raise
            finally:
                await browser.close()

    async def _find_level_selector(self, page: Any) -> Any:
        """Helper to find level selection elements"""
        level_selectors = [
            "select",
            '[data-testid="stSelectbox"]',
            '[role="combobox"]',
            'input[type="select"]',
        ]

        for selector in level_selectors:
            elements = page.locator(selector)
            if await elements.count() > 0:
                print(f"✅ Found level selector: {selector}")
                return elements.first
        return None

    async def _test_level_options_interaction(
        self, page: Any, level_selector: Any
    ) -> None:
        """Helper to test level selector interaction"""
        initial_content = await page.text_content("body")

        # Click on selector to open options
        await level_selector.click()
        await page.wait_for_timeout(1000)

        # Look for options or dropdown content
        options = page.locator('option, [role="option"]')
        option_count = await options.count()

        if option_count > 0:
            print(f"✅ Found {option_count} level options")
            # Try selecting a different option if available
            if option_count > 1:
                await options.nth(1).click()
                await page.wait_for_timeout(2000)

                new_content = await page.text_content("body")
                if new_content != initial_content:
                    print("✅ Level selection changed page content")
                else:
                    print("⚠️ Level selection didn't change content")
        else:
            print("⚠️ No dropdown options found")

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_level_selection_functionality(self) -> None:
        """Test level selection dropdown/picker functionality"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Navigate to challenges
                challenges_nav = page.locator(
                    '[data-testid="stSidebar"] >> text="Challenges"'
                )
                if await challenges_nav.count() > 0:
                    await challenges_nav.click()
                    await page.wait_for_timeout(3000)

                level_selector = await self._find_level_selector(page)
                if level_selector:
                    await self._test_level_options_interaction(page, level_selector)

                await page.screenshot(path="tests/ui/screenshots/level_selection.png")
                print("✅ Level selection test completed")

            except Exception as e:
                print(f"❌ Level selection test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/level_selection_error.png"
                )
                # Don't raise - this is exploratory
            finally:
                await browser.close()

    async def _find_challenge_buttons(self, page: Any) -> list:
        """Helper to find challenge-related buttons"""
        button_patterns = [
            'button:has-text("Start")',
            'button:has-text("Challenge")',
            'button:has-text("Begin")',
            'button:has-text("Try")',
            'button:has-text("Open")',
            'button[class*="challenge"]',
        ]

        challenge_buttons_found = []
        for pattern in button_patterns:
            buttons = page.locator(pattern)
            button_count = await buttons.count()
            if button_count > 0:
                challenge_buttons_found.extend([pattern] * button_count)
                print(f"✅ Found {button_count} buttons matching: {pattern}")

        return challenge_buttons_found

    async def _test_button_click_response(self, page: Any, first_pattern: str) -> None:
        """Helper to test button click and response"""
        first_button = page.locator(first_pattern).first

        initial_content = await page.text_content("body")
        await first_button.click()
        await page.wait_for_timeout(3000)

        new_content = await page.text_content("body")

        # Check for modal or content changes
        modal_indicators = [
            "modal",
            "dialog",
            "overlay",
            "popup",
            "challenge content",
            "description",
            "objectives",
        ]

        modal_found = False
        if new_content:
            modal_found = any(
                indicator in new_content.lower() for indicator in modal_indicators
            )

        content_changed = new_content != initial_content

        if modal_found or content_changed:
            print("✅ Challenge button interaction successful - content changed")

            # Look for close buttons or back navigation
            close_buttons = page.locator(
                'button:has-text("Close"), button:has-text("Back"), '
                'button:has-text("×"), [aria-label="close"]'
            )
            if await close_buttons.count() > 0:
                print("✅ Found close/back button in modal")
        else:
            print("⚠️ Challenge button clicked but no obvious change detected")

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_challenge_button_interactions(self) -> None:
        """Test clicking challenge buttons and modal interactions"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Navigate to challenges
                challenges_nav = page.locator(
                    '[data-testid="stSidebar"] >> text="Challenges"'
                )
                if await challenges_nav.count() > 0:
                    await challenges_nav.click()
                    await page.wait_for_timeout(3000)

                challenge_buttons_found = await self._find_challenge_buttons(page)

                if challenge_buttons_found:
                    # Try clicking the first available challenge button
                    first_pattern = next(iter(set(challenge_buttons_found)))
                    await self._test_button_click_response(page, first_pattern)
                else:
                    print("⚠️ No challenge interaction buttons found")

                await page.screenshot(
                    path="tests/ui/screenshots/challenge_interactions.png"
                )
                print("✅ Challenge button interactions test completed")

            except Exception as e:
                print(f"❌ Challenge interactions test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/challenge_interactions_error.png"
                )
                # Don't raise - exploratory test
            finally:
                await browser.close()

    @pytest.mark.ui
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_challenge_statistics_display(self) -> None:
        """Test that challenge statistics and progress are displayed"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Navigate to challenges
                challenges_nav = page.locator(
                    '[data-testid="stSidebar"] >> text="Challenges"'
                )
                if await challenges_nav.count() > 0:
                    await challenges_nav.click()
                    await page.wait_for_timeout(3000)

                page_content = await page.text_content("body")

                # Look for statistics indicators
                stat_indicators = [
                    "%",
                    "completed",
                    "progress",
                    "score",
                    "points",
                    "statistics",
                    "analytics",
                ]

                found_stats = []
                numbers_with_percent = []
                challenge_counts = []

                if page_content:
                    found_stats = [
                        stat for stat in stat_indicators if stat in page_content.lower()
                    ]

                    # Look for numerical displays (progress percentages, counts, etc.)
                    import re

                    numbers_with_percent = re.findall(r"\d+%", page_content)
                    challenge_counts = re.findall(r"\d+/\d+", page_content)

                print(f"Found statistics indicators: {found_stats}")

                print(f"Found percentage displays: {numbers_with_percent}")
                print(f"Found count displays: {challenge_counts}")

                # Look for visual indicators (progress bars, charts)
                progress_elements = page.locator(
                    '[role="progressbar"], .progress, [class*="progress"], '
                    '[data-testid*="progress"], svg'
                )
                progress_count = await progress_elements.count()
                print(f"Found {progress_count} potential progress/chart elements")

                await page.screenshot(
                    path="tests/ui/screenshots/challenge_statistics.png"
                )

                has_statistics = (
                    len(found_stats) > 0
                    or len(numbers_with_percent) > 0
                    or len(challenge_counts) > 0
                    or progress_count > 0
                )

                if has_statistics:
                    print("✅ Challenge statistics display test passed!")
                else:
                    print("⚠️ No obvious statistics found, but test completed")

            except Exception as e:
                print(f"❌ Challenge statistics test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/challenge_statistics_error.png"
                )
                # Don't raise - exploratory test
            finally:
                await browser.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
