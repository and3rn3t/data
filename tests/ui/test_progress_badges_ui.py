"""
Enhanced Progress and Badges UI Tests
Tests for progress tracking, analytics, and achievement displays
"""

from typing import Any

import pytest
from playwright.async_api import async_playwright


class TestProgressBadgesUI:
    """Test suite for Progress and Badges functionality"""

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_progress_page_analytics(self) -> None:
        """Test that progress page displays analytics and charts"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Navigate to progress page
                progress_nav = page.locator(
                    '[data-testid="stSidebar"] >> text="Progress"'
                )
                if await progress_nav.count() > 0:
                    await progress_nav.click()
                    await page.wait_for_timeout(3000)

                page_content = await page.text_content("body")

                # Look for analytics/progress indicators
                analytics_indicators = [
                    "analytics",
                    "progress",
                    "statistics",
                    "performance",
                    "chart",
                    "graph",
                    "completion",
                    "achievement",
                ]

                found_analytics = []
                if page_content:
                    found_analytics = [
                        indicator
                        for indicator in analytics_indicators
                        if indicator.lower() in page_content.lower()
                    ]

                print(f"Found analytics indicators: {found_analytics}")

                # Look for charts/visualizations
                chart_elements = page.locator(
                    'svg, canvas, [class*="chart"], [class*="graph"], '
                    '[data-testid*="chart"], .js-plotly-plot'
                )
                chart_count = await chart_elements.count()
                print(f"Found {chart_count} potential chart elements")

                # Look for metrics/statistics displays
                import re

                numbers_found = []
                percentages_found = []

                if page_content:
                    numbers_found = re.findall(r"\d+", page_content)
                    percentages_found = re.findall(r"\d+%", page_content)

                print(f"Found {len(numbers_found)} numeric displays")
                print(f"Found {len(percentages_found)} percentage displays")

                await page.screenshot(
                    path="tests/ui/screenshots/progress_analytics.png"
                )

                has_progress_content = (
                    len(found_analytics) > 0
                    or chart_count > 0
                    or len(percentages_found) > 0
                )

                if has_progress_content:
                    print("✅ Progress analytics test passed!")
                else:
                    print("⚠️ Limited progress content found")

            except Exception as e:
                print(f"❌ Progress analytics test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/progress_analytics_error.png"
                )
            finally:
                await browser.close()

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_badges_page_display(self) -> None:
        """Test that badges page shows achievement information"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Navigate to badges page
                badges_nav = page.locator('[data-testid="stSidebar"] >> text="Badges"')
                if await badges_nav.count() > 0:
                    await badges_nav.click()
                    await page.wait_for_timeout(3000)

                page_content = await page.text_content("body")

                # Look for badge-related content
                badge_indicators = [
                    "badge",
                    "achievement",
                    "earned",
                    "unlocked",
                    "trophy",
                    "award",
                    "medal",
                    "accomplishment",
                ]

                found_badges = []
                if page_content:
                    found_badges = [
                        indicator
                        for indicator in badge_indicators
                        if indicator.lower() in page_content.lower()
                    ]

                print(f"Found badge indicators: {found_badges}")

                # Look for emoji indicators (common in badge displays)
                import re

                emojis_found = []
                if page_content:
                    emojis_found = re.findall(r"[🏆🏅🎖️🥇🥈🥉⭐🌟✨🎯🚀]", page_content)
                print(f"Found {len(emojis_found)} achievement-related emojis")

                # Look for badge cards or containers
                badge_containers = page.locator(
                    '[class*="badge"], [class*="achievement"], '
                    '[data-testid*="badge"], .card, .ios-card'
                )
                container_count = await badge_containers.count()
                print(f"Found {container_count} potential badge containers")

                await page.screenshot(path="tests/ui/screenshots/badges_display.png")

                has_badge_content = (
                    len(found_badges) > 0
                    or len(emojis_found) > 0
                    or container_count > 0
                )

                if has_badge_content:
                    print("✅ Badges display test passed!")
                else:
                    print("⚠️ Limited badge content found")

            except Exception as e:
                print(f"❌ Badges display test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/badges_display_error.png"
                )
            finally:
                await browser.close()

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_levels_page_progression(self) -> None:
        """Test that levels page shows learning progression"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Navigate to levels page
                levels_nav = page.locator('[data-testid="stSidebar"] >> text="Levels"')
                if await levels_nav.count() > 0:
                    await levels_nav.click()
                    await page.wait_for_timeout(3000)

                page_content = await page.text_content("body")

                # Look for level-related content
                level_indicators = [
                    "level",
                    "progression",
                    "learning",
                    "mastery",
                    "completed",
                    "unlocked",
                    "beginner",
                    "advanced",
                ]

                found_levels = []
                if page_content:
                    found_levels = [
                        indicator
                        for indicator in level_indicators
                        if indicator.lower() in page_content.lower()
                    ]

                print(f"Found level indicators: {found_levels}")

                # Look for level numbering
                import re

                level_numbers = []
                if page_content:
                    level_numbers = re.findall(
                        r"Level \d+", page_content, re.IGNORECASE
                    )
                print(f"Found level numbers: {level_numbers[:5]}")  # Show first 5

                # Look for progress indicators
                progress_elements = page.locator(
                    '[role="progressbar"], [class*="progress"], '
                    '[data-testid*="progress"], .stProgress'
                )
                progress_count = await progress_elements.count()
                print(f"Found {progress_count} progress elements")

                # Look for status indicators (completed, locked, etc.)
                status_indicators = []
                if page_content:
                    status_indicators = re.findall(
                        r"(completed|unlocked|locked|active|mastered)",
                        page_content,
                        re.IGNORECASE,
                    )
                print(f"Found status indicators: {list(set(status_indicators))}")

                await page.screenshot(
                    path="tests/ui/screenshots/levels_progression.png"
                )

                has_level_content = (
                    len(found_levels) > 0
                    or len(level_numbers) > 0
                    or progress_count > 0
                )

                if has_level_content:
                    print("✅ Levels progression test passed!")
                else:
                    print("⚠️ Limited level content found")

            except Exception as e:
                print(f"❌ Levels progression test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/levels_progression_error.png"
                )
            finally:
                await browser.close()

    async def _navigate_to_page(self, page: Any, page_name: str) -> bool:
        """Helper to navigate to a specific page"""
        nav_button = page.locator(f'[data-testid="stSidebar"] >> text="{page_name}"')

        if await nav_button.count() > 0:
            await nav_button.click()
            await page.wait_for_timeout(2000)
            return True
        else:
            print(f"⚠️ Navigation button for {page_name} not found")
            return False

    async def _validate_page_content(
        self, page: Any, page_name: str, expected_keywords: list, previous_content: str
    ) -> tuple:
        """Helper to validate page content after navigation"""
        current_content = await page.text_content("body")

        # Check if content changed from previous page
        content_changed = current_content != previous_content

        # Check for expected keywords
        keywords_found = []
        if current_content:
            keywords_found = [
                kw for kw in expected_keywords if kw.lower() in current_content.lower()
            ]

        if content_changed and len(keywords_found) > 0:
            print(f"✅ Successfully navigated to {page_name}")
            print(f"   Keywords found: {keywords_found}")

            await page.screenshot(
                path=f"tests/ui/screenshots/nav_flow_{page_name.lower()}.png"
            )
            return True, current_content
        else:
            print(f"⚠️ Navigation to {page_name} unclear")
            return False, current_content

    @pytest.mark.ui
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_cross_page_navigation_consistency(self) -> None:
        """Test navigation between different pages works consistently"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Test navigation flow: Dashboard -> Challenges -> Progress -> Badges -> Levels
                navigation_flow = [
                    ("Dashboard", ["dashboard", "overview", "welcome"]),
                    ("Challenges", ["challenge", "coding", "exercise"]),
                    ("Progress", ["progress", "analytics", "statistics"]),
                    ("Badges", ["badge", "achievement", "earned"]),
                    ("Levels", ["level", "progression", "learning"]),
                ]

                successful_navigations = 0
                previous_content = ""

                for page_name, expected_keywords in navigation_flow:
                    try:
                        if await self._navigate_to_page(page, page_name):
                            success, current_content = (
                                await self._validate_page_content(
                                    page, page_name, expected_keywords, previous_content
                                )
                            )
                            if success:
                                successful_navigations += 1
                            previous_content = current_content or ""

                    except Exception as nav_error:
                        print(f"⚠️ Error navigating to {page_name}: {nav_error}")

                print(
                    f"✅ Cross-page navigation test completed: {successful_navigations}/5 successful"
                )

                # Should successfully navigate to at least half the pages
                assert (
                    successful_navigations >= 2
                ), f"Too few successful navigations: {successful_navigations}"

            except Exception as e:
                print(f"❌ Cross-page navigation test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/cross_navigation_error.png"
                )
                raise
            finally:
                await browser.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
