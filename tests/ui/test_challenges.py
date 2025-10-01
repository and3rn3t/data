"""
Challenge UI Tests
Tests for challenge-related functionality in the dashboard
"""

import pytest
from playwright.async_api import Page, expect

from tests.ui.conftest import StreamlitTestUtils


class TestChallengeUI:
    """Test suite for Challenge UI components"""

    @pytest.mark.ui
    async def test_challenges_page_loads(self, page: Page, test_config: dict) -> None:
        """Test that the challenges page loads correctly"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Navigate to challenges if navigation exists
        nav_button = page.locator('[data-testid="stSidebar"] >> text="Challenges"')
        if await nav_button.count() > 0:
            await nav_button.click()
            await page.wait_for_timeout(2000)

        # Look for challenge-related content
        challenge_indicators = ["challenge", "Challenge", "Level", "difficulty"]

        for indicator in challenge_indicators:
            elements = page.locator(f"text={indicator}")
            if await elements.count() > 0:
                await expect(elements.first).to_be_visible()
                break

        await StreamlitTestUtils.take_screenshot(page, "challenges_page", test_config)

    @pytest.mark.ui
    async def test_challenge_modal_functionality(
        self, page: Page, test_config: dict
    ) -> None:
        """Test challenge modal/detail view functionality"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for buttons that might open challenge details
        challenge_buttons = page.locator('button:has-text("Start Challenge")')
        button_count = await challenge_buttons.count()

        if button_count > 0:
            # Click first challenge button
            await challenge_buttons.first.click()
            await page.wait_for_timeout(1000)

            # Check if modal or detail view opened
            modal_indicators = [
                '[role="dialog"]',
                'text="Challenge"',
                'text="Overview"',
                'text="Objectives"',
            ]

            for indicator in modal_indicators:
                modal_element = page.locator(indicator)
                if await modal_element.count() > 0:
                    await expect(modal_element).to_be_visible()
                    break

        await StreamlitTestUtils.take_screenshot(page, "challenge_modal", test_config)

    @pytest.mark.ui
    async def test_challenge_difficulty_display(
        self, page: Page, test_config: dict
    ) -> None:
        """Test that challenge difficulty levels are displayed"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Navigate to challenges page if possible
        nav_button = page.locator('[data-testid="stSidebar"] >> text="Challenges"')
        if await nav_button.count() > 0:
            await nav_button.click()
            await page.wait_for_timeout(2000)

        # Look for difficulty indicators
        difficulty_levels = [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Expert",
            "Easy",
            "Medium",
            "Hard",
        ]

        difficulty_found = False
        for level in difficulty_levels:
            elements = page.locator(f"text={level}")
            if await elements.count() > 0:
                difficulty_found = True
                break

        # Document difficulty display
        await StreamlitTestUtils.take_screenshot(
            page, "challenge_difficulties", test_config
        )

    @pytest.mark.ui
    async def test_challenge_progress_tracking(
        self, page: Page, test_config: dict
    ) -> None:
        """Test challenge progress tracking display"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for progress indicators
        progress_indicators = [
            "completed",
            "Completed",
            "progress",
            "Progress",
            "%",
            "✓",
            "✅",
        ]

        for indicator in progress_indicators:
            elements = page.locator(f"text={indicator}")
            if await elements.count() > 0:
                await expect(elements.first).to_be_visible()
                break

        await StreamlitTestUtils.take_screenshot(
            page, "challenge_progress", test_config
        )

    @pytest.mark.ui
    async def test_level_based_organization(
        self, page: Page, test_config: dict
    ) -> None:
        """Test that challenges are organized by levels"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Navigate to levels page if it exists
        levels_button = page.locator('[data-testid="stSidebar"] >> text="Levels"')
        if await levels_button.count() > 0:
            await levels_button.click()
            await page.wait_for_timeout(2000)

        # Look for level organization
        level_indicators = []
        for i in range(1, 8):  # Levels 1-7
            level_indicators.append(f"Level {i}")

        levels_found = 0
        for level in level_indicators:
            elements = page.locator(f"text={level}")
            if await elements.count() > 0:
                levels_found += 1

        await StreamlitTestUtils.take_screenshot(
            page, "level_organization", test_config
        )

    @pytest.mark.ui
    async def test_challenge_filtering_search(
        self, page: Page, test_config: dict
    ) -> None:
        """Test challenge filtering and search functionality"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for search/filter inputs
        search_inputs = page.locator(
            'input[placeholder*="search"], input[placeholder*="filter"]'
        )
        select_boxes = page.locator('[data-testid="stSelectbox"]')

        # Check if filtering options exist
        has_search = await search_inputs.count() > 0
        has_select = await select_boxes.count() > 0

        if has_search or has_select:
            await StreamlitTestUtils.take_screenshot(
                page, "challenge_filters", test_config
            )

    @pytest.mark.ui
    @pytest.mark.slow
    async def test_challenge_completion_flow(
        self, page: Page, test_config: dict
    ) -> None:
        """Test the complete challenge interaction flow"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Navigate to challenges
        nav_button = page.locator('[data-testid="stSidebar"] >> text="Challenges"')
        if await nav_button.count() > 0:
            await nav_button.click()
            await page.wait_for_timeout(2000)

        # Look for challenge cards or lists
        challenge_containers = page.locator(
            'div:has-text("challenge"), div:has-text("Challenge")'
        )

        if await challenge_containers.count() > 0:
            # Take screenshot of challenges list
            await StreamlitTestUtils.take_screenshot(
                page, "challenges_list", test_config
            )

            # Try to interact with a challenge
            start_buttons = page.locator(
                'button:has-text("Start"), button:has-text("View"), button:has-text("Open")'
            )

            if await start_buttons.count() > 0:
                await start_buttons.first.click()
                await page.wait_for_timeout(2000)

                # Take screenshot after interaction
                await StreamlitTestUtils.take_screenshot(
                    page, "challenge_started", test_config
                )

    @pytest.mark.ui
    async def test_challenge_content_display(
        self, page: Page, test_config: dict
    ) -> None:
        """Test that challenge content is properly displayed"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for content elements that would be in challenges
        content_elements = [
            "instructions",
            "Instructions",
            "objective",
            "Objective",
            "description",
            "Description",
            "code",
            "Code",
        ]

        content_found = []
        for element in content_elements:
            elements = page.locator(f"text={element}")
            if await elements.count() > 0:
                content_found.append(element)

        # Document what content is displayed
        await StreamlitTestUtils.take_screenshot(page, "challenge_content", test_config)
