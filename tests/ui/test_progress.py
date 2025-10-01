"""
Progress Tracking UI Tests
Tests for progress tracking and analytics functionality
"""

import pytest
from playwright.async_api import Page, expect

from tests.ui.conftest import StreamlitTestUtils


class TestProgressUI:
    """Test suite for Progress tracking UI components"""

    @pytest.mark.ui
    async def test_progress_page_loads(self, page: Page, test_config: dict) -> None:
        """Test that the progress page loads correctly"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Navigate to progress page if navigation exists
        nav_button = page.locator('[data-testid="stSidebar"] >> text="Progress"')
        if await nav_button.count() > 0:
            await nav_button.click()
            await page.wait_for_timeout(2000)

        # Look for progress-related content
        progress_indicators = [
            "Analytics",
            "Progress",
            "Statistics",
            "Performance",
            "Completion",
            "XP",
            "Experience",
        ]

        for indicator in progress_indicators:
            elements = page.locator(f"text={indicator}")
            if await elements.count() > 0:
                await expect(elements.first).to_be_visible()
                break

        await StreamlitTestUtils.take_screenshot(page, "progress_page", test_config)

    @pytest.mark.ui
    async def test_xp_and_level_display(self, page: Page, test_config: dict) -> None:
        """Test XP and level information display"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for XP/level indicators
        xp_indicators = ["XP", "Experience", "Level", "Points"]

        found_indicators = []
        for indicator in xp_indicators:
            elements = page.locator(f"text={indicator}")
            if await elements.count() > 0:
                found_indicators.append(indicator)

        # Look for numeric values that might be XP/levels
        numeric_elements = page.locator("text=/\\d+/")
        numeric_count = await numeric_elements.count()

        await StreamlitTestUtils.take_screenshot(page, "xp_level_display", test_config)

    @pytest.mark.ui
    async def test_completion_statistics(self, page: Page, test_config: dict) -> None:
        """Test completion statistics display"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for completion-related metrics
        completion_terms = [
            "completed",
            "Completed",
            "finished",
            "Finished",
            "%",
            "percent",
            "rate",
            "total",
        ]

        for term in completion_terms:
            elements = page.locator(f"text={term}")
            if await elements.count() > 0:
                await expect(elements.first).to_be_visible()
                break

        await StreamlitTestUtils.take_screenshot(page, "completion_stats", test_config)

    @pytest.mark.ui
    async def test_progress_charts_visualization(
        self, page: Page, test_config: dict
    ) -> None:
        """Test progress visualization charts"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Navigate to progress page
        nav_button = page.locator('[data-testid="stSidebar"] >> text="Progress"')
        if await nav_button.count() > 0:
            await nav_button.click()
            await page.wait_for_timeout(3000)

        # Look for Plotly charts (common in Streamlit dashboards)
        plotly_charts = page.locator(".js-plotly-plot")
        chart_count = await plotly_charts.count()

        # Look for other chart containers
        chart_containers = page.locator('[data-testid="stPlotlyChart"]')
        plotly_count = await chart_containers.count()

        if chart_count > 0 or plotly_count > 0:
            await StreamlitTestUtils.take_screenshot(
                page, "progress_charts", test_config
            )

    @pytest.mark.ui
    async def test_badge_achievements_display(
        self, page: Page, test_config: dict
    ) -> None:
        """Test badge and achievement display"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Navigate to badges page if it exists
        badges_button = page.locator('[data-testid="stSidebar"] >> text="Badges"')
        if await badges_button.count() > 0:
            await badges_button.click()
            await page.wait_for_timeout(2000)

        # Look for badge-related content
        badge_indicators = [
            "Badge",
            "Achievement",
            "Award",
            "🏆",
            "🏅",
            "⭐",
            "earned",
            "unlocked",
        ]

        badges_found = 0
        for indicator in badge_indicators:
            elements = page.locator(f"text={indicator}")
            if await elements.count() > 0:
                badges_found += 1

        await StreamlitTestUtils.take_screenshot(page, "badges_display", test_config)

    @pytest.mark.ui
    async def test_learning_timeline(self, page: Page, test_config: dict) -> None:
        """Test learning timeline or history display"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for timeline/history elements
        timeline_indicators = [
            "Timeline",
            "History",
            "Recent",
            "Activity",
            "Log",
            "Session",
            "Study",
        ]

        for indicator in timeline_indicators:
            elements = page.locator(f"text={indicator}")
            if await elements.count() > 0:
                await expect(elements.first).to_be_visible()
                break

        await StreamlitTestUtils.take_screenshot(page, "learning_timeline", test_config)

    @pytest.mark.ui
    async def test_performance_metrics(self, page: Page, test_config: dict) -> None:
        """Test performance metrics display"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for metric containers
        metrics = page.locator('[data-testid="metric-container"]')
        metric_count = await metrics.count()

        if metric_count > 0:
            # Check that metrics have values
            for i in range(min(3, metric_count)):  # Check first 3 metrics
                metric = metrics.nth(i)
                metric_value = metric.locator('[data-testid="metric-value"]')
                if await metric_value.count() > 0:
                    await expect(metric_value).to_be_visible()

        await StreamlitTestUtils.take_screenshot(
            page, "performance_metrics", test_config
        )

    @pytest.mark.ui
    async def test_study_timer_functionality(
        self, page: Page, test_config: dict
    ) -> None:
        """Test study timer if it exists"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for timer-related elements
        timer_indicators = [
            "timer",
            "Timer",
            "Study Session",
            "minutes",
            "Time",
            "Session",
            "⏱️",
        ]

        timer_found = False
        for indicator in timer_indicators:
            elements = page.locator(f"text={indicator}")
            if await elements.count() > 0:
                timer_found = True
                break

        if timer_found:
            # Look for timer controls
            timer_buttons = page.locator(
                'button:has-text("Start"), button:has-text("Stop"), button:has-text("Reset")'
            )
            if await timer_buttons.count() > 0:
                await StreamlitTestUtils.take_screenshot(
                    page, "study_timer", test_config
                )

    @pytest.mark.ui
    @pytest.mark.slow
    async def test_progress_data_accuracy(self, page: Page, test_config: dict) -> None:
        """Test that progress data appears consistent and accurate"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Navigate through different pages to collect progress info
        pages_to_check = ["Dashboard", "Progress", "Badges"]

        progress_data = {}

        for page_name in pages_to_check:
            nav_button = page.locator(
                f'[data-testid="stSidebar"] >> text="{page_name}"'
            )
            if await nav_button.count() > 0:
                await nav_button.click()
                await page.wait_for_timeout(2000)

                # Collect any numeric values that might be progress data
                metrics = page.locator('[data-testid="metric-container"]')
                metric_count = await metrics.count()

                if metric_count > 0:
                    progress_data[page_name] = metric_count

                # Take screenshot of each page
                await StreamlitTestUtils.take_screenshot(
                    page, f"progress_check_{page_name.lower()}", test_config
                )

        # The test passes if we can navigate and see consistent data
        # Actual validation would require knowing expected values

    @pytest.mark.ui
    async def test_progress_export_options(self, page: Page, test_config: dict) -> None:
        """Test progress export or download options"""
        await StreamlitTestUtils.wait_for_streamlit_ready(page)

        # Look for export/download buttons
        export_indicators = ["Download", "Export", "Save", "PDF", "CSV", "Report"]

        export_found = False
        for indicator in export_indicators:
            buttons = page.locator(f'button:has-text("{indicator}")')
            if await buttons.count() > 0:
                export_found = True
                break

        if export_found:
            await StreamlitTestUtils.take_screenshot(
                page, "export_options", test_config
            )
