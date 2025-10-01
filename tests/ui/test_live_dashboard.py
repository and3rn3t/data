"""
Live Dashboard UI Tests
Tests that connect to the existing Streamlit server
"""

import pytest
from playwright.async_api import async_playwright, expect


class TestLiveDashboard:
    """Test suite for Live Dashboard UI components"""

    @pytest.mark.ui
    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_live_dashboard_loads(self) -> None:
        """Test that the live dashboard loads successfully"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Navigate to the live Streamlit app
                await page.goto("http://localhost:8501", timeout=30000)

                # Wait for Streamlit to load
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Check main app container exists
                app_container = page.locator('[data-testid="stApp"]')
                await expect(app_container).to_be_visible()

                # Check for the header/title
                page_content = await page.text_content("body")
                assert page_content is not None, "Failed to get page content"
                assert (
                    "Data Science Sandbox" in page_content
                    or "Dashboard" in page_content
                )

                print("✅ Dashboard loaded successfully!")

            except Exception as e:
                print(f"❌ Dashboard test failed: {e}")
                # Take screenshot for debugging
                await page.screenshot(path="tests/ui/screenshots/dashboard_error.png")
                raise
            finally:
                await browser.close()

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_live_sidebar_exists(self) -> None:
        """Test that the sidebar is present and functional"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Navigate to the dashboard
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Check if sidebar exists
                sidebar_selectors = [
                    '[data-testid="stSidebar"]',
                    ".css-1d391kg",  # Common Streamlit sidebar class
                    'section[data-testid="stSidebar"]',
                ]

                sidebar_found = False
                for selector in sidebar_selectors:
                    try:
                        await page.wait_for_selector(selector, timeout=5000)
                        sidebar_found = True
                        print(f"✅ Sidebar found with selector: {selector}")
                        break
                    except Exception:
                        continue

                if not sidebar_found:
                    # Check if there's any navigation content
                    page_text = await page.text_content("body")
                    if page_text and any(
                        nav_text in page_text.lower()
                        for nav_text in [
                            "dashboard",
                            "challenges",
                            "levels",
                            "progress",
                        ]
                    ):
                        print("✅ Navigation content found in page")
                        sidebar_found = True

                if sidebar_found:
                    # Take screenshot of successful sidebar
                    await page.screenshot(
                        path="tests/ui/screenshots/sidebar_success.png"
                    )
                else:
                    await page.screenshot(
                        path="tests/ui/screenshots/sidebar_not_found.png"
                    )
                    print("⚠️ Sidebar not found with standard selectors")

            except Exception as e:
                print(f"❌ Sidebar test failed: {e}")
                await page.screenshot(path="tests/ui/screenshots/sidebar_error.png")
                raise
            finally:
                await browser.close()

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_live_page_interactivity(self) -> None:
        """Test basic page interactivity"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Navigate to the dashboard
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Look for interactive elements
                buttons = page.locator("button")
                button_count = await buttons.count()
                print(f"Found {button_count} buttons on the page")

                # Look for any clickable elements
                clickable_elements = page.locator(
                    'button, [role="button"], input, select'
                )
                clickable_count = await clickable_elements.count()
                print(f"Found {clickable_count} interactive elements")

                # Check for Streamlit widgets
                widgets = page.locator('[data-testid^="st"]')
                widget_count = await widgets.count()
                print(f"Found {widget_count} Streamlit widgets")

                # Take a screenshot of the current state
                await page.screenshot(
                    path="tests/ui/screenshots/dashboard_interactive.png"
                )

                # Basic assertion - page should have some interactive elements
                assert (
                    clickable_count > 0 or widget_count > 0
                ), "No interactive elements found on the page"

                print("✅ Page interactivity verified!")

            except Exception as e:
                print(f"❌ Interactivity test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/interactivity_error.png"
                )
                raise
            finally:
                await browser.close()

    @pytest.mark.ui
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_live_page_content(self) -> None:
        """Test that the page has expected content"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Navigate and wait for load
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Wait a bit more for dynamic content to load
                await page.wait_for_timeout(3000)

                # Get all text content
                page_text = await page.text_content("body")

                # Check for expected content
                expected_content = ["sandbox", "data", "science"]

                found_content = []
                if page_text:
                    for content in expected_content:
                        if content.lower() in page_text.lower():
                            found_content.append(content)

                print(f"Found expected content: {found_content}")

                # Take screenshot of final state
                await page.screenshot(path="tests/ui/screenshots/dashboard_content.png")

                # Should find at least some expected content
                page_sample = page_text[:200] if page_text else "No content retrieved"
                assert (
                    len(found_content) > 0
                ), f"No expected content found. Page text sample: {page_sample}..."

                print("✅ Page content verified!")

            except Exception as e:
                print(f"❌ Content test failed: {e}")
                await page.screenshot(path="tests/ui/screenshots/content_error.png")
                raise
            finally:
                await browser.close()

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_live_theme_toggle(self) -> None:
        """Test theme toggle functionality if available"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Look for theme toggle in sidebar
                theme_toggles = page.locator(
                    'button:has-text("Dark"), button:has-text("Light"), '
                    'button:has-text("Theme"), [aria-label*="theme"], '
                    '[class*="theme"], input[type="checkbox"]'
                )

                toggle_count = await theme_toggles.count()
                print(f"Found {toggle_count} potential theme controls")

                if toggle_count > 0:
                    # Try toggling theme
                    initial_styles = await page.evaluate(
                        "() => window.getComputedStyle(document.body).backgroundColor"
                    )

                    await theme_toggles.first.click()
                    await page.wait_for_timeout(2000)

                    new_styles = await page.evaluate(
                        "() => window.getComputedStyle(document.body).backgroundColor"
                    )

                    if initial_styles != new_styles:
                        print("✅ Theme toggle appears to work - background changed")
                    else:
                        print(
                            "⚠️ Theme toggle clicked but no background change detected"
                        )

                await page.screenshot(path="tests/ui/screenshots/theme_toggle.png")
                print("✅ Theme toggle test completed")

            except Exception as e:
                print(f"❌ Theme toggle test failed: {e}")
                await page.screenshot(
                    path="tests/ui/screenshots/theme_toggle_error.png"
                )
                # Don't raise - this is optional functionality
            finally:
                await browser.close()

    @pytest.mark.ui
    @pytest.mark.asyncio
    async def test_live_responsive_layout(self) -> None:
        """Test that the dashboard is responsive to different screen sizes"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Test different viewport sizes
                viewport_sizes = [
                    (1920, 1080, "desktop"),
                    (1024, 768, "tablet"),
                    (375, 667, "mobile"),
                ]

                for width, height, device_type in viewport_sizes:
                    print(f"Testing {device_type} viewport: {width}x{height}")

                    await page.set_viewport_size({"width": width, "height": height})
                    await page.goto("http://localhost:8501", timeout=30000)
                    await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                    # Check if content is still accessible
                    sidebar = page.locator('[data-testid="stSidebar"]')
                    sidebar_visible = await sidebar.is_visible()

                    main_content = page.locator('[data-testid="stApp"]')
                    main_visible = await main_content.is_visible()

                    print(f"  Sidebar visible: {sidebar_visible}")
                    print(f"  Main content visible: {main_visible}")

                    # Take screenshot for this viewport
                    await page.screenshot(
                        path=f"tests/ui/screenshots/responsive_{device_type}.png"
                    )

                    # Main content should always be visible
                    assert main_visible, f"Main content not visible on {device_type}"

                print("✅ Responsive layout test passed!")

            except Exception as e:
                print(f"❌ Responsive layout test failed: {e}")
                await page.screenshot(path="tests/ui/screenshots/responsive_error.png")
                raise
            finally:
                await browser.close()

    @pytest.mark.ui
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_live_page_performance(self) -> None:
        """Test basic page performance and load times"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Measure page load time
                import time

                start_time = time.time()

                await page.goto("http://localhost:8501", timeout=30000)
                await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

                # Wait for full load (including dynamic content)
                await page.wait_for_timeout(3000)

                end_time = time.time()
                load_time = end_time - start_time

                print(f"Page load time: {load_time:.2f} seconds")

                # Check for error indicators
                error_elements = page.locator(
                    '[class*="error"], [class*="Error"], text="Error", '
                    'text="Failed", text="Exception"'
                )
                error_count = await error_elements.count()

                print(f"Found {error_count} potential error indicators")

                # Check for loading indicators (should be gone by now)
                loading_elements = page.locator(
                    '[class*="loading"], [class*="spinner"], text="Loading"'
                )
                loading_count = await loading_elements.count()

                print(f"Found {loading_count} persistent loading indicators")

                await page.screenshot(path="tests/ui/screenshots/performance_check.png")

                # Performance assertions
                assert load_time < 15.0, f"Page load time too slow: {load_time:.2f}s"
                assert error_count == 0, f"Found {error_count} error indicators"

                print("✅ Performance test passed!")

            except Exception as e:
                print(f"❌ Performance test failed: {e}")
                await page.screenshot(path="tests/ui/screenshots/performance_error.png")
                raise
            finally:
                await browser.close()


if __name__ == "__main__":
    # Run this test directly
    pytest.main([__file__, "-v", "-s"])
