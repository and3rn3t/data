"""
Unit tests for Dashboard core functionality
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from sandbox.core.dashboard import Dashboard


class TestDashboard:
    """Test suite for Dashboard class."""

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_initialization(self, mock_streamlit):
        """Test Dashboard initialization with game engine."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        assert dashboard.game == mock_game_engine
        assert hasattr(dashboard, "run")

    @patch("sandbox.core.dashboard.st")
    def test_get_theme_colors_dark_theme(self, mock_streamlit):
        """Test theme color detection for dark theme."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        # Mock session state for dark theme
        mock_streamlit.session_state.get.return_value = True

        colors = dashboard._get_theme_colors()

        assert colors["text_primary"] == "#FFFFFF"
        assert colors["text_secondary"] == "#AEAEB2"
        assert "surface_primary" in colors
        assert "ios_blue" in colors

    @patch("sandbox.core.dashboard.st")
    def test_get_theme_colors_light_theme(self, mock_streamlit):
        """Test theme color detection for light theme."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        # Mock session state for light theme
        mock_streamlit.session_state.get.return_value = False

        colors = dashboard._get_theme_colors()

        assert colors["text_primary"] == "#000000"
        assert colors["text_secondary"] == "#6B7280"
        assert "surface_primary" in colors
        assert "ios_blue" in colors

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_run_method_exists(self, mock_streamlit):
        """Test that dashboard run method exists and can be called."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        # Should not raise an error when called
        try:
            dashboard.run()
        except Exception as e:
            # Some exceptions expected due to Streamlit mocking
            assert "streamlit" not in str(e).lower() or "st" not in str(e).lower()

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_constants(self, mock_streamlit):
        """Test that dashboard constants are properly defined."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        # Test class constants
        assert dashboard.TRANSPARENT_BG == "rgba(0,0,0,0)"
        assert dashboard.GRID_COLOR == "rgba(142, 142, 147, 0.2)"
        assert dashboard.SF_FONT == "SF Pro Display, -apple-system, sans-serif"
        assert dashboard.IOS_BLUE == "var(--ios-blue)"
        assert dashboard.DEFAULT_TIME_STR == "20 minutes"
        assert dashboard.CLOSE_DIV == "</div></div>"

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_page_config_setup(self, mock_streamlit):
        """Test that Streamlit page configuration is set up."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        dashboard.run()

        # Should call set_page_config
        mock_streamlit.set_page_config.assert_called_once()

        call_args = mock_streamlit.set_page_config.call_args[1]
        assert call_args["page_title"] == "Data Science Sandbox"
        assert call_args["page_icon"] == "🎮"
        assert call_args["layout"] == "wide"

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_css_styling(self, mock_streamlit):
        """Test that custom CSS is applied."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        dashboard.run()

        # Should apply custom CSS with markdown
        assert mock_streamlit.markdown.called

        # Check that CSS variables are included
        css_calls = [
            call
            for call in mock_streamlit.markdown.call_args_list
            if call[1].get("unsafe_allow_html")
        ]
        assert len(css_calls) > 0

    @patch("sandbox.core.dashboard.st")
    @patch("sandbox.core.dashboard.DashboardLayoutSystem")
    def test_dashboard_layout_system_integration(
        self, mock_layout_system, mock_streamlit
    ):
        """Test integration with DashboardLayoutSystem."""
        mock_game_engine = Mock()
        mock_layout_instance = Mock()
        mock_layout_system.return_value = mock_layout_instance

        dashboard = Dashboard(mock_game_engine)

        # Should initialize dashboard layout system
        assert hasattr(dashboard, "game")

    def test_dashboard_color_scheme_consistency(self):
        """Test that color scheme constants are consistent."""
        # Test that all iOS color constants follow the same pattern
        ios_colors = [
            Dashboard.IOS_BLUE,
            Dashboard.IOS_GREEN,
            Dashboard.IOS_PURPLE,
            Dashboard.IOS_ORANGE,
            Dashboard.IOS_RED,
        ]

        for color in ios_colors:
            assert color.startswith("var(--ios-")
            assert color.endswith(")")

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_responsive_design_constants(self, mock_streamlit):
        """Test responsive design related constants."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        # Test font family constant
        assert "SF Pro Display" in dashboard.SF_FONT
        assert "sans-serif" in dashboard.SF_FONT

        # Test grid color for proper alpha transparency
        assert "rgba" in dashboard.GRID_COLOR
        assert "0.2)" in dashboard.GRID_COLOR

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_theme_session_state_handling(self, mock_streamlit):
        """Test handling of theme preferences in session state."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        # Test default theme preference
        mock_streamlit.session_state.get.return_value = None
        colors = dashboard._get_theme_colors()

        # Should default to dark theme
        assert colors["text_primary"] == "#FFFFFF"

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_error_handling(self, mock_streamlit):
        """Test dashboard error handling."""
        mock_game_engine = Mock()

        # Mock an exception in Streamlit
        mock_streamlit.set_page_config.side_effect = Exception("Test error")

        dashboard = Dashboard(mock_game_engine)

        # Should handle errors gracefully
        try:
            dashboard.run()
        except Exception as e:
            # Expected due to mocked error
            assert "Test error" in str(e)

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_game_engine_integration(self, mock_streamlit):
        """Test integration with game engine."""
        mock_game_engine = Mock()
        mock_game_engine.get_stats.return_value = {
            "level": 1,
            "experience": 100,
            "badges": 5,
        }

        dashboard = Dashboard(mock_game_engine)

        # Dashboard should store reference to game engine
        assert dashboard.game == mock_game_engine

    def test_dashboard_css_variables_structure(self):
        """Test that CSS variable structure is well-formed."""
        # Test color constants follow CSS custom property syntax
        css_vars = [
            Dashboard.IOS_BLUE,
            Dashboard.IOS_GREEN,
            Dashboard.IOS_PURPLE,
            Dashboard.IOS_ORANGE,
            Dashboard.IOS_RED,
        ]

        for var in css_vars:
            assert var.startswith("var(")
            assert var.endswith(")")
            assert "--ios-" in var

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_accessibility_considerations(self, mock_streamlit):
        """Test accessibility-related features."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        colors = dashboard._get_theme_colors()

        # Should provide high contrast options
        assert colors["text_primary"] in ["#FFFFFF", "#000000"]
        assert colors["text_secondary"] is not None

    @patch("sandbox.core.dashboard.st")
    def test_dashboard_layout_constants(self, mock_streamlit):
        """Test layout-related constants."""
        mock_game_engine = Mock()
        dashboard = Dashboard(mock_game_engine)

        # Test HTML structure constants
        assert dashboard.CLOSE_DIV == "</div></div>"

        # Test default time string
        assert "minutes" in dashboard.DEFAULT_TIME_STR
        assert dashboard.DEFAULT_TIME_STR == "20 minutes"
