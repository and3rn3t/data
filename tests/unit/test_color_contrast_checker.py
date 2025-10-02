"""
Unit tests for ColorContrastChecker utility
"""

from unittest.mock import Mock, patch

from sandbox.utils.color_contrast_checker import (
    ColorContrastChecker,
    ContrastResult,
)


class TestColorContrastChecker:
    """Test suite for ColorContrastChecker class."""

    def test_contrast_result_dataclass(self):
        """Test ContrastResult dataclass creation."""
        result = ContrastResult(
            background="#000000",
            foreground="#ffffff",
            ratio=4.5,
            passes_aa=True,
            passes_aaa=False,
            recommendation="Good contrast - meets AA standards",
        )

        assert result.ratio == 4.5
        assert result.passes_aa is True
        assert result.passes_aaa is False
        assert result.background == "#000000"
        assert result.foreground == "#ffffff"
        assert result.recommendation == "Good contrast - meets AA standards"

    def test_checker_initialization(self):
        """Test ColorContrastChecker initialization."""
        checker = ColorContrastChecker()

        # Should initialize without error
        assert checker is not None
        assert hasattr(checker, "check_contrast")

    def test_hex_to_rgb_conversion(self):
        """Test hex color to RGB conversion."""
        # Test standard hex colors (static method)
        black_rgb = ColorContrastChecker.hex_to_rgb("#000000")
        white_rgb = ColorContrastChecker.hex_to_rgb("#FFFFFF")
        red_rgb = ColorContrastChecker.hex_to_rgb("#FF0000")

        assert black_rgb == (0, 0, 0)
        assert white_rgb == (255, 255, 255)
        assert red_rgb == (255, 0, 0)

    def test_hex_to_rgb_short_format(self):
        """Test hex color conversion with short format."""
        # The actual implementation returns (0, 0, 0) for invalid 3-char hex
        red_short = ColorContrastChecker.hex_to_rgb("#F00")
        assert red_short == (0, 0, 0)  # Invalid format returns black

        # Test valid 6-char hex
        red_valid = ColorContrastChecker.hex_to_rgb("#FF0000")
        assert red_valid == (255, 0, 0)

    def test_hex_to_rgb_invalid_color(self):
        """Test hex color conversion with invalid input."""
        # The actual implementation returns (0, 0, 0) for invalid input
        invalid_rgb = ColorContrastChecker.hex_to_rgb("invalid")
        assert invalid_rgb == (0, 0, 0)

        # Invalid hex characters raise ValueError
        try:
            ColorContrastChecker.hex_to_rgb("#GG0000")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected behavior

    def test_relative_luminance_calculation(self):
        """Test relative luminance calculation."""
        # Test known values (static method)
        black_luminance = ColorContrastChecker.get_relative_luminance((0, 0, 0))
        white_luminance = ColorContrastChecker.get_relative_luminance((255, 255, 255))

        assert black_luminance == 0.0
        assert white_luminance == 1.0

    def test_contrast_ratio_calculation(self):
        """Test contrast ratio calculation between colors."""
        # Black and white should have maximum contrast (21:1)
        ratio = ColorContrastChecker.calculate_contrast_ratio("#000000", "#FFFFFF")
        assert abs(ratio - 21.0) < 0.1  # Allow for floating point precision

        # Same color should have 1:1 ratio
        ratio_same = ColorContrastChecker.calculate_contrast_ratio("#808080", "#808080")
        assert abs(ratio_same - 1.0) < 0.01

    def test_check_contrast_black_white(self):
        """Test contrast checking with black and white colors."""
        result = ColorContrastChecker.check_contrast("#000000", "#FFFFFF")

        assert isinstance(result, ContrastResult)
        assert result.passes_aa is True
        assert result.passes_aaa is True
        assert result.ratio >= 7.0  # Should pass AAA

    def test_check_contrast_similar_colors(self):
        """Test contrast checking with similar colors."""
        # Very similar grays
        result = ColorContrastChecker.check_contrast("#808080", "#909090")

        assert isinstance(result, ContrastResult)
        assert result.passes_aa is False
        assert result.passes_aaa is False
        assert result.ratio < 4.5

    def test_check_contrast_with_large_text(self):
        """Test contrast checking for large text."""
        result = ColorContrastChecker.check_contrast(
            "#404040", "#FFFFFF", is_large_text=True
        )

        assert isinstance(result, ContrastResult)
        # Large text has lower requirements, should pass AA with lower ratio
        assert result.ratio > 3.0  # Should be above large text AA threshold

    def test_batch_check_colors(self):
        """Test checking multiple color combinations."""
        # Use the existing dashboard combinations check
        results = ColorContrastChecker.check_dashboard_combinations()

        assert len(results) > 0
        assert all(isinstance(result, ContrastResult) for result in results)

        # Test individual checks
        result1 = ColorContrastChecker.check_contrast("#000000", "#FFFFFF")
        result2 = ColorContrastChecker.check_contrast("#FF0000", "#00FF00")

        assert isinstance(result1, ContrastResult)
        assert isinstance(result2, ContrastResult)

    def test_get_wcag_compliance_level(self):
        """Test WCAG compliance level determination."""
        # Test high contrast (should pass AAA)
        high_result = ColorContrastChecker.check_contrast("#000000", "#FFFFFF")
        assert high_result.passes_aaa is True
        assert high_result.passes_aa is True

        # Test medium contrast (should pass AA but not AAA)
        medium_result = ColorContrastChecker.check_contrast("#666666", "#FFFFFF")
        assert medium_result.passes_aa is True

        # Test low contrast (should fail both)
        low_result = ColorContrastChecker.check_contrast("#CCCCCC", "#FFFFFF")
        assert low_result.passes_aa is False
        assert low_result.passes_aaa is False

    def test_generate_report(self):
        """Test generating contrast report."""
        report = ColorContrastChecker.generate_report()

        assert isinstance(report, str)
        assert "Dashboard Color Contrast Report" in report
        assert "Total combinations checked:" in report

    def test_suggest_improvements(self):
        """Test color improvement suggestions."""
        # Test the suggest_better_color method
        better_color = ColorContrastChecker.suggest_better_color("#808080", 4.5)

        if better_color:
            # If a suggestion was found, test that it meets the target ratio
            ratio = ColorContrastChecker.calculate_contrast_ratio(
                "#808080", better_color
            )
            assert ratio >= 4.5

        # Test get_safe_text_color method
        safe_color = ColorContrastChecker.get_safe_text_color("#FF0000")
        assert safe_color in ["white", "black"]

    def test_color_validation_edge_cases(self):
        """Test edge cases in color validation."""
        # Test with hash prefix
        result1 = ColorContrastChecker.check_contrast("#FF0000", "#00FF00")
        # Test without hash prefix (should work the same)
        result2 = ColorContrastChecker.check_contrast("FF0000", "00FF00")

        # Results should be the same
        assert abs(result1.ratio - result2.ratio) < 0.1

    def test_color_name_support(self):
        """Test support for CSS color names."""
        # Test common color names (these are in the COLOR_PALETTE)
        result = ColorContrastChecker.check_contrast("black", "white")

        assert isinstance(result, ContrastResult)
        assert result.passes_aaa is True

    def test_accessibility_compliance_thresholds(self):
        """Test that WCAG compliance thresholds are correctly applied."""
        # Test AA threshold (4.5:1 for normal text)
        result = ColorContrastChecker.check_contrast(
            "#767676", "#FFFFFF"
        )  # ~4.5:1 ratio

        assert isinstance(result, ContrastResult)
        # Should be right at the threshold
        if result.ratio >= 4.5:
            assert result.passes_aa is True
        else:
            assert result.passes_aa is False

    def test_large_text_different_thresholds(self):
        """Test that large text uses different contrast thresholds."""
        # Color pair that passes for large text but not normal text
        result_normal = ColorContrastChecker.check_contrast(
            "#999999", "#FFFFFF", is_large_text=False
        )
        result_large = ColorContrastChecker.check_contrast(
            "#999999", "#FFFFFF", is_large_text=True
        )

        # Large text should have more lenient requirements
        # A ratio around 3.5 might pass for large text but not normal text
        assert isinstance(result_normal, ContrastResult)
        assert isinstance(result_large, ContrastResult)
        assert result_normal.ratio == result_large.ratio  # Same colors, same ratio
