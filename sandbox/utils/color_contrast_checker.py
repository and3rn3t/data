"""
Color Contrast Checker for Dashboard Components
Ensures WCAG compliance and visual accessibility
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ContrastResult:
    """Result of a contrast check"""

    background: str
    foreground: str
    ratio: float
    passes_aa: bool
    passes_aaa: bool
    recommendation: str


class ColorContrastChecker:
    """Utility to check color contrast ratios and WCAG compliance"""

    # WCAG contrast ratio requirements
    WCAG_AA_NORMAL = 4.5
    WCAG_AA_LARGE = 3.0
    WCAG_AAA_NORMAL = 7.0
    WCAG_AAA_LARGE = 4.5

    # Current dashboard color palette
    COLOR_PALETTE = {
        "--ios-blue": "#1D4ED8",
        "--ios-gray": "#6B7280",
        "--ios-gray-light": "#F9FAFB",
        "--ios-gray-dark": "#374151",
        "--ios-green": "#047857",
        "--ios-orange": "#C2410C",
        "--ios-red": "#DC2626",
        "--ios-purple": "#7C3AED",
        "--ios-pink": "#EC4899",
        "--ios-teal": "#06B6D4",
        "--ios-indigo": "#6366F1",
        "--ios-yellow": "#EAB308",
        "--text-primary": "#000000",
        "--text-secondary": "#6B7280",
        "--surface-primary": "rgba(255, 255, 255, 0.85)",
        "--surface-secondary": "rgba(242, 242, 247, 0.8)",
        "--surface-tertiary": "rgba(255, 255, 255, 0.6)",
        "white": "#FFFFFF",
        "black": "#000000",
    }

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return (0, 0, 0)  # Default to black for invalid hex
        rgb_values = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return (rgb_values[0], rgb_values[1], rgb_values[2])

    @staticmethod
    def rgba_to_rgb(
        rgba_str: str, background_rgb: Tuple[int, int, int] = (255, 255, 255)
    ) -> Tuple[int, int, int]:
        """Convert RGBA to RGB by blending with background"""
        # Extract RGBA values
        match = re.search(
            r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)", rgba_str
        )
        if not match:
            return background_rgb

        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        alpha = float(match.group(4)) if match.group(4) else 1.0

        # Blend with background
        r = int(r * alpha + background_rgb[0] * (1 - alpha))
        g = int(g * alpha + background_rgb[1] * (1 - alpha))
        b = int(b * alpha + background_rgb[2] * (1 - alpha))

        return (r, g, b)

    @staticmethod
    def get_relative_luminance(rgb: Tuple[int, int, int]) -> float:
        """Calculate relative luminance according to WCAG formula"""

        def linearize(value: int) -> float:
            c = value / 255.0
            if c <= 0.03928:
                return c / 12.92
            else:
                return math.pow((c + 0.055) / 1.055, 2.4)

        r, g, b = rgb
        return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)

    @classmethod
    def _resolve_css_var(cls, color: str) -> str:
        """Resolve CSS variable to actual color value"""
        if color.startswith("var(") and color.endswith(")"):
            var_name = color[4:-1]
            return cls.COLOR_PALETTE.get(var_name, "#000000")
        # Handle named colors
        return cls.COLOR_PALETTE.get(color, color)

    @classmethod
    def calculate_contrast_ratio(cls, color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors"""
        # Resolve CSS variables
        color1 = cls._resolve_css_var(color1)
        color2 = cls._resolve_css_var(color2)

        # Convert to RGB
        if color1.startswith("rgba"):
            rgb1 = cls.rgba_to_rgb(color1)
        else:
            rgb1 = cls.hex_to_rgb(color1)

        if color2.startswith("rgba"):
            rgb2 = cls.rgba_to_rgb(color2)
        else:
            rgb2 = cls.hex_to_rgb(color2)

        # Calculate luminance
        lum1 = cls.get_relative_luminance(rgb1)
        lum2 = cls.get_relative_luminance(rgb2)

        # Calculate contrast ratio
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)

        return (lighter + 0.05) / (darker + 0.05)

    @classmethod
    def check_contrast(
        cls, background: str, foreground: str, is_large_text: bool = False
    ) -> ContrastResult:
        """Check contrast between background and foreground colors"""
        ratio = cls.calculate_contrast_ratio(background, foreground)

        # Determine thresholds based on text size
        aa_threshold = cls.WCAG_AA_LARGE if is_large_text else cls.WCAG_AA_NORMAL
        aaa_threshold = cls.WCAG_AAA_LARGE if is_large_text else cls.WCAG_AAA_NORMAL

        passes_aa = ratio >= aa_threshold
        passes_aaa = ratio >= aaa_threshold

        # Generate recommendation
        if passes_aaa:
            recommendation = "Excellent contrast - exceeds AAA standards"
        elif passes_aa:
            recommendation = "Good contrast - meets AA standards"
        elif ratio >= 3.0:
            recommendation = "Marginal contrast - consider improving"
        else:
            recommendation = "Poor contrast - needs improvement"

        return ContrastResult(
            background=background,
            foreground=foreground,
            ratio=ratio,
            passes_aa=passes_aa,
            passes_aaa=passes_aaa,
            recommendation=recommendation,
        )

    @classmethod
    def check_dashboard_combinations(cls) -> List[ContrastResult]:
        """Check common color combinations used in the dashboard"""
        results = []

        # Common combinations to check
        text_primary = "var(--text-primary)"
        combinations = [
            # Background, Foreground, Large Text
            ("var(--ios-blue)", "white", False),
            ("var(--ios-green)", "white", False),
            ("var(--ios-orange)", "white", False),
            ("var(--ios-red)", "white", False),
            ("var(--ios-purple)", "white", False),
            ("var(--ios-gray)", "white", False),
            ("var(--ios-gray-dark)", "white", False),
            ("var(--surface-primary)", text_primary, False),
            ("var(--surface-secondary)", text_primary, False),
            ("var(--surface-tertiary)", text_primary, False),
            ("var(--ios-gray-light)", text_primary, False),
            ("white", "var(--ios-blue)", False),
            ("white", "var(--ios-green)", False),
            ("white", "var(--text-secondary)", False),
        ]

        for bg, fg, large in combinations:
            result = cls.check_contrast(bg, fg, large)
            results.append(result)

        return results

    @classmethod
    def generate_report(cls) -> str:
        """Generate a comprehensive contrast report"""
        results = cls.check_dashboard_combinations()

        report = ["Dashboard Color Contrast Report", "=" * 40, ""]

        # Group results
        excellent = [r for r in results if r.passes_aaa]
        good = [r for r in results if r.passes_aa and not r.passes_aaa]
        poor = [r for r in results if not r.passes_aa]

        # Summary
        report.extend(
            [
                f"Total combinations checked: {len(results)}",
                f"Excellent (AAA): {len(excellent)}",
                f"Good (AA): {len(good)}",
                f"Poor (Below AA): {len(poor)}",
                "",
            ]
        )

        # Detailed results
        categories = [
            ("EXCELLENT (AAA)", excellent),
            ("GOOD (AA)", good),
            ("NEEDS IMPROVEMENT", poor),
        ]
        for category, items in categories:
            if items:
                report.extend([f"{category}:", "-" * len(category)])
                for result in items:
                    bg_resolved = cls._resolve_css_var(result.background)
                    fg_resolved = cls._resolve_css_var(result.foreground)
                    report.append(
                        f"  {result.background} ({bg_resolved}) on {result.foreground} ({fg_resolved})"
                    )
                    report.append(
                        f"    Ratio: {result.ratio:.2f} - {result.recommendation}"
                    )
                report.append("")

        return "\n".join(report)

    @classmethod
    def get_safe_text_color(cls, background: str) -> str:
        """Get the safest text color (white or black) for a given background"""
        white_ratio = cls.calculate_contrast_ratio(background, "white")
        black_ratio = cls.calculate_contrast_ratio(background, "black")

        return "white" if white_ratio > black_ratio else "black"

    @classmethod
    def suggest_better_color(
        cls, background: str, target_ratio: float = 4.5
    ) -> Optional[str]:
        """Suggest a better foreground color that meets the target contrast ratio"""
        # Try different shades of grey
        for lightness in range(0, 256, 15):
            test_color = f"#{lightness:02x}{lightness:02x}{lightness:02x}"
            ratio = cls.calculate_contrast_ratio(background, test_color)
            if ratio >= target_ratio:
                return test_color

        return None


def main() -> None:
    """Run contrast check and print report"""
    checker = ColorContrastChecker()
    print(checker.generate_report())


if __name__ == "__main__":
    main()
