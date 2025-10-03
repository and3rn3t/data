"""
iOS 16+ Human Interface Guidelines Design System
Complete implementation following Apple's design principles
https://developer.apple.com/design/human-interface-guidelines/
"""

from typing import Optional


class IOSDesignSystem:
    """
    Complete iOS 16+ HIG Design System for Professional Applications
    Implements Apple's design principles for spacing, typography, colors, and components
    """

    # ==========================================
    # iOS 16+ Color System
    # ==========================================
    COLORS = {
        # System Colors (Light Mode)
        "system_blue": "#007AFF",
        "system_green": "#34C759",
        "system_indigo": "#5856D6",
        "system_orange": "#FF9500",
        "system_pink": "#FF2D92",
        "system_purple": "#AF52DE",
        "system_red": "#FF3B30",
        "system_teal": "#30D158",
        "system_yellow": "#FFCC00",
        # Gray Colors
        "system_gray": "#8E8E93",
        "system_gray2": "#AEAEB2",
        "system_gray3": "#C7C7CC",
        "system_gray4": "#D1D1D6",
        "system_gray5": "#E5E5EA",
        "system_gray6": "#F2F2F7",
        # Label Colors
        "label": "#000000",
        "secondary_label": "#3C3C434D",
        "tertiary_label": "#3C3C4326",
        "quaternary_label": "#3C3C431F",
        # Fill Colors
        "system_fill": "#78788033",
        "secondary_system_fill": "#78788028",
        "tertiary_system_fill": "#7676801E",
        "quaternary_system_fill": "#74748014",
        # Background Colors
        "system_background": "#FFFFFF",
        "secondary_system_background": "#F2F2F7",
        "tertiary_system_background": "#FFFFFF",
        # Grouped Background Colors
        "system_grouped_background": "#F2F2F7",
        "secondary_system_grouped_background": "#FFFFFF",
        "tertiary_system_grouped_background": "#F2F2F7",
        # Separator Colors
        "separator": "#3C3C4349",
        "opaque_separator": "#C6C6C8",
    }

    # Dark Mode Colors
    DARK_COLORS = {
        # System Colors (Dark Mode)
        "system_blue": "#0A84FF",
        "system_green": "#30D158",
        "system_indigo": "#5E5CE6",
        "system_orange": "#FF9F0A",
        "system_pink": "#FF375F",
        "system_purple": "#BF5AF2",
        "system_red": "#FF453A",
        "system_teal": "#40CBE0",
        "system_yellow": "#FFD60A",
        # Gray Colors (Dark)
        "system_gray": "#8E8E93",
        "system_gray2": "#636366",
        "system_gray3": "#48484A",
        "system_gray4": "#3A3A3C",
        "system_gray5": "#2C2C2E",
        "system_gray6": "#1C1C1E",
        # Label Colors (Dark)
        "label": "#FFFFFF",
        "secondary_label": "#EBEBF54C",
        "tertiary_label": "#EBEBF530",
        "quaternary_label": "#EBEBF52E",
        # Background Colors (Dark)
        "system_background": "#000000",
        "secondary_system_background": "#1C1C1E",
        "tertiary_system_background": "#2C2C2E",
        # Grouped Background Colors (Dark)
        "system_grouped_background": "#000000",
        "secondary_system_grouped_background": "#1C1C1E",
        "tertiary_system_grouped_background": "#2C2C2E",
    }

    # ==========================================
    # Typography System (iOS 16+ SF Pro)
    # ==========================================
    TYPOGRAPHY = {
        # Large Title
        "large_title": {
            "font_size": "34px",
            "line_height": "41px",
            "font_weight": "400",
            "letter_spacing": "0.37px",
        },
        # Title 1
        "title1": {
            "font_size": "28px",
            "line_height": "34px",
            "font_weight": "400",
            "letter_spacing": "0.36px",
        },
        # Title 2
        "title2": {
            "font_size": "22px",
            "line_height": "28px",
            "font_weight": "400",
            "letter_spacing": "0.35px",
        },
        # Title 3
        "title3": {
            "font_size": "20px",
            "line_height": "25px",
            "font_weight": "400",
            "letter_spacing": "0.38px",
        },
        # Headline
        "headline": {
            "font_size": "17px",
            "line_height": "22px",
            "font_weight": "600",
            "letter_spacing": "-0.43px",
        },
        # Body
        "body": {
            "font_size": "17px",
            "line_height": "22px",
            "font_weight": "400",
            "letter_spacing": "-0.43px",
        },
        # Callout
        "callout": {
            "font_size": "16px",
            "line_height": "21px",
            "font_weight": "400",
            "letter_spacing": "-0.32px",
        },
        # Subheadline
        "subheadline": {
            "font_size": "15px",
            "line_height": "20px",
            "font_weight": "400",
            "letter_spacing": "-0.24px",
        },
        # Footnote
        "footnote": {
            "font_size": "13px",
            "line_height": "18px",
            "font_weight": "400",
            "letter_spacing": "-0.08px",
        },
        # Caption 1
        "caption1": {
            "font_size": "12px",
            "line_height": "16px",
            "font_weight": "400",
            "letter_spacing": "0px",
        },
        # Caption 2
        "caption2": {
            "font_size": "11px",
            "line_height": "13px",
            "font_weight": "400",
            "letter_spacing": "0.07px",
        },
    }

    # ==========================================
    # Spacing System (iOS Standard Points)
    # ==========================================
    SPACING = {
        "xs": "4px",  # Extra small
        "sm": "8px",  # Small
        "md": "16px",  # Medium (Base unit)
        "lg": "24px",  # Large
        "xl": "32px",  # Extra large
        "xxl": "48px",  # Extra extra large
        "xxxl": "64px",  # Maximum spacing
    }

    # ==========================================
    # Corner Radius System
    # ==========================================
    CORNER_RADIUS = {
        "small": "8px",
        "medium": "12px",
        "large": "16px",
        "extra_large": "20px",
    }

    # ==========================================
    # Shadow System
    # ==========================================
    SHADOWS = {
        "small": "0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)",
        "medium": "0 3px 6px rgba(0, 0, 0, 0.16), 0 3px 6px rgba(0, 0, 0, 0.23)",
        "large": "0 10px 20px rgba(0, 0, 0, 0.19), 0 6px 6px rgba(0, 0, 0, 0.23)",
        "extra_large": "0 14px 28px rgba(0, 0, 0, 0.25), 0 10px 10px rgba(0, 0, 0, 0.22)",
    }

    @classmethod
    def get_css(cls) -> str:
        """Generate complete iOS HIG CSS system"""
        return f"""
        <style>
        /* ==========================================
           iOS 16+ HIG Design System CSS
           ========================================== */

        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@100;200;300;400;500;600;700;800;900&display=swap');

        /* CSS Custom Properties for Design System */
        :root {{
            /* Colors - Light Mode */
            --system-blue: {cls.COLORS['system_blue']};
            --system-green: {cls.COLORS['system_green']};
            --system-indigo: {cls.COLORS['system_indigo']};
            --system-orange: {cls.COLORS['system_orange']};
            --system-pink: {cls.COLORS['system_pink']};
            --system-purple: {cls.COLORS['system_purple']};
            --system-red: {cls.COLORS['system_red']};
            --system-teal: {cls.COLORS['system_teal']};
            --system-yellow: {cls.COLORS['system_yellow']};

            --system-gray: {cls.COLORS['system_gray']};
            --system-gray2: {cls.COLORS['system_gray2']};
            --system-gray3: {cls.COLORS['system_gray3']};
            --system-gray4: {cls.COLORS['system_gray4']};
            --system-gray5: {cls.COLORS['system_gray5']};
            --system-gray6: {cls.COLORS['system_gray6']};

            --label: {cls.COLORS['label']};
            --secondary-label: {cls.COLORS['secondary_label']};
            --tertiary-label: {cls.COLORS['tertiary_label']};
            --quaternary-label: {cls.COLORS['quaternary_label']};

            --system-background: {cls.COLORS['system_background']};
            --secondary-system-background: {cls.COLORS['secondary_system_background']};
            --tertiary-system-background: {cls.COLORS['tertiary_system_background']};

            --system-grouped-background: {cls.COLORS['system_grouped_background']};
            --secondary-system-grouped-background: {cls.COLORS['secondary_system_grouped_background']};
            --tertiary-system-grouped-background: {cls.COLORS['tertiary_system_grouped_background']};

            --separator: {cls.COLORS['separator']};
            --opaque-separator: {cls.COLORS['opaque_separator']};

            /* Spacing */
            --spacing-xs: {cls.SPACING['xs']};
            --spacing-sm: {cls.SPACING['sm']};
            --spacing-md: {cls.SPACING['md']};
            --spacing-lg: {cls.SPACING['lg']};
            --spacing-xl: {cls.SPACING['xl']};
            --spacing-xxl: {cls.SPACING['xxl']};
            --spacing-xxxl: {cls.SPACING['xxxl']};

            /* Corner Radius */
            --corner-radius-small: {cls.CORNER_RADIUS['small']};
            --corner-radius-medium: {cls.CORNER_RADIUS['medium']};
            --corner-radius-large: {cls.CORNER_RADIUS['large']};
            --corner-radius-extra-large: {cls.CORNER_RADIUS['extra_large']};

            /* Shadows */
            --shadow-small: {cls.SHADOWS['small']};
            --shadow-medium: {cls.SHADOWS['medium']};
            --shadow-large: {cls.SHADOWS['large']};
            --shadow-extra-large: {cls.SHADOWS['extra_large']};
        }}

        /* Dark Mode */
        @media (prefers-color-scheme: dark) {{
            :root {{
                --system-blue: {cls.DARK_COLORS['system_blue']};
                --system-green: {cls.DARK_COLORS['system_green']};
                --system-indigo: {cls.DARK_COLORS['system_indigo']};
                --system-orange: {cls.DARK_COLORS['system_orange']};
                --system-pink: {cls.DARK_COLORS['system_pink']};
                --system-purple: {cls.DARK_COLORS['system_purple']};
                --system-red: {cls.DARK_COLORS['system_red']};
                --system-teal: {cls.DARK_COLORS['system_teal']};
                --system-yellow: {cls.DARK_COLORS['system_yellow']};

                --system-gray: {cls.DARK_COLORS['system_gray']};
                --system-gray2: {cls.DARK_COLORS['system_gray2']};
                --system-gray3: {cls.DARK_COLORS['system_gray3']};
                --system-gray4: {cls.DARK_COLORS['system_gray4']};
                --system-gray5: {cls.DARK_COLORS['system_gray5']};
                --system-gray6: {cls.DARK_COLORS['system_gray6']};

                --label: {cls.DARK_COLORS['label']};
                --secondary-label: {cls.DARK_COLORS['secondary_label']};
                --tertiary-label: {cls.DARK_COLORS['tertiary_label']};
                --quaternary-label: {cls.DARK_COLORS['quaternary_label']};

                --system-background: {cls.DARK_COLORS['system_background']};
                --secondary-system-background: {cls.DARK_COLORS['secondary_system_background']};
                --tertiary-system-background: {cls.DARK_COLORS['tertiary_system_background']};

                --system-grouped-background: {cls.DARK_COLORS['system_grouped_background']};
                --secondary-system-grouped-background: {cls.DARK_COLORS['secondary_system_grouped_background']};
                --tertiary-system-grouped-background: {cls.DARK_COLORS['tertiary_system_grouped_background']};
            }}
        }}

        /* ==========================================
           Base Styles
           ========================================== */

        * {{
            box-sizing: border-box;
        }}

        body, .main, .main > div {{
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--system-grouped-background);
            color: var(--label);
            line-height: 1.47059;
            font-feature-settings: "kern";
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        /* ==========================================
           Typography Classes
           ========================================== */

        .large-title {{
            font-size: {cls.TYPOGRAPHY['large_title']['font_size']};
            line-height: {cls.TYPOGRAPHY['large_title']['line_height']};
            font-weight: {cls.TYPOGRAPHY['large_title']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['large_title']['letter_spacing']};
            color: var(--label);
        }}

        .title1 {{
            font-size: {cls.TYPOGRAPHY['title1']['font_size']};
            line-height: {cls.TYPOGRAPHY['title1']['line_height']};
            font-weight: {cls.TYPOGRAPHY['title1']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['title1']['letter_spacing']};
            color: var(--label);
        }}

        .title2 {{
            font-size: {cls.TYPOGRAPHY['title2']['font_size']};
            line-height: {cls.TYPOGRAPHY['title2']['line_height']};
            font-weight: {cls.TYPOGRAPHY['title2']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['title2']['letter_spacing']};
            color: var(--label);
        }}

        .title3 {{
            font-size: {cls.TYPOGRAPHY['title3']['font_size']};
            line_height: {cls.TYPOGRAPHY['title3']['line_height']};
            font-weight: {cls.TYPOGRAPHY['title3']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['title3']['letter_spacing']};
            color: var(--label);
        }}

        .headline {{
            font-size: {cls.TYPOGRAPHY['headline']['font_size']};
            line-height: {cls.TYPOGRAPHY['headline']['line_height']};
            font-weight: {cls.TYPOGRAPHY['headline']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['headline']['letter_spacing']};
            color: var(--label);
        }}

        .body {{
            font-size: {cls.TYPOGRAPHY['body']['font_size']};
            line-height: {cls.TYPOGRAPHY['body']['line_height']};
            font-weight: {cls.TYPOGRAPHY['body']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['body']['letter_spacing']};
            color: var(--label);
        }}

        .callout {{
            font-size: {cls.TYPOGRAPHY['callout']['font_size']};
            line-height: {cls.TYPOGRAPHY['callout']['line_height']};
            font-weight: {cls.TYPOGRAPHY['callout']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['callout']['letter_spacing']};
            color: var(--label);
        }}

        .subheadline {{
            font-size: {cls.TYPOGRAPHY['subheadline']['font_size']};
            line-height: {cls.TYPOGRAPHY['subheadline']['line_height']};
            font-weight: {cls.TYPOGRAPHY['subheadline']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['subheadline']['letter_spacing']};
            color: var(--secondary-label);
        }}

        .footnote {{
            font-size: {cls.TYPOGRAPHY['footnote']['font_size']};
            line-height: {cls.TYPOGRAPHY['footnote']['line_height']};
            font-weight: {cls.TYPOGRAPHY['footnote']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['footnote']['letter_spacing']};
            color: var(--secondary-label);
        }}

        .caption1 {{
            font-size: {cls.TYPOGRAPHY['caption1']['font_size']};
            line-height: {cls.TYPOGRAPHY['caption1']['line_height']};
            font-weight: {cls.TYPOGRAPHY['caption1']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['caption1']['letter_spacing']};
            color: var(--tertiary-label);
        }}

        .caption2 {{
            font-size: {cls.TYPOGRAPHY['caption2']['font_size']};
            line-height: {cls.TYPOGRAPHY['caption2']['line_height']};
            font-weight: {cls.TYPOGRAPHY['caption2']['font_weight']};
            letter-spacing: {cls.TYPOGRAPHY['caption2']['letter_spacing']};
            color: var(--tertiary-label);
        }}

        /* ==========================================
           Component Styles
           ========================================== */

        /* iOS Cards */
        .ios-card {{
            background-color: var(--secondary-system-grouped-background);
            border-radius: var(--corner-radius-large);
            padding: var(--spacing-lg);
            margin: var(--spacing-md) 0;
            box-shadow: var(--shadow-small);
            border: 0.5px solid var(--separator);
        }}

        .ios-card-inset {{
            background-color: var(--tertiary-system-grouped-background);
            border-radius: var(--corner-radius-medium);
            padding: var(--spacing-md);
            margin: var(--spacing-sm) 0;
        }}

        /* iOS Buttons */
        .ios-button-primary {{
            background-color: var(--system-blue);
            color: white;
            border: none;
            border-radius: var(--corner-radius-medium);
            padding: var(--spacing-md) var(--spacing-lg);
            font-weight: 600;
            font-size: 17px;
            line-height: 22px;
            letter-spacing: -0.43px;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            min-height: 44px; /* iOS minimum touch target */
        }}

        .ios-button-primary:hover {{
            background-color: #0051D5;
            transform: translateY(-1px);
        }}

        .ios-button-primary:active {{
            transform: translateY(0);
            background-color: #003DB8;
        }}

        .ios-button-secondary {{
            background-color: var(--secondary-system-background);
            color: var(--system-blue);
            border: 0.5px solid var(--separator);
            border-radius: var(--corner-radius-medium);
            padding: var(--spacing-md) var(--spacing-lg);
            font-weight: 600;
            font-size: 17px;
            line-height: 22px;
            letter-spacing: -0.43px;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            min-height: 44px;
        }}

        .ios-button-secondary:hover {{
            background-color: var(--system-gray6);
        }}

        /* iOS Lists */
        .ios-list {{
            background-color: var(--secondary-system-grouped-background);
            border-radius: var(--corner-radius-large);
            overflow: hidden;
            margin: var(--spacing-md) 0;
        }}

        .ios-list-item {{
            padding: var(--spacing-md) var(--spacing-lg);
            border-bottom: 0.5px solid var(--separator);
            background-color: var(--secondary-system-grouped-background);
            display: flex;
            align-items: center;
            min-height: 44px;
            cursor: pointer;
            transition: background-color 0.2s ease-in-out;
        }}

        .ios-list-item:last-child {{
            border-bottom: none;
        }}

        .ios-list-item:hover {{
            background-color: var(--tertiary-system-background);
        }}

        .ios-list-item:active {{
            background-color: var(--system-gray5);
        }}

        /* iOS Navigation */
        .ios-nav-bar {{
            background-color: var(--system-background);
            border-bottom: 0.5px solid var(--separator);
            padding: var(--spacing-md) var(--spacing-lg);
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 44px;
        }}

        .ios-nav-title {{
            font-size: 17px;
            font-weight: 600;
            line-height: 22px;
            letter-spacing: -0.43px;
            color: var(--label);
        }}

        /* iOS Progress Indicators */
        .ios-progress-bar {{
            width: 100%;
            height: 4px;
            background-color: var(--system-gray5);
            border-radius: 2px;
            overflow: hidden;
        }}

        .ios-progress-fill {{
            height: 100%;
            background-color: var(--system-blue);
            border-radius: 2px;
            transition: width 0.3s ease-in-out;
        }}

        /* iOS Metrics Cards */
        .ios-metric-card {{
            background-color: var(--secondary-system-grouped-background);
            border-radius: var(--corner-radius-large);
            padding: var(--spacing-lg);
            text-align: center;
            box-shadow: var(--shadow-small);
            border: 0.5px solid var(--separator);
        }}

        .ios-metric-value {{
            font-size: 34px;
            line-height: 41px;
            font-weight: 700;
            letter-spacing: 0.37px;
            color: var(--label);
        }}

        .ios-metric-label {{
            font-size: 13px;
            line-height: 18px;
            font-weight: 400;
            letter-spacing: -0.08px;
            color: var(--secondary-label);
            margin-top: var(--spacing-sm);
        }}

        /* iOS Badges */
        .ios-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 20px;
            height: 20px;
            padding: 0 var(--spacing-xs);
            background-color: var(--system-red);
            color: white;
            border-radius: 10px;
            font-size: 12px;
            font-weight: 600;
            line-height: 16px;
        }}

        /* iOS Separators */
        .ios-separator {{
            height: 0.5px;
            background-color: var(--separator);
            margin: var(--spacing-md) 0;
        }}

        /* iOS Safe Areas and Layout */
        .ios-safe-area {{
            padding-left: max(var(--spacing-lg), env(safe-area-inset-left));
            padding-right: max(var(--spacing-lg), env(safe-area-inset-right));
            padding-top: max(var(--spacing-md), env(safe-area-inset-top));
            padding-bottom: max(var(--spacing-md), env(safe-area-inset-bottom));
        }}

        .ios-section {{
            margin: var(--spacing-xl) 0;
        }}

        .ios-section-header {{
            font-size: 13px;
            line-height: 18px;
            font-weight: 400;
            letter-spacing: -0.08px;
            color: var(--secondary-label);
            text-transform: uppercase;
            margin-bottom: var(--spacing-sm);
            padding: 0 var(--spacing-lg);
        }}

        .ios-section-footer {{
            font-size: 13px;
            line-height: 18px;
            font-weight: 400;
            letter-spacing: -0.08px;
            color: var(--tertiary-label);
            margin-top: var(--spacing-sm);
            padding: 0 var(--spacing-lg);
        }}

        /* Hide Streamlit elements for clean iOS look */
        .css-1d391kg, .css-1y4p8pa, .css-12oz5g7 {{
            padding: 0;
        }}

        header[data-testid="stHeader"] {{
            display: none;
        }}

        .css-1rs6os {{
            background-color: var(--system-grouped-background);
        }}

        /* Override Streamlit button styles */
        .stButton > button {{
            background-color: var(--system-blue) !important;
            color: white !important;
            border: none !important;
            border-radius: var(--corner-radius-medium) !important;
            padding: var(--spacing-md) var(--spacing-lg) !important;
            font-weight: 600 !important;
            font-size: 17px !important;
            line-height: 22px !important;
            letter-spacing: -0.43px !important;
            min-height: 44px !important;
            transition: all 0.2s ease-in-out !important;
        }}

        .stButton > button:hover {{
            background-color: #0051D5 !important;
            transform: translateY(-1px) !important;
        }}

        /* Override Streamlit sidebar */
        .css-1d391kg {{
            background-color: var(--system-background) !important;
            border-right: 0.5px solid var(--separator) !important;
        }}

        /* Override Streamlit metrics */
        .css-1r6slb0 {{
            background-color: var(--secondary-system-grouped-background) !important;
            border: 0.5px solid var(--separator) !important;
            border-radius: var(--corner-radius-large) !important;
            padding: var(--spacing-lg) !important;
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .ios-safe-area {{
                padding-left: var(--spacing-md);
                padding-right: var(--spacing-md);
            }}

            .large-title {{
                font-size: 28px;
                line-height: 34px;
            }}

            .title1 {{
                font-size: 22px;
                line-height: 28px;
            }}
        }}

        /* Animation System */
        @keyframes ios-fade-in {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .ios-animate-in {{
            animation: ios-fade-in 0.5s ease-out;
        }}

        @keyframes ios-bounce {{
            0%, 20%, 53%, 80%, 100% {{
                transform: translate3d(0,0,0);
            }}
            40%, 43% {{
                transform: translate3d(0, -8px, 0);
            }}
            70% {{
                transform: translate3d(0, -4px, 0);
            }}
            90% {{
                transform: translate3d(0, -1px, 0);
            }}
        }}

        .ios-animate-bounce {{
            animation: ios-bounce 1s ease-in-out;
        }}
        </style>
        """

    @classmethod
    def create_navigation_bar(
        cls,
        title: str,
        left_button: Optional[str] = None,
        right_button: Optional[str] = None,
    ) -> str:
        """Create iOS-style navigation bar"""
        left_content = (
            f'<button class="ios-button-secondary">{left_button}</button>'
            if left_button
            else "<div></div>"
        )
        right_content = (
            f'<button class="ios-button-secondary">{right_button}</button>'
            if right_button
            else "<div></div>"
        )

        return f"""
        <div class="ios-nav-bar">
            {left_content}
            <div class="ios-nav-title">{title}</div>
            {right_content}
        </div>
        """

    @classmethod
    def create_card(cls, content: str, title: Optional[str] = None) -> str:
        """Create iOS-style card"""
        title_html = (
            f'<h3 class="headline" style="margin-bottom: var(--spacing-md);">{title}</h3>'
            if title
            else ""
        )
        return f"""
        <div class="ios-card ios-animate-in">
            {title_html}
            {content}
        </div>
        """

    @classmethod
    def create_metric_card(
        cls,
        value: str,
        label: str,
        color: Optional[str] = None,
        delta: Optional[str] = None,
    ) -> str:
        """Create iOS-style metric card"""
        value_color = f"color: {color};" if color else ""
        delta_html = (
            f'<div class="caption1" style="color: var(--system-green); margin-top: var(--spacing-xs);">{delta}</div>'
            if delta
            else ""
        )

        return f"""
        <div class="ios-metric-card">
            <div class="ios-metric-value" style="{value_color}">{value}</div>
            <div class="ios-metric-label">{label}</div>
            {delta_html}
        </div>
        """

    @classmethod
    def create_list_item(
        cls,
        title: str,
        subtitle: Optional[str] = None,
        icon: Optional[str] = None,
        accessory: Optional[str] = None,
    ) -> str:
        """Create iOS-style list item"""
        icon_html = (
            f'<span style="margin-right: var(--spacing-md); font-size: 20px;">{icon}</span>'
            if icon
            else ""
        )
        subtitle_html = f'<div class="subheadline">{subtitle}</div>' if subtitle else ""
        accessory_html = (
            f'<div style="margin-left: auto;">{accessory}</div>' if accessory else ""
        )

        return f"""
        <div class="ios-list-item">
            {icon_html}
            <div style="flex: 1;">
                <div class="body">{title}</div>
                {subtitle_html}
            </div>
            {accessory_html}
        </div>
        """

    @classmethod
    def create_section_header(cls, title: str) -> str:
        """Create iOS-style section header"""
        return f'<div class="ios-section-header">{title}</div>'

    @classmethod
    def create_progress_bar(cls, progress: float, color: Optional[str] = None) -> str:
        """Create iOS-style progress bar"""
        fill_color = color or "var(--system-blue)"
        return f"""
        <div class="ios-progress-bar">
            <div class="ios-progress-fill" style="width: {progress}%; background-color: {fill_color};"></div>
        </div>
        """
