"""
Dashboard Layout System for Cohesive UI Design
Provides consistent spacing, sections, and visual hierarchy
"""

import streamlit as st
from typing import Dict, List, Any, Optional


class DashboardLayoutSystem:
    """Centralized layout system for consistent UI design"""

    # Layout Constants
    SECTION_SPACING = "margin: 32px 0 24px 0;"
    CARD_SPACING = "margin: 16px 0;"
    SMALL_SPACING = "margin: 8px 0;"

    # Visual Hierarchy
    HEADER_H1 = "font-size: 2.5rem; font-weight: 700; margin-bottom: 16px;"
    HEADER_H2 = "font-size: 2rem; font-weight: 600; margin-bottom: 12px;"
    HEADER_H3 = "font-size: 1.5rem; font-weight: 600; margin-bottom: 8px;"
    HEADER_H4 = "font-size: 1.25rem; font-weight: 500; margin-bottom: 6px;"

    @staticmethod
    def create_page_header(
        title: str, subtitle: Optional[str] = None, icon: str = "■"
    ) -> None:
        """Create a consistent page header"""
        if subtitle:
            st.markdown(
                f"""
                <div class="ios-card" style="text-align: center; {DashboardLayoutSystem.SECTION_SPACING}">
                    <div style="font-size: 3rem; margin-bottom: 16px;">{icon}</div>
                    <h1 style="{DashboardLayoutSystem.HEADER_H1} color: var(--text-primary); margin-bottom: 8px;">{title}</h1>
                    <p style="font-size: 1.1rem; color: var(--text-secondary); margin: 0;">{subtitle}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; {DashboardLayoutSystem.SECTION_SPACING}">
                    <span style="font-size: 2.5rem; margin-right: 16px;">{icon}</span>
                    <h1 style="{DashboardLayoutSystem.HEADER_H1} color: var(--text-primary); margin: 0;">{title}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

    @staticmethod
    def create_section_header(title: str, icon: str = "•") -> None:
        """Create a consistent section header"""
        st.markdown(
            f"""
            <div style="{DashboardLayoutSystem.SECTION_SPACING}">
                <h2 style="{DashboardLayoutSystem.HEADER_H2} color: var(--text-primary); background: transparent;">
                    {title}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def create_card_section(
        title: str, content: str, icon: str = "", style: str = ""
    ) -> None:
        """Create a consistent card section"""
        header = (
            f'<h3 style="{DashboardLayoutSystem.HEADER_H3} margin-bottom: 16px;">{icon + " " if icon else ""}{title}</h3>'
            if title
            else ""
        )

        st.markdown(
            f"""
            <div class="ios-card" style="{DashboardLayoutSystem.CARD_SPACING} {style}
                 position: relative; z-index: 2; overflow: hidden;
                 background: var(--surface-primary);
                 backdrop-filter: blur(20px);">
                {header}
                <div style="position: relative; z-index: 3;">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def create_metric_cards(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
        """Create consistent metric cards in a grid layout"""
        cols = st.columns(columns)

        for i, metric in enumerate(metrics):
            with cols[i % columns]:
                icon = metric.get("icon", "■")
                value = metric.get("value", "0")
                label = metric.get("label", "Metric")
                sublabel = metric.get("sublabel", "")
                color = metric.get("color", "var(--ios-blue)")
                delta = metric.get("delta", "")

                delta_html = (
                    f'<div style="font-size: 0.8rem; color: {metric.get("delta_color", "var(--text-secondary)")}; margin-top: 4px;">{delta}</div>'
                    if delta
                    else ""
                )

                st.markdown(
                    f"""
                    <div class="ios-card" style="text-align: center; {DashboardLayoutSystem.CARD_SPACING}">
                        <div style="font-size: 2.5rem; margin-bottom: 12px;">{icon}</div>
                        <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 4px;">{value}</div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 2px;">{label}</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">{sublabel}</div>
                        {delta_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    @staticmethod
    def create_two_column_layout(
        left_content: callable, right_content: callable, ratio: List[int] = [2, 1]
    ) -> None:
        """Create a consistent two-column layout"""
        col1, col2 = st.columns(ratio, gap="large")

        with col1:
            st.markdown(
                '<div style="position: relative; z-index: 1; margin-right: 16px;">',
                unsafe_allow_html=True,
            )
            left_content()
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
                '<div style="position: relative; z-index: 1; margin-left: 16px;">',
                unsafe_allow_html=True,
            )
            right_content()
            st.markdown("</div>", unsafe_allow_html=True)

    @staticmethod
    def create_tabs(tab_data: List[Dict[str, Any]]) -> None:
        """Create consistent tabbed content"""
        tab_labels = [tab["label"] for tab in tab_data]
        tabs = st.tabs(tab_labels)

        for i, (tab, tab_info) in enumerate(zip(tabs, tab_data)):
            with tab:
                if "icon" in tab_info:
                    st.markdown(
                        f"<div style='font-size: 1.5rem; margin-bottom: 16px;'>{tab_info['icon']}</div>",
                        unsafe_allow_html=True,
                    )

                if callable(tab_info["content"]):
                    tab_info["content"]()
                else:
                    st.markdown(tab_info["content"])

    @staticmethod
    def create_progress_section(items: List[Dict[str, Any]]) -> None:
        """Create a consistent progress display section"""
        for item in items:
            progress = item.get("progress", 0)
            title = item.get("title", "Item")
            subtitle = item.get("subtitle", "")
            color = item.get("color", "var(--ios-blue)")

            st.markdown(
                f"""
                <div class="ios-card" style="{DashboardLayoutSystem.CARD_SPACING}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <div style="font-weight: 600; color: var(--text-primary);">{title}</div>
                            <div style="font-size: 0.9rem; color: var(--text-secondary);">{subtitle}</div>
                        </div>
                        <div style="font-weight: 700; color: {color};">{progress}%</div>
                    </div>
                    <div style="width: 100%; height: 6px; background: var(--surface-tertiary); border-radius: 3px; overflow: hidden;">
                        <div style="width: {progress}%; height: 100%; background: {color}; border-radius: 3px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    @staticmethod
    def create_action_buttons(buttons: List[Dict[str, Any]], columns: int = 3) -> None:
        """Create consistent action button grid"""
        cols = st.columns(columns)

        for i, button in enumerate(buttons):
            with cols[i % columns]:
                icon = button.get("icon", "🔘")
                title = button.get("title", "Action")
                description = button.get("description", "")
                color = button.get("color", "var(--ios-blue)")
                key = button.get("key", f"btn_{i}")
                callback = button.get("callback")

                st.markdown(
                    f"""
                    <div class="ios-card" style="text-align: center; background: {color}; color: white; {DashboardLayoutSystem.CARD_SPACING}">
                        <div style="font-size: 2.5rem; margin-bottom: 12px;">{icon}</div>
                        <h4 style="margin-bottom: 8px; font-weight: 600;">{title}</h4>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(f"{icon} {title}", key=key, help=description):
                    if callback:
                        callback()

    @staticmethod
    def create_status_indicator(
        status: str, message: str, details: Optional[str] = None
    ) -> None:
        """Create consistent status indicators"""
        status_config = {
            "success": {
                "color": "var(--ios-green)",
                "icon": "✅",
                "bg": "rgba(4, 120, 87, 0.1)",
            },
            "warning": {
                "color": "var(--ios-orange)",
                "icon": "⚠️",
                "bg": "rgba(194, 65, 12, 0.1)",
            },
            "error": {
                "color": "var(--ios-red)",
                "icon": "❌",
                "bg": "rgba(220, 38, 38, 0.1)",
            },
            "info": {
                "color": "var(--ios-blue)",
                "icon": "ℹ️",
                "bg": "rgba(29, 78, 216, 0.1)",
            },
        }

        config = status_config.get(status, status_config["info"])
        details_html = (
            f'<div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 8px;">{details}</div>'
            if details
            else ""
        )

        st.markdown(
            f"""
            <div class="ios-card" style="background: {config['bg']}; border-left: 4px solid {config['color']}; {DashboardLayoutSystem.CARD_SPACING}">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.5rem; margin-right: 12px;">{config['icon']}</span>
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: {config['color']};">{message}</div>
                        {details_html}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def add_spacing(type: str = "section") -> None:
        """Add consistent spacing"""
        spacing_map = {
            "section": DashboardLayoutSystem.SECTION_SPACING,
            "card": DashboardLayoutSystem.CARD_SPACING,
            "small": DashboardLayoutSystem.SMALL_SPACING,
        }

        spacing = spacing_map.get(type, DashboardLayoutSystem.CARD_SPACING)
        st.markdown(f"<div style='{spacing}'></div>", unsafe_allow_html=True)
