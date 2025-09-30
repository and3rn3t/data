"""
Dashboard Integration for Color Contrast Monitoring
Adds contrast status display to the dashboard settings
"""

import streamlit as st
from typing import Dict, Any
from sandbox.utils.color_contrast_checker import ColorContrastChecker


class DashboardContrastIntegration:
    """Integration class for displaying contrast status in dashboard"""

    @staticmethod
    def render_contrast_status() -> None:
        """Render contrast status in the dashboard settings page"""

        st.markdown(
            """
        <div class="ios-card" style="margin: 30px 0 20px 0;">
            <h3 style="margin-bottom: 20px;">🎨 Color Accessibility Status</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Run contrast check
        with st.spinner("Checking color accessibility..."):
            checker = ColorContrastChecker()
            results = checker.check_dashboard_combinations()

        # Calculate stats
        total = len(results)
        excellent = len([r for r in results if r.passes_aaa])
        good = len([r for r in results if r.passes_aa and not r.passes_aaa])
        poor = len([r for r in results if not r.passes_aa])

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Combinations", total)

        with col2:
            st.metric("Excellent (AAA)", excellent, delta=f"{excellent/total*100:.0f}%")

        with col3:
            st.metric("Good (AA)", good, delta=f"{good/total*100:.0f}%")

        with col4:
            st.metric(
                "Needs Work",
                poor,
                delta=f"{poor/total*100:.0f}%",
                delta_color="inverse",
            )

        # Overall status indicator
        if poor == 0:
            st.success(
                "✅ All color combinations meet WCAG AA accessibility standards!"
            )
        elif poor <= 2:
            st.warning(
                f"⚠️ {poor} color combination(s) need improvement for optimal accessibility."
            )
        else:
            st.error(
                f"❌ {poor} color combinations have poor contrast. Immediate attention needed."
            )

        # Detailed results in expandable sections
        if excellent > 0:
            with st.expander(
                f"🏆 Excellent Contrast ({excellent} combinations)", expanded=False
            ):
                for result in results:
                    if result.passes_aaa:
                        st.markdown(
                            f"- **{result.background}** on **{result.foreground}** - Ratio: {result.ratio:.2f}"
                        )

        if good > 0:
            with st.expander(f"✓ Good Contrast ({good} combinations)", expanded=False):
                for result in results:
                    if result.passes_aa and not result.passes_aaa:
                        st.markdown(
                            f"- **{result.background}** on **{result.foreground}** - Ratio: {result.ratio:.2f}"
                        )

        if poor > 0:
            with st.expander(
                f"⚠️ Needs Improvement ({poor} combinations)", expanded=True
            ):
                for result in results:
                    if not result.passes_aa:
                        st.markdown(
                            f"- **{result.background}** on **{result.foreground}** - Ratio: {result.ratio:.2f}"
                        )
                        st.caption(f"  {result.recommendation}")

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Run Full Contrast Report"):
                st.text_area("Full Report", checker.generate_report(), height=300)

        with col2:
            if st.button("💾 Download Report"):
                report = checker.generate_report()
                st.download_button(
                    label="Download Contrast Report",
                    data=report,
                    file_name=f"contrast_report_{st.session_state.get('timestamp', 'latest')}.txt",
                    mime="text/plain",
                )

    @staticmethod
    def get_contrast_score() -> Dict[str, Any]:
        """Get a simple contrast score for display"""
        try:
            checker = ColorContrastChecker()
            results = checker.check_dashboard_combinations()

            total = len(results)
            passing = len([r for r in results if r.passes_aa])

            return {
                "score": (passing / total) * 100,
                "status": (
                    "excellent"
                    if passing == total
                    else "good" if passing >= total * 0.8 else "poor"
                ),
                "total": total,
                "passing": passing,
            }
        except Exception:
            return {"score": 0, "status": "error", "total": 0, "passing": 0}


def render_contrast_widget() -> None:
    """Render a compact contrast status widget for sidebar"""
    score_data = DashboardContrastIntegration.get_contrast_score()

    if score_data["status"] == "error":
        st.sidebar.error("⚠️ Contrast check failed")
    else:
        score = score_data["score"]
        status = score_data["status"]

        # Status emoji and color
        if status == "excellent":
            emoji = "🟢"
            color = "var(--ios-green)"
        elif status == "good":
            emoji = "🟡"
            color = "var(--ios-orange)"
        else:
            emoji = "🔴"
            color = "var(--ios-red)"

        st.sidebar.markdown(
            f"""
        <div style="display: flex; align-items: center; padding: 8px; background: var(--surface-tertiary); border-radius: 8px; margin: 8px 0;">
            <span style="margin-right: 8px; font-size: 1.2em;">{emoji}</span>
            <div>
                <div style="font-weight: 600; font-size: 0.9em;">Accessibility Score</div>
                <div style="font-size: 0.8em; color: {color};">{score:.0f}% ({score_data['passing']}/{score_data['total']})</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
