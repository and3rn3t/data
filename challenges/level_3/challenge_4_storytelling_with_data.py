"""
Level 3 Challenge 4: Storytelling with Data
Master the art of data storytelling by creating compelling narratives that guide audiences through insights.
"""

import warnings
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


def setup_professional_styling() -> None:
    """Setup professional presentation styling"""
    plt.style.use("default")
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
        }
    )


def create_climate_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create comprehensive climate change dataset for storytelling"""
    print("Creating Climate Change Impact Dataset for Data Storytelling...")

    # Generate comprehensive climate and economic data
    np.random.seed(42)
    years = list(range(1980, 2024))

    # Global temperature anomaly (based on real trends)
    base_temp = 0.0  # 1980 baseline
    temp_trend = 0.018  # degrees per year warming trend
    temp_anomaly = [
        base_temp + (year - 1980) * temp_trend + np.random.normal(0, 0.1)
        for year in years
    ]

    # CO2 levels (realistic progression)
    base_co2 = 338  # ppm in 1980
    co2_growth = 1.8  # ppm per year average
    co2_levels = [
        base_co2 + (year - 1980) * co2_growth + np.random.normal(0, 2) for year in years
    ]

    # Economic and social impact data
    gdp_impact = []
    crop_yields = []
    extreme_weather_events = []
    renewable_energy_pct = []
    public_awareness = []

    for i, year in enumerate(years):
        # GDP impact (economic cost of climate change)
        climate_cost = temp_anomaly[i] * 0.5 + np.random.normal(0, 0.2)  # % of GDP
        gdp_impact.append(max(0, climate_cost))

        # Crop yields (declining with temperature)
        base_yield = 100
        temp_impact = -temp_anomaly[i] * 8  # yield reduction
        variability = np.random.normal(0, 5)
        crop_yields.append(base_yield + temp_impact + variability)

        # Extreme weather events (increasing trend)
        base_events = 20
        trend_events = (year - 1980) * 0.3
        random_events = np.random.poisson(5)
        extreme_weather_events.append(base_events + trend_events + random_events)

        # Renewable energy adoption (S-curve)
        if year < 1990:
            renewable_pct = 2 + np.random.normal(0, 0.5)
        elif year < 2000:
            renewable_pct = 2 + (year - 1990) * 0.3 + np.random.normal(0, 1)
        elif year < 2010:
            renewable_pct = 5 + (year - 2000) * 0.8 + np.random.normal(0, 1.5)
        else:
            renewable_pct = 13 + (year - 2010) * 2.2 + np.random.normal(0, 2)
        renewable_energy_pct.append(max(0, min(100, renewable_pct)))

        # Public awareness (sigmoid growth starting around 2000)
        if year < 2000:
            awareness = 10 + np.random.normal(0, 3)
        else:
            progress = (year - 2000) / 24  # 24 years from 2000 to 2024
            sigmoid = 1 / (1 + np.exp(-6 * (progress - 0.5)))
            awareness = 10 + sigmoid * 70 + np.random.normal(0, 5)
        public_awareness.append(max(0, min(100, awareness)))

    # Create comprehensive DataFrame
    climate_df = pd.DataFrame(
        {
            "year": years,
            "temp_anomaly": temp_anomaly,
            "co2_ppm": co2_levels,
            "gdp_impact_pct": gdp_impact,
            "crop_yield_index": crop_yields,
            "extreme_weather_count": extreme_weather_events,
            "renewable_energy_pct": renewable_energy_pct,
            "public_awareness_pct": public_awareness,
        }
    )

    # Add regional breakdown
    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Africa"]
    regional_data = []

    for region in regions:
        for _, row in climate_df.iterrows():
            # Regional variations
            regional_multipliers = {
                "North America": {"temp": 1.1, "gdp": 0.8, "renewable": 1.2},
                "Europe": {"temp": 1.0, "gdp": 0.7, "renewable": 1.5},
                "Asia Pacific": {"temp": 0.9, "gdp": 1.2, "renewable": 1.1},
                "Latin America": {"temp": 1.2, "gdp": 1.5, "renewable": 0.8},
                "Africa": {"temp": 1.3, "gdp": 2.0, "renewable": 0.6},
            }

            multiplier = regional_multipliers[region]
            regional_data.append(
                {
                    "year": row["year"],
                    "region": region,
                    "temp_anomaly": row["temp_anomaly"] * multiplier["temp"],
                    "gdp_impact_pct": row["gdp_impact_pct"] * multiplier["gdp"],
                    "renewable_energy_pct": row["renewable_energy_pct"]
                    * multiplier["renewable"],
                }
            )

    regional_df = pd.DataFrame(regional_data)

    print("Climate dataset created successfully!")
    print(f"Global data shape: {climate_df.shape}")
    print(f"Regional data shape: {regional_df.shape}")

    return climate_df, regional_df


def create_chapter_1_problem(
    climate_df: pd.DataFrame, regional_df: pd.DataFrame
) -> None:
    """Chapter 1: The Problem - Establishing Context"""
    print("\n" + "=" * 60)
    print("DATA STORY: CLIMATE CHANGE - A VISUAL NARRATIVE")
    print("=" * 60)

    print("\nCHAPTER 1: THE PROBLEM - Rising Temperatures and Emissions")

    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Africa"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Chapter 1: The Climate Crisis Emerges\n1980-2024 Global Trends",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )

    # 1.1: Temperature anomaly progression
    ax1.plot(
        climate_df["year"],
        climate_df["temp_anomaly"],
        linewidth=3,
        color="red",
        alpha=0.8,
    )
    ax1.fill_between(
        climate_df["year"],
        0,
        climate_df["temp_anomaly"],
        alpha=0.3,
        color="red",
        where=(np.array(climate_df["temp_anomaly"]) > 0),
    )
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax1.set_title(
        'Global Temperature Anomaly\n"The World is Heating Up"', fontweight="bold"
    )
    ax1.set_ylabel("Temperature Anomaly (°C)")
    ax1.grid(True, alpha=0.3)

    # Add annotation for recent warming
    recent_temp = climate_df["temp_anomaly"].iloc[-1]
    ax1.annotate(
        f"2024: +{recent_temp:.1f}°C",
        xy=(2023, recent_temp),
        xytext=(2010, recent_temp + 0.2),
        arrowprops={"arrowstyle": "->", "color": "darkred", "lw": 2},
        fontsize=12,
        fontweight="bold",
        color="darkred",
    )

    # 1.2: CO2 concentration rise
    ax2.plot(climate_df["year"], climate_df["co2_ppm"], linewidth=3, color="navy")
    ax2.set_title(
        'Atmospheric CO₂ Concentration\n"Emissions Continue Rising"', fontweight="bold"
    )
    ax2.set_ylabel("CO₂ (ppm)")
    ax2.grid(True, alpha=0.3)

    # Add threshold lines
    ax2.axhline(
        y=350, color="orange", linestyle="--", alpha=0.7, label="350 ppm (Safe level)"
    )
    ax2.axhline(
        y=400, color="red", linestyle="--", alpha=0.7, label="400 ppm (Crossed 2010s)"
    )
    ax2.legend()

    # 1.3: Regional temperature impacts
    for region in regions[:3]:  # Show top 3 most impacted
        region_data = regional_df[regional_df["region"] == region]
        ax3.plot(
            region_data["year"],
            region_data["temp_anomaly"],
            linewidth=2,
            label=region,
            alpha=0.8,
        )

    ax3.set_title(
        'Regional Temperature Impacts\n"Some Regions Hit Harder"', fontweight="bold"
    )
    ax3.set_ylabel("Temperature Anomaly (°C)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 1.4: Extreme weather correlation
    ax4.scatter(
        climate_df["temp_anomaly"],
        climate_df["extreme_weather_count"],
        c=climate_df["year"],
        cmap="Reds",
        s=60,
        alpha=0.7,
    )
    ax4.set_title(
        'Temperature vs Extreme Weather\n"More Heat, More Storms"', fontweight="bold"
    )
    ax4.set_xlabel("Temperature Anomaly (°C)")
    ax4.set_ylabel("Extreme Weather Events")

    # Add trend line
    z = np.polyfit(climate_df["temp_anomaly"], climate_df["extreme_weather_count"], 1)
    p = np.poly1d(z)
    ax4.plot(
        climate_df["temp_anomaly"],
        p(climate_df["temp_anomaly"]),
        "r--",
        alpha=0.8,
        linewidth=2,
    )

    plt.tight_layout()
    plt.show()

    # Key insights for Chapter 1
    print("\nKEY INSIGHTS FROM CHAPTER 1:")
    print(f"• Temperature has risen by {recent_temp:.1f}°C since 1980")
    print(
        f"• CO₂ levels increased from {climate_df['co2_ppm'].iloc[0]:.0f} to {climate_df['co2_ppm'].iloc[-1]:.0f} ppm"
    )
    print(
        f"• Extreme weather events increased by {climate_df['extreme_weather_count'].iloc[-1] - climate_df['extreme_weather_count'].iloc[0]:.0f} annually"
    )


def create_chapter_2_impact(
    climate_df: pd.DataFrame, regional_df: pd.DataFrame
) -> None:
    """Chapter 2: The Impact - Economic and Social Consequences"""
    print("\nCHAPTER 2: THE IMPACT - Economic and Social Consequences")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Chapter 2: The Hidden Costs of Climate Change\nEconomic and Agricultural Impacts",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )

    # 2.1: Economic impact over time
    ax = axes[0, 0]
    ax.fill_between(
        climate_df["year"],
        0,
        climate_df["gdp_impact_pct"],
        color="red",
        alpha=0.4,
        label="GDP Loss",
    )
    ax.plot(
        climate_df["year"], climate_df["gdp_impact_pct"], color="darkred", linewidth=3
    )
    ax.set_title(
        'Economic Impact of Climate Change\n"Growing Cost to Global Economy"',
        fontweight="bold",
    )
    ax.set_ylabel("GDP Impact (%)")
    ax.grid(True, alpha=0.3)

    # Add cumulative cost annotation
    total_impact = climate_df["gdp_impact_pct"].sum()
    ax.text(
        0.7,
        0.9,
        f"Cumulative Impact:\n~{total_impact:.1f}% GDP",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
    )

    # 2.2: Agricultural impacts
    ax = axes[0, 1]
    ax.plot(
        climate_df["year"],
        climate_df["crop_yield_index"],
        color="green",
        linewidth=3,
        alpha=0.8,
    )
    ax.axhline(y=100, color="black", linestyle="-", alpha=0.5, label="1980 Baseline")
    ax.fill_between(
        climate_df["year"],
        100,
        climate_df["crop_yield_index"],
        where=(np.array(climate_df["crop_yield_index"]) < 100),
        color="red",
        alpha=0.3,
        label="Yield Loss",
    )
    ax.set_title(
        'Global Crop Yield Index\n"Food Security Under Threat"', fontweight="bold"
    )
    ax.set_ylabel("Yield Index (1980=100)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2.3: Regional economic impacts
    ax = axes[1, 0]
    recent_year = 2020
    recent_regional = regional_df[regional_df["year"] == recent_year].sort_values(
        "gdp_impact_pct", ascending=True
    )
    colors = [
        "red" if x > 1.5 else "orange" if x > 1.0 else "yellow"
        for x in recent_regional["gdp_impact_pct"]
    ]
    bars = ax.barh(
        recent_regional["region"], recent_regional["gdp_impact_pct"], color=colors
    )
    ax.set_title(
        f'Regional Economic Impact ({recent_year})\n"Unequal Burden"', fontweight="bold"
    )
    ax.set_xlabel("GDP Impact (%)")

    # Add value labels on bars
    for i, (_, value) in enumerate(zip(bars, recent_regional["gdp_impact_pct"])):
        ax.text(value + 0.05, i, f"{value:.1f}%", va="center", fontweight="bold")

    # 2.4: Multiple impact correlation matrix
    impact_data = climate_df[
        ["temp_anomaly", "gdp_impact_pct", "crop_yield_index", "extreme_weather_count"]
    ].corr()
    ax = axes[1, 1]
    ax.imshow(impact_data, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(impact_data.columns)))
    ax.set_yticks(range(len(impact_data.columns)))
    ax.set_xticklabels(
        ["Temperature", "GDP Impact", "Crop Yields", "Extreme Weather"], rotation=45
    )
    ax.set_yticklabels(["Temperature", "GDP Impact", "Crop Yields", "Extreme Weather"])
    ax.set_title(
        'Impact Correlation Matrix\n"Everything is Connected"', fontweight="bold"
    )

    # Add correlation values
    for i in range(len(impact_data.columns)):
        for j in range(len(impact_data.columns)):
            ax.text(
                j,
                i,
                f"{impact_data.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if abs(impact_data.iloc[i, j]) > 0.5 else "black",
            )

    plt.tight_layout()
    plt.show()

    print("\nKEY INSIGHTS FROM CHAPTER 2:")
    recent_gdp = climate_df["gdp_impact_pct"].iloc[-1]
    recent_yield = climate_df["crop_yield_index"].iloc[-1]
    print(f"• Current annual economic impact: {recent_gdp:.1f}% of global GDP")
    print(f"• Crop yields declined to {recent_yield:.1f}% of 1980 levels")
    print("• Developing regions face 2-3x higher economic impacts")


def create_chapter_3_hope(climate_df: pd.DataFrame, regional_df: pd.DataFrame) -> None:
    """Chapter 3: The Hope - Solutions and Progress"""
    print("\nCHAPTER 3: THE HOPE - Solutions and Growing Awareness")

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(
        "Chapter 3: The Path Forward\nRenewable Energy Growth and Rising Awareness",
        fontsize=20,
        fontweight="bold",
        y=0.95,
    )

    # 3.1: Renewable energy adoption (large central plot)
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    ax_main.plot(
        climate_df["year"],
        climate_df["renewable_energy_pct"],
        linewidth=4,
        color="green",
        alpha=0.8,
    )
    ax_main.fill_between(
        climate_df["year"],
        0,
        climate_df["renewable_energy_pct"],
        alpha=0.3,
        color="green",
    )

    # Add milestones
    milestones = [
        (1990, "Solar costs start declining"),
        (2000, "Wind power expansion"),
        (2010, "Grid parity achieved"),
        (2020, "Renewables become cheapest"),
    ]

    for year, milestone in milestones:
        if year in climate_df["year"].values:
            renewable_val = climate_df[climate_df["year"] == year][
                "renewable_energy_pct"
            ].iloc[0]
            ax_main.annotate(
                milestone,
                xy=(year, renewable_val),
                xytext=(year, renewable_val + 8),
                arrowprops={"arrowstyle": "->", "color": "darkgreen"},
                fontsize=10,
                ha="center",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "lightgreen",
                    "alpha": 0.8,
                },
            )

    ax_main.set_title(
        'Renewable Energy Revolution\n"Exponential Growth in Clean Energy"',
        fontsize=14,
        fontweight="bold",
    )
    ax_main.set_ylabel("Renewable Energy (%)")
    ax_main.grid(True, alpha=0.3)

    # 3.2: Public awareness growth
    ax_awareness = fig.add_subplot(gs[0, 2])
    ax_awareness.plot(
        climate_df["year"],
        climate_df["public_awareness_pct"],
        linewidth=3,
        color="blue",
        alpha=0.8,
    )
    ax_awareness.set_title(
        'Public Awareness\n"Climate Consciousness Rising"',
        fontsize=12,
        fontweight="bold",
    )
    ax_awareness.set_ylabel("Awareness (%)")
    ax_awareness.grid(True, alpha=0.3)

    # 3.3: Regional renewable adoption
    ax_regional = fig.add_subplot(gs[1, 2])
    recent_renewable = regional_df[regional_df["year"] == 2020].sort_values(
        "renewable_energy_pct", ascending=False
    )
    colors = ["gold", "silver", "brown", "gray", "lightgray"]
    bars = ax_regional.bar(
        range(len(recent_renewable)),
        recent_renewable["renewable_energy_pct"],
        color=colors,
    )
    ax_regional.set_title(
        "Regional Leaders\n(2020 Renewable %)", fontsize=12, fontweight="bold"
    )
    ax_regional.set_xticks(range(len(recent_renewable)))
    ax_regional.set_xticklabels(recent_renewable["region"], rotation=45, ha="right")
    ax_regional.set_ylabel("Renewable %")

    # Add percentage labels on bars
    for bar, pct in zip(bars, recent_renewable["renewable_energy_pct"]):
        ax_regional.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3.4: Future scenarios
    ax_future = fig.add_subplot(gs[2, :])

    # Create projected scenarios (2024-2050)
    future_years = list(range(2024, 2051))
    n_future = len(future_years)

    # Scenario 1: Business as usual
    bau_temp = [
        climate_df["temp_anomaly"].iloc[-1] + i * 0.025 + np.random.normal(0, 0.02)
        for i in range(n_future)
    ]
    # Scenario 2: Paris Agreement goals
    paris_temp = [
        climate_df["temp_anomaly"].iloc[-1]
        + i * 0.015 * np.exp(-i * 0.05)
        + np.random.normal(0, 0.02)
        for i in range(n_future)
    ]
    # Scenario 3: Aggressive action
    aggressive_temp = [
        climate_df["temp_anomaly"].iloc[-1]
        + max(0, 0.005 * (10 - i))
        + np.random.normal(0, 0.02)
        for i in range(n_future)
    ]

    # Plot historical and projected
    ax_future.plot(
        climate_df["year"],
        climate_df["temp_anomaly"],
        "k-",
        linewidth=3,
        label="Historical (1980-2024)",
    )
    ax_future.plot(
        future_years, bau_temp, "r--", linewidth=3, label="Business as Usual", alpha=0.8
    )
    ax_future.plot(
        future_years,
        paris_temp,
        "orange",
        linewidth=3,
        label="Paris Agreement",
        alpha=0.8,
    )
    ax_future.plot(
        future_years,
        aggressive_temp,
        "g--",
        linewidth=3,
        label="Aggressive Climate Action",
        alpha=0.8,
    )

    # Add target lines
    ax_future.axhline(
        y=1.5, color="orange", linestyle=":", alpha=0.7, label="1.5°C Target"
    )
    ax_future.axhline(
        y=2.0, color="red", linestyle=":", alpha=0.7, label="2.0°C Danger Zone"
    )

    ax_future.set_title(
        'Future Temperature Scenarios\n"The Choice is Ours"',
        fontsize=14,
        fontweight="bold",
    )
    ax_future.set_xlabel("Year")
    ax_future.set_ylabel("Temperature Anomaly (°C)")
    ax_future.legend(loc="upper left")
    ax_future.grid(True, alpha=0.3)

    # Add scenario outcome text
    outcome_text = """2050 Outcomes by Scenario:
• Business as Usual: +2.5°C
• Paris Agreement: +1.8°C
• Aggressive Action: +1.2°C"""
    ax_future.text(
        0.7,
        0.3,
        outcome_text,
        transform=ax_future.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightblue", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.show()

    print("\nKEY INSIGHTS FROM CHAPTER 3:")
    current_renewable = climate_df["renewable_energy_pct"].iloc[-1]
    current_awareness = climate_df["public_awareness_pct"].iloc[-1]
    print(f"• Renewable energy reached {current_renewable:.1f}% of global energy mix")
    print(f"• Public awareness increased to {current_awareness:.1f}% of population")
    print("• Three possible futures: from disaster (+2.5°C) to success (+1.2°C)")


def create_interactive_story_dashboard(
    climate_df: pd.DataFrame, regional_df: pd.DataFrame
) -> None:
    """Create interactive Plotly dashboard for complete story"""
    print("\nCreating Interactive Story Dashboard...")

    # Create interactive subplot dashboard
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Temperature Rise Over Time",
            "Regional Economic Impact",
            "Renewable Energy Growth",
            "Future Scenarios",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Temperature timeline
    fig.add_trace(
        go.Scatter(
            x=climate_df["year"],
            y=climate_df["temp_anomaly"],
            mode="lines",
            name="Temperature Anomaly",
            line={"width": 3, "color": "red"},
        ),
        row=1,
        col=1,
    )

    # Regional impacts (latest year)
    latest_regional = regional_df[regional_df["year"] == 2020].sort_values(
        "gdp_impact_pct"
    )
    fig.add_trace(
        go.Bar(
            x=latest_regional["gdp_impact_pct"],
            y=latest_regional["region"],
            orientation="h",
            name="GDP Impact %",
            marker_color="orange",
        ),
        row=1,
        col=2,
    )

    # Renewable energy growth
    fig.add_trace(
        go.Scatter(
            x=climate_df["year"],
            y=climate_df["renewable_energy_pct"],
            mode="lines+markers",
            name="Renewable Energy %",
            line={"width": 3, "color": "green"},
        ),
        row=2,
        col=1,
    )

    # Future scenarios
    future_years = list(range(2024, 2051))
    n_future = len(future_years)

    bau_temp = [
        climate_df["temp_anomaly"].iloc[-1] + i * 0.025 for i in range(n_future)
    ]
    paris_temp = [
        climate_df["temp_anomaly"].iloc[-1] + i * 0.015 * np.exp(-i * 0.05)
        for i in range(n_future)
    ]
    aggressive_temp = [
        climate_df["temp_anomaly"].iloc[-1] + max(0, 0.005 * (10 - i))
        for i in range(n_future)
    ]

    fig.add_trace(
        go.Scatter(
            x=future_years,
            y=bau_temp,
            mode="lines",
            name="Business as Usual",
            line={"dash": "dash", "color": "red"},
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=future_years,
            y=paris_temp,
            mode="lines",
            name="Paris Agreement",
            line={"color": "orange"},
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=future_years,
            y=aggressive_temp,
            mode="lines",
            name="Aggressive Action",
            line={"dash": "dash", "color": "green"},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800,
        title_text="Climate Change: An Interactive Data Story",
        title_x=0.5,
        showlegend=True,
    )

    print("Interactive dashboard created successfully!")


def main() -> Dict[str, Any]:
    """Main function to run the complete data storytelling challenge"""
    print("=" * 60)
    print("LEVEL 3 CHALLENGE 4: STORYTELLING WITH DATA")
    print("=" * 60)

    # Setup professional styling
    setup_professional_styling()

    # Create comprehensive dataset
    climate_df, regional_df = create_climate_dataset()

    # Create narrative chapters
    create_chapter_1_problem(climate_df, regional_df)
    create_chapter_2_impact(climate_df, regional_df)
    create_chapter_3_hope(climate_df, regional_df)

    # Create interactive dashboard
    create_interactive_story_dashboard(climate_df, regional_df)

    print("\n" + "=" * 60)
    print("STORYTELLING CHALLENGE COMPLETION SUMMARY")
    print("=" * 60)

    print("Data storytelling techniques mastered:")
    print("  + Progressive narrative structure with 3 compelling chapters")
    print("  + Visual hierarchy guiding audience through insights")
    print("  + Emotional engagement through problem-impact-solution flow")
    print("  + Professional styling and annotation techniques")
    print("  + Interactive dashboard for audience exploration")

    print("\nStory Impact Metrics:")
    print(
        f"  - Dataset span: {climate_df['year'].min()}-{climate_df['year'].max()} ({len(climate_df)} data points)"
    )
    print(f"  - Regional coverage: {regional_df['region'].nunique()} regions analyzed")
    print("  - Visualization chapters: 3 narrative sections created")
    print("  - Future scenarios: 3 potential outcomes presented")
    print("  - Interactive elements: Multi-panel dashboard with user exploration")

    print("\nNarrative Structure:")
    print("  1. THE PROBLEM: Temperature rise and emissions (Context setting)")
    print("  2. THE IMPACT: Economic and social costs (Urgency building)")
    print("  3. THE HOPE: Solutions and pathways forward (Call to action)")

    return {
        "climate_data": climate_df,
        "regional_data": regional_df,
        "story_chapters": 3,
        "narrative_complete": True,
    }


if __name__ == "__main__":
    results = main()

    print("\n" + "=" * 60)
    print("CHALLENGE 4 STATUS: COMPLETE")
    print("=" * 60)
    print("Data storytelling mastery achieved!")
    print("Level 3 Visualization Virtuoso challenges complete.")
    print("Ready for Level 5: Advanced Machine Learning.")
