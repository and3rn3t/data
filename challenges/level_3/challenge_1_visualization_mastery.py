"""
Level 3 Challenge 1: Master Data Visualization
Master advanced data visualization techniques using matplotlib, seaborn, and plotly.
"""

import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


def setup_visualization_environment():
    """Setup advanced visualization environment"""
    plt.style.use("default")
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    print("Visualization environment configured successfully!")


def create_comprehensive_dataset():
    """Create comprehensive sales dataset for visualization"""
    print("Creating comprehensive sales dataset for visualization...")

    np.random.seed(42)
    n_samples = 1000

    # Generate dates for trend analysis
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")

    # Simulate company sales data with realistic patterns
    data = pd.DataFrame(
        {
            "date": dates,
            "region": np.random.choice(
                ["North", "South", "East", "West"], n_samples, p=[0.3, 0.25, 0.25, 0.2]
            ),
            "product_category": np.random.choice(
                ["Electronics", "Clothing", "Home", "Sports", "Books"], n_samples
            ),
            "sales_amount": np.random.gamma(2, 500, n_samples),
            "profit_margin": np.random.normal(0.15, 0.05, n_samples),
            "customer_satisfaction": np.random.normal(4.2, 0.8, n_samples),
            "marketing_spend": np.random.exponential(100, n_samples),
        }
    )

    # Add season mapping
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
    }
    data["season"] = data["date"].dt.month.map(season_map)

    # Calculate derived metrics
    data["profit_amount"] = data["sales_amount"] * data["profit_margin"]
    data["roi"] = data["profit_amount"] / data["marketing_spend"]
    data["month"] = data["date"].dt.month_name()
    data["quarter"] = data["date"].dt.quarter

    # Clip satisfaction to realistic range
    data["customer_satisfaction"] = np.clip(data["customer_satisfaction"], 1, 5)

    print(f"Dataset created: {data.shape[0]} records with {data.shape[1]} features")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    return data


def create_statistical_visualizations(data):
    """Create advanced statistical visualizations"""
    print("\\nCreating statistical visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Statistical Analysis Dashboard", fontsize=16, fontweight="bold")

    # 1. Distribution analysis with KDE
    ax = axes[0, 0]
    sns.histplot(data=data, x="sales_amount", hue="region", kde=True, ax=ax, alpha=0.7)
    ax.set_title("Sales Distribution by Region")
    ax.set_xlabel("Sales Amount ($)")
    ax.set_ylabel("Frequency")

    # 2. Box plots for category comparison
    ax = axes[0, 1]
    sns.boxplot(data=data, x="product_category", y="profit_margin", ax=ax)
    ax.set_title("Profit Margin by Product Category")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Profit Margin")
    ax.tick_params(axis="x", rotation=45)

    # 3. Correlation heatmap
    ax = axes[1, 0]
    numeric_cols = [
        "sales_amount",
        "profit_margin",
        "customer_satisfaction",
        "marketing_spend",
    ]
    corr_matrix = data[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")

    # 4. Scatter plot with trend line
    ax = axes[1, 1]
    sns.scatterplot(
        data=data,
        x="marketing_spend",
        y="sales_amount",
        hue="season",
        size="customer_satisfaction",
        ax=ax,
        alpha=0.6,
    )
    ax.set_title("Marketing Spend vs Sales (by Season)")
    ax.set_xlabel("Marketing Spend ($)")
    ax.set_ylabel("Sales Amount ($)")

    plt.tight_layout()
    plt.show()

    return fig


def create_time_series_analysis(data):
    """Create time series visualizations"""
    print("Creating time series analysis...")

    # Aggregate data by date for cleaner trends
    daily_sales = (
        data.groupby("date")
        .agg(
            {
                "sales_amount": "sum",
                "profit_amount": "sum",
                "customer_satisfaction": "mean",
            }
        )
        .reset_index()
    )

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle("Time Series Analysis Dashboard", fontsize=16, fontweight="bold")

    # 1. Sales trend over time
    axes[0].plot(
        daily_sales["date"], daily_sales["sales_amount"], linewidth=2, color="steelblue"
    )
    axes[0].fill_between(
        daily_sales["date"], daily_sales["sales_amount"], alpha=0.3, color="steelblue"
    )
    axes[0].set_title("Daily Sales Trend")
    axes[0].set_ylabel("Sales Amount ($)")
    axes[0].grid(True, alpha=0.3)

    # 2. Profit trend with moving average
    axes[1].plot(
        daily_sales["date"],
        daily_sales["profit_amount"],
        linewidth=1,
        alpha=0.7,
        label="Daily Profit",
    )

    # Calculate 7-day moving average
    daily_sales["profit_ma7"] = daily_sales["profit_amount"].rolling(window=7).mean()
    axes[1].plot(
        daily_sales["date"],
        daily_sales["profit_ma7"],
        linewidth=3,
        color="red",
        label="7-day MA",
    )

    axes[1].set_title("Profit Trend with Moving Average")
    axes[1].set_ylabel("Profit Amount ($)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Customer satisfaction trend
    axes[2].plot(
        daily_sales["date"],
        daily_sales["customer_satisfaction"],
        linewidth=2,
        color="green",
    )
    axes[2].axhline(
        y=daily_sales["customer_satisfaction"].mean(),
        color="red",
        linestyle="--",
        label=f'Average: {daily_sales["customer_satisfaction"].mean():.2f}',
    )
    axes[2].set_title("Customer Satisfaction Trend")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Satisfaction Score")
    axes[2].set_ylim([3.5, 5])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, daily_sales


def create_interactive_plotly_dashboard(data):
    """Create interactive plotly visualizations"""
    print("Creating interactive Plotly dashboard...")

    # Create subplot dashboard with proper specs for pie chart
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Regional Sales Performance",
            "Product Category Analysis",
            "Sales vs Marketing Efficiency",
            "Seasonal Trends",
        ),
        specs=[
            [{"secondary_y": False}, {"type": "domain"}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # 1. Regional sales performance (bar chart)
    regional_sales = (
        data.groupby("region")["sales_amount"].sum().sort_values(ascending=False)
    )
    fig.add_trace(
        go.Bar(
            x=regional_sales.index,
            y=regional_sales.values,
            name="Sales by Region",
            marker_color="lightblue",
        ),
        row=1,
        col=1,
    )

    # 2. Product category distribution (pie chart)
    category_sales = data.groupby("product_category")["sales_amount"].sum()
    fig.add_trace(
        go.Pie(
            labels=category_sales.index,
            values=category_sales.values,
            name="Category Distribution",
        ),
        row=1,
        col=2,
    )

    # 3. Sales vs Marketing scatter
    fig.add_trace(
        go.Scatter(
            x=data["marketing_spend"],
            y=data["sales_amount"],
            mode="markers",
            name="Sales vs Marketing",
            marker={
                "size": data["customer_satisfaction"] * 3,
                "color": data["profit_margin"],
                "colorscale": "Viridis",
                "showscale": True,
            },
            text=data["region"],
            hovertemplate="<b>%{text}</b><br>Marketing: $%{x}<br>Sales: $%{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 4. Seasonal trends
    seasonal_data = (
        data.groupby("season")
        .agg({"sales_amount": "mean", "customer_satisfaction": "mean"})
        .reset_index()
    )

    fig.add_trace(
        go.Bar(
            x=seasonal_data["season"],
            y=seasonal_data["sales_amount"],
            name="Seasonal Sales",
            marker_color="orange",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=800, title_text="Interactive Sales Analytics Dashboard")

    print("Interactive dashboard created successfully!")
    return fig


def create_advanced_visualizations(data):
    """Create advanced visualization techniques"""
    print("Creating advanced visualization techniques...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Advanced Visualization Techniques", fontsize=16, fontweight="bold")

    # 1. Violin plot for distribution comparison
    ax = axes[0, 0]
    sns.violinplot(data=data, x="season", y="sales_amount", ax=ax)
    ax.set_title("Sales Distribution by Season (Violin Plot)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Sales Amount ($)")

    # 2. Pair plot style correlation
    ax = axes[0, 1]
    sample_data = data.sample(200)  # Sample for performance
    scatter = ax.scatter(
        sample_data["sales_amount"],
        sample_data["profit_amount"],
        c=sample_data["customer_satisfaction"],
        cmap="viridis",
        s=sample_data["marketing_spend"] / 10,
        alpha=0.6,
    )
    ax.set_title("Multi-dimensional Analysis")
    ax.set_xlabel("Sales Amount ($)")
    ax.set_ylabel("Profit Amount ($)")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Customer Satisfaction")

    # 3. Stacked bar chart
    ax = axes[1, 0]
    pivot_data = data.pivot_table(
        values="sales_amount", index="region", columns="season", aggfunc="sum"
    )
    pivot_data.plot(kind="bar", stacked=True, ax=ax, colormap="Set3")
    ax.set_title("Regional Sales by Season (Stacked)")
    ax.set_xlabel("Region")
    ax.set_ylabel("Sales Amount ($)")
    ax.legend(title="Season", bbox_to_anchor=(1.05, 1), loc="upper left")

    # 4. Hexbin plot for density
    ax = axes[1, 1]
    hb = ax.hexbin(
        data["marketing_spend"], data["sales_amount"], gridsize=20, cmap="Blues"
    )
    ax.set_title("Marketing vs Sales Density (Hexbin)")
    ax.set_xlabel("Marketing Spend ($)")
    ax.set_ylabel("Sales Amount ($)")

    # Add colorbar for hexbin
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Density")

    plt.tight_layout()
    plt.show()

    return fig


def analyze_visualization_insights(data):
    """Generate insights from visualizations"""
    print("\\nGenerating visualization insights...")

    insights = {}

    # Sales performance insights
    regional_performance = data.groupby("region")["sales_amount"].agg(
        ["mean", "sum", "count"]
    )
    top_region = regional_performance["sum"].idxmax()
    insights["top_region"] = (
        f"{top_region} leads in total sales with ${regional_performance.loc[top_region, 'sum']:,.0f}"
    )

    # Product category insights
    category_performance = data.groupby("product_category")["profit_margin"].mean()
    most_profitable = category_performance.idxmax()
    insights["most_profitable"] = (
        f"{most_profitable} has highest profit margin at {category_performance[most_profitable]:.1%}"
    )

    # Seasonal patterns
    seasonal_sales = data.groupby("season")["sales_amount"].mean()
    peak_season = seasonal_sales.idxmax()
    insights["peak_season"] = (
        f"{peak_season} shows highest average sales at ${seasonal_sales[peak_season]:,.0f}"
    )

    # Marketing efficiency
    data["marketing_efficiency"] = data["sales_amount"] / data["marketing_spend"]
    avg_efficiency = data["marketing_efficiency"].mean()
    insights["marketing_efficiency"] = (
        f"Average marketing ROI: {avg_efficiency:.1f}x return on spend"
    )

    # Customer satisfaction correlation
    corr_sales_satisfaction = data["sales_amount"].corr(data["customer_satisfaction"])
    insights["satisfaction_correlation"] = (
        f"Sales-satisfaction correlation: {corr_sales_satisfaction:.3f}"
    )

    return insights


def main():
    """Main function to run all visualization challenges"""
    print("=" * 60)
    print("LEVEL 3 CHALLENGE 1: MASTER DATA VISUALIZATION")
    print("=" * 60)

    # Setup environment
    setup_visualization_environment()

    # Create comprehensive dataset
    data = create_comprehensive_dataset()

    # Create statistical visualizations
    print("\\n" + "=" * 50)
    print("SECTION 1: STATISTICAL VISUALIZATIONS")
    print("=" * 50)
    stat_fig = create_statistical_visualizations(data)

    # Create time series analysis
    print("\\n" + "=" * 50)
    print("SECTION 2: TIME SERIES ANALYSIS")
    print("=" * 50)
    ts_fig, daily_data = create_time_series_analysis(data)

    # Create interactive dashboard
    print("\\n" + "=" * 50)
    print("SECTION 3: INTERACTIVE PLOTLY DASHBOARD")
    print("=" * 50)
    interactive_fig = create_interactive_plotly_dashboard(data)

    # Create advanced visualizations
    print("\\n" + "=" * 50)
    print("SECTION 4: ADVANCED VISUALIZATION TECHNIQUES")
    print("=" * 50)
    advanced_fig = create_advanced_visualizations(data)

    # Generate insights
    insights = analyze_visualization_insights(data)

    print("\\n" + "=" * 60)
    print("VISUALIZATION INSIGHTS SUMMARY")
    print("=" * 60)

    for category, insight in insights.items():
        print(f"  • {insight}")

    print("\\n" + "=" * 60)
    print("CHALLENGE 1 COMPLETION SUMMARY")
    print("=" * 60)

    techniques_mastered = [
        "Statistical distributions with KDE overlays",
        "Box plots for category comparisons",
        "Correlation heatmaps for feature relationships",
        "Time series with moving averages",
        "Interactive plotly dashboards with subplots",
        "Multi-dimensional scatter plots with size/color encoding",
        "Violin plots for distribution analysis",
        "Stacked bar charts for categorical breakdowns",
        "Hexbin plots for density visualization",
    ]

    print("Visualization techniques mastered:")
    for i, technique in enumerate(techniques_mastered, 1):
        print(f"  {i}. {technique}")

    print("\\nDataset Analysis:")
    print(f"  - Records processed: {len(data):,}")
    print(f"  - Features analyzed: {data.shape[1]}")
    print(f"  - Regions covered: {data['region'].nunique()}")
    print(f"  - Product categories: {data['product_category'].nunique()}")
    print(f"  - Time period: {(data['date'].max() - data['date'].min()).days} days")

    return {
        "data": data,
        "daily_data": daily_data,
        "statistical_figure": stat_fig,
        "time_series_figure": ts_fig,
        "interactive_figure": interactive_fig,
        "advanced_figure": advanced_fig,
        "insights": insights,
    }


if __name__ == "__main__":
    results = main()

    print("\\n" + "=" * 60)
    print("CHALLENGE 1 STATUS: COMPLETE")
    print("=" * 60)
    print("Data visualization mastery achieved!")
    print("Ready for Challenge 2: Interactive Dashboards.")
