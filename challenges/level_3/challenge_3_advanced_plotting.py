"""
Level 3 Challenge 3: Advanced Plotting Techniques
Master sophisticated visualizat        for day_idx, day in enumerate(dates):
            stock_data.append(
                {
                    "date": day,
                    "stock": stock,
                    "sector": sectors[i],
                    "price": prices[day_idx, i],
                    "volume": rng.poisson(1000000) + 500000,
                    "market_cap": rng.uniform(1, 100),  # Billions
                    "pe_ratio": rng.gamma(2, 8),
                    "dividend_yield": rng.uniform(0, 0.06),
                    "volatility": rng.uniform(0.1, 0.4),
                }
            ) for complex data analysis and presentation.
"""

import warnings
from typing import Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.linalg import inv
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")


def generate_financial_data() -> pd.DataFrame:
    """Generate comprehensive multi-dimensional financial dataset"""
    print("Generating advanced financial market dataset...")

    # Set advanced styling
    plt.style.use("default")
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Create comprehensive multi-dimensional dataset
    rng = np.random.default_rng(42)

    # Simulate financial market data
    n_stocks = 8
    n_days = 500
    stock_names = [
        "TECH_A",
        "TECH_B",
        "FINANCE_A",
        "FINANCE_B",
        "ENERGY_A",
        "ENERGY_B",
        "RETAIL_A",
        "RETAIL_B",
    ]
    sectors = [
        "Technology",
        "Technology",
        "Finance",
        "Finance",
        "Energy",
        "Energy",
        "Retail",
        "Retail",
    ]

    # Generate stock price data with realistic correlations
    base_returns = rng.normal(0.001, 0.02, (n_days, n_stocks))

    # Add sector correlations
    tech_factor = rng.normal(0, 0.01, n_days)
    finance_factor = rng.normal(0, 0.015, n_days)
    energy_factor = rng.normal(0, 0.025, n_days)
    retail_factor = rng.normal(0, 0.012, n_days)

    base_returns[:, 0:2] += tech_factor[:, np.newaxis] * 0.7  # Technology stocks
    base_returns[:, 2:4] += finance_factor[:, np.newaxis] * 0.6  # Finance stocks
    base_returns[:, 4:6] += energy_factor[:, np.newaxis] * 0.8  # Energy stocks
    base_returns[:, 6:8] += retail_factor[:, np.newaxis] * 0.5  # Retail stocks

    # Calculate cumulative prices
    initial_prices = rng.uniform(50, 200, n_stocks)
    prices = initial_prices * np.exp(np.cumsum(base_returns, axis=0))

    # Create comprehensive DataFrame
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    stock_data = []

    for i, stock in enumerate(stock_names):
        for j, date in enumerate(dates):
            stock_data.append(
                {
                    "date": date,
                    "stock": stock,
                    "sector": sectors[i],
                    "price": prices[j, i],
                    "volume": rng.poisson(1000000) + 500000,
                    "market_cap": rng.uniform(1, 100),  # Billions
                    "pe_ratio": rng.gamma(2, 8),
                    "dividend_yield": rng.uniform(0, 0.06),
                    "volatility": rng.uniform(0.1, 0.4),
                }
            )

    df = pd.DataFrame(stock_data)

    # Calculate additional metrics
    df["returns"] = df.groupby("stock")["price"].pct_change()
    df["log_returns"] = np.log(df["price"] / df.groupby("stock")["price"].shift(1))
    df["sma_20"] = (
        df.groupby("stock")["price"].rolling(20).mean().reset_index(0, drop=True)
    )
    df["rsi"] = (
        df.groupby("stock")["returns"]
        .rolling(14)
        .apply(
            lambda x: (
                100 - (100 / (1 + np.abs(x[x > 0].sum() / x[x < 0].sum())))
                if len(x[x < 0]) > 0
                else 50
            )
        )
        .reset_index(0, drop=True)
    )

    print(f"Advanced plotting dataset ready! Data shape: {df.shape}")
    return df


def create_statistical_distributions(df: pd.DataFrame) -> Any:
    """Create advanced statistical distribution plots with confidence intervals"""
    print("Creating advanced statistical distribution analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Advanced Statistical Analysis", fontsize=16, fontweight="bold")

    # 1. Probability density with confidence intervals
    sector_returns = df.dropna().groupby("sector")["returns"].apply(list)

    for i, (sector, returns) in enumerate(sector_returns.items()):
        if i < 4:  # Plot first 4 sectors
            row, col = i // 2, i % 2

            # Fit normal distribution
            mu, sigma = stats.norm.fit(returns)

            # Create density plot with confidence intervals
            axes[row, col].hist(
                returns, bins=50, density=True, alpha=0.7, color=f"C{i}", label="Data"
            )

            # Plot fitted normal distribution
            x = np.linspace(min(returns), max(returns), 100)
            axes[row, col].plot(
                x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal fit"
            )

            # Add confidence intervals
            confidence = 0.95
            interval = stats.norm.interval(confidence, mu, sigma)
            axes[row, col].axvspan(
                interval[0],
                interval[1],
                alpha=0.2,
                color="red",
                label=f"{confidence*100}% CI",
            )

            axes[row, col].set_title(f"{sector} Returns Distribution")
            axes[row, col].set_xlabel("Daily Returns")
            axes[row, col].set_ylabel("Density")
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

    # 2. Q-Q plots for normality testing
    latest_data = df[df["date"] == df["date"].max()]
    axes[0, 2].set_title("Q-Q Plot: Normality Test")
    stats.probplot(latest_data["returns"].dropna(), dist="norm", plot=axes[0, 2])
    axes[0, 2].get_lines()[0].set_markerfacecolor("C0")
    axes[0, 2].get_lines()[1].set_color("red")

    # 3. Advanced box plots with statistical annotations
    sector_pe_data = [
        df[df["sector"] == sector]["pe_ratio"].dropna()
        for sector in df["sector"].unique()
    ]
    bp = axes[1, 2].boxplot(
        sector_pe_data, labels=df["sector"].unique(), patch_artist=True
    )

    colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Add mean markers
    for i, data in enumerate(sector_pe_data):
        mean_val = np.mean(data)
        axes[1, 2].plot(
            i + 1, mean_val, "ro", markersize=8, label="Mean" if i == 0 else ""
        )

    axes[1, 2].set_title("P/E Ratio Distribution by Sector")
    axes[1, 2].set_ylabel("P/E Ratio")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

    return fig


def create_correlation_analysis(df: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
    """Create multi-dimensional correlation analysis with clustering"""
    print("Creating multi-dimensional correlation analysis...")

    # Create correlation matrix for latest data points
    latest_stocks = (
        df[df["date"] == df["date"].max()]
        .pivot(index="date", columns="stock", values="price")
        .corr()
    )

    # Advanced heatmap with hierarchical clustering
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Standard correlation heatmap
    sns.heatmap(
        latest_stocks,
        annot=True,
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=axes[0],
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Stock Price Correlations")

    # 2. Clustermap-style correlation with dendrogram
    # Calculate distance matrix and perform clustering
    distance_matrix = 1 - latest_stocks.abs()
    linkage_matrix = linkage(squareform(distance_matrix), method="ward")

    # Create custom colormap - professional harmonious palette
    colors = ["#DC2626", "#C2410C", "#F9FAFB", "#047857", "#1D4ED8"]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Plot correlation with reordered indices based on clustering
    cluster_order = dendrogram(linkage_matrix, no_plot=True)["leaves"]
    reordered_corr = latest_stocks.iloc[cluster_order, cluster_order]

    sns.heatmap(
        reordered_corr,
        annot=True,
        cmap=cmap,
        center=0,
        square=True,
        ax=axes[1],
        cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title("Hierarchically Clustered Correlations")

    # 3. Partial correlation network visualization
    # Calculate partial correlations
    precision_matrix = inv(latest_stocks.values)
    partial_corr = -precision_matrix / np.sqrt(
        np.outer(np.diag(precision_matrix), np.diag(precision_matrix))
    )
    np.fill_diagonal(partial_corr, 1)

    partial_corr_df = pd.DataFrame(
        partial_corr, index=latest_stocks.index, columns=latest_stocks.columns
    )
    sns.heatmap(
        partial_corr_df,
        annot=True,
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=axes[2],
        cbar_kws={"shrink": 0.8},
    )
    axes[2].set_title("Partial Correlations (Network Effects)")

    plt.tight_layout()
    plt.show()

    return fig, latest_stocks


def create_technical_analysis(df: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
    """Create advanced time series visualization with technical indicators"""
    print("Creating advanced technical analysis charts...")

    # Select one stock for detailed analysis
    tech_stock = df[df["stock"] == "TECH_A"].copy()
    tech_stock = tech_stock.sort_values("date")

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.suptitle("Advanced Technical Analysis - TECH_A", fontsize=16, fontweight="bold")

    # 1. Candlestick-style price chart with moving averages
    # Simulate OHLC data
    tech_stock["open"] = tech_stock["price"].shift(1) * (
        1 + rng.normal(0, 0.001, len(tech_stock))
    )
    tech_stock["high"] = tech_stock[["price", "open"]].max(axis=1) * (
        1 + rng.uniform(0, 0.01, len(tech_stock))
    )
    tech_stock["low"] = tech_stock[["price", "open"]].min(axis=1) * (
        1 - rng.uniform(0, 0.01, len(tech_stock))
    )
    tech_stock["close"] = tech_stock["price"]

    # Plot price with Bollinger Bands
    tech_stock["bb_upper"] = tech_stock["sma_20"] + (
        tech_stock["price"].rolling(20).std() * 2
    )
    tech_stock["bb_lower"] = tech_stock["sma_20"] - (
        tech_stock["price"].rolling(20).std() * 2
    )

    axes[0].plot(
        tech_stock["date"],
        tech_stock["close"],
        color="black",
        linewidth=1,
        label="Close Price",
    )
    axes[0].plot(
        tech_stock["date"],
        tech_stock["sma_20"],
        color="blue",
        linewidth=1.5,
        label="20-day SMA",
    )
    axes[0].fill_between(
        tech_stock["date"],
        tech_stock["bb_upper"],
        tech_stock["bb_lower"],
        alpha=0.2,
        color="gray",
        label="Bollinger Bands",
    )
    axes[0].set_title("Price with Bollinger Bands and Moving Average")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Volume analysis with price overlay
    ax_vol = axes[1]
    ax_price = ax_vol.twinx()

    # Volume bars
    volume_colors = [
        "green" if ret > 0 else "red" for ret in tech_stock["returns"].fillna(0)
    ]
    ax_vol.bar(tech_stock["date"], tech_stock["volume"], alpha=0.6, color=volume_colors)
    ax_vol.set_ylabel("Volume", color="gray")

    # Price overlay
    ax_price.plot(tech_stock["date"], tech_stock["close"], color="black", linewidth=2)
    ax_price.set_ylabel("Price ($)", color="black")

    ax_vol.set_title("Volume Analysis with Price Overlay")
    ax_vol.grid(True, alpha=0.3)

    # 3. RSI with overbought/oversold levels
    axes[2].plot(tech_stock["date"], tech_stock["rsi"], color="purple", linewidth=2)
    axes[2].axhline(
        y=70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)"
    )
    axes[2].axhline(
        y=30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)"
    )
    axes[2].fill_between(
        tech_stock["date"], 30, 70, alpha=0.1, color="gray", label="Normal Range"
    )
    axes[2].set_title("Relative Strength Index (RSI)")
    axes[2].set_ylabel("RSI")
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 4. Returns distribution with risk metrics
    returns_clean = tech_stock["returns"].dropna()
    axes[3].hist(
        returns_clean,
        bins=50,
        alpha=0.7,
        density=True,
        color="lightblue",
        edgecolor="black",
    )

    # Add distribution statistics
    mean_return = returns_clean.mean()
    std_return = returns_clean.std()
    var_95 = np.percentile(returns_clean, 5)  # Value at Risk (95%)
    var_99 = np.percentile(returns_clean, 1)  # Value at Risk (99%)

    axes[3].axvline(
        mean_return,
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_return:.4f}",
    )
    axes[3].axvline(
        var_95,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"VaR 95%: {var_95:.4f}",
    )
    axes[3].axvline(
        var_99, color="red", linestyle="--", linewidth=2, label=f"VaR 99%: {var_99:.4f}"
    )

    # Fit and plot normal distribution
    x = np.linspace(returns_clean.min(), returns_clean.max(), 100)
    axes[3].plot(
        x,
        stats.norm.pdf(x, mean_return, std_return),
        "r-",
        linewidth=2,
        label="Normal Fit",
    )

    axes[3].set_title("Returns Distribution with Risk Metrics")
    axes[3].set_xlabel("Daily Returns")
    axes[3].set_ylabel("Density")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, tech_stock


def create_interactive_3d_analysis(df: pd.DataFrame) -> Tuple[Any, Any, pd.DataFrame]:
    """Create interactive 3D and advanced Plotly visualizations"""
    print("Creating interactive 3D portfolio analysis...")

    rng = np.random.default_rng(42)

    # Create multi-dimensional portfolio analysis
    portfolio_data = (
        df.groupby(["sector", "stock"])
        .agg(
            {
                "returns": "std",
                "price": "last",
                "market_cap": "last",
                "pe_ratio": "last",
                "dividend_yield": "last",
            }
        )
        .reset_index()
    )

    portfolio_data.columns = [
        "sector",
        "stock",
        "volatility",
        "price",
        "market_cap",
        "pe_ratio",
        "dividend_yield",
    ]
    portfolio_data["expected_return"] = rng.normal(0.08, 0.03, len(portfolio_data))

    # 1. 3D Risk-Return-Size Analysis
    fig_3d = go.Figure(
        data=go.Scatter3d(
            x=portfolio_data["volatility"],
            y=portfolio_data["expected_return"],
            z=portfolio_data["pe_ratio"],
            mode="markers+text",
            marker=dict(
                size=portfolio_data["market_cap"],
                color=portfolio_data["dividend_yield"],
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="Dividend Yield"),
                sizemode="diameter",
                sizeref=2.0 * max(portfolio_data["market_cap"]) / (40.0**2),
                sizemin=4,
            ),
            text=portfolio_data["stock"],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>"
            + "Volatility: %{x:.3f}<br>"
            + "Expected Return: %{y:.3f}<br>"
            + "P/E Ratio: %{z:.1f}<br>"
            + "Market Cap: %{marker.size:.1f}B<br>"
            + "Dividend Yield: %{marker.color:.3f}<br>"
            + "<extra></extra>",
        )
    )

    fig_3d.update_layout(
        title="3D Portfolio Analysis: Risk-Return-Valuation",
        scene=dict(
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            zaxis_title="P/E Ratio (Valuation)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        height=600,
    )

    print("3D Portfolio Analysis chart created successfully!")

    # 2. Advanced Subplot Dashboard with Secondary Axes
    fig_subplots = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Sector Performance",
            "Risk-Return Scatter",
            "Valuation Metrics",
            "Time Series Comparison",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": True}, {"secondary_y": False}],
        ],
    )

    # Sector performance
    sector_metrics = (
        portfolio_data.groupby("sector")
        .agg(
            {
                "expected_return": "mean",
                "volatility": "mean",
                "pe_ratio": "mean",
                "dividend_yield": "mean",
            }
        )
        .reset_index()
    )

    fig_subplots.add_trace(
        go.Bar(
            x=sector_metrics["sector"],
            y=sector_metrics["expected_return"],
            name="Expected Return",
            marker_color="lightblue",
        ),
        row=1,
        col=1,
    )

    # Risk-return scatter
    fig_subplots.add_trace(
        go.Scatter(
            x=portfolio_data["volatility"],
            y=portfolio_data["expected_return"],
            mode="markers",
            marker=dict(color=portfolio_data.index, colorscale="plasma", size=10),
            text=portfolio_data["stock"],
            name="Stocks",
        ),
        row=1,
        col=2,
    )

    # Valuation metrics with dual axis
    fig_subplots.add_trace(
        go.Bar(
            x=portfolio_data["stock"],
            y=portfolio_data["pe_ratio"],
            name="P/E Ratio",
            marker_color="orange",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    fig_subplots.add_trace(
        go.Scatter(
            x=portfolio_data["stock"],
            y=portfolio_data["dividend_yield"],
            mode="lines+markers",
            name="Dividend Yield",
            line=dict(color="red", width=3),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # Time series comparison
    sample_stocks = ["TECH_A", "FINANCE_A", "ENERGY_A"]
    for stock in sample_stocks:
        stock_ts = df[df["stock"] == stock].sort_values("date")
        fig_subplots.add_trace(
            go.Scatter(
                x=stock_ts["date"], y=stock_ts["price"], mode="lines", name=stock
            ),
            row=2,
            col=2,
        )

    fig_subplots.update_layout(
        height=800, title_text="Advanced Multi-Panel Analysis Dashboard"
    )
    fig_subplots.update_xaxes(tickangle=45, row=2, col=1)
    fig_subplots.update_yaxes(title_text="P/E Ratio", row=2, col=1, secondary_y=False)
    fig_subplots.update_yaxes(
        title_text="Dividend Yield", row=2, col=1, secondary_y=True
    )

    print("Advanced multi-panel dashboard created successfully!")

    return fig_3d, fig_subplots, portfolio_data


def create_publication_ready_plot(df: pd.DataFrame) -> Tuple[Any, dict]:
    """Create publication-ready figure with custom annotations and styling"""
    print("Creating publication-ready efficient frontier analysis...")

    rng = np.random.default_rng(42)

    # Create publication-ready figure with custom styling
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "axes.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.major.size": 5,
            "ytick.minor.size": 3,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create efficient frontier simulation
    n_portfolios = 1000
    returns = []
    volatilities = []
    sharpe_ratios = []

    stock_returns = df.pivot_table(
        values="returns", index="date", columns="stock"
    ).dropna()
    mean_returns = stock_returns.mean()
    cov_matrix = stock_returns.cov()

    for _ in range(n_portfolios):
        weights = rng.random(len(mean_returns))
        weights /= np.sum(weights)

        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix * 252, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        returns.append(portfolio_return)
        volatilities.append(portfolio_volatility)
        sharpe_ratios.append(portfolio_return / portfolio_volatility)

    # Create scatter plot with color mapping
    scatter = ax.scatter(
        volatilities,
        returns,
        c=sharpe_ratios,
        cmap="RdYlBu_r",
        alpha=0.6,
        s=50,
        edgecolors="white",
        linewidth=0.5,
    )

    # Add colorbar with custom formatting
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Sharpe Ratio", rotation=270, labelpad=20)

    # Find and highlight optimal portfolio (max Sharpe ratio)
    max_sharpe_idx = np.argmax(sharpe_ratios)
    ax.scatter(
        volatilities[max_sharpe_idx],
        returns[max_sharpe_idx],
        marker="*",
        color="red",
        s=300,
        edgecolors="black",
        linewidth=2,
        label="Optimal Portfolio",
        zorder=5,
    )

    # Add annotations with arrows
    ax.annotate(
        "Maximum Sharpe Ratio\nPortfolio",
        xy=(volatilities[max_sharpe_idx], returns[max_sharpe_idx]),
        xytext=(volatilities[max_sharpe_idx] + 0.05, returns[max_sharpe_idx] + 0.02),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0.3", color="black", lw=2
        ),
        fontsize=11,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Add risk-free rate line (Capital Market Line approximation)
    risk_free_rate = 0.02
    max_vol = max(volatilities)
    x_line = np.linspace(0, max_vol, 100)
    y_line = (
        risk_free_rate
        + (returns[max_sharpe_idx] - risk_free_rate)
        / volatilities[max_sharpe_idx]
        * x_line
    )

    ax.plot(
        x_line,
        y_line,
        "--",
        color="black",
        linewidth=2,
        alpha=0.7,
        label="Capital Market Line",
    )

    # Professional styling
    ax.set_xlabel("Portfolio Volatility (Risk)", fontweight="bold")
    ax.set_ylabel("Expected Annual Return", fontweight="bold")
    ax.set_title(
        "Efficient Frontier Analysis\nPortfolio Optimization with Risk-Return Trade-off",
        fontweight="bold",
        pad=20,
    )

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)

    # Add text box with summary statistics
    textstr = f"""Portfolio Analysis Summary:
• Total Portfolios: {n_portfolios:,}
• Optimal Sharpe Ratio: {max(sharpe_ratios):.3f}
• Optimal Return: {returns[max_sharpe_idx]:.1%}
• Optimal Risk: {volatilities[max_sharpe_idx]:.1%}
• Risk-Free Rate: {risk_free_rate:.1%}"""

    props = dict(boxstyle="round", facecolor="lightgray", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.show()

    # Reset matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)

    return fig, {
        "n_portfolios": n_portfolios,
        "max_sharpe_ratio": max(sharpe_ratios),
        "optimal_return": returns[max_sharpe_idx],
        "optimal_volatility": volatilities[max_sharpe_idx],
        "risk_free_rate": risk_free_rate,
    }


def main() -> dict:
    """Main function to run all advanced plotting challenges"""
    print("=" * 60)
    print("LEVEL 3 CHALLENGE 3: ADVANCED PLOTTING TECHNIQUES")
    print("=" * 60)

    # Generate comprehensive financial dataset
    df = generate_financial_data()

    print("\n" + "=" * 60)
    print("CHALLENGE 1: ADVANCED STATISTICAL DISTRIBUTIONS")
    print("=" * 60)

    # Create statistical distribution plots
    _ = create_statistical_distributions(df)

    print("\n" + "=" * 60)
    print("CHALLENGE 2: MULTI-DIMENSIONAL CORRELATION ANALYSIS")
    print("=" * 60)

    # Create correlation analysis
    _, correlation_matrix = create_correlation_analysis(df)

    print("\n" + "=" * 60)
    print("CHALLENGE 3: TIME SERIES WITH TECHNICAL INDICATORS")
    print("=" * 60)

    # Create technical analysis
    _, _ = create_technical_analysis(df)

    print("\n" + "=" * 60)
    print("CHALLENGE 4: INTERACTIVE 3D AND ADVANCED PLOTLY")
    print("=" * 60)

    # Create 3D interactive plots
    _, _, portfolio_data = create_interactive_3d_analysis(df)

    print("\n" + "=" * 60)
    print("CHALLENGE 5: CUSTOM ANNOTATIONS AND STYLING")
    print("=" * 60)

    # Create publication-ready plot
    _, portfolio_stats = create_publication_ready_plot(df)

    print("\n" + "=" * 60)
    print("CHALLENGE 3 COMPLETION SUMMARY")
    print("=" * 60)

    print("✅ Advanced plotting challenge completed successfully!")
    print("\nMastered visualization techniques:")
    print("• Statistical distributions with confidence intervals and Q-Q plots")
    print("• Multi-dimensional correlation analysis with hierarchical clustering")
    print("• Technical analysis with Bollinger Bands, RSI, and risk metrics")
    print("• Interactive 3D visualizations with multi-panel dashboards")
    print("• Publication-ready plots with professional annotations and styling")

    print("\nDataset Statistics:")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Unique stocks: {df['stock'].nunique()}")
    print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  - Sectors analyzed: {df['sector'].nunique()}")

    print("\nPortfolio Analysis Results:")
    print(f"  - Portfolios simulated: {portfolio_stats['n_portfolios']:,}")
    print(f"  - Maximum Sharpe ratio: {portfolio_stats['max_sharpe_ratio']:.3f}")
    print(f"  - Optimal annual return: {portfolio_stats['optimal_return']:.1%}")
    print(f"  - Optimal volatility: {portfolio_stats['optimal_volatility']:.1%}")

    return {
        "dataset": df,
        "portfolio_data": portfolio_data,
        "correlation_matrix": correlation_matrix,
        "portfolio_stats": portfolio_stats,
        "figures_created": 5,
    }


if __name__ == "__main__":
    results = main()

    print("\n" + "=" * 60)
    print("CHALLENGE 3 STATUS: COMPLETE")
    print("=" * 60)
    print("Advanced plotting techniques mastered!")
    print("Ready for Challenge 4: Storytelling with Data.")
