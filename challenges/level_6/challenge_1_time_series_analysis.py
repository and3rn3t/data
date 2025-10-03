"""
Level 6 - Challenge 1: Time Series Analysis & Forecasting
=========================================================

Master time series analysis, forecasting, and temporal pattern detection.
This challenge covers trend analysis, seasonality, ARIMA models, and advanced forecasting.

Learning Objectives:
- Understand time series components and patterns
- Learn decomposition techniques and stationarity testing
- Master ARIMA, SARIMA, and exponential smoothing models
- Explore advanced forecasting methods and evaluation
- Handle missing data and outliers in time series

Required Libraries: pandas, numpy, matplotlib, scipy, statsmodels
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import advanced time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print(
        "📚 Using simplified time series methods - install statsmodels for advanced features"
    )


def create_time_series_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Create comprehensive time series datasets for analysis and forecasting.

    Returns:
        Dictionary containing various time series datasets
    """
    print("📈 Creating Time Series Datasets...")

    datasets = {}
    rng = np.random.default_rng(42)

    # 1. Sales Time Series with Trend and Seasonality
    print("Creating sales time series...")

    # Generate 3 years of daily data
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    # Create components
    t = np.arange(n_days)

    # Trend component (gradual growth)
    trend = 1000 + 0.5 * t + 0.0001 * t**2

    # Seasonal component (yearly pattern)
    seasonal_yearly = 200 * np.sin(2 * np.pi * t / 365.25)

    # Weekly seasonality (higher sales on weekends)
    seasonal_weekly = 50 * np.sin(2 * np.pi * t / 7 + np.pi / 2)

    # Holiday effects (spikes during holidays)
    holiday_effect = np.zeros(n_days)
    for i, date in enumerate(dates):
        # Christmas effect (December)
        if date.month == 12 and date.day >= 20:
            holiday_effect[i] = 300
        # Black Friday effect (last Friday of November)
        elif date.month == 11 and date.weekday() == 4 and date.day >= 23:
            holiday_effect[i] = 400
        # Summer vacation effect (July-August)
        elif date.month in [7, 8]:
            holiday_effect[i] = 100

    # Random noise
    noise = rng.normal(0, 50, n_days)

    # Occasional outliers
    outlier_indices = rng.choice(n_days, size=20, replace=False)
    outliers = rng.normal(0, 200, 20)
    noise[outlier_indices] += outliers

    # Combine all components
    sales_values = trend + seasonal_yearly + seasonal_weekly + holiday_effect + noise
    sales_values = np.maximum(sales_values, 0)  # Sales can't be negative

    sales_df = pd.DataFrame(
        {
            "date": dates,
            "sales": sales_values,
            "trend": trend,
            "seasonal_yearly": seasonal_yearly,
            "seasonal_weekly": seasonal_weekly,
            "holiday_effect": holiday_effect,
            "noise": noise,
        }
    )
    sales_df.set_index("date", inplace=True)

    datasets["sales"] = {
        "data": sales_df,
        "target_column": "sales",
        "frequency": "D",
        "description": "Daily retail sales with trend, seasonality, and holiday effects",
    }

    # 2. Stock Price Time Series (Financial Data)
    print("Creating stock price time series...")

    # Generate 2 years of hourly stock data (trading hours only)
    stock_dates = []
    current_date = datetime(2022, 1, 3)  # Start on a Monday
    while current_date <= datetime(2023, 12, 29):
        # Only trading days (Monday-Friday)
        if current_date.weekday() < 5:
            # Trading hours (9 AM - 4 PM)
            for hour in range(9, 16):
                stock_dates.append(current_date.replace(hour=hour))
        current_date += timedelta(days=1)

    n_hours = len(stock_dates)

    # Stock price simulation (geometric Brownian motion)
    initial_price = 100.0
    drift = 0.0001  # Small positive trend
    volatility = 0.02

    # Generate random walks
    returns = rng.normal(drift, volatility, n_hours)

    # Add market hours effect (higher volatility at open/close)
    for i, dt in enumerate(stock_dates):
        if dt.hour in [9, 15]:  # Opening and closing hours
            returns[i] *= 1.5

    # Convert to prices
    log_prices = np.cumsum(returns)
    stock_prices = initial_price * np.exp(log_prices)

    # Add volume (correlated with volatility)
    base_volume = 1000000
    volume_multiplier = 1 + np.abs(returns) * 10
    volumes = base_volume * volume_multiplier * (1 + rng.normal(0, 0.3, n_hours))
    volumes = np.maximum(volumes, 10000)

    stock_df = pd.DataFrame(
        {
            "datetime": stock_dates,
            "price": stock_prices,
            "volume": volumes,
            "returns": returns,
            "high": stock_prices * (1 + np.abs(rng.normal(0, 0.005, n_hours))),
            "low": stock_prices * (1 - np.abs(rng.normal(0, 0.005, n_hours))),
        }
    )
    stock_df["open"] = stock_df["price"].shift(1).fillna(stock_df["price"].iloc[0])
    stock_df["close"] = stock_df["price"]
    stock_df.set_index("datetime", inplace=True)

    datasets["stock"] = {
        "data": stock_df,
        "target_column": "price",
        "frequency": "H",
        "description": "Hourly stock prices with OHLC data and volume",
    }

    # 3. IoT Sensor Data (Temperature)
    print("Creating IoT sensor time series...")

    # Generate 6 months of minute-level temperature data
    sensor_start = datetime(2023, 1, 1)
    sensor_end = datetime(2023, 6, 30)
    sensor_dates = pd.date_range(start=sensor_start, end=sensor_end, freq="1min")
    n_minutes = len(sensor_dates)

    # Base temperature pattern
    t_minutes = np.arange(n_minutes)

    # Daily temperature cycle (24-hour pattern)
    daily_temp = 20 + 8 * np.sin(2 * np.pi * t_minutes / (24 * 60) - np.pi / 2)

    # Seasonal trend (winter to summer)
    seasonal_temp = 5 * np.sin(2 * np.pi * t_minutes / (365.25 * 24 * 60) + np.pi / 2)

    # Weather events (random temperature drops/spikes)
    weather_events = np.zeros(n_minutes)
    event_indices = rng.choice(n_minutes, size=100, replace=False)
    event_magnitudes = rng.normal(0, 3, 100)
    weather_events[event_indices] = event_magnitudes

    # Apply smoothing to weather events (events last for hours)
    for i in range(1, n_minutes):
        weather_events[i] = 0.95 * weather_events[i - 1] + 0.05 * weather_events[i]

    # Sensor noise and drift
    sensor_noise = rng.normal(0, 0.5, n_minutes)
    sensor_drift = 0.00001 * t_minutes  # Gradual sensor calibration drift

    # Combine components
    temperature = (
        daily_temp + seasonal_temp + weather_events + sensor_noise + sensor_drift
    )

    # Add humidity (inversely correlated with temperature)
    humidity = 60 - 0.8 * (temperature - 20) + rng.normal(0, 5, n_minutes)
    humidity = np.clip(humidity, 0, 100)

    sensor_df = pd.DataFrame(
        {
            "datetime": sensor_dates,
            "temperature": temperature,
            "humidity": humidity,
            "daily_cycle": daily_temp,
            "seasonal_trend": seasonal_temp,
            "weather_events": weather_events,
        }
    )
    sensor_df.set_index("datetime", inplace=True)

    datasets["iot_sensor"] = {
        "data": sensor_df,
        "target_column": "temperature",
        "frequency": "1min",
        "description": "Minute-level IoT sensor data with daily and seasonal patterns",
    }

    # 4. Website Traffic Data
    print("Creating website traffic time series...")

    # Generate 1 year of hourly website traffic
    traffic_start = datetime(2023, 1, 1)
    traffic_end = datetime(2023, 12, 31)
    traffic_dates = pd.date_range(start=traffic_start, end=traffic_end, freq="H")
    n_traffic_hours = len(traffic_dates)

    # Base traffic level
    base_traffic = 1000

    # Daily pattern (higher during day, lower at night)
    daily_pattern = np.array(
        [
            0.3,
            0.2,
            0.15,
            0.1,
            0.1,
            0.2,
            0.4,
            0.7,
            1.0,
            1.2,
            1.3,
            1.4,
            1.5,
            1.4,
            1.3,
            1.2,
            1.1,
            1.0,
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
            0.4,
        ]
    )

    # Weekly pattern (lower on weekends)
    weekly_pattern = np.array([1.2, 1.3, 1.3, 1.2, 1.1, 0.8, 0.7])  # Mon-Sun

    # Monthly pattern (seasonal business effects)
    monthly_pattern = np.array(
        [0.9, 0.9, 1.0, 1.1, 1.1, 1.2, 1.0, 0.8, 1.1, 1.2, 1.3, 1.4]
    )

    traffic_values = []
    for i, dt in enumerate(traffic_dates):
        daily_mult = daily_pattern[dt.hour]
        weekly_mult = weekly_pattern[dt.weekday()]
        monthly_mult = monthly_pattern[dt.month - 1]

        # Marketing campaign effects (random spikes)
        campaign_effect = 1.0
        if rng.random() < 0.05:  # 5% chance of campaign effect
            campaign_effect = rng.uniform(1.5, 3.0)

        # Growth trend
        growth_mult = 1 + 0.0001 * i

        traffic = (
            base_traffic
            * daily_mult
            * weekly_mult
            * monthly_mult
            * campaign_effect
            * growth_mult
        )
        traffic += rng.normal(0, traffic * 0.1)  # 10% noise
        traffic_values.append(max(0, traffic))

    # Convert to page views, sessions, users
    page_views = np.array(traffic_values)
    sessions = page_views / rng.uniform(2.5, 3.5, n_traffic_hours)  # Pages per session
    users = sessions / rng.uniform(1.2, 1.8, n_traffic_hours)  # Sessions per user

    traffic_df = pd.DataFrame(
        {
            "datetime": traffic_dates,
            "page_views": page_views,
            "sessions": sessions,
            "users": users,
            "bounce_rate": rng.uniform(0.3, 0.7, n_traffic_hours),
            "avg_session_duration": rng.uniform(120, 300, n_traffic_hours),  # seconds
        }
    )
    traffic_df.set_index("datetime", inplace=True)

    datasets["website_traffic"] = {
        "data": traffic_df,
        "target_column": "page_views",
        "frequency": "H",
        "description": "Hourly website traffic with daily, weekly, and seasonal patterns",
    }

    print(f"Created {len(datasets)} time series datasets")
    return datasets


def analyze_time_series_components(data: pd.Series, title: str = "Time Series") -> None:
    """
    Analyze and visualize time series components.
    """
    print(f"\n📊 Time Series Analysis: {title}")
    print("=" * 50)

    # Basic statistics
    print(f"Period: {data.index[0]} to {data.index[-1]}")
    print(f"Length: {len(data)} observations")
    print(f"Mean: {data.mean():.2f}")
    print(f"Std Dev: {data.std():.2f}")
    print(f"Min: {data.min():.2f}, Max: {data.max():.2f}")

    # Missing values
    missing_count = data.isnull().sum()
    print(f"Missing values: {missing_count} ({missing_count/len(data)*100:.1f}%)")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original time series
    axes[0, 0].plot(data.index, data.values, linewidth=1)
    axes[0, 0].set_title(f"{title} - Original Series")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].grid(True, alpha=0.3)

    # Rolling statistics
    window_size = max(7, len(data) // 50)  # Adaptive window size
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()

    axes[0, 1].plot(data.index, data.values, alpha=0.3, label="Original")
    axes[0, 1].plot(
        rolling_mean.index,
        rolling_mean.values,
        color="red",
        label=f"Rolling Mean ({window_size})",
    )
    axes[0, 1].plot(
        rolling_std.index,
        rolling_std.values,
        color="green",
        label=f"Rolling Std ({window_size})",
    )
    axes[0, 1].set_title("Rolling Statistics")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Distribution
    axes[1, 0].hist(data.dropna().values, bins=50, alpha=0.7, edgecolor="black")
    axes[1, 0].set_title("Value Distribution")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # Autocorrelation (simplified)
    max_lags = min(50, len(data) // 4)
    autocorr_values = []
    lags = range(1, max_lags + 1)

    for lag in lags:
        if len(data) > lag:
            corr = data.corr(data.shift(lag))
            autocorr_values.append(corr if not np.isnan(corr) else 0)
        else:
            autocorr_values.append(0)

    axes[1, 1].bar(lags, autocorr_values, alpha=0.7)
    axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1, 1].axhline(y=0.2, color="red", linestyle="--", alpha=0.5, label="±0.2")
    axes[1, 1].axhline(y=-0.2, color="red", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Autocorrelation Function (ACF)")
    axes[1, 1].set_xlabel("Lag")
    axes[1, 1].set_ylabel("Autocorrelation")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Advanced decomposition if statsmodels is available
    if STATSMODELS_AVAILABLE and len(data) >= 24:
        try:
            # Determine period for decomposition
            if isinstance(data.index, pd.DatetimeIndex):
                freq = pd.infer_freq(data.index)
                if freq:
                    if "D" in freq:
                        period = 7  # Weekly seasonality for daily data
                    elif "H" in freq:
                        period = 24  # Daily seasonality for hourly data
                    elif "min" in freq:
                        period = 60  # Hourly seasonality for minute data
                    else:
                        period = 12  # Default quarterly
                else:
                    period = 12
            else:
                period = 12

            if len(data) >= 2 * period:
                decomposition = seasonal_decompose(
                    data.dropna(), model="additive", period=period
                )

                fig, axes = plt.subplots(4, 1, figsize=(15, 12))

                decomposition.observed.plot(ax=axes[0], title="Original")
                decomposition.trend.plot(ax=axes[1], title="Trend")
                decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
                decomposition.resid.plot(ax=axes[3], title="Residual")

                for ax in axes:
                    ax.grid(True, alpha=0.3)

                plt.suptitle(f"{title} - Seasonal Decomposition")
                plt.tight_layout()
                plt.show()

                print(f"\n📈 Decomposition Analysis:")
                print(
                    f"• Trend strength: {1 - decomposition.resid.var() / decomposition.observed.var():.3f}"
                )
                print(
                    f"• Seasonal strength: {1 - (decomposition.resid + decomposition.trend).var() / decomposition.observed.var():.3f}"
                )

        except Exception as e:
            print(f"⚠️ Could not perform seasonal decomposition: {e}")


def test_stationarity(data: pd.Series, title: str = "Series") -> Dict[str, Any]:
    """
    Test time series for stationarity using statistical tests and visual methods.
    """
    print(f"\n🔍 Stationarity Testing: {title}")
    print("=" * 50)

    results = {}

    # Simple statistical test (rolling statistics)
    window_size = max(10, len(data) // 20)

    # Split data into chunks and compare means/variances
    chunk_size = len(data) // 4
    chunks = [
        data.iloc[i * chunk_size : (i + 1) * chunk_size]
        for i in range(4)
        if (i + 1) * chunk_size <= len(data)
    ]

    chunk_means = [chunk.mean() for chunk in chunks if len(chunk) > 0]
    chunk_vars = [chunk.var() for chunk in chunks if len(chunk) > 0]

    mean_stability = (
        np.std(chunk_means) / np.mean(chunk_means) if chunk_means else float("inf")
    )
    var_stability = (
        np.std(chunk_vars) / np.mean(chunk_vars) if chunk_vars else float("inf")
    )

    results["mean_stability"] = mean_stability
    results["variance_stability"] = var_stability

    print(f"Mean stability (lower is better): {mean_stability:.4f}")
    print(f"Variance stability (lower is better): {var_stability:.4f}")

    # Simple stationarity assessment
    is_stationary_simple = (mean_stability < 0.1) and (var_stability < 0.5)
    results["is_stationary_simple"] = is_stationary_simple

    print(
        f"Simple stationarity assessment: {'✅ Likely stationary' if is_stationary_simple else '❌ Likely non-stationary'}"
    )

    # Augmented Dickey-Fuller test if statsmodels is available
    if STATSMODELS_AVAILABLE:
        try:
            adf_result = adfuller(data.dropna())
            results["adf_statistic"] = adf_result[0]
            results["adf_pvalue"] = adf_result[1]
            results["adf_critical_values"] = adf_result[4]

            print(f"\n📊 Augmented Dickey-Fuller Test:")
            print(f"• ADF Statistic: {adf_result[0]:.4f}")
            print(f"• p-value: {adf_result[1]:.4f}")
            print(f"• Critical Values:")
            for key, value in adf_result[4].items():
                print(f"  - {key}: {value:.4f}")

            is_stationary_adf = adf_result[1] < 0.05
            results["is_stationary_adf"] = is_stationary_adf
            print(
                f"• Result: {'✅ Stationary' if is_stationary_adf else '❌ Non-stationary'} (p < 0.05)"
            )

        except Exception as e:
            print(f"⚠️ Could not perform ADF test: {e}")
            results["adf_error"] = str(e)

    # Differencing analysis
    if not results.get("is_stationary_simple", False):
        print(f"\n🔄 Differencing Analysis:")

        # First difference
        diff1 = data.diff().dropna()
        diff1_mean_stability = (
            np.std(
                [
                    diff1.iloc[i * chunk_size : (i + 1) * chunk_size].mean()
                    for i in range(len(diff1) // chunk_size)
                ]
            )
            / abs(diff1.mean())
            if len(diff1) > chunk_size
            else 0
        )

        print(f"• First difference mean stability: {diff1_mean_stability:.4f}")

        if diff1_mean_stability < 0.1:
            print("• ✅ First differencing makes series more stationary")
            results["differencing_needed"] = 1
        else:
            # Second difference
            diff2 = diff1.diff().dropna()
            if len(diff2) > chunk_size:
                diff2_mean_stability = (
                    np.std(
                        [
                            diff2.iloc[i * chunk_size : (i + 1) * chunk_size].mean()
                            for i in range(len(diff2) // chunk_size)
                        ]
                    )
                    / abs(diff2.mean())
                    if diff2.mean() != 0
                    else 0
                )
                print(f"• Second difference mean stability: {diff2_mean_stability:.4f}")

                if diff2_mean_stability < 0.1:
                    print("• ✅ Second differencing makes series more stationary")
                    results["differencing_needed"] = 2
                else:
                    print("• ⚠️ Series may need more complex transformation")
                    results["differencing_needed"] = "complex"

    return results


def simple_forecast_methods(
    data: pd.Series, forecast_periods: int = 30
) -> Dict[str, pd.Series]:
    """
    Apply simple forecasting methods to time series data.
    """
    print(f"\n🔮 Simple Forecasting Methods")
    print("=" * 50)

    forecasts = {}

    # 1. Naive forecast (last value)
    last_value = data.iloc[-1]
    naive_forecast = pd.Series(
        [last_value] * forecast_periods,
        index=pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq="D",
        ),
    )
    forecasts["Naive"] = naive_forecast
    print(f"• Naive forecast: {last_value:.2f} (constant)")

    # 2. Simple moving average
    window_sizes = [7, 14, 30]
    for window in window_sizes:
        if len(data) >= window:
            ma_value = data.tail(window).mean()
            ma_forecast = pd.Series(
                [ma_value] * forecast_periods, index=naive_forecast.index
            )
            forecasts[f"MA_{window}"] = ma_forecast
            print(f"• Moving Average ({window} periods): {ma_value:.2f}")

    # 3. Linear trend
    if len(data) >= 10:
        x = np.arange(len(data))
        y = data.values

        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        # Forecast future values
        future_x = np.arange(len(data), len(data) + forecast_periods)
        trend_forecast = intercept + slope * future_x

        trend_forecast_series = pd.Series(trend_forecast, index=naive_forecast.index)
        forecasts["Linear_Trend"] = trend_forecast_series
        print(f"• Linear trend: slope={slope:.4f}, intercept={intercept:.2f}")

    # 4. Seasonal naive (if enough data)
    if len(data) >= 14:  # At least 2 weeks of data
        seasonal_period = 7  # Weekly seasonality
        seasonal_values = []

        for i in range(forecast_periods):
            seasonal_idx = len(data) - seasonal_period + (i % seasonal_period)
            if seasonal_idx >= 0:
                seasonal_values.append(data.iloc[seasonal_idx])
            else:
                seasonal_values.append(data.iloc[-1])

        seasonal_forecast = pd.Series(seasonal_values, index=naive_forecast.index)
        forecasts["Seasonal_Naive"] = seasonal_forecast
        print(f"• Seasonal naive: using {seasonal_period}-period seasonality")

    # 5. Exponential smoothing (simple)
    if len(data) >= 3:
        alpha = 0.3  # Smoothing parameter

        # Calculate exponentially weighted average
        weights = np.array([(1 - alpha) ** i for i in range(len(data))])
        weights = weights / weights.sum()  # Normalize

        smoothed_value = np.sum(data.values[::-1] * weights)

        exp_smooth_forecast = pd.Series(
            [smoothed_value] * forecast_periods, index=naive_forecast.index
        )
        forecasts["Exponential_Smoothing"] = exp_smooth_forecast
        print(f"• Exponential smoothing (α={alpha}): {smoothed_value:.2f}")

    return forecasts


def advanced_forecast_methods(
    data: pd.Series, forecast_periods: int = 30
) -> Dict[str, pd.Series]:
    """
    Apply advanced forecasting methods using statsmodels.
    """
    print(f"\n🚀 Advanced Forecasting Methods")
    print("=" * 50)

    forecasts = {}

    if not STATSMODELS_AVAILABLE:
        print("⚠️ Statsmodels not available. Using simplified methods.")
        return simple_forecast_methods(data, forecast_periods)

    try:
        # 1. ARIMA Model
        print("Fitting ARIMA model...")

        # Simple ARIMA parameter selection
        best_aic = float("inf")
        best_order = (1, 1, 1)

        # Test different parameter combinations
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(data.dropna(), order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue

        # Fit best model
        arima_model = ARIMA(data.dropna(), order=best_order)
        arima_fitted = arima_model.fit()

        # Forecast
        arima_forecast = arima_fitted.forecast(steps=forecast_periods)

        future_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq="D",
        )
        forecasts["ARIMA"] = pd.Series(arima_forecast, index=future_index)

        print(f"• ARIMA{best_order}: AIC={best_aic:.2f}")

    except Exception as e:
        print(f"⚠️ ARIMA fitting failed: {e}")

    try:
        # 2. Exponential Smoothing (Holt-Winters)
        print("Fitting Exponential Smoothing...")

        # Determine if we have enough data for seasonal model
        seasonal_periods = min(12, len(data) // 3) if len(data) >= 24 else None

        if seasonal_periods and seasonal_periods >= 4:
            exp_smooth_model = ExponentialSmoothing(
                data.dropna(),
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods,
            )
        else:
            exp_smooth_model = ExponentialSmoothing(data.dropna(), trend="add")

        exp_smooth_fitted = exp_smooth_model.fit()
        exp_smooth_forecast = exp_smooth_fitted.forecast(forecast_periods)

        future_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq="D",
        )
        forecasts["Holt_Winters"] = pd.Series(exp_smooth_forecast, index=future_index)

        print(f"• Holt-Winters: {'Seasonal' if seasonal_periods else 'Non-seasonal'}")

    except Exception as e:
        print(f"⚠️ Exponential Smoothing fitting failed: {e}")

    return forecasts


def evaluate_forecasts(
    actual: pd.Series, forecasts: Dict[str, pd.Series], test_periods: int = 30
) -> pd.DataFrame:
    """
    Evaluate forecast accuracy using multiple metrics.
    """
    print(f"\n📊 Forecast Evaluation")
    print("=" * 50)

    if len(actual) < test_periods:
        print(
            f"⚠️ Not enough data for evaluation. Need at least {test_periods} periods."
        )
        return pd.DataFrame()

    # Split data for evaluation
    train_data = actual.iloc[:-test_periods]
    test_data = actual.iloc[-test_periods:]

    results = []

    for method_name, forecast_series in forecasts.items():
        if len(forecast_series) >= len(test_data):
            # Align forecast with test data
            forecast_values = forecast_series.iloc[: len(test_data)]

            # Calculate metrics
            mae = np.mean(np.abs(test_data.values - forecast_values.values))
            mse = np.mean((test_data.values - forecast_values.values) ** 2)
            rmse = np.sqrt(mse)

            # Mean Absolute Percentage Error
            mape = (
                np.mean(
                    np.abs(
                        (test_data.values - forecast_values.values) / test_data.values
                    )
                )
                * 100
            )

            # Symmetric Mean Absolute Percentage Error
            smape = (
                np.mean(
                    2
                    * np.abs(test_data.values - forecast_values.values)
                    / (np.abs(test_data.values) + np.abs(forecast_values.values))
                )
                * 100
            )

            results.append(
                {
                    "Method": method_name,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "SMAPE": smape,
                }
            )

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("RMSE")

        print("Forecast Accuracy (sorted by RMSE):")
        print(results_df.round(4))

        return results_df
    else:
        print("⚠️ No valid forecasts to evaluate.")
        return pd.DataFrame()


def run_time_series_challenges() -> None:
    """
    Run all time series analysis challenges.
    """
    print("🚀 Starting Level 6 Challenge 1: Time Series Analysis & Forecasting")
    print("=" * 60)

    try:
        # Challenge 1: Create time series datasets
        print("\n" + "=" * 50)
        print("CHALLENGE 1: Time Series Dataset Creation")
        print("=" * 50)

        datasets = create_time_series_datasets()

        print(f"\n✅ Created {len(datasets)} time series datasets:")
        for name, info in datasets.items():
            data = info["data"]
            print(f"• {name}: {len(data)} observations, frequency: {info['frequency']}")
            print(f"  Target: {info['target_column']}")
            print(f"  Period: {data.index[0]} to {data.index[-1]}")

        # Challenge 2: Time series analysis
        print("\n" + "=" * 50)
        print("CHALLENGE 2: Time Series Component Analysis")
        print("=" * 50)

        # Analyze each dataset
        for name, info in datasets.items():
            data = info["data"]
            target_col = info["target_column"]
            target_series = data[target_col]

            analyze_time_series_components(
                target_series, f"{name.title()} {target_col}"
            )

        # Challenge 3: Stationarity testing
        print("\n" + "=" * 50)
        print("CHALLENGE 3: Stationarity Analysis")
        print("=" * 50)

        stationarity_results = {}
        for name, info in datasets.items():
            data = info["data"]
            target_col = info["target_column"]
            target_series = data[target_col]

            results = test_stationarity(target_series, f"{name.title()} {target_col}")
            stationarity_results[name] = results

        # Challenge 4: Forecasting
        print("\n" + "=" * 50)
        print("CHALLENGE 4: Forecasting Methods")
        print("=" * 50)

        # Apply forecasting to sales data (most suitable for demonstration)
        sales_data = datasets["sales"]["data"]["sales"]

        print(f"Forecasting sales data ({len(sales_data)} observations)...")

        # Simple methods
        simple_forecasts = simple_forecast_methods(sales_data, forecast_periods=30)

        # Advanced methods (if available)
        advanced_forecasts = advanced_forecast_methods(sales_data, forecast_periods=30)

        # Combine all forecasts
        all_forecasts = {**simple_forecasts, **advanced_forecasts}

        # Visualize forecasts
        plt.figure(figsize=(15, 8))

        # Plot historical data (last 90 days)
        recent_data = sales_data.tail(90)
        plt.plot(
            recent_data.index,
            recent_data.values,
            label="Historical Data",
            linewidth=2,
            color="black",
        )

        # Plot forecasts
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink"]
        for i, (method, forecast) in enumerate(all_forecasts.items()):
            color = colors[i % len(colors)]
            plt.plot(
                forecast.index,
                forecast.values,
                label=method,
                linestyle="--",
                alpha=0.7,
                color=color,
            )

        plt.title("Sales Forecasting - Multiple Methods")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Challenge 5: Forecast evaluation
        print("\n" + "=" * 50)
        print("CHALLENGE 5: Forecast Evaluation")
        print("=" * 50)

        # Evaluate forecasts using walk-forward validation
        evaluation_results = evaluate_forecasts(
            sales_data, all_forecasts, test_periods=30
        )

        if not evaluation_results.empty:
            # Visualize evaluation results
            plt.figure(figsize=(12, 6))

            methods = evaluation_results["Method"]
            rmse_values = evaluation_results["RMSE"]
            mape_values = evaluation_results["MAPE"]

            x_pos = np.arange(len(methods))
            width = 0.35

            plt.subplot(1, 2, 1)
            plt.bar(x_pos, rmse_values)
            plt.set_title("Root Mean Square Error (RMSE)")
            plt.set_ylabel("RMSE")
            plt.set_xticks(x_pos)
            plt.set_xticklabels(methods, rotation=45, ha="right")
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.bar(x_pos, mape_values)
            plt.set_title("Mean Absolute Percentage Error (MAPE)")
            plt.set_ylabel("MAPE (%)")
            plt.set_xticks(x_pos)
            plt.set_xticklabels(methods, rotation=45, ha="right")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        print("\n" + "🎉" * 20)
        print("LEVEL 6 CHALLENGE 1 COMPLETE!")
        print("🎉" * 20)

        print("\n📚 What You've Learned:")
        print("• Time series components: trend, seasonality, noise")
        print("• Stationarity testing and transformation methods")
        print("• Multiple forecasting approaches (naive to advanced)")
        print("• Forecast evaluation metrics and validation")
        print("• Real-world time series patterns and challenges")

        print("\n🚀 Next Steps:")
        print("• Explore multivariate time series analysis")
        print("• Learn about state space models and Kalman filters")
        print("• Try deep learning approaches (LSTM, Prophet)")
        print("• Apply to real business forecasting problems")
        print("• Move to Level 6 Challenge 2: Anomaly Detection")

        return datasets

    except Exception as e:
        print(f"❌ Error in time series challenges: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete time series challenge
    datasets = run_time_series_challenges()

    if datasets:
        print("\n" + "=" * 60)
        print("TIME SERIES CHALLENGE SUMMARY")
        print("=" * 60)

        print("\nDatasets Created:")
        for name, info in datasets.items():
            data = info["data"]
            print(f"• {name}: {len(data)} observations ({info['frequency']} frequency)")

        print("\nKey Time Series Concepts Covered:")
        concepts = [
            "Time series decomposition and component analysis",
            "Stationarity testing and transformation methods",
            "Forecasting techniques from simple to advanced",
            "Model evaluation and accuracy metrics",
            "Real-world temporal patterns and seasonality",
            "Handling missing data and outliers in time series",
        ]

        for i, concept in enumerate(concepts, 1):
            print(f"{i}. {concept}")

        print("\n✨ Ready for Level 6 Challenge 2: Anomaly Detection!")
