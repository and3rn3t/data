# Level 6: Advanced Analytics Expert

## Challenge 1: Time Series Analysis and Forecasting

Master advanced time series analysis, forecasting techniques, and temporal pattern recognition for business and scientific applications.

### Objective

Learn sophisticated time series analysis methods including decomposition, stationarity testing, ARIMA modeling, seasonal forecasting, and anomaly detection in temporal data.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Advanced analytics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from scipy.signal import find_peaks
import itertools

print("📈 Time Series Analysis and Forecasting")
print("=" * 45)

# Set random seed for reproducibility
np.random.seed(42)

print("📊 Creating Comprehensive Time Series Datasets...")

# CHALLENGE 1: MULTI-SEASONAL TIME SERIES GENERATION
print("\n" + "=" * 60)
print("🕒 CHALLENGE 1: COMPLEX TIME SERIES GENERATION & ANALYSIS")
print("=" * 60)

def create_complex_time_series(start_date='2020-01-01', periods=1095, freq='D'):
    """Create realistic multi-seasonal time series with various patterns"""

    # Generate date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)

    # Base trend (growth with occasional dips)
    trend = np.linspace(100, 200, periods)
    # Add some trend breaks
    trend[365:730] *= 1.2  # Growth acceleration
    trend[730:] *= 0.95   # Market correction

    # Yearly seasonality (stronger in winter, weaker in summer)
    yearly_seasonal = 20 * np.sin(2 * np.pi * np.arange(periods) / 365.25 - np.pi/2)

    # Weekly seasonality (lower on weekends)
    weekly_seasonal = 15 * np.sin(2 * np.pi * np.arange(periods) / 7)

    # Monthly seasonality (end-of-month effects)
    monthly_seasonal = 10 * np.sin(2 * np.pi * np.arange(periods) / 30.44)

    # Holiday effects (simulate major holidays)
    holiday_effect = np.zeros(periods)
    for year in [2020, 2021, 2022]:
        # Christmas/New Year effect
        christmas_start = (datetime(year, 12, 20) - datetime(2020, 1, 1)).days
        new_year_end = (datetime(year+1, 1, 5) - datetime(2020, 1, 1)).days
        if christmas_start < periods:
            holiday_effect[christmas_start:min(new_year_end, periods)] += 30

        # Summer vacation effect (July-August)
        summer_start = (datetime(year, 7, 1) - datetime(2020, 1, 1)).days
        summer_end = (datetime(year, 8, 31) - datetime(2020, 1, 1)).days
        if summer_start < periods:
            holiday_effect[summer_start:min(summer_end, periods)] -= 15

    # Economic events (simulate market crashes, recoveries)
    economic_shocks = np.zeros(periods)
    # COVID-19 impact (March-May 2020)
    covid_start = (datetime(2020, 3, 15) - datetime(2020, 1, 1)).days
    covid_end = (datetime(2020, 6, 1) - datetime(2020, 1, 1)).days
    economic_shocks[covid_start:covid_end] = -40 * np.exp(-np.arange(covid_end-covid_start)/30)

    # Supply chain issues (2021)
    supply_start = (datetime(2021, 9, 1) - datetime(2020, 1, 1)).days
    supply_end = (datetime(2022, 3, 1) - datetime(2020, 1, 1)).days
    if supply_start < periods:
        supply_duration = min(supply_end, periods) - supply_start
        economic_shocks[supply_start:min(supply_end, periods)] = -20 * np.sin(
            np.pi * np.arange(supply_duration) / supply_duration
        )

    # Random noise with varying volatility
    base_noise = np.random.normal(0, 5, periods)
    volatility_multiplier = 1 + 0.3 * np.sin(2 * np.pi * np.arange(periods) / 100)
    noise = base_noise * volatility_multiplier

    # Combine all components
    ts_data = (trend + yearly_seasonal + weekly_seasonal + monthly_seasonal +
              holiday_effect + economic_shocks + noise)

    # Ensure no negative values (representing sales/revenue)
    ts_data = np.maximum(ts_data, 10)

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': ts_data,
        'trend': trend,
        'yearly_seasonal': yearly_seasonal,
        'weekly_seasonal': weekly_seasonal,
        'monthly_seasonal': monthly_seasonal,
        'holiday_effect': holiday_effect,
        'economic_shocks': economic_shocks,
        'noise': noise
    })

    df.set_index('date', inplace=True)

    return df

# Generate primary time series (daily sales data)
print("Creating primary time series (Daily Sales)...")
sales_ts = create_complex_time_series('2020-01-01', 1095, 'D')

print(f"Time series shape: {sales_ts.shape}")
print(f"Date range: {sales_ts.index[0]} to {sales_ts.index[-1]}")
print(f"Value range: {sales_ts['value'].min():.2f} to {sales_ts['value'].max():.2f}")

# Create additional related time series
print("\nCreating additional time series...")

# Website traffic (leads sales by 1-2 days)
traffic_ts = sales_ts['value'].shift(-2) * (1.5 + 0.3 * np.random.randn(len(sales_ts)))
traffic_ts = traffic_ts.fillna(method='bfill')

# Marketing spend (affects sales with 3-7 day lag)
marketing_base = 1000 + 200 * np.sin(2 * np.pi * np.arange(len(sales_ts)) / 365)
marketing_noise = 100 * np.random.randn(len(sales_ts))
marketing_ts = marketing_base + marketing_noise

# Weather impact (temperature affects sales)
temperature_ts = 20 + 15 * np.sin(2 * np.pi * np.arange(len(sales_ts)) / 365.25) + 3 * np.random.randn(len(sales_ts))

# Combine into comprehensive dataset
comprehensive_ts = pd.DataFrame({
    'sales': sales_ts['value'],
    'traffic': traffic_ts,
    'marketing_spend': marketing_ts,
    'temperature': temperature_ts
}, index=sales_ts.index)

print("🔍 Time Series Exploration and Visualization")

# Basic time series statistics
print(f"\nTime Series Statistics:")
print(comprehensive_ts.describe())

# Visualize the comprehensive time series
plt.figure(figsize=(20, 16))

# Main time series with components
plt.subplot(3, 4, 1)
plt.plot(sales_ts.index, sales_ts['value'], alpha=0.8, linewidth=1)
plt.title('Complete Sales Time Series')
plt.ylabel('Sales Value')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Decompose into components
plt.subplot(3, 4, 2)
plt.plot(sales_ts.index, sales_ts['trend'], label='Trend', alpha=0.8)
plt.plot(sales_ts.index, sales_ts['yearly_seasonal'], label='Yearly', alpha=0.8)
plt.plot(sales_ts.index, sales_ts['weekly_seasonal'], label='Weekly', alpha=0.8)
plt.title('Time Series Components')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Seasonality patterns
plt.subplot(3, 4, 3)
monthly_avg = comprehensive_ts.groupby(comprehensive_ts.index.month)['sales'].mean()
plt.bar(range(1, 13), monthly_avg.values, alpha=0.7)
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.title('Monthly Seasonality')
plt.xticks(range(1, 13))
plt.grid(axis='y', alpha=0.3)

# Weekly patterns
plt.subplot(3, 4, 4)
weekly_avg = comprehensive_ts.groupby(comprehensive_ts.index.dayofweek)['sales'].mean()
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.bar(range(7), weekly_avg.values, alpha=0.7)
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.title('Weekly Seasonality')
plt.xticks(range(7), weekdays, rotation=45)
plt.grid(axis='y', alpha=0.3)

# All time series together
plt.subplot(3, 4, 5)
for col in comprehensive_ts.columns:
    # Normalize for comparison
    normalized = (comprehensive_ts[col] - comprehensive_ts[col].mean()) / comprehensive_ts[col].std()
    plt.plot(comprehensive_ts.index[::30], normalized[::30], label=col, alpha=0.7)  # Every 30th point for clarity
plt.title('Normalized Multiple Time Series')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Rolling statistics
plt.subplot(3, 4, 6)
rolling_mean = comprehensive_ts['sales'].rolling(window=30).mean()
rolling_std = comprehensive_ts['sales'].rolling(window=30).std()
plt.plot(comprehensive_ts.index, comprehensive_ts['sales'], alpha=0.5, label='Sales')
plt.plot(comprehensive_ts.index, rolling_mean, label='30-day Mean', linewidth=2)
plt.fill_between(comprehensive_ts.index,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.3, label='±1 Std')
plt.title('Rolling Statistics')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# CHALLENGE 2: STATIONARITY TESTING AND TRANSFORMATION
print("\n" + "=" * 60)
print("📊 CHALLENGE 2: STATIONARITY TESTING & TRANSFORMATION")
print("=" * 60)

def perform_stationarity_tests(ts_data, series_name="Time Series"):
    """Perform comprehensive stationarity tests"""

    results = {}

    # Augmented Dickey-Fuller test
    adf_result = adfuller(ts_data.dropna())
    results['ADF'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < 0.05
    }

    # KPSS test
    kpss_result = kpss(ts_data.dropna(), regression='ct')
    results['KPSS'] = {
        'statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'critical_values': kpss_result[3],
        'is_stationary': kpss_result[1] > 0.05
    }

    print(f"\n🔍 Stationarity Tests for {series_name}:")
    print(f"ADF Test: Statistic={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")
    print(f"  Stationary: {results['ADF']['is_stationary']}")

    print(f"KPSS Test: Statistic={kpss_result[0]:.4f}, p-value={kpss_result[1]:.4f}")
    print(f"  Stationary: {results['KPSS']['is_stationary']}")

    return results

# Test original series
sales_stationarity = perform_stationarity_tests(comprehensive_ts['sales'], "Original Sales")

# Apply transformations to achieve stationarity
print("\n🔄 Applying Transformations for Stationarity:")

# 1. First difference
sales_diff1 = comprehensive_ts['sales'].diff().dropna()
diff1_stationarity = perform_stationarity_tests(sales_diff1, "First Difference")

# 2. Log transformation + first difference
sales_log = np.log(comprehensive_ts['sales'])
sales_log_diff = sales_log.diff().dropna()
log_diff_stationarity = perform_stationarity_tests(sales_log_diff, "Log + First Difference")

# 3. Seasonal difference
sales_seasonal_diff = comprehensive_ts['sales'].diff(7).dropna()  # Weekly seasonal difference
seasonal_diff_stationarity = perform_stationarity_tests(sales_seasonal_diff, "Seasonal Difference (Weekly)")

# Plot transformations
plt.subplot(3, 4, 7)
plt.plot(comprehensive_ts.index[-365:], comprehensive_ts['sales'][-365:], label='Original')
plt.plot(comprehensive_ts.index[-364:], sales_diff1[-364:], label='First Diff', alpha=0.7)
plt.title('Stationarity Transformations')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# ACF and PACF plots for model identification
plt.subplot(3, 4, 8)
# Use the stationary series (first difference)
stationary_series = sales_diff1[-200:]  # Last 200 observations
autocorr = [stationary_series.autocorr(lag=i) for i in range(1, 21)]
plt.bar(range(1, 21), autocorr, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('ACF (First 20 lags)')
plt.grid(True, alpha=0.3)

# CHALLENGE 3: ADVANCED FORECASTING MODELS
print("\n" + "=" * 60)
print("🔮 CHALLENGE 3: ADVANCED FORECASTING MODELS")
print("=" * 60)

# Prepare data for modeling
train_size = int(0.8 * len(comprehensive_ts))
train_data = comprehensive_ts[:train_size]
test_data = comprehensive_ts[train_size:]

print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

# Dictionary to store model results
model_results = {}

print("\n🎯 Model 1: ARIMA Modeling")

def find_best_arima_order(ts_data, max_p=5, max_d=2, max_q=5):
    """Find best ARIMA parameters using AIC"""

    best_aic = float('inf')
    best_order = None
    best_model = None

    # Grid search for best parameters
    for p, d, q in itertools.product(range(max_p+1), range(max_d+1), range(max_q+1)):
        try:
            model = ARIMA(ts_data, order=(p, d, q))
            fitted_model = model.fit()

            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_order = (p, d, q)
                best_model = fitted_model

        except Exception:
            continue

    return best_order, best_model, best_aic

# Find best ARIMA model
print("Searching for optimal ARIMA parameters...")
best_arima_order, best_arima_model, best_aic = find_best_arima_order(
    train_data['sales'], max_p=3, max_d=2, max_q=3
)

print(f"Best ARIMA order: {best_arima_order}")
print(f"Best AIC: {best_aic:.2f}")

# Generate ARIMA forecasts
arima_forecast = best_arima_model.forecast(steps=len(test_data))
arima_forecast_index = test_data.index

# Calculate ARIMA performance
arima_mae = mean_absolute_error(test_data['sales'], arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test_data['sales'], arima_forecast))
arima_mape = np.mean(np.abs((test_data['sales'] - arima_forecast) / test_data['sales'])) * 100

model_results['ARIMA'] = {
    'forecast': arima_forecast,
    'mae': arima_mae,
    'rmse': arima_rmse,
    'mape': arima_mape,
    'model': best_arima_model
}

print(f"ARIMA Performance - MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.2f}%")

print("\n🎯 Model 2: Exponential Smoothing (Holt-Winters)")

# Holt-Winters model
hw_model = ExponentialSmoothing(
    train_data['sales'],
    trend='add',
    seasonal='add',
    seasonal_periods=7  # Weekly seasonality
)

hw_fitted = hw_model.fit()
hw_forecast = hw_fitted.forecast(steps=len(test_data))

# Calculate Holt-Winters performance
hw_mae = mean_absolute_error(test_data['sales'], hw_forecast)
hw_rmse = np.sqrt(mean_squared_error(test_data['sales'], hw_forecast))
hw_mape = np.mean(np.abs((test_data['sales'] - hw_forecast) / test_data['sales'])) * 100

model_results['Holt_Winters'] = {
    'forecast': hw_forecast,
    'mae': hw_mae,
    'rmse': hw_rmse,
    'mape': hw_mape,
    'model': hw_fitted
}

print(f"Holt-Winters Performance - MAE: {hw_mae:.2f}, RMSE: {hw_rmse:.2f}, MAPE: {hw_mape:.2f}%")

print("\n🎯 Model 3: Vector Autoregression (VAR)")

# Prepare multivariate data for VAR
var_data = train_data[['sales', 'traffic', 'marketing_spend']].dropna()

# Check stationarity and difference if needed
var_data_diff = var_data.diff().dropna()

# Fit VAR model
var_model = VAR(var_data_diff)
var_lag_order = var_model.select_order(maxlags=10)
optimal_lag = var_lag_order.aic
print(f"Optimal VAR lag order: {optimal_lag}")

var_fitted = var_model.fit(optimal_lag)

# Generate VAR forecasts
var_forecast_diff = var_fitted.forecast(var_data_diff.values, steps=len(test_data))

# Convert back to levels (reverse differencing)
last_levels = var_data.iloc[-1].values
var_forecast_levels = np.zeros_like(var_forecast_diff)
var_forecast_levels[0] = last_levels + var_forecast_diff[0]

for i in range(1, len(var_forecast_diff)):
    var_forecast_levels[i] = var_forecast_levels[i-1] + var_forecast_diff[i]

var_sales_forecast = var_forecast_levels[:, 0]  # Sales column

# Calculate VAR performance
var_mae = mean_absolute_error(test_data['sales'], var_sales_forecast)
var_rmse = np.sqrt(mean_squared_error(test_data['sales'], var_sales_forecast))
var_mape = np.mean(np.abs((test_data['sales'] - var_sales_forecast) / test_data['sales'])) * 100

model_results['VAR'] = {
    'forecast': var_sales_forecast,
    'mae': var_mae,
    'rmse': var_rmse,
    'mape': var_mape,
    'model': var_fitted
}

print(f"VAR Performance - MAE: {var_mae:.2f}, RMSE: {var_rmse:.2f}, MAPE: {var_mape:.2f}%")

print("\n🎯 Model 4: Machine Learning (Random Forest)")

# Feature engineering for ML model
def create_time_features(df):
    """Create time-based features"""
    features = pd.DataFrame(index=df.index)

    # Date features
    features['year'] = df.index.year
    features['month'] = df.index.month
    features['day'] = df.index.day
    features['dayofweek'] = df.index.dayofweek
    features['quarter'] = df.index.quarter
    features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Lag features
    for lag in [1, 7, 14, 30]:
        features[f'sales_lag_{lag}'] = df['sales'].shift(lag)

    # Rolling features
    for window in [7, 14, 30]:
        features[f'sales_rolling_mean_{window}'] = df['sales'].rolling(window).mean()
        features[f'sales_rolling_std_{window}'] = df['sales'].rolling(window).std()

    # External features
    if 'traffic' in df.columns:
        features['traffic'] = df['traffic']
        features['marketing_spend'] = df['marketing_spend']
        features['temperature'] = df['temperature']

    return features

# Create features for ML model
ml_features = create_time_features(comprehensive_ts)
ml_features = ml_features.dropna()

# Prepare ML train/test sets
ml_train_size = int(0.8 * len(ml_features))
X_train_ml = ml_features[:ml_train_size].drop(columns=[col for col in ml_features.columns if col.startswith('sales_')], errors='ignore')
X_train_ml = ml_features[:ml_train_size]
y_train_ml = comprehensive_ts.loc[X_train_ml.index, 'sales']

X_test_ml = ml_features[ml_train_size:]
y_test_ml = comprehensive_ts.loc[X_test_ml.index, 'sales']

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_ml, y_train_ml)

rf_forecast = rf_model.predict(X_test_ml)

# Calculate RF performance
rf_mae = mean_absolute_error(y_test_ml, rf_forecast)
rf_rmse = np.sqrt(mean_squared_error(y_test_ml, rf_forecast))
rf_mape = np.mean(np.abs((y_test_ml - rf_forecast) / y_test_ml)) * 100

model_results['Random_Forest'] = {
    'forecast': rf_forecast,
    'mae': rf_mae,
    'rmse': rf_rmse,
    'mape': rf_mape,
    'model': rf_model
}

print(f"Random Forest Performance - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, MAPE: {rf_mape:.2f}%")

# Plot forecasting results
plt.subplot(3, 4, 9)
plt.plot(train_data.index[-100:], train_data['sales'][-100:], label='Training', alpha=0.7)
plt.plot(test_data.index, test_data['sales'], label='Actual', linewidth=2)
plt.plot(test_data.index, model_results['ARIMA']['forecast'], label='ARIMA', alpha=0.8)
plt.plot(test_data.index, model_results['Holt_Winters']['forecast'], label='Holt-Winters', alpha=0.8)
plt.title('Forecasting Results Comparison')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# CHALLENGE 4: ANOMALY DETECTION IN TIME SERIES
print("\n" + "=" * 60)
print("🚨 CHALLENGE 4: TIME SERIES ANOMALY DETECTION")
print("=" * 60)

print("🔍 Multiple Anomaly Detection Techniques")

# Method 1: Statistical Outlier Detection (Z-score)
def detect_statistical_anomalies(ts_data, window=30, threshold=3):
    """Detect anomalies using rolling z-score"""

    rolling_mean = ts_data.rolling(window=window).mean()
    rolling_std = ts_data.rolling(window=window).std()

    z_scores = np.abs((ts_data - rolling_mean) / rolling_std)
    anomalies = z_scores > threshold

    return anomalies, z_scores

stat_anomalies, z_scores = detect_statistical_anomalies(comprehensive_ts['sales'])

print(f"Statistical anomalies detected: {stat_anomalies.sum()}")

# Method 2: Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)

# Use recent data for training
recent_data = comprehensive_ts['sales'][-500:].values.reshape(-1, 1)
iso_forest.fit(recent_data)

# Detect anomalies in full dataset
iso_anomalies = iso_forest.predict(comprehensive_ts['sales'].values.reshape(-1, 1))
iso_anomalies_bool = iso_anomalies == -1

print(f"Isolation Forest anomalies detected: {iso_anomalies_bool.sum()}")

# Method 3: Residual-based Detection (using ARIMA residuals)
arima_residuals = best_arima_model.resid
residual_threshold = 2.5 * np.std(arima_residuals)
residual_anomalies = np.abs(arima_residuals) > residual_threshold

print(f"Residual-based anomalies detected: {residual_anomalies.sum()}")

# Method 4: Seasonal Decomposition Anomaly Detection
def detect_decomposition_anomalies(ts_data, period=7, threshold=2.5):
    """Detect anomalies in seasonal decomposition residuals"""

    decomposition = seasonal_decompose(ts_data, model='additive', period=period)
    residual_std = np.std(decomposition.resid.dropna())

    anomalies = np.abs(decomposition.resid) > threshold * residual_std

    return anomalies.fillna(False), decomposition

decomp_anomalies, decomposition = detect_decomposition_anomalies(comprehensive_ts['sales'])

print(f"Decomposition-based anomalies detected: {decomp_anomalies.sum()}")

# Combine anomaly detection results
combined_anomalies = (stat_anomalies | iso_anomalies_bool |
                     decomp_anomalies.reindex(comprehensive_ts.index, fill_value=False))

print(f"Total unique anomalies detected: {combined_anomalies.sum()}")

# Plot anomaly detection results
plt.subplot(3, 4, 10)
plt.plot(comprehensive_ts.index, comprehensive_ts['sales'], alpha=0.7, label='Sales')
anomaly_points = comprehensive_ts[combined_anomalies]
plt.scatter(anomaly_points.index, anomaly_points['sales'], color='red', s=20, label='Anomalies', alpha=0.7)
plt.title('Anomaly Detection Results')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# CHALLENGE 5: ADVANCED SEASONAL ANALYSIS
print("\n" + "=" * 60)
print("🌊 CHALLENGE 5: ADVANCED SEASONAL ANALYSIS")
print("=" * 60)

print("📊 Multi-level Seasonal Pattern Analysis")

# Seasonal decomposition with different periods
decompositions = {}
periods = [7, 30, 365]  # Weekly, Monthly, Yearly

for period in periods:
    if len(comprehensive_ts) >= 2 * period:
        try:
            decomp = seasonal_decompose(comprehensive_ts['sales'],
                                      model='additive', period=period)
            decompositions[f'period_{period}'] = decomp
        except Exception as e:
            print(f"Could not decompose with period {period}: {e}")

# Analyze seasonal strength
def calculate_seasonal_strength(decomposition):
    """Calculate the strength of seasonality"""

    seasonal_var = np.var(decomposition.seasonal.dropna())
    residual_var = np.var(decomposition.resid.dropna())

    if residual_var > 0:
        seasonal_strength = seasonal_var / (seasonal_var + residual_var)
        return seasonal_strength
    return 0

seasonal_strengths = {}
for period_name, decomp in decompositions.items():
    strength = calculate_seasonal_strength(decomp)
    seasonal_strengths[period_name] = strength
    print(f"Seasonal strength for {period_name}: {strength:.4f}")

# Fourier analysis for frequency domain patterns
print("\n🌈 Fourier Analysis for Frequency Patterns")

from scipy.fft import fft, fftfreq

# Apply FFT to detect dominant frequencies
sales_values = comprehensive_ts['sales'].values
n = len(sales_values)
yf = fft(sales_values)
xf = fftfreq(n, 1)[:n//2]

# Find dominant frequencies
dominant_freqs = np.argsort(np.abs(yf[:n//2]))[-10:]  # Top 10 frequencies
print("Dominant frequency periods (days):")
for freq_idx in dominant_freqs[-5:]:  # Top 5
    if xf[freq_idx] > 0:
        period = 1 / xf[freq_idx]
        print(f"  Period: {period:.1f} days, Amplitude: {np.abs(yf[freq_idx]):.1f}")

# Plot decomposition for weekly pattern
if 'period_7' in decompositions:
    plt.subplot(3, 4, 11)
    decomp_7 = decompositions['period_7']
    plt.plot(comprehensive_ts.index[-365:], decomp_7.seasonal[-365:])
    plt.title('Weekly Seasonal Pattern')
    plt.ylabel('Seasonal Component')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

# Model performance comparison
plt.subplot(3, 4, 12)
model_names = list(model_results.keys())
mae_scores = [model_results[model]['mae'] for model in model_names]
rmse_scores = [model_results[model]['rmse'] for model in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

plt.bar(x_pos - width/2, mae_scores, width, label='MAE', alpha=0.7)
plt.bar(x_pos + width/2, rmse_scores, width, label='RMSE', alpha=0.7)

plt.xlabel('Models')
plt.ylabel('Error')
plt.title('Forecasting Model Performance')
plt.xticks(x_pos, model_names, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# CHALLENGE 6: TIME SERIES CROSS-VALIDATION
print("\n" + "=" * 60)
print("⏰ CHALLENGE 6: TIME SERIES CROSS-VALIDATION")
print("=" * 60)

print("🔄 Implementing Time Series Cross-Validation")

def time_series_cv_score(model_func, ts_data, n_splits=5, test_size=30):
    """Perform time series cross-validation"""

    scores = []
    n_samples = len(ts_data)

    for i in range(n_splits):
        # Calculate split points
        test_end = n_samples - i * test_size
        test_start = test_end - test_size
        train_end = test_start

        if train_end <= test_size:  # Need minimum training data
            break

        # Split data
        train_data = ts_data[:train_end]
        test_data = ts_data[test_start:test_end]

        try:
            # Train model and forecast
            forecast = model_func(train_data, len(test_data))

            # Calculate performance
            mae = mean_absolute_error(test_data, forecast)
            scores.append(mae)

        except Exception as e:
            print(f"Error in CV fold {i}: {e}")

    return scores

# Define model functions for CV
def arima_model_func(train_data, forecast_steps):
    best_order, best_model, _ = find_best_arima_order(train_data, max_p=2, max_d=1, max_q=2)
    return best_model.forecast(steps=forecast_steps)

def hw_model_func(train_data, forecast_steps):
    hw_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=7)
    hw_fitted = hw_model.fit()
    return hw_fitted.forecast(steps=forecast_steps)

# Perform cross-validation
print("Performing time series cross-validation...")

arima_cv_scores = time_series_cv_score(arima_model_func, comprehensive_ts['sales'], n_splits=5, test_size=30)
hw_cv_scores = time_series_cv_score(hw_model_func, comprehensive_ts['sales'], n_splits=5, test_size=30)

print(f"ARIMA CV MAE: {np.mean(arima_cv_scores):.2f} ± {np.std(arima_cv_scores):.2f}")
print(f"Holt-Winters CV MAE: {np.mean(hw_cv_scores):.2f} ± {np.std(hw_cv_scores):.2f}")

print("\n" + "=" * 60)
print("📈 TIME SERIES ANALYSIS INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

# Find best performing model
best_model_name = min(model_results.keys(), key=lambda k: model_results[k]['mae'])
best_model_performance = model_results[best_model_name]

print("📋 Key Findings:")
print(f"1. Best Forecasting Model: {best_model_name}")
print(f"   • MAE: {best_model_performance['mae']:.2f}")
print(f"   • RMSE: {best_model_performance['rmse']:.2f}")
print(f"   • MAPE: {best_model_performance['mape']:.2f}%")

print(f"\n2. Seasonality Analysis:")
for period_name, strength in seasonal_strengths.items():
    period = period_name.split('_')[1]
    print(f"   • {period}-day seasonality strength: {strength:.4f}")

print(f"\n3. Anomaly Detection:")
print(f"   • Statistical anomalies: {stat_anomalies.sum()}")
print(f"   • ML-based anomalies: {iso_anomalies_bool.sum()}")
print(f"   • Total unique anomalies: {combined_anomalies.sum()}")

print(f"\n4. Stationarity:")
print(f"   • Original series stationary: {sales_stationarity['ADF']['is_stationary']}")
print(f"   • First difference stationary: {diff1_stationarity['ADF']['is_stationary']}")
print(f"   • Recommended transformation: First differencing")

print(f"\n🎯 Business Recommendations:")
print("• Use ensemble of ARIMA and Holt-Winters for robust forecasting")
print("• Monitor for anomalies using multiple detection methods")
print("• Account for weekly and seasonal patterns in planning")
print("• Implement real-time model updates based on new data")
print("• Consider external factors (marketing, weather) in multivariate models")

print(f"\n📊 Technical Best Practices:")
print("• Always test for stationarity before ARIMA modeling")
print("• Use time series cross-validation for model selection")
print("• Combine statistical and ML approaches for better accuracy")
print("• Monitor forecast accuracy and retrain models regularly")
print("• Use residual analysis for model diagnostics")

print(f"\n🔧 Model Selection Guidelines:")
print("• ARIMA: Best for univariate series with clear patterns")
print("• Holt-Winters: Excellent for series with strong seasonality")
print("• VAR: Use when multiple related time series are available")
print("• ML models: Best when many external features are available")
print("• Ensemble: Combine methods for robust predictions")

print("\n✅ Time Series Analysis and Forecasting Challenge Completed!")
print("What you've mastered:")
print("• Advanced time series generation with multiple seasonal patterns")
print("• Comprehensive stationarity testing and transformation techniques")
print("• Multiple forecasting models (ARIMA, Holt-Winters, VAR, ML)")
print("• Sophisticated anomaly detection methods")
print("• Advanced seasonal analysis and frequency domain techniques")
print("• Time series cross-validation and model selection")

print(f"\n📈 You are now a Time Series Expert! Ready for advanced analytics!")
```

### Success Criteria

- Generate and analyze complex multi-seasonal time series data
- Master stationarity testing and appropriate transformations
- Implement multiple forecasting models and compare performance
- Develop sophisticated anomaly detection systems
- Perform advanced seasonal analysis using multiple techniques
- Apply proper time series cross-validation methods

### Learning Objectives

- Understand complex time series patterns and generation techniques
- Master statistical tests for stationarity and model assumptions
- Learn advanced forecasting methods including ARIMA, VAR, and ML approaches
- Practice comprehensive anomaly detection in temporal data
- Develop skills in seasonal analysis and frequency domain methods
- Build robust time series analysis and forecasting pipelines

---

_Pro tip: Time series analysis requires understanding the underlying patterns - always decompose, test assumptions, and validate with proper cross-validation!_
