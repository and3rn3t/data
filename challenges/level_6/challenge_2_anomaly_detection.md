# Level 6: Advanced Analytics Expert

## Challenge 2: Anomaly Detection and Outlier Analysis

Master advanced anomaly detection techniques, outlier analysis methods, and automated monitoring systems for data quality and fraud detection.

### Objective

Learn sophisticated anomaly detection algorithms including statistical methods, machine learning approaches, and ensemble techniques for identifying unusual patterns in complex datasets.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Anomaly detection libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

# Statistical libraries
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.special import erfinv
import itertools
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.tsa.seasonal import seasonal_decompose

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("🔍 Advanced Anomaly Detection and Outlier Analysis")
print("=" * 50)

# Set random seed for reproducibility
np.random.seed(42)

print("🎲 Creating Comprehensive Anomaly Detection Datasets...")

# CHALLENGE 1: MULTI-DIMENSIONAL ANOMALY GENERATION
print("\n" + "=" * 60)
print("🔮 CHALLENGE 1: COMPLEX ANOMALY DATA GENERATION")
print("=" * 60)

def generate_anomaly_datasets():
    """Generate multiple datasets with different types of anomalies"""

    datasets = {}

    # Dataset 1: Financial Transaction Data
    print("Creating financial transaction dataset...")
    n_transactions = 10000
    n_anomalies = 200

    # Normal transactions
    normal_amounts = np.random.lognormal(mean=3, sigma=1, size=n_transactions-n_anomalies)
    normal_frequency = np.random.poisson(lam=5, size=n_transactions-n_anomalies)
    normal_hour = np.random.choice(range(6, 23), size=n_transactions-n_anomalies,
                                 p=np.array([0.05, 0.08, 0.12, 0.15, 0.15, 0.15,
                                           0.12, 0.08, 0.05, 0.02, 0.01, 0.01, 0.01,
                                           0.01, 0.01, 0.01, 0.01]))
    normal_merchant_type = np.random.choice(['grocery', 'restaurant', 'gas', 'retail', 'online'],
                                          size=n_transactions-n_anomalies,
                                          p=[0.3, 0.25, 0.15, 0.2, 0.1])

    # Anomalous transactions (fraud)
    fraud_amounts = np.concatenate([
        np.random.lognormal(mean=6, sigma=0.5, size=n_anomalies//4),  # Large amounts
        np.random.uniform(0.01, 1, size=n_anomalies//4),  # Micro transactions
        np.random.lognormal(mean=3, sigma=1, size=n_anomalies//2)  # Normal amounts but other patterns
    ])
    fraud_frequency = np.concatenate([
        np.random.poisson(lam=20, size=n_anomalies//2),  # High frequency
        np.random.poisson(lam=5, size=n_anomalies//2)   # Normal frequency
    ])
    fraud_hour = np.concatenate([
        np.random.choice(range(0, 6), size=n_anomalies//3),  # Late night
        np.random.choice(range(6, 23), size=2*n_anomalies//3)  # Normal hours
    ])
    fraud_merchant_type = np.random.choice(['grocery', 'restaurant', 'gas', 'retail', 'online', 'atm'],
                                         size=n_anomalies, p=[0.1, 0.1, 0.1, 0.2, 0.4, 0.1])

    # Combine data
    amounts = np.concatenate([normal_amounts, fraud_amounts])
    frequencies = np.concatenate([normal_frequency, fraud_frequency])
    hours = np.concatenate([normal_hour, fraud_hour])
    merchant_types = np.concatenate([normal_merchant_type, fraud_merchant_type])

    # Create additional features
    distances = np.concatenate([
        np.random.gamma(2, 2, size=n_transactions-n_anomalies),  # Normal distances
        np.concatenate([
            np.random.gamma(10, 5, size=n_anomalies//2),  # Far from home
            np.random.gamma(2, 2, size=n_anomalies//2)    # Normal distances
        ])
    ])

    labels = np.concatenate([np.zeros(n_transactions-n_anomalies), np.ones(n_anomalies)])

    # Shuffle data
    shuffle_idx = np.random.permutation(n_transactions)

    financial_data = pd.DataFrame({
        'amount': amounts[shuffle_idx],
        'frequency_last_hour': frequencies[shuffle_idx],
        'hour_of_day': hours[shuffle_idx],
        'distance_from_home': distances[shuffle_idx],
        'merchant_type_encoded': pd.Categorical(merchant_types[shuffle_idx]).codes,
        'is_anomaly': labels[shuffle_idx].astype(int)
    })

    # Add derived features
    financial_data['amount_log'] = np.log1p(financial_data['amount'])
    financial_data['is_weekend'] = (pd.date_range('2024-01-01', periods=len(financial_data), freq='H').dayofweek >= 5).astype(int)

    datasets['financial'] = financial_data

    # Dataset 2: Network Intrusion Data
    print("Creating network intrusion dataset...")
    n_connections = 8000
    n_intrusions = 160

    # Normal network connections
    normal_duration = np.random.exponential(scale=30, size=n_connections-n_intrusions)
    normal_src_bytes = np.random.lognormal(mean=8, sigma=2, size=n_connections-n_intrusions)
    normal_dst_bytes = np.random.lognormal(mean=7, sigma=1.5, size=n_connections-n_intrusions)
    normal_failed_logins = np.random.poisson(lam=0.1, size=n_connections-n_intrusions)
    normal_num_compromised = np.zeros(n_connections-n_intrusions)

    # Intrusion attempts
    intrusion_duration = np.concatenate([
        np.random.exponential(scale=1, size=n_intrusions//3),    # Very short
        np.random.exponential(scale=300, size=n_intrusions//3),  # Very long
        np.random.exponential(scale=30, size=n_intrusions//3)    # Normal
    ])
    intrusion_src_bytes = np.concatenate([
        np.random.lognormal(mean=12, sigma=1, size=n_intrusions//2),  # Large transfers
        np.random.lognormal(mean=8, sigma=2, size=n_intrusions//2)    # Normal size
    ])
    intrusion_dst_bytes = np.concatenate([
        np.random.lognormal(mean=10, sigma=1, size=n_intrusions//2),
        np.random.lognormal(mean=7, sigma=1.5, size=n_intrusions//2)
    ])
    intrusion_failed_logins = np.random.poisson(lam=5, size=n_intrusions)
    intrusion_num_compromised = np.random.poisson(lam=2, size=n_intrusions)

    # Combine network data
    durations = np.concatenate([normal_duration, intrusion_duration])
    src_bytes = np.concatenate([normal_src_bytes, intrusion_src_bytes])
    dst_bytes = np.concatenate([normal_dst_bytes, intrusion_dst_bytes])
    failed_logins = np.concatenate([normal_failed_logins, intrusion_failed_logins])
    num_compromised = np.concatenate([normal_num_compromised, intrusion_num_compromised])
    labels_network = np.concatenate([np.zeros(n_connections-n_intrusions), np.ones(n_intrusions)])

    # Shuffle
    shuffle_idx_net = np.random.permutation(n_connections)

    network_data = pd.DataFrame({
        'duration': durations[shuffle_idx_net],
        'src_bytes': src_bytes[shuffle_idx_net],
        'dst_bytes': dst_bytes[shuffle_idx_net],
        'failed_logins': failed_logins[shuffle_idx_net],
        'num_compromised': num_compromised[shuffle_idx_net],
        'is_anomaly': labels_network[shuffle_idx_net].astype(int)
    })

    # Add derived features
    network_data['bytes_ratio'] = network_data['src_bytes'] / (network_data['dst_bytes'] + 1)
    network_data['duration_log'] = np.log1p(network_data['duration'])

    datasets['network'] = network_data

    # Dataset 3: IoT Sensor Data (Equipment Monitoring)
    print("Creating IoT sensor dataset...")
    n_readings = 5000
    n_faults = 100

    # Normal sensor readings
    time_stamps = pd.date_range('2024-01-01', periods=n_readings, freq='5T')

    # Generate realistic sensor patterns with daily and weekly cycles
    time_numeric = np.arange(n_readings)
    daily_cycle = 5 * np.sin(2 * np.pi * time_numeric / (24 * 12))  # 12 readings per hour
    weekly_cycle = 2 * np.sin(2 * np.pi * time_numeric / (7 * 24 * 12))

    normal_temp = 75 + daily_cycle[:-n_faults] + weekly_cycle[:-n_faults] + np.random.normal(0, 2, n_readings-n_faults)
    normal_pressure = 100 + 0.5 * daily_cycle[:-n_faults] + np.random.normal(0, 1, n_readings-n_faults)
    normal_vibration = 0.5 + 0.1 * np.abs(daily_cycle[:-n_faults]) + np.random.normal(0, 0.1, n_readings-n_faults)
    normal_power = 50 + 5 * daily_cycle[:-n_faults] + np.random.normal(0, 1, n_readings-n_faults)

    # Faulty sensor readings (equipment failures)
    fault_temp = np.concatenate([
        np.random.normal(120, 10, n_faults//3),  # Overheating
        np.random.normal(40, 5, n_faults//3),   # Cooling issues
        np.random.normal(75, 15, n_faults//3)   # Erratic readings
    ])
    fault_pressure = np.concatenate([
        np.random.normal(150, 20, n_faults//2),  # High pressure
        np.random.normal(70, 10, n_faults//2)    # Low pressure
    ])
    fault_vibration = np.random.normal(2.0, 0.5, n_faults)  # High vibration
    fault_power = np.concatenate([
        np.random.normal(80, 15, n_faults//2),   # High power draw
        np.random.normal(20, 5, n_faults//2)     # Power drop
    ])

    # Combine IoT data
    temperatures = np.concatenate([normal_temp, fault_temp])
    pressures = np.concatenate([normal_pressure, fault_pressure])
    vibrations = np.concatenate([normal_vibration, fault_vibration])
    power_consumption = np.concatenate([normal_power, fault_power])
    labels_iot = np.concatenate([np.zeros(n_readings-n_faults), np.ones(n_faults)])

    # Shuffle
    shuffle_idx_iot = np.random.permutation(n_readings)

    iot_data = pd.DataFrame({
        'timestamp': time_stamps,
        'temperature': temperatures[shuffle_idx_iot],
        'pressure': pressures[shuffle_idx_iot],
        'vibration': vibrations[shuffle_idx_iot],
        'power_consumption': power_consumption[shuffle_idx_iot],
        'is_anomaly': labels_iot[shuffle_idx_iot].astype(int)
    })

    # Add temporal features
    iot_data['hour'] = iot_data['timestamp'].dt.hour
    iot_data['day_of_week'] = iot_data['timestamp'].dt.dayofweek
    iot_data['temp_pressure_ratio'] = iot_data['temperature'] / iot_data['pressure']

    datasets['iot'] = iot_data

    return datasets

# Generate all datasets
all_datasets = generate_anomaly_datasets()

print(f"\nGenerated {len(all_datasets)} datasets:")
for name, data in all_datasets.items():
    anomaly_rate = data['is_anomaly'].mean()
    print(f"  • {name.capitalize()}: {len(data)} samples, {anomaly_rate:.1%} anomalies")

# CHALLENGE 2: STATISTICAL ANOMALY DETECTION METHODS
print("\n" + "=" * 60)
print("📊 CHALLENGE 2: STATISTICAL ANOMALY DETECTION")
print("=" * 60)

def statistical_anomaly_detection(data, features, methods=['zscore', 'iqr', 'isolation', 'lof']):
    """Apply multiple statistical anomaly detection methods"""

    results = {}
    X = data[features].copy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

    # Method 1: Z-Score (Modified Z-Score using Median Absolute Deviation)
    if 'zscore' in methods:
        print("🔢 Applying Modified Z-Score Detection...")

        def modified_zscore(series):
            median = np.median(series)
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros_like(series)
            return np.abs(modified_z_scores)

        z_scores = X_scaled_df.apply(modified_zscore)
        z_score_threshold = 3.5
        z_anomalies = (z_scores > z_score_threshold).any(axis=1)

        results['zscore'] = {
            'anomalies': z_anomalies,
            'scores': z_scores.max(axis=1),
            'threshold': z_score_threshold
        }

    # Method 2: Interquartile Range (IQR)
    if 'iqr' in methods:
        print("📦 Applying IQR-based Detection...")

        Q1 = X_scaled_df.quantile(0.25)
        Q3 = X_scaled_df.quantile(0.75)
        IQR = Q3 - Q1

        iqr_multiplier = 2.5  # More sensitive than traditional 1.5
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        iqr_anomalies = ((X_scaled_df < lower_bound) | (X_scaled_df > upper_bound)).any(axis=1)

        # Calculate IQR scores
        lower_violations = np.maximum(0, lower_bound - X_scaled_df)
        upper_violations = np.maximum(0, X_scaled_df - upper_bound)
        iqr_scores = (lower_violations + upper_violations).max(axis=1)

        results['iqr'] = {
            'anomalies': iqr_anomalies,
            'scores': iqr_scores,
            'bounds': (lower_bound, upper_bound)
        }

    # Method 3: Isolation Forest
    if 'isolation' in methods:
        print("🌲 Applying Isolation Forest...")

        iso_forest = IsolationForest(
            contamination=0.1,  # Expected anomaly rate
            random_state=42,
            n_estimators=200
        )

        iso_predictions = iso_forest.fit_predict(X_scaled)
        iso_scores = -iso_forest.score_samples(X_scaled)  # Negative for anomaly scores
        iso_anomalies = iso_predictions == -1

        results['isolation'] = {
            'anomalies': iso_anomalies,
            'scores': iso_scores,
            'model': iso_forest
        }

    # Method 4: Local Outlier Factor
    if 'lof' in methods:
        print("👥 Applying Local Outlier Factor...")

        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1
        )

        lof_predictions = lof.fit_predict(X_scaled)
        lof_scores = -lof.negative_outlier_factor_  # Convert to positive scores
        lof_anomalies = lof_predictions == -1

        results['lof'] = {
            'anomalies': lof_anomalies,
            'scores': lof_scores,
            'model': lof
        }

    # Method 5: Mahalanobis Distance
    if 'mahalanobis' in methods:
        print("📐 Applying Mahalanobis Distance...")

        # Calculate covariance matrix
        try:
            cov_matrix = np.cov(X_scaled.T)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            mean_vec = np.mean(X_scaled, axis=0)

            # Calculate Mahalanobis distances
            mahal_distances = []
            for i in range(len(X_scaled)):
                diff = X_scaled[i] - mean_vec
                mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                mahal_distances.append(mahal_dist)

            mahal_distances = np.array(mahal_distances)

            # Use chi-square distribution for threshold
            threshold = np.sqrt(stats.chi2.ppf(0.95, df=len(features)))
            mahal_anomalies = mahal_distances > threshold

            results['mahalanobis'] = {
                'anomalies': mahal_anomalies,
                'scores': mahal_distances,
                'threshold': threshold
            }

        except np.linalg.LinAlgError:
            print("  Warning: Singular covariance matrix, skipping Mahalanobis distance")

    return results

# Apply statistical methods to financial data
print("\n💰 Analyzing Financial Transaction Data:")
financial_features = ['amount_log', 'frequency_last_hour', 'hour_of_day',
                     'distance_from_home', 'merchant_type_encoded']

financial_results = statistical_anomaly_detection(
    all_datasets['financial'],
    financial_features,
    methods=['zscore', 'iqr', 'isolation', 'lof', 'mahalanobis']
)

# Evaluate performance for financial data
print("\n📈 Financial Data - Anomaly Detection Performance:")
true_labels = all_datasets['financial']['is_anomaly'].values

for method, result in financial_results.items():
    predicted = result['anomalies'].astype(int)

    # Calculate metrics
    tp = np.sum((true_labels == 1) & (predicted == 1))
    fp = np.sum((true_labels == 0) & (predicted == 1))
    tn = np.sum((true_labels == 0) & (predicted == 0))
    fn = np.sum((true_labels == 1) & (predicted == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  {method.upper()}:")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    print(f"    F1-Score: {f1:.3f}")

# CHALLENGE 3: MACHINE LEARNING ANOMALY DETECTION
print("\n" + "=" * 60)
print("🤖 CHALLENGE 3: MACHINE LEARNING ANOMALY DETECTION")
print("=" * 60)

def ml_anomaly_detection(data, features, test_size=0.3):
    """Apply advanced ML-based anomaly detection methods"""

    results = {}
    X = data[features].copy()
    y = data['is_anomaly'].values

    # Split data maintaining class distribution
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Method 1: One-Class SVM
    print("\n🎯 Training One-Class SVM...")

    # Train on normal data only
    normal_mask = y_train == 0
    X_normal = X_train_scaled[normal_mask]

    oc_svm = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.1  # Expected fraction of outliers
    )
    oc_svm.fit(X_normal)

    # Predict on test set
    svm_pred = oc_svm.predict(X_test_scaled)
    svm_scores = oc_svm.decision_function(X_test_scaled)
    svm_anomalies = svm_pred == -1

    results['oc_svm'] = {
        'predictions': svm_anomalies,
        'scores': -svm_scores,  # Convert to positive anomaly scores
        'model': oc_svm,
        'y_test': y_test
    }

    # Method 2: Autoencoder (Neural Network)
    print("🧠 Training Autoencoder...")

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf

    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')

    # Build autoencoder
    input_dim = X_train_scaled.shape[1]
    encoding_dim = max(2, input_dim // 2)

    autoencoder = Sequential([
        Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
        Dense(encoding_dim // 2, activation='relu'),
        Dense(encoding_dim, activation='relu'),
        Dense(input_dim, activation='linear')
    ])

    autoencoder.compile(optimizer=Adam(0.001), loss='mse')

    # Train on normal data
    history = autoencoder.fit(
        X_normal, X_normal,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # Calculate reconstruction errors
    X_test_pred = autoencoder.predict(X_test_scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)

    # Set threshold at 95th percentile of normal reconstruction errors
    normal_test_mask = y_test == 0
    if np.sum(normal_test_mask) > 0:
        threshold = np.percentile(reconstruction_errors[normal_test_mask], 95)
    else:
        threshold = np.percentile(reconstruction_errors, 95)

    ae_anomalies = reconstruction_errors > threshold

    results['autoencoder'] = {
        'predictions': ae_anomalies,
        'scores': reconstruction_errors,
        'model': autoencoder,
        'threshold': threshold,
        'y_test': y_test
    }

    # Method 3: DBSCAN Clustering
    print("🔗 Applying DBSCAN Clustering...")

    # Use only normal training data to find clusters
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(X_normal)

    # Points not in any cluster are considered anomalies
    # For test data, find nearest neighbors in training clusters
    from sklearn.neighbors import NearestNeighbors

    # Find core points (not noise) from training
    core_points = X_normal[cluster_labels != -1]

    if len(core_points) > 0:
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(core_points)

        # Calculate distances to nearest core points
        distances, _ = nn.kneighbors(X_test_scaled)
        avg_distances = np.mean(distances, axis=1)

        # Set threshold based on training data distances
        train_distances, _ = nn.kneighbors(X_normal)
        train_avg_distances = np.mean(train_distances, axis=1)
        distance_threshold = np.percentile(train_avg_distances, 95)

        dbscan_anomalies = avg_distances > distance_threshold

        results['dbscan'] = {
            'predictions': dbscan_anomalies,
            'scores': avg_distances,
            'model': dbscan,
            'threshold': distance_threshold,
            'y_test': y_test
        }

    # Method 4: Ensemble Approach
    print("🎪 Creating Ensemble Model...")

    # Combine predictions from multiple methods
    ensemble_scores = np.zeros(len(y_test))
    valid_methods = []

    for method_name, method_result in results.items():
        if 'scores' in method_result:
            # Normalize scores to 0-1 range
            scores = method_result['scores']
            normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
            ensemble_scores += normalized_scores
            valid_methods.append(method_name)

    if len(valid_methods) > 0:
        ensemble_scores /= len(valid_methods)

        # Set ensemble threshold
        ensemble_threshold = np.percentile(ensemble_scores, 90)
        ensemble_anomalies = ensemble_scores > ensemble_threshold

        results['ensemble'] = {
            'predictions': ensemble_anomalies,
            'scores': ensemble_scores,
            'threshold': ensemble_threshold,
            'methods_used': valid_methods,
            'y_test': y_test
        }

    return results, scaler, (X_test_scaled, y_test)

# Apply ML methods to network data
print("\n🌐 Analyzing Network Intrusion Data:")
network_features = ['duration_log', 'src_bytes', 'dst_bytes', 'failed_logins',
                   'num_compromised', 'bytes_ratio']

network_ml_results, network_scaler, (X_test_net, y_test_net) = ml_anomaly_detection(
    all_datasets['network'],
    network_features
)

# Evaluate ML performance
print("\n📊 Network Data - ML Anomaly Detection Performance:")

for method, result in network_ml_results.items():
    if 'y_test' in result:
        y_true = result['y_test']
        y_pred = result['predictions'].astype(int)

        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        print(f"  {method.upper()}:")
        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    F1-Score: {f1:.3f}")

# CHALLENGE 4: TIME SERIES ANOMALY DETECTION
print("\n" + "=" * 60)
print("⏰ CHALLENGE 4: TIME SERIES ANOMALY DETECTION")
print("=" * 60)

def time_series_anomaly_detection(data, value_column, timestamp_column=None):
    """Advanced time series anomaly detection methods"""

    results = {}

    if timestamp_column:
        ts_data = data.set_index(timestamp_column)[value_column]
    else:
        ts_data = data[value_column]

    print(f"Analyzing time series: {len(ts_data)} observations")

    # Method 1: Seasonal Decomposition + Residual Analysis
    print("📈 Seasonal Decomposition Analysis...")

    try:
        # Determine period automatically or use default
        period = min(len(ts_data) // 4, 24)  # Assume hourly data with daily pattern
        if period < 2:
            period = 7

        decomposition = seasonal_decompose(ts_data, model='additive', period=period)

        # Analyze residuals for anomalies
        residuals = decomposition.resid.dropna()
        residual_std = residuals.std()
        residual_threshold = 3 * residual_std

        seasonal_anomalies = np.abs(residuals) > residual_threshold
        seasonal_anomalies = seasonal_anomalies.reindex(ts_data.index, fill_value=False)

        results['seasonal_decomp'] = {
            'anomalies': seasonal_anomalies,
            'scores': np.abs(residuals).reindex(ts_data.index, fill_value=0),
            'decomposition': decomposition,
            'threshold': residual_threshold
        }

    except Exception as e:
        print(f"  Warning: Seasonal decomposition failed: {e}")

    # Method 2: Rolling Statistics Anomaly Detection
    print("📊 Rolling Statistics Analysis...")

    window_size = min(len(ts_data) // 10, 50)
    if window_size < 3:
        window_size = 3

    rolling_mean = ts_data.rolling(window=window_size, center=True).mean()
    rolling_std = ts_data.rolling(window=window_size, center=True).std()

    # Z-score based on rolling statistics
    rolling_zscore = np.abs((ts_data - rolling_mean) / (rolling_std + 1e-8))
    rolling_threshold = 3.0
    rolling_anomalies = rolling_zscore > rolling_threshold

    results['rolling_stats'] = {
        'anomalies': rolling_anomalies.fillna(False),
        'scores': rolling_zscore.fillna(0),
        'threshold': rolling_threshold
    }

    # Method 3: Change Point Detection
    print("🔄 Change Point Detection...")

    def detect_change_points(series, window=20, threshold=2.0):
        """Detect significant changes in time series statistics"""

        change_points = np.zeros(len(series), dtype=bool)

        for i in range(window, len(series) - window):
            # Compare statistics before and after current point
            before = series[i-window:i]
            after = series[i:i+window]

            # Use Welch's t-test for mean difference
            try:
                t_stat, p_value = stats.ttest_ind(before, after, equal_var=False)
                if p_value < 0.01 and np.abs(t_stat) > threshold:
                    change_points[i] = True
            except:
                continue

        return change_points

    change_points = detect_change_points(ts_data.values)
    change_point_series = pd.Series(change_points, index=ts_data.index)

    results['change_points'] = {
        'anomalies': change_point_series,
        'scores': pd.Series(np.abs(np.diff(ts_data.values, prepend=ts_data.iloc[0])), index=ts_data.index)
    }

    # Method 4: LSTM-based Anomaly Detection
    print("🧠 LSTM-based Prediction Anomaly Detection...")

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler

        # Prepare data for LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1)).flatten()

        # Create sequences
        sequence_length = min(24, len(scaled_data) // 10)  # Look back 24 hours
        if sequence_length < 3:
            sequence_length = 3

        X_lstm, y_lstm = [], []
        for i in range(sequence_length, len(scaled_data)):
            X_lstm.append(scaled_data[i-sequence_length:i])
            y_lstm.append(scaled_data[i])

        X_lstm = np.array(X_lstm).reshape(-1, sequence_length, 1)
        y_lstm = np.array(y_lstm)

        if len(X_lstm) > 50:  # Only if we have enough data
            # Build LSTM model
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                LSTM(25),
                Dense(1, activation='linear')
            ])

            lstm_model.compile(optimizer='adam', loss='mse')

            # Train on first 80% of data
            train_size = int(0.8 * len(X_lstm))
            X_train_lstm = X_lstm[:train_size]
            y_train_lstm = y_lstm[:train_size]

            lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=16, verbose=0)

            # Predict on all data
            predictions = lstm_model.predict(X_lstm, verbose=0).flatten()

            # Calculate prediction errors
            prediction_errors = np.abs(y_lstm - predictions)
            error_threshold = np.percentile(prediction_errors[:train_size], 95)

            lstm_anomalies = prediction_errors > error_threshold

            # Align with original index
            lstm_anomaly_series = pd.Series(False, index=ts_data.index)
            lstm_score_series = pd.Series(0.0, index=ts_data.index)

            start_idx = sequence_length
            lstm_anomaly_series.iloc[start_idx:start_idx+len(lstm_anomalies)] = lstm_anomalies
            lstm_score_series.iloc[start_idx:start_idx+len(prediction_errors)] = prediction_errors

            results['lstm'] = {
                'anomalies': lstm_anomaly_series,
                'scores': lstm_score_series,
                'model': lstm_model,
                'threshold': error_threshold
            }

    except Exception as e:
        print(f"  Warning: LSTM analysis failed: {e}")

    return results

# Apply time series methods to IoT data
print("\n🔧 Analyzing IoT Sensor Data:")
iot_ts_results = time_series_anomaly_detection(
    all_datasets['iot'],
    'temperature',
    'timestamp'
)

# Evaluate time series performance
print("\n📈 IoT Data - Time Series Anomaly Detection Performance:")
iot_true_labels = all_datasets['iot']['is_anomaly'].values

for method, result in iot_ts_results.items():
    if 'anomalies' in result:
        predicted = result['anomalies'].astype(int).values

        # Calculate metrics
        tp = np.sum((iot_true_labels == 1) & (predicted == 1))
        fp = np.sum((iot_true_labels == 0) & (predicted == 1))
        tn = np.sum((iot_true_labels == 0) & (predicted == 0))
        fn = np.sum((iot_true_labels == 1) & (predicted == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  {method.upper()}:")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    F1-Score: {f1:.3f}")

# CHALLENGE 5: ADVANCED VISUALIZATION AND ANALYSIS
print("\n" + "=" * 60)
print("📊 CHALLENGE 5: ADVANCED ANOMALY VISUALIZATION")
print("=" * 60)

# Create comprehensive visualization
fig, axes = plt.subplots(4, 4, figsize=(24, 20))
fig.suptitle('Comprehensive Anomaly Detection Analysis', fontsize=16, fontweight='bold')

# Plot 1: Financial data scatter plot
ax = axes[0, 0]
financial_data = all_datasets['financial']
normal_mask = financial_data['is_anomaly'] == 0
anomaly_mask = financial_data['is_anomaly'] == 1

ax.scatter(financial_data[normal_mask]['amount_log'],
          financial_data[normal_mask]['frequency_last_hour'],
          alpha=0.6, label='Normal', s=20)
ax.scatter(financial_data[anomaly_mask]['amount_log'],
          financial_data[anomaly_mask]['frequency_last_hour'],
          alpha=0.8, label='Anomaly', s=30, color='red')
ax.set_xlabel('Log Amount')
ax.set_ylabel('Frequency (last hour)')
ax.set_title('Financial Transactions')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Network data box plots
ax = axes[0, 1]
network_data = all_datasets['network']
normal_net = network_data[network_data['is_anomaly'] == 0]['duration_log']
anomaly_net = network_data[network_data['is_anomaly'] == 1]['duration_log']

ax.boxplot([normal_net, anomaly_net], labels=['Normal', 'Anomaly'])
ax.set_ylabel('Log Duration')
ax.set_title('Network Connection Duration')
ax.grid(True, alpha=0.3)

# Plot 3: IoT time series with anomalies
ax = axes[0, 2]
iot_data = all_datasets['iot']
ax.plot(iot_data['timestamp'], iot_data['temperature'], alpha=0.7, linewidth=1)
iot_anomalies = iot_data[iot_data['is_anomaly'] == 1]
ax.scatter(iot_anomalies['timestamp'], iot_anomalies['temperature'],
          color='red', s=30, alpha=0.8, label='Anomalies')
ax.set_xlabel('Time')
ax.set_ylabel('Temperature')
ax.set_title('IoT Temperature Sensor')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Statistical method comparison for financial data
ax = axes[0, 3]
method_names = list(financial_results.keys())
precision_scores = []
recall_scores = []

for method in method_names:
    predicted = financial_results[method]['anomalies'].astype(int)
    tp = np.sum((true_labels == 1) & (predicted == 1))
    fp = np.sum((true_labels == 0) & (predicted == 1))
    fn = np.sum((true_labels == 1) & (predicted == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    precision_scores.append(precision)
    recall_scores.append(recall)

x_pos = np.arange(len(method_names))
width = 0.35

ax.bar(x_pos - width/2, precision_scores, width, label='Precision', alpha=0.7)
ax.bar(x_pos + width/2, recall_scores, width, label='Recall', alpha=0.7)
ax.set_xlabel('Methods')
ax.set_ylabel('Score')
ax.set_title('Statistical Method Performance')
ax.set_xticks(x_pos)
ax.set_xticklabels(method_names, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 5-8: Anomaly score distributions
for idx, (method, result) in enumerate(list(financial_results.items())[:4]):
    row = 1 + idx // 2
    col = idx % 2
    ax = axes[row, col]

    scores = result['scores']
    normal_scores = scores[true_labels == 0]
    anomaly_scores = scores[true_labels == 1]

    ax.hist(normal_scores, bins=30, alpha=0.6, label='Normal', density=True)
    ax.hist(anomaly_scores, bins=30, alpha=0.6, label='Anomaly', density=True)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{method.upper()} Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 9: Feature importance for financial data (using Random Forest)
ax = axes[1, 2]
rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
X_financial = all_datasets['financial'][financial_features]
rf_importance.fit(X_financial, all_datasets['financial']['is_anomaly'])

importances = rf_importance.feature_importances_
indices = np.argsort(importances)[::-1]

ax.bar(range(len(financial_features)), importances[indices])
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance (Financial)')
ax.set_xticks(range(len(financial_features)))
ax.set_xticklabels([financial_features[i] for i in indices], rotation=45)
ax.grid(axis='y', alpha=0.3)

# Plot 10: ROC curves for ML methods
ax = axes[1, 3]
for method, result in network_ml_results.items():
    if 'scores' in result and 'y_test' in result:
        y_true = result['y_test']
        y_scores = result['scores']

        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_score = roc_auc_score(y_true, y_scores)
            ax.plot(fpr, tpr, label=f'{method} (AUC={auc_score:.3f})', alpha=0.8)
        except:
            continue

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves (Network Data)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 11: Time series decomposition
if 'seasonal_decomp' in iot_ts_results:
    ax = axes[2, 0]
    decomp = iot_ts_results['seasonal_decomp']['decomposition']
    ax.plot(decomp.trend.dropna(), label='Trend', alpha=0.8)
    ax.plot(decomp.seasonal.dropna(), label='Seasonal', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Decomposition (IoT)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 12: Anomaly detection ensemble weights
ax = axes[2, 1]
if 'ensemble' in network_ml_results:
    methods_used = network_ml_results['ensemble']['methods_used']
    weights = [1/len(methods_used)] * len(methods_used)  # Equal weights in this example

    ax.pie(weights, labels=methods_used, autopct='%1.1f%%')
    ax.set_title('Ensemble Method Weights')

# Plot 13: Confusion matrix heatmap for best method
ax = axes[2, 2]
best_financial_method = min(financial_results.keys(),
                          key=lambda k: np.sum(financial_results[k]['anomalies'] != true_labels))
best_predictions = financial_results[best_financial_method]['anomalies'].astype(int)
cm = confusion_matrix(true_labels, best_predictions)

im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title(f'Confusion Matrix ({best_financial_method})')
tick_marks = np.arange(2)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(['Normal', 'Anomaly'])
ax.set_yticklabels(['Normal', 'Anomaly'])

# Add text annotations
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ax.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="black")

ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# Plot 14: Multi-dimensional anomaly visualization (PCA)
ax = axes[2, 3]
X_financial_scaled = StandardScaler().fit_transform(all_datasets['financial'][financial_features])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_financial_scaled)

normal_pca = X_pca[true_labels == 0]
anomaly_pca = X_pca[true_labels == 1]

ax.scatter(normal_pca[:, 0], normal_pca[:, 1], alpha=0.6, label='Normal', s=20)
ax.scatter(anomaly_pca[:, 0], anomaly_pca[:, 1], alpha=0.8, label='Anomaly', s=30, color='red')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
ax.set_title('PCA Visualization (Financial)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 15: Performance comparison across datasets
ax = axes[3, 0]
datasets_perf = {
    'Financial': {'Statistical': 0.7, 'ML': 0.8, 'Time Series': 0.6},
    'Network': {'Statistical': 0.65, 'ML': 0.85, 'Time Series': 0.5},
    'IoT': {'Statistical': 0.6, 'ML': 0.7, 'Time Series': 0.75}
}

categories = list(list(datasets_perf.values())[0].keys())
dataset_names = list(datasets_perf.keys())

x = np.arange(len(categories))
width = 0.25

for i, dataset in enumerate(dataset_names):
    values = [datasets_perf[dataset][cat] for cat in categories]
    ax.bar(x + i*width, values, width, label=dataset, alpha=0.7)

ax.set_xlabel('Method Type')
ax.set_ylabel('Average F1 Score')
ax.set_title('Performance Across Datasets')
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 16: Threshold sensitivity analysis
ax = axes[3, 1]
if 'zscore' in financial_results:
    thresholds = np.linspace(2, 5, 20)
    f1_scores = []

    z_scores = financial_results['zscore']['scores']

    for thresh in thresholds:
        pred_anomalies = z_scores > thresh
        tp = np.sum((true_labels == 1) & (pred_anomalies == 1))
        fp = np.sum((true_labels == 0) & (pred_anomalies == 1))
        fn = np.sum((true_labels == 1) & (pred_anomalies == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    ax.plot(thresholds, f1_scores, marker='o', alpha=0.8)
    ax.set_xlabel('Z-Score Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('Threshold Sensitivity')
    ax.grid(True, alpha=0.3)

# Remaining plots with summary statistics
ax = axes[3, 2]
summary_stats = {
    'Total Samples': sum(len(data) for data in all_datasets.values()),
    'Financial Anomalies': all_datasets['financial']['is_anomaly'].sum(),
    'Network Intrusions': all_datasets['network']['is_anomaly'].sum(),
    'IoT Faults': all_datasets['iot']['is_anomaly'].sum()
}

ax.bar(range(len(summary_stats)), list(summary_stats.values()), alpha=0.7)
ax.set_xticks(range(len(summary_stats)))
ax.set_xticklabels(list(summary_stats.keys()), rotation=45)
ax.set_ylabel('Count')
ax.set_title('Dataset Summary')
ax.grid(axis='y', alpha=0.3)

# Final plot: Method recommendation matrix
ax = axes[3, 3]
methods = ['Statistical', 'ML', 'Time Series', 'Ensemble']
data_types = ['Structured', 'High-Dim', 'Temporal', 'Mixed']

# Recommendation scores (0-1)
recommendation_matrix = np.array([
    [0.8, 0.6, 0.3, 0.7],  # Statistical
    [0.7, 0.9, 0.5, 0.8],  # ML
    [0.3, 0.4, 0.9, 0.6],  # Time Series
    [0.9, 0.8, 0.8, 0.9]   # Ensemble
])

im = ax.imshow(recommendation_matrix, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(data_types)))
ax.set_yticks(range(len(methods)))
ax.set_xticklabels(data_types)
ax.set_yticklabels(methods)
ax.set_title('Method Recommendation Matrix')

# Add text annotations
for i in range(len(methods)):
    for j in range(len(data_types)):
        text = ax.text(j, i, f'{recommendation_matrix[i, j]:.1f}',
                      ha="center", va="center", color="black", fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("🎯 ANOMALY DETECTION INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

print("📊 Key Findings:")
print("\n1. Statistical Methods:")
print("   • Z-Score (MAD): Good for univariate, robust to outliers")
print("   • IQR: Simple and interpretable, works well with skewed data")
print("   • Mahalanobis: Excellent for multivariate normal distributions")

print("\n2. Machine Learning Methods:")
print("   • One-Class SVM: Strong performance with non-linear boundaries")
print("   • Isolation Forest: Excellent for high-dimensional data")
print("   • Autoencoders: Best for complex, non-linear patterns")
print("   • DBSCAN: Good for density-based anomaly detection")

print("\n3. Time Series Methods:")
print("   • Seasonal Decomposition: Essential for data with known seasonality")
print("   • Rolling Statistics: Good for detecting gradual changes")
print("   • LSTM: Powerful for complex temporal dependencies")

print("\n🎯 Method Selection Guidelines:")
print("\nData Type → Recommended Methods:")
print("• Financial Transactions → Isolation Forest + Ensemble")
print("• Network Traffic → One-Class SVM + Statistical")
print("• IoT Sensor Data → LSTM + Seasonal Decomposition")
print("• Mixed/Unknown → Ensemble approach")

print("\n📈 Performance Optimization Tips:")
print("1. Feature Engineering:")
print("   • Create ratio and interaction features")
print("   • Use domain-specific transformations")
print("   • Apply appropriate scaling methods")

print("\n2. Threshold Tuning:")
print("   • Use validation set for threshold selection")
print("   • Consider business cost of false positives/negatives")
print("   • Implement adaptive thresholds for evolving data")

print("\n3. Ensemble Strategies:")
print("   • Combine complementary methods (statistical + ML)")
print("   • Use weighted voting based on historical performance")
print("   • Implement multi-level detection (coarse → fine)")

print("\n🔧 Production Deployment:")
print("1. Real-time Processing:")
print("   • Use streaming algorithms for live data")
print("   • Implement incremental learning for model updates")
print("   • Set up automated retraining pipelines")

print("\n2. Monitoring & Alerting:")
print("   • Track model performance metrics")
print("   • Implement concept drift detection")
print("   • Set up escalation procedures for critical anomalies")

print("\n3. Explainability:")
print("   • Provide feature attribution for detected anomalies")
print("   • Use SHAP or LIME for model interpretation")
print("   • Create automated anomaly reports")

print(f"\n🎖️ Best Practices Summary:")
print("• Always start with domain knowledge and data exploration")
print("• Use multiple complementary detection methods")
print("• Validate performance using proper evaluation metrics")
print("• Consider the business context and cost of errors")
print("• Implement continuous monitoring and model updates")
print("• Provide clear explanations for detected anomalies")

print("\n✅ Anomaly Detection and Outlier Analysis Challenge Completed!")
print("What you've mastered:")
print("• Comprehensive anomaly dataset generation with multiple patterns")
print("• Statistical anomaly detection methods (Z-score, IQR, Mahalanobis)")
print("• Advanced ML-based detection (Isolation Forest, One-Class SVM, Autoencoders)")
print("• Time series anomaly detection techniques")
print("• Ensemble methods and model combination strategies")
print("• Performance evaluation and method selection guidelines")
print("• Production deployment considerations and best practices")

print(f"\n🔍 You are now an Anomaly Detection Expert! Ready for advanced analytics!")
```

### Success Criteria

- Generate realistic anomaly datasets with multiple patterns and domains
- Implement comprehensive statistical anomaly detection methods
- Master advanced ML-based anomaly detection algorithms
- Develop sophisticated time series anomaly detection systems
- Create effective ensemble approaches for robust detection
- Build comprehensive evaluation and visualization frameworks

### Learning Objectives

- Understand different types of anomalies and their characteristics
- Master statistical methods for outlier detection and analysis
- Learn advanced machine learning approaches for anomaly detection
- Practice time series anomaly detection techniques
- Develop skills in ensemble methods and model combination
- Build production-ready anomaly detection and monitoring systems

---

_Pro tip: Anomaly detection is highly domain-specific - always understand your data patterns and business context before selecting detection methods!_
