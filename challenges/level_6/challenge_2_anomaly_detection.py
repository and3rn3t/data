"""
Level 6 - Challenge 2: Anomaly Detection & Outlier Analysis
==========================================================

Master anomaly detection techniques for identifying unusual patterns in data.
This challenge covers statistical methods, machine learning approaches, and real-world applications.

Learning Objectives:
- Understand different types of anomalies and outliers
- Learn statistical anomaly detection methods
- Master machine learning approaches (Isolation Forest, One-Class SVM)
- Explore time series anomaly detection
- Handle imbalanced anomaly datasets and evaluation

Required Libraries: pandas, numpy, matplotlib, scikit-learn
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")


def create_anomaly_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Create comprehensive datasets with various types of anomalies.

    Returns:
        Dictionary containing datasets with known anomalies for testing detection methods
    """
    print("🔍 Creating Anomaly Detection Datasets...")

    datasets = {}
    rng = np.random.default_rng(42)

    # 1. Credit Card Transaction Dataset
    print("Creating credit card transaction dataset...")

    n_transactions = 10000
    n_anomalies = 50  # 0.5% fraud rate (realistic)

    # Normal transaction features
    normal_transactions = []

    for _ in range(n_transactions - n_anomalies):
        # Normal transaction patterns
        amount = rng.lognormal(mean=3.0, sigma=1.5)  # Most transactions are small

        # Time of day (normal business hours bias)
        hour_probs = np.array(
            [
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.02,  # 0-5 AM
                0.03,
                0.05,
                0.07,
                0.09,
                0.08,
                0.07,  # 6-11 AM
                0.08,
                0.07,
                0.06,
                0.05,
                0.06,
                0.08,  # 12-5 PM
                0.09,
                0.08,
                0.06,
                0.04,
                0.03,
                0.02,
            ]
        )  # 6-11 PM
        hour_probs = hour_probs / hour_probs.sum()  # Normalize to sum to 1
        hour = rng.choice(24, p=hour_probs)

        # Day of week (weekday bias)
        day_of_week = rng.choice(7, p=[0.15, 0.16, 0.16, 0.16, 0.16, 0.12, 0.09])

        # Merchant category (normal distribution)
        merchant_category = rng.choice(
            ["grocery", "gas", "restaurant", "retail", "online"],
            p=[0.3, 0.2, 0.25, 0.15, 0.1],
        )

        # Location consistency (most transactions near home)
        distance_from_home = rng.exponential(scale=5.0)  # km

        # Account age and history
        account_age_days = rng.uniform(365, 3650)  # 1-10 years
        avg_monthly_spending = rng.uniform(500, 3000)

        # Derived features
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if hour < 6 or hour > 22 else 0
        amount_vs_avg = amount / (avg_monthly_spending / 30)

        normal_transactions.append(
            [
                amount,
                hour,
                day_of_week,
                distance_from_home,
                account_age_days,
                avg_monthly_spending,
                is_weekend,
                is_night,
                amount_vs_avg,
                merchant_category == "grocery",
                merchant_category == "gas",
                merchant_category == "restaurant",
                merchant_category == "retail",
                merchant_category == "online",
                0,  # Not fraud
            ]
        )

    # Fraudulent transactions (anomalies)
    fraudulent_transactions = []

    for _ in range(n_anomalies):
        # Fraud patterns - unusual amounts, times, locations
        amount = rng.choice(
            [
                rng.uniform(2000, 10000),  # Large unusual amounts
                rng.uniform(0.01, 1.0),  # Very small amounts (testing)
                rng.uniform(100, 500),  # Round amounts
            ]
        )

        # Unusual times (very late/early)
        hour = rng.choice([2, 3, 4, 23, 24, 1])

        # Random day
        day_of_week = rng.choice(7)

        # Unusual locations (far from home)
        distance_from_home = rng.uniform(500, 2000)  # Very far

        # Account details
        account_age_days = rng.uniform(1, 3650)
        avg_monthly_spending = rng.uniform(200, 5000)

        # Unusual merchant categories
        merchant_category = rng.choice(["online", "retail", "gas"], p=[0.6, 0.3, 0.1])

        # Derived features
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if hour < 6 or hour > 22 else 0
        amount_vs_avg = amount / (avg_monthly_spending / 30)

        fraudulent_transactions.append(
            [
                amount,
                hour,
                day_of_week,
                distance_from_home,
                account_age_days,
                avg_monthly_spending,
                is_weekend,
                is_night,
                amount_vs_avg,
                merchant_category == "grocery",
                merchant_category == "gas",
                merchant_category == "restaurant",
                merchant_category == "retail",
                merchant_category == "online",
                1,  # Fraud
            ]
        )

    # Combine and create DataFrame
    all_transactions = normal_transactions + fraudulent_transactions
    rng.shuffle(all_transactions)

    credit_df = pd.DataFrame(
        all_transactions,
        columns=[
            "amount",
            "hour",
            "day_of_week",
            "distance_from_home",
            "account_age_days",
            "avg_monthly_spending",
            "is_weekend",
            "is_night",
            "amount_vs_avg",
            "is_grocery",
            "is_gas",
            "is_restaurant",
            "is_retail",
            "is_online",
            "is_fraud",
        ],
    )

    datasets["credit_fraud"] = {
        "data": credit_df,
        "target_column": "is_fraud",
        "feature_columns": [col for col in credit_df.columns if col != "is_fraud"],
        "anomaly_rate": n_anomalies / n_transactions,
        "description": "Credit card transactions with fraud detection",
    }

    # 2. Network Intrusion Detection Dataset
    print("Creating network intrusion dataset...")

    n_connections = 5000
    n_intrusions = 75  # 1.5% intrusion rate

    # Normal network connections
    normal_connections = []

    for _ in range(n_connections - n_intrusions):
        # Normal connection patterns
        duration = rng.exponential(scale=30.0)  # seconds
        bytes_sent = rng.exponential(scale=1000.0)
        bytes_received = rng.exponential(scale=2000.0)

        # Protocol distribution
        protocol = rng.choice(["tcp", "udp", "icmp"], p=[0.7, 0.25, 0.05])

        # Service distribution
        service = rng.choice(
            ["http", "ftp", "smtp", "ssh", "dns"], p=[0.4, 0.1, 0.1, 0.2, 0.2]
        )

        # Connection state
        flag = rng.choice(["SF", "S0", "REJ"], p=[0.8, 0.15, 0.05])  # SF = successful

        # Error rates (low for normal)
        src_error_rate = rng.beta(1, 20)  # Low error rate
        dst_error_rate = rng.beta(1, 20)

        # Connection counts (normal patterns)
        same_host_conn = rng.poisson(lam=5)
        diff_host_conn = rng.poisson(lam=2)

        normal_connections.append(
            [
                duration,
                bytes_sent,
                bytes_received,
                src_error_rate,
                dst_error_rate,
                same_host_conn,
                diff_host_conn,
                protocol == "tcp",
                protocol == "udp",
                service == "http",
                service == "ftp",
                service == "smtp",
                service == "ssh",
                service == "dns",
                flag == "SF",
                0,  # Not intrusion
            ]
        )

    # Intrusion connections (anomalies)
    intrusion_connections = []

    for _ in range(n_intrusions):
        # Intrusion patterns - unusual duration, data transfer, error rates
        intrusion_type = rng.choice(["dos", "probe", "r2l", "u2r"])

        if intrusion_type == "dos":  # Denial of Service
            duration = rng.uniform(0, 1)  # Very short
            bytes_sent = rng.uniform(0, 100)
            bytes_received = rng.uniform(0, 100)
            src_error_rate = rng.uniform(0.5, 1.0)  # High error rate
            dst_error_rate = rng.uniform(0.5, 1.0)
            same_host_conn = rng.uniform(100, 1000)  # Many connections

        elif intrusion_type == "probe":  # Network probing
            duration = rng.uniform(0, 5)
            bytes_sent = rng.uniform(0, 50)
            bytes_received = rng.uniform(0, 50)
            src_error_rate = rng.uniform(0.3, 0.8)
            dst_error_rate = rng.uniform(0.3, 0.8)
            same_host_conn = rng.uniform(50, 200)

        else:  # r2l or u2r attacks
            duration = rng.uniform(100, 1000)  # Long connections
            bytes_sent = rng.uniform(5000, 50000)
            bytes_received = rng.uniform(5000, 50000)
            src_error_rate = rng.uniform(0.1, 0.5)
            dst_error_rate = rng.uniform(0.1, 0.5)
            same_host_conn = rng.uniform(1, 10)

        diff_host_conn = rng.uniform(0, 5)
        protocol = rng.choice(["tcp", "udp", "icmp"])
        service = rng.choice(["http", "ftp", "smtp", "ssh", "dns"])
        flag = rng.choice(["SF", "S0", "REJ"])

        intrusion_connections.append(
            [
                duration,
                bytes_sent,
                bytes_received,
                src_error_rate,
                dst_error_rate,
                same_host_conn,
                diff_host_conn,
                protocol == "tcp",
                protocol == "udp",
                service == "http",
                service == "ftp",
                service == "smtp",
                service == "ssh",
                service == "dns",
                flag == "SF",
                1,  # Intrusion
            ]
        )

    # Combine and create DataFrame
    all_connections = normal_connections + intrusion_connections
    rng.shuffle(all_connections)

    network_df = pd.DataFrame(
        all_connections,
        columns=[
            "duration",
            "bytes_sent",
            "bytes_received",
            "src_error_rate",
            "dst_error_rate",
            "same_host_conn",
            "diff_host_conn",
            "is_tcp",
            "is_udp",
            "is_http",
            "is_ftp",
            "is_smtp",
            "is_ssh",
            "is_dns",
            "is_successful",
            "is_intrusion",
        ],
    )

    datasets["network_intrusion"] = {
        "data": network_df,
        "target_column": "is_intrusion",
        "feature_columns": [col for col in network_df.columns if col != "is_intrusion"],
        "anomaly_rate": n_intrusions / n_connections,
        "description": "Network connections with intrusion detection",
    }

    # 3. Manufacturing Quality Control Dataset
    print("Creating manufacturing quality dataset...")

    n_products = 3000
    n_defects = 30  # 1% defect rate

    # Normal products
    normal_products = []

    for _ in range(n_products - n_defects):
        # Normal manufacturing parameters (within specifications)
        temperature = rng.normal(220, 5)  # Target: 220°C ± 5°C
        pressure = rng.normal(15, 1)  # Target: 15 bar ± 1 bar
        speed = rng.normal(100, 3)  # Target: 100 rpm ± 3 rpm

        # Material properties (normal distribution)
        density = rng.normal(2.5, 0.1)
        thickness = rng.normal(10.0, 0.2)

        # Quality measurements (correlated with parameters)
        strength = (
            1000 + 2 * temperature + 10 * pressure - 0.5 * speed + rng.normal(0, 20)
        )
        flexibility = (
            50 + 0.1 * temperature - 2 * pressure + 0.2 * speed + rng.normal(0, 5)
        )

        # Surface quality (affected by all parameters)
        surface_roughness = abs(rng.normal(1.0, 0.1))

        normal_products.append(
            [
                temperature,
                pressure,
                speed,
                density,
                thickness,
                strength,
                flexibility,
                surface_roughness,
                0,  # Not defective
            ]
        )

    # Defective products (anomalies)
    defective_products = []

    for _ in range(n_defects):
        # Various defect patterns
        defect_type = rng.choice(["temperature", "pressure", "material", "combined"])

        if defect_type == "temperature":
            temperature = rng.choice(
                [rng.normal(180, 10), rng.normal(260, 10)]
            )  # Too hot/cold
            pressure = rng.normal(15, 1)
            speed = rng.normal(100, 3)

        elif defect_type == "pressure":
            temperature = rng.normal(220, 5)
            pressure = rng.choice(
                [rng.normal(10, 2), rng.normal(20, 2)]
            )  # Too low/high
            speed = rng.normal(100, 3)

        elif defect_type == "material":
            temperature = rng.normal(220, 5)
            pressure = rng.normal(15, 1)
            speed = rng.normal(100, 3)

        else:  # combined defects
            temperature = rng.normal(240, 15)
            pressure = rng.normal(18, 3)
            speed = rng.normal(120, 10)

        # Material properties (may be off-spec for material defects)
        if defect_type == "material":
            density = rng.choice([rng.normal(2.2, 0.1), rng.normal(2.8, 0.1)])
            thickness = rng.choice([rng.normal(9.0, 0.3), rng.normal(11.0, 0.3)])
        else:
            density = rng.normal(2.5, 0.1)
            thickness = rng.normal(10.0, 0.2)

        # Quality measurements (degraded due to defects)
        strength = (
            1000 + 2 * temperature + 10 * pressure - 0.5 * speed + rng.normal(-100, 50)
        )
        flexibility = (
            50 + 0.1 * temperature - 2 * pressure + 0.2 * speed + rng.normal(-10, 10)
        )
        surface_roughness = abs(rng.normal(2.0, 0.5))  # Higher roughness

        defective_products.append(
            [
                temperature,
                pressure,
                speed,
                density,
                thickness,
                strength,
                flexibility,
                surface_roughness,
                1,  # Defective
            ]
        )

    # Combine and create DataFrame
    all_products = normal_products + defective_products
    rng.shuffle(all_products)

    manufacturing_df = pd.DataFrame(
        all_products,
        columns=[
            "temperature",
            "pressure",
            "speed",
            "density",
            "thickness",
            "strength",
            "flexibility",
            "surface_roughness",
            "is_defective",
        ],
    )

    datasets["manufacturing_quality"] = {
        "data": manufacturing_df,
        "target_column": "is_defective",
        "feature_columns": [
            col for col in manufacturing_df.columns if col != "is_defective"
        ],
        "anomaly_rate": n_defects / n_products,
        "description": "Manufacturing quality control with defect detection",
    }

    # 4. Time Series Anomaly Dataset (Server Metrics)
    print("Creating time series anomaly dataset...")

    # Generate 30 days of hourly server metrics
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=24 * 30, freq="H")

    # Normal server behavior
    cpu_usage = []
    memory_usage = []
    disk_io = []
    network_traffic = []
    response_times = []

    for i, dt in enumerate(dates):
        # Daily patterns
        hour = dt.hour

        # Business hours effect
        business_factor = 1.5 if 9 <= hour <= 17 else 0.5

        # Weekend effect
        weekend_factor = 0.3 if dt.weekday() >= 5 else 1.0

        # Base metrics with daily patterns
        base_cpu = 20 + 30 * business_factor * weekend_factor
        base_memory = 40 + 20 * business_factor * weekend_factor
        base_disk = 100 + 200 * business_factor * weekend_factor
        base_network = 500 + 1000 * business_factor * weekend_factor
        base_response = 50 + 100 * business_factor * weekend_factor

        # Add noise
        cpu_usage.append(max(0, min(100, base_cpu + rng.normal(0, 5))))
        memory_usage.append(max(0, min(100, base_memory + rng.normal(0, 3))))
        disk_io.append(max(0, base_disk + rng.normal(0, 20)))
        network_traffic.append(max(0, base_network + rng.normal(0, 100)))
        response_times.append(max(1, base_response + rng.normal(0, 10)))

    # Inject anomalies
    anomaly_indices = rng.choice(len(dates), size=20, replace=False)
    anomaly_labels = np.zeros(len(dates))

    for idx in anomaly_indices:
        anomaly_type = rng.choice(
            ["cpu_spike", "memory_leak", "disk_failure", "network_attack"]
        )

        if anomaly_type == "cpu_spike":
            cpu_usage[idx] = rng.uniform(85, 99)
            response_times[idx] *= rng.uniform(2, 5)

        elif anomaly_type == "memory_leak":
            memory_usage[idx] = rng.uniform(90, 99)
            response_times[idx] *= rng.uniform(1.5, 3)

        elif anomaly_type == "disk_failure":
            disk_io[idx] = rng.uniform(2000, 5000)
            response_times[idx] *= rng.uniform(3, 8)

        elif anomaly_type == "network_attack":
            network_traffic[idx] = rng.uniform(5000, 20000)
            response_times[idx] *= rng.uniform(2, 6)

        anomaly_labels[idx] = 1

    server_df = pd.DataFrame(
        {
            "timestamp": dates,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_io": disk_io,
            "network_traffic": network_traffic,
            "response_time": response_times,
            "is_anomaly": anomaly_labels,
        }
    )

    datasets["server_monitoring"] = {
        "data": server_df,
        "target_column": "is_anomaly",
        "feature_columns": [
            "cpu_usage",
            "memory_usage",
            "disk_io",
            "network_traffic",
            "response_time",
        ],
        "timestamp_column": "timestamp",
        "anomaly_rate": len(anomaly_indices) / len(dates),
        "description": "Server monitoring metrics with system anomalies",
    }

    print(f"Created {len(datasets)} anomaly detection datasets")
    return datasets


def statistical_anomaly_detection(
    data: pd.DataFrame,
    features: List[str],
    method: str = "zscore",
    threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply statistical methods for anomaly detection.
    """
    print(f"\n📊 Statistical Anomaly Detection: {method.upper()}")
    print("=" * 50)

    # Ensure numeric data only
    X = data[features].select_dtypes(include=[np.number]).to_numpy()
    if X.shape[1] != len(features):
        # Filter to numeric features only
        numeric_features = (
            data[features].select_dtypes(include=[np.number]).columns.tolist()
        )
        X = data[numeric_features].to_numpy()
        print(f"⚠️ Using {len(numeric_features)} numeric features: {numeric_features}")

    anomaly_scores = np.zeros(len(data))

    if method == "zscore":
        # Z-score method
        mean_vals = np.mean(X, axis=0)
        std_vals = np.std(X, axis=0)

        z_scores = np.abs((X - mean_vals) / (std_vals + 1e-8))
        anomaly_scores = np.max(z_scores, axis=1)  # Max z-score across features

        predictions = (anomaly_scores > threshold).astype(int)

        print(f"• Z-score threshold: {threshold}")
        print(f"• Mean anomaly score: {np.mean(anomaly_scores):.3f}")
        print(
            f"• Detected anomalies: {np.sum(predictions)} ({np.mean(predictions)*100:.1f}%)"
        )

    elif method == "iqr":
        # Interquartile Range method
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Count outliers per sample
        outliers = (X < lower_bound) | (X > upper_bound)
        anomaly_scores = (
            np.sum(outliers, axis=1) / X.shape[1]
        )  # Fraction of outlier features

        predictions = (anomaly_scores > 0.2).astype(int)  # 20% of features are outliers

        print(f"• IQR method with 1.5×IQR bounds")
        print(f"• Mean outlier fraction: {np.mean(anomaly_scores):.3f}")
        print(
            f"• Detected anomalies: {np.sum(predictions)} ({np.mean(predictions)*100:.1f}%)"
        )

    elif method == "modified_zscore":
        # Modified Z-score using median absolute deviation
        median_vals = np.median(X, axis=0)
        mad_vals = np.median(np.abs(X - median_vals), axis=0)

        modified_z_scores = 0.6745 * (X - median_vals) / (mad_vals + 1e-8)
        anomaly_scores = np.max(np.abs(modified_z_scores), axis=1)

        predictions = (anomaly_scores > threshold).astype(int)

        print(f"• Modified Z-score threshold: {threshold}")
        print(f"• Mean anomaly score: {np.mean(anomaly_scores):.3f}")
        print(
            f"• Detected anomalies: {np.sum(predictions)} ({np.mean(predictions)*100:.1f}%)"
        )

    return predictions, anomaly_scores


def ml_anomaly_detection(
    data: pd.DataFrame, features: List[str]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply machine learning methods for anomaly detection.
    """
    print("\n🤖 Machine Learning Anomaly Detection")
    print("=" * 50)

    # Ensure numeric data only
    X = data[features].select_dtypes(include=[np.number]).to_numpy()
    if X.shape[1] != len(features):
        # Filter to numeric features only
        numeric_features = (
            data[features].select_dtypes(include=[np.number]).columns.tolist()
        )
        X = data[numeric_features].to_numpy()
        print(f"⚠️ Using {len(numeric_features)} numeric features: {numeric_features}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # 1. Isolation Forest
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    iso_predictions = iso_forest.fit_predict(X_scaled)
    iso_scores = iso_forest.decision_function(X_scaled)

    # Convert -1/1 to 0/1
    iso_predictions = (iso_predictions == -1).astype(int)

    results["Isolation_Forest"] = (
        iso_predictions,
        -iso_scores,
    )  # Negative for consistency

    print(f"• Isolation Forest detected: {np.sum(iso_predictions)} anomalies")

    # 2. One-Class SVM
    print("Training One-Class SVM...")
    oc_svm = OneClassSVM(nu=0.1, gamma="auto")
    svm_predictions = oc_svm.fit_predict(X_scaled)
    svm_scores = oc_svm.decision_function(X_scaled)

    # Convert -1/1 to 0/1
    svm_predictions = (svm_predictions == -1).astype(int)

    results["One_Class_SVM"] = (
        svm_predictions,
        -svm_scores,
    )  # Negative for consistency

    print(f"• One-Class SVM detected: {np.sum(svm_predictions)} anomalies")

    # 3. Local Outlier Factor
    print("Computing Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof_predictions = lof.fit_predict(X_scaled)
    lof_scores = lof.negative_outlier_factor_

    # Convert -1/1 to 0/1
    lof_predictions = (lof_predictions == -1).astype(int)

    results["Local_Outlier_Factor"] = (lof_predictions, -lof_scores)

    print(f"• LOF detected: {np.sum(lof_predictions)} anomalies")

    # 4. Elliptic Envelope (Robust Covariance)
    print("Training Elliptic Envelope...")
    try:
        elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
        elliptic_predictions = elliptic.fit_predict(X_scaled)
        elliptic_scores = elliptic.decision_function(X_scaled)

        # Convert -1/1 to 0/1
        elliptic_predictions = (elliptic_predictions == -1).astype(int)

        results["Elliptic_Envelope"] = (elliptic_predictions, -elliptic_scores)

        print(f"• Elliptic Envelope detected: {np.sum(elliptic_predictions)} anomalies")

    except Exception as e:
        print(f"⚠️ Elliptic Envelope failed: {e}")

    return results


def evaluate_anomaly_detection(
    y_true: np.ndarray, predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """
    Evaluate anomaly detection performance.
    """
    print(f"\n📈 Anomaly Detection Evaluation")
    print("=" * 50)

    results = []

    for method_name, (y_pred, scores) in predictions_dict.items():
        # Basic metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        accuracy = (tp + tn) / len(y_true)

        # ROC AUC (if scores available)
        try:
            roc_auc = roc_auc_score(y_true, scores)
        except:
            roc_auc = np.nan

        results.append(
            {
                "Method": method_name,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1_score,
                "Accuracy": accuracy,
                "ROC_AUC": roc_auc,
                "True_Positives": tp,
                "False_Positives": fp,
                "True_Negatives": tn,
                "False_Negatives": fn,
            }
        )

    results_df = pd.DataFrame(results)

    print("Performance Metrics:")
    display_df = results_df[
        ["Method", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    ].round(3)
    print(display_df.to_string(index=False))

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = results_df["Method"]

    # Precision, Recall, F1
    axes[0, 0].bar(methods, results_df["Precision"], alpha=0.7, label="Precision")
    axes[0, 0].bar(methods, results_df["Recall"], alpha=0.7, label="Recall")
    axes[0, 0].bar(methods, results_df["F1_Score"], alpha=0.7, label="F1-Score")
    axes[0, 0].set_title("Precision, Recall, F1-Score")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # ROC AUC
    valid_auc = ~pd.isna(results_df["ROC_AUC"])
    if valid_auc.any():
        axes[0, 1].bar(
            methods[valid_auc], results_df.loc[valid_auc, "ROC_AUC"], alpha=0.7
        )
        axes[0, 1].axhline(
            y=0.5, color="red", linestyle="--", alpha=0.5, label="Random"
        )
        axes[0, 1].set_title("ROC AUC")
        axes[0, 1].set_ylabel("AUC Score")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

    # Confusion Matrix Summary (True Positives vs False Positives)
    axes[1, 0].bar(
        methods, results_df["True_Positives"], alpha=0.7, label="True Positives"
    )
    axes[1, 0].bar(
        methods, results_df["False_Positives"], alpha=0.7, label="False Positives"
    )
    axes[1, 0].set_title("True vs False Positives")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Detection Rate vs False Alarm Rate
    detection_rate = results_df["Recall"]  # True Positive Rate
    false_alarm_rate = results_df["False_Positives"] / (
        results_df["False_Positives"] + results_df["True_Negatives"]
    )

    axes[1, 1].scatter(false_alarm_rate, detection_rate, s=100, alpha=0.7)
    for i, method in enumerate(methods):
        axes[1, 1].annotate(
            method,
            (false_alarm_rate.iloc[i], detection_rate.iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    axes[1, 1].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Random")
    axes[1, 1].set_xlabel("False Alarm Rate")
    axes[1, 1].set_ylabel("Detection Rate")
    axes[1, 1].set_title("Detection Rate vs False Alarm Rate")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results_df


def time_series_anomaly_detection(
    data: pd.DataFrame,
    timestamp_col: str,
    feature_cols: List[str],
    window_size: int = 24,
) -> Dict[str, np.ndarray]:
    """
    Apply time series specific anomaly detection methods.
    """
    print(f"\n⏰ Time Series Anomaly Detection")
    print("=" * 50)

    results = {}

    # Sort by timestamp
    data_sorted = data.sort_values(timestamp_col).reset_index(drop=True)

    for feature in feature_cols:
        values = data_sorted[feature].values

        # 1. Rolling statistics anomalies
        rolling_mean = (
            pd.Series(values).rolling(window=window_size, min_periods=1).mean()
        )
        rolling_std = pd.Series(values).rolling(window=window_size, min_periods=1).std()

        # Z-score based on rolling statistics
        z_scores = np.abs((values - rolling_mean) / (rolling_std + 1e-8))
        rolling_anomalies = (z_scores > 3.0).astype(int)

        results[f"{feature}_rolling"] = rolling_anomalies

        # 2. Rate of change anomalies
        diff_values = np.diff(values, prepend=values[0])
        diff_threshold = np.percentile(np.abs(diff_values), 95)  # 95th percentile
        change_anomalies = (np.abs(diff_values) > diff_threshold).astype(int)

        results[f"{feature}_change"] = change_anomalies

        # 3. Seasonal decomposition anomalies (if enough data)
        if len(values) >= 2 * window_size:
            try:
                # Simple seasonal decomposition
                seasonal_period = min(window_size, len(values) // 4)

                # Calculate seasonal component
                seasonal = np.zeros_like(values)
                for i in range(len(values)):
                    seasonal_indices = range(
                        i % seasonal_period, len(values), seasonal_period
                    )
                    seasonal_values = values[seasonal_indices]
                    seasonal[i] = np.mean(seasonal_values)

                # Residual component
                residual = values - seasonal

                # Anomalies in residual
                residual_threshold = np.percentile(np.abs(residual), 95)
                seasonal_anomalies = (np.abs(residual) > residual_threshold).astype(int)

                results[f"{feature}_seasonal"] = seasonal_anomalies

            except Exception as e:
                print(f"⚠️ Seasonal decomposition failed for {feature}: {e}")

    # Combine anomalies across features
    all_anomalies = []
    for feature in feature_cols:
        if f"{feature}_rolling" in results:
            all_anomalies.append(results[f"{feature}_rolling"])

    if all_anomalies:
        # Any feature anomaly
        combined_any = np.logical_or.reduce(all_anomalies).astype(int)
        results["combined_any"] = combined_any

        # Majority vote
        anomaly_votes = np.sum(all_anomalies, axis=0)
        combined_majority = (anomaly_votes > len(all_anomalies) // 2).astype(int)
        results["combined_majority"] = combined_majority

    print(f"• Processed {len(feature_cols)} features")
    print(f"• Window size: {window_size}")
    print(f"• Generated {len(results)} anomaly indicators")

    return results


def run_anomaly_detection_challenges() -> None:
    """
    Run all anomaly detection challenges.
    """
    print("🚀 Starting Level 6 Challenge 2: Anomaly Detection & Outlier Analysis")
    print("=" * 60)

    try:
        # Challenge 1: Create anomaly datasets
        print("\n" + "=" * 50)
        print("CHALLENGE 1: Anomaly Dataset Creation")
        print("=" * 50)

        datasets = create_anomaly_datasets()

        print(f"\n✅ Created {len(datasets)} anomaly detection datasets:")
        for name, info in datasets.items():
            data = info["data"]
            target_col = info["target_column"]
            anomaly_rate = info["anomaly_rate"]

            n_anomalies = data[target_col].sum()
            print(
                f"• {name}: {len(data)} samples, {n_anomalies} anomalies ({anomaly_rate*100:.1f}%)"
            )
            print(f"  Features: {len(info['feature_columns'])}")

        # Challenge 2: Statistical anomaly detection
        print("\n" + "=" * 50)
        print("CHALLENGE 2: Statistical Anomaly Detection")
        print("=" * 50)

        # Apply statistical methods to credit fraud dataset
        credit_data = datasets["credit_fraud"]
        X_credit = credit_data["data"]
        y_true_credit = X_credit[credit_data["target_column"]].values
        feature_cols_credit = credit_data["feature_columns"]

        stat_results = {}

        # Z-score method
        zscore_pred, zscore_scores = statistical_anomaly_detection(
            X_credit, feature_cols_credit, method="zscore", threshold=2.5
        )
        stat_results["Z_Score"] = (zscore_pred, zscore_scores)

        # IQR method
        iqr_pred, iqr_scores = statistical_anomaly_detection(
            X_credit, feature_cols_credit, method="iqr"
        )
        stat_results["IQR"] = (iqr_pred, iqr_scores)

        # Modified Z-score
        mod_zscore_pred, mod_zscore_scores = statistical_anomaly_detection(
            X_credit, feature_cols_credit, method="modified_zscore", threshold=3.5
        )
        stat_results["Modified_Z_Score"] = (mod_zscore_pred, mod_zscore_scores)

        # Challenge 3: Machine learning anomaly detection
        print("\n" + "=" * 50)
        print("CHALLENGE 3: Machine Learning Anomaly Detection")
        print("=" * 50)

        # Apply ML methods to manufacturing dataset
        manufacturing_data = datasets["manufacturing_quality"]
        X_manufacturing = manufacturing_data["data"]
        y_true_manufacturing = X_manufacturing[
            manufacturing_data["target_column"]
        ].values
        feature_cols_manufacturing = manufacturing_data["feature_columns"]

        ml_results = ml_anomaly_detection(X_manufacturing, feature_cols_manufacturing)

        # Challenge 4: Method comparison and evaluation
        print("\n" + "=" * 50)
        print("CHALLENGE 4: Method Comparison & Evaluation")
        print("=" * 50)

        # Evaluate statistical methods on credit fraud
        print("Evaluating Statistical Methods on Credit Fraud Dataset:")
        stat_eval = evaluate_anomaly_detection(y_true_credit, stat_results)

        print("\nEvaluating ML Methods on Manufacturing Quality Dataset:")
        ml_eval = evaluate_anomaly_detection(y_true_manufacturing, ml_results)

        # Challenge 5: Time series anomaly detection
        print("\n" + "=" * 50)
        print("CHALLENGE 5: Time Series Anomaly Detection")
        print("=" * 50)

        # Apply time series methods to server monitoring dataset
        server_data = datasets["server_monitoring"]
        X_server = server_data["data"]
        y_true_server = X_server[server_data["target_column"]].values
        timestamp_col = server_data["timestamp_column"]
        feature_cols_server = server_data["feature_columns"]

        ts_results = time_series_anomaly_detection(
            X_server, timestamp_col, feature_cols_server, window_size=24
        )

        # Evaluate time series methods
        ts_eval_results = {}
        for method_name, predictions in ts_results.items():
            # Use predictions as scores (binary, so scores = predictions)
            ts_eval_results[method_name] = (predictions, predictions.astype(float))

        print("\nEvaluating Time Series Methods on Server Monitoring Dataset:")
        ts_eval = evaluate_anomaly_detection(y_true_server, ts_eval_results)

        # Visualize time series anomalies
        plt.figure(figsize=(15, 10))

        # Plot first feature with anomalies
        feature_to_plot = feature_cols_server[0]  # CPU usage
        timestamps = X_server[timestamp_col]
        values = X_server[feature_to_plot]

        plt.subplot(2, 1, 1)
        plt.plot(timestamps, values, label=f"{feature_to_plot}", alpha=0.7)

        # Highlight true anomalies
        anomaly_mask = y_true_server == 1
        plt.scatter(
            timestamps[anomaly_mask],
            values[anomaly_mask],
            color="red",
            s=50,
            label="True Anomalies",
            zorder=5,
        )

        plt.title(f"Server {feature_to_plot} with True Anomalies")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot detected anomalies for best method
        if ts_eval_results:
            best_method = ts_eval.loc[ts_eval["F1_Score"].idxmax(), "Method"]
            detected_anomalies = ts_eval_results[best_method][0]

            plt.subplot(2, 1, 2)
            plt.plot(timestamps, values, alpha=0.7, color="blue")

            detected_mask = detected_anomalies == 1
            plt.scatter(
                timestamps[detected_mask],
                values[detected_mask],
                color="orange",
                s=30,
                label=f"Detected by {best_method}",
                zorder=5,
            )

            plt.scatter(
                timestamps[anomaly_mask],
                values[anomaly_mask],
                color="red",
                s=50,
                label="True Anomalies",
                zorder=6,
                marker="x",
            )

            plt.title(f"Anomaly Detection Results - {best_method}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\n" + "🎉" * 20)
        print("LEVEL 6 CHALLENGE 2 COMPLETE!")
        print("🎉" * 20)

        print("\n📚 What You've Learned:")
        print(
            "• Statistical anomaly detection methods (Z-score, IQR, Modified Z-score)"
        )
        print("• Machine learning approaches (Isolation Forest, One-Class SVM, LOF)")
        print("• Time series specific anomaly detection techniques")
        print("• Evaluation metrics for imbalanced anomaly detection")
        print("• Real-world anomaly patterns across different domains")

        print("\n🚀 Next Steps:")
        print("• Explore deep learning anomaly detection (Autoencoders)")
        print("• Learn about ensemble anomaly detection methods")
        print("• Study domain-specific anomaly detection techniques")
        print("• Apply to real-world monitoring and fraud detection")
        print("• Move to Level 6 Challenge 3: NLP & Text Analytics")

        return datasets

    except Exception as e:
        print(f"❌ Error in anomaly detection challenges: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete anomaly detection challenge
    datasets = run_anomaly_detection_challenges()

    if datasets:
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION CHALLENGE SUMMARY")
        print("=" * 60)

        print("\nDatasets Created:")
        for name, info in datasets.items():
            data = info["data"]
            target_col = info["target_column"]
            n_anomalies = data[target_col].sum()
            print(f"• {name}: {len(data)} samples, {n_anomalies} anomalies")

        print("\nKey Anomaly Detection Concepts Covered:")
        concepts = [
            "Statistical methods for outlier detection",
            "Machine learning approaches for novelty detection",
            "Time series anomaly detection techniques",
            "Evaluation metrics for imbalanced anomaly detection",
            "Domain-specific anomaly patterns and applications",
            "Comparison of detection methods across different data types",
        ]

        for i, concept in enumerate(concepts, 1):
            print(f"{i}. {concept}")

        print("\n✨ Ready for Level 6 Challenge 3: NLP & Text Analytics!")
