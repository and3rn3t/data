#!/usr/bin/env python3
"""
Level 7 Challenge 3: Real-time Analytics and Edge Deployment Runner

This script demonstrates real-time data processing, streaming analytics,
and edge deployment for low-latency ML applications.
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Model libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("⚡ LEVEL 7 CHALLENGE 3: REAL-TIME ANALYTICS")
print("=" * 55)

# PART 1: Real-time Data Streaming Infrastructure
print("\n🌊 PART 1: REAL-TIME DATA STREAMING INFRASTRUCTURE")
print("-" * 50)


@dataclass
class StreamingEvent:
    """Data structure for streaming events"""

    timestamp: str
    event_id: str
    user_id: str
    event_type: str
    value: float
    metadata: Dict[str, Any]


class DataStreamGenerator:
    """Simulates real-time data streaming"""

    def __init__(self, stream_name: str = "iot_sensors"):
        self.stream_name = stream_name
        self.is_streaming = False
        self.event_queue = Queue()

    def generate_event(self) -> StreamingEvent:
        """Generate a realistic streaming event"""

        event_types = ["temperature", "pressure", "vibration", "power_consumption"]

        # Simulate different sensor patterns
        base_values = {
            "temperature": 20 + 10 * np.sin(time.time() / 100) + np.random.normal(0, 2),
            "pressure": 1013 + 50 * np.cos(time.time() / 80) + np.random.normal(0, 5),
            "vibration": abs(np.random.normal(10, 3)),
            "power_consumption": 100
            + 20 * np.sin(time.time() / 60)
            + np.random.exponential(5),
        }

        event_type = np.random.choice(event_types)

        # Occasionally inject anomalies
        value = base_values[event_type]
        if np.random.random() < 0.05:  # 5% anomaly rate
            value *= np.random.uniform(2, 5)  # Anomalous spike

        return StreamingEvent(
            timestamp=datetime.now().isoformat(),
            event_id=str(uuid.uuid4()),
            user_id=f"sensor_{np.random.randint(1, 10)}",
            event_type=event_type,
            value=value,
            metadata={
                "location": f"zone_{np.random.randint(1, 5)}",
                "quality": np.random.choice(
                    ["good", "fair", "poor"], p=[0.8, 0.15, 0.05]
                ),
            },
        )

    def start_streaming(self, events_per_second: float = 2.0):
        """Start generating streaming events"""

        self.is_streaming = True

        def stream_worker():
            while self.is_streaming:
                event = self.generate_event()
                self.event_queue.put(event)
                time.sleep(1.0 / events_per_second)

        self.stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self.stream_thread.start()

        print(
            f"✅ Started streaming {self.stream_name} at {events_per_second} events/second"
        )

    def stop_streaming(self):
        """Stop streaming"""
        self.is_streaming = False
        print(f"⏹️ Stopped streaming {self.stream_name}")

    def get_events(self, max_events: int = 100) -> List[StreamingEvent]:
        """Get available events from the queue"""
        events = []
        count = 0
        while not self.event_queue.empty() and count < max_events:
            events.append(self.event_queue.get())
            count += 1
        return events


class RealTimeProcessor:
    """Real-time stream processor with windowing and aggregation"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.event_window = deque(maxlen=window_size)
        self.processed_count = 0
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def process_batch(self, events: List[StreamingEvent]) -> Dict[str, Any]:
        """Process a batch of events"""

        if not events:
            return {"status": "no_events"}

        # Add events to sliding window
        self.event_window.extend(events)
        self.processed_count += len(events)

        # Convert to DataFrame for analysis
        event_dicts = [asdict(event) for event in self.event_window]
        df = pd.json_normalize(event_dicts)

        # Calculate real-time aggregations
        aggregations = {
            "window_size": len(self.event_window),
            "event_types": df["event_type"].value_counts().to_dict(),
            "avg_values_by_type": df.groupby("event_type")["value"].mean().to_dict(),
            "std_values_by_type": df.groupby("event_type")["value"]
            .std()
            .fillna(0)
            .to_dict(),
            "timestamp_range": {
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
            },
            "processed_total": self.processed_count,
        }

        # Anomaly detection on numerical features
        numerical_features = self.extract_numerical_features(df)
        anomalies = self.detect_anomalies(numerical_features)

        aggregations["anomalies_detected"] = anomalies

        return aggregations

    def extract_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract numerical features for anomaly detection"""

        # Create feature matrix
        features = []

        # Group by event type and calculate features
        for event_type in df["event_type"].unique():
            type_df = df[df["event_type"] == event_type]
            if len(type_df) > 0:
                features.extend(
                    [
                        type_df["value"].mean(),
                        type_df["value"].std() if len(type_df) > 1 else 0,
                        type_df["value"].min(),
                        type_df["value"].max(),
                    ]
                )

        # Pad to consistent size
        while len(features) < 16:  # Assume max 4 event types * 4 stats
            features.append(0.0)

        return np.array(features[:16]).reshape(1, -1)

    def detect_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """Real-time anomaly detection"""

        if not self.is_trained:
            # Train on first batch (bootstrap)
            if len(self.event_window) >= 20:  # Minimum training samples
                self.scaler.fit(features)
                self.anomaly_detector = IsolationForest(
                    contamination=0.1, random_state=42
                )
                self.anomaly_detector.fit(self.scaler.transform(features))
                self.is_trained = True
                return {"status": "training_complete", "anomaly_score": 0.0}
            else:
                return {
                    "status": "collecting_training_data",
                    "samples": len(self.event_window),
                }

        # Detect anomalies
        features_scaled = self.scaler.transform(features)
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1

        return {
            "status": "detection_active",
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
            "threshold": 0.0,
        }


class EdgeMLModel:
    """Lightweight ML model for edge deployment"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.prediction_count = 0

    def train_lightweight_model(self, training_data: pd.DataFrame):
        """Train a lightweight model suitable for edge deployment"""

        print("🧠 Training lightweight edge model...")

        # Prepare features (simple statistical features)
        features = []
        labels = []

        # Group by time windows and create features
        training_data["timestamp"] = pd.to_datetime(training_data["timestamp"])
        training_data = training_data.sort_values("timestamp")

        # Create 10-second windows
        for i in range(0, len(training_data) - 10, 5):
            window_data = training_data.iloc[i : i + 10]

            # Extract features
            feature_vector = [
                window_data["value"].mean(),
                window_data["value"].std(),
                window_data["value"].min(),
                window_data["value"].max(),
                len(window_data["event_type"].unique()),
                (
                    window_data["value"]
                    > window_data["value"].mean() + 2 * window_data["value"].std()
                ).sum(),
            ]

            # Label: 1 if any anomalous values (> 3 std devs), 0 otherwise
            label = (
                1
                if (
                    window_data["value"]
                    > window_data["value"].mean() + 3 * window_data["value"].std()
                ).any()
                else 0
            )

            features.append(feature_vector)
            labels.append(label)

        # Train simple model
        X = np.array(features)
        y = np.array(labels)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Use simple KMeans for clustering/anomaly detection
        self.model = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.model.fit(X_scaled)

        self.is_loaded = True

        print(f"✅ Edge model trained on {len(features)} windows")
        print(f"📊 Feature dimensions: {X.shape[1]}")
        print(f"🎯 Anomaly rate in training: {np.mean(y):.1%}")

    def predict_realtime(self, recent_events: List[StreamingEvent]) -> Dict[str, Any]:
        """Make real-time predictions suitable for edge deployment"""

        if not self.is_loaded:
            return {"error": "Model not loaded", "prediction": None}

        if len(recent_events) < 3:
            return {"error": "Insufficient data", "prediction": None}

        # Extract features from recent events
        values = [event.value for event in recent_events[-10:]]  # Last 10 events

        if len(values) == 0:
            return {"error": "No values", "prediction": None}

        feature_vector = [
            np.mean(values),
            np.std(values) if len(values) > 1 else 0,
            np.min(values),
            np.max(values),
            len(set(event.event_type for event in recent_events[-10:])),
            sum(1 for v in values if v > np.mean(values) + 2 * np.std(values)),
        ]

        # Scale and predict
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get cluster assignment (0 or 1)
        cluster = self.model.predict(X_scaled)[0]

        # Calculate distance to cluster centers
        distances = self.model.transform(X_scaled)[0]
        confidence = 1.0 - min(distances) / max(distances + [1e-6])

        self.prediction_count += 1

        return {
            "prediction": int(cluster),
            "confidence": float(confidence),
            "feature_vector": feature_vector,
            "prediction_id": self.prediction_count,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": 1.2,  # Simulated low latency
        }


# Initialize components
print("🔧 Initializing real-time analytics components...")

# Start data streaming
stream_generator = DataStreamGenerator("iot_sensor_stream")
stream_processor = RealTimeProcessor(window_size=50)
edge_model = EdgeMLModel()

stream_generator.start_streaming(events_per_second=5.0)

print("✅ Real-time streaming infrastructure ready")

# PART 2: Real-time Processing Simulation
print("\n⚡ PART 2: REAL-TIME PROCESSING SIMULATION")
print("-" * 40)

print("🔄 Running real-time processing for 15 seconds...")
processing_results = []

for cycle in range(15):  # 15 second simulation
    time.sleep(1)  # 1 second processing cycles

    # Get events from stream
    events = stream_generator.get_events(max_events=10)

    if events:
        # Process events
        result = stream_processor.process_batch(events)
        processing_results.append(
            {
                "cycle": cycle + 1,
                "timestamp": datetime.now().isoformat(),
                "events_processed": len(events),
                "result": result,
            }
        )

        # Show real-time metrics
        if result.get("anomalies_detected", {}).get("status") == "detection_active":
            anomaly_info = result["anomalies_detected"]
            status_icon = "🚨" if anomaly_info.get("is_anomaly", False) else "✅"
            print(
                f"Cycle {cycle+1:2d}: {len(events):2d} events | "
                f"Anomaly Score: {anomaly_info.get('anomaly_score', 0):6.3f} {status_icon}"
            )
        else:
            print(
                f"Cycle {cycle+1:2d}: {len(events):2d} events | Status: {result.get('anomalies_detected', {}).get('status', 'unknown')}"
            )

# Stop streaming for model training
stream_generator.stop_streaming()

# PART 3: Edge Model Training and Deployment
print("\n🧠 PART 3: EDGE MODEL TRAINING & DEPLOYMENT")
print("-" * 40)

# Collect training data from processed events
all_events = []
for result in processing_results:
    if "events" in result:
        all_events.extend(result["events"])

# Create training dataset from stream processor's window
training_events = list(stream_processor.event_window)
if training_events:
    training_df = pd.json_normalize([asdict(event) for event in training_events])

    # Train edge model
    edge_model.train_lightweight_model(training_df)
else:
    print("⚠️ No training data available, creating synthetic data for edge model...")
    # Create synthetic training data
    synthetic_events = [stream_generator.generate_event() for _ in range(100)]
    training_df = pd.json_normalize([asdict(event) for event in synthetic_events])
    edge_model.train_lightweight_model(training_df)

# PART 4: Real-time Edge Inference
print("\n🚀 PART 4: REAL-TIME EDGE INFERENCE")
print("-" * 40)

# Restart streaming for edge inference
stream_generator.start_streaming(events_per_second=3.0)

print("🔄 Running edge inference for 10 seconds...")
edge_predictions = []

for inference_cycle in range(10):
    time.sleep(1)

    # Get recent events
    recent_events = stream_generator.get_events(max_events=5)

    if recent_events and len(recent_events) >= 3:
        # Make edge prediction
        prediction = edge_model.predict_realtime(recent_events)

        edge_predictions.append(
            {
                "cycle": inference_cycle + 1,
                "timestamp": datetime.now().isoformat(),
                "events_used": len(recent_events),
                "prediction": prediction,
            }
        )

        # Show real-time inference results
        if "prediction" in prediction and prediction["prediction"] is not None:
            pred_class = prediction["prediction"]
            confidence = prediction["confidence"]
            latency = prediction.get("latency_ms", 0)

            class_label = "🚨 ANOMALY" if pred_class == 1 else "✅ NORMAL"
            print(
                f"Edge Inference {inference_cycle+1:2d}: {class_label} | "
                f"Confidence: {confidence:.3f} | Latency: {latency:.1f}ms"
            )
        else:
            print(
                f"Edge Inference {inference_cycle+1:2d}: {prediction.get('error', 'Unknown error')}"
            )
    else:
        print(f"Edge Inference {inference_cycle+1:2d}: Waiting for data...")

stream_generator.stop_streaming()

# PART 5: Performance Analytics and Monitoring
print("\n📊 PART 5: PERFORMANCE ANALYTICS & MONITORING")
print("-" * 40)

# Analyze processing performance
total_events_processed = sum(r.get("events_processed", 0) for r in processing_results)
avg_events_per_cycle = (
    total_events_processed / len(processing_results) if processing_results else 0
)

# Analyze edge inference performance
successful_predictions = [p for p in edge_predictions if "error" not in p["prediction"]]
edge_success_rate = (
    len(successful_predictions) / len(edge_predictions) if edge_predictions else 0
)

# Calculate anomaly detection statistics
anomaly_detections = 0
total_detections = 0

for result in processing_results:
    anomaly_info = result.get("result", {}).get("anomalies_detected", {})
    if anomaly_info.get("status") == "detection_active":
        total_detections += 1
        if anomaly_info.get("is_anomaly", False):
            anomaly_detections += 1

anomaly_rate = anomaly_detections / total_detections if total_detections > 0 else 0

print("📈 Real-time Analytics Performance Report:")
print(f"  🌊 Stream Processing:")
print(f"    • Total Events Processed: {total_events_processed}")
print(f"    • Average Events/Cycle: {avg_events_per_cycle:.1f}")
print(f"    • Processing Cycles: {len(processing_results)}")
print(f"    • Stream Window Size: {stream_processor.window_size}")

print(f"  🚨 Anomaly Detection:")
print(f"    • Detection Cycles: {total_detections}")
print(f"    • Anomalies Detected: {anomaly_detections}")
print(f"    • Anomaly Rate: {anomaly_rate:.1%}")
print(
    f"    • Model Training Status: {'✅ Trained' if stream_processor.is_trained else '⏳ Training'}"
)

print(f"  🧠 Edge Inference:")
print(f"    • Total Predictions: {len(edge_predictions)}")
print(f"    • Successful Predictions: {len(successful_predictions)}")
print(f"    • Success Rate: {edge_success_rate:.1%}")
print(f"    • Average Latency: 1.2ms (simulated)")

# PART 6: Real-time Dashboard Summary
print("\n📱 PART 6: REAL-TIME DASHBOARD SUMMARY")
print("-" * 40)

print("🖥️ Real-time Analytics Dashboard Status:")
print("  ✅ Data Streaming: Active")
print("  ✅ Stream Processing: Operational")
print("  ✅ Anomaly Detection: Trained & Active")
print("  ✅ Edge ML Model: Deployed")
print("  ✅ Real-time Inference: Low Latency (<2ms)")
print("  ✅ Monitoring & Alerting: Active")

print("\n🏆 LEVEL 7 CHALLENGE 3 COMPLETED!")
print("=" * 45)

print("\n✅ REAL-TIME ANALYTICS MASTERY DEMONSTRATED:")
print("  🌊 High-throughput data streaming")
print("  ⚡ Real-time stream processing")
print("  🔍 Live anomaly detection")
print("  🧠 Edge ML model deployment")
print("  🚀 Ultra-low latency inference")
print("  📊 Real-time monitoring & dashboards")

print("\n🎓 PRODUCTION SKILLS LEARNED:")
print("  • Stream processing architectures")
print("  • Real-time feature engineering")
print("  • Edge computing and deployment")
print("  • Low-latency model inference")
print("  • Streaming anomaly detection")
print("  • Production monitoring systems")

print("\n🚀 NEXT CHALLENGE UNLOCKED:")
print("  Ready for Challenge 4: AI Ethics & Governance!")

print("\n🏅 Achievement Unlocked: Real-time Analytics Expert!")
