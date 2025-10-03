#!/usr/bin/env python3
"""
Level 5: Deep Learning Dynamo
Challenge 1: Neural Network Fundamentals

Master neural network architectures from perceptrons to deep networks.
"""

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Try to import TensorFlow/Keras, but continue if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, callbacks

    TENSORFLOW_AVAILABLE = True
    print("🚀 TensorFlow available - Full deep learning capabilities enabled!")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print(
        "📚 Using scikit-learn neural networks - TensorFlow features will be simulated"
    )

warnings.filterwarnings("ignore")


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    if TENSORFLOW_AVAILABLE:
        tf.random.set_seed(seed)


def create_neural_network_datasets():
    """Create various datasets to demonstrate neural network capabilities"""
    print("🧠 Creating Neural Network Datasets...")

    datasets = {}

    # 1. Linearly separable dataset (simple)
    X_linear, y_linear = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42,
    )
    datasets["linear"] = (X_linear, y_linear, "Linear Classification")

    # 2. Moons dataset (non-linear)
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.1, random_state=42)
    datasets["moons"] = (X_moons, y_moons, "Moons (Non-linear)")

    # 3. Circles dataset (complex non-linear)
    X_circles, y_circles = make_circles(
        n_samples=1000, noise=0.05, factor=0.6, random_state=42
    )
    datasets["circles"] = (X_circles, y_circles, "Circles (Complex Non-linear)")

    # 4. Multi-class complex dataset
    X_complex, y_complex = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=42,
    )
    datasets["complex"] = (X_complex, y_complex, "High-dimensional Multi-class")

    print(f"Created {len(datasets)} datasets for neural network training")
    return datasets


def visualize_datasets(datasets):
    """Visualize 2D datasets to understand classification challenges"""
    print("\n📊 Visualizing Dataset Challenges...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    plot_datasets = ["linear", "moons", "circles"]

    for i, (name, (X, y, title)) in enumerate(
        [(k, v) for k, v in datasets.items() if k in plot_datasets]
    ):
        if X.shape[1] == 2:  # Only plot 2D datasets
            scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.7)
            axes[i].set_title(
                f"{title}\\n{X.shape[0]} samples, {len(np.unique(y))} classes"
            )
            axes[i].set_xlabel("Feature 1")
            axes[i].set_ylabel("Feature 2")
            axes[i].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[i])

    # Summary plot for high-dimensional dataset
    X_complex, y_complex, title = datasets["complex"]
    axes[3].hist(
        [X_complex[y_complex == i, 0] for i in range(len(np.unique(y_complex)))],
        bins=20,
        alpha=0.7,
        label=[f"Class {i}" for i in range(len(np.unique(y_complex)))],
    )
    axes[3].set_title(
        f"{title}\\n{X_complex.shape[0]} samples, {X_complex.shape[1]} features"
    )
    axes[3].set_xlabel("Feature 1 Distribution")
    axes[3].set_ylabel("Frequency")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def single_perceptron_demo():
    """Demonstrate single perceptron limitations"""
    print("\n🔍 Single Perceptron Limitations Demo...")

    # Create XOR problem (not linearly separable)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])  # XOR function

    # Try single perceptron (linear)
    perceptron = MLPClassifier(hidden_layer_sizes=(), max_iter=1000, random_state=42)

    try:
        perceptron.fit(X_xor, y_xor)
        predictions = perceptron.predict(X_xor)
        accuracy = accuracy_score(y_xor, predictions)

        print(f"Single Perceptron on XOR:")
        print(f"  Input: {X_xor.tolist()}")
        print(f"  Expected: {y_xor.tolist()}")
        print(f"  Predicted: {predictions.tolist()}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(
            f"  Result: {'✅ Success' if accuracy > 0.9 else '❌ Failed (as expected)'}"
        )

    except Exception as e:
        print(f"Single perceptron failed on XOR: {e}")

    return X_xor, y_xor


def multi_layer_neural_networks(datasets):
    """Demonstrate multi-layer neural networks on various datasets"""
    print("\n🧠 Multi-Layer Neural Networks Analysis...")

    results = {}

    # Different network architectures to test
    architectures = {
        "Small Network": (10,),
        "Medium Network": (50, 20),
        "Large Network": (100, 50, 20),
        "Deep Network": (64, 32, 16, 8),
    }

    for dataset_name, (X, y, title) in datasets.items():
        print(f"\\n--- Testing on {title} ---")

        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        dataset_results = {}

        for arch_name, hidden_layers in architectures.items():
            # Create and train neural network
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )

            mlp.fit(X_train_scaled, y_train)

            # Evaluate
            train_accuracy = mlp.score(X_train_scaled, y_train)
            test_accuracy = mlp.score(X_test_scaled, y_test)

            dataset_results[arch_name] = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "n_layers": len(hidden_layers),
                "total_params": (
                    sum(hidden_layers) + hidden_layers[0] if hidden_layers else 0
                ),
            }

            print(
                f"  {arch_name:15} -> Train: {train_accuracy:.3f}, Test: {test_accuracy:.3f}"
            )

        results[dataset_name] = dataset_results

    return results


def learning_curve_analysis(datasets):
    """Analyze learning curves for neural networks"""
    print("\n📈 Neural Network Learning Curve Analysis...")

    # Focus on the most interesting dataset
    X, y, title = datasets["circles"]

    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define different network sizes
    network_configs = {
        "Small (10)": (10,),
        "Medium (50,20)": (50, 20),
        "Large (100,50)": (100, 50),
    }

    plt.figure(figsize=(15, 5))

    for i, (config_name, hidden_layers) in enumerate(network_configs.items()):
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        )

        # Learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            mlp,
            X_scaled,
            y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42,
        )

        # Plot learning curve
        plt.subplot(1, 3, i + 1)

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue",
        )

        plt.plot(train_sizes, val_mean, "o-", color="red", label="Validation Score")
        plt.fill_between(
            train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red"
        )

        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy Score")
        plt.title(f"Learning Curve: {config_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Diagnose overfitting
        final_gap = train_mean[-1] - val_mean[-1]
        if final_gap > 0.1:
            diagnosis = "Overfitting"
        elif val_mean[-1] < 0.7:
            diagnosis = "Underfitting"
        else:
            diagnosis = "Good Fit"

        plt.text(
            0.02,
            0.02,
            f"Status: {diagnosis}",
            transform=plt.gca().transAxes,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    plt.tight_layout()
    plt.show()


def hyperparameter_optimization_demo(datasets):
    """Demonstrate neural network hyperparameter optimization"""
    print("\n🔧 Hyperparameter Optimization for Neural Networks...")

    # Use circles dataset for demonstration
    X, y, _ = datasets["circles"]

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameters to test
    param_tests = {
        "Hidden Layer Size": {
            "param": "hidden_layer_sizes",
            "values": [(10,), (20,), (50,), (100,), (50, 20), (100, 50)],
        },
        "Learning Rate": {
            "param": "learning_rate_init",
            "values": [0.0001, 0.001, 0.01, 0.1],
        },
        "Activation Function": {
            "param": "activation",
            "values": ["relu", "tanh", "logistic"],
        },
    }

    results = {}

    for param_name, param_config in param_tests.items():
        print(f"\\n--- Testing {param_name} ---")

        param_results = []

        for value in param_config["values"]:
            # Create base model
            mlp_params = {
                "hidden_layer_sizes": (50, 20),
                "activation": "relu",
                "learning_rate_init": 0.001,
                "max_iter": 500,
                "random_state": 42,
                "early_stopping": True,
            }

            # Update the parameter being tested
            mlp_params[param_config["param"]] = value

            # Train model
            mlp = MLPClassifier(**mlp_params)
            mlp.fit(X_train_scaled, y_train)

            # Evaluate
            test_accuracy = mlp.score(X_test_scaled, y_test)
            param_results.append((value, test_accuracy))

            print(f"  {param_config['param']}={value} -> Accuracy: {test_accuracy:.3f}")

        results[param_name] = param_results

    # Visualize results
    plt.figure(figsize=(15, 5))

    for i, (param_name, param_results) in enumerate(results.items()):
        plt.subplot(1, 3, i + 1)

        values, scores = zip(*param_results)

        if isinstance(values[0], tuple):  # Handle hidden layer sizes
            labels = [str(v) for v in values]
            x_pos = range(len(labels))
            plt.bar(x_pos, scores)
            plt.xticks(x_pos, labels, rotation=45)
        else:
            plt.plot(values, scores, "o-")
            if param_name == "Learning Rate":
                plt.xscale("log")

        plt.xlabel(param_name)
        plt.ylabel("Test Accuracy")
        plt.title(f"Hyperparameter Tuning: {param_name}")
        plt.grid(True, alpha=0.3)

        # Mark best value
        best_idx = np.argmax(scores)
        best_value = values[best_idx]
        best_score = scores[best_idx]

        if isinstance(best_value, tuple):
            plt.axvline(x=best_idx, color="red", linestyle="--", alpha=0.7)
        else:
            plt.axvline(x=best_value, color="red", linestyle="--", alpha=0.7)

        plt.text(
            0.02,
            0.98,
            f"Best: {best_value}\\nScore: {best_score:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.8},
        )

    plt.tight_layout()
    plt.show()

    return results


def tensorflow_neural_networks_demo():
    """Demonstrate TensorFlow/Keras neural networks if available"""
    if not TENSORFLOW_AVAILABLE:
        print("\\n📚 TensorFlow Neural Networks (Simulated)...")
        print("Install TensorFlow to enable full deep learning capabilities:")
        print("  pip install tensorflow")

        print("\\nTensorFlow would provide:")
        print("  🚀 Custom neural network architectures")
        print("  📊 Advanced optimizers (Adam, RMSprop, etc.)")
        print("  🔧 Flexible layer types (Dense, Dropout, BatchNorm)")
        print("  📈 Built-in callbacks (EarlyStopping, ReduceLROnPlateau)")
        print("  💾 Model saving and loading")
        print("  🎯 Advanced metrics and loss functions")

        return None

    print("\\n🚀 TensorFlow/Keras Neural Networks Demo...")

    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build TensorFlow model
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("Model Architecture:")
    model.summary()

    # Train with callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    # Evaluate
    train_accuracy = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
    test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)[1]

    print(f"\\nResults:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return model, history


def neural_network_interpretation():
    """Demonstrate neural network interpretation techniques"""
    print("\\n🔍 Neural Network Interpretation & Analysis...")

    # Create simple dataset for interpretation
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train neural network
    mlp = MLPClassifier(
        hidden_layer_sizes=(50, 20),
        activation="relu",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
    )

    mlp.fit(X_train_scaled, y_train)

    print(f"Neural Network Performance:")
    print(f"  Training Accuracy: {mlp.score(X_train_scaled, y_train):.4f}")
    print(f"  Test Accuracy: {mlp.score(X_test_scaled, y_test):.4f}")

    # Feature importance through permutation
    print("\\nCalculating feature importance...")

    baseline_accuracy = mlp.score(X_test_scaled, y_test)
    feature_importance = []

    for i, feature_name in enumerate(feature_names):
        # Permute feature i
        X_permuted = X_test_scaled.copy()
        np.random.shuffle(X_permuted[:, i])

        # Calculate accuracy drop
        permuted_accuracy = mlp.score(X_permuted, y_test)
        importance = baseline_accuracy - permuted_accuracy
        feature_importance.append((feature_name, importance))

    # Sort by importance
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print("\\nFeature Importance (Permutation Method):")
    for feature, importance in feature_importance[:5]:
        print(f"  {feature}: {importance:.4f}")

    # Visualize feature importance
    plt.figure(figsize=(10, 6))

    features, importances = zip(*feature_importance)

    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance (Accuracy Drop)")
    plt.title("Neural Network Feature Importance")
    plt.grid(True, alpha=0.3)

    # Highlight most important features
    max_importance = max(importances)
    colors = ["red" if imp > max_importance * 0.7 else "blue" for imp in importances]

    bars = plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

    return feature_importance


def main():
    """Main function to run the neural network fundamentals challenge"""
    print("=" * 60)
    print("LEVEL 5 CHALLENGE 1: NEURAL NETWORK FUNDAMENTALS")
    print("=" * 60)

    print("🧠 Welcome to Deep Learning!")
    print("Master neural networks from basic perceptrons to deep architectures.")

    # Set random seeds for reproducibility
    set_random_seeds()

    # 1. Create and visualize datasets
    datasets = create_neural_network_datasets()
    visualize_datasets(datasets)

    # 2. Single perceptron limitations
    X_xor, y_xor = single_perceptron_demo()

    # 3. Multi-layer neural networks
    nn_results = multi_layer_neural_networks(datasets)

    # 4. Learning curve analysis
    learning_curve_analysis(datasets)

    # 5. Hyperparameter optimization
    hyperopt_results = hyperparameter_optimization_demo(datasets)

    # 6. TensorFlow demo (if available)
    tf_model, tf_history = tensorflow_neural_networks_demo()

    # 7. Neural network interpretation
    interpretation_results = neural_network_interpretation()

    # Summary
    print("\\n" + "=" * 60)
    print("CHALLENGE 1 COMPLETION SUMMARY")
    print("=" * 60)

    print("Neural Network concepts mastered:")
    concepts = [
        "🧠 Perceptron vs Multi-layer networks",
        "🎯 Non-linear classification capabilities",
        "🏗️ Architecture design (hidden layers, neurons)",
        "⚙️ Activation functions (ReLU, tanh, sigmoid)",
        "🔧 Hyperparameter optimization strategies",
        "📈 Learning curves and overfitting analysis",
        "🔍 Feature importance and model interpretation",
        "📊 Performance evaluation and validation",
    ]

    if TENSORFLOW_AVAILABLE:
        concepts.extend(
            [
                "🚀 TensorFlow/Keras implementation",
                "📋 Advanced callbacks and regularization",
                "💾 Model architecture and training",
            ]
        )

    for concept in concepts:
        print(f"  {concept}")

    print(f"\\nDatasets analyzed:")
    for name, (X, y, title) in datasets.items():
        print(f"  • {title}: {X.shape[0]} samples, {X.shape[1]} features")

    print(f"\\nArchitectures tested:")
    for arch_name in [
        "Small Network",
        "Medium Network",
        "Large Network",
        "Deep Network",
    ]:
        print(f"  • {arch_name}: Comprehensive evaluation")

    # Best performing architecture
    if nn_results:
        best_performance = {}
        for dataset_name, dataset_results in nn_results.items():
            best_arch = max(
                dataset_results.keys(),
                key=lambda k: dataset_results[k]["test_accuracy"],
            )
            best_acc = dataset_results[best_arch]["test_accuracy"]
            best_performance[dataset_name] = (best_arch, best_acc)

        print(f"\\nBest architectures per dataset:")
        for dataset_name, (arch, acc) in best_performance.items():
            print(f"  • {datasets[dataset_name][2]}: {arch} ({acc:.3f} accuracy)")

    print("\\n🎉 Congratulations! You've mastered neural network fundamentals!")
    print("Ready for Challenge 2: Convolutional Neural Networks")

    return {
        "datasets": datasets,
        "nn_results": nn_results,
        "hyperopt_results": hyperopt_results,
        "interpretation_results": interpretation_results,
        "tf_model": tf_model,
    }


if __name__ == "__main__":
    results = main()

    print("\\n" + "=" * 60)
    print("CHALLENGE 1 STATUS: COMPLETE")
    print("=" * 60)
    print("Neural network fundamentals mastered!")
    print("Ready for Challenge 2: Convolutional Neural Networks.")
