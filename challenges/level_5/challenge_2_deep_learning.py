"""
Level 5 Challenge 2: Deep Learning and Neural Networks
Master neural networks and deep learning architectures for complex ML tasks.
"""

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Try to import TensorFlow, but continue if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import callbacks, layers, optimizers, regularizers
    from tensorflow.keras.utils import to_categorical

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - using scikit-learn MLPs only")

warnings.filterwarnings("ignore")


def set_random_seeds():
    """Set random seeds for reproducibility"""
    np.random.seed(42)
    if TENSORFLOW_AVAILABLE:
        tf.random.set_seed(42)


def create_pattern_dataset(n_samples=3000, pattern_size=20):
    """Create synthetic 2D pattern dataset for neural networks"""
    print(
        f"Creating pattern recognition dataset: {n_samples} samples, {pattern_size}x{pattern_size}"
    )

    X = np.random.randn(n_samples, pattern_size, pattern_size)
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        pattern = X[i]

        # Pattern 1: Diagonal lines (class 0)
        if np.mean(np.diag(pattern)) > 0:
            y[i] = 0

        # Pattern 2: Center concentration (class 1)
        elif np.mean(pattern[5:15, 5:15]) > np.mean(pattern):
            y[i] = 1

        # Pattern 3: Edge activation (class 2)
        elif (
            np.mean(pattern[0, :])
            + np.mean(pattern[-1, :])
            + np.mean(pattern[:, 0])
            + np.mean(pattern[:, -1])
        ) > 0:
            y[i] = 2

        # Pattern 4: Quadrant patterns (class 3)
        else:
            y[i] = 3

    # Flatten for neural networks
    X_flat = X.reshape(n_samples, -1)

    print(f"Pattern dataset created: {X_flat.shape}, Classes: {np.bincount(y)}")
    return X_flat, y


def create_time_series_dataset(n_samples=2000, sequence_length=50):
    """Create synthetic time series dataset for sequence modeling"""
    print(
        f"Creating time series dataset: {n_samples} sequences, length {sequence_length}"
    )

    X = np.zeros((n_samples, sequence_length))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        # Generate different time series patterns
        t = np.linspace(0, 4 * np.pi, sequence_length)

        if i % 4 == 0:  # Sine wave with trend
            X[i] = np.sin(t) + 0.1 * t + np.random.normal(0, 0.1, sequence_length)
            y[i] = np.mean(X[i][-10:])  # Predict average of last 10 values

        elif i % 4 == 1:  # Cosine with noise
            X[i] = np.cos(t) + np.random.normal(0, 0.2, sequence_length)
            y[i] = np.std(X[i])  # Predict volatility

        elif i % 4 == 2:  # Exponential decay
            X[i] = np.exp(-t / 2) + np.random.normal(0, 0.1, sequence_length)
            y[i] = X[i][-1] - X[i][0]  # Predict total change

        else:  # Random walk
            X[i] = np.cumsum(np.random.normal(0, 1, sequence_length))
            y[i] = 1 if X[i][-1] > X[i][0] else 0  # Predict direction

    print(
        f"Time series dataset created: {X.shape}, Target range: [{y.min():.3f}, {y.max():.3f}]"
    )
    return X, y


def create_sklearn_neural_networks(
    X_train, X_test, y_train, y_test, task_type="classification"
):
    """Create and train scikit-learn neural networks"""
    print(f"Training scikit-learn neural networks for {task_type}...")

    results = {}

    if task_type == "classification":
        # Different MLP architectures for classification
        mlp_configs = {
            "Small MLP": MLPClassifier(
                hidden_layer_sizes=(50,),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
            ),
            "Deep MLP": MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
            ),
            "Wide MLP": MLPClassifier(
                hidden_layer_sizes=(200, 200),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
            ),
            "Regularized MLP": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                solver="adam",
                alpha=0.01,  # L2 regularization
                max_iter=500,
                random_state=42,
            ),
        }

        for name, model in mlp_configs.items():
            start_time = datetime.now()

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average="macro")

            training_time = (datetime.now() - start_time).total_seconds()

            results[name] = {
                "model": model,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "training_time": training_time,
                "n_iterations": model.n_iter_,
            }

            print(
                f"  {name}: Acc={test_acc:.3f}, F1={test_f1:.3f}, Iter={model.n_iter_}, Time={training_time:.2f}s"
            )

    else:  # regression
        # Different MLP architectures for regression
        mlp_configs = {
            "Small MLP": MLPRegressor(
                hidden_layer_sizes=(50,),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
            ),
            "Deep MLP": MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
            ),
            "Regularized MLP": MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                solver="adam",
                alpha=0.01,
                max_iter=500,
                random_state=42,
            ),
        }

        for name, model in mlp_configs.items():
            start_time = datetime.now()

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)

            training_time = (datetime.now() - start_time).total_seconds()

            results[name] = {
                "model": model,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "test_r2": test_r2,
                "training_time": training_time,
                "n_iterations": model.n_iter_,
            }

            print(
                f"  {name}: R²={test_r2:.3f}, MSE={test_mse:.3f}, Iter={model.n_iter_}, Time={training_time:.2f}s"
            )

    return results


def create_tensorflow_models(
    X_train, X_test, y_train, y_test, task_type="classification"
):
    """Create and train TensorFlow/Keras models"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - skipping TensorFlow models")
        return {}

    print(f"Training TensorFlow models for {task_type}...")

    results = {}

    if task_type == "classification":
        n_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, n_classes)
        y_test_cat = to_categorical(y_test, n_classes)

        # Model architectures
        models_config = {
            "Simple Dense": {
                "layers": [
                    layers.Dense(
                        64, activation="relu", input_shape=(X_train.shape[1],)
                    ),
                    layers.Dropout(0.3),
                    layers.Dense(n_classes, activation="softmax"),
                ],
                "optimizer": "adam",
                "loss": "categorical_crossentropy",
            },
            "Deep Network": {
                "layers": [
                    layers.Dense(
                        128, activation="relu", input_shape=(X_train.shape[1],)
                    ),
                    layers.Dropout(0.4),
                    layers.Dense(64, activation="relu"),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(n_classes, activation="softmax"),
                ],
                "optimizer": "adam",
                "loss": "categorical_crossentropy",
            },
            "Regularized Network": {
                "layers": [
                    layers.Dense(
                        100,
                        activation="relu",
                        input_shape=(X_train.shape[1],),
                        kernel_regularizer=regularizers.l2(0.001),
                    ),
                    layers.Dropout(0.5),
                    layers.Dense(
                        50, activation="relu", kernel_regularizer=regularizers.l2(0.001)
                    ),
                    layers.Dropout(0.3),
                    layers.Dense(n_classes, activation="softmax"),
                ],
                "optimizer": optimizers.Adam(learning_rate=0.001),
                "loss": "categorical_crossentropy",
            },
        }

        for name, config in models_config.items():
            model = keras.Sequential(config["layers"])
            model.compile(
                optimizer=config["optimizer"], loss=config["loss"], metrics=["accuracy"]
            )

            # Early stopping
            early_stop = callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )

            start_time = datetime.now()

            history = model.fit(
                X_train,
                y_train_cat,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test_cat),
                callbacks=[early_stop],
                verbose=0,
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # Evaluate
            train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

            results[name] = {
                "model": model,
                "history": history,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "training_time": training_time,
                "epochs_trained": len(history.history["loss"]),
            }

            print(
                f"  {name}: Acc={test_acc:.3f}, Loss={test_loss:.3f}, "
                f"Epochs={len(history.history['loss'])}, Time={training_time:.2f}s"
            )

    else:  # regression
        models_config = {
            "Simple Regressor": {
                "layers": [
                    layers.Dense(
                        64, activation="relu", input_shape=(X_train.shape[1],)
                    ),
                    layers.Dropout(0.3),
                    layers.Dense(1),
                ],
                "optimizer": "adam",
                "loss": "mse",
            },
            "Deep Regressor": {
                "layers": [
                    layers.Dense(
                        128, activation="relu", input_shape=(X_train.shape[1],)
                    ),
                    layers.Dropout(0.4),
                    layers.Dense(64, activation="relu"),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1),
                ],
                "optimizer": "adam",
                "loss": "mse",
            },
        }

        for name, config in models_config.items():
            model = keras.Sequential(config["layers"])
            model.compile(
                optimizer=config["optimizer"], loss=config["loss"], metrics=["mae"]
            )

            early_stop = callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )

            start_time = datetime.now()

            history = model.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0,
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # Evaluate
            test_pred = model.predict(X_test, verbose=0)
            test_mse = mean_squared_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)

            results[name] = {
                "model": model,
                "history": history,
                "test_mse": test_mse,
                "test_r2": test_r2,
                "training_time": training_time,
                "epochs_trained": len(history.history["loss"]),
            }

            print(
                f"  {name}: R²={test_r2:.3f}, MSE={test_mse:.3f}, "
                f"Epochs={len(history.history['loss'])}, Time={training_time:.2f}s"
            )

    return results


def analyze_neural_network_performance(
    sklearn_results, tf_results, task_type="classification"
):
    """Analyze and visualize neural network performance"""
    print("\\nAnalyzing neural network performance...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Neural Network Performance Analysis", fontsize=16, fontweight="bold")

    # Combine results for analysis
    all_models = []

    if task_type == "classification":
        # Scikit-learn results
        for name, result in sklearn_results.items():
            all_models.append(
                {
                    "Model": f"SKL: {name}",
                    "Framework": "Scikit-Learn",
                    "Test_Accuracy": result["test_accuracy"],
                    "Test_F1": result["test_f1"],
                    "Training_Time": result["training_time"],
                }
            )

        # TensorFlow results
        for name, result in tf_results.items():
            all_models.append(
                {
                    "Model": f"TF: {name}",
                    "Framework": "TensorFlow",
                    "Test_Accuracy": result["test_accuracy"],
                    "Test_F1": result.get(
                        "test_f1", 0
                    ),  # TF doesn't compute F1 by default
                    "Training_Time": result["training_time"],
                }
            )

        results_df = pd.DataFrame(all_models)

        # 1. Accuracy comparison
        ax = axes[0, 0]
        sns.barplot(
            data=results_df, x="Test_Accuracy", y="Model", hue="Framework", ax=ax
        )
        ax.set_title("Test Accuracy Comparison")
        ax.set_xlim(0, 1)

        # 2. Training time comparison
        ax = axes[0, 1]
        sns.barplot(
            data=results_df, x="Training_Time", y="Model", hue="Framework", ax=ax
        )
        ax.set_title("Training Time Comparison")
        ax.set_xlabel("Training Time (seconds)")

    else:  # regression
        # Scikit-learn results
        for name, result in sklearn_results.items():
            all_models.append(
                {
                    "Model": f"SKL: {name}",
                    "Framework": "Scikit-Learn",
                    "Test_R2": result["test_r2"],
                    "Test_MSE": result["test_mse"],
                    "Training_Time": result["training_time"],
                }
            )

        # TensorFlow results
        for name, result in tf_results.items():
            all_models.append(
                {
                    "Model": f"TF: {name}",
                    "Framework": "TensorFlow",
                    "Test_R2": result["test_r2"],
                    "Test_MSE": result["test_mse"],
                    "Training_Time": result["training_time"],
                }
            )

        results_df = pd.DataFrame(all_models)

        # 1. R² comparison
        ax = axes[0, 0]
        sns.barplot(data=results_df, x="Test_R2", y="Model", hue="Framework", ax=ax)
        ax.set_title("Test R² Comparison")

        # 2. MSE comparison
        ax = axes[0, 1]
        sns.barplot(data=results_df, x="Test_MSE", y="Model", hue="Framework", ax=ax)
        ax.set_title("Test MSE Comparison")

    # 3. Framework comparison
    ax = axes[1, 0]
    framework_perf = (
        results_df.groupby("Framework")
        .agg(
            {
                "Test_Accuracy" if task_type == "classification" else "Test_R2": "mean",
                "Training_Time": "mean",
            }
        )
        .reset_index()
    )

    metric_name = "Test_Accuracy" if task_type == "classification" else "Test_R2"

    x_pos = np.arange(len(framework_perf))
    ax.bar(x_pos, framework_perf[metric_name], alpha=0.7)
    ax.set_xlabel("Framework")
    ax.set_ylabel(metric_name.replace("_", " "))
    ax.set_title(f'Average {metric_name.replace("_", " ")} by Framework')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(framework_perf["Framework"])

    # 4. Performance vs Time scatter
    ax = axes[1, 1]
    for framework in results_df["Framework"].unique():
        framework_data = results_df[results_df["Framework"] == framework]
        ax.scatter(
            framework_data["Training_Time"],
            framework_data[metric_name],
            label=framework,
            alpha=0.7,
            s=100,
        )

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel(metric_name.replace("_", " "))
    ax.set_title("Performance vs Training Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results_df


def main():
    """Main function to run deep learning challenge"""
    print("=" * 60)
    print("LEVEL 5 CHALLENGE 2: DEEP LEARNING & NEURAL NETWORKS")
    print("=" * 60)

    # Set random seeds
    set_random_seeds()

    # Check TensorFlow availability
    if TENSORFLOW_AVAILABLE:
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    else:
        print("TensorFlow not available - using scikit-learn only")

    # Classification Task
    print("\\n" + "=" * 50)
    print("CLASSIFICATION TASK: PATTERN RECOGNITION")
    print("=" * 50)

    X_class, y_class = create_pattern_dataset(n_samples=3000, pattern_size=20)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    # Scale features
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    print(
        f"Classification data: {X_train_c_scaled.shape[0]} train, {X_test_c_scaled.shape[0]} test"
    )

    # Train models
    sklearn_class_results = create_sklearn_neural_networks(
        X_train_c_scaled, X_test_c_scaled, y_train_c, y_test_c, "classification"
    )

    tf_class_results = create_tensorflow_models(
        X_train_c_scaled, X_test_c_scaled, y_train_c, y_test_c, "classification"
    )

    # Regression Task
    print("\\n" + "=" * 50)
    print("REGRESSION TASK: TIME SERIES PREDICTION")
    print("=" * 50)

    X_reg, y_reg = create_time_series_dataset(n_samples=2000, sequence_length=50)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Scale features
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)

    print(
        f"Regression data: {X_train_r_scaled.shape[0]} train, {X_test_r_scaled.shape[0]} test"
    )

    # Train models
    sklearn_reg_results = create_sklearn_neural_networks(
        X_train_r_scaled, X_test_r_scaled, y_train_r, y_test_r, "regression"
    )

    tf_reg_results = create_tensorflow_models(
        X_train_r_scaled, X_test_r_scaled, y_train_r, y_test_r, "regression"
    )

    # Analysis
    print("\\n" + "=" * 50)
    print("PERFORMANCE ANALYSIS")
    print("=" * 50)

    class_results_df = analyze_neural_network_performance(
        sklearn_class_results, tf_class_results, "classification"
    )

    reg_results_df = analyze_neural_network_performance(
        sklearn_reg_results, tf_reg_results, "regression"
    )

    # Summary
    print("\\n" + "=" * 60)
    print("CHALLENGE 2 COMPLETION SUMMARY")
    print("=" * 60)

    total_models = (
        len(sklearn_class_results)
        + len(tf_class_results)
        + len(sklearn_reg_results)
        + len(tf_reg_results)
    )

    neural_techniques = [
        "Multi-layer Perceptrons (MLPs)",
        "Deep feedforward networks",
        "Dropout regularization",
        "L2 weight regularization",
        "Adam optimization",
        "Early stopping callbacks",
        "Batch normalization concepts",
        "Classification with softmax",
        "Regression with MSE loss",
        "Pattern recognition tasks",
        "Time series prediction",
    ]

    if TENSORFLOW_AVAILABLE:
        neural_techniques.extend(
            [
                "TensorFlow/Keras implementation",
                "Sequential model building",
                "Custom layer configurations",
            ]
        )

    print(f"Total models trained: {total_models}")
    print(
        f"Frameworks used: {'Scikit-Learn, TensorFlow' if TENSORFLOW_AVAILABLE else 'Scikit-Learn'}"
    )
    print("\\nNeural network techniques mastered:")
    for i, technique in enumerate(neural_techniques, 1):
        print(f"  {i}. {technique}")

    print("\\nDataset Statistics:")
    print(
        f"  - Pattern recognition: {len(X_class):,} samples, {X_class.shape[1]} features"
    )
    print(f"  - Time series: {len(X_reg):,} sequences, {X_reg.shape[1]} timesteps")
    print(f"  - Classification classes: {len(np.unique(y_class))}")
    print(f"  - Regression target range: [{y_reg.min():.2f}, {y_reg.max():.2f}]")

    return {
        "sklearn_classification": sklearn_class_results,
        "tensorflow_classification": tf_class_results,
        "sklearn_regression": sklearn_reg_results,
        "tensorflow_regression": tf_reg_results,
        "class_results_df": class_results_df,
        "reg_results_df": reg_results_df,
    }


if __name__ == "__main__":
    results = main()

    print("\\n" + "=" * 60)
    print("CHALLENGE 2 STATUS: COMPLETE")
    print("=" * 60)
    print("Deep learning and neural network mastery achieved!")
    print("Ready for Challenge 3: Advanced Feature Engineering.")
