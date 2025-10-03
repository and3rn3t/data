"""
Level 5 - Challenge 4: Advanced Neural Architectures
===================================================

Explore cutting-edge neural network architectures and techniques.
This challenge covers advanced concepts, ensemble methods, and modern architectures.

Learning Objectives:
- Understand ensemble methods and model combination
- Explore attention mechanisms and transformers (conceptual)
- Learn about regularization and optimization techniques
- Master model interpretation and explainability
- Understand modern architecture patterns

Required Libraries: numpy, matplotlib, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")


class SimpleEnsemble(BaseEstimator, ClassifierMixin):
    """
    Simple ensemble classifier combining multiple base models.
    """

    def __init__(self, base_models: List[Any], voting: str = "hard"):
        self.base_models = base_models
        self.voting = voting
        self.fitted_models = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all base models."""
        self.fitted_models = []
        for model in self.base_models:
            fitted_model = model.fit(X, y)
            self.fitted_models.append(fitted_model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []

        for model in self.fitted_models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        if self.voting == "hard":
            # Majority voting
            ensemble_pred = []
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                ensemble_pred.append(majority_class)
            return np.array(ensemble_pred)
        else:
            # Soft voting (average probabilities)
            if hasattr(self.fitted_models[0], "predict_proba"):
                probas = []
                for model in self.fitted_models:
                    proba = model.predict_proba(X)
                    probas.append(proba)
                avg_proba = np.mean(probas, axis=0)
                return np.argmax(avg_proba, axis=1)
            else:
                # Fallback to hard voting
                return self.predict_with_hard_voting(X)


def create_advanced_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Create datasets for testing advanced architectures.

    Returns:
        Dictionary containing various challenging datasets
    """
    print("🔬 Creating Advanced Neural Network Datasets...")

    datasets = {}
    rng = np.random.default_rng(42)

    # 1. Multi-modal Dataset (different feature types)
    print("Creating multi-modal dataset...")
    n_samples = 1000

    # Numerical features
    numerical_features = rng.normal(0, 1, (n_samples, 5))

    # Categorical-like features (embedded)
    categorical_features = rng.choice([0, 1], size=(n_samples, 3))  # Binary categories
    categorical_embedded = np.zeros((n_samples, 6))  # One-hot like
    for i, cat_vals in enumerate(categorical_features):
        for j, val in enumerate(cat_vals):
            categorical_embedded[i, j * 2 + val] = 1

    # Sequential features (time series like)
    seq_length = 10
    sequential_features = []
    for _ in range(n_samples):
        seq = np.cumsum(rng.normal(0, 0.1, seq_length))
        sequential_features.append(seq)
    sequential_features = np.array(sequential_features)

    # Combine all features
    X_multimodal = np.hstack(
        [numerical_features, categorical_embedded, sequential_features]
    )

    # Complex target based on interactions
    y_multimodal = []
    for i in range(n_samples):
        numerical_sum = np.sum(numerical_features[i])
        categorical_sum = np.sum(categorical_features[i])
        sequential_trend = sequential_features[i, -1] - sequential_features[i, 0]

        score = 0.3 * numerical_sum + 0.4 * categorical_sum + 0.3 * sequential_trend

        if score > 0.5:
            y_multimodal.append(2)  # High
        elif score > -0.5:
            y_multimodal.append(1)  # Medium
        else:
            y_multimodal.append(0)  # Low

    datasets["multimodal"] = {
        "X": X_multimodal,
        "y": np.array(y_multimodal),
        "classes": ["Low", "Medium", "High"],
        "feature_types": {
            "numerical": list(range(5)),
            "categorical": list(range(5, 11)),
            "sequential": list(range(11, 21)),
        },
        "description": "Multi-modal data with numerical, categorical, and sequential features",
    }

    # 2. Hierarchical Dataset (nested classes)
    print("Creating hierarchical classification dataset...")
    n_hierarchical = 800

    X_hierarchical = rng.normal(0, 1, (n_hierarchical, 8))

    # Create hierarchical structure: Animal -> Mammal/Bird -> Specific species
    y_hierarchical_coarse = []  # Mammal=0, Bird=1
    y_hierarchical_fine = []  # Cat=0, Dog=1, Eagle=2, Robin=3

    for i in range(n_hierarchical):
        features = X_hierarchical[i]

        # Coarse classification (Mammal vs Bird)
        mammal_score = features[0] + features[1] + features[2]
        bird_score = features[5] + features[6] + features[7]

        if mammal_score > bird_score:
            coarse_class = 0  # Mammal
            # Fine classification within mammals
            if features[3] > 0:
                fine_class = 0  # Cat
            else:
                fine_class = 1  # Dog
        else:
            coarse_class = 1  # Bird
            # Fine classification within birds
            if features[4] > 0:
                fine_class = 2  # Eagle
            else:
                fine_class = 3  # Robin

        y_hierarchical_coarse.append(coarse_class)
        y_hierarchical_fine.append(fine_class)

    datasets["hierarchical"] = {
        "X": X_hierarchical,
        "y_coarse": np.array(y_hierarchical_coarse),
        "y_fine": np.array(y_hierarchical_fine),
        "coarse_classes": ["Mammal", "Bird"],
        "fine_classes": ["Cat", "Dog", "Eagle", "Robin"],
        "description": "Hierarchical classification with coarse and fine-grained labels",
    }

    # 3. Attention-like Dataset (importance weighting needed)
    print("Creating attention-mechanism dataset...")
    n_attention = 600
    feature_dim = 12
    important_features = [2, 5, 8]  # Only these features matter

    X_attention = rng.normal(0, 1, (n_attention, feature_dim))
    y_attention = []

    for i in range(n_attention):
        # Only important features contribute to classification
        important_values = X_attention[i, important_features]

        # Dynamic importance (attention-like)
        attention_weights = np.abs(important_values) / (
            np.sum(np.abs(important_values)) + 1e-8
        )
        weighted_sum = np.sum(attention_weights * important_values)

        if weighted_sum > 0.3:
            y_attention.append(1)
        else:
            y_attention.append(0)

    datasets["attention"] = {
        "X": X_attention,
        "y": np.array(y_attention),
        "important_features": important_features,
        "classes": ["Low Attention", "High Attention"],
        "description": "Dataset requiring attention to specific features",
    }

    print(f"Created {len(datasets)} advanced datasets")
    return datasets


def demonstrate_ensemble_methods(datasets: Dict[str, Dict[str, Any]]) -> None:
    """
    Demonstrate ensemble learning techniques.
    """
    print("\n🎭 Ensemble Methods Demonstration")
    print("=" * 50)

    # Use multimodal dataset
    data = datasets["multimodal"]
    X, y = data["X"], data["y"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]} dimensions")
    print(f"Classes: {data['classes']}")

    # Create different base models
    base_models = {
        "Neural Network 1": MLPClassifier(
            hidden_layer_sizes=(50,),
            activation="relu",
            learning_rate_init=0.01,
            max_iter=500,
            random_state=1,
        ),
        "Neural Network 2": MLPClassifier(
            hidden_layer_sizes=(30, 20),
            activation="tanh",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=2,
        ),
        "Neural Network 3": MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="logistic",
            learning_rate_init=0.005,
            max_iter=500,
            random_state=3,
        ),
    }

    # Train individual models
    individual_scores = {}
    fitted_models = {}

    print("\n📊 Individual Model Performance:")
    for name, model in base_models.items():
        fitted_model = model.fit(X_train_scaled, y_train)
        y_pred = fitted_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        individual_scores[name] = accuracy
        fitted_models[name] = fitted_model

        print(f"• {name}: {accuracy:.3f}")

    # Create ensemble models
    print("\n🎯 Ensemble Performance:")

    # 1. Voting Classifier (hard voting)
    voting_hard = VotingClassifier(
        estimators=[(name, model) for name, model in base_models.items()], voting="hard"
    )
    voting_hard.fit(X_train_scaled, y_train)
    y_pred_hard = voting_hard.predict(X_test_scaled)
    accuracy_hard = accuracy_score(y_test, y_pred_hard)

    print(f"• Hard Voting: {accuracy_hard:.3f}")

    # 2. Voting Classifier (soft voting)
    voting_soft = VotingClassifier(
        estimators=[(name, model) for name, model in base_models.items()], voting="soft"
    )
    voting_soft.fit(X_train_scaled, y_train)
    y_pred_soft = voting_soft.predict(X_test_scaled)
    accuracy_soft = accuracy_score(y_test, y_pred_soft)

    print(f"• Soft Voting: {accuracy_soft:.3f}")

    # 3. Bagging
    bagging = BaggingClassifier(
        estimator=MLPClassifier(hidden_layer_sizes=(50,), max_iter=300),
        n_estimators=5,
        random_state=42,
    )
    bagging.fit(X_train_scaled, y_train)
    y_pred_bagging = bagging.predict(X_test_scaled)
    accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

    print(f"• Bagging: {accuracy_bagging:.3f}")

    # Visualize results
    methods = ["NN1", "NN2", "NN3", "Hard Vote", "Soft Vote", "Bagging"]
    accuracies = [
        individual_scores["Neural Network 1"],
        individual_scores["Neural Network 2"],
        individual_scores["Neural Network 3"],
        accuracy_hard,
        accuracy_soft,
        accuracy_bagging,
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies)

    # Color individual models differently from ensembles
    for i, bar in enumerate(bars):
        if i < 3:
            bar.set_color("lightblue")  # Individual models
        else:
            bar.set_color("lightgreen")  # Ensemble methods

    plt.title("Individual vs Ensemble Model Performance")
    plt.ylabel("Accuracy")
    plt.ylim(0, max(accuracies) * 1.1)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    print(f"\n🎯 Ensemble Analysis:")
    best_individual = max(individual_scores.values())
    best_ensemble = max(accuracy_hard, accuracy_soft, accuracy_bagging)

    print(f"• Best individual model: {best_individual:.3f}")
    print(f"• Best ensemble method: {best_ensemble:.3f}")
    print(f"• Ensemble improvement: {best_ensemble - best_individual:.3f}")

    if best_ensemble > best_individual:
        print("✅ Ensemble outperformed individual models!")
    else:
        print("⚠️ Ensemble didn't improve over best individual model")


def demonstrate_attention_mechanism(datasets: Dict[str, Dict[str, Any]]) -> None:
    """
    Demonstrate attention-like mechanisms conceptually.
    """
    print("\n🔍 Attention Mechanism Simulation")
    print("=" * 50)

    # Use attention dataset
    data = datasets["attention"]
    X, y = data["X"], data["y"]
    important_features = data["important_features"]

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Important features: {important_features}")
    print(f"Task: Only important features should contribute to prediction")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Method 1: Standard neural network (no attention)
    print(f"\n📊 Comparing Standard vs Attention-like Models:")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Standard model
    standard_model = MLPClassifier(
        hidden_layer_sizes=(50, 30),
        activation="relu",
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
    )
    standard_model.fit(X_train_scaled, y_train)
    y_pred_standard = standard_model.predict(X_test_scaled)
    accuracy_standard = accuracy_score(y_test, y_pred_standard)

    print(f"• Standard NN: {accuracy_standard:.3f}")

    # Method 2: Feature selection (simulate attention)
    X_train_attention = X_train[:, important_features]
    X_test_attention = X_test[:, important_features]

    scaler_attention = StandardScaler()
    X_train_attention_scaled = scaler_attention.fit_transform(X_train_attention)
    X_test_attention_scaled = scaler_attention.transform(X_test_attention)

    attention_model = MLPClassifier(
        hidden_layer_sizes=(30, 20),
        activation="relu",
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
    )
    attention_model.fit(X_train_attention_scaled, y_train)
    y_pred_attention = attention_model.predict(X_test_attention_scaled)
    accuracy_attention = accuracy_score(y_test, y_pred_attention)

    print(f"• Attention-like (feature selection): {accuracy_attention:.3f}")

    # Method 3: Weighted features (simulate learned attention)
    def compute_attention_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute simple attention weights based on feature importance."""
        weights = []

        for feat_idx in range(X.shape[1]):
            # Compute correlation between feature and target
            correlation = np.abs(np.corrcoef(X[:, feat_idx], y)[0, 1])
            weights.append(correlation)

        weights = np.array(weights)
        # Normalize to sum to 1 (attention weights)
        weights = weights / (np.sum(weights) + 1e-8)
        return weights

    attention_weights = compute_attention_weights(X_train_scaled, y_train)

    # Apply attention weights to features
    X_train_weighted = X_train_scaled * attention_weights
    X_test_weighted = X_test_scaled * attention_weights

    weighted_model = MLPClassifier(
        hidden_layer_sizes=(50, 30),
        activation="relu",
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
    )
    weighted_model.fit(X_train_weighted, y_train)
    y_pred_weighted = weighted_model.predict(X_test_weighted)
    accuracy_weighted = accuracy_score(y_test, y_pred_weighted)

    print(f"• Weighted features: {accuracy_weighted:.3f}")

    # Visualize attention weights
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Attention weights
    ax1.bar(range(len(attention_weights)), attention_weights)
    ax1.set_title("Learned Attention Weights")
    ax1.set_xlabel("Feature Index")
    ax1.set_ylabel("Attention Weight")
    ax1.grid(True, alpha=0.3)

    # Highlight important features
    for idx in important_features:
        ax1.bar(idx, attention_weights[idx], color="red", alpha=0.7)

    # Performance comparison
    methods = ["Standard NN", "Attention-like", "Weighted Features"]
    accuracies = [accuracy_standard, accuracy_attention, accuracy_weighted]

    bars = ax2.bar(methods, accuracies)
    ax2.set_title("Model Performance Comparison")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, max(accuracies) * 1.1)
    ax2.grid(True, alpha=0.3)

    # Color best method
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_color("gold")

    # Add value labels
    for i, v in enumerate(accuracies):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    print(f"\n🔍 Attention Analysis:")
    print(f"• Ground truth important features: {important_features}")
    top_attention_features = np.argsort(attention_weights)[-3:][::-1]
    print(f"• Top 3 attention features: {top_attention_features}")

    overlap = len(set(important_features) & set(top_attention_features))
    print(f"• Attention-ground truth overlap: {overlap}/3 features")

    if overlap >= 2:
        print("✅ Attention mechanism successfully identified important features!")
    else:
        print("⚠️ Attention mechanism needs improvement")


def demonstrate_regularization_techniques() -> None:
    """
    Demonstrate various regularization techniques.
    """
    print("\n🛡️ Regularization Techniques")
    print("=" * 50)

    # Create overfitting-prone dataset (small dataset, high dimensional)
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 50

    X = rng.normal(0, 1, (n_samples, n_features))
    # Create complex target with noise
    true_coefficients = rng.normal(0, 1, n_features) * (rng.random(n_features) > 0.7)
    y_continuous = np.dot(X, true_coefficients) + rng.normal(0, 0.5, n_samples)
    y = (y_continuous > np.median(y_continuous)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
    print("High-dimensional, small dataset → prone to overfitting")

    # Different regularization approaches
    models = {
        "No Regularization": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            alpha=0,  # No L2 regularization
            learning_rate_init=0.01,
            max_iter=500,
            random_state=42,
        ),
        "L2 Regularization": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            alpha=0.01,  # L2 regularization
            learning_rate_init=0.01,
            max_iter=500,
            random_state=42,
        ),
        "Strong L2 Regularization": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            alpha=0.1,  # Strong L2 regularization
            learning_rate_init=0.01,
            max_iter=500,
            random_state=42,
        ),
        "Smaller Network": MLPClassifier(
            hidden_layer_sizes=(30, 15),  # Smaller capacity
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.01,
            max_iter=500,
            random_state=42,
        ),
        "Early Stopping": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            alpha=0.001,
            learning_rate_init=0.01,
            max_iter=100,  # Early stopping via fewer iterations
            random_state=42,
        ),
    }

    results = {}

    print(f"\n📊 Regularization Results:")

    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Evaluate
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)

        results[name] = {
            "train_acc": train_accuracy,
            "test_acc": test_accuracy,
            "overfitting": train_accuracy - test_accuracy,
        }

        print(f"• {name}:")
        print(f"  Train Accuracy: {train_accuracy:.3f}")
        print(f"  Test Accuracy: {test_accuracy:.3f}")
        print(f"  Overfitting Gap: {train_accuracy - test_accuracy:.3f}")
        print()

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = list(results.keys())
    train_accs = [results[method]["train_acc"] for method in methods]
    test_accs = [results[method]["test_acc"] for method in methods]
    overfitting_gaps = [results[method]["overfitting"] for method in methods]

    # Accuracy comparison
    x_pos = np.arange(len(methods))
    width = 0.35

    ax1.bar(x_pos - width / 2, train_accs, width, label="Train Accuracy", alpha=0.7)
    ax1.bar(x_pos + width / 2, test_accs, width, label="Test Accuracy", alpha=0.7)
    ax1.set_title("Train vs Test Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Overfitting analysis
    bars = ax2.bar(methods, overfitting_gaps)
    ax2.set_title("Overfitting Analysis")
    ax2.set_ylabel("Overfitting Gap (Train - Test)")
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    # Color bars based on overfitting level
    for i, bar in enumerate(bars):
        gap = overfitting_gaps[i]
        if gap < 0.05:
            bar.set_color("green")  # Good generalization
        elif gap < 0.15:
            bar.set_color("yellow")  # Moderate overfitting
        else:
            bar.set_color("red")  # High overfitting

    plt.tight_layout()
    plt.show()

    print(f"🎯 Regularization Analysis:")
    best_method = min(results.keys(), key=lambda k: results[k]["overfitting"])
    best_gap = results[best_method]["overfitting"]

    print(f"• Best regularization method: {best_method}")
    print(f"• Lowest overfitting gap: {best_gap:.3f}")
    print(f"• Test accuracy: {results[best_method]['test_acc']:.3f}")

    print(f"\n💡 Regularization Techniques:")
    print(f"• L2 regularization: Penalizes large weights")
    print(f"• Network size: Smaller networks have less capacity to overfit")
    print(f"• Early stopping: Prevents training too long")
    print(f"• Dropout: Random neuron deactivation (not shown here)")


def model_interpretation_demo() -> None:
    """
    Demonstrate model interpretation techniques.
    """
    print("\n🔬 Model Interpretation and Explainability")
    print("=" * 50)

    # Create interpretable dataset
    rng = np.random.default_rng(42)
    n_samples = 500

    # Features with known importance
    feature_names = [
        "Age",
        "Income",
        "Education",
        "Experience",
        "Location",
        "Noise1",
        "Noise2",
        "Noise3",
    ]
    n_features = len(feature_names)

    X = rng.normal(0, 1, (n_samples, n_features))

    # Create interpretable target
    # Important features: Age, Income, Education (indices 0, 1, 2)
    # Less important: Experience (index 3)
    # Irrelevant: Location, Noise features

    true_importance = np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    y_score = np.dot(X, true_importance) + rng.normal(0, 0.2, n_samples)
    y = (y_score > np.median(y_score)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = MLPClassifier(
        hidden_layer_sizes=(50, 30),
        activation="relu",
        alpha=0.01,
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.3f}")
    print(f"Features: {feature_names}")
    print(f"True importance: {true_importance}")

    # Method 1: Feature permutation importance
    print(f"\n🔄 Permutation Feature Importance:")

    def permutation_importance(model, X, y, feature_names):
        """Compute permutation importance."""
        baseline_score = model.score(X, y)
        importances = []

        for i in range(X.shape[1]):
            # Permute feature i
            X_permuted = X.copy()
            X_permuted[:, i] = rng.permutation(X_permuted[:, i])

            # Calculate score drop
            permuted_score = model.score(X_permuted, y)
            importance = baseline_score - permuted_score
            importances.append(importance)

            print(f"• {feature_names[i]}: {importance:.4f}")

        return np.array(importances)

    permutation_imp = permutation_importance(
        model, X_test_scaled, y_test, feature_names
    )

    # Method 2: Simple gradient-based importance (for first layer)
    print(f"\n⚡ Weight-based Feature Importance:")

    # Get first layer weights
    first_layer_weights = model.coefs_[0]  # Shape: (n_features, n_hidden)

    # Compute feature importance as sum of absolute weights
    weight_importance = np.sum(np.abs(first_layer_weights), axis=1)
    weight_importance = weight_importance / np.sum(weight_importance)  # Normalize

    for i, (name, imp) in enumerate(zip(feature_names, weight_importance)):
        print(f"• {name}: {imp:.4f}")

    # Visualize all importance methods
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True importance
    axes[0].bar(feature_names, true_importance)
    axes[0].set_title("True Feature Importance")
    axes[0].set_ylabel("Importance")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Permutation importance
    axes[1].bar(feature_names, permutation_imp)
    axes[1].set_title("Permutation Importance")
    axes[1].set_ylabel("Accuracy Drop")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3)

    # Weight-based importance
    axes[2].bar(feature_names, weight_importance)
    axes[2].set_title("Weight-based Importance")
    axes[2].set_ylabel("Normalized Weight Sum")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Correlation analysis
    print(f"\n📊 Importance Method Comparison:")

    # Only compare on features that should have importance > 0
    important_features = true_importance > 0

    if np.any(important_features):
        perm_corr = np.corrcoef(
            true_importance[important_features], permutation_imp[important_features]
        )[0, 1]
        weight_corr = np.corrcoef(
            true_importance[important_features], weight_importance[important_features]
        )[0, 1]

        print(f"• Permutation vs True importance correlation: {perm_corr:.3f}")
        print(f"• Weight-based vs True importance correlation: {weight_corr:.3f}")

        # Find top features by each method
        true_top3 = np.argsort(true_importance)[-3:][::-1]
        perm_top3 = np.argsort(permutation_imp)[-3:][::-1]
        weight_top3 = np.argsort(weight_importance)[-3:][::-1]

        print(f"\nTop 3 features by each method:")
        print(f"• True: {[feature_names[i] for i in true_top3]}")
        print(f"• Permutation: {[feature_names[i] for i in perm_top3]}")
        print(f"• Weight-based: {[feature_names[i] for i in weight_top3]}")

        # Check overlap
        perm_overlap = len(set(true_top3) & set(perm_top3))
        weight_overlap = len(set(true_top3) & set(weight_top3))

        print(f"\nTop-3 overlap with true importance:")
        print(f"• Permutation method: {perm_overlap}/3")
        print(f"• Weight-based method: {weight_overlap}/3")


def run_advanced_challenges() -> None:
    """
    Run all advanced neural architecture challenges.
    """
    print("🚀 Starting Level 5 Challenge 4: Advanced Neural Architectures")
    print("=" * 60)

    try:
        # Challenge 1: Create advanced datasets
        print("\n" + "=" * 50)
        print("CHALLENGE 1: Advanced Dataset Creation")
        print("=" * 50)

        datasets = create_advanced_datasets()

        print(f"\n✅ Created {len(datasets)} advanced datasets:")
        for name, data in datasets.items():
            if "X" in data:
                print(f"• {name}: {data['X'].shape}")
            print(f"  Description: {data['description']}")

        # Challenge 2: Ensemble methods
        print("\n" + "=" * 50)
        print("CHALLENGE 2: Ensemble Learning")
        print("=" * 50)

        demonstrate_ensemble_methods(datasets)

        # Challenge 3: Attention mechanisms
        print("\n" + "=" * 50)
        print("CHALLENGE 3: Attention Mechanisms")
        print("=" * 50)

        demonstrate_attention_mechanism(datasets)

        # Challenge 4: Regularization
        print("\n" + "=" * 50)
        print("CHALLENGE 4: Regularization Techniques")
        print("=" * 50)

        demonstrate_regularization_techniques()

        # Challenge 5: Model interpretation
        print("\n" + "=" * 50)
        print("CHALLENGE 5: Model Interpretation")
        print("=" * 50)

        model_interpretation_demo()

        print("\n" + "🎉" * 20)
        print("LEVEL 5 CHALLENGE 4 COMPLETE!")
        print("🎉" * 20)

        print("\n📚 What You've Learned:")
        print("• Ensemble methods: voting, bagging, model combination")
        print("• Attention mechanisms: feature weighting and selection")
        print("• Regularization: L2, network size, early stopping")
        print("• Model interpretation: permutation importance, weight analysis")
        print("• Advanced datasets: multi-modal, hierarchical, attention-based")
        print("• Performance evaluation: overfitting analysis, generalization")

        print("\n🏆 LEVEL 5: DEEP LEARNING DYNAMO - COMPLETE!")
        print("=" * 60)
        print("🎓 Congratulations! You've mastered:")
        print("• Neural Network Fundamentals")
        print("• Convolutional Neural Networks (CNNs)")
        print("• Recurrent Neural Networks (RNNs)")
        print("• Advanced Neural Architectures")

        print("\n🚀 Next Steps for Continued Learning:")
        print("• Implement with TensorFlow/PyTorch")
        print("• Explore transformer architectures")
        print("• Try generative models (VAEs, GANs)")
        print("• Apply to real-world projects")
        print("• Explore reinforcement learning")
        print("• Study MLOps and model deployment")

        return datasets

    except Exception as e:
        print(f"❌ Error in advanced challenges: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete advanced architectures challenge
    datasets = run_advanced_challenges()

    if datasets:
        print("\n" + "=" * 60)
        print("ADVANCED ARCHITECTURES CHALLENGE SUMMARY")
        print("=" * 60)

        print("\nDatasets Created:")
        for name, data in datasets.items():
            if "X" in data:
                print(f"• {name}: {data['X'].shape}")

        print("\nAdvanced Concepts Mastered:")
        concepts = [
            "Ensemble learning and model combination strategies",
            "Attention mechanisms and feature importance",
            "Regularization techniques for preventing overfitting",
            "Model interpretation and explainability methods",
            "Multi-modal and hierarchical data handling",
            "Performance analysis and generalization assessment",
        ]

        for i, concept in enumerate(concepts, 1):
            print(f"{i}. {concept}")

        print(
            f"\n✨ Deep Learning Journey Complete! Ready for real-world applications! 🌟"
        )
