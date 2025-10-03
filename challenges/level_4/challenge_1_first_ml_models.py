#!/usr/bin/env python3
"""
Level 4: Machine Learning Novice
Challenge 1: Your First ML Models

Build and evaluate your first prediction models using classification and regression.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")


def create_classification_dataset(n_samples=1000, random_state=42):
    """Create a synthetic classification dataset"""
    rng = np.random.default_rng(random_state)

    # Generate features
    X = rng.standard_normal((n_samples, 4))

    # Create target based on complex relationships
    target_prob = (
        0.3 * X[:, 0]
        + 0.2 * X[:, 1]
        - 0.1 * X[:, 2]
        + 0.4 * X[:, 3]
        + 0.1 * X[:, 0] * X[:, 1]
        + rng.standard_normal(n_samples) * 0.2
    )

    y = (target_prob > np.percentile(target_prob, 50)).astype(int)

    # Create feature names
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]

    return X, y, feature_names


def create_regression_dataset(n_samples=1000, random_state=42):
    """Create a synthetic regression dataset"""
    rng = np.random.default_rng(random_state)

    # Generate features
    X = rng.standard_normal((n_samples, 3))

    # Create target with non-linear relationships
    y = (
        2.5 * X[:, 0]
        + 1.8 * X[:, 1]
        - 0.7 * X[:, 2]
        + 0.3 * X[:, 0] ** 2
        + 0.2 * X[:, 1] * X[:, 2]
        + rng.standard_normal(n_samples) * 0.5
    )

    feature_names = ["temperature", "pressure", "humidity"]

    return X, y, feature_names


def iris_classification_challenge():
    """Complete classification challenge using Iris dataset"""
    print("=== CLASSIFICATION CHALLENGE ===")

    # Load and explore the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target
    iris_df["species"] = iris_df["target"].map(
        {0: "setosa", 1: "versicolor", 2: "virginica"}
    )

    print("Iris dataset overview:")
    print(iris_df.head())
    print(f"Shape: {iris_df.shape}")
    print(f"Classes: {iris_df['species'].unique()}")

    # Exploratory Data Analysis
    print("\nClass distribution:")
    print(iris_df["species"].value_counts())

    # Feature correlation
    print("\nFeature correlations:")
    correlation_matrix = iris_df[iris.feature_names].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Iris Features Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # Prepare data for classification
    X = iris_df[iris.feature_names]
    y = iris_df["target"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train multiple classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42, probability=True),
    }

    classification_results = {}

    print("\n🔬 Training Classification Models:")
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")

        # Use scaled data for SVM and Logistic Regression, original for Random Forest
        if name in ["SVM", "Logistic Regression"]:
            classifier.fit(X_train_scaled, y_train)
            y_pred = classifier.predict(X_test_scaled)
            y_pred_proba = classifier.predict_proba(X_test_scaled)
        else:
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            y_pred_proba = classifier.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        classification_results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Detailed classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Compare models
    print("\n📊 Model Comparison:")
    results_df = pd.DataFrame(classification_results).T
    print(results_df[["accuracy", "precision", "recall", "f1"]].round(4))

    # Best model
    best_model = results_df["accuracy"].idxmax()
    print(f"\n🏆 Best Classification Model: {best_model}")
    print(f"Best Accuracy: {results_df.loc[best_model, 'accuracy']:.4f}")

    # Confusion Matrix for best model
    best_predictions = classification_results[best_model]["predictions"]
    cm = confusion_matrix(y_test, best_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )
    plt.title(f"Confusion Matrix - {best_model}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    return classification_results


def housing_regression_challenge():
    """Complete regression challenge using California housing dataset"""
    print("\n=== REGRESSION CHALLENGE ===")

    # Load California housing dataset
    housing = fetch_california_housing()
    housing_df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    housing_df["target"] = housing.target

    # Sample for faster processing
    housing_sample = housing_df.sample(n=1000, random_state=42)

    print("California Housing dataset overview:")
    print(housing_sample.head())
    print(f"Shape: {housing_sample.shape}")
    print(
        f"Target range: ${housing_sample['target'].min():.1f}k - ${housing_sample['target'].max():.1f}k"
    )

    # Basic statistics
    print("\nTarget statistics:")
    print(housing_sample["target"].describe())

    # Feature correlation with target
    print("\nFeature correlations with target:")
    correlations = (
        housing_sample[housing.feature_names]
        .corrwith(housing_sample["target"])
        .sort_values(ascending=False)
    )
    print(correlations)

    # Prepare data for regression
    X = housing_sample[housing.feature_names]
    y = housing_sample["target"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train multiple regressors
    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel="rbf"),
    }

    regression_results = {}

    print("\n🔬 Training Regression Models:")
    for name, regressor in regressors.items():
        print(f"\nTraining {name}...")

        # Use scaled data for SVR and Linear Regression, original for Random Forest
        if name in ["SVR", "Linear Regression"]:
            regressor.fit(X_train_scaled, y_train)
            y_pred = regressor.predict(X_test_scaled)
        else:
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        regression_results[name] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "predictions": y_pred,
        }

        print(f"RMSE: ${rmse:.2f}k")
        print(f"MAE: ${mae:.2f}k")
        print(f"R² Score: {r2:.4f}")

    # Compare models
    print("\n📊 Model Comparison:")
    results_df = pd.DataFrame(regression_results).T
    print(results_df[["rmse", "mae", "r2"]].round(4))

    # Best model (highest R²)
    best_model = results_df["r2"].idxmax()
    print(f"\n🏆 Best Regression Model: {best_model}")
    print(f"Best R² Score: {results_df.loc[best_model, 'r2']:.4f}")

    # Prediction vs Actual plot for best model
    best_predictions = regression_results[best_model]["predictions"]

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, best_predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Housing Price ($100k)")
    plt.ylabel("Predicted Housing Price ($100k)")
    plt.title(f"Actual vs Predicted - {best_model}")
    plt.grid(True, alpha=0.3)

    # Add R² score to plot
    r2_score_best = results_df.loc[best_model, "r2"]
    plt.text(
        0.05,
        0.95,
        f"R² = {r2_score_best:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    return regression_results


def cross_validation_analysis():
    """Perform cross-validation analysis"""
    print("\n=== CROSS-VALIDATION ANALYSIS ===")

    # Use Iris for classification CV
    iris = load_iris()
    X, y = iris.data, iris.target

    # Models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42),
    }

    print("Cross-validation scores (5-fold):")
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        cv_results[name] = scores

        print(f"\n{name}:")
        print(f"CV Scores: {scores}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # Visualize CV results
    plt.figure(figsize=(10, 6))
    plt.boxplot(cv_results.values(), labels=cv_results.keys())
    plt.title("Cross-Validation Score Distribution")
    plt.ylabel("Accuracy Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return cv_results


def main():
    """Main function to run all ML challenges"""
    print("=" * 60)
    print("LEVEL 4 CHALLENGE 1: YOUR FIRST ML MODELS")
    print("=" * 60)

    print("🚀 Welcome to Machine Learning!")
    print("You'll learn to build and evaluate classification and regression models.")

    # Classification Challenge
    classification_results = iris_classification_challenge()

    # Regression Challenge
    regression_results = housing_regression_challenge()

    # Cross-validation Analysis
    cv_results = cross_validation_analysis()

    # Summary
    print("\n" + "=" * 60)
    print("CHALLENGE 1 COMPLETION SUMMARY")
    print("=" * 60)

    print("Machine Learning concepts mastered:")
    concepts = [
        "📊 Classification vs Regression problems",
        "🔄 Train/Test data splitting and stratification",
        "📏 Feature scaling and preprocessing",
        "🎯 Multiple algorithm comparison (Logistic, Random Forest, SVM)",
        "📈 Model evaluation metrics (Accuracy, Precision, Recall, F1, RMSE, R²)",
        "✅ Cross-validation for robust model assessment",
        "🎨 Confusion matrices and prediction visualizations",
        "🏆 Model selection based on performance metrics",
    ]

    for concept in concepts:
        print(f"  {concept}")

    print(f"\nDatasets analyzed:")
    print(f"  • Iris Classification: 150 samples, 3 species")
    print(f"  • California Housing: 1,000 samples, price prediction")

    print(f"\nModels trained and evaluated:")
    print(f"  • Classification models: {len(classification_results)}")
    print(f"  • Regression models: {len(regression_results)}")
    print(f"  • Cross-validation: 5-fold validation completed")

    print("\n🎉 Congratulations! You've completed your first ML challenge!")
    print("You're now ready for Challenge 2: Feature Engineering")

    return {
        "classification_results": classification_results,
        "regression_results": regression_results,
        "cv_results": cv_results,
    }


if __name__ == "__main__":
    results = main()

    print("\n" + "=" * 60)
    print("CHALLENGE 1 STATUS: COMPLETE")
    print("=" * 60)
    print("Machine learning fundamentals mastered!")
    print("Ready for Challenge 2: Feature Engineering.")
