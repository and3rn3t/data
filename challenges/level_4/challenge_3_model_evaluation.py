#!/usr/bin/env python3
"""
Level 4: Machine Learning Novice
Challenge 3: Model Evaluation Mastery

Master comprehensive model evaluation techniques and metrics for robust ML assessment.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")


def create_classification_evaluation_dataset(n_samples=1000, random_state=42):
    """Create a classification dataset for evaluation demonstration"""
    print("📊 Creating Classification Dataset for Evaluation...")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=random_state,
    )

    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]

    return X, y, feature_names


def create_regression_evaluation_dataset(n_samples=1000, random_state=42):
    """Create a regression dataset for evaluation demonstration"""
    print("📊 Creating Regression Dataset for Evaluation...")

    X, y = make_regression(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=random_state,
    )

    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]

    return X, y, feature_names


def classification_metrics_deep_dive(X, y):
    """Comprehensive classification metrics analysis"""
    print("\\n=== CLASSIFICATION METRICS DEEP DIVE ===")

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train multiple models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42, probability=True),
    }

    evaluation_results = {}

    for name, model in models.items():
        print(f"\\n--- {name} Evaluation ---")

        # Fit model (use scaled data for LR and SVM)
        if name in ["Logistic Regression", "SVM"]:
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            y_pred_proba = model.predict_proba(x_test_scaled)[:, 1]
        else:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)[:, 1]

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        evaluation_results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        # Detailed classification report
        print("\\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))

    # Comparison visualization
    metrics_df = pd.DataFrame(evaluation_results).T

    plt.figure(figsize=(15, 10))

    # Metrics comparison
    plt.subplot(2, 3, 1)
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metrics_df[metrics_to_plot].plot(kind="bar", ax=plt.gca())
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)

    # Confusion matrices
    for i, (name, results) in enumerate(evaluation_results.items()):
        plt.subplot(2, 3, i + 2)
        cm = confusion_matrix(y_test, results["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

    # ROC curves
    plt.subplot(2, 3, 5)
    for name, results in evaluation_results.items():
        fpr, tpr, _ = roc_curve(y_test, results["y_pred_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={results['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Precision-Recall curves
    plt.subplot(2, 3, 6)
    for name, results in evaluation_results.items():
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_test, results["y_pred_proba"]
        )
        plt.plot(recall_vals, precision_vals, label=f"{name}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return evaluation_results, y_test


def regression_metrics_deep_dive(X, y):
    """Comprehensive regression metrics analysis"""
    print("\\n=== REGRESSION METRICS DEEP DIVE ===")

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train multiple models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel="rbf"),
    }

    evaluation_results = {}

    for name, model in models.items():
        print(f"\\n--- {name} Evaluation ---")

        # Fit model (use scaled data for LR and SVR)
        if name in ["Linear Regression", "SVR"]:
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        residuals = y_test - y_pred

        evaluation_results[name] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "y_pred": y_pred,
            "residuals": residuals,
        }

        print(f"MSE:   {mse:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"MAE:   {mae:.4f}")
        print(f"R²:    {r2:.4f}")
        print(f"MAPE:  {mape:.2f}%")

    # Visualization
    plt.figure(figsize=(15, 10))

    # Metrics comparison
    plt.subplot(2, 3, 1)
    metrics_df = pd.DataFrame(evaluation_results).T
    metrics_df[["rmse", "mae", "r2"]].plot(kind="bar", ax=plt.gca())
    plt.title("Regression Metrics Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()

    # Actual vs Predicted plots
    for i, (name, results) in enumerate(evaluation_results.items()):
        plt.subplot(2, 3, i + 2)
        plt.scatter(y_test, results["y_pred"], alpha=0.6)
        plt.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f'Actual vs Predicted: {name}\\nR² = {results["r2"]:.3f}')
        plt.grid(True, alpha=0.3)

    # Residual analysis
    plt.subplot(2, 3, 5)
    for name, results in evaluation_results.items():
        plt.hist(results["residuals"], bins=20, alpha=0.7, label=name)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residual Distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Residuals vs Predicted
    plt.subplot(2, 3, 6)
    best_model = max(
        evaluation_results.keys(), key=lambda k: evaluation_results[k]["r2"]
    )
    results = evaluation_results[best_model]
    plt.scatter(results["y_pred"], results["residuals"], alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot: {best_model}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return evaluation_results, y_test


def cross_validation_analysis(X, y, task_type="classification"):
    """Comprehensive cross-validation analysis"""
    print(f"\\n=== CROSS-VALIDATION ANALYSIS ({task_type.upper()}) ===")

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "SVM": SVC(random_state=42),
        }
        scoring = "accuracy"
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
            "SVR": SVR(kernel="rbf"),
        }
        scoring = "r2"

    cv_results = {}

    print(f"\\n{scoring.upper()} Scores (5-fold Cross-Validation):")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        cv_results[name] = scores

        print(f"\\n{name}:")
        print(f"  CV Scores: {scores}")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std:  {scores.std():.4f}")
        print(
            f"  95% CI: [{scores.mean() - 1.96*scores.std():.4f}, {scores.mean() + 1.96*scores.std():.4f}]"
        )

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.boxplot(cv_results.values(), labels=cv_results.keys())
    plt.title(f"Cross-Validation {scoring.upper()} Distribution")
    plt.ylabel(f"{scoring.upper()} Score")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    means = [scores.mean() for scores in cv_results.values()]
    stds = [scores.std() for scores in cv_results.values()]
    plt.errorbar(range(len(means)), means, yerr=stds, fmt="o", capsize=5)
    plt.xticks(range(len(cv_results)), cv_results.keys(), rotation=45)
    plt.ylabel(f"Mean {scoring.upper()} Score")
    plt.title(f"Mean {scoring.upper()} with Error Bars")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return cv_results


def learning_curve_analysis(X, y, model, model_name, task_type="classification"):
    """Analyze learning curves to detect overfitting/underfitting"""
    print(f"\\n=== LEARNING CURVE ANALYSIS: {model_name} ===")

    scoring = "accuracy" if task_type == "classification" else "r2"

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring,
        random_state=42,
    )

    # Calculate means and stds
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Plot learning curves
    plt.figure(figsize=(10, 6))

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
    plt.ylabel(f"{scoring.upper()} Score")
    plt.title(f"Learning Curve: {model_name}")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    # Analysis
    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.1:
        diagnosis = "OVERFITTING: Large gap between training and validation scores"
    elif val_mean[-1] < 0.7:
        diagnosis = "UNDERFITTING: Both scores are relatively low"
    else:
        diagnosis = "GOOD FIT: Training and validation scores are close and reasonable"

    plt.text(
        0.02,
        0.98,
        diagnosis,
        transform=plt.gca().transAxes,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        verticalalignment="top",
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()

    print(f"Final training score: {train_mean[-1]:.4f} (+/- {train_std[-1]:.4f})")
    print(f"Final validation score: {val_mean[-1]:.4f} (+/- {val_std[-1]:.4f})")
    print(f"Training-Validation gap: {final_gap:.4f}")
    print(f"Diagnosis: {diagnosis}")

    return {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "val_scores": val_scores,
        "diagnosis": diagnosis,
    }


def validation_curve_analysis(
    X, y, model, param_name, param_range, model_name, task_type="classification"
):
    """Analyze validation curves for hyperparameter tuning"""
    print(f"\\n=== VALIDATION CURVE ANALYSIS: {model_name} - {param_name} ===")

    scoring = "accuracy" if task_type == "classification" else "r2"

    train_scores, val_scores = validation_curve(
        model,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
    )

    # Calculate means and stds
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Plot validation curves
    plt.figure(figsize=(10, 6))

    plt.plot(param_range, train_mean, "o-", color="blue", label="Training Score")
    plt.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )

    plt.plot(param_range, val_mean, "o-", color="red", label="Validation Score")
    plt.fill_between(
        param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red"
    )

    plt.xlabel(param_name.replace("_", " ").title())
    plt.ylabel(f"{scoring.upper()} Score")
    plt.title(f"Validation Curve: {model_name} - {param_name}")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    # Find optimal parameter
    best_idx = np.argmax(val_mean)
    best_param = param_range[best_idx]
    best_score = val_mean[best_idx]

    plt.axvline(
        x=best_param,
        color="green",
        linestyle="--",
        alpha=0.8,
        label=f"Best {param_name}: {best_param}",
    )
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()

    print(f"Best {param_name}: {best_param}")
    print(f"Best validation score: {best_score:.4f}")

    return {
        "param_range": param_range,
        "train_scores": train_scores,
        "val_scores": val_scores,
        "best_param": best_param,
        "best_score": best_score,
    }


def model_interpretation_analysis(X, y, feature_names):
    """Analyze feature importance and model interpretability"""
    print("\\n=== MODEL INTERPRETATION ANALYSIS ===")

    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Feature importance analysis
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\\nTop 10 Most Important Features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Visualization
    plt.figure(figsize=(12, 8))

    # Feature importance plot
    plt.subplot(2, 2, 1)
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features["importance"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Feature Importances")
    plt.gca().invert_yaxis()

    # Feature importance distribution
    plt.subplot(2, 2, 2)
    plt.hist(feature_importance["importance"], bins=20, alpha=0.7, edgecolor="black")
    plt.xlabel("Feature Importance")
    plt.ylabel("Frequency")
    plt.title("Feature Importance Distribution")
    plt.grid(True, alpha=0.3)

    # Cumulative importance
    plt.subplot(2, 2, 3)
    cumulative_importance = np.cumsum(feature_importance["importance"])
    plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance)
    plt.xlabel("Number of Features")
    plt.ylabel("Cumulative Importance")
    plt.title("Cumulative Feature Importance")
    plt.grid(True, alpha=0.3)

    # Feature selection threshold analysis
    plt.subplot(2, 2, 4)
    thresholds = [0.8, 0.9, 0.95, 0.99]
    features_needed = []
    for threshold in thresholds:
        n_features = np.where(cumulative_importance >= threshold)[0][0] + 1
        features_needed.append(n_features)

    plt.bar(range(len(thresholds)), features_needed)
    plt.xticks(range(len(thresholds)), [f"{t*100:.0f}%" for t in thresholds])
    plt.xlabel("Importance Threshold")
    plt.ylabel("Number of Features Needed")
    plt.title("Features Needed for Different Thresholds")

    for i, v in enumerate(features_needed):
        plt.text(i, v + 0.5, str(v), ha="center")

    plt.tight_layout()
    plt.show()

    return feature_importance


def main():
    """Main function to run the model evaluation challenge"""
    print("=" * 60)
    print("LEVEL 4 CHALLENGE 3: MODEL EVALUATION MASTERY")
    print("=" * 60)

    print("📊 Welcome to Model Evaluation!")
    print("Learn comprehensive techniques for assessing model performance.")

    # 1. Create datasets
    x_class, y_class, class_features = create_classification_evaluation_dataset()
    x_reg, y_reg, reg_features = create_regression_evaluation_dataset()

    # 2. Classification evaluation
    class_results, y_test_class = classification_metrics_deep_dive(x_class, y_class)

    # 3. Regression evaluation
    reg_results, y_test_reg = regression_metrics_deep_dive(x_reg, y_reg)

    # 4. Cross-validation analysis
    cv_class_results = cross_validation_analysis(x_class, y_class, "classification")
    cv_reg_results = cross_validation_analysis(x_reg, y_reg, "regression")

    # 5. Learning curve analysis
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    learning_results = learning_curve_analysis(
        x_class, y_class, rf_classifier, "Random Forest Classifier", "classification"
    )

    # 6. Validation curve analysis
    param_range = [10, 20, 50, 100, 200]
    validation_results = validation_curve_analysis(
        x_class,
        y_class,
        RandomForestClassifier(random_state=42),
        "n_estimators",
        param_range,
        "Random Forest",
        "classification",
    )

    # 7. Model interpretation
    interpretation_results = model_interpretation_analysis(
        x_class, y_class, class_features
    )

    # Summary
    print("\\n" + "=" * 60)
    print("CHALLENGE 3 COMPLETION SUMMARY")
    print("=" * 60)

    print("Model Evaluation techniques mastered:")
    techniques = [
        "📊 Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)",
        "📈 Regression metrics (MSE, RMSE, MAE, R², MAPE)",
        "🎯 Confusion matrices and error analysis",
        "📉 ROC curves and Precision-Recall curves",
        "✅ Cross-validation for robust assessment",
        "📚 Learning curves (overfitting/underfitting detection)",
        "🔧 Validation curves (hyperparameter optimization)",
        "🔍 Feature importance and model interpretation",
        "📋 Residual analysis for regression models",
    ]

    for technique in techniques:
        print(f"  {technique}")

    print(f"\\nEvaluation analysis completed:")
    print(f"  • Classification models evaluated: {len(class_results)}")
    print(f"  • Regression models evaluated: {len(reg_results)}")
    print(f"  • Cross-validation folds: 5")
    print(f"  • Learning curve analysis: Complete")
    print(f"  • Hyperparameter validation: Complete")

    # Best performing models
    best_classifier = max(
        class_results.keys(), key=lambda k: class_results[k]["roc_auc"]
    )
    best_regressor = max(reg_results.keys(), key=lambda k: reg_results[k]["r2"])

    print(f"\\nBest performing models:")
    print(
        f"  • Classification: {best_classifier} (ROC-AUC: {class_results[best_classifier]['roc_auc']:.4f})"
    )
    print(
        f"  • Regression: {best_regressor} (R²: {reg_results[best_regressor]['r2']:.4f})"
    )

    print("\\n🎉 Congratulations! You've mastered model evaluation!")
    print("You're ready for Challenge 4: Hyperparameter Tuning")

    return {
        "classification_results": class_results,
        "regression_results": reg_results,
        "cv_results": {
            "classification": cv_class_results,
            "regression": cv_reg_results,
        },
        "learning_curve": learning_results,
        "validation_curve": validation_results,
        "interpretation": interpretation_results,
    }


if __name__ == "__main__":
    results = main()

    print("\\n" + "=" * 60)
    print("CHALLENGE 3 STATUS: COMPLETE")
    print("=" * 60)
    print("Model evaluation mastery achieved!")
    print("Ready for Challenge 4: Hyperparameter Tuning.")
