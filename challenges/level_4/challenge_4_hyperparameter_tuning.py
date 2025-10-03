#!/usr/bin/env python3
"""
Level 4: Machine Learning Novice
Challenge 4: Hyperparameter Tuning Mastery

Master hyperparameter optimization techniques for peak model performance.
"""

import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
    validation_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")


def create_tuning_datasets(random_state=42):
    """Create datasets for hyperparameter tuning demonstrations"""
    print("📊 Creating Datasets for Hyperparameter Tuning...")

    # Classification dataset
    x_class, y_class = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=random_state,
    )

    # Regression dataset
    x_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=random_state,
    )

    print(
        f"Classification dataset: {x_class.shape[0]} samples, {x_class.shape[1]} features"
    )
    print(f"Regression dataset: {x_reg.shape[0]} samples, {x_reg.shape[1]} features")

    return x_class, y_class, x_reg, y_reg


def manual_hyperparameter_tuning(X, y, task_type="classification"):
    """Demonstrate manual hyperparameter tuning with validation curves"""
    print(f"\\n=== MANUAL HYPERPARAMETER TUNING ({task_type.upper()}) ===")

    if task_type == "classification":
        model = RandomForestClassifier(random_state=42)
        param_name = "n_estimators"
        param_range = [10, 25, 50, 100, 200, 500]
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(random_state=42)
        param_name = "n_estimators"
        param_range = [10, 25, 50, 100, 200, 500]
        scoring = "r2"

    # Manual tuning using validation curves
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

    # Find best parameter
    best_idx = np.argmax(val_mean)
    best_param = param_range[best_idx]
    best_score = val_mean[best_idx]

    print(f"Manual tuning results for {param_name}:")
    for i, param in enumerate(param_range):
        print(f"  {param_name}={param}: {val_mean[i]:.4f} (+/- {val_std[i]:.4f})")

    print(f"\\nBest {param_name}: {best_param}")
    print(f"Best {scoring} score: {best_score:.4f}")

    # Visualization
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

    plt.axvline(
        x=best_param,
        color="green",
        linestyle="--",
        alpha=0.8,
        label=f"Best {param_name}: {best_param}",
    )

    plt.xlabel(param_name.replace("_", " ").title())
    plt.ylabel(f"{scoring.upper()} Score")
    plt.title(f"Manual Hyperparameter Tuning: {param_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "best_param": best_param,
        "best_score": best_score,
        "param_range": param_range,
        "train_scores": train_scores,
        "val_scores": val_scores,
    }


def grid_search_tuning(X, y, task_type="classification"):
    """Demonstrate systematic grid search hyperparameter tuning"""
    print(f"\\n=== GRID SEARCH TUNING ({task_type.upper()}) ===")

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if task_type == "classification" else None,
    )

    if task_type == "classification":
        # Random Forest Classification
        rf_model = RandomForestClassifier(random_state=42)
        rf_params = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        # SVM Classification
        svm_model = SVC(random_state=42)
        svm_params = {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto", 0.001, 0.01],
        }

        scoring = "accuracy"

    else:
        # Random Forest Regression
        rf_model = RandomForestRegressor(random_state=42)
        rf_params = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        # SVR
        svm_model = SVR()
        svm_params = {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto", 0.001, 0.01],
        }

        scoring = "r2"

    models_and_params = [
        ("Random Forest", rf_model, rf_params),
        ("SVM", svm_model, svm_params),
    ]

    grid_results = {}

    for name, model, params in models_and_params:
        print(f"\\n--- Grid Search for {name} ---")
        print(
            f"Parameter combinations to test: {np.prod([len(v) for v in params.values()])}"
        )

        # Perform grid search
        grid_search = GridSearchCV(
            model, params, cv=5, scoring=scoring, n_jobs=-1, verbose=0
        )

        # For SVM, scale the data
        if name == "SVM":
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            grid_search.fit(x_train_scaled, y_train)

            # Evaluate on test set
            if task_type == "classification":
                test_score = accuracy_score(y_test, grid_search.predict(x_test_scaled))
            else:
                test_score = r2_score(y_test, grid_search.predict(x_test_scaled))
        else:
            grid_search.fit(x_train, y_train)

            # Evaluate on test set
            if task_type == "classification":
                test_score = accuracy_score(y_test, grid_search.predict(x_test))
            else:
                test_score = r2_score(y_test, grid_search.predict(x_test))

        grid_results[name] = {
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "test_score": test_score,
            "grid_search": grid_search,
        }

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Test set score: {test_score:.4f}")

    # Compare models
    print(f"\\n=== GRID SEARCH COMPARISON ===")
    comparison_df = pd.DataFrame(
        {
            name: {
                "Best CV Score": results["best_cv_score"],
                "Test Score": results["test_score"],
            }
            for name, results in grid_results.items()
        }
    ).T

    print(comparison_df.round(4))

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    comparison_df.plot(kind="bar", ax=plt.gca())
    plt.title("Grid Search Results Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Best model performance visualization
    best_model_name = comparison_df["Test Score"].idxmax()
    best_grid_search = grid_results[best_model_name]["grid_search"]

    plt.subplot(1, 2, 2)
    # Show top 10 parameter combinations
    results_df = pd.DataFrame(best_grid_search.cv_results_)
    top_results = results_df.nlargest(10, "mean_test_score")

    plt.barh(range(len(top_results)), top_results["mean_test_score"])
    plt.yticks(
        range(len(top_results)), [f"Config {i+1}" for i in range(len(top_results))]
    )
    plt.xlabel(f"{scoring.upper()} Score")
    plt.title(f"Top 10 Configurations: {best_model_name}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return grid_results


def randomized_search_tuning(X, y, task_type="classification"):
    """Demonstrate randomized search for efficient hyperparameter tuning"""
    print(f"\\n=== RANDOMIZED SEARCH TUNING ({task_type.upper()}) ===")

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if task_type == "classification" else None,
    )

    if task_type == "classification":
        model = RandomForestClassifier(random_state=42)
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(random_state=42)
        scoring = "r2"

    # Define parameter distributions for randomized search
    param_distributions = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    # Compare different numbers of iterations
    n_iter_values = [10, 50, 100]
    random_results = {}

    for n_iter in n_iter_values:
        print(f"\\n--- Randomized Search with {n_iter} iterations ---")

        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )

        random_search.fit(x_train, y_train)

        # Evaluate on test set
        if task_type == "classification":
            test_score = accuracy_score(y_test, random_search.predict(x_test))
        else:
            test_score = r2_score(y_test, random_search.predict(x_test))

        random_results[n_iter] = {
            "best_params": random_search.best_params_,
            "best_cv_score": random_search.best_score_,
            "test_score": test_score,
            "random_search": random_search,
        }

        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        print(f"Test set score: {test_score:.4f}")

    # Compare iterations
    print(f"\\n=== RANDOMIZED SEARCH ITERATION COMPARISON ===")
    iteration_df = pd.DataFrame(
        {
            f"{n_iter} iterations": {
                "Best CV Score": results["best_cv_score"],
                "Test Score": results["test_score"],
            }
            for n_iter, results in random_results.items()
        }
    ).T

    print(iteration_df.round(4))

    # Visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    iteration_df.plot(kind="bar", ax=plt.gca())
    plt.title("Randomized Search: Iterations Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Parameter importance analysis
    best_random_search = random_results[max(n_iter_values)]["random_search"]
    results_df = pd.DataFrame(best_random_search.cv_results_)

    # Analyze parameter correlations with performance
    param_columns = [col for col in results_df.columns if col.startswith("param_")]

    plt.subplot(2, 2, 2)
    # Show distribution of scores
    plt.hist(results_df["mean_test_score"], bins=20, alpha=0.7, edgecolor="black")
    plt.xlabel(f"{scoring.upper()} Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution (Randomized Search)")
    plt.grid(True, alpha=0.3)

    # Show parameter value distributions for top performers
    plt.subplot(2, 2, 3)
    top_10_percent = results_df.nlargest(int(len(results_df) * 0.1), "mean_test_score")

    # Focus on numeric parameters
    numeric_params = [
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
    ]
    numeric_data = []
    labels = []

    for param in numeric_params:
        if param in top_10_percent.columns:
            values = pd.to_numeric(top_10_percent[param], errors="coerce").dropna()
            if len(values) > 0:
                numeric_data.append(values)
                labels.append(param.replace("param_", ""))

    if numeric_data:
        plt.boxplot(numeric_data, labels=labels)
        plt.title("Top 10% Parameter Distributions")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    # Convergence analysis
    plt.subplot(2, 2, 4)
    all_scores = []
    for n_iter in n_iter_values:
        scores = pd.DataFrame(random_results[n_iter]["random_search"].cv_results_)[
            "mean_test_score"
        ]
        cumulative_best = scores.expanding().max()
        all_scores.append(cumulative_best.values)
        plt.plot(
            range(1, len(cumulative_best) + 1),
            cumulative_best,
            label=f"{n_iter} iterations",
            marker="o",
            markersize=3,
        )

    plt.xlabel("Iteration")
    plt.ylabel(f"Best {scoring.upper()} Score So Far")
    plt.title("Randomized Search Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return random_results


def bayesian_optimization_simulation(X, y, task_type="classification"):
    """Simulate Bayesian optimization concepts using intelligent sampling"""
    print(f"\\n=== BAYESIAN OPTIMIZATION SIMULATION ({task_type.upper()}) ===")

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if task_type == "classification" else None,
    )

    if task_type == "classification":
        model = RandomForestClassifier(random_state=42)
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(random_state=42)
        scoring = "r2"

    print("Simulating Bayesian Optimization approach...")
    print("(Using intelligent parameter selection based on previous results)")

    # Define search space
    param_space = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 15],
    }

    # Simulate Bayesian optimization with intelligent sampling
    n_iterations = 20
    evaluated_params = []
    evaluated_scores = []

    rng = np.random.default_rng(42)

    for iteration in range(n_iterations):
        if iteration < 5:
            # Initial random sampling
            params = {
                "n_estimators": rng.choice(param_space["n_estimators"]),
                "max_depth": rng.choice(param_space["max_depth"]),
                "min_samples_split": rng.choice(param_space["min_samples_split"]),
            }
        else:
            # Intelligent sampling (simulate acquisition function)
            # In real Bayesian optimization, this would use Gaussian processes
            if evaluated_scores:
                best_idx = np.argmax(evaluated_scores)
                best_params = evaluated_params[best_idx]

                # Sample near best parameters with some exploration
                params = {}
                for param_name, param_value in best_params.items():
                    space = param_space[param_name]
                    if param_value in space:
                        idx = space.index(param_value)
                        # Sample nearby values
                        nearby_indices = [
                            max(0, idx - 1),
                            idx,
                            min(len(space) - 1, idx + 1),
                        ]
                        params[param_name] = space[rng.choice(nearby_indices)]
                    else:
                        params[param_name] = rng.choice(space)

        # Evaluate parameters
        temp_model = model.__class__(**params, random_state=42)
        cv_scores = cross_val_score(temp_model, x_train, y_train, cv=3, scoring=scoring)
        score = cv_scores.mean()

        evaluated_params.append(params)
        evaluated_scores.append(score)

        if iteration % 5 == 0 or iteration < 5:
            print(f"Iteration {iteration+1}: {params} -> {score:.4f}")

    # Find best parameters
    best_idx = np.argmax(evaluated_scores)
    best_params = evaluated_params[best_idx]
    best_score = evaluated_scores[best_idx]

    print(f"\\nBest parameters found: {best_params}")
    print(f"Best CV score: {best_score:.4f}")

    # Train final model and evaluate on test set
    final_model = model.__class__(**best_params, random_state=42)
    final_model.fit(x_train, y_train)

    if task_type == "classification":
        test_score = accuracy_score(y_test, final_model.predict(x_test))
    else:
        test_score = r2_score(y_test, final_model.predict(x_test))

    print(f"Test set score: {test_score:.4f}")

    # Visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(evaluated_scores) + 1), evaluated_scores, "bo-")
    plt.axhline(
        y=best_score, color="r", linestyle="--", label=f"Best: {best_score:.4f}"
    )
    plt.xlabel("Iteration")
    plt.ylabel(f"{scoring.upper()} Score")
    plt.title("Bayesian Optimization Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    # Running best score
    running_best = []
    current_best = -np.inf
    for score in evaluated_scores:
        if score > current_best:
            current_best = score
        running_best.append(current_best)

    plt.plot(range(1, len(running_best) + 1), running_best, "go-")
    plt.xlabel("Iteration")
    plt.ylabel(f"Best {scoring.upper()} Score So Far")
    plt.title("Best Score Convergence")
    plt.grid(True, alpha=0.3)

    # Parameter exploration visualization
    plt.subplot(2, 2, 3)
    n_estimators_values = [p["n_estimators"] for p in evaluated_params]
    plt.scatter(
        range(len(n_estimators_values)),
        n_estimators_values,
        c=evaluated_scores,
        cmap="viridis",
    )
    plt.colorbar(label=f"{scoring.upper()} Score")
    plt.xlabel("Iteration")
    plt.ylabel("n_estimators")
    plt.title("Parameter Exploration: n_estimators")

    plt.subplot(2, 2, 4)
    max_depth_values = [
        p["max_depth"] if p["max_depth"] is not None else 25 for p in evaluated_params
    ]
    plt.scatter(
        range(len(max_depth_values)),
        max_depth_values,
        c=evaluated_scores,
        cmap="viridis",
    )
    plt.colorbar(label=f"{scoring.upper()} Score")
    plt.xlabel("Iteration")
    plt.ylabel("max_depth (None=25)")
    plt.title("Parameter Exploration: max_depth")

    plt.tight_layout()
    plt.show()

    return {
        "best_params": best_params,
        "best_score": best_score,
        "test_score": test_score,
        "evaluated_params": evaluated_params,
        "evaluated_scores": evaluated_scores,
    }


def hyperparameter_tuning_comparison(x_class, y_class, x_reg, y_reg):
    """Compare different hyperparameter tuning approaches"""
    print("\\n=== HYPERPARAMETER TUNING METHODS COMPARISON ===")

    # Run all methods on classification data
    print("\\n--- CLASSIFICATION COMPARISON ---")

    # Manual tuning
    manual_class = manual_hyperparameter_tuning(x_class, y_class, "classification")

    # Grid search (simplified for comparison)
    simple_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
        cv=3,
        scoring="accuracy",
    )
    simple_grid.fit(x_class, y_class)

    # Randomized search
    simple_random = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200, 300], "max_depth": [5, 10, 15, None]},
        n_iter=20,
        cv=3,
        scoring="accuracy",
        random_state=42,
    )
    simple_random.fit(x_class, y_class)

    # Bayesian simulation
    bayesian_class = bayesian_optimization_simulation(
        x_class, y_class, "classification"
    )

    # Comparison table
    comparison_results = {
        "Manual Tuning": manual_class["best_score"],
        "Grid Search": simple_grid.best_score_,
        "Randomized Search": simple_random.best_score_,
        "Bayesian Optimization": bayesian_class["best_score"],
    }

    print("\\nClassification Results Comparison:")
    for method, score in comparison_results.items():
        print(f"  {method}: {score:.4f}")

    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    methods = list(comparison_results.keys())
    scores = list(comparison_results.values())
    bars = plt.bar(methods, scores, color=["blue", "orange", "green", "red"])
    plt.ylabel("Accuracy Score")
    plt.title("Hyperparameter Tuning Methods Comparison")
    plt.xticks(rotation=45)
    plt.ylim(min(scores) - 0.01, max(scores) + 0.01)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{score:.4f}",
            ha="center",
            va="bottom",
        )

    plt.grid(True, alpha=0.3)

    # Efficiency comparison (conceptual)
    plt.subplot(1, 2, 2)
    efficiency_data = {
        "Manual Tuning": 6,  # number of evaluations
        "Grid Search": 9,  # 3*3 combinations
        "Randomized Search": 20,  # n_iter
        "Bayesian Optimization": 20,  # n_iterations
    }

    methods = list(efficiency_data.keys())
    evaluations = list(efficiency_data.values())
    bars = plt.bar(methods, evaluations, color=["blue", "orange", "green", "red"])
    plt.ylabel("Number of Evaluations")
    plt.title("Computational Efficiency Comparison")
    plt.xticks(rotation=45)

    # Add value labels
    for bar, evals in zip(bars, evaluations):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(evals),
            ha="center",
            va="bottom",
        )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return comparison_results


def main():
    """Main function to run the hyperparameter tuning challenge"""
    print("=" * 60)
    print("LEVEL 4 CHALLENGE 4: HYPERPARAMETER TUNING MASTERY")
    print("=" * 60)

    print("🔧 Welcome to Hyperparameter Tuning!")
    print("Learn systematic approaches to optimize model performance.")

    # 1. Create datasets
    x_class, y_class, x_reg, y_reg = create_tuning_datasets()

    # 2. Manual hyperparameter tuning
    manual_results = manual_hyperparameter_tuning(x_class, y_class)

    # 3. Grid search tuning
    grid_results = grid_search_tuning(x_class, y_class)

    # 4. Randomized search tuning
    random_results = randomized_search_tuning(x_class, y_class)

    # 5. Bayesian optimization simulation
    bayesian_results = bayesian_optimization_simulation(x_class, y_class)

    # 6. Comparison of methods
    comparison_results = hyperparameter_tuning_comparison(
        x_class, y_class, x_reg, y_reg
    )

    # Summary
    print("\\n" + "=" * 60)
    print("CHALLENGE 4 COMPLETION SUMMARY")
    print("=" * 60)

    print("Hyperparameter Tuning techniques mastered:")
    techniques = [
        "📊 Manual tuning with validation curves",
        "🔍 Exhaustive Grid Search optimization",
        "🎲 Efficient Randomized Search",
        "🧠 Bayesian Optimization concepts",
        "📈 Parameter space exploration strategies",
        "⚡ Computational efficiency analysis",
        "📋 Cross-validation integration",
        "🎯 Model performance optimization",
        "📊 Hyperparameter importance analysis",
    ]

    for technique in techniques:
        print(f"  {technique}")

    print(f"\\nTuning methods comparison:")
    for method, score in comparison_results.items():
        print(f"  • {method}: {score:.4f} accuracy")

    best_method = max(comparison_results.keys(), key=lambda k: comparison_results[k])
    print(f"\\n🏆 Best performing method: {best_method}")
    print(f"Best score achieved: {comparison_results[best_method]:.4f}")

    print(f"\\nOptimization insights:")
    print(f"  • Grid Search: Exhaustive but expensive")
    print(f"  • Randomized Search: Efficient for large spaces")
    print(f"  • Bayesian Optimization: Intelligent exploration")
    print(f"  • Manual Tuning: Good for understanding individual parameters")

    print("\\n🎉 Congratulations! You've mastered hyperparameter tuning!")
    print("Ready for the Level 4 Capstone Challenge!")

    return {
        "manual_results": manual_results,
        "grid_results": grid_results,
        "random_results": random_results,
        "bayesian_results": bayesian_results,
        "comparison_results": comparison_results,
    }


if __name__ == "__main__":
    results = main()

    print("\\n" + "=" * 60)
    print("CHALLENGE 4 STATUS: COMPLETE")
    print("=" * 60)
    print("Hyperparameter tuning mastery achieved!")
    print("Ready for the Level 4 Capstone Challenge.")
