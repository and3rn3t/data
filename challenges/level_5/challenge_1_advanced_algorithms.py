"""
Level 5 Challenge 1: Advanced Algorithms and Ensemble Methods
Master sophisticated machine learning algorithms and ensemble techniques.
"""

import warnings
from datetime import datetime

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def create_complex_dataset():
    """Create complex multi-modal dataset for advanced ML"""
    print("Creating complex multi-modal dataset...")

    np.random.seed(42)
    n_customers = 3000

    # Customer demographics and behavior
    data = pd.DataFrame(
        {
            # Demographics
            "age": np.random.normal(35, 12, n_customers),
            "income": np.random.lognormal(10.5, 0.5, n_customers),
            "education_years": np.random.normal(14, 3, n_customers),
            "household_size": np.random.poisson(2.5, n_customers) + 1,
            # Digital behavior
            "web_sessions_month": np.random.poisson(15, n_customers),
            "mobile_sessions_month": np.random.poisson(25, n_customers),
            "pages_per_session": np.random.gamma(2, 3, n_customers),
            "session_duration": np.random.exponential(300, n_customers),  # seconds
            # Purchase behavior
            "purchases_last_year": np.random.poisson(8, n_customers),
            "avg_order_value": np.random.gamma(3, 50, n_customers),
            "days_since_last_purchase": np.random.exponential(30, n_customers),
            "returns_rate": np.random.beta(1, 10, n_customers),
            # Engagement metrics
            "email_open_rate": np.random.beta(3, 5, n_customers),
            "social_media_followers": np.random.pareto(1, n_customers) * 100,
            "customer_service_calls": np.random.poisson(2, n_customers),
            "loyalty_program_tier": np.random.choice(
                [0, 1, 2, 3], n_customers, p=[0.4, 0.3, 0.2, 0.1]
            ),
            # Geographic and temporal
            "region": np.random.choice(["North", "South", "East", "West"], n_customers),
            "urban_rural": np.random.choice(
                ["Urban", "Suburban", "Rural"], n_customers, p=[0.5, 0.3, 0.2]
            ),
            "timezone": np.random.choice(["EST", "CST", "MST", "PST"], n_customers),
        }
    )

    # Clip and normalize some values
    data["age"] = np.clip(data["age"], 18, 80)
    data["education_years"] = np.clip(data["education_years"], 8, 22)
    data["household_size"] = np.clip(data["household_size"], 1, 8)
    data["returns_rate"] = np.clip(data["returns_rate"], 0, 1)
    data["email_open_rate"] = np.clip(data["email_open_rate"], 0, 1)

    # Create complex target variable based on multiple factors
    # Customer value tier (0: Low, 1: Medium, 2: High, 3: Premium)

    # Calculate customer lifetime value score
    clv_score = (
        data["income"] / 100000 * 0.3
        + data["purchases_last_year"] / 20 * 0.2
        + data["avg_order_value"] / 200 * 0.2
        + (1 - data["returns_rate"]) * 0.1
        + data["email_open_rate"] * 0.1
        + data["loyalty_program_tier"] / 3 * 0.1
    )

    # Add some noise and non-linear interactions
    interaction_term = (
        data["web_sessions_month"] * data["pages_per_session"] / 100 * 0.05
        + np.log1p(data["social_media_followers"]) / 10 * 0.03
        + (data["session_duration"] > 300).astype(int) * 0.02
    )

    clv_score += interaction_term + np.random.normal(0, 0.1, n_customers)

    # Convert to categorical target
    clv_percentiles = np.percentile(clv_score, [25, 50, 75])
    data["customer_tier"] = pd.cut(
        clv_score,
        bins=[-np.inf] + list(clv_percentiles) + [np.inf],
        labels=[0, 1, 2, 3],
    )

    # Convert categorical variables
    le_region = LabelEncoder()
    le_urban = LabelEncoder()
    le_timezone = LabelEncoder()

    data["region_encoded"] = le_region.fit_transform(data["region"])
    data["urban_rural_encoded"] = le_urban.fit_transform(data["urban_rural"])
    data["timezone_encoded"] = le_timezone.fit_transform(data["timezone"])

    print(f"Dataset created: {data.shape[0]} customers with {data.shape[1]} features")
    print(
        f"Target distribution: {data['customer_tier'].value_counts().sort_index().to_dict()}"
    )

    return data, [le_region, le_urban, le_timezone]


def prepare_features(data):
    """Prepare features for modeling"""
    print("Preparing features for modeling...")

    # Select numeric and encoded categorical features
    feature_columns = [
        "age",
        "income",
        "education_years",
        "household_size",
        "web_sessions_month",
        "mobile_sessions_month",
        "pages_per_session",
        "session_duration",
        "purchases_last_year",
        "avg_order_value",
        "days_since_last_purchase",
        "returns_rate",
        "email_open_rate",
        "social_media_followers",
        "customer_service_calls",
        "loyalty_program_tier",
        "region_encoded",
        "urban_rural_encoded",
        "timezone_encoded",
    ]

    X = data[feature_columns].copy()
    y = data["customer_tier"].astype(int)

    # Feature engineering
    X["income_per_household"] = X["income"] / X["household_size"]
    X["engagement_score"] = (X["web_sessions_month"] + X["mobile_sessions_month"]) * X[
        "pages_per_session"
    ]
    X["purchase_frequency"] = X["purchases_last_year"] / (
        X["days_since_last_purchase"] + 1
    )
    X["digital_native"] = (X["mobile_sessions_month"] > X["web_sessions_month"]).astype(
        int
    )

    print(f"Features prepared: {X.shape[1]} features")
    print(f"Target classes: {sorted(y.unique())}")

    return X, y


def train_individual_models(X_train, X_test, y_train, y_test):
    """Train individual models for comparison"""
    print("Training individual models...")

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss"
        ),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42, probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
        ),
    }

    results = {}

    for name, model in models.items():
        start_time = datetime.now()

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Probabilities for AUC (if available)
        try:
            test_proba = model.predict_proba(X_test)
            # For multiclass, use macro average AUC
            auc_score = roc_auc_score(
                y_test, test_proba, multi_class="ovr", average="macro"
            )
        except:
            auc_score = None

        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")

        training_time = (datetime.now() - start_time).total_seconds()

        results[name] = {
            "model": model,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_auc": auc_score,
            "training_time": training_time,
        }

        print(
            f"  {name}: Test Acc={test_acc:.3f}, F1={test_f1:.3f}, Time={training_time:.2f}s"
        )

    return results


def create_ensemble_models(X_train, X_test, y_train, y_test, base_models):
    """Create advanced ensemble models"""
    print("Creating ensemble models...")

    # Extract best performing models for ensembles
    model_performance = [
        (name, results["test_f1"]) for name, results in base_models.items()
    ]
    model_performance.sort(key=lambda x: x[1], reverse=True)

    # Select top 5 models for ensembles
    top_models = [
        (name, base_models[name]["model"]) for name, _ in model_performance[:5]
    ]
    print(f"Top performing models for ensembles: {[name for name, _ in top_models]}")

    ensemble_results = {}

    # 1. Voting Classifier (Hard and Soft)
    voting_models = [(name, model) for name, model in top_models]

    # Hard voting
    hard_voting = VotingClassifier(estimators=voting_models, voting="hard")
    hard_voting.fit(X_train, y_train)
    hard_pred = hard_voting.predict(X_test)
    hard_acc = accuracy_score(y_test, hard_pred)
    hard_f1 = f1_score(y_test, hard_pred, average="macro")

    # Soft voting
    soft_voting = VotingClassifier(estimators=voting_models, voting="soft")
    soft_voting.fit(X_train, y_train)
    soft_pred = soft_voting.predict(X_test)
    soft_acc = accuracy_score(y_test, soft_pred)
    soft_f1 = f1_score(y_test, soft_pred, average="macro")

    ensemble_results["Hard Voting"] = {
        "model": hard_voting,
        "test_accuracy": hard_acc,
        "test_f1": hard_f1,
    }

    ensemble_results["Soft Voting"] = {
        "model": soft_voting,
        "test_accuracy": soft_acc,
        "test_f1": soft_f1,
    }

    # 2. Bagging with different base learners
    bagging_dt = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=8),
        n_estimators=50,
        random_state=42,
    )
    bagging_dt.fit(X_train, y_train)
    bagging_pred = bagging_dt.predict(X_test)
    bagging_acc = accuracy_score(y_test, bagging_pred)
    bagging_f1 = f1_score(y_test, bagging_pred, average="macro")

    ensemble_results["Bagging"] = {
        "model": bagging_dt,
        "test_accuracy": bagging_acc,
        "test_f1": bagging_f1,
    }

    # 3. Stacking Classifier
    # Use top 4 as base learners and logistic regression as meta-learner
    base_learners = [(name, model) for name, model in top_models[:4]]

    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(random_state=42),
        cv=3,
        stack_method="predict_proba",
    )
    stacking.fit(X_train, y_train)
    stacking_pred = stacking.predict(X_test)
    stacking_acc = accuracy_score(y_test, stacking_pred)
    stacking_f1 = f1_score(y_test, stacking_pred, average="macro")

    ensemble_results["Stacking"] = {
        "model": stacking,
        "test_accuracy": stacking_acc,
        "test_f1": stacking_f1,
    }

    print("Ensemble Results:")
    for name, results in ensemble_results.items():
        print(
            f"  {name}: Accuracy={results['test_accuracy']:.3f}, F1={results['test_f1']:.3f}"
        )

    return ensemble_results


def analyze_model_performance(base_results, ensemble_results, X_test, y_test):
    """Comprehensive performance analysis"""
    print("\\nAnalyzing model performance...")

    # Combine all results
    all_results = {**base_results, **ensemble_results}

    # Create performance comparison
    performance_df = pd.DataFrame(
        {
            "Model": list(all_results.keys()),
            "Test_Accuracy": [r.get("test_accuracy", 0) for r in all_results.values()],
            "Test_F1": [r.get("test_f1", 0) for r in all_results.values()],
            "Test_AUC": [r.get("test_auc", 0) for r in all_results.values()],
            "Training_Time": [r.get("training_time", 0) for r in all_results.values()],
        }
    )

    performance_df = performance_df.sort_values("Test_F1", ascending=False)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Model Performance Analysis", fontsize=16, fontweight="bold")

    # 1. Accuracy comparison
    ax = axes[0, 0]
    bars = ax.barh(
        performance_df["Model"], performance_df["Test_Accuracy"], color="skyblue"
    )
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlim(0, 1)

    # Add value labels
    for bar, acc in zip(bars, performance_df["Test_Accuracy"]):
        ax.text(
            acc + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}",
            va="center",
            fontsize=8,
        )

    # 2. F1-Score comparison
    ax = axes[0, 1]
    bars = ax.barh(
        performance_df["Model"], performance_df["Test_F1"], color="lightgreen"
    )
    ax.set_xlabel("Test F1-Score")
    ax.set_title("Model F1-Score Comparison")
    ax.set_xlim(0, 1)

    # Add value labels
    for bar, f1 in zip(bars, performance_df["Test_F1"]):
        ax.text(
            f1 + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{f1:.3f}",
            va="center",
            fontsize=8,
        )

    # 3. Training time analysis
    ax = axes[1, 0]
    base_models_mask = performance_df["Training_Time"] > 0
    base_models_df = performance_df[base_models_mask]

    ax.scatter(
        base_models_df["Training_Time"],
        base_models_df["Test_F1"],
        s=100,
        alpha=0.6,
        c="orange",
    )

    for i, row in base_models_df.iterrows():
        ax.annotate(
            row["Model"],
            (row["Training_Time"], row["Test_F1"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Test F1-Score")
    ax.set_title("Performance vs Training Time")
    ax.grid(True, alpha=0.3)

    # 4. Performance distribution
    ax = axes[1, 1]
    metrics = ["Test_Accuracy", "Test_F1"]
    x_pos = np.arange(len(metrics))

    means = [performance_df["Test_Accuracy"].mean(), performance_df["Test_F1"].mean()]
    stds = [performance_df["Test_Accuracy"].std(), performance_df["Test_F1"].std()]

    ax.bar(x_pos, means, yerr=stds, capsize=5, color=["skyblue", "lightgreen"])
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("Average Performance Across All Models")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f"{mean:.3f}±{std:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    return performance_df


def feature_importance_analysis(models_dict, X_test, y_test, feature_names):
    """Analyze feature importance across models"""
    print("Analyzing feature importance...")

    # Select models that have feature importance
    importance_models = {}

    for name, results in models_dict.items():
        model = results["model"]

        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importance_models[name] = model.feature_importances_

        # For ensemble models, try to get feature importance
        elif hasattr(model, "estimators_") and hasattr(
            model.estimators_[0], "feature_importances_"
        ):
            # Average importance across estimators
            importances = np.mean(
                [est.feature_importances_ for est in model.estimators_], axis=0
            )
            importance_models[name] = importances

    if not importance_models:
        print("No models with feature importance found.")
        return None

    # Create feature importance dataframe
    importance_df = pd.DataFrame(importance_models, index=feature_names)

    # Calculate permutation importance for top model
    best_model_name = max(
        models_dict.keys(), key=lambda x: models_dict[x].get("test_f1", 0)
    )
    best_model = models_dict[best_model_name]["model"]

    perm_importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=42
    )

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle("Feature Importance Analysis", fontsize=16, fontweight="bold")

    # 1. Feature importance heatmap
    ax = axes[0]
    sns.heatmap(importance_df, annot=True, cmap="viridis", ax=ax, fmt=".3f")
    ax.set_title("Feature Importance Across Models")
    ax.set_xlabel("Models")
    ax.set_ylabel("Features")

    # 2. Permutation importance
    ax = axes[1]
    perm_imp_mean = perm_importance.importances_mean
    perm_imp_std = perm_importance.importances_std

    sorted_idx = perm_imp_mean.argsort()
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    ax.barh(pos, perm_imp_mean[sorted_idx], xerr=perm_imp_std[sorted_idx])
    ax.set_yticks(pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Permutation Importance")
    ax.set_title(f"Permutation Importance - {best_model_name}")

    plt.tight_layout()
    plt.show()

    return importance_df, perm_importance


def main():
    """Main function to run advanced algorithms challenge"""
    print("=" * 60)
    print("LEVEL 5 CHALLENGE 1: ADVANCED ALGORITHMS & ENSEMBLES")
    print("=" * 60)

    # Create dataset
    data, encoders = create_complex_dataset()

    # Prepare features
    X, y = prepare_features(data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\\nData split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # Train individual models
    print("\\n" + "=" * 50)
    print("TRAINING INDIVIDUAL MODELS")
    print("=" * 50)
    base_results = train_individual_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Create ensemble models
    print("\\n" + "=" * 50)
    print("CREATING ENSEMBLE MODELS")
    print("=" * 50)
    ensemble_results = create_ensemble_models(
        X_train_scaled, X_test_scaled, y_train, y_test, base_results
    )

    # Analyze performance
    print("\\n" + "=" * 50)
    print("PERFORMANCE ANALYSIS")
    print("=" * 50)
    performance_df = analyze_model_performance(
        base_results, ensemble_results, X_test_scaled, y_test
    )

    # Feature importance analysis
    print("\\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    importance_df, perm_imp = feature_importance_analysis(
        base_results, X_test_scaled, y_test, X.columns.tolist()
    )

    # Summary
    print("\\n" + "=" * 60)
    print("CHALLENGE 1 COMPLETION SUMMARY")
    print("=" * 60)

    best_model = performance_df.iloc[0]
    print(f"Best performing model: {best_model['Model']}")
    print(f"  - Test Accuracy: {best_model['Test_Accuracy']:.3f}")
    print(f"  - Test F1-Score: {best_model['Test_F1']:.3f}")

    algorithms_implemented = [
        "Decision Trees with pruning",
        "Random Forest ensemble",
        "Extra Trees randomization",
        "Gradient Boosting optimization",
        "AdaBoost adaptive boosting",
        "XGBoost extreme gradient boosting",
        "LightGBM efficient boosting",
        "Hard & Soft Voting ensembles",
        "Bagging with diversity",
        "Stacking meta-learning",
        "Support Vector Machines",
        "Neural Networks",
    ]

    print("\\nAlgorithms implemented:")
    for i, algorithm in enumerate(algorithms_implemented, 1):
        print(f"  {i}. {algorithm}")

    print(f"\\nDataset Statistics:")
    print(f"  - Customers analyzed: {len(data):,}")
    print(f"  - Features engineered: {X.shape[1]}")
    print(f"  - Classes predicted: {len(y.unique())}")
    print(f"  - Models trained: {len(base_results) + len(ensemble_results)}")

    return {
        "data": data,
        "performance_df": performance_df,
        "base_results": base_results,
        "ensemble_results": ensemble_results,
        "feature_importance": importance_df,
    }


if __name__ == "__main__":
    results = main()

    print("\\n" + "=" * 60)
    print("CHALLENGE 1 STATUS: COMPLETE")
    print("=" * 60)
    print("Advanced algorithms and ensemble methods mastery achieved!")
    print("Ready for Challenge 2: Deep Learning Architectures.")
