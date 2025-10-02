"""
Level 5 Challenge 3: Advanced Feature Engineering and Automated ML
Master sophisticated feature engineering and automated ML pipelines.
"""

import warnings
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def create_comprehensive_dataset():
    """Create comprehensive multi-modal dataset for feature engineering"""
    print("Creating comprehensive multi-modal dataset...")

    np.random.seed(42)
    n_samples = 5000

    # Customer analytics dataset with multiple data types
    data = pd.DataFrame(
        {
            # Demographic features
            "age": np.random.normal(35, 12, n_samples),
            "income": np.random.lognormal(10.5, 0.8, n_samples),
            "education_score": np.random.normal(75, 15, n_samples),
            "household_size": np.random.poisson(2.5, n_samples) + 1,
            "years_experience": np.random.exponential(5, n_samples),
            # Behavioral metrics (skewed distributions)
            "web_sessions": np.random.pareto(1, n_samples) * 10,
            "page_views": np.random.gamma(2, 5, n_samples),
            "time_on_site": np.random.weibull(2, n_samples) * 300,
            "bounce_rate": np.random.beta(2, 5, n_samples),
            "conversion_rate": np.random.beta(1, 10, n_samples),
            # Financial metrics
            "credit_score": np.random.normal(650, 80, n_samples),
            "debt_ratio": np.random.beta(2, 3, n_samples),
            "savings_rate": np.random.beta(3, 7, n_samples),
            "investment_amount": np.random.pareto(1.5, n_samples) * 1000,
            # Geographic and categorical
            "region": np.random.choice(["North", "South", "East", "West"], n_samples),
            "city_tier": np.random.choice(
                ["Tier1", "Tier2", "Tier3"], n_samples, p=[0.3, 0.5, 0.2]
            ),
            "device_type": np.random.choice(
                ["Mobile", "Desktop", "Tablet"], n_samples, p=[0.6, 0.3, 0.1]
            ),
            "subscription_type": np.random.choice(
                ["Free", "Basic", "Premium"], n_samples, p=[0.5, 0.3, 0.2]
            ),
        }
    )

    # Clip values to realistic ranges
    data["age"] = np.clip(data["age"], 18, 80)
    data["education_score"] = np.clip(data["education_score"], 0, 100)
    data["household_size"] = np.clip(data["household_size"], 1, 8)
    data["years_experience"] = np.clip(data["years_experience"], 0, 45)
    data["credit_score"] = np.clip(data["credit_score"], 300, 850)
    data["bounce_rate"] = np.clip(data["bounce_rate"], 0, 1)
    data["conversion_rate"] = np.clip(data["conversion_rate"], 0, 1)
    data["debt_ratio"] = np.clip(data["debt_ratio"], 0, 1)
    data["savings_rate"] = np.clip(data["savings_rate"], 0, 1)

    # Create text features (simulate product reviews)
    review_templates = [
        "excellent product quality service",
        "poor customer support experience disappointing",
        "good value money satisfied purchase",
        "terrible delivery slow shipping",
        "amazing quality highly recommend",
        "average product nothing special",
        "outstanding service quick response",
        "defective product return refund",
    ]

    data["review_text"] = np.random.choice(review_templates, n_samples)

    # Create complex target variable
    # Customer lifetime value category (0: Low, 1: Medium, 2: High)
    clv_score = (
        np.log1p(data["income"]) * 0.3
        + data["education_score"] / 100 * 0.15
        + (1 - data["bounce_rate"]) * 0.2
        + data["conversion_rate"] * 0.2
        + data["credit_score"] / 850 * 0.15
    )

    # Add interaction effects and noise
    interaction = (data["age"] / 80) * (data["years_experience"] / 45) * 0.1
    clv_score += interaction + np.random.normal(0, 0.05, n_samples)

    # Convert to categorical
    clv_percentiles = np.percentile(clv_score, [33, 67])
    data["customer_value"] = pd.cut(
        clv_score, bins=[-np.inf] + list(clv_percentiles) + [np.inf], labels=[0, 1, 2]
    )

    print(f"Dataset created: {data.shape}")
    print(
        f"Target distribution: {data['customer_value'].value_counts().sort_index().to_dict()}"
    )
    print(f"Missing values: {data.isnull().sum().sum()}")

    return data


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for advanced feature engineering"""

    def __init__(
        self,
        include_interactions=True,
        include_ratios=True,
        include_aggregates=True,
        polynomial_degree=2,
    ):
        self.include_interactions = include_interactions
        self.include_ratios = include_ratios
        self.include_aggregates = include_aggregates
        self.polynomial_degree = polynomial_degree

    def fit(self, X, y=None):
        # Identify numeric columns
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()

        # Identify important column pairs for interactions
        if self.include_interactions and len(self.numeric_columns_) > 1:
            self.interaction_pairs_ = list(
                combinations(self.numeric_columns_[:8], 2)
            )  # Limit pairs
        else:
            self.interaction_pairs_ = []

        return self

    def transform(self, X):
        X_new = X.copy()

        if self.include_interactions:
            # Create interaction features
            for col1, col2 in self.interaction_pairs_:
                X_new[f"{col1}_x_{col2}"] = X[col1] * X[col2]

        if self.include_ratios:
            # Create ratio features (avoid division by zero)
            for i, col1 in enumerate(self.numeric_columns_):
                for col2 in self.numeric_columns_[i + 1 :]:
                    if col2 != col1:
                        # Safe division
                        denominator = X[col2].replace(0, np.finfo(float).eps)
                        X_new[f"{col1}_div_{col2}"] = X[col1] / denominator

        if self.include_aggregates:
            # Statistical aggregates
            numeric_data = X[self.numeric_columns_]
            X_new["row_mean"] = numeric_data.mean(axis=1)
            X_new["row_std"] = numeric_data.std(axis=1)
            X_new["row_max"] = numeric_data.max(axis=1)
            X_new["row_min"] = numeric_data.min(axis=1)
            X_new["row_range"] = X_new["row_max"] - X_new["row_min"]

        return X_new


def create_text_features(data):
    """Create advanced text features"""
    print("Creating text features...")

    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=50, stop_words="english", ngram_range=(1, 2))
    tfidf_features = tfidf.fit_transform(data["review_text"])

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])],
    )

    # Text statistics
    text_stats = pd.DataFrame(
        {
            "text_length": data["review_text"].str.len(),
            "word_count": data["review_text"].str.split().str.len(),
            "avg_word_length": data["review_text"].apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
            ),
            "sentiment_positive": data["review_text"]
            .str.contains("excellent|good|amazing|outstanding")
            .astype(int),
            "sentiment_negative": data["review_text"]
            .str.contains("poor|terrible|disappointing|defective")
            .astype(int),
        }
    )

    # Combine text features
    text_features = pd.concat([tfidf_df, text_stats], axis=1)

    print(f"Text features created: {text_features.shape[1]} features")
    return text_features


def create_categorical_features(data):
    """Create advanced categorical feature encodings"""
    print("Creating categorical features...")

    categorical_cols = ["region", "city_tier", "device_type", "subscription_type"]

    # One-hot encoding
    one_hot = pd.get_dummies(data[categorical_cols], prefix=categorical_cols)

    # Target encoding (mean encoding)
    target_encoded = pd.DataFrame()
    for col in categorical_cols:
        # Calculate mean target for each category
        target_means = data.groupby(col)["customer_value"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean()
        )
        target_encoded[f"{col}_target_enc"] = data[col].map(target_means)

    # Frequency encoding
    freq_encoded = pd.DataFrame()
    for col in categorical_cols:
        freq_counts = data[col].value_counts()
        freq_encoded[f"{col}_freq"] = data[col].map(freq_counts)

    categorical_features = pd.concat([one_hot, target_encoded, freq_encoded], axis=1)

    print(f"Categorical features created: {categorical_features.shape[1]} features")
    return categorical_features


def advanced_feature_selection(X, y, feature_names):
    """Perform advanced feature selection"""
    print("Performing advanced feature selection...")

    results = {}

    # 1. Variance Threshold
    var_threshold = VarianceThreshold(threshold=0.01)
    X_var = var_threshold.fit_transform(X)
    var_selected = var_threshold.get_support()
    results["variance_threshold"] = {
        "n_features": X_var.shape[1],
        "selected_features": np.array(feature_names)[var_selected].tolist(),
    }

    # 2. Univariate Selection
    selector_chi2 = SelectKBest(chi2, k=50)
    X_chi2 = selector_chi2.fit_transform(np.abs(X), y)  # Chi2 needs non-negative
    chi2_selected = selector_chi2.get_support()
    results["chi2_selection"] = {
        "n_features": X_chi2.shape[1],
        "scores": selector_chi2.scores_,
        "selected_features": np.array(feature_names)[chi2_selected].tolist(),
    }

    # 3. Mutual Information
    selector_mi = SelectKBest(mutual_info_classif, k=50)
    X_mi = selector_mi.fit_transform(X, y)
    mi_selected = selector_mi.get_support()
    results["mutual_info_selection"] = {
        "n_features": X_mi.shape[1],
        "scores": selector_mi.scores_,
        "selected_features": np.array(feature_names)[mi_selected].tolist(),
    }

    # 4. Model-based selection (Random Forest)
    rf_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    X_rf = rf_selector.fit_transform(X, y)
    rf_selected = rf_selector.get_support()
    results["random_forest_selection"] = {
        "n_features": X_rf.shape[1],
        "feature_importances": rf_selector.estimator_.feature_importances_,
        "selected_features": np.array(feature_names)[rf_selected].tolist(),
    }

    # 5. Recursive Feature Elimination
    rfe_selector = RFE(LogisticRegression(random_state=42), n_features_to_select=50)
    X_rfe = rfe_selector.fit_transform(X, y)
    rfe_selected = rfe_selector.get_support()
    results["rfe_selection"] = {
        "n_features": X_rfe.shape[1],
        "ranking": rfe_selector.ranking_,
        "selected_features": np.array(feature_names)[rfe_selected].tolist(),
    }

    # Print summary
    for method, result in results.items():
        print(f"  {method}: {result['n_features']} features selected")

    return results


def create_dimensionality_reduction_features(X, feature_names):
    """Create features using dimensionality reduction"""
    print("Creating dimensionality reduction features...")

    # Standardize first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reduction_features = pd.DataFrame()

    # 1. PCA
    pca = PCA(n_components=20, random_state=42)
    pca_features = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_features, columns=[f"pca_{i}" for i in range(20)])

    print(
        f"  PCA explained variance ratio: {pca.explained_variance_ratio_[:5].sum():.3f} (first 5 components)"
    )

    # 2. Truncated SVD
    svd = TruncatedSVD(n_components=20, random_state=42)
    svd_features = svd.fit_transform(X_scaled)
    svd_df = pd.DataFrame(svd_features, columns=[f"svd_{i}" for i in range(20)])

    # 3. Factor Analysis
    fa = FactorAnalysis(n_components=15, random_state=42)
    fa_features = fa.fit_transform(X_scaled)
    fa_df = pd.DataFrame(fa_features, columns=[f"fa_{i}" for i in range(15)])

    # 4. Independent Component Analysis
    ica = FastICA(n_components=15, random_state=42)
    ica_features = ica.fit_transform(X_scaled)
    ica_df = pd.DataFrame(ica_features, columns=[f"ica_{i}" for i in range(15)])

    # 5. Clustering-based features
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    cluster_distances = kmeans.transform(X_scaled)

    cluster_df = pd.DataFrame(
        {
            "cluster_label": cluster_labels,
            **{f"cluster_dist_{i}": cluster_distances[:, i] for i in range(10)},
        }
    )

    # Combine all reduction features
    reduction_features = pd.concat([pca_df, svd_df, fa_df, ica_df, cluster_df], axis=1)

    print(
        f"Dimensionality reduction features created: {reduction_features.shape[1]} features"
    )
    return reduction_features


def create_automated_ml_pipeline(X, y):
    """Create automated ML pipeline with feature engineering"""
    print("Creating automated ML pipeline...")

    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "poly",
                            PolynomialFeatures(
                                degree=2, include_bias=False, interaction_only=True
                            ),
                        ),
                        ("selector", SelectKBest(f_classif, k=100)),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    # Model pipelines with different algorithms
    pipelines = {
        "rf_pipeline": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        ),
        "gb_pipeline": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "svm_pipeline": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", SVC(random_state=42, probability=True)),
            ]
        ),
        "lr_pipeline": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        ),
    }

    # Hyperparameter grids
    param_grids = {
        "rf_pipeline": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [10, 20, None],
            "preprocessor__num__selector__k": [50, 100, 150],
        },
        "gb_pipeline": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "preprocessor__num__selector__k": [50, 100],
        },
        "svm_pipeline": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["rbf", "linear"],
            "preprocessor__num__selector__k": [50, 100],
        },
        "lr_pipeline": {
            "classifier__C": [0.1, 1, 10],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear", "saga"],
            "preprocessor__num__selector__k": [50, 100],
        },
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and evaluate pipelines
    results = {}

    for name, pipeline in pipelines.items():
        print(f"  Training {name}...")

        # Randomized search for hyperparameter tuning
        search = RandomizedSearchCV(
            pipeline,
            param_grids[name],
            n_iter=20,
            cv=3,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
        )

        start_time = datetime.now()
        search.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        best_model = search.best_estimator_
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")

        results[name] = {
            "best_model": best_model,
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "training_time": training_time,
        }

        print(f"    Best CV F1: {search.best_score_:.3f}")
        print(f"    Test Accuracy: {test_acc:.3f}")
        print(f"    Test F1: {test_f1:.3f}")

    return results, X_test, y_test


def visualize_feature_engineering_results(
    selection_results, reduction_features, automl_results
):
    """Visualize feature engineering results"""
    print("Creating feature engineering visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Advanced Feature Engineering Results", fontsize=16, fontweight="bold")

    # 1. Feature selection comparison
    ax = axes[0, 0]
    methods = list(selection_results.keys())
    n_features = [result["n_features"] for result in selection_results.values()]

    bars = ax.bar(range(len(methods)), n_features, color="skyblue")
    ax.set_xlabel("Selection Method")
    ax.set_ylabel("Number of Features Selected")
    ax.set_title("Feature Selection Methods Comparison")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace("_", "\\n") for m in methods], rotation=45)

    # Add value labels
    for bar, n in zip(bars, n_features):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(n),
            ha="center",
            va="bottom",
        )

    # 2. Dimensionality reduction features distribution
    ax = axes[0, 1]
    reduction_types = ["PCA", "SVD", "FA", "ICA", "Cluster"]
    reduction_counts = [20, 20, 15, 15, 11]  # Based on our implementation

    ax.pie(reduction_counts, labels=reduction_types, autopct="%1.1f%%", startangle=90)
    ax.set_title("Dimensionality Reduction Features")

    # 3. AutoML pipeline performance
    ax = axes[1, 0]
    pipeline_names = [
        name.replace("_pipeline", "").upper() for name in automl_results.keys()
    ]
    test_f1_scores = [result["test_f1"] for result in automl_results.values()]

    bars = ax.barh(pipeline_names, test_f1_scores, color="lightgreen")
    ax.set_xlabel("Test F1-Score")
    ax.set_title("AutoML Pipeline Performance")
    ax.set_xlim(0, 1)

    # Add value labels
    for bar, score in zip(bars, test_f1_scores):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
        )

    # 4. Training time vs performance
    ax = axes[1, 1]
    training_times = [result["training_time"] for result in automl_results.values()]

    scatter = ax.scatter(
        training_times,
        test_f1_scores,
        c=range(len(pipeline_names)),
        cmap="viridis",
        s=100,
    )

    for i, name in enumerate(pipeline_names):
        ax.annotate(
            name,
            (training_times[i], test_f1_scores[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Test F1-Score")
    ax.set_title("Performance vs Training Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main function to run advanced feature engineering challenge"""
    print("=" * 70)
    print("LEVEL 5 CHALLENGE 3: ADVANCED FEATURE ENGINEERING & AUTOML")
    print("=" * 70)

    # Create dataset
    data = create_comprehensive_dataset()

    # Separate features and target
    X_basic = data.drop(["customer_value", "review_text"], axis=1)
    y = data["customer_value"].astype(int)

    print(f"\\nBasic features: {X_basic.shape}")

    # Advanced feature engineering
    print("\\n" + "=" * 50)
    print("ADVANCED FEATURE ENGINEERING")
    print("=" * 50)

    # Custom feature engineering
    feature_engineer = AdvancedFeatureEngineer(
        include_interactions=True, include_ratios=True, include_aggregates=True
    )

    X_engineered = feature_engineer.fit_transform(X_basic)
    print(f"After custom engineering: {X_engineered.shape}")

    # Text features
    text_features = create_text_features(data)

    # Categorical features
    categorical_features = create_categorical_features(data)

    # Combine all features
    X_combined = pd.concat(
        [
            X_engineered.reset_index(drop=True),
            text_features.reset_index(drop=True),
            categorical_features.reset_index(drop=True),
        ],
        axis=1,
    )

    print(f"Combined features: {X_combined.shape}")

    # Feature Selection
    print("\\n" + "=" * 50)
    print("FEATURE SELECTION")
    print("=" * 50)

    # Convert to numpy for feature selection
    X_numeric = X_combined.select_dtypes(include=[np.number]).fillna(0)
    feature_names = X_numeric.columns.tolist()

    selection_results = advanced_feature_selection(
        X_numeric.values, y.values, feature_names
    )

    # Dimensionality Reduction
    print("\\n" + "=" * 50)
    print("DIMENSIONALITY REDUCTION")
    print("=" * 50)

    reduction_features = create_dimensionality_reduction_features(
        X_numeric.values, feature_names
    )

    # Automated ML
    print("\\n" + "=" * 50)
    print("AUTOMATED ML PIPELINES")
    print("=" * 50)

    automl_results, X_test, y_test = create_automated_ml_pipeline(X_numeric, y)

    # Visualization
    print("\\n" + "=" * 50)
    print("RESULTS VISUALIZATION")
    print("=" * 50)

    visualize_feature_engineering_results(
        selection_results, reduction_features, automl_results
    )

    # Summary
    print("\\n" + "=" * 70)
    print("CHALLENGE 3 COMPLETION SUMMARY")
    print("=" * 70)

    best_pipeline = max(
        automl_results.keys(), key=lambda x: automl_results[x]["test_f1"]
    )
    best_performance = automl_results[best_pipeline]

    print(f"Best performing pipeline: {best_pipeline}")
    print(f"  - Test F1-Score: {best_performance['test_f1']:.3f}")
    print(f"  - Test Accuracy: {best_performance['test_accuracy']:.3f}")
    print(f"  - Best CV Score: {best_performance['best_cv_score']:.3f}")

    feature_engineering_techniques = [
        "Custom interaction features",
        "Ratio and proportion features",
        "Statistical aggregate features",
        "TF-IDF text vectorization",
        "N-gram text features",
        "Target encoding for categories",
        "Frequency encoding",
        "Variance threshold selection",
        "Chi-square feature selection",
        "Mutual information selection",
        "Model-based selection (Random Forest)",
        "Recursive feature elimination",
        "Principal Component Analysis (PCA)",
        "Truncated Singular Value Decomposition",
        "Factor Analysis",
        "Independent Component Analysis",
        "K-means clustering features",
        "Automated ML pipelines",
        "Hyperparameter optimization",
        "Cross-validation scoring",
    ]

    print("\\nFeature engineering techniques mastered:")
    for i, technique in enumerate(feature_engineering_techniques, 1):
        print(f"  {i}. {technique}")

    print(f"\\nDataset Statistics:")
    print(f"  - Original features: {X_basic.shape[1]}")
    print(f"  - Engineered features: {X_combined.shape[1]}")
    print(f"  - Samples processed: {len(data):,}")
    print(f"  - Text features created: {text_features.shape[1]}")
    print(f"  - Categorical encodings: {categorical_features.shape[1]}")
    print(f"  - Reduction features: {reduction_features.shape[1]}")

    return {
        "data": data,
        "X_combined": X_combined,
        "selection_results": selection_results,
        "reduction_features": reduction_features,
        "automl_results": automl_results,
    }


if __name__ == "__main__":
    results = main()

    print("\\n" + "=" * 70)
    print("CHALLENGE 3 STATUS: COMPLETE")
    print("=" * 70)
    print("Advanced feature engineering and automated ML mastery achieved!")
    print("Ready for Challenge 4: Production ML Systems.")
