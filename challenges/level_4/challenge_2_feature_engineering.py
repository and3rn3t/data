#!/usr/bin/env python3
"""
Level 4: Machine Learning Novice
Challenge 2: Feature Engineering Mastery

Master the art of feature engineering - transforming raw data into powerful predictive features.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore")


def create_customer_dataset(n_samples=1000, random_state=42):
    """Create a comprehensive customer dataset with various feature types"""
    rng = np.random.default_rng(random_state)

    print("📊 Creating Rich Dataset for Feature Engineering...")

    # Generate customer dataset with mixed feature types
    data = pd.DataFrame(
        {
            # Numerical features
            "age": np.clip(rng.normal(35, 12, n_samples), 18, 80),
            "income": rng.lognormal(10.5, 0.8, n_samples),
            "credit_score": np.clip(rng.normal(650, 100, n_samples), 300, 850),
            "years_employed": np.clip(rng.exponential(5, n_samples), 0, 40),
            "debt_to_income": rng.beta(2, 5, n_samples) * 0.8,
            # Categorical features
            "education": rng.choice(
                ["High School", "Bachelor", "Master", "PhD"],
                size=n_samples,
                p=[0.4, 0.35, 0.2, 0.05],
            ),
            "employment_type": rng.choice(
                ["Full-time", "Part-time", "Contract", "Self-employed"],
                size=n_samples,
                p=[0.6, 0.15, 0.15, 0.1],
            ),
            "marital_status": rng.choice(
                ["Single", "Married", "Divorced"], size=n_samples, p=[0.4, 0.45, 0.15]
            ),
            # Transaction features
            "monthly_transactions": rng.poisson(25, n_samples),
            "avg_transaction_amount": rng.gamma(2, 50, n_samples),
            "account_balance": rng.normal(5000, 3000, n_samples),
        }
    )

    # Add some missing values to simulate real data
    missing_indices = rng.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    data.loc[missing_indices, "credit_score"] = np.nan

    missing_indices = rng.choice(n_samples, size=int(0.03 * n_samples), replace=False)
    data.loc[missing_indices, "years_employed"] = np.nan

    # Create target variable based on complex relationships
    risk_score = (
        -0.01 * data["age"]
        + -0.0001 * data["income"]
        + -0.005 * data["credit_score"].fillna(data["credit_score"].mean())
        + -0.02 * data["years_employed"].fillna(data["years_employed"].mean())
        + 2.0 * data["debt_to_income"]
        + 0.01 * data["monthly_transactions"]
        + rng.normal(0, 0.3, n_samples)
    )

    data["default_risk"] = (risk_score > risk_score.median()).astype(int)

    return data


def handle_missing_values(data):
    """Demonstrate different missing value handling strategies"""
    print("\n=== MISSING VALUE HANDLING ===")

    # Analyze missing values
    missing_summary = data.isnull().sum()
    print("Missing values by column:")
    print(missing_summary[missing_summary > 0])

    # Strategy 1: Simple imputation
    print("\n1. Simple Imputation (Mean/Mode)")
    simple_imputer_num = SimpleImputer(strategy="mean")
    simple_imputer_cat = SimpleImputer(strategy="most_frequent")

    data_simple = data.copy()
    numeric_cols = data_simple.select_dtypes(include=[np.number]).columns
    categorical_cols = data_simple.select_dtypes(include=["object"]).columns

    data_simple[numeric_cols] = simple_imputer_num.fit_transform(
        data_simple[numeric_cols]
    )
    data_simple[categorical_cols] = simple_imputer_cat.fit_transform(
        data_simple[categorical_cols]
    )

    print(f"Missing values after simple imputation: {data_simple.isnull().sum().sum()}")

    # Strategy 2: KNN imputation (numeric only for demo)
    print("\n2. KNN Imputation (Numeric features)")
    knn_imputer = KNNImputer(n_neighbors=5)
    data_knn = data.copy()
    data_knn[numeric_cols] = knn_imputer.fit_transform(data_knn[numeric_cols])

    # Fill categorical with mode for KNN version
    for col in categorical_cols:
        data_knn[col] = data_knn[col].fillna(data_knn[col].mode()[0])

    print(f"Missing values after KNN imputation: {data_knn.isnull().sum().sum()}")

    return data_simple, data_knn


def encode_categorical_features(data):
    """Demonstrate different categorical encoding techniques"""
    print("\n=== CATEGORICAL ENCODING ===")

    categorical_cols = ["education", "employment_type", "marital_status"]
    encoded_data = data.copy()

    # 1. Label Encoding
    print("1. Label Encoding")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        encoded_data[f"{col}_label"] = le.fit_transform(encoded_data[col])
        label_encoders[col] = le
        print(f"   {col}: {len(le.classes_)} unique values")

    # 2. One-Hot Encoding
    print("\n2. One-Hot Encoding")
    # Use pandas get_dummies for simplicity
    one_hot_encoded = pd.get_dummies(data[categorical_cols], prefix=categorical_cols)
    encoded_data = pd.concat([encoded_data, one_hot_encoded], axis=1)

    print(f"   Original categorical columns: {len(categorical_cols)}")
    print(f"   One-hot encoded columns: {one_hot_encoded.shape[1]}")

    # 3. Ordinal Encoding for Education (natural order)
    education_order = ["High School", "Bachelor", "Master", "PhD"]
    encoded_data["education_ordinal"] = encoded_data["education"].map(
        {edu: i for i, edu in enumerate(education_order)}
    )

    print(
        "   Education ordinal mapping:",
        {edu: i for i, edu in enumerate(education_order)},
    )

    return encoded_data, label_encoders


def create_new_features(data):
    """Create new features through feature engineering"""
    print("\n=== FEATURE CREATION ===")

    engineered_data = data.copy()

    # 1. Mathematical transformations
    print("1. Mathematical Transformations")
    engineered_data["log_income"] = np.log1p(engineered_data["income"])
    engineered_data["sqrt_age"] = np.sqrt(engineered_data["age"])
    engineered_data["credit_score_squared"] = engineered_data["credit_score"] ** 2

    # 2. Ratio features
    print("2. Ratio and Interaction Features")
    engineered_data["income_per_year_employed"] = engineered_data["income"] / (
        engineered_data["years_employed"] + 1
    )
    engineered_data["transaction_frequency_ratio"] = (
        engineered_data["monthly_transactions"] / 30  # per day
    )
    engineered_data["balance_to_income_ratio"] = (
        engineered_data["account_balance"] / engineered_data["income"]
    )

    # 3. Binning continuous variables
    print("3. Binning Continuous Variables")
    engineered_data["age_group"] = pd.cut(
        engineered_data["age"],
        bins=[0, 25, 35, 50, 100],
        labels=["Young", "Adult", "Middle-aged", "Senior"],
    )

    engineered_data["income_tier"] = pd.qcut(
        engineered_data["income"], q=4, labels=["Low", "Medium", "High", "Premium"]
    )

    # 4. Interaction features
    print("4. Advanced Interaction Features")
    # Age and income interaction
    engineered_data["age_income_interaction"] = (
        engineered_data["age"] * engineered_data["log_income"]
    )

    # Credit utilization proxy
    engineered_data["credit_utilization"] = (
        engineered_data["debt_to_income"]
        * engineered_data["income"]
        / (engineered_data["credit_score"] / 100)
    )

    print(f"   Original features: {data.shape[1]}")
    print(f"   After feature engineering: {engineered_data.shape[1]}")
    print(f"   New features created: {engineered_data.shape[1] - data.shape[1]}")

    return engineered_data


def scale_features(data, target_col):
    """Demonstrate different scaling techniques"""
    print("\n=== FEATURE SCALING ===")

    # Select numeric features for scaling
    numeric_features = data.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != target_col]

    scaling_data = data[numeric_features].copy()

    # 1. Standard Scaling (Z-score normalization)
    print("1. Standard Scaling (Mean=0, Std=1)")
    scaler_standard = StandardScaler()
    scaled_standard = pd.DataFrame(
        scaler_standard.fit_transform(scaling_data),
        columns=scaling_data.columns,
        index=scaling_data.index,
    )

    # 2. Min-Max Scaling
    print("2. Min-Max Scaling (Range 0-1)")
    scaler_minmax = MinMaxScaler()
    scaled_minmax = pd.DataFrame(
        scaler_minmax.fit_transform(scaling_data),
        columns=scaling_data.columns,
        index=scaling_data.index,
    )

    # 3. Robust Scaling (using median and IQR)
    print("3. Robust Scaling (Median and IQR)")
    scaler_robust = RobustScaler()
    scaled_robust = pd.DataFrame(
        scaler_robust.fit_transform(scaling_data),
        columns=scaling_data.columns,
        index=scaling_data.index,
    )

    # Compare scaling effects
    comparison_data = {
        "Original": scaling_data.describe().loc[["mean", "std", "min", "max"]],
        "Standard": scaled_standard.describe().loc[["mean", "std", "min", "max"]],
        "MinMax": scaled_minmax.describe().loc[["mean", "std", "min", "max"]],
        "Robust": scaled_robust.describe().loc[["mean", "std", "min", "max"]],
    }

    print("\\nScaling comparison (first 3 features):")
    for scale_type, stats in comparison_data.items():
        print(f"\\n{scale_type}:")
        print(stats.iloc[:, :3].round(3))

    return {
        "standard": scaled_standard,
        "minmax": scaled_minmax,
        "robust": scaled_robust,
        "scalers": {
            "standard": scaler_standard,
            "minmax": scaler_minmax,
            "robust": scaler_robust,
        },
    }


def select_features(X, y, feature_names):
    """Demonstrate feature selection techniques"""
    print("\n=== FEATURE SELECTION ===")

    # 1. Univariate Feature Selection
    print("1. Univariate Feature Selection (SelectKBest)")
    selector_univariate = SelectKBest(score_func=f_classif, k=10)
    X_selected_univariate = selector_univariate.fit_transform(X, y)

    selected_features_univariate = np.array(feature_names)[
        selector_univariate.get_support()
    ]
    scores = selector_univariate.scores_[selector_univariate.get_support()]

    print(f"   Selected {len(selected_features_univariate)} features:")
    for feature, score in zip(selected_features_univariate[:5], scores[:5]):
        print(f"   - {feature}: {score:.2f}")

    # 2. L1-based Feature Selection
    print("\\n2. L1-based Feature Selection (Lasso)")
    lasso = Lasso(alpha=0.01, random_state=42)
    selector_l1 = SelectFromModel(lasso)
    X_selected_l1 = selector_l1.fit_transform(X, y)

    selected_features_l1 = np.array(feature_names)[selector_l1.get_support()]
    print(f"   Selected {len(selected_features_l1)} features with L1 regularization")

    # 3. Tree-based Feature Selection
    print("\\n3. Tree-based Feature Selection")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector_tree = SelectFromModel(rf)
    X_selected_tree = selector_tree.fit_transform(X, y)

    selected_features_tree = np.array(feature_names)[selector_tree.get_support()]

    # Get feature importances
    rf.fit(X, y)
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"   Selected {len(selected_features_tree)} features using Random Forest")
    print("   Top 5 most important features:")
    for _, row in feature_importance.head().iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")

    return {
        "univariate": (X_selected_univariate, selected_features_univariate),
        "l1": (X_selected_l1, selected_features_l1),
        "tree": (X_selected_tree, selected_features_tree),
        "feature_importance": feature_importance,
    }


def create_polynomial_features(X, feature_names, degree=2):
    """Create polynomial features"""
    print(f"\n=== POLYNOMIAL FEATURES (Degree {degree}) ===")

    # Use only first 5 features to avoid explosion
    X_subset = X[:, :5]
    subset_names = feature_names[:5]

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_subset)

    poly_feature_names = poly.get_feature_names_out(subset_names)

    print(f"Original features: {X_subset.shape[1]}")
    print(f"Polynomial features: {X_poly.shape[1]}")
    print(f"Example new features: {list(poly_feature_names[-5:])}")

    return X_poly, poly_feature_names


def dimensionality_reduction_demo(X, feature_names, n_components=10):
    """Demonstrate dimensionality reduction with PCA"""
    print(f"\n=== DIMENSIONALITY REDUCTION (PCA) ===")

    # Standardize features first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    print(f"Original dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: {X_pca.shape[1]}")
    print(
        f"Variance explained by {n_components} components: {cumulative_variance[-1]:.3f}"
    )

    # Show explained variance by component
    print("\\nExplained variance by component:")
    for i, variance in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"   PC{i+1}: {variance:.3f}")

    # Visualize explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, "bo-")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA: Cumulative Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return X_pca, pca


def compare_feature_engineering_impact(original_data, engineered_data, target_col):
    """Compare model performance with and without feature engineering"""
    print("\n=== FEATURE ENGINEERING IMPACT ANALYSIS ===")

    # Prepare original data
    X_original = original_data.select_dtypes(include=[np.number]).drop(
        columns=[target_col]
    )
    y = original_data[target_col]

    # Prepare engineered data (numeric only)
    X_engineered = engineered_data.select_dtypes(include=[np.number]).drop(
        columns=[target_col]
    )

    # Handle any remaining missing values
    X_original = X_original.fillna(X_original.mean())
    X_engineered = X_engineered.fillna(X_engineered.mean())

    # Train-test split
    (X_orig_train, X_orig_test, X_eng_train, X_eng_test, y_train, y_test) = (
        train_test_split(
            X_original, X_engineered, y, test_size=0.3, random_state=42, stratify=y
        )
    )

    # Test multiple models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\\n{model_name}:")

        # Original features
        model_orig = model.__class__(**model.get_params())
        model_orig.fit(X_orig_train, y_train)
        orig_accuracy = accuracy_score(y_test, model_orig.predict(X_orig_test))

        # Engineered features
        model_eng = model.__class__(**model.get_params())
        model_eng.fit(X_eng_train, y_train)
        eng_accuracy = accuracy_score(y_test, model_eng.predict(X_eng_test))

        improvement = ((eng_accuracy - orig_accuracy) / orig_accuracy) * 100

        results[model_name] = {
            "original_accuracy": orig_accuracy,
            "engineered_accuracy": eng_accuracy,
            "improvement_pct": improvement,
        }

        print(f"   Original features ({X_original.shape[1]}): {orig_accuracy:.4f}")
        print(f"   Engineered features ({X_engineered.shape[1]}): {eng_accuracy:.4f}")
        print(f"   Improvement: {improvement:+.2f}%")

    return results


def main():
    """Main function to run the feature engineering challenge"""
    print("=" * 60)
    print("LEVEL 4 CHALLENGE 2: FEATURE ENGINEERING MASTERY")
    print("=" * 60)

    print("🔧 Welcome to Feature Engineering!")
    print("Learn to transform raw data into powerful predictive features.")

    # 1. Create comprehensive dataset
    data = create_customer_dataset()
    print(f"\\nDataset created: {data.shape[0]} samples, {data.shape[1]} features")
    print("\\nDataset overview:")
    print(data.head())

    # 2. Handle missing values
    data_simple, data_knn = handle_missing_values(data)

    # 3. Encode categorical features
    encoded_data, label_encoders = encode_categorical_features(data_simple)

    # 4. Create new features
    engineered_data = create_new_features(encoded_data)

    # 5. Scale features
    scaling_results = scale_features(engineered_data, "default_risk")

    # 6. Feature selection
    X = engineered_data.select_dtypes(include=[np.number]).drop(
        columns=["default_risk"]
    )
    y = engineered_data["default_risk"]
    X = X.fillna(X.mean())  # Handle any remaining missing values

    selection_results = select_features(X.values, y.values, X.columns.tolist())

    # 7. Polynomial features
    X_poly, poly_names = create_polynomial_features(X.values, X.columns.tolist())

    # 8. Dimensionality reduction
    X_pca, pca_model = dimensionality_reduction_demo(X.values, X.columns.tolist())

    # 9. Compare impact
    impact_results = compare_feature_engineering_impact(
        data_simple, engineered_data, "default_risk"
    )

    # Summary
    print("\\n" + "=" * 60)
    print("CHALLENGE 2 COMPLETION SUMMARY")
    print("=" * 60)

    print("Feature Engineering techniques mastered:")
    techniques = [
        "🔍 Missing value handling (Simple & KNN imputation)",
        "🏷️ Categorical encoding (Label, One-hot, Ordinal)",
        "⚙️ Feature creation (Mathematical, Ratios, Binning, Interactions)",
        "📏 Feature scaling (Standard, Min-Max, Robust)",
        "🎯 Feature selection (Univariate, L1-based, Tree-based)",
        "📐 Polynomial feature generation",
        "📊 Dimensionality reduction (PCA)",
        "📈 Impact analysis and model performance comparison",
    ]

    for technique in techniques:
        print(f"  {technique}")

    print(f"\\nDataset transformation:")
    print(f"  • Original features: {data.shape[1]}")
    print(f"  • After engineering: {engineered_data.shape[1]}")
    print(
        f"  • Feature improvement: {(engineered_data.shape[1] - data.shape[1])} new features"
    )

    print(f"\\nModel performance improvement:")
    for model, results in impact_results.items():
        print(f"  • {model}: {results['improvement_pct']:+.2f}% improvement")

    print("\\n🎉 Congratulations! You've mastered feature engineering!")
    print("You're ready for Challenge 3: Model Evaluation")

    return {
        "original_data": data,
        "engineered_data": engineered_data,
        "scaling_results": scaling_results,
        "selection_results": selection_results,
        "impact_results": impact_results,
    }


if __name__ == "__main__":
    results = main()

    print("\\n" + "=" * 60)
    print("CHALLENGE 2 STATUS: COMPLETE")
    print("=" * 60)
    print("Feature engineering mastery achieved!")
    print("Ready for Challenge 3: Model Evaluation.")
