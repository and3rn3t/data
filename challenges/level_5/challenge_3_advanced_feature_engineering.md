# Level 5: Algorithm Architect

## Challenge 3: Advanced Feature Engineering and Automated ML

Master sophisticated feature engineering techniques and automated machine learning pipelines to build production-ready ML systems.

### Objective

Learn advanced feature engineering, automated feature selection, and AutoML techniques to build scalable and robust machine learning pipelines.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold,
                                   GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                 PolynomialFeatures, QuantileTransformer, PowerTransformer)
from sklearn.feature_selection import (SelectKBest, SelectPercentile, RFE, RFECV,
                                     chi2, f_classif, mutual_info_classif,
                                     VarianceThreshold, SelectFromModel)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                            GradientBoostingClassifier, IsolationForest)
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("🔬 Advanced Feature Engineering and Automated ML")
print("=" * 50)

# Create comprehensive multi-modal dataset
np.random.seed(42)

print("📊 Creating Complex Multi-Modal Dataset...")

# Simulate e-commerce customer dataset with multiple data types
n_customers = 3000

# Demographic features
demographic_data = pd.DataFrame({
    'age': np.random.gamma(2, 20, n_customers).clip(18, 80),
    'income': np.random.lognormal(10.5, 0.8, n_customers),
    'education_level': np.random.choice(['high_school', 'bachelor', 'master', 'phd'],
                                      size=n_customers, p=[0.3, 0.4, 0.25, 0.05]),
    'marital_status': np.random.choice(['single', 'married', 'divorced'],
                                     size=n_customers, p=[0.35, 0.55, 0.1]),
    'location': np.random.choice(['urban', 'suburban', 'rural'],
                               size=n_customers, p=[0.5, 0.35, 0.15])
})

# Behavioral features
behavioral_data = pd.DataFrame({
    'website_visits_per_month': np.random.negative_binomial(15, 0.3, n_customers),
    'session_duration_avg': np.random.lognormal(3, 0.5, n_customers),
    'pages_per_session': np.random.gamma(2, 3, n_customers).clip(1, 50),
    'bounce_rate': np.random.beta(2, 5, n_customers),
    'time_on_site_total': np.random.exponential(100, n_customers).clip(1, 500),
    'device_type': np.random.choice(['mobile', 'desktop', 'tablet'],
                                  size=n_customers, p=[0.6, 0.3, 0.1]),
    'browser': np.random.choice(['chrome', 'firefox', 'safari', 'edge'],
                              size=n_customers, p=[0.6, 0.2, 0.15, 0.05])
})

# Transaction features
transaction_data = pd.DataFrame({
    'purchase_frequency': np.random.gamma(1.5, 2, n_customers),
    'avg_order_value': np.random.lognormal(4.2, 0.6, n_customers),
    'total_spent': np.random.lognormal(6, 1, n_customers),
    'discount_usage_rate': np.random.beta(3, 7, n_customers),
    'return_rate': np.random.beta(2, 8, n_customers),
    'payment_method': np.random.choice(['credit', 'debit', 'paypal', 'crypto'],
                                     size=n_customers, p=[0.5, 0.3, 0.15, 0.05]),
    'preferred_category': np.random.choice(['electronics', 'clothing', 'books', 'home', 'sports'],
                                        size=n_customers, p=[0.25, 0.3, 0.15, 0.2, 0.1])
})

# Engagement features
engagement_data = pd.DataFrame({
    'email_open_rate': np.random.beta(4, 6, n_customers),
    'click_through_rate': np.random.beta(2, 8, n_customers),
    'social_media_follows': np.random.poisson(3, n_customers),
    'review_count': np.random.poisson(2, n_customers),
    'referral_count': np.random.poisson(1, n_customers),
    'loyalty_program_level': np.random.choice(['bronze', 'silver', 'gold', 'platinum'],
                                           size=n_customers, p=[0.4, 0.3, 0.2, 0.1]),
    'customer_service_tickets': np.random.poisson(1.5, n_customers)
})

# Temporal features (simulated time-based data)
temporal_data = pd.DataFrame({
    'account_age_months': np.random.exponential(24, n_customers).clip(1, 120),
    'last_purchase_days_ago': np.random.exponential(30, n_customers).clip(0, 365),
    'seasonal_activity': np.random.choice(['spring', 'summer', 'fall', 'winter'],
                                        size=n_customers, p=[0.25, 0.25, 0.25, 0.25]),
    'weekday_activity': np.random.uniform(0, 1, n_customers),  # Activity score for weekdays
    'weekend_activity': np.random.uniform(0, 1, n_customers)   # Activity score for weekends
})

# Text features (simulated product reviews)
def generate_review_text():
    """Generate synthetic review text with sentiment"""
    positive_words = ['excellent', 'great', 'amazing', 'perfect', 'wonderful', 'fantastic', 'love']
    negative_words = ['terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'bad']
    neutral_words = ['okay', 'average', 'decent', 'normal', 'standard', 'typical', 'regular']

    reviews = []
    sentiments = []

    for i in range(n_customers):
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.5, 0.3, 0.2])

        if sentiment == 'positive':
            words = np.random.choice(positive_words, size=np.random.randint(3, 8))
            sentiments.append(1)
        elif sentiment == 'negative':
            words = np.random.choice(negative_words, size=np.random.randint(3, 8))
            sentiments.append(0)
        else:
            words = np.random.choice(neutral_words, size=np.random.randint(3, 6))
            sentiments.append(2)

        review = ' '.join(words) + ' product quality service'
        reviews.append(review)

    return reviews, sentiments

reviews, review_sentiments = generate_review_text()
text_data = pd.DataFrame({
    'review_text': reviews,
    'review_sentiment': review_sentiments
})

# Combine all data
customer_data = pd.concat([demographic_data, behavioral_data, transaction_data,
                          engagement_data, temporal_data, text_data], axis=1)

# Create complex target variable (customer lifetime value category)
# Use multiple features to create realistic CLV segments
clv_score = (
    (customer_data['total_spent'] / 1000) * 0.3 +
    (customer_data['purchase_frequency'] * 10) * 0.2 +
    (customer_data['avg_order_value'] / 100) * 0.2 +
    (customer_data['account_age_months'] / 12) * 0.1 +
    (customer_data['email_open_rate'] * 100) * 0.1 +
    (customer_data['loyalty_program_level'].map({'bronze': 1, 'silver': 2, 'gold': 3, 'platinum': 4})) * 0.1
)

# Add some noise and create categories
clv_score += np.random.normal(0, 1, n_customers)
customer_data['clv_category'] = pd.cut(clv_score,
                                     bins=[-np.inf, 2, 5, 8, np.inf],
                                     labels=['Low', 'Medium', 'High', 'VIP']).astype(str)

print(f"Dataset shape: {customer_data.shape}")
print(f"CLV Category distribution:")
print(customer_data['clv_category'].value_counts())

# CHALLENGE 1: ADVANCED FEATURE TRANSFORMATION
print("\n" + "=" * 60)
print("🔄 CHALLENGE 1: ADVANCED FEATURE TRANSFORMATION")
print("=" * 60)

# Separate features by type
numeric_features = ['age', 'income', 'website_visits_per_month', 'session_duration_avg',
                   'pages_per_session', 'bounce_rate', 'time_on_site_total',
                   'purchase_frequency', 'avg_order_value', 'total_spent',
                   'discount_usage_rate', 'return_rate', 'email_open_rate',
                   'click_through_rate', 'social_media_follows', 'review_count',
                   'referral_count', 'customer_service_tickets', 'account_age_months',
                   'last_purchase_days_ago', 'weekday_activity', 'weekend_activity']

categorical_features = ['education_level', 'marital_status', 'location', 'device_type',
                       'browser', 'payment_method', 'preferred_category',
                       'loyalty_program_level', 'seasonal_activity']

text_features = ['review_text']

print("🧪 Custom Feature Transformations")

class AdvancedFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for advanced feature engineering"""

    def __init__(self, create_interactions=True, create_ratios=True,
                 create_binned=True, create_temporal=True):
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_binned = create_binned
        self.create_temporal = create_temporal
        self.feature_names_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()

        # Interaction features
        if self.create_interactions:
            # Behavioral interactions
            X_new['engagement_score'] = (X['email_open_rate'] * X['click_through_rate'] *
                                       X['social_media_follows'])
            X_new['activity_intensity'] = X['website_visits_per_month'] * X['session_duration_avg']
            X_new['purchase_efficiency'] = X['purchase_frequency'] / (X['website_visits_per_month'] + 1)

            # Value interactions
            X_new['value_per_visit'] = X['total_spent'] / (X['website_visits_per_month'] + 1)
            X_new['loyalty_value'] = X['avg_order_value'] * X['account_age_months']

        # Ratio features
        if self.create_ratios:
            X_new['income_to_spending_ratio'] = X['income'] / (X['total_spent'] + 1)
            X_new['engagement_to_purchase_ratio'] = X['email_open_rate'] / (X['purchase_frequency'] + 0.1)
            X_new['activity_balance'] = X['weekday_activity'] / (X['weekend_activity'] + 0.1)
            X_new['efficiency_ratio'] = X['purchase_frequency'] / (X['return_rate'] + 0.01)

        # Binned features
        if self.create_binned:
            X_new['age_group'] = pd.cut(X['age'], bins=[0, 25, 35, 50, 65, 100],
                                      labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
            X_new['income_tier'] = pd.qcut(X['income'], q=5,
                                         labels=['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High'])
            X_new['spending_tier'] = pd.qcut(X['total_spent'], q=4,
                                           labels=['Light', 'Moderate', 'Heavy', 'VIP'])

        # Temporal features
        if self.create_temporal:
            X_new['recency_score'] = 1 / (X['last_purchase_days_ago'] + 1)
            X_new['tenure_category'] = pd.cut(X['account_age_months'],
                                            bins=[0, 6, 12, 24, 60, 200],
                                            labels=['New', 'Growing', 'Established', 'Mature', 'Veteran'])
            X_new['activity_trend'] = (X['weekday_activity'] + X['weekend_activity']) / 2

        return X_new

# Apply advanced transformations
print("Applying custom feature transformations...")
advanced_transformer = AdvancedFeatureTransformer()
customer_transformed = advanced_transformer.fit_transform(customer_data[numeric_features])

print(f"Original features: {len(numeric_features)}")
print(f"After transformation: {customer_transformed.shape[1]}")
print("New features created:", [col for col in customer_transformed.columns if col not in numeric_features])

# CHALLENGE 2: AUTOMATED FEATURE SELECTION
print("\n" + "=" * 60)
print("🎯 CHALLENGE 2: AUTOMATED FEATURE SELECTION")
print("=" * 60)

# Prepare complete dataset for feature selection
X_complete = customer_data.drop(['clv_category', 'review_text'], axis=1)
y_complete = customer_data['clv_category']

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoders = {}
X_encoded = X_complete.copy()

for cat_col in categorical_features:
    if cat_col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[cat_col] = le.fit_transform(X_encoded[cat_col].astype(str))
        label_encoders[cat_col] = le

# Handle target encoding
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_complete)

print("🔍 Multiple Feature Selection Techniques")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Variance Threshold
print("\n1. Variance Threshold Feature Selection")
variance_selector = VarianceThreshold(threshold=0.01)
X_train_variance = variance_selector.fit_transform(X_train_scaled)
selected_variance_features = X_train.columns[variance_selector.get_support()]
print(f"Features after variance threshold: {X_train_variance.shape[1]}/{X_train.shape[1]}")
print(f"Removed low-variance features: {X_train.shape[1] - X_train_variance.shape[1]}")

# 2. Univariate Statistical Tests
print("\n2. Univariate Statistical Feature Selection")
# For classification, use f_classif
univariate_selector = SelectKBest(score_func=f_classif, k=15)
X_train_univariate = univariate_selector.fit_transform(X_train_scaled, y_train)
selected_univariate_features = X_train.columns[univariate_selector.get_support()]

# Get feature scores
feature_scores = pd.DataFrame({
    'feature': X_train.columns,
    'score': univariate_selector.scores_,
    'p_value': univariate_selector.pvalues_
}).sort_values('score', ascending=False)

print(f"Top 5 features by F-score:")
for idx, (_, row) in enumerate(feature_scores.head().iterrows()):
    print(f"  {idx+1}. {row['feature']}: {row['score']:.2f} (p={row['p_value']:.6f})")

# 3. Mutual Information
print("\n3. Mutual Information Feature Selection")
mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k=15)
X_train_mutual_info = mutual_info_selector.fit_transform(X_train_scaled, y_train)
selected_mutual_info_features = X_train.columns[mutual_info_selector.get_support()]

mutual_info_scores = pd.DataFrame({
    'feature': X_train.columns,
    'mutual_info_score': mutual_info_selector.scores_
}).sort_values('mutual_info_score', ascending=False)

print(f"Top 5 features by Mutual Information:")
for idx, (_, row) in enumerate(mutual_info_scores.head().iterrows()):
    print(f"  {idx+1}. {row['feature']}: {row['mutual_info_score']:.4f}")

# 4. Recursive Feature Elimination (RFE)
print("\n4. Recursive Feature Elimination")
rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=15)
X_train_rfe = rfe_selector.fit_transform(X_train_scaled, y_train)
selected_rfe_features = X_train.columns[rfe_selector.get_support()]

print(f"Features selected by RFE: {X_train_rfe.shape[1]}")
print("RFE selected features:", list(selected_rfe_features[:5]))

# 5. Recursive Feature Elimination with Cross-Validation
print("\n5. RFE with Cross-Validation")
rfecv_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
rfecv_selector = RFECV(estimator=rfecv_estimator, step=1, cv=3, scoring='f1_weighted')
X_train_rfecv = rfecv_selector.fit_transform(X_train_scaled, y_train)
selected_rfecv_features = X_train.columns[rfecv_selector.get_support()]

print(f"Optimal number of features: {rfecv_selector.n_features_}")
print(f"Best cross-validation score: {rfecv_selector.grid_scores_.max():.4f}")

# 6. Model-based Feature Selection
print("\n6. Model-based Feature Selection")
# Using different models for feature importance
models_for_selection = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'Lasso': Lasso(alpha=0.01, random_state=42)
}

model_based_results = {}

for model_name, model in models_for_selection.items():
    if model_name == 'Lasso':
        # For Lasso, we need to handle multiclass
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(model)

    model_selector = SelectFromModel(model, threshold='median')

    try:
        X_train_model = model_selector.fit_transform(X_train_scaled, y_train)
        selected_model_features = X_train.columns[model_selector.get_support()]

        model_based_results[model_name] = {
            'n_features': X_train_model.shape[1],
            'features': list(selected_model_features)
        }

        print(f"  {model_name}: {X_train_model.shape[1]} features selected")
    except Exception as e:
        print(f"  {model_name}: Error - {str(e)}")

# Compare feature selection methods
print("\n📊 Feature Selection Comparison")
all_selection_results = {
    'Univariate (F-test)': list(selected_univariate_features),
    'Mutual Information': list(selected_mutual_info_features),
    'RFE': list(selected_rfe_features),
    'RFECV': list(selected_rfecv_features)
}

# Add model-based results
for model_name, result in model_based_results.items():
    all_selection_results[f'Model-based ({model_name})'] = result['features']

# Find feature overlap
feature_overlap = {}
all_methods = list(all_selection_results.keys())

for method1, method2 in combinations(all_methods[:4], 2):  # Compare first 4 methods
    features1 = set(all_selection_results[method1])
    features2 = set(all_selection_results[method2])
    overlap = len(features1.intersection(features2))
    feature_overlap[f"{method1} ∩ {method2}"] = overlap

print("Feature selection overlap:")
for comparison, overlap_count in feature_overlap.items():
    print(f"  {comparison}: {overlap_count} common features")

# CHALLENGE 3: DIMENSIONALITY REDUCTION TECHNIQUES
print("\n" + "=" * 60)
print("📉 CHALLENGE 3: DIMENSIONALITY REDUCTION")
print("=" * 60)

print("🔄 Multiple Dimensionality Reduction Techniques")

# Apply various dimensionality reduction methods
dimensionality_results = {}

# 1. Principal Component Analysis (PCA)
print("\n1. Principal Component Analysis")
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original dimensions: {X_train_scaled.shape[1]}")
print(f"PCA dimensions: {X_train_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

dimensionality_results['PCA'] = {
    'n_components': X_train_pca.shape[1],
    'explained_variance': pca.explained_variance_ratio_.sum(),
    'X_train': X_train_pca,
    'X_test': X_test_pca
}

# 2. Truncated SVD (for sparse matrices)
print("\n2. Truncated SVD")
svd = TruncatedSVD(n_components=15, random_state=42)
X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd = svd.transform(X_test_scaled)

print(f"SVD dimensions: {X_train_svd.shape[1]}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

dimensionality_results['SVD'] = {
    'n_components': X_train_svd.shape[1],
    'explained_variance': svd.explained_variance_ratio_.sum(),
    'X_train': X_train_svd,
    'X_test': X_test_svd
}

# 3. Independent Component Analysis (ICA)
print("\n3. Independent Component Analysis")
ica = FastICA(n_components=15, random_state=42, max_iter=1000)
try:
    X_train_ica = ica.fit_transform(X_train_scaled)
    X_test_ica = ica.transform(X_test_scaled)

    dimensionality_results['ICA'] = {
        'n_components': X_train_ica.shape[1],
        'X_train': X_train_ica,
        'X_test': X_test_ica
    }
    print(f"ICA dimensions: {X_train_ica.shape[1]}")
except:
    print("ICA failed to converge, skipping...")

# 4. Factor Analysis
print("\n4. Factor Analysis")
fa = FactorAnalysis(n_components=15, random_state=42, max_iter=1000)
try:
    X_train_fa = fa.fit_transform(X_train_scaled)
    X_test_fa = fa.transform(X_test_scaled)

    dimensionality_results['FactorAnalysis'] = {
        'n_components': X_train_fa.shape[1],
        'X_train': X_train_fa,
        'X_test': X_test_fa
    }
    print(f"Factor Analysis dimensions: {X_train_fa.shape[1]}")
except:
    print("Factor Analysis failed, skipping...")

# 5. t-SNE (for visualization only, not for training)
print("\n5. t-SNE (for visualization)")
# Use a subset for t-SNE due to computational cost
subset_size = min(500, X_train_scaled.shape[0])
X_subset = X_train_scaled[:subset_size]
y_subset = y_train[:subset_size]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_train_tsne = tsne.fit_transform(X_subset)

print(f"t-SNE dimensions: {X_train_tsne.shape[1]} (visualization only)")

# CHALLENGE 4: AUTOMATED PIPELINE CONSTRUCTION
print("\n" + "=" * 60)
print("🏭 CHALLENGE 4: AUTOMATED PIPELINE CONSTRUCTION")
print("=" * 60)

print("🔧 Building Automated ML Pipelines")

class AutoMLPipeline:
    """Automated ML pipeline with feature selection and model selection"""

    def __init__(self):
        self.best_pipeline = None
        self.best_score = 0
        self.results = {}

    def create_pipelines(self):
        """Create multiple pipeline configurations"""

        # Feature selection methods
        feature_selectors = {
            'univariate': SelectKBest(f_classif, k=15),
            'rfe': RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=15),
            'model_based': SelectFromModel(ExtraTreesClassifier(n_estimators=50, random_state=42))
        }

        # Dimensionality reduction methods
        dim_reducers = {
            'pca': PCA(n_components=0.95),
            'none': 'passthrough'
        }

        # Classifiers
        classifiers = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }

        pipelines = {}

        # Create combinations of preprocessing and models
        for fs_name, fs in feature_selectors.items():
            for dr_name, dr in dim_reducers.items():
                for clf_name, clf in classifiers.items():

                    pipeline_name = f"{fs_name}_{dr_name}_{clf_name}"

                    if dr_name == 'none':
                        pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('feature_selector', fs),
                            ('classifier', clf)
                        ])
                    else:
                        pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('feature_selector', fs),
                            ('dim_reducer', dr),
                            ('classifier', clf)
                        ])

                    pipelines[pipeline_name] = pipeline

        return pipelines

    def evaluate_pipelines(self, X_train, y_train, cv=3):
        """Evaluate all pipeline configurations"""

        pipelines = self.create_pipelines()

        for name, pipeline in pipelines.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(pipeline, X_train, y_train,
                                          cv=cv, scoring='f1_weighted', n_jobs=-1)

                mean_score = cv_scores.mean()
                std_score = cv_scores.std()

                self.results[name] = {
                    'pipeline': pipeline,
                    'cv_mean': mean_score,
                    'cv_std': std_score,
                    'cv_scores': cv_scores
                }

                # Track best pipeline
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_pipeline = pipeline

                print(f"  {name}: {mean_score:.4f} ± {std_score:.4f}")

            except Exception as e:
                print(f"  {name}: Error - {str(e)}")

    def get_best_pipeline(self):
        """Return the best performing pipeline"""
        return self.best_pipeline

# Run AutoML pipeline evaluation
print("Evaluating automated ML pipelines...")
automl = AutoMLPipeline()
automl.evaluate_pipelines(X_train, y_train, cv=3)

print(f"\nBest pipeline score: {automl.best_score:.4f}")

# Train and evaluate best pipeline
best_pipeline = automl.get_best_pipeline()
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Best pipeline test accuracy: {test_accuracy:.4f}")
print(f"Best pipeline test F1-score: {test_f1:.4f}")

# CHALLENGE 5: TEXT FEATURE ENGINEERING
print("\n" + "=" * 60)
print("📝 CHALLENGE 5: TEXT FEATURE ENGINEERING")
print("=" * 60)

print("🔤 Advanced Text Processing")

# Prepare text data
review_texts = customer_data['review_text'].values
review_sentiments = customer_data['review_sentiment'].values

# Split text data
X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
    review_texts, review_sentiments, test_size=0.25, random_state=42, stratify=review_sentiments
)

# 1. Bag of Words
print("\n1. Bag of Words (CountVectorizer)")
count_vectorizer = CountVectorizer(max_features=100, stop_words='english')
X_text_bow_train = count_vectorizer.fit_transform(X_text_train)
X_text_bow_test = count_vectorizer.transform(X_text_test)

print(f"Bag of Words shape: {X_text_bow_train.shape}")

# 2. TF-IDF
print("\n2. TF-IDF Vectorization")
tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
X_text_tfidf_train = tfidf_vectorizer.fit_transform(X_text_train)
X_text_tfidf_test = tfidf_vectorizer.transform(X_text_test)

print(f"TF-IDF shape: {X_text_tfidf_train.shape}")

# 3. Text-based features
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract custom text features"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = pd.DataFrame()

        # Text length features
        features['text_length'] = [len(text) for text in X]
        features['word_count'] = [len(text.split()) for text in X]
        features['avg_word_length'] = [np.mean([len(word) for word in text.split()])
                                     if len(text.split()) > 0 else 0 for text in X]

        # Sentiment-related features
        positive_words = ['excellent', 'great', 'amazing', 'perfect', 'wonderful', 'fantastic', 'love']
        negative_words = ['terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'bad']

        features['positive_word_count'] = [
            sum(1 for word in text.lower().split() if word in positive_words) for text in X
        ]
        features['negative_word_count'] = [
            sum(1 for word in text.lower().split() if word in negative_words) for text in X
        ]
        features['sentiment_ratio'] = features['positive_word_count'] / (features['negative_word_count'] + 1)

        return features.values

text_feature_extractor = TextFeatureExtractor()
X_text_features_train = text_feature_extractor.fit_transform(X_text_train)
X_text_features_test = text_feature_extractor.transform(X_text_test)

print(f"Custom text features shape: {X_text_features_train.shape}")

# Combine text features with original features
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50, stop_words='english')),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

text_pipeline.fit(X_text_train, y_text_train)
text_pred = text_pipeline.predict(X_text_test)
text_accuracy = accuracy_score(y_text_test, text_pred)

print(f"Text classification accuracy: {text_accuracy:.4f}")

# Visualize results
plt.figure(figsize=(20, 16))

# Feature selection comparison
plt.subplot(3, 4, 1)
selection_methods = list(all_selection_results.keys())[:4]
n_features = [len(all_selection_results[method]) for method in selection_methods]
plt.bar(selection_methods, n_features, alpha=0.7, color='skyblue')
plt.ylabel('Number of Features Selected')
plt.title('Feature Selection Methods Comparison')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Feature importance from best model (if available)
plt.subplot(3, 4, 2)
try:
    # Get feature importance from the best pipeline
    if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
        feature_importance = best_pipeline.named_steps['classifier'].feature_importances_
        # Get the selected features after preprocessing
        selected_indices = best_pipeline.named_steps['feature_selector'].get_support()
        selected_feature_names = X_train.columns[selected_indices]

        # Plot top 10 features
        top_indices = np.argsort(feature_importance)[-10:]
        plt.barh(range(10), feature_importance[top_indices])
        plt.yticks(range(10), [selected_feature_names[i] for i in top_indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Feature Importance')
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center')
        plt.title('Feature Importance')
except:
    plt.text(0.5, 0.5, 'Unable to plot\nfeature importance', ha='center', va='center')
    plt.title('Feature Importance')

# PCA explained variance
plt.subplot(3, 4, 3)
cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'b-', marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)

# t-SNE visualization
plt.subplot(3, 4, 4)
scatter = plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1],
                     c=y_subset, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization')

# AutoML pipeline performance comparison
plt.subplot(3, 4, 5)
# Get top 10 pipelines
sorted_results = sorted(automl.results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
top_10_results = sorted_results[:10]

pipeline_names = [name.split('_') for name, _ in top_10_results]
pipeline_scores = [result['cv_mean'] for _, result in top_10_results]

plt.barh(range(len(pipeline_names)), pipeline_scores, alpha=0.7)
plt.yticks(range(len(pipeline_names)), [f"{p[0]}+{p[1]}+{p[2]}" for p in pipeline_names])
plt.xlabel('CV F1-Score')
plt.title('Top 10 AutoML Pipelines')
plt.grid(axis='x', alpha=0.3)

# Dimensionality reduction comparison
plt.subplot(3, 4, 6)
dr_methods = list(dimensionality_results.keys())
dr_components = [dimensionality_results[method]['n_components'] for method in dr_methods]
dr_explained_var = [dimensionality_results[method].get('explained_variance', 0)
                   for method in dr_methods]

x_pos = np.arange(len(dr_methods))
plt.bar(x_pos, dr_components, alpha=0.7, color='lightcoral', label='Components')
plt.bar(x_pos, [var * 100 for var in dr_explained_var], alpha=0.7, color='lightblue', label='Explained Var %')
plt.xticks(x_pos, dr_methods, rotation=45)
plt.ylabel('Number of Components / Explained Variance %')
plt.title('Dimensionality Reduction Comparison')
plt.legend()

# Feature correlation heatmap (top features)
plt.subplot(3, 4, 7)
top_features = feature_scores.head(10)['feature'].values
corr_matrix = X_train[top_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
           square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Top Features Correlation')

# Text feature distribution
plt.subplot(3, 4, 8)
text_features_df = pd.DataFrame(X_text_features_train,
                               columns=['text_length', 'word_count', 'avg_word_length',
                                      'positive_words', 'negative_words', 'sentiment_ratio'])
text_features_df['sentiment'] = y_text_train

# Plot sentiment ratio distribution by class
for sentiment in sorted(text_features_df['sentiment'].unique()):
    subset = text_features_df[text_features_df['sentiment'] == sentiment]
    plt.hist(subset['sentiment_ratio'], alpha=0.6, label=f'Sentiment {sentiment}', bins=20)

plt.xlabel('Sentiment Ratio')
plt.ylabel('Frequency')
plt.title('Sentiment Ratio Distribution')
plt.legend()

# Model performance comparison across different data representations
plt.subplot(3, 4, 9)
performance_comparison = {
    'Original': test_f1,
    'PCA': 0.0,  # Placeholder
    'Selected Features': 0.0,  # Placeholder
    'Text Features': text_accuracy
}

# Evaluate some key representations
try:
    # Evaluate PCA representation
    pca_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    pca_scores = cross_val_score(pca_classifier, X_train_pca, y_train, cv=3, scoring='f1_weighted')
    performance_comparison['PCA'] = pca_scores.mean()

    # Evaluate selected features
    selected_X_train = X_train_scaled[:, univariate_selector.get_support()]
    selected_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    selected_scores = cross_val_score(selected_classifier, selected_X_train, y_train, cv=3, scoring='f1_weighted')
    performance_comparison['Selected Features'] = selected_scores.mean()
except:
    pass

methods = list(performance_comparison.keys())
scores = list(performance_comparison.values())
plt.bar(methods, scores, alpha=0.7, color='lightgreen')
plt.ylabel('Performance Score')
plt.title('Data Representation Performance')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Feature selection overlap visualization
plt.subplot(3, 4, 10)
overlap_data = list(feature_overlap.values())
overlap_labels = [label.split(' ∩ ') for label in feature_overlap.keys()]
overlap_short_labels = [f"{l[0].split()[0]}\n∩\n{l[1].split()[0]}" for l in overlap_labels]

plt.bar(range(len(overlap_data)), overlap_data, alpha=0.7, color='orange')
plt.xticks(range(len(overlap_data)), overlap_short_labels, rotation=45)
plt.ylabel('Number of Common Features')
plt.title('Feature Selection Overlap')
plt.grid(axis='y', alpha=0.3)

# Pipeline component analysis
plt.subplot(3, 4, 11)
# Analyze which components work best together
component_performance = {}
for name, result in sorted_results[:15]:
    components = name.split('_')
    fs_method = components[0]

    if fs_method not in component_performance:
        component_performance[fs_method] = []
    component_performance[fs_method].append(result['cv_mean'])

fs_methods = list(component_performance.keys())
avg_performance = [np.mean(component_performance[method]) for method in fs_methods]

plt.bar(fs_methods, avg_performance, alpha=0.7, color='purple')
plt.ylabel('Average CV F1-Score')
plt.title('Feature Selection Method Performance')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Advanced feature engineering impact
plt.subplot(3, 4, 12)
# Compare original vs engineered features
original_features = len(numeric_features)
engineered_features = customer_transformed.shape[1] - original_features

categories = ['Original', 'Engineered', 'Total']
counts = [original_features, engineered_features, customer_transformed.shape[1]]
colors = ['lightblue', 'lightcoral', 'lightgreen']

plt.bar(categories, counts, color=colors, alpha=0.7)
plt.ylabel('Number of Features')
plt.title('Feature Engineering Impact')
plt.grid(axis='y', alpha=0.3)

# Add count labels on bars
for i, count in enumerate(counts):
    plt.text(i, count + 1, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("🧬 ADVANCED FEATURE ENGINEERING INSIGHTS")
print("=" * 60)

print("📋 Key Findings:")
print(f"1. Feature Selection:")
print(f"   • Univariate selection: {len(selected_univariate_features)} features")
print(f"   • Mutual information: {len(selected_mutual_info_features)} features")
print(f"   • RFE: {len(selected_rfe_features)} features")
print(f"   • RFECV optimal: {rfecv_selector.n_features_} features")

print(f"\n2. Dimensionality Reduction:")
print(f"   • PCA: {pca.n_components_} components for 95% variance")
print(f"   • SVD: {svd.n_components} components, {svd.explained_variance_ratio_.sum():.3f} variance")
print(f"   • Original dimensions: {X_train.shape[1]}")

print(f"\n3. Automated ML:")
print(f"   • Best pipeline: {automl.best_score:.4f} CV F1-score")
print(f"   • Test performance: {test_f1:.4f} F1-score")
print(f"   • Evaluated {len(automl.results)} pipeline configurations")

print(f"\n4. Text Processing:")
print(f"   • Text classification accuracy: {text_accuracy:.4f}")
print(f"   • TF-IDF features: {X_text_tfidf_train.shape[1]}")
print(f"   • Custom text features: {X_text_features_train.shape[1]}")

print(f"\n🔧 Production Recommendations:")
print("• Use RFECV for automatic feature selection with cross-validation")
print("• Combine multiple feature selection methods for robustness")
print("• Apply PCA when interpretability is not critical")
print("• Use automated pipeline evaluation for systematic model selection")
print("• Implement custom transformers for domain-specific feature engineering")
print("• Monitor feature importance drift in production systems")

print(f"\n💡 Advanced Techniques:")
print("• Custom transformer classes enable sophisticated feature engineering")
print("• Pipeline automation reduces manual hyperparameter tuning")
print("• Feature selection ensemble methods improve stability")
print("• Text feature engineering beyond bag-of-words improves performance")
print("• Dimensionality reduction preserves information while reducing complexity")

print("\n✅ Advanced Feature Engineering and AutoML Challenge Completed!")
print("What you've mastered:")
print("• Advanced feature transformation and interaction creation")
print("• Multiple feature selection techniques and comparison")
print("• Dimensionality reduction methods (PCA, SVD, ICA, Factor Analysis)")
print("• Automated ML pipeline construction and evaluation")
print("• Text feature engineering and natural language processing")
print("• Production-ready feature engineering frameworks")

print(f"\n🔬 You are now a Feature Engineering Expert! Ready for production systems!")
```

### Success Criteria

- Implement advanced feature transformation and interaction techniques
- Master multiple feature selection methods and automated selection
- Apply dimensionality reduction techniques effectively
- Build automated ML pipelines with systematic evaluation
- Create sophisticated text feature engineering workflows
- Develop production-ready feature engineering frameworks

### Learning Objectives

- Understand advanced feature engineering principles and techniques
- Master automated feature selection and dimensionality reduction
- Learn to build and evaluate automated ML pipelines
- Practice advanced text processing and feature extraction
- Develop skills in systematic feature engineering workflows
- Create scalable and robust feature engineering systems

---

_Pro tip: Great features beat great algorithms - invest time in understanding your data and creating meaningful feature representations!_
