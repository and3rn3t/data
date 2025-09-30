# Level 5: Algorithm Architect

## Challenge 1: Advanced Algorithms and Ensemble Methods

Master sophisticated machine learning algorithms and powerful ensemble techniques to build state-of-the-art predictive models.

### Objective

Learn advanced algorithms including boosting, bagging, stacking, and sophisticated ensemble strategies that power modern ML systems.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold,
                                   learning_curve, validation_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier,
                            VotingClassifier, StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report,
                           confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

print("🏗️ Advanced Algorithms and Ensemble Methods")
print("=" * 50)

# Create complex multi-class dataset for advanced algorithms
np.random.seed(42)

print("📊 Creating Complex Multi-Modal Dataset...")

# Simulate customer behavior analysis with multiple data sources
n_customers = 3000

# Generate multiple data modalities
demographic_data = pd.DataFrame({
    'age': np.random.gamma(2, 20, n_customers).clip(18, 80),
    'income': np.random.lognormal(10.5, 0.8, n_customers),
    'education_years': np.random.normal(14, 3, n_customers).clip(8, 25),
    'family_size': np.random.poisson(2.5, n_customers).clip(1, 8)
})

behavioral_data = pd.DataFrame({
    'website_visits': np.random.negative_binomial(10, 0.3, n_customers),
    'session_duration': np.random.exponential(25, n_customers).clip(1, 200),
    'pages_viewed': np.random.poisson(8, n_customers),
    'bounce_rate': np.random.beta(2, 5, n_customers),
    'conversion_events': np.random.poisson(3, n_customers)
})

transactional_data = pd.DataFrame({
    'purchase_frequency': np.random.gamma(1.5, 2, n_customers),
    'avg_order_value': np.random.lognormal(4, 0.6, n_customers),
    'discount_usage': np.random.beta(3, 7, n_customers),
    'return_rate': np.random.beta(2, 8, n_customers),
    'loyalty_points': np.random.exponential(500, n_customers)
})

engagement_data = pd.DataFrame({
    'email_opens': np.random.negative_binomial(5, 0.4, n_customers),
    'social_shares': np.random.poisson(2, n_customers),
    'review_count': np.random.poisson(1.5, n_customers),
    'referrals': np.random.poisson(0.8, n_customers),
    'support_tickets': np.random.poisson(1.2, n_customers)
})

# Combine all data sources
customer_data = pd.concat([demographic_data, behavioral_data,
                          transactional_data, engagement_data], axis=1)

# Create complex customer value segments with realistic interactions
# High-value customers
high_value_score = (
    (customer_data['income'] > customer_data['income'].quantile(0.7)) * 3 +
    (customer_data['purchase_frequency'] > customer_data['purchase_frequency'].quantile(0.8)) * 2 +
    (customer_data['avg_order_value'] > customer_data['avg_order_value'].quantile(0.75)) * 2 +
    (customer_data['loyalty_points'] > customer_data['loyalty_points'].quantile(0.8)) * 1 +
    (customer_data['session_duration'] > customer_data['session_duration'].quantile(0.7)) * 1
)

# Medium-value customers
medium_value_score = (
    (customer_data['website_visits'] > customer_data['website_visits'].quantile(0.5)) * 2 +
    (customer_data['conversion_events'] > 2) * 2 +
    (customer_data['email_opens'] > customer_data['email_opens'].quantile(0.6)) * 1 +
    (customer_data['education_years'] > 12) * 1
)

# Create final segments with some noise
segment_scores = np.zeros(n_customers)
segment_scores += high_value_score * 0.4
segment_scores += medium_value_score * 0.3
segment_scores += np.random.normal(0, 1, n_customers)  # Add complexity

# Convert to discrete customer segments (0: Low, 1: Medium, 2: High, 3: Premium)
customer_data['value_segment'] = pd.cut(segment_scores,
                                       bins=[-np.inf, -1, 1, 3, np.inf],
                                       labels=[0, 1, 2, 3]).astype(int)

print(f"Dataset shape: {customer_data.shape}")
print(f"Segment distribution:")
print(customer_data['value_segment'].value_counts().sort_index())

# Prepare data for modeling
X = customer_data.drop('value_segment', axis=1)
y = customer_data['value_segment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features for algorithms that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# CHALLENGE 1: ADVANCED BOOSTING ALGORITHMS
print("\n" + "=" * 60)
print("🚀 CHALLENGE 1: ADVANCED BOOSTING ALGORITHMS")
print("=" * 60)

print("🏆 Gradient Boosting Classifier")

# Implement Gradient Boosting with different configurations
gb_models = {}

# Standard Gradient Boosting
gb_standard = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_standard.fit(X_train, y_train)
gb_standard_pred = gb_standard.predict(X_test)
gb_standard_score = f1_score(y_test, gb_standard_pred, average='weighted')
gb_models['Standard GB'] = {'model': gb_standard, 'score': gb_standard_score}

print(f"Standard GB F1-score: {gb_standard_score:.4f}")

# Aggressive Gradient Boosting (higher learning rate, more estimators)
gb_aggressive = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.2,
    max_depth=7,
    min_samples_split=5,
    random_state=42
)

gb_aggressive.fit(X_train, y_train)
gb_aggressive_pred = gb_aggressive.predict(X_test)
gb_aggressive_score = f1_score(y_test, gb_aggressive_pred, average='weighted')
gb_models['Aggressive GB'] = {'model': gb_aggressive, 'score': gb_aggressive_score}

print(f"Aggressive GB F1-score: {gb_aggressive_score:.4f}")

# Conservative Gradient Boosting (lower learning rate, regularization)
gb_conservative = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

gb_conservative.fit(X_train, y_train)
gb_conservative_pred = gb_conservative.predict(X_test)
gb_conservative_score = f1_score(y_test, gb_conservative_pred, average='weighted')
gb_models['Conservative GB'] = {'model': gb_conservative, 'score': gb_conservative_score}

print(f"Conservative GB F1-score: {gb_conservative_score:.4f}")

print("\n🚀 XGBoost Implementation")

# XGBoost with advanced parameters
xgb_classifier = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='mlogloss'
)

xgb_classifier.fit(X_train, y_train)
xgb_pred = xgb_classifier.predict(X_test)
xgb_score = f1_score(y_test, xgb_pred, average='weighted')
gb_models['XGBoost'] = {'model': xgb_classifier, 'score': xgb_score}

print(f"XGBoost F1-score: {xgb_score:.4f}")

print("\n🚀 LightGBM Implementation")

# LightGBM with advanced parameters
lgb_classifier = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

lgb_classifier.fit(X_train, y_train)
lgb_pred = lgb_classifier.predict(X_test)
lgb_score = f1_score(y_test, lgb_pred, average='weighted')
gb_models['LightGBM'] = {'model': lgb_classifier, 'score': lgb_score}

print(f"LightGBM F1-score: {lgb_score:.4f}")

# CHALLENGE 2: ADVANCED BAGGING AND RANDOM FORESTS
print("\n" + "=" * 60)
print("🌳 CHALLENGE 2: ADVANCED BAGGING TECHNIQUES")
print("=" * 60)

print("🎯 Random Forest Variations")

# Standard Random Forest
rf_standard = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_standard.fit(X_train, y_train)
rf_standard_pred = rf_standard.predict(X_test)
rf_standard_score = f1_score(y_test, rf_standard_pred, average='weighted')

print(f"Standard Random Forest F1-score: {rf_standard_score:.4f}")

# Extra Trees (Extremely Randomized Trees)
et_classifier = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

et_classifier.fit(X_train, y_train)
et_pred = et_classifier.predict(X_test)
et_score = f1_score(y_test, et_pred, average='weighted')

print(f"Extra Trees F1-score: {et_score:.4f}")

# Custom Bagging with different base estimators
print("\n🎲 Advanced Bagging Strategies")

# Bagging with Decision Trees
bagging_dt = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=8),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

bagging_dt.fit(X_train, y_train)
bagging_dt_pred = bagging_dt.predict(X_test)
bagging_dt_score = f1_score(y_test, bagging_dt_pred, average='weighted')

print(f"Bagging Decision Trees F1-score: {bagging_dt_score:.4f}")

# Bagging with SVM
bagging_svm = BaggingClassifier(
    base_estimator=SVC(kernel='rbf', C=1.0),
    n_estimators=50,  # Fewer estimators for SVM due to computational cost
    max_samples=0.8,
    random_state=42,
    n_jobs=-1
)

bagging_svm.fit(X_train_scaled, y_train)
bagging_svm_pred = bagging_svm.predict(X_test_scaled)
bagging_svm_score = f1_score(y_test, bagging_svm_pred, average='weighted')

print(f"Bagging SVM F1-score: {bagging_svm_score:.4f}")

# CHALLENGE 3: VOTING AND STACKING ENSEMBLES
print("\n" + "=" * 60)
print("🗳️ CHALLENGE 3: VOTING AND STACKING ENSEMBLES")
print("=" * 60)

print("🎭 Voting Classifiers")

# Create diverse base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)),
    ('nb', GaussianNB())
]

# Hard Voting Classifier
hard_voting = VotingClassifier(
    estimators=base_models,
    voting='hard'
)

# We need to use scaled data for SVM and MLP
# Create pipelines to handle scaling automatically
from sklearn.pipeline import Pipeline

base_models_with_scaling = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))])),
    ('mlp', Pipeline([('scaler', StandardScaler()),
                     ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500))])),
    ('nb', Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())]))
]

# Hard Voting
hard_voting = VotingClassifier(
    estimators=base_models_with_scaling,
    voting='hard'
)

hard_voting.fit(X_train, y_train)
hard_voting_pred = hard_voting.predict(X_test)
hard_voting_score = f1_score(y_test, hard_voting_pred, average='weighted')

print(f"Hard Voting Ensemble F1-score: {hard_voting_score:.4f}")

# Soft Voting Classifier
soft_voting = VotingClassifier(
    estimators=base_models_with_scaling,
    voting='soft'
)

soft_voting.fit(X_train, y_train)
soft_voting_pred = soft_voting.predict(X_test)
soft_voting_score = f1_score(y_test, soft_voting_pred, average='weighted')

print(f"Soft Voting Ensemble F1-score: {soft_voting_score:.4f}")

print("\n🏗️ Stacking Ensemble")

# Create stacking ensemble with diverse base models
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42))
]

# Meta-learner
meta_learner = LogisticRegression(random_state=42, max_iter=1000)

# Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba'
)

stacking_classifier.fit(X_train, y_train)
stacking_pred = stacking_classifier.predict(X_test)
stacking_score = f1_score(y_test, stacking_pred, average='weighted')

print(f"Stacking Ensemble F1-score: {stacking_score:.4f}")

# Multi-level Stacking
print("\n🏗️ Multi-Level Stacking")

# Level 1 base learners
level1_learners = [
    ('rf1', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
    ('rf2', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=43)),
    ('gb1', GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)),
    ('gb2', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=43)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42))
]

# Level 2 meta-learner
level2_meta = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)

multilevel_stacking = StackingClassifier(
    estimators=level1_learners,
    final_estimator=level2_meta,
    cv=5
)

multilevel_stacking.fit(X_train, y_train)
multilevel_pred = multilevel_stacking.predict(X_test)
multilevel_score = f1_score(y_test, multilevel_pred, average='weighted')

print(f"Multi-Level Stacking F1-score: {multilevel_score:.4f}")

# CHALLENGE 4: ADAPTIVE BOOSTING (ADABOOST)
print("\n" + "=" * 60)
print("⚡ CHALLENGE 4: ADAPTIVE BOOSTING TECHNIQUES")
print("=" * 60)

print("🎯 AdaBoost Variations")

# Standard AdaBoost
ada_standard = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

ada_standard.fit(X_train, y_train)
ada_standard_pred = ada_standard.predict(X_test)
ada_standard_score = f1_score(y_test, ada_standard_pred, average='weighted')

print(f"Standard AdaBoost F1-score: {ada_standard_score:.4f}")

# AdaBoost with different base estimators
ada_deep_trees = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)

ada_deep_trees.fit(X_train, y_train)
ada_deep_pred = ada_deep_trees.predict(X_test)
ada_deep_score = f1_score(y_test, ada_deep_pred, average='weighted')

print(f"AdaBoost Deep Trees F1-score: {ada_deep_score:.4f}")

# Visualize algorithm performance comparison
plt.figure(figsize=(20, 16))

# Collect all results
all_results = {
    'Standard GB': gb_standard_score,
    'Aggressive GB': gb_aggressive_score,
    'Conservative GB': gb_conservative_score,
    'XGBoost': xgb_score,
    'LightGBM': lgb_score,
    'Random Forest': rf_standard_score,
    'Extra Trees': et_score,
    'Bagging DT': bagging_dt_score,
    'Bagging SVM': bagging_svm_score,
    'Hard Voting': hard_voting_score,
    'Soft Voting': soft_voting_score,
    'Stacking': stacking_score,
    'Multi-Level Stack': multilevel_score,
    'AdaBoost Standard': ada_standard_score,
    'AdaBoost Deep': ada_deep_score
}

# Algorithm performance comparison
plt.subplot(2, 4, 1)
algorithms = list(all_results.keys())
scores = list(all_results.values())
colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))

bars = plt.bar(range(len(algorithms)), scores, color=colors, alpha=0.7)
plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
plt.ylabel('F1 Score')
plt.title('Algorithm Performance Comparison')
plt.grid(axis='y', alpha=0.3)

# Highlight best performing algorithms
best_idx = np.argmax(scores)
bars[best_idx].set_color('gold')
bars[best_idx].set_edgecolor('black')

# Boosting algorithms comparison
plt.subplot(2, 4, 2)
boosting_algos = ['Standard GB', 'Aggressive GB', 'Conservative GB', 'XGBoost', 'LightGBM', 'AdaBoost Standard']
boosting_scores = [all_results[algo] for algo in boosting_algos]

plt.bar(boosting_algos, boosting_scores, alpha=0.7, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('F1 Score')
plt.title('Boosting Algorithms Comparison')
plt.grid(axis='y', alpha=0.3)

# Ensemble methods comparison
plt.subplot(2, 4, 3)
ensemble_algos = ['Random Forest', 'Extra Trees', 'Bagging DT', 'Hard Voting', 'Soft Voting', 'Stacking']
ensemble_scores = [all_results[algo] for algo in ensemble_algos]

plt.bar(ensemble_algos, ensemble_scores, alpha=0.7, color='lightcoral')
plt.xticks(rotation=45, ha='right')
plt.ylabel('F1 Score')
plt.title('Ensemble Methods Comparison')
plt.grid(axis='y', alpha=0.3)

# Feature importance analysis for best model
plt.subplot(2, 4, 4)
best_model = gb_models[max(gb_models.keys(), key=lambda k: gb_models[k]['score'])]['model']
feature_importance = best_model.feature_importances_
feature_names = X.columns

# Plot top 10 most important features
top_features_idx = np.argsort(feature_importance)[-10:]
plt.barh(range(10), feature_importance[top_features_idx])
plt.yticks(range(10), [feature_names[i] for i in top_features_idx])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importance (Best GB Model)')

# Learning curves for different boosting algorithms
plt.subplot(2, 4, 5)
train_sizes = np.linspace(0.1, 1.0, 10)
for name, config in [('Standard GB', gb_standard), ('XGBoost', xgb_classifier)]:
    train_sizes_abs, train_scores, val_scores = learning_curve(
        config, X_train, y_train, train_sizes=train_sizes, cv=3, scoring='f1_weighted', n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.plot(train_sizes_abs, train_mean, 'o-', label=f'{name} (Train)')
    plt.plot(train_sizes_abs, val_mean, 's-', label=f'{name} (Val)', linestyle='--')

plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.title('Learning Curves: Boosting Algorithms')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion matrix for best ensemble model
plt.subplot(2, 4, 6)
best_ensemble_score = max(soft_voting_score, stacking_score, multilevel_score)
if best_ensemble_score == soft_voting_score:
    best_ensemble_pred = soft_voting_pred
    best_ensemble_name = 'Soft Voting'
elif best_ensemble_score == stacking_score:
    best_ensemble_pred = stacking_pred
    best_ensemble_name = 'Stacking'
else:
    best_ensemble_pred = multilevel_pred
    best_ensemble_name = 'Multi-Level Stacking'

cm = confusion_matrix(y_test, best_ensemble_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix: {best_ensemble_name}')

# Algorithm complexity vs performance
plt.subplot(2, 4, 7)
# Estimate relative complexity (arbitrary units for demonstration)
complexity_estimates = {
    'Random Forest': 3, 'Extra Trees': 3, 'Standard GB': 5, 'XGBoost': 6,
    'LightGBM': 4, 'Soft Voting': 7, 'Stacking': 8, 'Multi-Level Stack': 10
}

selected_algos = list(complexity_estimates.keys())
selected_scores = [all_results[algo] for algo in selected_algos]
selected_complexity = list(complexity_estimates.values())

plt.scatter(selected_complexity, selected_scores, s=100, alpha=0.7)
for i, algo in enumerate(selected_algos):
    plt.annotate(algo, (selected_complexity[i], selected_scores[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('Algorithm Complexity (Relative)')
plt.ylabel('F1 Score')
plt.title('Complexity vs Performance Trade-off')
plt.grid(True, alpha=0.3)

# ROC curves for top 3 algorithms
plt.subplot(2, 4, 8)
top_3_algos = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:3]

# For multi-class ROC, we'll show ROC for class 2 (high-value customers)
for algo_name, _ in top_3_algos:
    if algo_name == 'Soft Voting':
        y_proba = soft_voting.predict_proba(X_test)[:, 2]
    elif algo_name == 'Stacking':
        y_proba = stacking_classifier.predict_proba(X_test)[:, 2]
    elif algo_name == 'XGBoost':
        y_proba = xgb_classifier.predict_proba(X_test)[:, 2]
    else:
        continue  # Skip if we don't have predict_proba

    # Binary classification for class 2 vs rest
    y_binary = (y_test == 2).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, y_proba)
    auc_score = roc_auc_score(y_binary, y_proba)

    plt.plot(fpr, tpr, label=f'{algo_name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Class 2 vs Rest')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# CHALLENGE 5: PERFORMANCE ANALYSIS AND INSIGHTS
print("\n" + "=" * 60)
print("📊 CHALLENGE 5: ALGORITHM PERFORMANCE ANALYSIS")
print("=" * 60)

# Detailed performance comparison
results_df = pd.DataFrame({
    'Algorithm': list(all_results.keys()),
    'F1_Score': list(all_results.values())
})

results_df = results_df.sort_values('F1_Score', ascending=False)

print("🏆 Algorithm Rankings:")
print("=" * 40)
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    print(f"{i:2d}. {row['Algorithm']:20s} | F1: {row['F1_Score']:.4f}")

# Statistical significance testing
print(f"\n📈 Performance Analysis:")
print(f"Best Algorithm: {results_df.iloc[0]['Algorithm']} (F1: {results_df.iloc[0]['F1_Score']:.4f})")
print(f"Worst Algorithm: {results_df.iloc[-1]['Algorithm']} (F1: {results_df.iloc[-1]['F1_Score']:.4f})")
print(f"Performance Spread: {results_df['F1_Score'].max() - results_df['F1_Score'].min():.4f}")
print(f"Mean Performance: {results_df['F1_Score'].mean():.4f} ± {results_df['F1_Score'].std():.4f}")

# Ensemble performance analysis
ensemble_methods = ['Hard Voting', 'Soft Voting', 'Stacking', 'Multi-Level Stack']
ensemble_scores = [all_results[method] for method in ensemble_methods]
individual_methods = [method for method in all_results.keys() if method not in ensemble_methods]
individual_scores = [all_results[method] for method in individual_methods]

print(f"\n🎭 Ensemble vs Individual Performance:")
print(f"Best Individual: {max(individual_scores):.4f}")
print(f"Best Ensemble: {max(ensemble_scores):.4f}")
print(f"Average Individual: {np.mean(individual_scores):.4f}")
print(f"Average Ensemble: {np.mean(ensemble_scores):.4f}")

if max(ensemble_scores) > max(individual_scores):
    improvement = ((max(ensemble_scores) - max(individual_scores)) / max(individual_scores)) * 100
    print(f"Ensemble Improvement: +{improvement:.1f}%")
else:
    decrease = ((max(individual_scores) - max(ensemble_scores)) / max(individual_scores)) * 100
    print(f"Ensemble Decrease: -{decrease:.1f}%")

print("\n" + "=" * 60)
print("🎯 ADVANCED ALGORITHMS INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

print("📋 Key Findings:")
print("1. Boosting Algorithms:")
print(f"   • Best Boosting: {max(gb_models.keys(), key=lambda k: gb_models[k]['score'])} (F1: {max(gb_models.values(), key=lambda x: x['score'])['score']:.4f})")
print("   • XGBoost and LightGBM provide excellent performance with built-in regularization")
print("   • Conservative settings often prevent overfitting in complex datasets")

print("\n2. Ensemble Methods:")
if max(ensemble_scores) > max(individual_scores):
    print("   • Ensemble methods outperform individual algorithms")
    print("   • Soft voting typically better than hard voting for probabilistic models")
    print("   • Stacking can capture complex model interactions")
else:
    print("   • Individual algorithms competitive with ensembles on this dataset")
    print("   • Model diversity crucial for ensemble effectiveness")

print("\n3. Algorithm Selection Guidelines:")
print("   • Use XGBoost/LightGBM for production systems (speed + performance)")
print("   • Use Random Forest for interpretable baseline models")
print("   • Use Stacking when you have diverse, well-performing base models")
print("   • Use Gradient Boosting when you can afford longer training times")

print(f"\n🚀 Production Recommendations:")
print(f"• Primary Model: {results_df.iloc[0]['Algorithm']} (highest F1-score)")
print(f"• Backup Model: {results_df.iloc[1]['Algorithm']} (second highest)")
print("• Consider ensemble of top 3 algorithms for critical applications")
print("• Implement proper cross-validation and hyperparameter tuning")

print("\n✅ Advanced Algorithms and Ensemble Methods Challenge Completed!")
print("What you've mastered:")
print("• Advanced boosting algorithms (Gradient Boosting, XGBoost, LightGBM)")
print("• Sophisticated bagging techniques (Random Forest, Extra Trees, Custom Bagging)")
print("• Ensemble methods (Voting, Stacking, Multi-level ensembles)")
print("• Adaptive boosting (AdaBoost variations)")
print("• Performance analysis and algorithm selection strategies")
print("• Production-ready ensemble architectures")

print(f"\n🏗️ You are now an Algorithm Architect! Ready for deep learning and neural networks!")
```

### Success Criteria

- Implement and compare multiple advanced boosting algorithms
- Build sophisticated ensemble methods including stacking and voting
- Master bagging techniques and custom ensemble strategies
- Analyze algorithm performance and complexity trade-offs
- Create production-ready model selection frameworks
- Develop expertise in algorithm architecture decisions

### Learning Objectives

- Understand advanced boosting algorithms and their variations
- Master ensemble techniques for improved model performance
- Learn to build complex stacking and voting classifiers
- Develop skills in algorithm performance analysis
- Practice production-ready model selection strategies
- Build comprehensive algorithm comparison frameworks

---

_Pro tip: The best algorithm is often not a single model but a carefully crafted ensemble of diverse, high-performing base learners!_
