# Level 4: Machine Learning Novice

## Challenge 4: Hyperparameter Tuning and Model Selection

Master the art of hyperparameter optimization and systematic model selection to squeeze every bit of performance from your models.

### Objective

Learn advanced hyperparameter tuning techniques, automated model selection, and optimization strategies to build the best possible models for any dataset.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, GridSearchCV, RandomizedSearchCV,
                                   cross_val_score, StratifiedKFold, validation_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                           mean_squared_error, r2_score, classification_report)
from scipy.stats import uniform, randint
import time
import warnings
warnings.filterwarnings('ignore')

print("🎛️ Hyperparameter Tuning and Model Selection Challenge")
print("=" * 57)

# Create challenging dataset for hyperparameter tuning
np.random.seed(42)

print("📊 Creating Complex Dataset for Optimization...")

# Multi-class classification dataset (Customer Segmentation)
n_customers = 2000

customer_data = pd.DataFrame({
    'age': np.random.normal(40, 15, n_customers).clip(18, 80),
    'income': np.random.lognormal(10.8, 0.6, n_customers),
    'spending_score': np.random.normal(50, 25, n_customers).clip(1, 100),
    'loyalty_years': np.random.exponential(3, n_customers).clip(0, 20),
    'purchase_frequency': np.random.poisson(8, n_customers),
    'avg_order_value': np.random.gamma(2, 50, n_customers),
    'seasonal_purchases': np.random.binomial(4, 0.6, n_customers),
    'digital_engagement': np.random.beta(2, 3, n_customers) * 100,
    'support_tickets': np.random.poisson(2, n_customers),
    'referrals_made': np.random.poisson(1.5, n_customers)
})

# Create complex customer segments with realistic relationships
segment_scores = np.zeros(n_customers)

# High-value customers (segment 2)
high_value_mask = (customer_data['income'] > customer_data['income'].quantile(0.7)) & \
                 (customer_data['spending_score'] > 60) & \
                 (customer_data['loyalty_years'] > 2)
segment_scores[high_value_mask] += 2

# Medium-value customers (segment 1)
medium_value_mask = (customer_data['income'] > customer_data['income'].quantile(0.3)) & \
                   (customer_data['spending_score'] > 30) & \
                   (customer_data['purchase_frequency'] > 5)
segment_scores[medium_value_mask & ~high_value_mask] += 1

# Add some noise and edge cases
noise = np.random.normal(0, 0.3, n_customers)
segment_scores = segment_scores + noise

# Convert to discrete segments
customer_data['segment'] = np.clip(np.round(segment_scores), 0, 2).astype(int)

print(f"Dataset shape: {customer_data.shape}")
print(f"Segment distribution:")
print(customer_data['segment'].value_counts().sort_index())

# CHALLENGE 1: GRID SEARCH HYPERPARAMETER TUNING
print("\n" + "=" * 60)
print("🔍 CHALLENGE 1: GRID SEARCH HYPERPARAMETER TUNING")
print("=" * 60)

# Prepare data
X = customer_data.drop('segment', axis=1)
y = customer_data['segment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features for algorithms that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🔧 Grid Search: Random Forest Classifier")

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Perform grid search
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
rf_grid.fit(X_train, y_train)
rf_grid_time = time.time() - start_time

print(f"Grid Search completed in {rf_grid_time:.2f} seconds")
print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best cross-validation score: {rf_grid.best_score_:.4f}")

# Test best model
rf_best = rf_grid.best_estimator_
rf_test_score = f1_score(y_test, rf_best.predict(X_test), average='macro')
print(f"Test set F1-score: {rf_test_score:.4f}")

print("\n🔧 Grid Search: SVM Classifier")

# SVM parameter grid
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

svm_grid = GridSearchCV(
    SVC(random_state=42),
    svm_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

start_time = time.time()
svm_grid.fit(X_train_scaled, y_train)
svm_grid_time = time.time() - start_time

print(f"SVM Grid Search completed in {svm_grid_time:.2f} seconds")
print(f"Best parameters: {svm_grid.best_params_}")
print(f"Best cross-validation score: {svm_grid.best_score_:.4f}")

# CHALLENGE 2: RANDOMIZED SEARCH FOR EFFICIENCY
print("\n" + "=" * 60)
print("🎲 CHALLENGE 2: RANDOMIZED SEARCH OPTIMIZATION")
print("=" * 60)

print("🚀 Randomized Search: Gradient Boosting Classifier")

# Define parameter distributions for randomized search
gb_param_dist = {
    'n_estimators': randint(50, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'subsample': uniform(0.6, 0.4),
    'max_features': ['sqrt', 'log2', None]
}

gb_random = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_dist,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
gb_random.fit(X_train, y_train)
gb_random_time = time.time() - start_time

print(f"Randomized Search completed in {gb_random_time:.2f} seconds")
print(f"Best parameters: {gb_random.best_params_}")
print(f"Best cross-validation score: {gb_random.best_score_:.4f}")

# Compare with default parameters
gb_default = GradientBoostingClassifier(random_state=42)
gb_default.fit(X_train, y_train)
default_score = f1_score(y_test, gb_default.predict(X_test), average='macro')
tuned_score = f1_score(y_test, gb_random.best_estimator_.predict(X_test), average='macro')

print(f"\nPerformance comparison:")
print(f"Default GB F1-score: {default_score:.4f}")
print(f"Tuned GB F1-score: {tuned_score:.4f}")
print(f"Improvement: {((tuned_score - default_score) / default_score * 100):.1f}%")

# Visualize search results
plt.figure(figsize=(15, 10))

# Grid search results visualization
plt.subplot(2, 3, 1)
rf_results = pd.DataFrame(rf_grid.cv_results_)
plt.scatter(rf_results['param_n_estimators'], rf_results['mean_test_score'], alpha=0.6)
plt.xlabel('n_estimators')
plt.ylabel('CV Score')
plt.title('RF Grid Search: n_estimators vs Score')

plt.subplot(2, 3, 2)
plt.scatter(rf_results['param_max_depth'].fillna(-1), rf_results['mean_test_score'], alpha=0.6)
plt.xlabel('max_depth (None=-1)')
plt.ylabel('CV Score')
plt.title('RF Grid Search: max_depth vs Score')

# Randomized search results
plt.subplot(2, 3, 3)
gb_results = pd.DataFrame(gb_random.cv_results_)
plt.scatter(gb_results['param_n_estimators'], gb_results['mean_test_score'], alpha=0.6)
plt.xlabel('n_estimators')
plt.ylabel('CV Score')
plt.title('GB Random Search: n_estimators vs Score')

plt.subplot(2, 3, 4)
plt.scatter(gb_results['param_learning_rate'], gb_results['mean_test_score'], alpha=0.6)
plt.xlabel('learning_rate')
plt.ylabel('CV Score')
plt.title('GB Random Search: learning_rate vs Score')

# Search efficiency comparison
plt.subplot(2, 3, 5)
search_methods = ['RF Grid Search', 'SVM Grid Search', 'GB Random Search']
search_times = [rf_grid_time, svm_grid_time, gb_random_time]
search_scores = [rf_grid.best_score_, svm_grid.best_score_, gb_random.best_score_]

plt.scatter(search_times, search_scores, s=100)
for i, method in enumerate(search_methods):
    plt.annotate(method, (search_times[i], search_scores[i]),
                xytext=(5, 5), textcoords='offset points')
plt.xlabel('Search Time (seconds)')
plt.ylabel('Best CV Score')
plt.title('Search Efficiency Comparison')

# Parameter importance for Random Forest
plt.subplot(2, 3, 6)
param_importance = {}
for param in ['param_n_estimators', 'param_max_depth', 'param_min_samples_split']:
    if param in rf_results.columns:
        correlation = rf_results[param].fillna(-1).corr(rf_results['mean_test_score'])
        param_importance[param.replace('param_', '')] = abs(correlation)

plt.bar(param_importance.keys(), param_importance.values())
plt.ylabel('Absolute Correlation with CV Score')
plt.title('RF Parameter Importance')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# CHALLENGE 3: ADVANCED OPTIMIZATION STRATEGIES
print("\n" + "=" * 60)
print("🧠 CHALLENGE 3: ADVANCED OPTIMIZATION STRATEGIES")
print("=" * 60)

# Multi-level hyperparameter tuning
print("🔥 Multi-Level Hyperparameter Tuning")

# Step 1: Coarse grid search
print("\nStep 1: Coarse parameter exploration")
coarse_param_grid = {
    'n_estimators': [50, 200, 500],
    'max_depth': [5, 15, None],
    'learning_rate': [0.05, 0.1, 0.2]
}

coarse_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    coarse_param_grid,
    cv=3,  # Fewer folds for speed
    scoring='f1_macro',
    n_jobs=-1
)

coarse_search.fit(X_train, y_train)
print(f"Coarse search best score: {coarse_search.best_score_:.4f}")
print(f"Coarse search best params: {coarse_search.best_params_}")

# Step 2: Fine-tune around best parameters
print("\nStep 2: Fine-tuning around best parameters")
best_coarse = coarse_search.best_params_

fine_param_grid = {
    'n_estimators': [max(50, best_coarse['n_estimators'] - 50),
                    best_coarse['n_estimators'],
                    best_coarse['n_estimators'] + 50],
    'max_depth': [best_coarse['max_depth'] - 2 if best_coarse['max_depth'] else 13,
                 best_coarse['max_depth'] if best_coarse['max_depth'] else 15,
                 best_coarse['max_depth'] + 2 if best_coarse['max_depth'] else 17] if best_coarse['max_depth'] else [13, 15, 17],
    'learning_rate': [max(0.01, best_coarse['learning_rate'] - 0.02),
                     best_coarse['learning_rate'],
                     min(0.3, best_coarse['learning_rate'] + 0.02)]
}

fine_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    fine_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

fine_search.fit(X_train, y_train)
print(f"Fine search best score: {fine_search.best_score_:.4f}")
print(f"Fine search best params: {fine_search.best_params_}")

# CHALLENGE 4: AUTOMATED MODEL SELECTION
print("\n" + "=" * 60)
print("🤖 CHALLENGE 4: AUTOMATED MODEL SELECTION")
print("=" * 60)

print("🏆 Comprehensive Model Comparison with Tuning")

# Define multiple algorithms with their parameter spaces
algorithms = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'scaled': True
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        'scaled': False
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'scaled': False
    },
    'SVM': {
        'model': SVC(random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.1],
            'kernel': ['rbf', 'poly']
        },
        'scaled': True
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'scaled': True
    }
}

# Perform automated model selection
model_results = {}
best_models = {}

print("Training and tuning multiple algorithms...")
for name, config in algorithms.items():
    print(f"\n🔧 Optimizing {name}...")

    # Choose appropriate data (scaled or not)
    X_train_data = X_train_scaled if config['scaled'] else X_train
    X_test_data = X_test_scaled if config['scaled'] else X_test

    # Perform grid search
    grid_search = GridSearchCV(
        config['model'],
        config['params'],
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )

    start_time = time.time()
    grid_search.fit(X_train_data, y_train)
    training_time = time.time() - start_time

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    test_predictions = best_model.predict(X_test_data)

    results = {
        'best_cv_score': grid_search.best_score_,
        'test_f1_score': f1_score(y_test, test_predictions, average='macro'),
        'test_accuracy': accuracy_score(y_test, test_predictions),
        'training_time': training_time,
        'best_params': grid_search.best_params_
    }

    model_results[name] = results
    best_models[name] = best_model

    print(f"  Best CV Score: {results['best_cv_score']:.4f}")
    print(f"  Test F1 Score: {results['test_f1_score']:.4f}")
    print(f"  Training Time: {results['training_time']:.2f}s")

# Create comprehensive comparison
results_df = pd.DataFrame(model_results).T
print(f"\n📊 Model Selection Results:")
print(results_df.round(4))

# Rank models by test performance
ranking = results_df.sort_values('test_f1_score', ascending=False)
print(f"\n🏅 Model Ranking (by Test F1-Score):")
for i, (model, metrics) in enumerate(ranking.iterrows(), 1):
    print(f"{i}. {model}: {metrics['test_f1_score']:.4f}")

# Visualize model comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Automated Model Selection Results', fontsize=16, fontweight='bold')

# CV Score vs Test Score
axes[0, 0].scatter(results_df['best_cv_score'], results_df['test_f1_score'], s=100)
for i, model in enumerate(results_df.index):
    axes[0, 0].annotate(model, (results_df.iloc[i]['best_cv_score'],
                               results_df.iloc[i]['test_f1_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[0, 0].plot([0.3, 0.8], [0.3, 0.8], 'r--', alpha=0.5)
axes[0, 0].set_xlabel('CV F1 Score')
axes[0, 0].set_ylabel('Test F1 Score')
axes[0, 0].set_title('CV vs Test Performance')

# Training time vs Performance
axes[0, 1].scatter(results_df['training_time'], results_df['test_f1_score'], s=100)
for i, model in enumerate(results_df.index):
    axes[0, 1].annotate(model, (results_df.iloc[i]['training_time'],
                               results_df.iloc[i]['test_f1_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[0, 1].set_xlabel('Training Time (seconds)')
axes[0, 1].set_ylabel('Test F1 Score')
axes[0, 1].set_title('Efficiency vs Performance')

# Model performance comparison
axes[0, 2].bar(results_df.index, results_df['test_f1_score'], alpha=0.7)
axes[0, 2].set_ylabel('Test F1 Score')
axes[0, 2].set_title('Model Performance Comparison')
axes[0, 2].tick_params(axis='x', rotation=45)

# CV vs Test score difference (overfitting indicator)
overfitting = results_df['best_cv_score'] - results_df['test_f1_score']
axes[1, 0].bar(results_df.index, overfitting, alpha=0.7)
axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1, 0].set_ylabel('CV Score - Test Score')
axes[1, 0].set_title('Overfitting Analysis')
axes[1, 0].tick_params(axis='x', rotation=45)

# Training time comparison
axes[1, 1].bar(results_df.index, results_df['training_time'], alpha=0.7)
axes[1, 1].set_ylabel('Training Time (seconds)')
axes[1, 1].set_title('Training Efficiency')
axes[1, 1].tick_params(axis='x', rotation=45)

# Accuracy vs F1 score
axes[1, 2].scatter(results_df['test_accuracy'], results_df['test_f1_score'], s=100)
for i, model in enumerate(results_df.index):
    axes[1, 2].annotate(model, (results_df.iloc[i]['test_accuracy'],
                               results_df.iloc[i]['test_f1_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[1, 2].set_xlabel('Test Accuracy')
axes[1, 2].set_ylabel('Test F1 Score')
axes[1, 2].set_title('Accuracy vs F1 Score')

plt.tight_layout()
plt.show()

# CHALLENGE 5: ENSEMBLE AND PIPELINE OPTIMIZATION
print("\n" + "=" * 60)
print("🎭 CHALLENGE 5: ENSEMBLE OPTIMIZATION")
print("=" * 60)

print("🎯 Optimizing Ensemble Methods")

# Voting ensemble with optimized base models
from sklearn.ensemble import VotingClassifier

# Get top 3 models
top_3_models = ranking.head(3).index.tolist()
ensemble_estimators = []

for model_name in top_3_models:
    if model_name in best_models:
        # Use scaled or unscaled data based on model requirements
        if algorithms[model_name]['scaled']:
            # Create a pipeline that includes scaling
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', best_models[model_name])
            ])
            ensemble_estimators.append((model_name.lower().replace(' ', '_'), pipeline))
        else:
            ensemble_estimators.append((model_name.lower().replace(' ', '_'), best_models[model_name]))

# Create voting ensemble
voting_ensemble = VotingClassifier(
    estimators=ensemble_estimators,
    voting='soft'  # Use predict_proba for soft voting
)

print(f"Creating ensemble with: {[name for name, _ in ensemble_estimators]}")

# Train ensemble
voting_ensemble.fit(X_train, y_train)
ensemble_predictions = voting_ensemble.predict(X_test)

ensemble_f1 = f1_score(y_test, ensemble_predictions, average='macro')
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

print(f"Ensemble Performance:")
print(f"  F1 Score: {ensemble_f1:.4f}")
print(f"  Accuracy: {ensemble_accuracy:.4f}")

# Compare with best individual model
best_individual = ranking.iloc[0]['test_f1_score']
ensemble_improvement = ((ensemble_f1 - best_individual) / best_individual) * 100

print(f"\nEnsemble vs Best Individual Model:")
print(f"  Best Individual: {best_individual:.4f}")
print(f"  Ensemble: {ensemble_f1:.4f}")
print(f"  Improvement: {ensemble_improvement:.1f}%")

# CHALLENGE 6: HYPERPARAMETER OPTIMIZATION ANALYSIS
print("\n" + "=" * 60)
print("📈 CHALLENGE 6: OPTIMIZATION ANALYSIS & INSIGHTS")
print("=" * 60)

# Analyze parameter sensitivity
print("🔍 Parameter Sensitivity Analysis")

# For the best model, analyze how parameters affect performance
best_model_name = ranking.index[0]
best_config = algorithms[best_model_name]

print(f"\nAnalyzing {best_model_name} parameter sensitivity...")

# Create validation curves for key parameters
if best_model_name == 'Random Forest':
    # Analyze n_estimators
    param_range = [10, 25, 50, 100, 200, 300, 500, 1000]
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42),
        X_train, y_train,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='f1_macro', n_jobs=-1
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')

    plt.xlabel('n_estimators')
    plt.ylabel('F1 Score')
    plt.title('Random Forest: n_estimators vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Analyze max_depth
    depth_range = [3, 5, 7, 10, 15, 20, None]
    train_scores_depth, val_scores_depth = validation_curve(
        RandomForestClassifier(random_state=42, n_estimators=100),
        X_train, y_train,
        param_name='max_depth', param_range=depth_range,
        cv=5, scoring='f1_macro', n_jobs=-1
    )

    plt.subplot(1, 3, 2)
    train_mean_depth = train_scores_depth.mean(axis=1)
    val_mean_depth = val_scores_depth.mean(axis=1)

    x_pos = range(len(depth_range))
    plt.plot(x_pos, train_mean_depth, 'o-', color='blue', label='Training score')
    plt.plot(x_pos, val_mean_depth, 'o-', color='red', label='Cross-validation score')

    plt.xticks(x_pos, [str(d) if d else 'None' for d in depth_range])
    plt.xlabel('max_depth')
    plt.ylabel('F1 Score')
    plt.title('Random Forest: max_depth vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Search strategy comparison summary
plt.subplot(1, 3, 3)
strategies = ['Grid Search\n(RF)', 'Grid Search\n(SVM)', 'Random Search\n(GB)', 'Multi-level\n(GB)']
times = [rf_grid_time, svm_grid_time, gb_random_time,
         coarse_search.cv_results_['mean_fit_time'].sum() + fine_search.cv_results_['mean_fit_time'].sum()]
scores = [rf_grid.best_score_, svm_grid.best_score_, gb_random.best_score_, fine_search.best_score_]

colors = ['blue', 'green', 'orange', 'purple']
plt.scatter(times, scores, c=colors, s=100)
for i, strategy in enumerate(strategies):
    plt.annotate(strategy, (times[i], scores[i]), xytext=(5, 5),
                textcoords='offset points', fontsize=9)

plt.xlabel('Total Time (seconds)')
plt.ylabel('Best CV Score')
plt.title('Optimization Strategy Comparison')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final recommendations
print("\n" + "=" * 60)
print("🎯 HYPERPARAMETER TUNING INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

print("📋 Key Findings:")
print(f"1. Best performing algorithm: {ranking.index[0]} (F1: {ranking.iloc[0]['test_f1_score']:.4f})")
print(f"2. Most efficient algorithm: {results_df.loc[results_df['training_time'].idxmin()].name}")
print(f"3. Most stable algorithm: {results_df.loc[(results_df['best_cv_score'] - results_df['test_f1_score']).abs().idxmin()].name}")

if ensemble_improvement > 0:
    print(f"4. Ensemble provides {ensemble_improvement:.1f}% improvement over best individual model")
else:
    print(f"4. Best individual model outperforms ensemble by {abs(ensemble_improvement):.1f}%")

print(f"\n🎛️ Tuning Strategy Recommendations:")
print("• Use randomized search for initial exploration with many parameters")
print("• Follow up with grid search for fine-tuning around promising regions")
print("• Consider multi-level search for complex models like Gradient Boosting")
print("• Always validate on held-out test set to detect overfitting")
print("• Consider ensemble methods when individual models have similar performance")

print(f"\n💡 Parameter Insights:")
for model_name, results in model_results.items():
    print(f"• {model_name}: {results['best_params']}")

print("\n✅ Hyperparameter Tuning and Model Selection Challenge Completed!")
print("What you've mastered:")
print("• Grid search and randomized search hyperparameter optimization")
print("• Multi-level search strategies for efficiency")
print("• Automated model selection and comparison frameworks")
print("• Ensemble method optimization")
print("• Parameter sensitivity analysis and validation curves")
print("• Production-ready model selection pipelines")

print(f"\n🎓 You now have the expertise to optimize any ML model systematically!")
```

### Success Criteria

- Implement comprehensive hyperparameter tuning strategies (grid, random, multi-level)
- Build automated model selection and comparison frameworks
- Optimize ensemble methods for improved performance
- Analyze parameter sensitivity and model behavior
- Create efficient search strategies balancing time and performance
- Develop production-ready optimization pipelines

### Learning Objectives

- Master systematic hyperparameter optimization techniques
- Understand trade-offs between search strategies and computational efficiency
- Learn to build automated model selection systems
- Practice ensemble optimization and combination strategies
- Develop skills in parameter sensitivity analysis
- Create scalable optimization frameworks for real-world applications

---

_Pro tip: The best hyperparameters are dataset-specific - always validate your optimization strategy and never trust a single metric!_
