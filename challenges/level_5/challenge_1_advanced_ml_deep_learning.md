# Level 5: Algorithm Architect

## Challenge 1: Advanced Machine Learning & Deep Learning

Welcome to the advanced realm of AI! Master sophisticated algorithms and neural networks.

### Objective
Build and deploy advanced machine learning models including neural networks, ensemble methods, and automated ML pipelines.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("🧠 Welcome to Advanced Machine Learning!")
print("==========================================")

# Generate complex synthetic dataset
np.random.seed(42)
n_samples = 2000

# Create multi-class classification problem
X = np.random.randn(n_samples, 20)  # 20 features
# Create complex non-linear relationships
feature_interactions = (X[:, 0] * X[:, 1] + 
                       np.sin(X[:, 2]) * X[:, 3] + 
                       X[:, 4] ** 2 + 
                       X[:, 5] * X[:, 6] * X[:, 7])

# Create target variable with multiple classes
y = np.zeros(n_samples)
y[feature_interactions < -1] = 0  # Class 0
y[(feature_interactions >= -1) & (feature_interactions < 1)] = 1  # Class 1
y[feature_interactions >= 1] = 2  # Class 2

# Add some noise
noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
y[noise_indices] = np.random.choice([0, 1, 2], size=len(noise_indices))

# Create DataFrame
feature_names = [f'feature_{i:02d}' for i in range(20)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y.astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{pd.Series(y).value_counts().sort_index()}")

# Your tasks:
# 1. ADVANCED FEATURE ENGINEERING
print("\n=== ADVANCED FEATURE ENGINEERING ===")

# Split data
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection with multiple methods
print("🔍 Feature Selection:")

# Statistical selection
selector_chi2 = SelectKBest(chi2, k=10)
X_train_chi2 = selector_chi2.fit_transform(np.abs(X_train_scaled), y_train)
X_test_chi2 = selector_chi2.transform(np.abs(X_test_scaled))

selected_features_chi2 = [feature_names[i] for i in selector_chi2.get_support(indices=True)]
print(f"Chi2 selected features: {selected_features_chi2[:5]}...")

# Recursive Feature Elimination
rfe = RFE(GradientBoostingClassifier(random_state=42), n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

selected_features_rfe = [feature_names[i] for i in rfe.get_support(indices=True)]
print(f"RFE selected features: {selected_features_rfe[:5]}...")

# Principal Component Analysis
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA components: {pca.n_components_} (explained variance: {pca.explained_variance_ratio_.sum():.3f})")

# 2. ENSEMBLE METHODS
print("\n=== ENSEMBLE METHODS ===")

# Individual models
models = {
    'gradient_boost': GradientBoostingClassifier(random_state=42),
    'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    'voting': VotingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42))
        ],
        voting='soft'
    )
}

ensemble_results = {}

print("🚀 Training Ensemble Models:")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    
    if y_pred_proba is not None:
        # Multi-class AUC (one vs rest)
        auc_scores = []
        for i in range(3):  # 3 classes
            y_binary = (y_test == i).astype(int)
            if len(np.unique(y_binary)) > 1:  # Only if both classes present
                auc = roc_auc_score(y_binary, y_pred_proba[:, i])
                auc_scores.append(auc)
        avg_auc = np.mean(auc_scores) if auc_scores else 0
    else:
        avg_auc = 0
    
    ensemble_results[name] = {
        'accuracy': accuracy,
        'auc': avg_auc
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Average AUC: {avg_auc:.4f}")

# Advanced Stacking Ensemble
print("\n🏗️ Building Stacking Ensemble:")
stacking_clf = StackingClassifier(
    estimators=[
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42))
    ],
    final_estimator=MLPClassifier(hidden_layer_sizes=(25,), max_iter=1000, random_state=42),
    cv=5
)

stacking_clf.fit(X_train_scaled, y_train)
y_pred_stack = stacking_clf.predict(X_test_scaled)
y_pred_stack_proba = stacking_clf.predict_proba(X_test_scaled)

stack_accuracy = (y_pred_stack == y_test).mean()
stack_auc_scores = []
for i in range(3):
    y_binary = (y_test == i).astype(int)
    if len(np.unique(y_binary)) > 1:
        auc = roc_auc_score(y_binary, y_pred_stack_proba[:, i])
        stack_auc_scores.append(auc)
stack_avg_auc = np.mean(stack_auc_scores)

print(f"Stacking Accuracy: {stack_accuracy:.4f}")
print(f"Stacking Average AUC: {stack_avg_auc:.4f}")

# 3. DEEP LEARNING WITH TENSORFLOW
print("\n=== DEEP LEARNING WITH TENSORFLOW ===")

# Prepare data for TensorFlow
X_train_tf = X_train_scaled.astype(np.float32)
X_test_tf = X_test_scaled.astype(np.float32)
y_train_tf = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_tf = tf.keras.utils.to_categorical(y_test, num_classes=3)

print("🧠 Building Deep Neural Network:")

# Build deep neural network
def create_deep_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(20,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train model
deep_model = create_deep_model()

print("Model architecture:")
deep_model.summary()

# Training with callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

print("\n🔥 Training Deep Neural Network...")
history = deep_model.fit(
    X_train_tf, y_train_tf,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=0
)

# Evaluate deep model
deep_loss, deep_accuracy = deep_model.evaluate(X_test_tf, y_test_tf, verbose=0)
y_pred_deep_proba = deep_model.predict(X_test_tf, verbose=0)
y_pred_deep = np.argmax(y_pred_deep_proba, axis=1)

print(f"Deep Learning Accuracy: {deep_accuracy:.4f}")

# Calculate AUC for deep model
deep_auc_scores = []
for i in range(3):
    y_binary = (y_test == i).astype(int)
    if len(np.unique(y_binary)) > 1:
        auc = roc_auc_score(y_binary, y_pred_deep_proba[:, i])
        deep_auc_scores.append(auc)
deep_avg_auc = np.mean(deep_auc_scores)

print(f"Deep Learning Average AUC: {deep_avg_auc:.4f}")

# Training history visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 4. HYPERPARAMETER OPTIMIZATION
print("\n=== HYPERPARAMETER OPTIMIZATION ===")

print("🎯 Advanced Hyperparameter Tuning:")

# Randomized search for neural network
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 50, 25)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [1000]
}

mlp_random = RandomizedSearchCV(
    MLPClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

mlp_random.fit(X_train_scaled, y_train)

print(f"Best MLP parameters: {mlp_random.best_params_}")
print(f"Best cross-validation score: {mlp_random.best_score_:.4f}")

# Test optimized model
y_pred_optimized = mlp_random.predict(X_test_scaled)
optimized_accuracy = (y_pred_optimized == y_test).mean()
print(f"Optimized model test accuracy: {optimized_accuracy:.4f}")

# 5. MODEL INTERPRETATION AND ANALYSIS
print("\n=== MODEL INTERPRETATION ===")

# Feature importance from gradient boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)

feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("🔍 Top 10 Most Important Features:")
print(feature_importance_df.head(10))

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Confusion matrices comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Confusion Matrix Comparison', fontsize=16)

models_comparison = {
    'Gradient Boosting': (gb_model.predict(X_test_scaled), axes[0, 0]),
    'Stacking Ensemble': (y_pred_stack, axes[0, 1]),
    'Deep Neural Network': (y_pred_deep, axes[1, 0]),
    'Optimized MLP': (y_pred_optimized, axes[1, 1])
}

for name, (predictions, ax) in models_comparison.items():
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 6. AUTOMATED ML PIPELINE
print("\n=== AUTOMATED ML PIPELINE ===")

print("🤖 Building Automated ML Pipeline:")

# Create comprehensive pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(chi2, k=15)),
    ('classifier', StackingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
        ],
        final_estimator=GradientBoostingClassifier(random_state=42),
        cv=5
    ))
])

# Fit pipeline
pipeline.fit(X_train, y_train)
y_pred_pipeline = pipeline.predict(X_test)
pipeline_accuracy = (y_pred_pipeline == y_test).mean()

print(f"Automated Pipeline Accuracy: {pipeline_accuracy:.4f}")

# 7. MODEL PERFORMANCE SUMMARY
print("\n=== PERFORMANCE SUMMARY ===")

results_summary = pd.DataFrame({
    'Model': ['Gradient Boosting', 'MLP', 'Voting Ensemble', 'Stacking Ensemble', 
              'Deep Neural Network', 'Optimized MLP', 'Automated Pipeline'],
    'Accuracy': [
        ensemble_results['gradient_boost']['accuracy'],
        ensemble_results['mlp']['accuracy'],
        ensemble_results['voting']['accuracy'],
        stack_accuracy,
        deep_accuracy,
        optimized_accuracy,
        pipeline_accuracy
    ],
    'AUC': [
        ensemble_results['gradient_boost']['auc'],
        ensemble_results['mlp']['auc'],
        ensemble_results['voting']['auc'],
        stack_avg_auc,
        deep_avg_auc,
        0,  # Not calculated for optimized MLP
        0   # Not calculated for pipeline
    ]
}).sort_values('Accuracy', ascending=False)

print("\n📊 Model Performance Ranking:")
print(results_summary.round(4))

# Best model visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(data=results_summary, x='Accuracy', y='Model')
plt.title('Model Accuracy Comparison')
plt.xlabel('Accuracy Score')

plt.subplot(1, 2, 2)
auc_data = results_summary[results_summary['AUC'] > 0]
if len(auc_data) > 0:
    sns.barplot(data=auc_data, x='AUC', y='Model')
plt.title('Model AUC Comparison')
plt.xlabel('AUC Score')

plt.tight_layout()
plt.show()

print("\n✅ Advanced Machine Learning Challenge Completed!")
print("\n🎓 Congratulations! You've mastered:")
print("• Advanced ensemble methods (voting, stacking)")
print("• Deep neural networks with TensorFlow/Keras")
print("• Sophisticated hyperparameter optimization")
print("• Automated ML pipelines")
print("• Feature selection and dimensionality reduction")
print("• Model interpretation and comparison")
print("• Production-ready ML workflows")
print("\n🏆 You're now an Algorithm Architect!")
```

### Success Criteria
- Implement multiple ensemble methods with improved performance
- Build and train deep neural networks using TensorFlow
- Perform advanced hyperparameter optimization
- Create automated ML pipelines
- Compare model performance comprehensively
- Demonstrate feature engineering and selection techniques

### Learning Objectives
- Master ensemble learning techniques
- Understand deep learning fundamentals
- Learn automated machine learning concepts
- Practice model interpretation and comparison
- Develop production ML pipeline skills
- Gain experience with advanced optimization techniques

### Next Steps
Ready for Level 6? You'll tackle real-world projects combining multiple data science domains including NLP, computer vision, and time series analysis!

---

*Pro tip: The best model isn't always the most complex one. Always consider interpretability, training time, and deployment requirements in your model selection!*