# Level 5: Algorithm Architect

## Challenge 2: Deep Learning and Neural Networks

Master neural networks and deep learning architectures to solve complex pattern recognition and prediction problems.

### Objective

Learn to build, train, and optimize neural networks using both scikit-learn and TensorFlow/Keras for various machine learning tasks.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                           confusion_matrix, mean_squared_error, r2_score)
from sklearn.datasets import make_classification, make_regression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, regularizers
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🧠 Deep Learning and Neural Networks")
print("=" * 40)

# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Create complex datasets for neural network training
print("📊 Creating Complex Neural Network Datasets...")

# Dataset 1: Image-like pattern recognition (flattened 2D patterns)
def create_pattern_dataset(n_samples=3000, pattern_size=20):
    """Create synthetic 2D pattern dataset"""
    X = np.random.randn(n_samples, pattern_size, pattern_size)
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Create different pattern types
        pattern_type = i % 4

        if pattern_type == 0:  # Diagonal pattern
            for j in range(pattern_size):
                X[i, j, j] += 3
                if j < pattern_size - 1:
                    X[i, j, j+1] += 2
            y[i] = 0

        elif pattern_type == 1:  # Corner pattern
            X[i, :5, :5] += 3
            X[i, -5:, -5:] += 3
            y[i] = 1

        elif pattern_type == 2:  # Cross pattern
            mid = pattern_size // 2
            X[i, mid-2:mid+3, :] += 3
            X[i, :, mid-2:mid+3] += 3
            y[i] = 2

        else:  # Circle pattern
            center = pattern_size // 2
            for r in range(pattern_size):
                for c in range(pattern_size):
                    dist = np.sqrt((r - center)**2 + (c - center)**2)
                    if 5 <= dist <= 8:
                        X[i, r, c] += 3
            y[i] = 3

    # Flatten for traditional ML
    X_flat = X.reshape(n_samples, -1)
    return X, X_flat, y

# Dataset 2: Time series-like sequential data
def create_sequential_dataset(n_samples=2000, sequence_length=50, n_features=10):
    """Create synthetic sequential dataset"""
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.zeros(n_samples)

    for i in range(n_samples):
        # Create different sequential patterns
        if i % 3 == 0:  # Increasing trend
            trend = np.linspace(0, 2, sequence_length)
            X[i, :, 0] += trend
            y[i] = 0
        elif i % 3 == 1:  # Periodic pattern
            time = np.arange(sequence_length)
            X[i, :, 1] += 2 * np.sin(2 * np.pi * time / 20)
            y[i] = 1
        else:  # Random walk with drift
            drift = np.cumsum(np.random.randn(sequence_length) * 0.1 + 0.05)
            X[i, :, 2] += drift
            y[i] = 2

    return X, y

# Create datasets
print("Creating pattern recognition dataset...")
X_patterns, X_patterns_flat, y_patterns = create_pattern_dataset(n_samples=2000, pattern_size=16)

print("Creating sequential dataset...")
X_sequential, y_sequential = create_sequential_dataset(n_samples=1500, sequence_length=40, n_features=8)

print("Creating tabular dataset...")
X_tabular, y_tabular = make_classification(
    n_samples=2500, n_features=50, n_informative=30, n_redundant=10,
    n_classes=4, n_clusters_per_class=2, class_sep=1.5, random_state=42
)

print(f"Pattern dataset: {X_patterns_flat.shape}, Classes: {len(np.unique(y_patterns))}")
print(f"Sequential dataset: {X_sequential.shape}, Classes: {len(np.unique(y_sequential))}")
print(f"Tabular dataset: {X_tabular.shape}, Classes: {len(np.unique(y_tabular))}")

# CHALLENGE 1: SCIKIT-LEARN NEURAL NETWORKS
print("\n" + "=" * 60)
print("🔥 CHALLENGE 1: SCIKIT-LEARN NEURAL NETWORKS")
print("=" * 60)

print("🎯 Multi-Layer Perceptron Classification")

# Prepare tabular data
X_tab_train, X_tab_test, y_tab_train, y_tab_test = train_test_split(
    X_tabular, y_tabular, test_size=0.25, random_state=42, stratify=y_tabular
)

scaler_tab = StandardScaler()
X_tab_train_scaled = scaler_tab.fit_transform(X_tab_train)
X_tab_test_scaled = scaler_tab.transform(X_tab_test)

# Different MLP architectures
mlp_configs = {
    'Small MLP': {
        'hidden_layer_sizes': (50,),
        'activation': 'relu',
        'alpha': 0.001,
        'learning_rate_init': 0.001,
        'max_iter': 500
    },
    'Medium MLP': {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'alpha': 0.01,
        'learning_rate_init': 0.001,
        'max_iter': 500
    },
    'Large MLP': {
        'hidden_layer_sizes': (200, 100, 50),
        'activation': 'relu',
        'alpha': 0.01,
        'learning_rate_init': 0.001,
        'max_iter': 500
    },
    'Deep MLP': {
        'hidden_layer_sizes': (150, 100, 75, 50, 25),
        'activation': 'relu',
        'alpha': 0.05,
        'learning_rate_init': 0.001,
        'max_iter': 800
    }
}

mlp_results = {}

for name, config in mlp_configs.items():
    print(f"\n🔧 Training {name}...")

    mlp = MLPClassifier(random_state=42, **config)
    mlp.fit(X_tab_train_scaled, y_tab_train)

    # Evaluate
    train_pred = mlp.predict(X_tab_train_scaled)
    test_pred = mlp.predict(X_tab_test_scaled)

    train_acc = accuracy_score(y_tab_train, train_pred)
    test_acc = accuracy_score(y_tab_test, test_pred)
    test_f1 = f1_score(y_tab_test, test_pred, average='weighted')

    mlp_results[name] = {
        'model': mlp,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'n_layers': len(config['hidden_layer_sizes']) + 1,
        'n_params': sum(config['hidden_layer_sizes']) + config['hidden_layer_sizes'][0] * X_tabular.shape[1]
    }

    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Training Iterations: {mlp.n_iter_}")

# Different activation functions
print("\n🧪 Activation Function Comparison")

activations = ['logistic', 'tanh', 'relu']
activation_results = {}

for activation in activations:
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation=activation,
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )

    mlp.fit(X_tab_train_scaled, y_tab_train)
    test_pred = mlp.predict(X_tab_test_scaled)
    test_f1 = f1_score(y_tab_test, test_pred, average='weighted')

    activation_results[activation] = test_f1
    print(f"  {activation.capitalize()}: F1 = {test_f1:.4f}")

# MLP Regression
print("\n🎯 Multi-Layer Perceptron Regression")

# Create regression dataset
X_reg, y_reg = make_regression(
    n_samples=1500, n_features=20, n_informative=15,
    noise=0.1, random_state=42
)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=42
)

scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# Different solver comparison
solvers = ['lbfgs', 'sgd', 'adam']
solver_results = {}

for solver in solvers:
    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver=solver,
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )

    mlp_reg.fit(X_reg_train_scaled, y_reg_train)
    reg_pred = mlp_reg.predict(X_reg_test_scaled)

    mse = mean_squared_error(y_reg_test, reg_pred)
    r2 = r2_score(y_reg_test, reg_pred)

    solver_results[solver] = {'mse': mse, 'r2': r2}
    print(f"  {solver.upper()}: MSE = {mse:.2f}, R² = {r2:.4f}")

# CHALLENGE 2: TENSORFLOW/KERAS DEEP LEARNING
print("\n" + "=" * 60)
print("🚀 CHALLENGE 2: TENSORFLOW/KERAS DEEP LEARNING")
print("=" * 60)

print("🔥 Building Deep Neural Networks with Keras")

# Prepare pattern data for deep learning
X_pat_train, X_pat_test, y_pat_train, y_pat_test = train_test_split(
    X_patterns_flat, y_patterns, test_size=0.25, random_state=42, stratify=y_patterns
)

scaler_pat = StandardScaler()
X_pat_train_scaled = scaler_pat.fit_transform(X_pat_train)
X_pat_test_scaled = scaler_pat.transform(X_pat_test)

# Convert labels to categorical
y_pat_train_cat = to_categorical(y_pat_train, num_classes=4)
y_pat_test_cat = to_categorical(y_pat_test, num_classes=4)

print(f"Pattern data prepared: {X_pat_train_scaled.shape} -> {y_pat_train_cat.shape}")

# Basic Dense Neural Network
print("\n🏗️ Basic Dense Neural Network")

def create_dense_model(input_dim, num_classes, architecture='medium'):
    """Create different dense neural network architectures"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,))
    ])

    if architecture == 'shallow':
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
    elif architecture == 'medium':
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
    elif architecture == 'deep':
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))

    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Train different architectures
architectures = ['shallow', 'medium', 'deep']
keras_results = {}

for arch in architectures:
    print(f"\n🔧 Training {arch.capitalize()} Architecture...")

    model = create_dense_model(X_pat_train_scaled.shape[1], 4, arch)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train with validation split
    history = model.fit(
        X_pat_train_scaled, y_pat_train_cat,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_pat_test_scaled, y_pat_test_cat, verbose=0)

    keras_results[arch] = {
        'model': model,
        'history': history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'epochs_trained': len(history.history['loss'])
    }

    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Epochs Trained: {len(history.history['loss'])}")

# Advanced Neural Network with Regularization
print("\n🛡️ Regularized Deep Neural Network")

def create_regularized_model(input_dim, num_classes):
    """Create a deep neural network with various regularization techniques"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # First block with dropout
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Second block
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Third block
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Fourth block
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Train regularized model
regularized_model = create_regularized_model(X_pat_train_scaled.shape[1], 4)

regularized_model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training regularized model...")
reg_history = regularized_model.fit(
    X_pat_train_scaled, y_pat_train_cat,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    verbose=0,
    callbacks=[
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.3, patience=7, min_lr=1e-7)
    ]
)

reg_test_loss, reg_test_acc = regularized_model.evaluate(X_pat_test_scaled, y_pat_test_cat, verbose=0)
print(f"Regularized Model - Test Accuracy: {reg_test_acc:.4f}, Test Loss: {reg_test_loss:.4f}")

# CHALLENGE 3: CONVOLUTIONAL NEURAL NETWORKS
print("\n" + "=" * 60)
print("🖼️ CHALLENGE 3: CONVOLUTIONAL NEURAL NETWORKS")
print("=" * 60)

print("🔍 CNN for Pattern Recognition")

# Reshape pattern data for CNN (add channel dimension)
X_pat_2d_train = X_patterns[:len(X_pat_train)].reshape(-1, 16, 16, 1)
X_pat_2d_test = X_patterns[len(X_pat_train):].reshape(-1, 16, 16, 1)

# Normalize pixel values
X_pat_2d_train = X_pat_2d_train.astype('float32') / np.std(X_pat_2d_train)
X_pat_2d_test = X_pat_2d_test.astype('float32') / np.std(X_pat_2d_test)

print(f"CNN input shape: {X_pat_2d_train.shape}")

def create_cnn_model(input_shape, num_classes):
    """Create a CNN for pattern recognition"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Train CNN
cnn_model = create_cnn_model((16, 16, 1), 4)

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training CNN...")
cnn_history = cnn_model.fit(
    X_pat_2d_train, y_pat_train_cat,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    verbose=0,
    callbacks=[
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)

cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_pat_2d_test, y_pat_test_cat, verbose=0)
print(f"CNN - Test Accuracy: {cnn_test_acc:.4f}, Test Loss: {cnn_test_loss:.4f}")

# CHALLENGE 4: RECURRENT NEURAL NETWORKS
print("\n" + "=" * 60)
print("🔄 CHALLENGE 4: RECURRENT NEURAL NETWORKS")
print("=" * 60)

print("🎵 LSTM for Sequential Data")

# Prepare sequential data
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
    X_sequential, y_sequential, test_size=0.25, random_state=42, stratify=y_sequential
)

# Normalize sequential data
scaler_seq = StandardScaler()
X_seq_train_scaled = scaler_seq.fit_transform(X_seq_train.reshape(-1, X_seq_train.shape[-1]))
X_seq_train_scaled = X_seq_train_scaled.reshape(X_seq_train.shape)

X_seq_test_scaled = scaler_seq.transform(X_seq_test.reshape(-1, X_seq_test.shape[-1]))
X_seq_test_scaled = X_seq_test_scaled.reshape(X_seq_test.shape)

# Convert labels to categorical
y_seq_train_cat = to_categorical(y_seq_train, num_classes=3)
y_seq_test_cat = to_categorical(y_seq_test, num_classes=3)

print(f"Sequential data shape: {X_seq_train_scaled.shape}")

def create_lstm_model(input_shape, num_classes, model_type='simple'):
    """Create different LSTM architectures"""
    model = keras.Sequential([
        layers.Input(shape=input_shape)
    ])

    if model_type == 'simple':
        model.add(layers.LSTM(50, activation='tanh'))
    elif model_type == 'stacked':
        model.add(layers.LSTM(64, return_sequences=True, activation='tanh'))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(32, activation='tanh'))
    elif model_type == 'bidirectional':
        model.add(layers.Bidirectional(layers.LSTM(32, activation='tanh')))

    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Train different RNN architectures
rnn_types = ['simple', 'stacked', 'bidirectional']
rnn_results = {}

for rnn_type in rnn_types:
    print(f"\n🔧 Training {rnn_type.capitalize()} LSTM...")

    lstm_model = create_lstm_model((X_seq_train_scaled.shape[1], X_seq_train_scaled.shape[2]), 3, rnn_type)

    lstm_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    lstm_history = lstm_model.fit(
        X_seq_train_scaled, y_seq_train_cat,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    lstm_test_loss, lstm_test_acc = lstm_model.evaluate(X_seq_test_scaled, y_seq_test_cat, verbose=0)

    rnn_results[rnn_type] = {
        'model': lstm_model,
        'history': lstm_history,
        'test_acc': lstm_test_acc,
        'test_loss': lstm_test_loss
    }

    print(f"  Test Accuracy: {lstm_test_acc:.4f}")
    print(f"  Test Loss: {lstm_test_loss:.4f}")

# CHALLENGE 5: NEURAL NETWORK OPTIMIZATION AND ANALYSIS
print("\n" + "=" * 60)
print("📊 CHALLENGE 5: NEURAL NETWORK OPTIMIZATION")
print("=" * 60)

print("🔧 Advanced Optimization Techniques")

# Learning rate scheduling
def create_optimized_model(input_dim, num_classes):
    """Create an optimized neural network with advanced techniques"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Input layer with noise injection for regularization
        layers.GaussianNoise(0.1),

        # First block
        layers.Dense(256, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Second block with different activation
        layers.Dense(128, activation='elu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Third block with Swish activation
        layers.Dense(64, activation='swish',
                    kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Output layer
        layers.Dense(num_classes, activation='softmax',
                    kernel_initializer='glorot_normal')
    ])

    return model

# Custom learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.9
    else:
        return lr * 0.95

# Train optimized model
optimized_model = create_optimized_model(X_pat_train_scaled.shape[1], 4)

# Use different optimizers
optimizers_to_test = {
    'Adam': optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    'RMSprop': optimizers.RMSprop(learning_rate=0.001),
    'AdamW': optimizers.AdamW(learning_rate=0.001, weight_decay=0.004)
}

optimizer_results = {}

for opt_name, optimizer in optimizers_to_test.items():
    print(f"\n🚀 Testing {opt_name} Optimizer...")

    model = create_optimized_model(X_pat_train_scaled.shape[1], 4)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_pat_train_scaled, y_pat_train_cat,
        batch_size=32,
        epochs=30,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            callbacks.LearningRateScheduler(scheduler),
            callbacks.EarlyStopping(patience=8, restore_best_weights=True)
        ]
    )

    test_loss, test_acc = model.evaluate(X_pat_test_scaled, y_pat_test_cat, verbose=0)

    optimizer_results[opt_name] = {
        'test_acc': test_acc,
        'test_loss': test_loss,
        'final_lr': float(model.optimizer.learning_rate)
    }

    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Final Learning Rate: {model.optimizer.learning_rate:.6f}")

# Visualize all results
plt.figure(figsize=(20, 16))

# MLP Architecture Comparison
plt.subplot(3, 4, 1)
mlp_names = list(mlp_results.keys())
mlp_scores = [mlp_results[name]['test_f1'] for name in mlp_names]
mlp_layers = [mlp_results[name]['n_layers'] for name in mlp_names]

plt.scatter(mlp_layers, mlp_scores, s=100, alpha=0.7)
for i, name in enumerate(mlp_names):
    plt.annotate(name, (mlp_layers[i], mlp_scores[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
plt.xlabel('Number of Layers')
plt.ylabel('Test F1 Score')
plt.title('MLP Architecture vs Performance')
plt.grid(True, alpha=0.3)

# Activation Function Comparison
plt.subplot(3, 4, 2)
activations = list(activation_results.keys())
act_scores = list(activation_results.values())
plt.bar(activations, act_scores, alpha=0.7, color='skyblue')
plt.ylabel('Test F1 Score')
plt.title('Activation Function Comparison')
plt.grid(axis='y', alpha=0.3)

# Keras Architecture Comparison
plt.subplot(3, 4, 3)
keras_names = list(keras_results.keys())
keras_scores = [keras_results[name]['test_acc'] for name in keras_names]
plt.bar(keras_names, keras_scores, alpha=0.7, color='lightgreen')
plt.ylabel('Test Accuracy')
plt.title('Keras Architecture Comparison')
plt.grid(axis='y', alpha=0.3)

# Training History for Best Keras Model
plt.subplot(3, 4, 4)
best_keras = max(keras_results.keys(), key=lambda k: keras_results[k]['test_acc'])
history = keras_results[best_keras]['history']
plt.plot(history.history['accuracy'], label='Training Accuracy', alpha=0.8)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Training History: {best_keras.capitalize()}')
plt.legend()
plt.grid(True, alpha=0.3)

# CNN Performance
plt.subplot(3, 4, 5)
cnn_acc_history = cnn_history.history['accuracy']
cnn_val_acc_history = cnn_history.history['val_accuracy']
plt.plot(cnn_acc_history, label='Training', alpha=0.8)
plt.plot(cnn_val_acc_history, label='Validation', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

# RNN Comparison
plt.subplot(3, 4, 6)
rnn_names = list(rnn_results.keys())
rnn_scores = [rnn_results[name]['test_acc'] for name in rnn_names]
plt.bar(rnn_names, rnn_scores, alpha=0.7, color='coral')
plt.ylabel('Test Accuracy')
plt.title('RNN Architecture Comparison')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Optimizer Comparison
plt.subplot(3, 4, 7)
opt_names = list(optimizer_results.keys())
opt_scores = [optimizer_results[name]['test_acc'] for name in opt_names]
plt.bar(opt_names, opt_scores, alpha=0.7, color='gold')
plt.ylabel('Test Accuracy')
plt.title('Optimizer Comparison')
plt.grid(axis='y', alpha=0.3)

# Learning Rate Evolution
plt.subplot(3, 4, 8)
# Simulate learning rate schedule
epochs = np.arange(30)
lrs = [scheduler(epoch, 0.001) for epoch in epochs]
plt.plot(epochs, lrs, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)

# Model Complexity vs Performance
plt.subplot(3, 4, 9)
model_names = ['Small MLP', 'Medium MLP', 'Deep CNN', 'Simple LSTM', 'Stacked LSTM']
complexity_scores = [0.75, 0.82, 0.89, 0.78, 0.85]  # Simulated complexity scores
performance_scores = [mlp_results['Small MLP']['test_f1'],
                     mlp_results['Medium MLP']['test_f1'],
                     cnn_test_acc,
                     rnn_results['simple']['test_acc'],
                     rnn_results['stacked']['test_acc']]

plt.scatter(complexity_scores, performance_scores, s=100, alpha=0.7)
for i, name in enumerate(model_names):
    plt.annotate(name, (complexity_scores[i], performance_scores[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
plt.xlabel('Model Complexity (Relative)')
plt.ylabel('Test Performance')
plt.title('Complexity vs Performance Trade-off')
plt.grid(True, alpha=0.3)

# Loss Comparison
plt.subplot(3, 4, 10)
loss_comparison = {
    'Dense (Medium)': keras_results['medium']['test_loss'],
    'CNN': cnn_test_loss,
    'Simple LSTM': rnn_results['simple']['test_loss'],
    'Regularized': reg_test_loss
}
plt.bar(loss_comparison.keys(), loss_comparison.values(), alpha=0.7, color='purple')
plt.ylabel('Test Loss')
plt.title('Model Loss Comparison')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Feature Learning Visualization (CNN filters)
plt.subplot(3, 4, 11)
# Get first layer weights from CNN
if len(cnn_model.layers) > 0 and hasattr(cnn_model.layers[0], 'get_weights'):
    try:
        weights = cnn_model.layers[0].get_weights()[0]
        # Visualize some filters
        n_filters = min(4, weights.shape[-1])
        for i in range(n_filters):
            plt.subplot(3, 4, 11)
            if i == 0:
                plt.imshow(weights[:, :, 0, i], cmap='viridis')
                plt.title('CNN Filter Visualization')
                plt.colorbar()
                break
    except:
        plt.text(0.5, 0.5, 'Filter visualization\nnot available',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('CNN Filters')

# Summary Statistics
plt.subplot(3, 4, 12)
all_neural_scores = (list(mlp_results.values()) +
                    list(keras_results.values()) +
                    list(rnn_results.values()))

# Extract test scores
test_scores = []
for result in all_neural_scores:
    if 'test_f1' in result:
        test_scores.append(result['test_f1'])
    elif 'test_acc' in result:
        test_scores.append(result['test_acc'])

plt.hist(test_scores, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(np.mean(test_scores), color='red', linestyle='--', label=f'Mean: {np.mean(test_scores):.3f}')
plt.xlabel('Test Performance')
plt.ylabel('Frequency')
plt.title('Performance Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("🧠 DEEP LEARNING INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

# Collect best results
best_mlp = max(mlp_results.keys(), key=lambda k: mlp_results[k]['test_f1'])
best_keras = max(keras_results.keys(), key=lambda k: keras_results[k]['test_acc'])
best_rnn = max(rnn_results.keys(), key=lambda k: rnn_results[k]['test_acc'])
best_optimizer = max(optimizer_results.keys(), key=lambda k: optimizer_results[k]['test_acc'])

print("📋 Key Findings:")
print(f"1. Best MLP Architecture: {best_mlp} (F1: {mlp_results[best_mlp]['test_f1']:.4f})")
print(f"2. Best Dense Network: {best_keras.capitalize()} (Acc: {keras_results[best_keras]['test_acc']:.4f})")
print(f"3. CNN Performance: {cnn_test_acc:.4f} accuracy on pattern data")
print(f"4. Best RNN: {best_rnn.capitalize()} LSTM (Acc: {rnn_results[best_rnn]['test_acc']:.4f})")
print(f"5. Best Optimizer: {best_optimizer} (Acc: {optimizer_results[best_optimizer]['test_acc']:.4f})")

print(f"\n🎯 Architecture Recommendations:")
print("• Use CNNs for image-like and spatial pattern data")
print("• Use RNNs/LSTMs for sequential and time-series data")
print("• Use dense networks for tabular data with proper regularization")
print("• Batch normalization and dropout are crucial for deep networks")
print("• Learning rate scheduling improves convergence")

print(f"\n⚙️ Training Best Practices:")
print("• Start with simpler architectures and increase complexity gradually")
print("• Use early stopping to prevent overfitting")
print("• Experiment with different optimizers (Adam, AdamW, RMSprop)")
print("• Apply proper data preprocessing and normalization")
print("• Use appropriate activation functions (ReLU, ELU, Swish)")
print("• Consider ensemble methods for critical applications")

print("\n✅ Deep Learning and Neural Networks Challenge Completed!")
print("What you've mastered:")
print("• Multi-layer perceptrons with scikit-learn")
print("• Deep neural networks with TensorFlow/Keras")
print("• Convolutional Neural Networks for pattern recognition")
print("• Recurrent Neural Networks for sequential data")
print("• Advanced optimization techniques and regularization")
print("• Neural network architecture design and analysis")

print(f"\n🧠 You are now a Neural Network Architect! Ready for advanced analytics!")
```

### Success Criteria

- Build and compare multiple neural network architectures
- Implement CNNs for spatial pattern recognition
- Create RNNs/LSTMs for sequential data processing
- Master advanced optimization and regularization techniques
- Analyze neural network performance and complexity trade-offs
- Develop production-ready deep learning pipelines

### Learning Objectives

- Understand neural network fundamentals and architectures
- Master TensorFlow/Keras for deep learning implementation
- Learn specialized architectures (CNN, RNN, LSTM)
- Practice advanced optimization and regularization techniques
- Develop skills in neural network performance analysis
- Build comprehensive deep learning comparison frameworks

---

_Pro tip: The best neural network is one that balances complexity with generalization - start simple and add complexity only when needed!_
