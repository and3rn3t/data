"""
Level 5 - Challenge 3: Recurrent Neural Networks (RNNs)
======================================================

Master sequential data processing with RNN architectures.
This challenge covers RNN fundamentals, LSTM/GRU networks, and sequence modeling.

Learning Objectives:
- Understand sequential data and temporal dependencies
- Build RNN architectures for sequence tasks
- Learn LSTM and GRU for long-term memory
- Master text processing and time series prediction
- Explore attention mechanisms and transformers

Required Libraries: numpy, matplotlib, collections (optional: tensorflow/pytorch)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


def create_sequence_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Create synthetic sequence datasets for RNN experiments.

    Returns:
        Dictionary containing various sequence datasets for RNN training
    """
    print("🔄 Creating Sequential Datasets for RNN Training...")

    datasets = {}

    # 1. Simple Sequence Classification
    print("Creating sequence classification dataset...")
    n_sequences = 1000
    seq_length = 20

    X_seq_class = []
    y_seq_class = []

    for _ in range(n_sequences):
        # Generate sequences with different patterns
        pattern_type = np.random.choice(
            ["increasing", "decreasing", "oscillating", "random"]
        )

        if pattern_type == "increasing":
            base = np.linspace(0, 1, seq_length)
            noise = np.random.normal(0, 0.1, seq_length)
            sequence = base + noise
            label = 0

        elif pattern_type == "decreasing":
            base = np.linspace(1, 0, seq_length)
            noise = np.random.normal(0, 0.1, seq_length)
            sequence = base + noise
            label = 1

        elif pattern_type == "oscillating":
            t = np.linspace(0, 4 * np.pi, seq_length)
            frequency = np.random.uniform(0.5, 2.0)
            amplitude = np.random.uniform(0.3, 0.7)
            sequence = amplitude * np.sin(frequency * t) + 0.5
            noise = np.random.normal(0, 0.05, seq_length)
            sequence += noise
            label = 2

        else:  # random
            sequence = np.random.uniform(0, 1, seq_length)
            label = 3

        X_seq_class.append(sequence.reshape(-1, 1))
        y_seq_class.append(label)

    datasets["sequence_classification"] = {
        "X": np.array(X_seq_class),
        "y": np.array(y_seq_class),
        "classes": ["Increasing", "Decreasing", "Oscillating", "Random"],
        "description": "Sequential patterns for RNN classification",
    }

    # 2. Time Series Prediction
    print("Creating time series prediction dataset...")
    n_series = 500
    series_length = 50
    prediction_steps = 5

    X_timeseries = []
    y_timeseries = []

    for _ in range(n_series):
        # Generate synthetic time series
        t = np.linspace(0, 10, series_length + prediction_steps)

        # Combine multiple components
        trend = np.random.uniform(-0.1, 0.1) * t
        seasonal = np.random.uniform(0.2, 0.8) * np.sin(2 * np.pi * t / 10)
        noise = np.random.normal(0, 0.1, len(t))

        full_series = trend + seasonal + noise

        # Input: first 50 points, Target: next 5 points
        X_timeseries.append(full_series[:series_length].reshape(-1, 1))
        y_timeseries.append(
            full_series[series_length : series_length + prediction_steps]
        )

    datasets["time_series"] = {
        "X": np.array(X_timeseries),
        "y": np.array(y_timeseries),
        "input_length": series_length,
        "prediction_steps": prediction_steps,
        "description": "Time series for multi-step prediction",
    }

    # 3. Text Sequences (Character-level)
    print("Creating character sequence dataset...")

    # Simple vocabulary
    vocabulary = list("abcdefghijklmnopqrstuvwxyz ")
    vocab_size = len(vocabulary)
    char_to_idx = {char: idx for idx, char in enumerate(vocabulary)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Generate simple text patterns
    text_patterns = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test sequence",
        "machine learning with neural networks",
        "recurrent networks process sequences",
        "lstm and gru handle long dependencies",
    ]

    X_text = []
    y_text = []
    context_length = 10

    for pattern in text_patterns:
        for _ in range(20):  # Multiple variations
            # Add some random variations
            text = pattern
            if np.random.random() > 0.5:
                text = text.replace(" ", "_")

            # Create sequences
            for i in range(len(text) - context_length):
                context = text[i : i + context_length]
                target = text[i + context_length]

                if all(c in char_to_idx for c in context + target):
                    context_indices = [char_to_idx[c] for c in context]
                    target_idx = char_to_idx[target]

                    X_text.append(context_indices)
                    y_text.append(target_idx)

    datasets["text_sequences"] = {
        "X": np.array(X_text),
        "y": np.array(y_text),
        "vocabulary": vocabulary,
        "vocab_size": vocab_size,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "context_length": context_length,
        "description": "Character-level text prediction",
    }

    print(f"Created {len(datasets)} sequence datasets")
    return datasets


def demonstrate_vanilla_rnn() -> None:
    """
    Demonstrate basic RNN operations and computations.
    """
    print("\n🧠 Vanilla RNN Demonstration")
    print("=" * 50)

    # RNN parameters
    input_size = 3
    hidden_size = 4
    sequence_length = 5

    # Initialize weights (normally distributed)
    W_input = np.random.normal(0, 0.1, (hidden_size, input_size))
    W_hidden = np.random.normal(0, 0.1, (hidden_size, hidden_size))
    bias = np.zeros((hidden_size, 1))

    # Sample input sequence
    X = np.random.randn(input_size, sequence_length)

    print(f"RNN Configuration:")
    print(f"• Input size: {input_size}")
    print(f"• Hidden size: {hidden_size}")
    print(f"• Sequence length: {sequence_length}")
    print(f"• Weight matrices: W_input {W_input.shape}, W_hidden {W_hidden.shape}")

    # Forward pass through RNN
    hidden_states = []
    h = np.zeros((hidden_size, 1))  # Initial hidden state

    print(f"\nForward Pass:")
    for t in range(sequence_length):
        x_t = X[:, t : t + 1]  # Input at time t

        # RNN computation: h_t = tanh(W_input * x_t + W_hidden * h_{t-1} + bias)
        linear_output = np.dot(W_input, x_t) + np.dot(W_hidden, h) + bias
        h = np.tanh(linear_output)

        hidden_states.append(h.copy())

        print(f"Time step {t+1}:")
        print(f"  Input: {x_t.flatten()}")
        print(f"  Hidden state: {h.flatten()}")
        print(f"  Hidden magnitude: {np.linalg.norm(h):.3f}")

    # Visualize hidden state evolution
    hidden_matrix = np.hstack(hidden_states)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot input sequence
    ax1.imshow(X, aspect="auto", cmap="RdBu", interpolation="nearest")
    ax1.set_title("Input Sequence")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Input Features")
    ax1.set_xticks(range(sequence_length))
    ax1.set_xticklabels(range(1, sequence_length + 1))

    # Plot hidden states
    ax2.imshow(hidden_matrix, aspect="auto", cmap="RdBu", interpolation="nearest")
    ax2.set_title("Hidden State Evolution")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Hidden Units")
    ax2.set_xticks(range(sequence_length))
    ax2.set_xticklabels(range(1, sequence_length + 1))

    plt.tight_layout()
    plt.show()

    # Analyze gradient flow (simplified)
    print(f"\n🔍 RNN Analysis:")
    print(
        f"• Initial hidden state norm: {np.linalg.norm(np.zeros((hidden_size, 1))):.3f}"
    )
    print(f"• Final hidden state norm: {np.linalg.norm(h):.3f}")
    print(f"• Hidden state variance: {np.var(hidden_matrix):.3f}")

    # Demonstrate vanishing gradient problem
    print(f"\n⚠️ Gradient Flow Analysis:")
    gradient_norms = []

    for t in range(sequence_length):
        # Simulate gradient magnitude (derivative of tanh)
        h_t = hidden_states[t]
        tanh_derivative = 1 - np.tanh(h_t) ** 2
        gradient_norm = np.mean(tanh_derivative)
        gradient_norms.append(gradient_norm)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, sequence_length + 1), gradient_norms, "ro-")
    plt.title("Gradient Flow Through Time (Simplified)")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Gradient Magnitude")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"• Average gradient magnitude: {np.mean(gradient_norms):.3f}")
    print(f"• This demonstrates potential vanishing gradients in long sequences")


def demonstrate_lstm_concepts() -> None:
    """
    Demonstrate LSTM concepts and gating mechanisms.
    """
    print("\n🔐 LSTM Gating Mechanisms")
    print("=" * 50)

    # LSTM components
    input_size = 2
    hidden_size = 3

    # Sample input
    x = np.random.randn(input_size, 1)
    h_prev = np.random.randn(hidden_size, 1)
    c_prev = np.random.randn(hidden_size, 1)

    # LSTM weights (simplified - normally these are larger matrices)
    W_f = np.random.normal(
        0, 0.1, (hidden_size, input_size + hidden_size)
    )  # Forget gate
    W_i = np.random.normal(
        0, 0.1, (hidden_size, input_size + hidden_size)
    )  # Input gate
    W_c = np.random.normal(0, 0.1, (hidden_size, input_size + hidden_size))  # Candidate
    W_o = np.random.normal(
        0, 0.1, (hidden_size, input_size + hidden_size)
    )  # Output gate

    # Biases
    b_f = np.zeros((hidden_size, 1))
    b_i = np.zeros((hidden_size, 1))
    b_c = np.zeros((hidden_size, 1))
    b_o = np.zeros((hidden_size, 1))

    print(f"LSTM Configuration:")
    print(f"• Input size: {input_size}")
    print(f"• Hidden size: {hidden_size}")
    print(f"• Previous hidden state: {h_prev.flatten()}")
    print(f"• Previous cell state: {c_prev.flatten()}")
    print(f"• Current input: {x.flatten()}")

    # Concatenate input and previous hidden state
    concat_input = np.vstack([x, h_prev])

    # LSTM forward pass
    print(f"\nLSTM Gates Computation:")

    # Forget gate
    f_gate = 1 / (1 + np.exp(-(np.dot(W_f, concat_input) + b_f)))
    print(f"1. Forget gate: {f_gate.flatten()} (what to forget from cell state)")

    # Input gate
    i_gate = 1 / (1 + np.exp(-(np.dot(W_i, concat_input) + b_i)))
    print(f"2. Input gate: {i_gate.flatten()} (what new info to store)")

    # Candidate values
    c_candidate = np.tanh(np.dot(W_c, concat_input) + b_c)
    print(f"3. Candidate: {c_candidate.flatten()} (new candidate values)")

    # Update cell state
    c_new = f_gate * c_prev + i_gate * c_candidate
    print(f"4. New cell state: {c_new.flatten()}")

    # Output gate
    o_gate = 1 / (1 + np.exp(-(np.dot(W_o, concat_input) + b_o)))
    print(f"5. Output gate: {o_gate.flatten()} (what to output)")

    # New hidden state
    h_new = o_gate * np.tanh(c_new)
    print(f"6. New hidden state: {h_new.flatten()}")

    # Visualize gates
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    gates_data = [
        ("Forget Gate", f_gate, "What to forget"),
        ("Input Gate", i_gate, "What to remember"),
        ("Output Gate", o_gate, "What to output"),
        ("Candidate", c_candidate, "New information"),
        ("Cell State", c_new, "Memory storage"),
        ("Hidden State", h_new, "Output activation"),
    ]

    for idx, (name, values, description) in enumerate(gates_data):
        row = idx // 3
        col = idx % 3

        bars = axes[row, col].bar(range(len(values)), values.flatten())
        axes[row, col].set_title(f"{name}\n{description}")
        axes[row, col].set_ylabel("Activation")
        axes[row, col].set_ylim(-1.1, 1.1)
        axes[row, col].grid(True, alpha=0.3)

        # Color code bars
        for i, bar in enumerate(bars):
            if name in ["Forget Gate", "Input Gate", "Output Gate"]:
                bar.set_color("lightblue")  # Gates are 0-1
            else:
                bar.set_color("lightgreen")  # States can be negative

    plt.tight_layout()
    plt.suptitle("LSTM Gate Activations", y=1.02, fontsize=16)
    plt.show()

    print(f"\n🎯 LSTM Key Concepts:")
    print(f"• Forget gate controls what to discard from previous cell state")
    print(f"• Input gate controls what new information to store")
    print(f"• Output gate controls what to output based on cell state")
    print(f"• Cell state maintains long-term memory")
    print(f"• Hidden state is the filtered output for current time step")

    print(f"\n✨ LSTM Benefits:")
    print(f"• Solves vanishing gradient problem")
    print(f"• Can learn long-term dependencies")
    print(f"• Selective memory through gating")


def sequence_prediction_demo() -> None:
    """
    Demonstrate sequence prediction with a simple RNN-like model.
    """
    print("\n📈 Sequence Prediction Demo")
    print("=" * 50)

    # Generate a predictable sequence
    sequence_length = 100
    prediction_length = 20

    # Create a complex pattern
    t = np.linspace(0, 8 * np.pi, sequence_length + prediction_length)
    pattern1 = np.sin(t)
    pattern2 = 0.5 * np.sin(2 * t)
    pattern3 = 0.3 * np.cos(0.5 * t)

    full_sequence = pattern1 + pattern2 + pattern3
    noise = np.random.normal(0, 0.1, len(full_sequence))
    full_sequence += noise

    # Split into training and prediction parts
    train_sequence = full_sequence[:sequence_length]
    true_prediction = full_sequence[sequence_length:]

    # Simple sequence prediction using linear model (simulating RNN)
    window_size = 10

    # Prepare training data
    X_train = []
    y_train = []

    for i in range(window_size, len(train_sequence)):
        X_train.append(train_sequence[i - window_size : i])
        y_train.append(train_sequence[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Simple linear regression for prediction (simulating trained RNN)
    # In practice, this would be an RNN with learned weights
    coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

    # Make predictions
    predictions = []
    current_window = train_sequence[-window_size:].copy()

    for _ in range(prediction_length):
        # Predict next value
        next_value = np.dot(current_window, coefficients)
        predictions.append(next_value)

        # Update window (slide forward)
        current_window = np.roll(current_window, -1)
        current_window[-1] = next_value

    predictions = np.array(predictions)

    # Calculate prediction metrics
    mse = np.mean((true_prediction - predictions) ** 2)
    mae = np.mean(np.abs(true_prediction - predictions))

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Full sequence view
    time_train = np.arange(len(train_sequence))
    time_pred = np.arange(len(train_sequence), len(train_sequence) + prediction_length)

    ax1.plot(time_train, train_sequence, "b-", label="Training Data", linewidth=2)
    ax1.plot(time_pred, true_prediction, "g-", label="True Future", linewidth=2)
    ax1.plot(time_pred, predictions, "r--", label="Predictions", linewidth=2)
    ax1.axvline(
        len(train_sequence),
        color="black",
        linestyle=":",
        alpha=0.7,
        label="Prediction Start",
    )
    ax1.set_title("Sequence Prediction Results")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Prediction accuracy view
    ax2.plot(true_prediction, "g-", label="True Values", linewidth=2)
    ax2.plot(predictions, "r--", label="Predictions", linewidth=2)
    ax2.fill_between(
        range(len(predictions)),
        true_prediction - np.abs(true_prediction - predictions),
        true_prediction + np.abs(true_prediction - predictions),
        alpha=0.2,
        color="red",
        label="Prediction Error",
    )
    ax2.set_title("Prediction vs Truth (Zoomed)")
    ax2.set_xlabel("Prediction Steps")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n📊 Prediction Results:")
    print(f"• Training sequence length: {len(train_sequence)}")
    print(f"• Prediction length: {prediction_length}")
    print(f"• Window size: {window_size}")
    print(f"• Mean Squared Error: {mse:.4f}")
    print(f"• Mean Absolute Error: {mae:.4f}")

    # Show prediction breakdown
    print(f"\n🔍 Step-by-step Analysis:")
    for i in range(min(5, prediction_length)):
        error = abs(true_prediction[i] - predictions[i])
        print(
            f"Step {i+1}: True={true_prediction[i]:.3f}, Pred={predictions[i]:.3f}, Error={error:.3f}"
        )

    if prediction_length > 5:
        print(f"... (showing first 5 of {prediction_length} predictions)")


def text_generation_demo() -> None:
    """
    Demonstrate simple text generation with character-level RNN concepts.
    """
    print("\n📝 Text Generation Demo")
    print("=" * 50)

    # Simple text corpus
    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
        "the fish swam in the sea",
        "the fox jumped over the fence",
    ]

    # Build vocabulary
    all_chars = set()
    for text in texts:
        all_chars.update(text)

    vocabulary = sorted(list(all_chars))
    vocab_size = len(vocabulary)
    char_to_idx = {char: idx for idx, char in enumerate(vocabulary)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print(f"Vocabulary: {vocabulary}")
    print(f"Vocabulary size: {vocab_size}")

    # Build character transition matrix (simple n-gram model)
    context_length = 3
    transitions = defaultdict(lambda: defaultdict(int))

    # Count transitions
    for text in texts:
        for i in range(len(text) - context_length):
            context = text[i : i + context_length]
            next_char = text[i + context_length]
            transitions[context][next_char] += 1

    # Convert to probabilities
    for context in transitions:
        total = sum(transitions[context].values())
        for next_char in transitions[context]:
            transitions[context][next_char] /= total

    print(f"\nLearned {len(transitions)} context patterns")
    print("Sample transitions:")
    for i, (context, next_chars) in enumerate(list(transitions.items())[:5]):
        print(f"  '{context}' → {dict(next_chars)}")

    # Text generation function
    def generate_text(seed: str, length: int = 50) -> str:
        """Generate text using learned transitions."""
        if len(seed) < context_length:
            seed = seed.ljust(context_length)

        generated = seed
        current_context = seed[-context_length:]

        for _ in range(length):
            if current_context in transitions:
                # Get possible next characters and their probabilities
                possible_chars = list(transitions[current_context].keys())
                probabilities = list(transitions[current_context].values())

                # Sample next character
                if possible_chars:
                    next_char = np.random.choice(possible_chars, p=probabilities)
                    generated += next_char
                    current_context = current_context[1:] + next_char
                else:
                    # If no learned transition, pick random character
                    next_char = np.random.choice(vocabulary)
                    generated += next_char
                    current_context = current_context[1:] + next_char
            else:
                # Unknown context, pick random character
                next_char = np.random.choice(vocabulary)
                generated += next_char
                current_context = current_context[1:] + next_char

        return generated

    # Generate samples
    print(f"\n🎭 Generated Text Samples:")
    seeds = ["the", "cat", "ran"]

    for seed in seeds:
        generated = generate_text(seed, length=30)
        print(f"Seed: '{seed}' → '{generated}'")

    # Analyze character frequency
    char_counts = defaultdict(int)
    for text in texts:
        for char in text:
            char_counts[char] += 1

    # Visualize character frequency
    chars = list(char_counts.keys())
    counts = list(char_counts.values())

    plt.figure(figsize=(12, 4))
    bars = plt.bar(chars, counts)
    plt.title("Character Frequency in Training Data")
    plt.xlabel("Characters")
    plt.ylabel("Frequency")

    # Highlight space character
    space_idx = chars.index(" ") if " " in chars else -1
    if space_idx >= 0:
        bars[space_idx].set_color("red")
        bars[space_idx].set_label("Space")
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\n📊 Text Analysis:")
    print(f"• Total unique characters: {len(chars)}")
    print(f"• Most common character: '{max(char_counts, key=char_counts.get)}'")
    print(f"• Context patterns learned: {len(transitions)}")
    print(
        f"• Average characters per pattern: {np.mean([len(transitions[ctx]) for ctx in transitions]):.1f}"
    )


def run_rnn_challenges() -> None:
    """
    Run all RNN challenges with comprehensive analysis.
    """
    print("🚀 Starting Level 5 Challenge 3: Recurrent Neural Networks")
    print("=" * 60)

    try:
        # Challenge 1: Create sequence datasets
        print("\n" + "=" * 50)
        print("CHALLENGE 1: Sequential Dataset Creation")
        print("=" * 50)

        datasets = create_sequence_datasets()

        print(f"\n✅ Created {len(datasets)} sequence datasets:")
        for name, data in datasets.items():
            if "X" in data and hasattr(data["X"], "shape"):
                print(f"• {name}: {data['X'].shape[0]} sequences")
                if len(data["X"].shape) > 2:
                    print(f"  Sequence shape: {data['X'].shape[1:]} per sequence")
                print(f"  Description: {data['description']}")

        # Challenge 2: Vanilla RNN demonstration
        print("\n" + "=" * 50)
        print("CHALLENGE 2: Vanilla RNN Mechanics")
        print("=" * 50)

        demonstrate_vanilla_rnn()

        # Challenge 3: LSTM concepts
        print("\n" + "=" * 50)
        print("CHALLENGE 3: LSTM Gating Mechanisms")
        print("=" * 50)

        demonstrate_lstm_concepts()

        # Challenge 4: Sequence prediction
        print("\n" + "=" * 50)
        print("CHALLENGE 4: Sequence Prediction")
        print("=" * 50)

        sequence_prediction_demo()

        # Challenge 5: Text generation
        print("\n" + "=" * 50)
        print("CHALLENGE 5: Text Generation")
        print("=" * 50)

        text_generation_demo()

        print("\n" + "🎉" * 20)
        print("LEVEL 5 CHALLENGE 3 COMPLETE!")
        print("🎉" * 20)

        print("\n📚 What You've Learned:")
        print("• RNN fundamentals: sequential processing, hidden states")
        print("• Vanilla RNN mechanics: forward pass, gradient flow")
        print("• LSTM architecture: gates, cell states, memory")
        print("• Sequence modeling: time series prediction, pattern recognition")
        print("• Text processing: character-level modeling, generation")
        print("• Temporal dependencies: short and long-term memory")

        print("\n🚀 Next Steps:")
        print("• Implement RNNs with TensorFlow/PyTorch")
        print("• Explore GRU (Gated Recurrent Units)")
        print("• Try bidirectional RNNs")
        print("• Learn about attention mechanisms")
        print("• Move to Level 5 Challenge 4: Advanced Architectures")

        return datasets

    except Exception as e:
        print(f"❌ Error in RNN challenges: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete RNN challenge
    datasets = run_rnn_challenges()

    if datasets:
        print("\n" + "=" * 60)
        print("RNN CHALLENGE SUMMARY")
        print("=" * 60)

        print("\nDatasets Created:")
        for name, data in datasets.items():
            if "X" in data and hasattr(data["X"], "shape"):
                print(f"• {name}: {data['X'].shape}")

        print("\nKey RNN Concepts Covered:")
        concepts = [
            "Sequential data processing and temporal patterns",
            "Vanilla RNN architecture and computations",
            "Hidden state evolution and memory",
            "LSTM gating mechanisms and cell states",
            "Sequence prediction and forecasting",
            "Text generation and language modeling",
            "Gradient flow and vanishing gradients",
        ]

        for i, concept in enumerate(concepts, 1):
            print(f"{i}. {concept}")

        print("\n✨ Ready for Level 5 Challenge 4: Advanced Architectures!")
