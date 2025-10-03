"""
Level 5 - Challenge 2: Convolutional Neural Networks (CNNs)
==========================================================

Master image processing and computer vision with CNN architectures.
This challenge covers CNN fundamentals, image classification, and visual understanding.

Learning Objectives:
- Understand convolution operations and filters
- Build CNN architectures for image tasks
- Learn pooling, activation, and regularization
- Explore transfer learning and pre-trained models
- Master image augmentation and preprocessing

Required Libraries: numpy, matplotlib, scikit-image (optional: tensorflow/pytorch)
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


def create_image_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Create synthetic image datasets for CNN experiments.

    Returns:
        Dictionary containing various image datasets for CNN training
    """
    print("🖼️ Creating Image Datasets for CNN Training...")

    datasets = {}

    # 1. Simple Pattern Recognition Dataset
    print("Creating pattern recognition dataset...")
    n_samples = 1000
    image_size = 28

    # Create simple geometric patterns
    X_patterns = []
    y_patterns = []

    for i in range(n_samples):
        img = np.zeros((image_size, image_size))
        pattern_type = np.random.randint(0, 4)

        if pattern_type == 0:  # Circle
            center = image_size // 2
            radius = np.random.randint(5, 10)
            y, x = np.ogrid[:image_size, :image_size]
            mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
            img[mask] = 1.0

        elif pattern_type == 1:  # Square
            size = np.random.randint(8, 16)
            start = (image_size - size) // 2
            img[start : start + size, start : start + size] = 1.0

        elif pattern_type == 2:  # Cross
            center = image_size // 2
            thickness = 2
            img[center - thickness : center + thickness, :] = 1.0
            img[:, center - thickness : center + thickness] = 1.0

        else:  # Triangle
            for row in range(image_size // 2, image_size):
                width = (row - image_size // 2) * 2
                start = (image_size - width) // 2
                img[row, start : start + width] = 1.0

        # Add noise
        noise = np.random.normal(0, 0.1, img.shape)
        img = np.clip(img + noise, 0, 1)

        X_patterns.append(img.reshape(image_size, image_size, 1))
        y_patterns.append(pattern_type)

    datasets["patterns"] = {
        "X": np.array(X_patterns),
        "y": np.array(y_patterns),
        "classes": ["Circle", "Square", "Cross", "Triangle"],
        "description": "Simple geometric patterns for CNN classification",
    }

    # 2. Texture Classification Dataset
    print("Creating texture dataset...")
    n_texture_samples = 800
    texture_size = 32

    X_textures = []
    y_textures = []

    for i in range(n_texture_samples):
        texture_type = np.random.randint(0, 3)
        img = np.zeros((texture_size, texture_size))

        if texture_type == 0:  # Horizontal stripes
            stripe_width = np.random.randint(2, 6)
            for y in range(0, texture_size, stripe_width * 2):
                img[y : y + stripe_width, :] = 1.0

        elif texture_type == 1:  # Vertical stripes
            stripe_width = np.random.randint(2, 6)
            for x in range(0, texture_size, stripe_width * 2):
                img[:, x : x + stripe_width] = 1.0

        else:  # Checkerboard
            block_size = np.random.randint(3, 8)
            for y in range(0, texture_size, block_size):
                for x in range(0, texture_size, block_size):
                    if (y // block_size + x // block_size) % 2 == 0:
                        img[y : y + block_size, x : x + block_size] = 1.0

        # Add noise and normalize
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)

        X_textures.append(img.reshape(texture_size, texture_size, 1))
        y_textures.append(texture_type)

    datasets["textures"] = {
        "X": np.array(X_textures),
        "y": np.array(y_textures),
        "classes": ["Horizontal", "Vertical", "Checkerboard"],
        "description": "Texture patterns for CNN feature learning",
    }

    # 3. Edge Detection Dataset
    print("Creating edge detection dataset...")
    n_edge_samples = 600
    edge_size = 24

    X_edges = []
    y_edges = []

    for i in range(n_edge_samples):
        img = np.random.normal(0.5, 0.1, (edge_size, edge_size))
        edge_type = np.random.randint(0, 4)

        if edge_type == 0:  # Horizontal edge
            split = np.random.randint(edge_size // 4, 3 * edge_size // 4)
            img[:split, :] = np.random.normal(0.2, 0.05, img[:split, :].shape)
            img[split:, :] = np.random.normal(0.8, 0.05, img[split:, :].shape)

        elif edge_type == 1:  # Vertical edge
            split = np.random.randint(edge_size // 4, 3 * edge_size // 4)
            img[:, :split] = np.random.normal(0.2, 0.05, img[:, :split].shape)
            img[:, split:] = np.random.normal(0.8, 0.05, img[:, split:].shape)

        elif edge_type == 2:  # Diagonal edge (top-left to bottom-right)
            for y in range(edge_size):
                for x in range(edge_size):
                    if y > x:
                        img[y, x] = np.random.normal(0.8, 0.05)
                    else:
                        img[y, x] = np.random.normal(0.2, 0.05)

        else:  # No edge (random texture)
            img = np.random.normal(0.5, 0.15, (edge_size, edge_size))

        img = np.clip(img, 0, 1)
        X_edges.append(img.reshape(edge_size, edge_size, 1))
        y_edges.append(edge_type)

    datasets["edges"] = {
        "X": np.array(X_edges),
        "y": np.array(y_edges),
        "classes": ["Horizontal", "Vertical", "Diagonal", "No Edge"],
        "description": "Edge detection patterns for filter learning",
    }

    print(f"Created {len(datasets)} image datasets")
    return datasets


def cnn_filter_visualization() -> None:
    """
    Demonstrate CNN filters and convolution operations.
    """
    print("\n🔍 CNN Filter Visualization")
    print("=" * 50)

    # Create sample image
    img_size = 20
    sample_img = np.zeros((img_size, img_size))

    # Add some features
    sample_img[5:15, 8:12] = 1.0  # Vertical bar
    sample_img[8:12, 5:15] = 1.0  # Horizontal bar (creates cross)

    # Define common CNN filters
    filters = {
        "Vertical Edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        "Horizontal Edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        "Diagonal Edge": np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
        "Blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(sample_img, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Apply each filter
    for idx, (name, filter_kernel) in enumerate(filters.items()):
        if idx >= 6:  # Only show first 6 filters
            break

        # Simple convolution implementation
        output = np.zeros((img_size - 2, img_size - 2))
        for i in range(img_size - 2):
            for j in range(img_size - 2):
                output[i, j] = np.sum(sample_img[i : i + 3, j : j + 3] * filter_kernel)

        row = (idx + 1) // 4
        col = (idx + 1) % 4

        axes[row, col].imshow(output, cmap="gray")
        axes[row, col].set_title(name)
        axes[row, col].axis("off")

    # Hide unused subplot
    if len(filters) < 7:
        axes[1, 3].axis("off")

    plt.tight_layout()
    plt.suptitle("CNN Filter Operations", y=1.02, fontsize=16)
    plt.show()

    print("\n📊 Filter Analysis:")
    for name, kernel in filters.items():
        print(f"{name}:")
        print(f"  Purpose: {get_filter_purpose(name)}")
        print(f"  Kernel shape: {kernel.shape}")
        print(f"  Sum: {np.sum(kernel):.2f}")
        print()


def get_filter_purpose(filter_name: str) -> str:
    """Get the purpose description for a filter."""
    purposes = {
        "Vertical Edge": "Detects vertical edges and lines",
        "Horizontal Edge": "Detects horizontal edges and lines",
        "Diagonal Edge": "Detects diagonal edges and corners",
        "Blur": "Smooths and reduces noise",
        "Sharpen": "Enhances edges and details",
        "Emboss": "Creates 3D embossed effect",
    }
    return purposes.get(filter_name, "Custom filter operation")


def simulate_cnn_architecture() -> None:
    """
    Simulate a CNN architecture and show feature maps.
    """
    print("\n🏗️ CNN Architecture Simulation")
    print("=" * 50)

    # Create a more complex test image
    img_size = 32
    test_img = np.zeros((img_size, img_size))

    # Add multiple features
    test_img[8:24, 10:12] = 1.0  # Vertical line
    test_img[10:12, 8:24] = 1.0  # Horizontal line
    test_img[20:28, 20:28] = 0.5  # Square
    test_img[4:8, 4:8] = 1.0  # Small square

    # Add some noise
    noise = np.random.normal(0, 0.05, test_img.shape)
    test_img = np.clip(test_img + noise, 0, 1)

    print("Simulating CNN layers:")

    # Layer 1: Convolution + ReLU
    conv1_filters = 8
    conv1_size = 3
    stride = 1

    after_conv1_size = (img_size - conv1_size) // stride + 1
    conv1_output = np.random.rand(conv1_filters, after_conv1_size, after_conv1_size)

    # Apply ReLU simulation
    conv1_output = np.maximum(0, conv1_output - 0.3)

    print(
        f"Conv Layer 1: {img_size}x{img_size}x1 → {after_conv1_size}x{after_conv1_size}x{conv1_filters}"
    )

    # Layer 2: Max Pooling
    pool_size = 2
    after_pool1_size = after_conv1_size // pool_size

    pool1_output = np.zeros((conv1_filters, after_pool1_size, after_pool1_size))
    for f in range(conv1_filters):
        for i in range(after_pool1_size):
            for j in range(after_pool1_size):
                pool1_output[f, i, j] = np.max(
                    conv1_output[
                        f,
                        i * pool_size : (i + 1) * pool_size,
                        j * pool_size : (j + 1) * pool_size,
                    ]
                )

    print(
        f"Max Pool 1:   {after_conv1_size}x{after_conv1_size}x{conv1_filters} → {after_pool1_size}x{after_pool1_size}x{conv1_filters}"
    )

    # Layer 3: Second Convolution
    conv2_filters = 16
    after_conv2_size = (after_pool1_size - conv1_size) // stride + 1

    conv2_output = np.random.rand(conv2_filters, after_conv2_size, after_conv2_size)
    conv2_output = np.maximum(0, conv2_output - 0.2)

    print(
        f"Conv Layer 2: {after_pool1_size}x{after_pool1_size}x{conv1_filters} → {after_conv2_size}x{after_conv2_size}x{conv2_filters}"
    )

    # Layer 4: Global Average Pooling
    global_pool_output = np.mean(conv2_output, axis=(1, 2))

    print(
        f"Global Pool:  {after_conv2_size}x{after_conv2_size}x{conv2_filters} → {conv2_filters}"
    )

    # Layer 5: Fully Connected
    num_classes = 4
    fc_weights = np.random.randn(conv2_filters, num_classes) * 0.1
    fc_output = np.dot(global_pool_output, fc_weights)

    # Apply softmax
    exp_output = np.exp(fc_output - np.max(fc_output))
    softmax_output = exp_output / np.sum(exp_output)

    print(f"Fully Conn:   {conv2_filters} → {num_classes}")

    # Visualize architecture
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Original image
    axes[0, 0].imshow(test_img, cmap="gray")
    axes[0, 0].set_title("Input\n32x32x1")
    axes[0, 0].axis("off")

    # Conv1 feature maps (show first 3)
    for i in range(min(3, conv1_filters)):
        axes[0, 1 + i].imshow(conv1_output[i], cmap="viridis")
        axes[0, 1 + i].set_title(f"Conv1 F{i+1}\n{after_conv1_size}x{after_conv1_size}")
        axes[0, 1 + i].axis("off")

    # Pool1 feature maps (show first 3)
    for i in range(min(3, conv1_filters)):
        axes[1, i].imshow(pool1_output[i], cmap="viridis")
        axes[1, i].set_title(f"Pool1 F{i+1}\n{after_pool1_size}x{after_pool1_size}")
        axes[1, i].axis("off")

    # Final prediction
    class_names = ["Circle", "Square", "Cross", "Triangle"]
    predicted_class = np.argmax(softmax_output)

    axes[1, 3].bar(range(num_classes), softmax_output)
    axes[1, 3].set_xticks(range(num_classes))
    axes[1, 3].set_xticklabels(class_names, rotation=45)
    axes[1, 3].set_title(f"Output\nPredicted: {class_names[predicted_class]}")
    axes[1, 3].set_ylabel("Probability")

    plt.tight_layout()
    plt.suptitle("CNN Architecture Flow", y=1.02, fontsize=16)
    plt.show()

    print("\n📈 Architecture Analysis:")
    print(
        f"Total parameters: ~{estimate_parameters(img_size, conv1_filters, conv2_filters, num_classes):,}"
    )
    print(
        f"Memory usage: ~{estimate_memory_usage(img_size, conv1_filters, conv2_filters):.1f} MB"
    )
    print(
        f"Predicted class: {class_names[predicted_class]} ({softmax_output[predicted_class]:.2%} confidence)"
    )


def estimate_parameters(
    input_size: int, conv1_filters: int, conv2_filters: int, num_classes: int
) -> int:
    """Estimate total parameters in the CNN."""
    # Conv1: 3x3 filters + bias
    conv1_params = (3 * 3 * 1 * conv1_filters) + conv1_filters

    # Conv2: 3x3 filters + bias
    conv2_params = (3 * 3 * conv1_filters * conv2_filters) + conv2_filters

    # FC layer
    fc_params = (conv2_filters * num_classes) + num_classes

    return conv1_params + conv2_params + fc_params


def estimate_memory_usage(
    input_size: int, conv1_filters: int, conv2_filters: int
) -> float:
    """Estimate memory usage in MB (rough approximation)."""
    # Assume 32-bit floats (4 bytes each)
    bytes_per_element = 4

    # Input
    input_memory = input_size * input_size * bytes_per_element

    # Conv1 output
    conv1_size = input_size - 2  # 3x3 conv with no padding
    conv1_memory = conv1_size * conv1_size * conv1_filters * bytes_per_element

    # Pool1 output
    pool1_size = conv1_size // 2
    pool1_memory = pool1_size * pool1_size * conv1_filters * bytes_per_element

    # Conv2 output
    conv2_size = pool1_size - 2
    conv2_memory = conv2_size * conv2_size * conv2_filters * bytes_per_element

    total_bytes = input_memory + conv1_memory + pool1_memory + conv2_memory
    return total_bytes / (1024 * 1024)  # Convert to MB


def image_augmentation_demo() -> None:
    """
    Demonstrate image augmentation techniques for CNN training.
    """
    print("\n🎭 Image Augmentation Techniques")
    print("=" * 50)

    # Create a test image with distinctive features
    img_size = 28
    original = np.zeros((img_size, img_size))

    # Create an arrow pattern
    for i in range(img_size // 4, 3 * img_size // 4):
        original[i, img_size // 2] = 1.0  # Vertical line
    for i in range(img_size // 2, 3 * img_size // 4):
        original[img_size // 2, i] = 1.0  # Horizontal line

    # Arrow head
    for i in range(5):
        original[img_size // 2 - i, 3 * img_size // 4 - i] = 1.0
        original[img_size // 2 + i, 3 * img_size // 4 - i] = 1.0

    augmented_images = {}

    # 1. Rotation
    def rotate_image(img: np.ndarray, angle_degrees: float) -> np.ndarray:
        """Simple rotation simulation."""
        # This is a simplified rotation - in practice, use scipy or cv2
        if angle_degrees == 90:
            return np.rot90(img, 1)
        elif angle_degrees == 180:
            return np.rot90(img, 2)
        elif angle_degrees == 270:
            return np.rot90(img, 3)
        else:
            # For other angles, just add some transformation effect
            return np.fliplr(img)  # Simplified

    augmented_images["Rotate 90°"] = rotate_image(original, 90)
    augmented_images["Rotate 180°"] = rotate_image(original, 180)

    # 2. Flipping
    augmented_images["Horizontal Flip"] = np.fliplr(original)
    augmented_images["Vertical Flip"] = np.flipud(original)

    # 3. Scaling (zoom)
    def scale_image(img: np.ndarray, scale_factor: float) -> np.ndarray:
        """Simple scaling simulation."""
        if scale_factor > 1:  # Zoom in
            crop_size = int(img_size / scale_factor)
            start = (img_size - crop_size) // 2
            cropped = img[start : start + crop_size, start : start + crop_size]
            # Simulate upsampling
            scaled = np.zeros_like(img)
            for i in range(crop_size):
                for j in range(crop_size):
                    # Simple nearest neighbor upsampling
                    scaled_i = int(i * scale_factor)
                    scaled_j = int(j * scale_factor)
                    if scaled_i < img_size and scaled_j < img_size:
                        scaled[scaled_i, scaled_j] = cropped[i, j]
            return scaled
        else:  # Zoom out
            scaled = np.zeros_like(img)
            new_size = int(img_size * scale_factor)
            start = (img_size - new_size) // 2
            for i in range(new_size):
                for j in range(new_size):
                    orig_i = int(i / scale_factor)
                    orig_j = int(j / scale_factor)
                    if orig_i < img_size and orig_j < img_size:
                        scaled[start + i, start + j] = img[orig_i, orig_j]
            return scaled

    augmented_images["Zoom In 1.5x"] = scale_image(original, 1.5)
    augmented_images["Zoom Out 0.7x"] = scale_image(original, 0.7)

    # 4. Translation
    def translate_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate image by dx, dy pixels."""
        translated = np.zeros_like(img)
        for i in range(img_size):
            for j in range(img_size):
                new_i = i + dy
                new_j = j + dx
                if 0 <= new_i < img_size and 0 <= new_j < img_size:
                    translated[new_i, new_j] = img[i, j]
        return translated

    augmented_images["Translate Right"] = translate_image(original, 5, 0)
    augmented_images["Translate Down"] = translate_image(original, 0, 5)

    # 5. Noise addition
    def add_noise(img: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, noise_level, img.shape)
        return np.clip(img + noise, 0, 1)

    augmented_images["Gaussian Noise"] = add_noise(original, 0.1)

    # 6. Brightness adjustment
    def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        return np.clip(img * factor, 0, 1)

    augmented_images["Brighter"] = adjust_brightness(original, 1.3)
    augmented_images["Darker"] = adjust_brightness(original, 0.7)

    # Visualize augmentations
    n_augmentations = len(augmented_images)
    n_cols = 4
    n_rows = (n_augmentations + n_cols) // n_cols + 1  # +1 for original

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # Show original
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Show augmentations
    for idx, (name, aug_img) in enumerate(augmented_images.items()):
        if idx + 1 < len(axes):
            axes[idx + 1].imshow(aug_img, cmap="gray")
            axes[idx + 1].set_title(name)
            axes[idx + 1].axis("off")

    # Hide unused subplots
    for idx in range(len(augmented_images) + 1, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.suptitle("Image Augmentation Techniques", y=1.02, fontsize=16)
    plt.show()

    print("\n🎯 Augmentation Benefits:")
    print("• Rotation: Handles objects at different orientations")
    print("• Flipping: Doubles dataset size, learns mirror invariance")
    print("• Scaling: Handles objects at different distances/sizes")
    print("• Translation: Learns position invariance")
    print("• Noise: Improves robustness to real-world imperfections")
    print("• Brightness: Handles different lighting conditions")
    print("\n💡 Data Augmentation increases training data variety without")
    print("   collecting more samples, improving model generalization!")


def cnn_transfer_learning_demo() -> None:
    """
    Demonstrate transfer learning concepts for CNNs.
    """
    print("\n🔄 Transfer Learning Simulation")
    print("=" * 50)

    # Simulate pre-trained model features
    print("Simulating pre-trained CNN (e.g., ImageNet model)...")

    # Pre-trained model architecture
    pretrained_layers = [
        ("Conv1", "Low-level features: edges, corners, textures"),
        ("Conv2", "Mid-level features: shapes, patterns"),
        ("Conv3", "High-level features: object parts"),
        ("Conv4", "Complex features: object-specific patterns"),
        ("FC", "Classification head (original task)"),
    ]

    print("\nPre-trained Model Layers:")
    for i, (layer_name, description) in enumerate(pretrained_layers):
        print(f"{i+1}. {layer_name}: {description}")

    # Transfer learning strategies
    strategies = {
        "Feature Extraction": {
            "description": "Freeze pre-trained layers, train only new classifier",
            "frozen_layers": ["Conv1", "Conv2", "Conv3", "Conv4"],
            "trainable_layers": ["New FC"],
            "use_case": "Small dataset, similar to pre-trained domain",
        },
        "Fine-tuning": {
            "description": "Unfreeze some layers, train with low learning rate",
            "frozen_layers": ["Conv1", "Conv2"],
            "trainable_layers": ["Conv3", "Conv4", "New FC"],
            "use_case": "Medium dataset, somewhat different domain",
        },
        "Full Training": {
            "description": "Train all layers with pre-trained weights as initialization",
            "frozen_layers": [],
            "trainable_layers": ["Conv1", "Conv2", "Conv3", "Conv4", "New FC"],
            "use_case": "Large dataset, different domain",
        },
    }

    print("\n🎯 Transfer Learning Strategies:")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (strategy_name, details) in enumerate(strategies.items()):
        ax = axes[idx]

        print(f"\n{idx+1}. {strategy_name}:")
        print(f"   {details['description']}")
        print(f"   Use case: {details['use_case']}")
        print(f"   Frozen: {details['frozen_layers']}")
        print(f"   Trainable: {details['trainable_layers']}")

        # Visualize strategy
        layer_names = ["Conv1", "Conv2", "Conv3", "Conv4", "New FC"]
        colors = []

        for layer in layer_names:
            if layer in details["frozen_layers"]:
                colors.append("lightgray")  # Frozen
            elif layer in details["trainable_layers"]:
                colors.append("lightgreen")  # Trainable
            else:
                colors.append("lightblue")  # Pre-trained

        bars = ax.bar(range(len(layer_names)), [1] * len(layer_names), color=colors)
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45)
        ax.set_title(strategy_name)
        ax.set_ylabel("Layer Status")
        ax.set_ylim(0, 1.2)

        # Add legend
        if idx == 0:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="lightgray", label="Frozen"),
                Patch(facecolor="lightgreen", label="Trainable"),
                Patch(facecolor="lightblue", label="Pre-trained"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        # Remove y-axis ticks
        ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle("Transfer Learning Strategies", y=1.02, fontsize=16)
    plt.show()

    # Performance simulation
    print("\n📊 Expected Performance Comparison:")

    # Simulate training scenarios
    scenarios = {
        "Train from Scratch": {
            "accuracy": 0.75,
            "training_time": 10,
            "data_needed": "Large",
        },
        "Feature Extraction": {
            "accuracy": 0.85,
            "training_time": 2,
            "data_needed": "Small",
        },
        "Fine-tuning": {"accuracy": 0.90, "training_time": 4, "data_needed": "Medium"},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    methods = list(scenarios.keys())
    accuracies = [scenarios[method]["accuracy"] for method in methods]
    times = [scenarios[method]["training_time"] for method in methods]

    ax1.bar(methods, accuracies, color=["red", "orange", "green"])
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Model Performance")
    ax1.set_ylim(0.7, 0.95)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Training time comparison
    ax2.bar(methods, times, color=["red", "orange", "green"])
    ax2.set_ylabel("Training Time (relative)")
    ax2.set_title("Training Efficiency")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

    print("\n💡 Transfer Learning Benefits:")
    print("• Faster training: Leverage pre-learned features")
    print("• Better performance: Especially with limited data")
    print("• Lower computational cost: Less training required")
    print("• Robust features: Benefit from large-scale pre-training")

    print("\n🎯 When to use each strategy:")
    for strategy, details in strategies.items():
        print(f"• {strategy}: {details['use_case']}")


def run_cnn_challenges() -> None:
    """
    Run all CNN challenges with comprehensive analysis.
    """
    print("🚀 Starting Level 5 Challenge 2: Convolutional Neural Networks")
    print("=" * 60)

    try:
        # Challenge 1: Create and analyze image datasets
        print("\n" + "=" * 50)
        print("CHALLENGE 1: Image Dataset Creation")
        print("=" * 50)

        datasets = create_image_datasets()

        print(f"\n✅ Created {len(datasets)} image datasets:")
        for name, data in datasets.items():
            print(
                f"• {name}: {data['X'].shape[0]} samples, {data['X'].shape[1]}x{data['X'].shape[2]} images"
            )
            print(f"  Classes: {data['classes']}")
            print(f"  Description: {data['description']}")

        # Visualize sample images from each dataset
        fig, axes = plt.subplots(len(datasets), 4, figsize=(16, 4 * len(datasets)))
        if len(datasets) == 1:
            axes = [axes]

        for dataset_idx, (dataset_name, data) in enumerate(datasets.items()):
            for class_idx in range(len(data["classes"])):
                # Find first example of this class
                class_indices = np.where(data["y"] == class_idx)[0]
                if len(class_indices) > 0:
                    sample_idx = class_indices[0]
                    img = data["X"][sample_idx].squeeze()

                    axes[dataset_idx][class_idx].imshow(img, cmap="gray")
                    axes[dataset_idx][class_idx].set_title(
                        f"{data['classes'][class_idx]}"
                    )
                    axes[dataset_idx][class_idx].axis("off")

        plt.suptitle("Sample Images from Each Dataset", fontsize=16)
        plt.tight_layout()
        plt.show()

        # Challenge 2: CNN Filter Analysis
        print("\n" + "=" * 50)
        print("CHALLENGE 2: CNN Filter Operations")
        print("=" * 50)

        cnn_filter_visualization()

        # Challenge 3: Architecture Understanding
        print("\n" + "=" * 50)
        print("CHALLENGE 3: CNN Architecture Flow")
        print("=" * 50)

        simulate_cnn_architecture()

        # Challenge 4: Data Augmentation
        print("\n" + "=" * 50)
        print("CHALLENGE 4: Image Augmentation")
        print("=" * 50)

        image_augmentation_demo()

        # Challenge 5: Transfer Learning
        print("\n" + "=" * 50)
        print("CHALLENGE 5: Transfer Learning")
        print("=" * 50)

        cnn_transfer_learning_demo()

        print("\n" + "🎉" * 20)
        print("LEVEL 5 CHALLENGE 2 COMPLETE!")
        print("🎉" * 20)

        print("\n📚 What You've Learned:")
        print("• CNN fundamentals: convolution, pooling, activation")
        print("• Filter operations: edge detection, feature extraction")
        print("• Architecture design: layer stacking, parameter estimation")
        print("• Data augmentation: improving robustness and generalization")
        print("• Transfer learning: leveraging pre-trained models")
        print("• Computer vision: image classification workflows")

        print("\n🚀 Next Steps:")
        print("• Try implementing CNNs with TensorFlow/PyTorch")
        print("• Experiment with different architectures (ResNet, VGG)")
        print("• Apply CNNs to real image classification tasks")
        print("• Explore advanced techniques: attention, normalization")
        print("• Move to Level 5 Challenge 3: Recurrent Neural Networks")

        return datasets

    except Exception as e:
        print(f"❌ Error in CNN challenges: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete CNN challenge
    datasets = run_cnn_challenges()

    if datasets:
        print("\n" + "=" * 60)
        print("CNN CHALLENGE SUMMARY")
        print("=" * 60)

        print("\nDatasets Created:")
        for name, data in datasets.items():
            print(f"• {name}: {len(data['classes'])} classes, {data['X'].shape}")

        print("\nKey CNN Concepts Covered:")
        concepts = [
            "Convolution operations and filters",
            "Pooling layers and dimensionality reduction",
            "CNN architecture design and flow",
            "Feature map visualization and interpretation",
            "Image augmentation techniques",
            "Transfer learning strategies",
            "Parameter estimation and memory usage",
        ]

        for i, concept in enumerate(concepts, 1):
            print(f"{i}. {concept}")

        print("\n✨ Ready for Level 5 Challenge 3: Recurrent Neural Networks!")
