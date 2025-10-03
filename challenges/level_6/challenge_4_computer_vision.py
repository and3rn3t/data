"""
Level 6 - Challenge 4: Computer Vision & Image Processing
========================================================

Master computer vision techniques for image analysis and processing.
This challenge covers image preprocessing, feature extraction, object detection, and classification.

Learning Objectives:
- Understand image data structures and preprocessing
- Learn feature extraction techniques (HOG, LBP, edge detection)
- Master image classification with traditional ML methods
- Explore object detection and segmentation basics
- Apply computer vision to real-world problems

Required Libraries: numpy, matplotlib, scikit-learn, scipy, PIL (optional)
"""

import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import filters, measurements
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def generate_synthetic_images(
    n_samples: int = 1000, image_size: Tuple[int, int] = (32, 32)
) -> Dict[str, Any]:
    """
    Generate synthetic image datasets for computer vision tasks.

    Args:
        n_samples: Number of samples per class
        image_size: Size of generated images (height, width)

    Returns:
        Dictionary containing image datasets and labels
    """
    print("🖼️ Generating Synthetic Image Datasets...")

    height, width = image_size
    rng = np.random.default_rng(42)

    datasets = {}

    # 1. Geometric Shapes Dataset
    print("Creating geometric shapes dataset...")

    shapes = ["circle", "square", "triangle", "rectangle"]
    images = []
    labels = []

    for shape_idx, shape in enumerate(shapes):
        for _ in range(n_samples):
            # Create blank image
            img = np.zeros((height, width))

            if shape == "circle":
                # Generate circle
                center_x = rng.integers(width // 4, 3 * width // 4)
                center_y = rng.integers(height // 4, 3 * height // 4)
                radius = rng.integers(min(height, width) // 8, min(height, width) // 4)

                y, x = np.ogrid[:height, :width]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
                img[mask] = 1.0

            elif shape == "square":
                # Generate square
                size = rng.integers(min(height, width) // 4, min(height, width) // 2)
                start_x = rng.integers(0, width - size)
                start_y = rng.integers(0, height - size)

                img[start_y : start_y + size, start_x : start_x + size] = 1.0

            elif shape == "triangle":
                # Generate triangle (simple)
                base = rng.integers(min(height, width) // 4, min(height, width) // 2)
                height_tri = int(base * 0.866)  # Equilateral triangle

                start_x = rng.integers(0, width - base)
                start_y = rng.integers(0, height - height_tri)

                for i in range(height_tri):
                    left = start_x + i * base // (2 * height_tri)
                    right = start_x + base - i * base // (2 * height_tri)
                    img[start_y + i, left:right] = 1.0

            elif shape == "rectangle":
                # Generate rectangle
                rect_width = rng.integers(
                    min(height, width) // 6, min(height, width) // 2
                )
                rect_height = rng.integers(
                    min(height, width) // 6, min(height, width) // 3
                )

                start_x = rng.integers(0, width - rect_width)
                start_y = rng.integers(0, height - rect_height)

                img[start_y : start_y + rect_height, start_x : start_x + rect_width] = (
                    1.0
                )

            # Add noise
            noise = rng.normal(0, 0.1, img.shape)
            img = np.clip(img + noise, 0, 1)

            images.append(img)
            labels.append(shape)

    # Convert to arrays
    shapes_images = np.array(images)
    shapes_labels = np.array(labels)

    # Shuffle data
    indices = np.arange(len(shapes_images))
    rng.shuffle(indices)
    shapes_images = shapes_images[indices]
    shapes_labels = shapes_labels[indices]

    datasets["geometric_shapes"] = {
        "images": shapes_images,
        "labels": shapes_labels,
        "classes": shapes,
        "description": "Geometric shapes classification",
    }

    # 2. Texture Patterns Dataset
    print("Creating texture patterns dataset...")

    textures = [
        "horizontal_lines",
        "vertical_lines",
        "diagonal_lines",
        "checkerboard",
        "random_dots",
    ]
    texture_images = []
    texture_labels = []

    for texture_idx, texture in enumerate(textures):
        for _ in range(n_samples):
            img = np.zeros((height, width))

            if texture == "horizontal_lines":
                # Horizontal stripes
                line_width = rng.integers(2, 6)
                for y in range(0, height, line_width * 2):
                    img[y : y + line_width, :] = 1.0

            elif texture == "vertical_lines":
                # Vertical stripes
                line_width = rng.integers(2, 6)
                for x in range(0, width, line_width * 2):
                    img[:, x : x + line_width] = 1.0

            elif texture == "diagonal_lines":
                # Diagonal pattern
                for i in range(height):
                    for j in range(width):
                        if (i + j) % 8 < 4:
                            img[i, j] = 1.0

            elif texture == "checkerboard":
                # Checkerboard pattern
                block_size = rng.integers(4, 8)
                for i in range(0, height, block_size):
                    for j in range(0, width, block_size):
                        if ((i // block_size) + (j // block_size)) % 2 == 0:
                            img[i : i + block_size, j : j + block_size] = 1.0

            elif texture == "random_dots":
                # Random dots
                n_dots = rng.integers(50, 200)
                for _ in range(n_dots):
                    x = rng.integers(0, width)
                    y = rng.integers(0, height)
                    dot_size = rng.integers(1, 3)
                    img[
                        max(0, y - dot_size) : min(height, y + dot_size + 1),
                        max(0, x - dot_size) : min(width, x + dot_size + 1),
                    ] = 1.0

            # Add noise
            noise = rng.normal(0, 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)

            texture_images.append(img)
            texture_labels.append(texture)

    # Convert to arrays and shuffle
    texture_images = np.array(texture_images)
    texture_labels = np.array(texture_labels)

    indices = np.arange(len(texture_images))
    rng.shuffle(indices)
    texture_images = texture_images[indices]
    texture_labels = texture_labels[indices]

    datasets["texture_patterns"] = {
        "images": texture_images,
        "labels": texture_labels,
        "classes": textures,
        "description": "Texture pattern recognition",
    }

    # 3. Digit-like Patterns Dataset
    print("Creating digit-like patterns dataset...")

    digit_patterns = [
        "vertical_line",
        "horizontal_line",
        "L_shape",
        "T_shape",
        "plus_sign",
    ]
    digit_images = []
    digit_labels = []

    for pattern_idx, pattern in enumerate(digit_patterns):
        for _ in range(n_samples):
            img = np.zeros((height, width))

            # Center coordinates
            center_x, center_y = width // 2, height // 2
            thickness = rng.integers(2, 4)

            if pattern == "vertical_line":
                # Vertical line
                line_height = rng.integers(height // 2, 3 * height // 4)
                start_y = (height - line_height) // 2
                img[
                    start_y : start_y + line_height,
                    center_x - thickness : center_x + thickness,
                ] = 1.0

            elif pattern == "horizontal_line":
                # Horizontal line
                line_width = rng.integers(width // 2, 3 * width // 4)
                start_x = (width - line_width) // 2
                img[
                    center_y - thickness : center_y + thickness,
                    start_x : start_x + line_width,
                ] = 1.0

            elif pattern == "L_shape":
                # L shape
                arm_length = rng.integers(
                    min(height, width) // 3, min(height, width) // 2
                )
                # Vertical part
                img[
                    center_y : center_y + arm_length,
                    center_x - thickness : center_x + thickness,
                ] = 1.0
                # Horizontal part
                img[
                    center_y
                    + arm_length
                    - thickness : center_y
                    + arm_length
                    + thickness,
                    center_x : center_x + arm_length,
                ] = 1.0

            elif pattern == "T_shape":
                # T shape
                arm_length = rng.integers(
                    min(height, width) // 3, min(height, width) // 2
                )
                # Horizontal part (top)
                img[
                    center_y - thickness : center_y + thickness,
                    center_x - arm_length // 2 : center_x + arm_length // 2,
                ] = 1.0
                # Vertical part
                img[
                    center_y : center_y + arm_length,
                    center_x - thickness : center_x + thickness,
                ] = 1.0

            elif pattern == "plus_sign":
                # Plus sign
                arm_length = rng.integers(
                    min(height, width) // 4, min(height, width) // 3
                )
                # Horizontal line
                img[
                    center_y - thickness : center_y + thickness,
                    center_x - arm_length : center_x + arm_length,
                ] = 1.0
                # Vertical line
                img[
                    center_y - arm_length : center_y + arm_length,
                    center_x - thickness : center_x + thickness,
                ] = 1.0

            # Add rotation
            angle = rng.uniform(-15, 15)  # Small rotation
            img = ndimage.rotate(img, angle, reshape=False, mode="constant", cval=0)

            # Add noise
            noise = rng.normal(0, 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)

            digit_images.append(img)
            digit_labels.append(pattern)

    # Convert to arrays and shuffle
    digit_images = np.array(digit_images)
    digit_labels = np.array(digit_labels)

    indices = np.arange(len(digit_images))
    rng.shuffle(indices)
    digit_images = digit_images[indices]
    digit_labels = digit_labels[indices]

    datasets["digit_patterns"] = {
        "images": digit_images,
        "labels": digit_labels,
        "classes": digit_patterns,
        "description": "Digit-like pattern recognition",
    }

    print(f"Created {len(datasets)} image datasets")
    return datasets


def extract_basic_features(images: np.ndarray) -> np.ndarray:
    """
    Extract basic statistical features from images.

    Args:
        images: Array of images with shape (n_samples, height, width)

    Returns:
        Feature matrix with shape (n_samples, n_features)
    """
    print("\n📊 Extracting Basic Image Features...")

    n_samples = len(images)
    features = []

    for img in images:
        # Basic statistical features
        feature_vector = [
            np.mean(img),  # Mean intensity
            np.std(img),  # Standard deviation
            np.min(img),  # Minimum intensity
            np.max(img),  # Maximum intensity
            np.median(img),  # Median intensity
            np.sum(img > 0.5),  # Number of bright pixels
            np.sum(img < 0.1),  # Number of dark pixels
        ]

        # Moments
        feature_vector.extend(
            [
                np.mean(img**2),  # Second moment
                np.mean(img**3),  # Third moment (skewness related)
                np.mean(img**4),  # Fourth moment (kurtosis related)
            ]
        )

        # Edge-related features (simple gradients)
        grad_x = np.abs(np.diff(img, axis=1))
        grad_y = np.abs(np.diff(img, axis=0))

        feature_vector.extend(
            [
                np.mean(grad_x),  # Mean horizontal gradient
                np.mean(grad_y),  # Mean vertical gradient
                np.std(grad_x),  # Std horizontal gradient
                np.std(grad_y),  # Std vertical gradient
            ]
        )

        # Spatial features
        height, width = img.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Weighted center of mass
        total_intensity = np.sum(img)
        if total_intensity > 0:
            center_x = np.sum(x_coords * img) / total_intensity
            center_y = np.sum(y_coords * img) / total_intensity
        else:
            center_x = width / 2
            center_y = height / 2

        feature_vector.extend(
            [
                center_x / width,  # Normalized center X
                center_y / height,  # Normalized center Y
            ]
        )

        # Symmetry features
        left_half = img[:, : width // 2]
        right_half = img[:, width // 2 :]
        right_half_flipped = np.fliplr(right_half)

        # Adjust for size mismatch
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        horizontal_symmetry = np.corrcoef(
            left_half[:, :min_width].flatten(),
            right_half_flipped[:, :min_width].flatten(),
        )[0, 1]

        if np.isnan(horizontal_symmetry):
            horizontal_symmetry = 0

        top_half = img[: height // 2, :]
        bottom_half = img[height // 2 :, :]
        bottom_half_flipped = np.flipud(bottom_half)

        min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
        vertical_symmetry = np.corrcoef(
            top_half[:min_height, :].flatten(),
            bottom_half_flipped[:min_height, :].flatten(),
        )[0, 1]

        if np.isnan(vertical_symmetry):
            vertical_symmetry = 0

        feature_vector.extend(
            [
                horizontal_symmetry,  # Horizontal symmetry
                vertical_symmetry,  # Vertical symmetry
            ]
        )

        features.append(feature_vector)

    features_array = np.array(features)
    print(f"• Extracted {features_array.shape[1]} features from {n_samples} images")

    return features_array


def extract_hog_features(
    images: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    Simplified implementation without skimage dependency.
    """
    print(f"\n🔍 Extracting HOG Features...")

    features = []

    for img in images:
        # Compute gradients
        grad_x = filters.sobel(img, axis=1)
        grad_y = filters.sobel(img, axis=0)

        # Compute gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
        direction[direction < 0] += 180  # Convert to 0-180 range

        # Cell-based histogram computation (simplified)
        height, width = img.shape
        cell_h, cell_w = pixels_per_cell

        n_cells_y = height // cell_h
        n_cells_x = width // cell_w

        # Compute histogram for each cell
        cell_histograms = []

        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                # Extract cell
                y_start, y_end = cy * cell_h, (cy + 1) * cell_h
                x_start, x_end = cx * cell_w, (cx + 1) * cell_w

                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_dir = direction[y_start:y_end, x_start:x_end]

                # Compute histogram
                hist, _ = np.histogram(
                    cell_dir.flatten(),
                    bins=orientations,
                    range=(0, 180),
                    weights=cell_mag.flatten(),
                )

                cell_histograms.append(hist)

        # Simple normalization (L2 norm per block)
        cell_histograms = np.array(cell_histograms)

        # Block normalization (simplified)
        block_features = []
        blocks_y = n_cells_y - cells_per_block[0] + 1
        blocks_x = n_cells_x - cells_per_block[1] + 1

        for by in range(blocks_y):
            for bx in range(blocks_x):
                # Collect histograms in this block
                block_hist = []
                for cy in range(by, by + cells_per_block[0]):
                    for cx in range(bx, bx + cells_per_block[1]):
                        cell_idx = cy * n_cells_x + cx
                        block_hist.extend(cell_histograms[cell_idx])

                # L2 normalization
                block_hist = np.array(block_hist)
                norm = np.linalg.norm(block_hist)
                if norm > 0:
                    block_hist = block_hist / norm

                block_features.extend(block_hist)

        features.append(block_features)

    features_array = np.array(features)
    print(
        f"• Extracted {features_array.shape[1]} HOG features from {len(images)} images"
    )

    return features_array


def extract_lbp_features(
    images: np.ndarray, radius: int = 1, n_points: int = 8
) -> np.ndarray:
    """
    Extract Local Binary Pattern (LBP) features.
    Simplified implementation.
    """
    print(f"\n🔢 Extracting LBP Features...")

    features = []

    for img in images:
        height, width = img.shape

        # LBP computation
        lbp = np.zeros_like(img)

        for y in range(radius, height - radius):
            for x in range(radius, width - radius):
                center = img[y, x]

                # Sample points around the center
                binary_string = []
                for i in range(n_points):
                    angle = 2 * np.pi * i / n_points
                    # Neighbor coordinates
                    ny = int(y + radius * np.sin(angle))
                    nx = int(x + radius * np.cos(angle))

                    # Ensure coordinates are within bounds
                    ny = max(0, min(height - 1, ny))
                    nx = max(0, min(width - 1, nx))

                    # Compare with center
                    binary_string.append(1 if img[ny, nx] >= center else 0)

                # Convert binary pattern to decimal
                lbp_value = sum([binary_string[i] * (2**i) for i in range(n_points)])
                lbp[y, x] = lbp_value

        # Create histogram of LBP values
        hist, _ = np.histogram(lbp.flatten(), bins=2**n_points, range=(0, 2**n_points))

        # Normalize histogram
        hist = hist.astype(float)
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)

        features.append(hist)

    features_array = np.array(features)
    print(
        f"• Extracted {features_array.shape[1]} LBP features from {len(images)} images"
    )

    return features_array


def apply_image_filters(images: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply various image filters and return filtered versions.
    """
    print("\n🎨 Applying Image Filters...")

    filtered_images = {}

    # 1. Gaussian blur
    print("• Applying Gaussian blur...")
    blurred = np.array([filters.gaussian_filter(img, sigma=1.0) for img in images])
    filtered_images["gaussian_blur"] = blurred

    # 2. Edge detection (Sobel)
    print("• Applying Sobel edge detection...")
    edges = []
    for img in images:
        grad_x = filters.sobel(img, axis=1)
        grad_y = filters.sobel(img, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edges.append(edge_magnitude)
    filtered_images["sobel_edges"] = np.array(edges)

    # 3. Laplacian (edge detection)
    print("• Applying Laplacian filter...")
    laplacian = np.array([filters.laplace(img) for img in images])
    filtered_images["laplacian"] = np.abs(laplacian)

    # 4. Median filter (noise reduction)
    print("• Applying median filter...")
    median_filtered = np.array([filters.median_filter(img, size=3) for img in images])
    filtered_images["median"] = median_filtered

    # 5. Maximum filter (morphological dilation)
    print("• Applying maximum filter...")
    max_filtered = np.array([filters.maximum_filter(img, size=3) for img in images])
    filtered_images["maximum"] = max_filtered

    # 6. Minimum filter (morphological erosion)
    print("• Applying minimum filter...")
    min_filtered = np.array([filters.minimum_filter(img, size=3) for img in images])
    filtered_images["minimum"] = min_filtered

    print(f"• Applied {len(filtered_images)} different filters")

    return filtered_images


def image_classification_pipeline(
    dataset: Dict[str, Any], feature_method: str = "basic"
) -> Dict[str, Any]:
    """
    Complete image classification pipeline with different feature extraction methods.
    """
    print(f"\n🎯 Image Classification Pipeline - {feature_method.upper()}")
    print("=" * 60)

    images = dataset["images"]
    labels = dataset["labels"]
    classes = dataset["classes"]

    print(f"Dataset: {dataset['description']}")
    print(f"• Images: {len(images)} samples of size {images[0].shape}")
    print(f"• Classes: {len(classes)} classes")

    # Feature extraction
    if feature_method == "basic":
        features = extract_basic_features(images)
    elif feature_method == "hog":
        features = extract_hog_features(images)
    elif feature_method == "lbp":
        features = extract_lbp_features(images)
    else:
        raise ValueError("Feature method must be 'basic', 'hog', or 'lbp'")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\n• Training samples: {len(X_train)}")
    print(f"• Testing samples: {len(X_test)}")
    print(f"• Feature dimensions: {features.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "K-NN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}

    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")

        # Train classifier
        classifier.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            "classifier": classifier,
            "accuracy": accuracy,
            "predictions": y_pred,
            "y_test": y_test,
        }

        print(f"• {name} Accuracy: {accuracy:.3f}")

        # Detailed classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=classes))

    # Visualize results
    plt.figure(figsize=(15, 12))

    # Sample images with labels
    plt.subplot(3, 4, 1)
    n_samples_show = min(16, len(images))
    fig_samples = plt.figure(figsize=(12, 8))

    for i in range(n_samples_show):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"{labels[i]}", fontsize=10)
        plt.axis("off")

    plt.suptitle(f'Sample Images - {dataset["description"]}')
    plt.tight_layout()
    plt.show()

    # Results visualization
    plt.figure(figsize=(15, 10))

    # Accuracy comparison
    plt.subplot(2, 3, 1)
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in names]

    colors = ["skyblue", "lightcoral", "lightgreen", "gold"]
    plt.bar(names, accuracies, alpha=0.7, color=colors[: len(names)])
    plt.title(f"Classification Accuracy - {feature_method.upper()} Features")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Feature importance for Random Forest
    plt.subplot(2, 3, 2)
    if "Random Forest" in results:
        rf_model = results["Random Forest"]["classifier"]
        importances = rf_model.feature_importances_

        # Show top 20 features
        top_indices = np.argsort(importances)[-20:]
        top_importances = importances[top_indices]

        plt.barh(range(len(top_importances)), top_importances, alpha=0.7)
        plt.title("Top Feature Importances (Random Forest)")
        plt.xlabel("Importance")
        plt.yticks(range(len(top_importances)), [f"Feature {i}" for i in top_indices])

    # Confusion matrices for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_results = results[best_model_name]

    plt.subplot(2, 3, 3)
    cm = confusion_matrix(
        best_results["y_test"], best_results["predictions"], labels=classes
    )

    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.colorbar()

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Class distribution
    plt.subplot(2, 3, 4)
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.pie(counts, labels=unique_labels, autopct="%1.1f%%")
    plt.title("Class Distribution")

    # Feature distribution (first few features)
    plt.subplot(2, 3, 5)
    n_features_show = min(4, features.shape[1])
    for i in range(n_features_show):
        plt.hist(features[:, i], alpha=0.5, label=f"Feature {i}", bins=20)
    plt.title("Feature Distributions")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.legend()

    # PCA visualization
    plt.subplot(2, 3, 6)
    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(StandardScaler().fit_transform(features))

        # Color map for classes
        colors_map = plt.cm.Set1(np.linspace(0, 1, len(classes)))

        for i, class_name in enumerate(classes):
            mask = labels == class_name
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors_map[i]],
                label=class_name,
                alpha=0.6,
            )

        plt.title("PCA Visualization (2D)")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    results["features"] = features
    results["scaler"] = scaler
    results["best_model"] = best_model_name
    results["feature_method"] = feature_method

    return results


def object_detection_simulation(
    images: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    """
    Simulate basic object detection using template matching and connected components.
    """
    print("\n🎯 Object Detection Simulation")
    print("=" * 40)

    detection_results = []

    # Create simple templates for each class
    templates = {}
    unique_labels = np.unique(labels)

    print("Creating templates from training data...")

    for label in unique_labels:
        # Get all images of this class
        label_images = images[labels == label]

        # Create template by averaging
        template = np.mean(label_images[:10], axis=0)  # Use first 10 images
        templates[label] = template

    print(f"Created {len(templates)} templates")

    # Template matching simulation
    print("Performing template matching...")

    detection_scores = []
    detected_objects = []

    for i, img in enumerate(images[:100]):  # Test on first 100 images
        scores = {}

        # Match against all templates
        for template_label, template in templates.items():
            # Simple correlation-based matching
            correlation = np.corrcoef(img.flatten(), template.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0
            scores[template_label] = correlation

        # Find best match
        best_match = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_match]

        detection_scores.append(best_score)
        detected_objects.append(best_match)

    # Connected components analysis
    print("Analyzing connected components...")

    component_stats = []

    for img in images[:20]:  # Analyze first 20 images
        # Threshold image
        binary = img > 0.5

        # Find connected components
        labeled_array, num_features = measurements.label(binary)

        # Get component properties
        component_sizes = measurements.sum(
            binary, labeled_array, range(1, num_features + 1)
        )

        if len(component_sizes) > 0:
            stats = {
                "num_components": num_features,
                "largest_component": (
                    np.max(component_sizes) if len(component_sizes) > 0 else 0
                ),
                "total_area": np.sum(component_sizes),
                "mean_component_size": (
                    np.mean(component_sizes) if len(component_sizes) > 0 else 0
                ),
            }
        else:
            stats = {
                "num_components": 0,
                "largest_component": 0,
                "total_area": 0,
                "mean_component_size": 0,
            }

        component_stats.append(stats)

    # Visualize detection results
    plt.figure(figsize=(15, 8))

    # Template matching scores
    plt.subplot(2, 3, 1)
    plt.hist(detection_scores, bins=20, alpha=0.7, color="skyblue")
    plt.title("Template Matching Scores")
    plt.xlabel("Correlation Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Detection accuracy
    plt.subplot(2, 3, 2)
    true_labels = labels[:100]
    detected_labels = np.array(detected_objects)
    accuracy = np.mean(true_labels == detected_labels)

    plt.bar(
        ["Correct", "Incorrect"],
        [accuracy, 1 - accuracy],
        color=["green", "red"],
        alpha=0.7,
    )
    plt.title(f"Detection Accuracy: {accuracy:.2f}")
    plt.ylabel("Proportion")

    # Show templates
    plt.subplot(2, 3, 3)
    n_templates = len(templates)
    template_grid = np.zeros((32 * ((n_templates + 1) // 2), 32 * 2))

    for i, (label, template) in enumerate(templates.items()):
        row = i // 2
        col = i % 2
        template_grid[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32] = template

    plt.imshow(template_grid, cmap="gray")
    plt.title("Generated Templates")
    plt.axis("off")

    # Connected components statistics
    plt.subplot(2, 3, 4)
    num_components = [stats["num_components"] for stats in component_stats]
    plt.hist(num_components, bins=10, alpha=0.7, color="lightcoral")
    plt.title("Number of Connected Components")
    plt.xlabel("Component Count")
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 5)
    component_areas = [stats["largest_component"] for stats in component_stats]
    plt.hist(component_areas, bins=15, alpha=0.7, color="lightgreen")
    plt.title("Largest Component Sizes")
    plt.xlabel("Component Size (pixels)")
    plt.ylabel("Frequency")

    # Sample detection results
    plt.subplot(2, 3, 6)
    sample_results = []
    for i in range(min(6, len(images))):
        if i < len(detected_objects):
            sample_results.append(f"{labels[i]} -> {detected_objects[i]}")

    plt.text(
        0.1,
        0.9,
        "Sample Detections:",
        transform=plt.gca().transAxes,
        fontweight="bold",
        fontsize=12,
    )

    for i, result in enumerate(sample_results):
        plt.text(0.1, 0.8 - i * 0.1, result, transform=plt.gca().transAxes, fontsize=10)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "templates": templates,
        "detection_scores": detection_scores,
        "detected_objects": detected_objects,
        "component_stats": component_stats,
        "detection_accuracy": accuracy,
    }


def run_computer_vision_challenges() -> None:
    """
    Run all computer vision and image processing challenges.
    """
    print("🚀 Starting Level 6 Challenge 4: Computer Vision & Image Processing")
    print("=" * 70)

    try:
        # Challenge 1: Generate image datasets
        print("\n" + "=" * 50)
        print("CHALLENGE 1: Synthetic Image Dataset Creation")
        print("=" * 50)

        datasets = generate_synthetic_images(n_samples=500, image_size=(32, 32))

        print(f"\n✅ Created {len(datasets)} image datasets:")
        for name, data in datasets.items():
            images = data["images"]
            labels = data["labels"]
            classes = data["classes"]

            print(f"• {name}: {len(images)} images of size {images[0].shape}")
            print(f"  Classes: {classes}")
            print(f"  Samples per class: {len(images) // len(classes)}")

        # Challenge 2: Basic feature extraction and classification
        print("\n" + "=" * 50)
        print("CHALLENGE 2: Basic Feature Extraction & Classification")
        print("=" * 50)

        # Test on geometric shapes with basic features
        shapes_dataset = datasets["geometric_shapes"]
        basic_results = image_classification_pipeline(
            shapes_dataset, feature_method="basic"
        )

        print(f"\n✅ Basic Feature Classification Complete")
        print(f"• Best model: {basic_results['best_model']}")
        best_accuracy = basic_results[basic_results["best_model"]]["accuracy"]
        print(f"• Best accuracy: {best_accuracy:.3f}")

        # Challenge 3: Advanced feature extraction (HOG)
        print("\n" + "=" * 50)
        print("CHALLENGE 3: Advanced Feature Extraction (HOG)")
        print("=" * 50)

        # Test on digit patterns with HOG features
        digit_dataset = datasets["digit_patterns"]
        hog_results = image_classification_pipeline(digit_dataset, feature_method="hog")

        print(f"\n✅ HOG Feature Classification Complete")
        print(f"• Best model: {hog_results['best_model']}")
        hog_accuracy = hog_results[hog_results["best_model"]]["accuracy"]
        print(f"• Best accuracy: {hog_accuracy:.3f}")

        # Challenge 4: Texture analysis with LBP
        print("\n" + "=" * 50)
        print("CHALLENGE 4: Texture Analysis (LBP Features)")
        print("=" * 50)

        # Test on texture patterns with LBP features
        texture_dataset = datasets["texture_patterns"]
        lbp_results = image_classification_pipeline(
            texture_dataset, feature_method="lbp"
        )

        print(f"\n✅ LBP Feature Classification Complete")
        print(f"• Best model: {lbp_results['best_model']}")
        lbp_accuracy = lbp_results[lbp_results["best_model"]]["accuracy"]
        print(f"• Best accuracy: {lbp_accuracy:.3f}")

        # Challenge 5: Image filtering and preprocessing
        print("\n" + "=" * 50)
        print("CHALLENGE 5: Image Filtering & Preprocessing")
        print("=" * 50)

        # Apply filters to sample images
        sample_images = shapes_dataset["images"][:20]
        filtered_images = apply_image_filters(sample_images)

        # Visualize filters
        plt.figure(figsize=(15, 10))

        original_img = sample_images[0]
        filter_names = list(filtered_images.keys())

        # Show original and filtered versions
        plt.subplot(2, 4, 1)
        plt.imshow(original_img, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        for i, (filter_name, filtered) in enumerate(filtered_images.items()):
            if i < 7:  # Show up to 7 filters
                plt.subplot(2, 4, i + 2)
                plt.imshow(filtered[0], cmap="gray")
                plt.title(filter_name.replace("_", " ").title())
                plt.axis("off")

        plt.suptitle("Image Filtering Results")
        plt.tight_layout()
        plt.show()

        print(f"\n✅ Image Filtering Complete")
        print(f"• Applied {len(filtered_images)} different filters")

        # Challenge 6: Object detection simulation
        print("\n" + "=" * 50)
        print("CHALLENGE 6: Object Detection Simulation")
        print("=" * 50)

        # Use shapes dataset for object detection
        detection_results = object_detection_simulation(
            shapes_dataset["images"], shapes_dataset["labels"]
        )

        print(f"\n✅ Object Detection Complete")
        print(
            f"• Template matching accuracy: {detection_results['detection_accuracy']:.3f}"
        )
        print(f"• Created {len(detection_results['templates'])} object templates")

        # Challenge 7: Method comparison
        print("\n" + "=" * 50)
        print("CHALLENGE 7: Feature Method Comparison")
        print("=" * 50)

        # Compare all methods on the same dataset
        comparison_results = {
            "Basic Features": basic_results[basic_results["best_model"]]["accuracy"],
            "HOG Features": hog_results[hog_results["best_model"]]["accuracy"],
            "LBP Features": lbp_results[lbp_results["best_model"]]["accuracy"],
        }

        # Visualization
        plt.figure(figsize=(12, 8))

        # Method comparison
        plt.subplot(2, 2, 1)
        methods = list(comparison_results.keys())
        accuracies = list(comparison_results.values())

        colors = ["skyblue", "lightcoral", "lightgreen"]
        bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
        plt.title("Feature Extraction Method Comparison")
        plt.ylabel("Best Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        # Dataset complexity comparison
        plt.subplot(2, 2, 2)
        dataset_names = list(datasets.keys())
        n_classes = [len(data["classes"]) for data in datasets.values()]

        plt.bar(dataset_names, n_classes, color="gold", alpha=0.7)
        plt.title("Dataset Complexity (Number of Classes)")
        plt.ylabel("Number of Classes")
        plt.xticks(rotation=45)

        # Feature dimensionality comparison
        plt.subplot(2, 2, 3)
        feature_dims = [
            basic_results["features"].shape[1],
            hog_results["features"].shape[1],
            lbp_results["features"].shape[1],
        ]

        plt.bar(
            methods,
            feature_dims,
            color=["skyblue", "lightcoral", "lightgreen"],
            alpha=0.7,
        )
        plt.title("Feature Dimensionality")
        plt.ylabel("Number of Features")
        plt.xticks(rotation=45)
        plt.yscale("log")

        # Summary statistics
        plt.subplot(2, 2, 4)
        stats_text = f"""
Computer Vision Challenge Summary:

Datasets Created: {len(datasets)}
• Geometric Shapes: {len(shapes_dataset['classes'])} classes
• Texture Patterns: {len(texture_dataset['classes'])} classes
• Digit Patterns: {len(digit_dataset['classes'])} classes

Feature Methods Tested: 3
• Basic: {basic_results['features'].shape[1]} features
• HOG: {hog_results['features'].shape[1]} features
• LBP: {lbp_results['features'].shape[1]} features

Best Overall Accuracy: {max(comparison_results.values()):.3f}
Image Filters Applied: {len(filtered_images)}
Detection Accuracy: {detection_results['detection_accuracy']:.3f}
        """

        plt.text(
            0.05,
            0.95,
            stats_text.strip(),
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        print("\n" + "🎉" * 20)
        print("LEVEL 6 CHALLENGE 4 COMPLETE!")
        print("🎉" * 20)

        print("\n📚 What You've Learned:")
        print("• Synthetic image dataset generation and manipulation")
        print("• Basic statistical feature extraction from images")
        print("• Advanced feature extraction (HOG, LBP)")
        print("• Image classification with multiple ML algorithms")
        print("• Image filtering and preprocessing techniques")
        print("• Basic object detection and template matching")
        print("• Comparative evaluation of CV methods")

        print("\n🚀 Next Steps:")
        print("• Explore deep learning for computer vision (CNNs)")
        print("• Learn advanced object detection (YOLO, R-CNN)")
        print("• Study image segmentation and semantic analysis")
        print("• Apply to real-world image datasets")
        print("• Move to Level 6 Challenge 5: Recommendation Systems")

        return datasets

    except Exception as e:
        print(f"❌ Error in computer vision challenges: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete computer vision challenge
    datasets = run_computer_vision_challenges()

    if datasets:
        print("\n" + "=" * 70)
        print("COMPUTER VISION CHALLENGE SUMMARY")
        print("=" * 70)

        print("\nDatasets Created:")
        for name, data in datasets.items():
            images = data["images"]
            classes = data["classes"]
            print(f"• {name}: {len(images)} images, {len(classes)} classes")

        print("\nKey Computer Vision Concepts Covered:")
        concepts = [
            "Synthetic image generation and dataset creation",
            "Basic statistical feature extraction from images",
            "Histogram of Oriented Gradients (HOG) features",
            "Local Binary Patterns (LBP) for texture analysis",
            "Image filtering and preprocessing techniques",
            "Template matching and basic object detection",
            "Multi-class image classification pipelines",
            "Comparative evaluation of feature extraction methods",
        ]

        for i, concept in enumerate(concepts, 1):
            print(f"{i}. {concept}")

        print("\n✨ Ready for Level 6 Challenge 5: Recommendation Systems!")
