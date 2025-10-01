# Level 6: Data Science Master

## Challenge 3: Computer Vision and Image Analytics

Master comprehensive image processing, computer vision techniques, and modern CV applications for extracting insights from visual data in business and scientific contexts.

### Objective

Learn advanced computer vision techniques including image preprocessing, feature extraction, object detection, image classification, and automated visual analysis to build production-ready computer vision systems.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core image processing libraries
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV available")
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not available. Install with: pip install opencv-python")

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import base64

# Scientific image processing
from skimage import (
    filters, feature, measure, morphology, segmentation,
    transform, util, exposure, restoration, color
)
from skimage.feature import (
    corner_harris, corner_peaks, hog, local_binary_pattern,
    greycomatrix, greycoprops, blob_dog, blob_log, blob_doh
)
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

# Machine learning for computer vision
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Advanced visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

print("👁️ Computer Vision and Image Analytics")
print("=" * 45)

# Set random seed for reproducibility
np.random.seed(42)

print("🖼️ Creating Comprehensive Image Datasets...")

# CHALLENGE 1: SYNTHETIC IMAGE DATASET GENERATION
print("\n" + "=" * 70)
print("🎨 CHALLENGE 1: SYNTHETIC IMAGE GENERATION & BASIC PROCESSING")
print("=" * 70)

class SyntheticImageGenerator:
    """Generate realistic synthetic images for computer vision training"""

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.width, self.height = image_size

    def generate_geometric_shapes(self, num_images=100):
        """Generate images with geometric shapes for classification"""
        images = []
        labels = []

        shape_types = ['circle', 'rectangle', 'triangle', 'ellipse']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for _ in range(num_images):
            # Create blank image
            img = Image.new('RGB', self.image_size, 'white')
            draw = ImageDraw.Draw(img)

            # Random shape and color
            shape = np.random.choice(shape_types)
            color = np.random.choice(colors)

            # Random position and size
            x1 = np.random.randint(20, self.width - 80)
            y1 = np.random.randint(20, self.height - 80)
            size = np.random.randint(30, 60)
            x2 = x1 + size
            y2 = y1 + size

            if shape == 'circle':
                draw.ellipse([x1, y1, x2, y2], fill=color)
            elif shape == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=color)
            elif shape == 'ellipse':
                x2 = x1 + np.random.randint(30, 80)
                y2 = y1 + np.random.randint(20, 50)
                draw.ellipse([x1, y1, x2, y2], fill=color)
            elif shape == 'triangle':
                points = [(x1, y2), ((x1+x2)//2, y1), (x2, y2)]
                draw.polygon(points, fill=color)

            # Add some noise
            img_array = np.array(img)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            images.append(np.array(img))
            labels.append(shape)

        return np.array(images), np.array(labels)

    def generate_texture_patterns(self, num_images=80):
        """Generate textured images for pattern analysis"""
        images = []
        labels = []

        texture_types = ['stripes', 'dots', 'grid', 'noise']

        for _ in range(num_images):
            img = Image.new('RGB', self.image_size, 'white')
            draw = ImageDraw.Draw(img)

            texture = np.random.choice(texture_types)

            if texture == 'stripes':
                stripe_width = np.random.randint(5, 15)
                for x in range(0, self.width, stripe_width * 2):
                    draw.rectangle([x, 0, x + stripe_width, self.height],
                                 fill=(0, 0, 0))

            elif texture == 'dots':
                dot_spacing = np.random.randint(15, 25)
                for x in range(dot_spacing, self.width, dot_spacing):
                    for y in range(dot_spacing, self.height, dot_spacing):
                        radius = np.random.randint(3, 8)
                        draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                                   fill=(0, 0, 0))

            elif texture == 'grid':
                line_spacing = np.random.randint(20, 30)
                for x in range(0, self.width, line_spacing):
                    draw.line([x, 0, x, self.height], fill=(0, 0, 0), width=2)
                for y in range(0, self.height, line_spacing):
                    draw.line([0, y, self.width, y], fill=(0, 0, 0), width=2)

            elif texture == 'noise':
                img_array = np.array(img)
                noise = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
                img = Image.fromarray(noise)

            images.append(np.array(img))
            labels.append(texture)

        return np.array(images), np.array(labels)

    def generate_document_images(self, num_images=60):
        """Generate synthetic document-like images"""
        images = []
        labels = []

        doc_types = ['text_heavy', 'table', 'form', 'mixed']

        for _ in range(num_images):
            img = Image.new('RGB', self.image_size, 'white')
            draw = ImageDraw.Draw(img)

            doc_type = np.random.choice(doc_types)

            if doc_type == 'text_heavy':
                # Draw horizontal lines to simulate text
                for y in range(20, self.height - 20, 15):
                    line_width = np.random.randint(self.width//2, int(self.width*0.9))
                    draw.rectangle([20, y, 20 + line_width, y + 8], fill=(0, 0, 0))

            elif doc_type == 'table':
                # Draw table structure
                rows, cols = np.random.randint(3, 6), np.random.randint(2, 5)
                cell_width = (self.width - 40) // cols
                cell_height = (self.height - 40) // rows

                for r in range(rows + 1):
                    y = 20 + r * cell_height
                    draw.line([20, y, self.width - 20, y], fill=(0, 0, 0), width=1)

                for c in range(cols + 1):
                    x = 20 + c * cell_width
                    draw.line([x, 20, x, self.height - 20], fill=(0, 0, 0), width=1)

            elif doc_type == 'form':
                # Draw form fields
                field_height = 25
                for i in range(5):
                    y = 30 + i * 40
                    # Label area
                    draw.rectangle([20, y, 100, y + 15], fill=(200, 200, 200))
                    # Input field
                    draw.rectangle([120, y, self.width - 40, y + field_height],
                                 outline=(0, 0, 0), width=1)

            elif doc_type == 'mixed':
                # Combination of text and visual elements
                # Header
                draw.rectangle([20, 20, self.width - 20, 50], fill=(100, 100, 100))
                # Text lines
                for y in range(70, 150, 12):
                    line_width = np.random.randint(100, self.width - 60)
                    draw.rectangle([30, y, 30 + line_width, y + 6], fill=(0, 0, 0))
                # Box element
                draw.rectangle([30, 160, 120, 200], outline=(0, 0, 0), width=2)

            images.append(np.array(img))
            labels.append(doc_type)

        return np.array(images), np.array(labels)

# Generate synthetic datasets
print("🎨 Generating synthetic image datasets...")
generator = SyntheticImageGenerator((128, 128))

shapes_images, shapes_labels = generator.generate_geometric_shapes(100)
texture_images, texture_labels = generator.generate_texture_patterns(80)
doc_images, doc_labels = generator.generate_document_images(60)

print(f"✅ Generated {len(shapes_images)} geometric shapes")
print(f"✅ Generated {len(texture_images)} texture patterns")
print(f"✅ Generated {len(doc_images)} document images")

# Combine all datasets
all_images = np.concatenate([shapes_images, texture_images, doc_images])
all_labels = np.concatenate([shapes_labels, texture_labels, doc_labels])

print(f"📊 Total dataset: {len(all_images)} images with {len(np.unique(all_labels))} classes")
print(f"Class distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

# Visualize sample images
print("\\n🖼️ Sample Generated Images:")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

sample_indices = np.random.choice(len(all_images), 8, replace=False)
for i, idx in enumerate(sample_indices):
    axes[i].imshow(all_images[idx])
    axes[i].set_title(f'{all_labels[idx]}', fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# CHALLENGE 2: ADVANCED IMAGE PREPROCESSING AND ENHANCEMENT
print("\n" + "=" * 70)
print("🔧 CHALLENGE 2: ADVANCED IMAGE PREPROCESSING & ENHANCEMENT")
print("=" * 70)

class ImagePreprocessor:
    """Comprehensive image preprocessing and enhancement toolkit"""

    def __init__(self):
        pass

    def basic_preprocessing(self, image):
        """Basic image preprocessing operations"""
        results = {'original': image}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else color.rgb2gray(image)
            results['grayscale'] = gray
        else:
            gray = image
            results['grayscale'] = gray

        # Resize to standard size
        resized = transform.resize(gray, (128, 128), preserve_range=True).astype(np.uint8)
        results['resized'] = resized

        # Normalize pixel values
        normalized = exposure.rescale_intensity(resized, out_range=(0, 1))
        results['normalized'] = normalized

        return results

    def noise_reduction(self, image):
        """Various noise reduction techniques"""
        results = {}

        # Gaussian blur
        results['gaussian_blur'] = filters.gaussian(image, sigma=1)

        # Median filter
        results['median_filter'] = filters.median(image, morphology.disk(3))

        # Bilateral filter (if OpenCV available)
        if CV2_AVAILABLE and len(image.shape) == 2:
            bilateral = cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75)
            results['bilateral'] = bilateral

        # Non-local means denoising
        results['nlm_denoised'] = restoration.denoise_nl_means(
            image, patch_size=7, patch_distance=11, h=0.1
        )

        return results

    def edge_detection(self, image):
        """Multiple edge detection algorithms"""
        results = {}

        # Sobel edge detection
        results['sobel'] = filters.sobel(image)

        # Canny edge detection
        results['canny'] = feature.canny(image, sigma=1, low_threshold=0.1, high_threshold=0.2)

        # Prewitt edge detection
        results['prewitt'] = filters.prewitt(image)

        # Laplacian of Gaussian
        results['log'] = filters.laplace(filters.gaussian(image, sigma=1))

        # Roberts cross-gradient
        results['roberts'] = filters.roberts(image)

        return results

    def morphological_operations(self, binary_image):
        """Morphological image processing operations"""
        results = {}

        # Ensure binary image
        if binary_image.max() > 1:
            binary_image = binary_image > 0.5

        # Erosion
        results['erosion'] = morphology.erosion(binary_image, morphology.disk(3))

        # Dilation
        results['dilation'] = morphology.dilation(binary_image, morphology.disk(3))

        # Opening (erosion followed by dilation)
        results['opening'] = morphology.opening(binary_image, morphology.disk(3))

        # Closing (dilation followed by erosion)
        results['closing'] = morphology.closing(binary_image, morphology.disk(3))

        # Skeletonization
        results['skeleton'] = morphology.skeletonize(binary_image)

        return results

    def histogram_enhancement(self, image):
        """Histogram-based image enhancement techniques"""
        results = {}

        # Histogram equalization
        results['hist_eq'] = exposure.equalize_hist(image)

        # Adaptive histogram equalization (CLAHE)
        results['clahe'] = exposure.equalize_adapthist(image, clip_limit=0.03)

        # Gamma correction
        results['gamma_corrected'] = exposure.adjust_gamma(image, gamma=1.5)

        # Log transformation
        results['log_transform'] = exposure.adjust_log(image)

        # Sigmoid transformation
        results['sigmoid'] = exposure.adjust_sigmoid(image, cutoff=0.5, gain=10)

        return results

# Test preprocessing on sample images
print("🧪 Testing Image Preprocessing Pipeline...")

preprocessor = ImagePreprocessor()
sample_image = all_images[0]

# Basic preprocessing
basic_results = preprocessor.basic_preprocessing(sample_image)
gray_image = basic_results['normalized']

# Test different preprocessing techniques
noise_results = preprocessor.noise_reduction(gray_image)
edge_results = preprocessor.edge_detection(gray_image)
hist_results = preprocessor.histogram_enhancement(gray_image)

# Create binary image for morphological operations
binary_image = gray_image > 0.5
morph_results = preprocessor.morphological_operations(binary_image)

print("✅ Preprocessing pipeline tested successfully")

# Visualize preprocessing results
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

# Row 1: Basic preprocessing
axes[0, 0].imshow(sample_image)
axes[0, 0].set_title('Original')
axes[0, 1].imshow(basic_results['grayscale'], cmap='gray')
axes[0, 1].set_title('Grayscale')
axes[0, 2].imshow(noise_results['gaussian_blur'], cmap='gray')
axes[0, 2].set_title('Gaussian Blur')
axes[0, 3].imshow(hist_results['hist_eq'], cmap='gray')
axes[0, 3].set_title('Hist Equalization')

# Row 2: Edge detection
axes[1, 0].imshow(edge_results['sobel'], cmap='gray')
axes[1, 0].set_title('Sobel Edges')
axes[1, 1].imshow(edge_results['canny'], cmap='gray')
axes[1, 1].set_title('Canny Edges')
axes[1, 2].imshow(edge_results['prewitt'], cmap='gray')
axes[1, 2].set_title('Prewitt Edges')
axes[1, 3].imshow(edge_results['log'], cmap='gray')
axes[1, 3].set_title('LoG')

# Row 3: Morphological operations
axes[2, 0].imshow(binary_image, cmap='gray')
axes[2, 0].set_title('Binary')
axes[2, 1].imshow(morph_results['erosion'], cmap='gray')
axes[2, 1].set_title('Erosion')
axes[2, 2].imshow(morph_results['dilation'], cmap='gray')
axes[2, 2].set_title('Dilation')
axes[2, 3].imshow(morph_results['skeleton'], cmap='gray')
axes[2, 3].set_title('Skeleton')

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()

# CHALLENGE 3: FEATURE EXTRACTION AND DESCRIPTORS
print("\n" + "=" * 70)
print("🎯 CHALLENGE 3: FEATURE EXTRACTION & IMAGE DESCRIPTORS")
print("=" * 70)

class FeatureExtractor:
    """Comprehensive feature extraction for computer vision"""

    def __init__(self):
        pass

    def extract_color_features(self, image):
        """Extract color-based features"""
        features = {}

        if len(image.shape) == 3:
            # Color histogram
            for i, color in enumerate(['red', 'green', 'blue']):
                hist, _ = np.histogram(image[:, :, i], bins=32, range=(0, 256))
                features[f'{color}_hist'] = hist / hist.sum()  # Normalize

            # Color moments
            features['mean_red'] = np.mean(image[:, :, 0])
            features['mean_green'] = np.mean(image[:, :, 1])
            features['mean_blue'] = np.mean(image[:, :, 2])

            features['std_red'] = np.std(image[:, :, 0])
            features['std_green'] = np.std(image[:, :, 1])
            features['std_blue'] = np.std(image[:, :, 2])

        return features

    def extract_texture_features(self, gray_image):
        """Extract texture-based features"""
        features = {}

        # Local Binary Patterns
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                                  range=(0, n_points + 2), density=True)
        features['lbp_histogram'] = lbp_hist

        # Gray Level Co-occurrence Matrix (GLCM)
        # Convert to uint8 for GLCM
        gray_uint8 = (gray_image * 255).astype(np.uint8)
        glcm = greycomatrix(gray_uint8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True, normed=True)

        # GLCM properties
        features['glcm_contrast'] = np.mean(greycoprops(glcm, 'contrast'))
        features['glcm_dissimilarity'] = np.mean(greycoprops(glcm, 'dissimilarity'))
        features['glcm_homogeneity'] = np.mean(greycoprops(glcm, 'homogeneity'))
        features['glcm_energy'] = np.mean(greycoprops(glcm, 'energy'))
        features['glcm_correlation'] = np.mean(greycoprops(glcm, 'correlation'))

        # Histogram of Oriented Gradients (HOG)
        hog_features = hog(gray_image, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False)
        features['hog_features'] = hog_features[:50]  # Take first 50 HOG features

        return features

    def extract_shape_features(self, binary_image):
        """Extract shape-based features"""
        features = {}

        # Ensure binary image
        if binary_image.max() > 1:
            binary_image = binary_image > 0.5

        # Label connected components
        labeled_image = measure.label(binary_image)
        regions = measure.regionprops(labeled_image)

        if regions:
            # Use the largest region
            largest_region = max(regions, key=lambda x: x.area)

            features['area'] = largest_region.area
            features['perimeter'] = largest_region.perimeter
            features['eccentricity'] = largest_region.eccentricity
            features['solidity'] = largest_region.solidity
            features['extent'] = largest_region.extent
            features['orientation'] = largest_region.orientation
            features['major_axis_length'] = largest_region.major_axis_length
            features['minor_axis_length'] = largest_region.minor_axis_length

            # Derived features
            features['aspect_ratio'] = largest_region.major_axis_length / largest_region.minor_axis_length
            features['circularity'] = 4 * np.pi * largest_region.area / (largest_region.perimeter ** 2)
        else:
            # Default values if no regions found
            for key in ['area', 'perimeter', 'eccentricity', 'solidity',
                       'extent', 'orientation', 'major_axis_length',
                       'minor_axis_length', 'aspect_ratio', 'circularity']:
                features[key] = 0

        return features

    def extract_corner_features(self, gray_image):
        """Extract corner and keypoint features"""
        features = {}

        # Harris corner detection
        corner_response = corner_harris(gray_image)
        corners = corner_peaks(corner_response, min_distance=5)

        features['num_corners'] = len(corners)
        features['corner_strength'] = np.mean(corner_response[corner_response > 0]) if np.any(corner_response > 0) else 0

        # Blob detection
        blobs_log = blob_log(gray_image, max_sigma=30, num_sigma=10, threshold=0.1)
        blobs_dog = blob_dog(gray_image, max_sigma=30, threshold=0.1)

        features['num_blobs_log'] = len(blobs_log)
        features['num_blobs_dog'] = len(blobs_dog)

        return features

    def extract_comprehensive_features(self, image):
        """Extract all types of features from an image"""
        all_features = {}

        # Prepare grayscale version
        if len(image.shape) == 3:
            gray_image = color.rgb2gray(image)
        else:
            gray_image = image

        # Normalize grayscale image
        gray_image = exposure.rescale_intensity(gray_image, out_range=(0, 1))

        # Extract different types of features
        if len(image.shape) == 3:
            color_features = self.extract_color_features(image)
            all_features.update(color_features)

        texture_features = self.extract_texture_features(gray_image)
        all_features.update(texture_features)

        # Create binary image for shape analysis
        binary_image = gray_image > 0.5
        shape_features = self.extract_shape_features(binary_image)
        all_features.update(shape_features)

        corner_features = self.extract_corner_features(gray_image)
        all_features.update(corner_features)

        return all_features

# Test feature extraction
print("🎯 Testing Feature Extraction...")

extractor = FeatureExtractor()

# Extract features from sample images
feature_vectors = []
feature_labels = []

print("Extracting features from all images...")
for i in range(0, len(all_images), 10):  # Sample every 10th image for speed
    try:
        features = extractor.extract_comprehensive_features(all_images[i])

        # Convert to numerical vector
        feature_vector = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                feature_vector.extend(value.flatten())
            else:
                feature_vector.append(value)

        feature_vectors.append(feature_vector)
        feature_labels.append(all_labels[i])

    except Exception as e:
        print(f"Error processing image {i}: {e}")
        continue

feature_vectors = np.array(feature_vectors)
feature_labels = np.array(feature_labels)

print(f"✅ Extracted features from {len(feature_vectors)} images")
print(f"Feature vector dimension: {feature_vectors.shape[1]}")

# Feature analysis
print("\\n📊 Feature Analysis:")
print(f"Feature vector statistics:")
print(f"  Mean: {np.mean(feature_vectors, axis=0)[:5]} ...")
print(f"  Std:  {np.std(feature_vectors, axis=0)[:5]} ...")

# CHALLENGE 4: IMAGE CLASSIFICATION AND MACHINE LEARNING
print("\n" + "=" * 70)
print("🤖 CHALLENGE 4: IMAGE CLASSIFICATION & MACHINE LEARNING")
print("=" * 70)

# Prepare data for classification
print("🔄 Preparing data for image classification...")

# Handle any NaN or infinite values
feature_vectors = np.nan_to_num(feature_vectors, nan=0, posinf=0, neginf=0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_vectors)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(feature_labels)

print(f"Classes: {list(label_encoder.classes_)}")
print(f"Class distribution: {dict(zip(*np.unique(feature_labels, return_counts=True)))}")

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Test multiple classifiers
print("\\n🎯 Testing Image Classification Algorithms...")

classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
}

classification_results = {}

for name, classifier in classifiers.items():
    print(f"\\nTraining {name}...")

    # Train classifier
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    classification_results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.3f}")

    # Detailed report for best performer
    if name == 'Random Forest':
        print("\\nDetailed Classification Report (Random Forest):")
        print(classification_report(y_test, y_pred,
                                  target_names=label_encoder.classes_))

print(f"\\n🏆 Best Classifier: {max(classification_results, key=classification_results.get)} "
      f"({max(classification_results.values()):.3f})")

# CHALLENGE 5: OBJECT DETECTION AND SEGMENTATION
print("\n" + "=" * 70)
print("🎯 CHALLENGE 5: OBJECT DETECTION & SEGMENTATION")
print("=" * 70)

class ObjectDetectionAnalyzer:
    """Simple object detection and segmentation methods"""

    def __init__(self):
        pass

    def blob_detection(self, gray_image):
        """Detect blob-like objects in image"""
        # Normalize image
        image_norm = exposure.rescale_intensity(gray_image, out_range=(0, 1))

        # Different blob detection methods
        blobs_log = blob_log(image_norm, min_sigma=1, max_sigma=10,
                           num_sigma=10, threshold=0.1)
        blobs_dog = blob_dog(image_norm, min_sigma=1, max_sigma=10,
                           threshold=0.1)
        blobs_doh = blob_doh(image_norm, min_sigma=1, max_sigma=10,
                           threshold=0.01)

        return {
            'log_blobs': blobs_log,
            'dog_blobs': blobs_dog,
            'doh_blobs': blobs_doh
        }

    def edge_based_detection(self, gray_image):
        """Edge-based object detection"""
        # Multiple edge detection approaches
        edges_canny = feature.canny(gray_image)
        edges_sobel = filters.sobel(gray_image)

        # Find contours from edge image
        labeled_edges = measure.label(edges_canny)
        regions = measure.regionprops(labeled_edges)

        # Filter regions by size and shape
        significant_regions = []
        for region in regions:
            if region.area > 50:  # Minimum area threshold
                significant_regions.append({
                    'bbox': region.bbox,
                    'area': region.area,
                    'centroid': region.centroid,
                    'eccentricity': region.eccentricity
                })

        return {
            'edges_canny': edges_canny,
            'edges_sobel': edges_sobel,
            'regions': significant_regions
        }

    def watershed_segmentation(self, gray_image):
        """Watershed-based image segmentation"""
        # Preprocessing
        denoised = restoration.denoise_nl_means(gray_image)

        # Find local maxima as markers
        local_maxima = feature.peak_local_maxima(denoised, min_distance=20)
        markers = np.zeros_like(denoised, dtype=int)
        for i, (y, x) in enumerate(local_maxima[0:10]):  # Limit markers
            markers[y, x] = i + 1

        # Watershed segmentation
        gradient = filters.sobel(denoised)
        watersheds = segmentation.watershed(gradient, markers)

        return {
            'original': gray_image,
            'denoised': denoised,
            'gradient': gradient,
            'markers': markers,
            'watersheds': watersheds,
            'num_segments': len(np.unique(watersheds)) - 1
        }

    def clustering_segmentation(self, image):
        """K-means clustering for image segmentation"""
        # Prepare image data
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
        else:
            pixels = image.reshape(-1, 1)

        # K-means clustering
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pixels)

        # Reshape back to image
        segmented_image = cluster_labels.reshape(image.shape[:2])

        return {
            'segmented': segmented_image,
            'cluster_centers': kmeans.cluster_centers_,
            'n_clusters': n_clusters
        }

# Test object detection and segmentation
print("🔍 Testing Object Detection and Segmentation...")

detector = ObjectDetectionAnalyzer()

# Test on a sample image with clear objects (geometric shapes)
shape_sample_idx = np.where(all_labels == 'circle')[0][0]
test_image = all_images[shape_sample_idx]
test_gray = color.rgb2gray(test_image)

print(f"Testing on image with label: {all_labels[shape_sample_idx]}")

# Blob detection
blob_results = detector.blob_detection(test_gray)
print(f"Detected blobs - LoG: {len(blob_results['log_blobs'])}, "
      f"DoG: {len(blob_results['dog_blobs'])}, "
      f"DoH: {len(blob_results['doh_blobs'])}")

# Edge-based detection
edge_results = detector.edge_based_detection(test_gray)
print(f"Found {len(edge_results['regions'])} significant edge regions")

# Watershed segmentation
watershed_results = detector.watershed_segmentation(test_gray)
print(f"Watershed segmentation created {watershed_results['num_segments']} segments")

# Clustering segmentation
cluster_results = detector.clustering_segmentation(test_image)
print(f"K-means segmentation with {cluster_results['n_clusters']} clusters")

# Visualize detection and segmentation results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Original and blob detection
axes[0, 0].imshow(test_image)
axes[0, 0].set_title('Original Image')

axes[0, 1].imshow(test_gray, cmap='gray')
# Overlay LoG blobs
for blob in blob_results['log_blobs']:
    y, x, r = blob
    circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
    axes[0, 1].add_patch(circle)
axes[0, 1].set_title(f'LoG Blobs ({len(blob_results["log_blobs"])})')

axes[0, 2].imshow(edge_results['edges_canny'], cmap='gray')
axes[0, 2].set_title('Canny Edges')

axes[0, 3].imshow(edge_results['edges_sobel'], cmap='gray')
axes[0, 3].set_title('Sobel Edges')

# Row 2: Segmentation results
axes[1, 0].imshow(watershed_results['gradient'], cmap='gray')
axes[1, 0].set_title('Gradient')

axes[1, 1].imshow(watershed_results['watersheds'], cmap='tab10')
axes[1, 1].set_title('Watershed Segments')

axes[1, 2].imshow(cluster_results['segmented'], cmap='tab10')
axes[1, 2].set_title('K-means Segments')

# Feature visualization
axes[1, 3].imshow(test_gray, cmap='gray')
axes[1, 3].set_title('Original Grayscale')

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()

# CHALLENGE 6: BUSINESS COMPUTER VISION APPLICATIONS
print("\n" + "=" * 70)
print("💼 CHALLENGE 6: BUSINESS CV APPLICATIONS & ANALYSIS PIPELINE")
print("=" * 70)

class BusinessVisionPipeline:
    """Production-ready computer vision pipeline for business applications"""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.extractor = FeatureExtractor()
        self.detector = ObjectDetectionAnalyzer()
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        self.is_trained = False

    def train_pipeline(self, images, labels):
        """Train the complete CV pipeline for business use"""
        print("🏗️  Training Business Computer Vision Pipeline...")

        # Extract features from all images
        feature_vectors = []
        valid_labels = []

        print(f"  🎯 Extracting features from {len(images)} images...")
        for i, (image, label) in enumerate(zip(images, labels)):
            try:
                # Comprehensive feature extraction
                features = self.extractor.extract_comprehensive_features(image)

                # Convert to numerical vector
                feature_vector = []
                for key, value in features.items():
                    if isinstance(value, np.ndarray):
                        feature_vector.extend(value.flatten())
                    else:
                        feature_vector.append(value)

                feature_vectors.append(feature_vector)
                valid_labels.append(label)

            except Exception as e:
                print(f"    ⚠️  Error processing image {i}: {e}")
                continue

        # Prepare training data
        X = np.array(feature_vectors)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(valid_labels)

        # Train classifier
        self.classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.classifier.fit(X_scaled, y)

        print(f"  ✅ Pipeline trained on {len(X)} samples")
        print(f"  📊 Classes: {list(self.label_encoder.classes_)}")

        self.is_trained = True

    def analyze_image(self, image):
        """Comprehensive analysis of a business image"""
        if not self.is_trained:
            raise ValueError("Pipeline must be trained first")

        results = {
            'image_shape': image.shape,
            'processing_timestamp': datetime.now().isoformat()
        }

        # Preprocessing analysis
        preprocessing = self.preprocessor.basic_preprocessing(image)
        gray_image = preprocessing['normalized']

        # Feature extraction
        features = self.extractor.extract_comprehensive_features(image)
        results['extracted_features'] = len(features)

        # Classification
        feature_vector = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                feature_vector.extend(value.flatten())
            else:
                feature_vector.append(value)

        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector = np.nan_to_num(feature_vector, nan=0, posinf=0, neginf=0)
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # Predictions
        predicted_class_idx = self.classifier.predict(feature_vector_scaled)[0]
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        prediction_proba = self.classifier.predict_proba(feature_vector_scaled)[0]

        results['classification'] = {
            'predicted_class': predicted_class,
            'confidence': float(max(prediction_proba)),
            'class_probabilities': dict(zip(
                self.label_encoder.classes_,
                prediction_proba
            ))
        }

        # Object detection analysis
        detection_results = self.detector.blob_detection(gray_image)
        results['object_detection'] = {
            'num_objects_detected': len(detection_results['log_blobs']),
            'detection_method': 'blob_log'
        }

        # Segmentation analysis
        segmentation_results = self.detector.clustering_segmentation(image)
        results['segmentation'] = {
            'num_segments': segmentation_results['n_clusters'],
            'segmentation_method': 'k_means_clustering'
        }

        # Quality metrics
        results['quality_metrics'] = {
            'image_contrast': float(np.std(gray_image)),
            'image_brightness': float(np.mean(gray_image)),
            'edge_density': float(np.mean(filters.sobel(gray_image)))
        }

        return results

    def batch_analyze(self, images):
        """Analyze multiple images efficiently for business reporting"""
        results = []

        for i, image in enumerate(images):
            try:
                analysis = self.analyze_image(image)
                analysis['image_id'] = i
                results.append(analysis)
            except Exception as e:
                results.append({
                    'image_id': i,
                    'error': str(e),
                    'processing_timestamp': datetime.now().isoformat()
                })

        return results

    def generate_business_report(self, images):
        """Generate comprehensive business insights from image analysis"""
        analyses = self.batch_analyze(images)

        # Aggregate business insights
        insights = {
            'total_images_processed': len(analyses),
            'successful_analyses': len([a for a in analyses if 'error' not in a]),
            'processing_timestamp': datetime.now().isoformat(),
            'classification_summary': {},
            'quality_summary': {},
            'detection_summary': {}
        }

        # Aggregate successful analyses
        successful_analyses = [a for a in analyses if 'error' not in a]

        if successful_analyses:
            # Classification insights
            predicted_classes = [a['classification']['predicted_class'] for a in successful_analyses]
            class_distribution = dict(zip(*np.unique(predicted_classes, return_counts=True)))
            insights['classification_summary'] = {
                'class_distribution': class_distribution,
                'most_common_class': max(class_distribution, key=class_distribution.get),
                'avg_confidence': np.mean([a['classification']['confidence'] for a in successful_analyses])
            }

            # Quality insights
            contrasts = [a['quality_metrics']['image_contrast'] for a in successful_analyses]
            brightnesses = [a['quality_metrics']['image_brightness'] for a in successful_analyses]

            insights['quality_summary'] = {
                'avg_contrast': float(np.mean(contrasts)),
                'avg_brightness': float(np.mean(brightnesses)),
                'quality_score': float(np.mean(contrasts) * np.mean(brightnesses))  # Custom metric
            }

            # Detection insights
            object_counts = [a['object_detection']['num_objects_detected'] for a in successful_analyses]
            insights['detection_summary'] = {
                'avg_objects_per_image': float(np.mean(object_counts)),
                'max_objects_detected': int(np.max(object_counts)),
                'images_with_objects': len([c for c in object_counts if c > 0])
            }

        return insights

# Train and test business pipeline
print("🏢 Training Business Computer Vision Pipeline...")

# Use all generated images for training
business_pipeline = BusinessVisionPipeline()
business_pipeline.train_pipeline(all_images[:50], all_labels[:50])  # Use subset for speed

# Test on sample business scenarios
print("\\n📊 Testing Business Image Analysis...")

test_images = all_images[50:55]  # Different images for testing
test_labels = all_labels[50:55]

for i, (image, true_label) in enumerate(zip(test_images, test_labels)):
    print(f"\\n--- Business Image Analysis {i+1} ---")
    analysis = business_pipeline.analyze_image(image)

    predicted = analysis['classification']['predicted_class']
    confidence = analysis['classification']['confidence']
    objects_detected = analysis['object_detection']['num_objects_detected']
    contrast = analysis['quality_metrics']['image_contrast']

    print(f"True Label: {true_label}")
    print(f"Predicted: {predicted} (confidence: {confidence:.3f})")
    print(f"Objects Detected: {objects_detected}")
    print(f"Image Quality - Contrast: {contrast:.3f}")

# Generate comprehensive business report
print("\\n📈 Generating Business Computer Vision Report...")
business_report = business_pipeline.generate_business_report(all_images[:30])

print("🎯 Business Computer Vision Insights:")
print(f"• Total images processed: {business_report['total_images_processed']}")
print(f"• Successful analyses: {business_report['successful_analyses']}")

if 'classification_summary' in business_report and business_report['classification_summary']:
    class_summary = business_report['classification_summary']
    print(f"• Most common class: {class_summary['most_common_class']}")
    print(f"• Average confidence: {class_summary['avg_confidence']:.3f}")
    print(f"• Class distribution: {class_summary['class_distribution']}")

if 'quality_summary' in business_report and business_report['quality_summary']:
    quality_summary = business_report['quality_summary']
    print(f"• Average image quality score: {quality_summary['quality_score']:.3f}")

if 'detection_summary' in business_report and business_report['detection_summary']:
    detection_summary = business_report['detection_summary']
    print(f"• Average objects per image: {detection_summary['avg_objects_per_image']:.1f}")

print("\\n" + "=" * 70)
print("🎉 CONGRATULATIONS! COMPUTER VISION MASTERY COMPLETE!")
print("=" * 70)

print("\\n🏆 You have successfully mastered:")
print("• Advanced image preprocessing and enhancement techniques")
print("• Comprehensive feature extraction from visual data")
print("• Multi-algorithm image classification and machine learning")
print("• Object detection and image segmentation methods")
print("• Production-ready computer vision pipeline development")
print("• Business-focused image analysis and reporting systems")

print("\\n💼 Business Applications Mastered:")
print("• Quality control and defect detection in manufacturing")
print("• Document processing and automated form analysis")
print("• Medical image analysis and diagnostic assistance")
print("• Retail product recognition and inventory management")
print("• Security and surveillance system automation")
print("• Agricultural monitoring and crop analysis")

print("\\n🚀 Ready for Advanced Applications:")
print("• Deep learning computer vision with CNNs and transfer learning")
print("• Real-time video processing and analysis")
print("• 3D computer vision and depth estimation")
print("• Augmented reality and computer graphics integration")
print("• Edge deployment and mobile computer vision")

print("\\n📈 Business Impact & ROI:")
print("• Automated visual inspection reduces manual QC costs by 40-70%")
print("• Document processing automation saves 50-80% of manual data entry effort")
print("• Real-time defect detection prevents costly production issues")
print("• Visual analytics provides insights impossible with manual analysis")

```

### Success Criteria

- Implement comprehensive image preprocessing and enhancement pipelines
- Master feature extraction techniques for color, texture, shape, and spatial analysis
- Build and compare multiple image classification models with high accuracy
- Develop object detection and segmentation capabilities for business applications
- Create production-ready computer vision systems with real-world performance
- Generate actionable business insights from visual data analysis

### Learning Objectives

- Understand modern computer vision techniques and their business applications
- Master image preprocessing, enhancement, and noise reduction methods
- Learn comprehensive feature extraction from visual data
- Practice advanced classification algorithms for image data
- Develop skills in object detection and image segmentation
- Build complete computer vision pipelines for business automation and intelligence

### Business Impact & Real-World Applications

**🏢 Enterprise Use Cases:**

- **Manufacturing Quality Control**: Automated visual inspection for defect detection, product grading, and quality assurance with 99%+ accuracy
- **Document Processing**: Intelligent OCR, form recognition, and automated data extraction from invoices, contracts, and legal documents
- **Medical Imaging**: Diagnostic assistance, anomaly detection in X-rays/MRIs, and treatment monitoring support systems
- **Retail Intelligence**: Product recognition, inventory management, customer behavior analysis, and visual search capabilities
- **Security & Surveillance**: Automated threat detection, facial recognition, and real-time monitoring with intelligent alerting
- **Agricultural Technology**: Crop health monitoring, yield prediction, pest detection, and precision farming automation

**💰 ROI Considerations:**

- Computer vision automation can reduce manual visual inspection costs by 40-70% while improving consistency and accuracy
- Automated document processing saves 50-80% of manual data entry effort and eliminates human error
- Real-time defect detection prevents costly production issues, potentially saving millions in recalls and rework
- Visual analytics enables insights impossible with manual analysis, creating new revenue opportunities and competitive advantages
- Business stakeholders should expect 3-6 month development cycles for production CV systems with measurable ROI within 6-12 months

**📊 Key Success Metrics:**

- Processing speed: 100+ images per second for real-time applications
- Classification accuracy: >95% for domain-specific visual recognition tasks
- Cost reduction: 50-70% decrease in manual visual inspection and data entry effort
- Quality improvement: 90%+ reduction in defects missed by human inspection
- Response time: Real-time analysis enabling immediate business decision making and automated responses

---

_Pro tip: The most successful computer vision systems solve specific visual problems that are tedious, error-prone, or impossible for humans to handle at scale. Always focus on measurable business outcomes and user workflow integration!_
