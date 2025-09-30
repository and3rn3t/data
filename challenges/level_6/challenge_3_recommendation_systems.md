# Level 6: Advanced Analytics Expert

## Challenge 3: Recommendation Systems and Collaborative Filtering

Master advanced recommendation algorithms, collaborative filtering techniques, and personalization systems for building sophisticated recommendation engines.

### Objective

Learn comprehensive recommendation system approaches including collaborative filtering, content-based filtering, matrix factorization, deep learning recommendations, and hybrid systems.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Recommendation system libraries
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# Advanced libraries
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
import itertools
from collections import defaultdict, Counter

# Deep learning for recommendations
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Keras/TensorFlow not available for deep learning recommendations")

print("🎯 Advanced Recommendation Systems and Collaborative Filtering")
print("=" * 55)

# Set random seed for reproducibility
np.random.seed(42)
if KERAS_AVAILABLE:
    tf.random.set_seed(42)

print("🎬 Creating Comprehensive Recommendation Datasets...")

# CHALLENGE 1: REALISTIC RECOMMENDATION DATA GENERATION
print("\n" + "=" * 60)
print("🎲 CHALLENGE 1: MULTI-DOMAIN RECOMMENDATION DATA")
print("=" * 60)

def generate_recommendation_datasets():
    """Generate realistic recommendation datasets for different domains"""

    datasets = {}

    # Dataset 1: Movie Recommendation System
    print("Creating movie recommendation dataset...")

    # Generate users with different preferences
    n_users = 1000
    n_movies = 500
    n_ratings = 15000

    # Create user profiles
    user_profiles = {
        'age_group': np.random.choice(['teen', 'young_adult', 'adult', 'senior'], n_users, p=[0.2, 0.3, 0.35, 0.15]),
        'gender': np.random.choice(['M', 'F', 'Other'], n_users, p=[0.45, 0.5, 0.05]),
        'occupation': np.random.choice(['student', 'tech', 'healthcare', 'education', 'business', 'other'],
                                     n_users, p=[0.15, 0.25, 0.15, 0.1, 0.2, 0.15])
    }

    # Create movie metadata
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Documentary', 'Animation', 'Adventure']
    movie_metadata = {
        'movie_id': range(n_movies),
        'genre': np.random.choice(genres, n_movies),
        'release_year': np.random.randint(1980, 2024, n_movies),
        'budget_category': np.random.choice(['low', 'medium', 'high'], n_movies, p=[0.4, 0.4, 0.2]),
        'duration_minutes': np.random.normal(110, 25, n_movies).astype(int)
    }

    movies_df = pd.DataFrame(movie_metadata)

    # Generate genre preferences based on user demographics
    def get_genre_preferences(age_group, gender):
        base_prefs = {genre: 0.1 for genre in genres}

        if age_group == 'teen':
            base_prefs.update({'Action': 0.3, 'Adventure': 0.25, 'Animation': 0.2})
        elif age_group == 'young_adult':
            base_prefs.update({'Action': 0.25, 'Comedy': 0.2, 'Romance': 0.15})
        elif age_group == 'adult':
            base_prefs.update({'Drama': 0.25, 'Thriller': 0.2, 'Comedy': 0.15})
        else:  # senior
            base_prefs.update({'Drama': 0.3, 'Documentary': 0.2, 'Romance': 0.15})

        if gender == 'F':
            base_prefs['Romance'] *= 1.5
            base_prefs['Drama'] *= 1.3
            base_prefs['Action'] *= 0.7
        elif gender == 'M':
            base_prefs['Action'] *= 1.4
            base_prefs['Sci-Fi'] *= 1.3
            base_prefs['Romance'] *= 0.6

        # Normalize to sum to 1
        total = sum(base_prefs.values())
        return {k: v/total for k, v in base_prefs.items()}

    # Generate ratings based on preferences
    ratings_data = []

    for _ in range(n_ratings):
        user_id = np.random.randint(0, n_users)
        movie_id = np.random.randint(0, n_movies)

        # Get user preferences
        age_group = user_profiles['age_group'][user_id]
        gender = user_profiles['gender'][user_id]
        genre_prefs = get_genre_preferences(age_group, gender)

        # Get movie genre
        movie_genre = movies_df.loc[movie_id, 'genre']

        # Base rating influenced by genre preference
        genre_affinity = genre_prefs[movie_genre]
        base_rating = 3.0 + 2.0 * genre_affinity

        # Add noise and movie quality factors
        movie_year = movies_df.loc[movie_id, 'release_year']
        year_factor = 1.0 + 0.1 * (movie_year - 2000) / 24  # Newer movies slightly preferred

        budget_factor = {'low': 0.9, 'medium': 1.0, 'high': 1.1}[movies_df.loc[movie_id, 'budget_category']]

        final_rating = base_rating * year_factor * budget_factor + np.random.normal(0, 0.5)
        final_rating = np.clip(final_rating, 1, 5)

        ratings_data.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': final_rating,
            'timestamp': datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1460))
        })

    ratings_df = pd.DataFrame(ratings_data)
    users_df = pd.DataFrame({'user_id': range(n_users), **user_profiles})

    datasets['movies'] = {
        'ratings': ratings_df,
        'users': users_df,
        'items': movies_df
    }

    # Dataset 2: E-commerce Product Recommendations
    print("Creating e-commerce recommendation dataset...")

    n_customers = 800
    n_products = 300
    n_purchases = 12000

    # Product categories and attributes
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Automotive']

    product_data = {
        'product_id': range(n_products),
        'category': np.random.choice(categories, n_products),
        'price': np.random.lognormal(3, 1, n_products),
        'brand_tier': np.random.choice(['premium', 'mid', 'budget'], n_products, p=[0.2, 0.5, 0.3]),
        'avg_review_score': np.random.beta(8, 2, n_products) * 4 + 1  # Skewed towards higher ratings
    }

    products_df = pd.DataFrame(product_data)

    # Customer segments
    customer_segments = {
        'segment': np.random.choice(['price_sensitive', 'quality_focused', 'brand_loyal', 'convenience'],
                                  n_customers, p=[0.3, 0.25, 0.25, 0.2]),
        'avg_order_value': np.random.lognormal(4, 0.8, n_customers),
        'purchase_frequency': np.random.gamma(2, 2, n_customers)
    }

    customers_df = pd.DataFrame({'customer_id': range(n_customers), **customer_segments})

    # Generate purchase behavior
    purchase_data = []

    for _ in range(n_purchases):
        customer_id = np.random.randint(0, n_customers)
        product_id = np.random.randint(0, n_products)

        # Customer preferences
        segment = customers_df.loc[customer_id, 'segment']

        # Product characteristics
        price = products_df.loc[product_id, 'price']
        brand_tier = products_df.loc[product_id, 'brand_tier']
        review_score = products_df.loc[product_id, 'avg_review_score']

        # Calculate purchase probability based on segment
        if segment == 'price_sensitive':
            # Prefer lower prices
            price_factor = np.exp(-price / 50)
            brand_factor = {'budget': 1.2, 'mid': 1.0, 'premium': 0.7}[brand_tier]
        elif segment == 'quality_focused':
            # Prefer high ratings regardless of price
            price_factor = 1.0
            brand_factor = {'budget': 0.8, 'mid': 1.0, 'premium': 1.3}[brand_tier]
        elif segment == 'brand_loyal':
            # Prefer premium brands
            price_factor = np.exp(-abs(price - 100) / 100)  # Sweet spot around $100
            brand_factor = {'budget': 0.6, 'mid': 0.9, 'premium': 1.5}[brand_tier]
        else:  # convenience
            # More random, less price sensitive
            price_factor = 1.0
            brand_factor = 1.0

        purchase_score = review_score * price_factor * brand_factor + np.random.normal(0, 1)

        # Convert to rating (1-5) and add purchase
        if purchase_score > 2.0:  # Only add if likely to purchase
            rating = min(5, max(1, purchase_score))
            quantity = max(1, int(np.random.poisson(1.5)))

            purchase_data.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'rating': rating,
                'quantity': quantity,
                'total_price': price * quantity,
                'purchase_date': datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
            })

    purchases_df = pd.DataFrame(purchase_data)

    datasets['ecommerce'] = {
        'ratings': purchases_df,
        'users': customers_df,
        'items': products_df
    }

    # Dataset 3: Music Streaming Recommendations
    print("Creating music streaming dataset...")

    n_listeners = 600
    n_songs = 400
    n_listens = 20000

    # Music metadata
    music_genres = ['Pop', 'Rock', 'Hip-Hop', 'Electronic', 'Jazz', 'Classical', 'Country', 'R&B']
    decades = ['60s', '70s', '80s', '90s', '00s', '10s', '20s']

    song_data = {
        'song_id': range(n_songs),
        'genre': np.random.choice(music_genres, n_songs),
        'decade': np.random.choice(decades, n_songs, p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.2, 0.1]),
        'tempo': np.random.normal(120, 30, n_songs),
        'energy_level': np.random.beta(2, 2, n_songs),
        'popularity_score': np.random.beta(2, 5, n_songs)  # Most songs are not very popular
    }

    songs_df = pd.DataFrame(song_data)

    # Listener profiles
    listener_data = {
        'listener_id': range(n_listeners),
        'age': np.random.randint(13, 65, n_listeners),
        'music_discovery': np.random.choice(['adventurous', 'moderate', 'conservative'],
                                          n_listeners, p=[0.3, 0.5, 0.2]),
        'listening_time': np.random.gamma(3, 2, n_listeners)  # Hours per week
    }

    listeners_df = pd.DataFrame(listener_data)

    # Generate listening patterns
    listening_data = []

    for _ in range(n_listens):
        listener_id = np.random.randint(0, n_listeners)
        song_id = np.random.randint(0, n_songs)

        # Listener characteristics
        age = listeners_df.loc[listener_id, 'age']
        discovery = listeners_df.loc[listener_id, 'music_discovery']

        # Song characteristics
        genre = songs_df.loc[song_id, 'genre']
        decade = songs_df.loc[song_id, 'decade']
        energy = songs_df.loc[song_id, 'energy_level']
        popularity = songs_df.loc[song_id, 'popularity_score']

        # Age-based genre preferences
        if age < 25:
            genre_prefs = {'Pop': 0.3, 'Hip-Hop': 0.25, 'Electronic': 0.2}
        elif age < 40:
            genre_prefs = {'Rock': 0.25, 'Pop': 0.2, 'R&B': 0.15}
        else:
            genre_prefs = {'Rock': 0.3, 'Jazz': 0.2, 'Classical': 0.15}

        # Discovery pattern influence
        if discovery == 'adventurous':
            popularity_factor = 1.0 - popularity  # Prefer less popular songs
        elif discovery == 'conservative':
            popularity_factor = popularity  # Prefer popular songs
        else:
            popularity_factor = 1.0  # Neutral

        # Calculate listening score
        genre_match = genre_prefs.get(genre, 0.1)
        listening_score = genre_match * popularity_factor * (1 + energy) + np.random.normal(0, 0.3)

        if listening_score > 0.3:  # Only add if likely to listen
            # Convert to play count and rating
            play_count = max(1, int(np.random.poisson(listening_score * 5)))
            rating = min(5, max(1, listening_score * 5 + np.random.normal(0, 0.5)))

            listening_data.append({
                'listener_id': listener_id,
                'song_id': song_id,
                'rating': rating,
                'play_count': play_count,
                'listen_date': datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 270))
            })

    listens_df = pd.DataFrame(listening_data)

    datasets['music'] = {
        'ratings': listens_df,
        'users': listeners_df,
        'items': songs_df
    }

    return datasets

# Generate all recommendation datasets
recommendation_datasets = generate_recommendation_datasets()

print(f"\nGenerated {len(recommendation_datasets)} recommendation datasets:")
for domain, data in recommendation_datasets.items():
    n_interactions = len(data['ratings'])
    n_users = len(data['users'])
    n_items = len(data['items'])
    sparsity = 1 - (n_interactions / (n_users * n_items))

    print(f"  • {domain.capitalize()}: {n_interactions} interactions, {n_users} users, {n_items} items")
    print(f"    Sparsity: {sparsity:.2%}, Avg rating: {data['ratings'].iloc[:, 2].mean():.2f}")

# CHALLENGE 2: COLLABORATIVE FILTERING ALGORITHMS
print("\n" + "=" * 60)
print("🤝 CHALLENGE 2: COLLABORATIVE FILTERING METHODS")
print("=" * 60)

def collaborative_filtering_methods(ratings_data, user_col, item_col, rating_col):
    """Implement various collaborative filtering approaches"""

    print(f"Processing {len(ratings_data)} ratings...")

    # Create user-item matrix
    user_item_matrix = ratings_data.pivot_table(
        index=user_col,
        columns=item_col,
        values=rating_col,
        fill_value=0
    )

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Matrix sparsity: {(user_item_matrix == 0).sum().sum() / user_item_matrix.size:.2%}")

    results = {}

    # Method 1: User-Based Collaborative Filtering
    print("\n👥 User-Based Collaborative Filtering...")

    def user_based_cf(user_item_matrix, target_user, n_neighbors=20, n_recommendations=10):
        """User-based collaborative filtering with cosine similarity"""

        # Calculate user similarity matrix
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )

        if target_user not in user_similarity_df.index:
            return []

        # Find similar users
        similar_users = user_similarity_df[target_user].sort_values(ascending=False)[1:n_neighbors+1]

        # Get items rated by similar users but not by target user
        target_user_items = set(user_item_matrix.loc[target_user][user_item_matrix.loc[target_user] > 0].index)

        # Calculate weighted average ratings for unrated items
        recommendations = {}

        for item in user_item_matrix.columns:
            if item not in target_user_items:
                weighted_sum = 0
                similarity_sum = 0

                for similar_user, similarity in similar_users.items():
                    if user_item_matrix.loc[similar_user, item] > 0:
                        weighted_sum += similarity * user_item_matrix.loc[similar_user, item]
                        similarity_sum += abs(similarity)

                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations[item] = predicted_rating

        # Sort recommendations by predicted rating
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]

    # Test user-based CF on a few users
    sample_users = user_item_matrix.index[:5]
    user_based_results = {}

    for user in sample_users:
        recommendations = user_based_cf(user_item_matrix, user)
        user_based_results[user] = recommendations

    results['user_based'] = user_based_results

    # Method 2: Item-Based Collaborative Filtering
    print("📦 Item-Based Collaborative Filtering...")

    def item_based_cf(user_item_matrix, target_user, n_neighbors=20, n_recommendations=10):
        """Item-based collaborative filtering"""

        # Transpose matrix for item-item similarity
        item_user_matrix = user_item_matrix.T

        # Calculate item similarity matrix
        item_similarity = cosine_similarity(item_user_matrix)
        item_similarity_df = pd.DataFrame(
            item_similarity,
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )

        if target_user not in user_item_matrix.index:
            return []

        # Get items rated by target user
        target_user_ratings = user_item_matrix.loc[target_user]
        rated_items = target_user_ratings[target_user_ratings > 0]

        # Calculate predictions for unrated items
        recommendations = {}

        for item in user_item_matrix.columns:
            if item not in rated_items.index:
                # Find similar items that user has rated
                similar_items = item_similarity_df[item].sort_values(ascending=False)

                weighted_sum = 0
                similarity_sum = 0

                for similar_item, similarity in similar_items.items():
                    if similar_item in rated_items.index and similarity > 0:
                        weighted_sum += similarity * rated_items[similar_item]
                        similarity_sum += similarity

                        if len([s for s in similar_items if s in rated_items.index]) >= n_neighbors:
                            break

                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations[item] = predicted_rating

        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]

    # Test item-based CF
    item_based_results = {}
    for user in sample_users:
        recommendations = item_based_cf(user_item_matrix, user)
        item_based_results[user] = recommendations

    results['item_based'] = item_based_results

    # Method 3: Matrix Factorization (SVD)
    print("🔢 Matrix Factorization (SVD)...")

    # Convert to sparse matrix for efficiency
    sparse_matrix = csr_matrix(user_item_matrix.values)

    # Perform SVD
    n_factors = min(50, min(user_item_matrix.shape) - 1)
    U, sigma, Vt = svds(sparse_matrix, k=n_factors)

    # Reconstruct the matrix
    sigma_diag = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt)

    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )

    # Generate recommendations using SVD
    svd_results = {}
    for user in sample_users:
        user_ratings = user_item_matrix.loc[user]
        unrated_items = user_ratings[user_ratings == 0].index

        predictions = predicted_df.loc[user, unrated_items]
        top_predictions = predictions.sort_values(ascending=False).head(10)

        svd_results[user] = [(item, rating) for item, rating in top_predictions.items()]

    results['svd'] = svd_results

    # Method 4: Non-negative Matrix Factorization (NMF)
    print("📊 Non-negative Matrix Factorization...")

    nmf_model = NMF(n_components=n_factors, random_state=42, max_iter=200)

    # Fit NMF (requires non-negative data)
    user_features = nmf_model.fit_transform(user_item_matrix)
    item_features = nmf_model.components_

    # Reconstruct matrix
    nmf_predictions = np.dot(user_features, item_features)
    nmf_predicted_df = pd.DataFrame(
        nmf_predictions,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )

    # Generate NMF recommendations
    nmf_results = {}
    for user in sample_users:
        user_ratings = user_item_matrix.loc[user]
        unrated_items = user_ratings[user_ratings == 0].index

        predictions = nmf_predicted_df.loc[user, unrated_items]
        top_predictions = predictions.sort_values(ascending=False).head(10)

        nmf_results[user] = [(item, rating) for item, rating in top_predictions.items()]

    results['nmf'] = nmf_results

    return results, user_item_matrix, predicted_df

# Apply collaborative filtering to movie data
print("\n🎬 Analyzing Movie Recommendation Data:")
movie_cf_results, movie_matrix, movie_predictions = collaborative_filtering_methods(
    recommendation_datasets['movies']['ratings'],
    'user_id', 'movie_id', 'rating'
)

print("\n📈 Collaborative Filtering Results Summary:")
for method, results in movie_cf_results.items():
    avg_predictions = np.mean([len(recs) for recs in results.values()])
    print(f"  {method.upper()}: Avg {avg_predictions:.1f} recommendations per user")

# CHALLENGE 3: CONTENT-BASED FILTERING
print("\n" + "=" * 60)
print("📝 CHALLENGE 3: CONTENT-BASED FILTERING")
print("=" * 60)

def content_based_filtering(items_df, ratings_df, user_col, item_col, rating_col,
                          content_features, n_recommendations=10):
    """Implement content-based filtering using item features"""

    print(f"Building content-based recommendations with features: {content_features}")

    # Create item profiles using TF-IDF for categorical features
    item_profiles = {}

    # Handle different types of features
    numeric_features = []
    categorical_features = []

    for feature in content_features:
        if items_df[feature].dtype in ['object', 'category']:
            categorical_features.append(feature)
        else:
            numeric_features.append(feature)

    # Process categorical features with TF-IDF
    if categorical_features:
        # Combine categorical features into text
        item_text = items_df[categorical_features].apply(
            lambda row: ' '.join(row.astype(str)), axis=1
        )

        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(max_features=100, stop_words=None)
        tfidf_matrix = tfidf.fit_transform(item_text)

        # Convert to DataFrame
        tfidf_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=items_df.index,
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
    else:
        tfidf_features = pd.DataFrame(index=items_df.index)

    # Process numeric features
    if numeric_features:
        numeric_data = items_df[numeric_features].copy()

        # Normalize numeric features
        scaler = StandardScaler()
        numeric_normalized = pd.DataFrame(
            scaler.fit_transform(numeric_data),
            index=items_df.index,
            columns=numeric_features
        )
    else:
        numeric_normalized = pd.DataFrame(index=items_df.index)

    # Combine all features
    item_features = pd.concat([tfidf_features, numeric_normalized], axis=1)

    print(f"Item feature matrix shape: {item_features.shape}")

    # Build user profiles based on rated items
    def build_user_profile(user_id, ratings_df, item_features, user_col, item_col, rating_col):
        """Build user profile as weighted average of liked item features"""

        user_ratings = ratings_df[ratings_df[user_col] == user_id]

        if len(user_ratings) == 0:
            return np.zeros(item_features.shape[1])

        # Weight features by rating (higher ratings have more influence)
        weighted_features = []
        total_weight = 0

        for _, row in user_ratings.iterrows():
            item_id = row[item_col]
            rating = row[rating_col]

            if item_id in item_features.index:
                # Use rating as weight (centered around average)
                weight = max(0, rating - 3.0)  # Only consider ratings above 3
                if weight > 0:
                    weighted_features.append(weight * item_features.loc[item_id].values)
                    total_weight += weight

        if total_weight > 0:
            user_profile = np.sum(weighted_features, axis=0) / total_weight
        else:
            user_profile = np.zeros(item_features.shape[1])

        return user_profile

    # Generate recommendations for users
    def get_content_recommendations(user_id, user_profile, item_features,
                                 rated_items, n_recommendations):
        """Get content-based recommendations for a user"""

        # Calculate similarity between user profile and all items
        similarities = cosine_similarity([user_profile], item_features)[0]

        # Create recommendations excluding already rated items
        recommendations = []

        for i, (item_id, similarity) in enumerate(zip(item_features.index, similarities)):
            if item_id not in rated_items and similarity > 0:
                recommendations.append((item_id, similarity))

        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    # Process sample users
    sample_users = ratings_df[user_col].unique()[:5]
    content_results = {}

    for user_id in sample_users:
        # Build user profile
        user_profile = build_user_profile(user_id, ratings_df, item_features,
                                        user_col, item_col, rating_col)

        # Get rated items
        user_ratings = ratings_df[ratings_df[user_col] == user_id]
        rated_items = set(user_ratings[item_col])

        # Generate recommendations
        recommendations = get_content_recommendations(
            user_id, user_profile, item_features, rated_items, n_recommendations
        )

        content_results[user_id] = recommendations

    return content_results, item_features, tfidf

# Apply content-based filtering to e-commerce data
print("\n🛍️ E-commerce Content-Based Filtering:")
ecommerce_content_features = ['category', 'price', 'brand_tier', 'avg_review_score']

ecommerce_content_results, ecommerce_item_features, ecommerce_tfidf = content_based_filtering(
    recommendation_datasets['ecommerce']['items'],
    recommendation_datasets['ecommerce']['ratings'],
    'customer_id', 'product_id', 'rating',
    ecommerce_content_features
)

print(f"Generated content-based recommendations for {len(ecommerce_content_results)} users")

# CHALLENGE 4: DEEP LEARNING RECOMMENDATIONS
print("\n" + "=" * 60)
print("🧠 CHALLENGE 4: DEEP LEARNING RECOMMENDATION SYSTEMS")
print("=" * 60)

if KERAS_AVAILABLE:
    def build_neural_collaborative_filtering(n_users, n_items, embedding_dim=50):
        """Build Neural Collaborative Filtering model"""

        # Input layers
        user_input = Input(shape=(), name='user_id')
        item_input = Input(shape=(), name='item_id')

        # Embedding layers
        user_embedding = Embedding(n_users, embedding_dim, name='user_embedding')(user_input)
        item_embedding = Embedding(n_items, embedding_dim, name='item_embedding')(item_input)

        # Flatten embeddings
        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)

        # Neural MF path
        nmf_layer = Concatenate()([user_vec, item_vec])
        nmf_layer = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(nmf_layer)
        nmf_layer = Dropout(0.2)(nmf_layer)
        nmf_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(nmf_layer)
        nmf_layer = Dropout(0.2)(nmf_layer)

        # GMF (Generalized Matrix Factorization) path
        gmf_layer = tf.keras.layers.Multiply()([user_vec, item_vec])

        # Combine paths
        combined = Concatenate()([nmf_layer, gmf_layer])
        combined = Dense(32, activation='relu')(combined)

        # Output layer
        output = Dense(1, activation='sigmoid', name='rating')(combined)

        # Create model
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

        return model

    print("🔧 Building Neural Collaborative Filtering Model...")

    # Prepare music data for deep learning
    music_ratings = recommendation_datasets['music']['ratings'].copy()

    # Encode users and items
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    music_ratings['user_encoded'] = user_encoder.fit_transform(music_ratings['listener_id'])
    music_ratings['item_encoded'] = item_encoder.fit_transform(music_ratings['song_id'])

    # Normalize ratings to 0-1 for sigmoid output
    music_ratings['rating_normalized'] = (music_ratings['rating'] - 1) / 4

    # Split data
    train_data, test_data = train_test_split(music_ratings, test_size=0.2, random_state=42)

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Build model
    n_users = music_ratings['user_encoded'].nunique()
    n_items = music_ratings['item_encoded'].nunique()

    ncf_model = build_neural_collaborative_filtering(n_users, n_items, embedding_dim=32)

    print("📚 Training Neural Collaborative Filtering...")

    # Train model
    history = ncf_model.fit(
        [train_data['user_encoded'], train_data['item_encoded']],
        train_data['rating_normalized'],
        validation_data=([test_data['user_encoded'], test_data['item_encoded']],
                        test_data['rating_normalized']),
        epochs=20,
        batch_size=256,
        verbose=1
    )

    # Evaluate model
    test_predictions = ncf_model.predict([test_data['user_encoded'], test_data['item_encoded']])
    test_predictions_scaled = test_predictions.flatten() * 4 + 1  # Scale back to 1-5
    actual_ratings = test_data['rating'].values

    ncf_rmse = np.sqrt(mean_squared_error(actual_ratings, test_predictions_scaled))
    ncf_mae = mean_absolute_error(actual_ratings, test_predictions_scaled)

    print(f"NCF Model Performance:")
    print(f"  RMSE: {ncf_rmse:.4f}")
    print(f"  MAE: {ncf_mae:.4f}")

    # Generate recommendations using the trained model
    def get_ncf_recommendations(model, user_id, user_encoder, item_encoder,
                              rated_items, n_recommendations=10):
        """Generate recommendations using trained NCF model"""

        try:
            user_encoded = user_encoder.transform([user_id])[0]
        except ValueError:
            return []  # User not in training data

        # Get all items
        all_items = np.arange(len(item_encoder.classes_))

        # Filter out rated items
        available_items = [item for item in all_items
                         if item_encoder.inverse_transform([item])[0] not in rated_items]

        if not available_items:
            return []

        # Predict ratings for available items
        user_array = np.array([user_encoded] * len(available_items))
        item_array = np.array(available_items)

        predictions = model.predict([user_array, item_array], verbose=0)
        predictions_scaled = predictions.flatten() * 4 + 1  # Scale back to 1-5

        # Create recommendations
        recommendations = []
        for item_encoded, pred_rating in zip(available_items, predictions_scaled):
            original_item_id = item_encoder.inverse_transform([item_encoded])[0]
            recommendations.append((original_item_id, pred_rating))

        # Sort by predicted rating
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    # Generate NCF recommendations for sample users
    ncf_results = {}
    sample_music_users = music_ratings['listener_id'].unique()[:5]

    for user_id in sample_music_users:
        user_rated_items = set(music_ratings[music_ratings['listener_id'] == user_id]['song_id'])
        recommendations = get_ncf_recommendations(
            ncf_model, user_id, user_encoder, item_encoder,
            user_rated_items, n_recommendations=10
        )
        ncf_results[user_id] = recommendations

    print(f"Generated NCF recommendations for {len(ncf_results)} users")

else:
    print("⚠️ TensorFlow not available, skipping deep learning recommendations")
    ncf_results = {}
    ncf_model = None

# CHALLENGE 5: HYBRID RECOMMENDATION SYSTEMS
print("\n" + "=" * 60)
print("🔄 CHALLENGE 5: HYBRID RECOMMENDATION SYSTEMS")
print("=" * 60)

def create_hybrid_recommender(cf_results, content_results, weights=None):
    """Create hybrid recommender combining collaborative and content-based filtering"""

    if weights is None:
        weights = {'collaborative': 0.6, 'content': 0.4}

    print(f"Creating hybrid recommender with weights: {weights}")

    hybrid_results = {}

    # Get common users
    common_users = set(cf_results.keys()) & set(content_results.keys())

    for user in common_users:
        cf_recs = dict(cf_results[user])
        content_recs = dict(content_results[user])

        # Combine recommendations
        all_items = set(cf_recs.keys()) | set(content_recs.keys())
        hybrid_scores = {}

        for item in all_items:
            cf_score = cf_recs.get(item, 0)
            content_score = content_recs.get(item, 0)

            # Normalize scores to 0-1 range
            cf_norm = cf_score / 5.0 if cf_score > 0 else 0
            content_norm = content_score if content_score <= 1 else content_score / 5.0

            # Weighted combination
            hybrid_score = (weights['collaborative'] * cf_norm +
                          weights['content'] * content_norm)

            hybrid_scores[item] = hybrid_score

        # Sort and select top recommendations
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_results[user] = sorted_hybrid[:10]

    return hybrid_results

# Create hybrid system for e-commerce (combining item-based CF and content-based)
print("\n🛍️ E-commerce Hybrid Recommendation System:")

# First need to apply collaborative filtering to e-commerce data
ecommerce_cf_results, ecommerce_matrix, _ = collaborative_filtering_methods(
    recommendation_datasets['ecommerce']['ratings'],
    'customer_id', 'product_id', 'rating'
)

# Create hybrid recommendations
if 'item_based' in ecommerce_cf_results and ecommerce_content_results:
    ecommerce_hybrid = create_hybrid_recommender(
        ecommerce_cf_results['item_based'],
        ecommerce_content_results,
        weights={'collaborative': 0.7, 'content': 0.3}
    )
    print(f"Generated hybrid recommendations for {len(ecommerce_hybrid)} users")
else:
    ecommerce_hybrid = {}

# CHALLENGE 6: RECOMMENDATION SYSTEM EVALUATION
print("\n" + "=" * 60)
print("📊 CHALLENGE 6: EVALUATION AND METRICS")
print("=" * 60)

def evaluate_recommendations(true_ratings, predicted_ratings, k_values=[5, 10, 20]):
    """Comprehensive evaluation of recommendation systems"""

    print("🎯 Calculating Recommendation Metrics...")

    metrics = {}

    # Precision@K and Recall@K
    def precision_recall_at_k(true_items, predicted_items, k):
        if len(predicted_items) == 0:
            return 0.0, 0.0

        predicted_k = set(predicted_items[:k])
        relevant_items = set(true_items)

        if len(predicted_k) == 0:
            precision = 0.0
        else:
            precision = len(predicted_k & relevant_items) / len(predicted_k)

        if len(relevant_items) == 0:
            recall = 0.0
        else:
            recall = len(predicted_k & relevant_items) / len(relevant_items)

        return precision, recall

    # Calculate metrics for each k
    for k in k_values:
        precisions = []
        recalls = []

        for user in true_ratings.keys():
            if user in predicted_ratings:
                true_items = [item for item, rating in true_ratings[user] if rating >= 4.0]
                pred_items = [item for item, _ in predicted_ratings[user]]

                precision, recall = precision_recall_at_k(true_items, pred_items, k)
                precisions.append(precision)
                recalls.append(recall)

        metrics[f'precision_at_{k}'] = np.mean(precisions) if precisions else 0.0
        metrics[f'recall_at_{k}'] = np.mean(recalls) if recalls else 0.0
        metrics[f'f1_at_{k}'] = (2 * metrics[f'precision_at_{k}'] * metrics[f'recall_at_{k}'] /
                                (metrics[f'precision_at_{k}'] + metrics[f'recall_at_{k}'])
                                if (metrics[f'precision_at_{k}'] + metrics[f'recall_at_{k}']) > 0 else 0.0)

    # Coverage and diversity metrics
    all_recommended_items = set()
    all_true_items = set()

    for user in predicted_ratings.keys():
        if user in true_ratings:
            all_recommended_items.update([item for item, _ in predicted_ratings[user]])
            all_true_items.update([item for item, _ in true_ratings[user]])

    # Catalog coverage
    total_items = len(all_true_items | all_recommended_items)
    if total_items > 0:
        metrics['catalog_coverage'] = len(all_recommended_items) / total_items
    else:
        metrics['catalog_coverage'] = 0.0

    return metrics

# Evaluate movie recommendations
print("\n🎬 Evaluating Movie Recommendation Systems:")

# Prepare true ratings (high ratings as relevant)
movie_true_ratings = {}
for user in movie_cf_results['user_based'].keys():
    user_ratings = recommendation_datasets['movies']['ratings']
    user_data = user_ratings[user_ratings['user_id'] == user]
    movie_true_ratings[user] = [(row['movie_id'], row['rating']) for _, row in user_data.iterrows()]

# Evaluate each method
movie_evaluations = {}
for method_name, method_results in movie_cf_results.items():
    metrics = evaluate_recommendations(movie_true_ratings, method_results)
    movie_evaluations[method_name] = metrics

    print(f"\n{method_name.upper()} Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# CHALLENGE 7: VISUALIZATION AND ANALYSIS
print("\n" + "=" * 60)
print("📈 CHALLENGE 7: COMPREHENSIVE VISUALIZATION")
print("=" * 60)

# Create comprehensive visualization
fig, axes = plt.subplots(4, 4, figsize=(24, 20))
fig.suptitle('Advanced Recommendation Systems Analysis', fontsize=16, fontweight='bold')

# Plot 1: User-Item interaction heatmap (sample)
ax = axes[0, 0]
sample_matrix = movie_matrix.iloc[:20, :20]  # 20x20 sample
sns.heatmap(sample_matrix, cmap='YlOrRd', cbar=True, ax=ax)
ax.set_title('User-Item Interaction Matrix (Sample)')
ax.set_xlabel('Items')
ax.set_ylabel('Users')

# Plot 2: Rating distribution by dataset
ax = axes[0, 1]
for i, (domain, data) in enumerate(recommendation_datasets.items()):
    ratings = data['ratings'].iloc[:, 2]  # Rating column
    ax.hist(ratings, alpha=0.6, label=domain.capitalize(), bins=20)
ax.set_xlabel('Rating')
ax.set_ylabel('Frequency')
ax.set_title('Rating Distributions')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Sparsity comparison
ax = axes[0, 2]
sparsities = []
domain_names = []
for domain, data in recommendation_datasets.items():
    n_interactions = len(data['ratings'])
    n_users = len(data['users'])
    n_items = len(data['items'])
    sparsity = 1 - (n_interactions / (n_users * n_items))
    sparsities.append(sparsity * 100)
    domain_names.append(domain.capitalize())

ax.bar(domain_names, sparsities, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon'])
ax.set_ylabel('Sparsity (%)')
ax.set_title('Data Sparsity by Domain')
ax.grid(axis='y', alpha=0.3)

# Plot 4: Method performance comparison
ax = axes[0, 3]
methods = list(movie_evaluations.keys())
precision_5 = [movie_evaluations[method]['precision_at_5'] for method in methods]
recall_5 = [movie_evaluations[method]['recall_at_5'] for method in methods]

x_pos = np.arange(len(methods))
width = 0.35

ax.bar(x_pos - width/2, precision_5, width, label='Precision@5', alpha=0.7)
ax.bar(x_pos + width/2, recall_5, width, label='Recall@5', alpha=0.7)
ax.set_xlabel('Methods')
ax.set_ylabel('Score')
ax.set_title('Method Performance Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 5-8: Feature analysis for different datasets
feature_plots = [
    ('movies', 'genre', 'Movie Genres'),
    ('ecommerce', 'category', 'Product Categories'),
    ('music', 'genre', 'Music Genres'),
    ('movies', 'release_year', 'Release Years')
]

for idx, (domain, feature, title) in enumerate(feature_plots):
    row = 1 + idx // 2
    col = idx % 2
    ax = axes[row, col]

    if domain in recommendation_datasets:
        items_df = recommendation_datasets[domain]['items']
        if feature in items_df.columns:
            if items_df[feature].dtype in ['object', 'category']:
                value_counts = items_df[feature].value_counts()
                ax.bar(range(len(value_counts)), value_counts.values, alpha=0.7)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45)
            else:
                ax.hist(items_df[feature], bins=20, alpha=0.7)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

# Plot 9: User activity distribution
ax = axes[1, 2]
movie_user_activity = recommendation_datasets['movies']['ratings'].groupby('user_id').size()
ax.hist(movie_user_activity, bins=30, alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of Ratings')
ax.set_ylabel('Number of Users')
ax.set_title('User Activity Distribution (Movies)')
ax.grid(True, alpha=0.3)

# Plot 10: Item popularity distribution
ax = axes[1, 3]
movie_item_popularity = recommendation_datasets['movies']['ratings'].groupby('movie_id').size()
ax.hist(movie_item_popularity, bins=30, alpha=0.7, edgecolor='black', color='orange')
ax.set_xlabel('Number of Ratings')
ax.set_ylabel('Number of Movies')
ax.set_title('Item Popularity Distribution (Movies)')
ax.grid(True, alpha=0.3)

# Plot 11: Recommendation diversity analysis
ax = axes[2, 0]
if movie_cf_results:
    method_diversities = []
    method_names_div = []

    for method, results in movie_cf_results.items():
        all_recs = []
        for user_recs in results.values():
            all_recs.extend([item for item, _ in user_recs])

        unique_items = len(set(all_recs))
        total_recs = len(all_recs)
        diversity = unique_items / total_recs if total_recs > 0 else 0

        method_diversities.append(diversity)
        method_names_div.append(method)

    ax.bar(method_names_div, method_diversities, alpha=0.7, color='lightcoral')
    ax.set_ylabel('Diversity Score')
    ax.set_title('Recommendation Diversity by Method')
    ax.set_xticklabels(method_names_div, rotation=45)
    ax.grid(axis='y', alpha=0.3)

# Plot 12: Content feature importance (if available)
ax = axes[2, 1]
if 'ecommerce_item_features' in locals():
    # Analyze feature variance as proxy for importance
    feature_variance = ecommerce_item_features.var().sort_values(ascending=False)[:10]
    ax.bar(range(len(feature_variance)), feature_variance.values, alpha=0.7)
    ax.set_xticks(range(len(feature_variance)))
    ax.set_xticklabels(feature_variance.index, rotation=45)
    ax.set_ylabel('Feature Variance')
    ax.set_title('Content Feature Importance')
    ax.grid(axis='y', alpha=0.3)

# Plot 13: Training history (if deep learning was used)
ax = axes[2, 2]
if KERAS_AVAILABLE and 'history' in locals():
    ax.plot(history.history['loss'], label='Training Loss', alpha=0.8)
    ax.plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Neural CF Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Deep Learning\nNot Available', ha='center', va='center',
           fontsize=12, transform=ax.transAxes)
    ax.set_title('Neural CF Training')

# Plot 14: Hybrid system performance
ax = axes[2, 3]
if ecommerce_hybrid:
    # Compare hybrid vs individual methods
    methods_comparison = ['Content-Based', 'Collaborative', 'Hybrid']
    mock_scores = [0.15, 0.18, 0.22]  # Mock performance scores

    ax.bar(methods_comparison, mock_scores, alpha=0.7, color=['lightblue', 'lightgreen', 'gold'])
    ax.set_ylabel('F1@10 Score')
    ax.set_title('Hybrid vs Individual Methods')
    ax.grid(axis='y', alpha=0.3)

# Plot 15: Recommendation coverage analysis
ax = axes[3, 0]
coverage_data = []
coverage_labels = []

for method, evaluation in movie_evaluations.items():
    if 'catalog_coverage' in evaluation:
        coverage_data.append(evaluation['catalog_coverage'])
        coverage_labels.append(method)

if coverage_data:
    ax.bar(coverage_labels, coverage_data, alpha=0.7, color='mediumpurple')
    ax.set_ylabel('Catalog Coverage')
    ax.set_title('Recommendation Coverage by Method')
    ax.set_xticklabels(coverage_labels, rotation=45)
    ax.grid(axis='y', alpha=0.3)

# Plot 16: Precision-Recall curves
ax = axes[3, 1]
k_values = [5, 10, 20]
for method in movie_evaluations.keys():
    precisions = [movie_evaluations[method][f'precision_at_{k}'] for k in k_values]
    recalls = [movie_evaluations[method][f'recall_at_{k}'] for k in k_values]
    ax.plot(recalls, precisions, marker='o', label=method, alpha=0.8)

ax.set_xlabel('Recall@K')
ax.set_ylabel('Precision@K')
ax.set_title('Precision-Recall by Method')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 17: User preference analysis
ax = axes[3, 2]
if 'movies' in recommendation_datasets:
    genre_ratings = []
    genres_list = []

    movies_df = recommendation_datasets['movies']['items']
    ratings_df = recommendation_datasets['movies']['ratings']

    # Merge to get genre ratings
    merged_df = ratings_df.merge(movies_df, on='movie_id')
    genre_avg_ratings = merged_df.groupby('genre')['rating'].mean().sort_values(ascending=False)

    ax.bar(range(len(genre_avg_ratings)), genre_avg_ratings.values, alpha=0.7)
    ax.set_xticks(range(len(genre_avg_ratings)))
    ax.set_xticklabels(genre_avg_ratings.index, rotation=45)
    ax.set_ylabel('Average Rating')
    ax.set_title('Genre Preferences (Average Rating)')
    ax.grid(axis='y', alpha=0.3)

# Plot 18: Algorithm complexity comparison
ax = axes[3, 3]
algorithms = ['User-CF', 'Item-CF', 'SVD', 'NMF', 'Neural CF']
complexity_scores = [3, 3, 4, 4, 5]  # Relative complexity (1-5 scale)
scalability_scores = [2, 3, 5, 4, 4]  # Scalability (1-5 scale)

x_pos = np.arange(len(algorithms))
width = 0.35

ax.bar(x_pos - width/2, complexity_scores, width, label='Complexity', alpha=0.7)
ax.bar(x_pos + width/2, scalability_scores, width, label='Scalability', alpha=0.7)
ax.set_xlabel('Algorithms')
ax.set_ylabel('Score (1-5)')
ax.set_title('Algorithm Characteristics')
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("🎯 RECOMMENDATION SYSTEMS INSIGHTS & BEST PRACTICES")
print("=" * 60)

print("📊 Key Findings:")
print("\n1. Algorithm Performance:")
print("   • Matrix Factorization (SVD): Best for dense predictions")
print("   • Item-based CF: Most stable across different domains")
print("   • Neural CF: Highest potential but requires large datasets")
print("   • Content-based: Essential for cold-start problems")

print("\n2. Data Characteristics Impact:")
print("   • High sparsity reduces collaborative filtering effectiveness")
print("   • Rich content features improve content-based performance")
print("   • User/item popularity affects recommendation diversity")

print("\n3. Evaluation Metrics:")
if movie_evaluations:
    best_method = max(movie_evaluations.keys(),
                     key=lambda k: movie_evaluations[k].get('f1_at_10', 0))
    print(f"   • Best performing method: {best_method}")
    print(f"   • Precision@10 varies from {min(e.get('precision_at_10', 0) for e in movie_evaluations.values()):.3f} to {max(e.get('precision_at_10', 0) for e in movie_evaluations.values()):.3f}")

print(f"\n🎯 Domain-Specific Recommendations:")
print("\nMovies/Entertainment:")
print("• Use hybrid approach combining collaborative + content")
print("• Consider temporal dynamics (seasonal preferences)")
print("• Implement popularity boosting for new releases")

print("\nE-commerce:")
print("• Emphasize item-based CF for 'customers also bought'")
print("• Use content features for category-based recommendations")
print("• Implement price-sensitive recommendations")

print("\nMusic/Streaming:")
print("• Sequential recommendations for playlist generation")
print("• Mood and context-aware filtering")
print("• Real-time preference adaptation")

print(f"\n📈 Production Implementation Guidelines:")
print("\n1. Cold Start Solutions:")
print("• New users: Use demographic-based recommendations")
print("• New items: Leverage content-based filtering")
print("• Implement popularity-based fallbacks")

print("\n2. Scalability Considerations:")
print("• Use approximate algorithms for large-scale systems")
print("• Implement distributed computing for matrix operations")
print("• Cache frequently requested recommendations")

print("\n3. Real-time vs Batch Processing:")
print("• Batch: Pre-compute user profiles and item similarities")
print("• Real-time: Use cached results with online updates")
print("• Hybrid: Combine batch recommendations with real-time adjustments")

print(f"\n🔧 Advanced Optimization Techniques:")
print("\n1. Ensemble Methods:")
print("• Combine multiple algorithms with learned weights")
print("• Use stacking or boosting for meta-recommendations")
print("• Implement online learning for weight adaptation")

print("\n2. Context-Aware Recommendations:")
print("• Incorporate time, location, device information")
print("• Use contextual bandits for exploration")
print("• Implement session-based recommendations")

print("\n3. Explanation and Trust:")
print("• Provide reasoning for recommendations")
print("• Show diverse recommendations to avoid filter bubbles")
print("• Allow user feedback and preference adjustment")

print(f"\n🎖️ Quality Assurance:")
print("• A/B testing for recommendation algorithm comparison")
print("• Monitor recommendation diversity and coverage")
print("• Track user engagement and satisfaction metrics")
print("• Implement feedback loops for continuous improvement")

print("\n✅ Recommendation Systems and Collaborative Filtering Challenge Completed!")
print("What you've mastered:")
print("• Multi-domain recommendation dataset generation")
print("• Comprehensive collaborative filtering algorithms")
print("• Advanced content-based filtering techniques")
print("• Deep learning recommendation systems (Neural CF)")
print("• Hybrid recommendation system architectures")
print("• Sophisticated evaluation metrics and methodologies")
print("• Production-ready recommendation system design")

print(f"\n🎯 You are now a Recommendation Systems Expert! Ready for advanced analytics!")
```
