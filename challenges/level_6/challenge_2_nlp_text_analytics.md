# Level 6: Data Science Master

## Challenge 2: Natural Language Processing and Text Analytics

Master comprehensive text analytics, natural language processing techniques, and modern NLP applications for extracting insights from unstructured text data.

### Objective

Learn advanced NLP techniques including text preprocessing, sentiment analysis, topic modeling, named entity recognition, and text classification to build production-ready text analytics systems.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Core NLP libraries
import nltk
import spacy
from textblob import TextBlob

# Text processing and vectorization
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion

# Machine learning for NLP
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Advanced NLP techniques
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Text similarity and distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import jaccard
from difflib import SequenceMatcher

# Visualization and analysis
import wordcloud
from collections import Counter, defaultdict
import re
import string
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

print("📚 Natural Language Processing and Text Analytics")
print("=" * 50)

# Set random seed for reproducibility
np.random.seed(42)

print("🔤 Creating Comprehensive Text Datasets...")

# CHALLENGE 1: MULTI-DOMAIN TEXT DATASET GENERATION
print("\n" + "=" * 70)
print("📝 CHALLENGE 1: COMPREHENSIVE TEXT DATA GENERATION & PREPROCESSING")
print("=" * 70)

# Generate realistic business text datasets
def generate_customer_reviews():
    """Generate realistic customer reviews with sentiment labels"""

    positive_templates = [
        "Excellent {product}! Great {quality} and fast delivery. Highly recommend!",
        "Love this {product}. {quality} is outstanding and customer service was helpful.",
        "Amazing {product} with fantastic {quality}. Will definitely buy again!",
        "Perfect {product} for the price. {quality} exceeded my expectations.",
        "Outstanding {product}! Top-notch {quality} and arrived quickly.",
    ]

    negative_templates = [
        "Terrible {product}. Poor {quality} and took forever to arrive. Very disappointed.",
        "Awful {product} with terrible {quality}. Waste of money!",
        "Horrible {product}. {quality} is unacceptable and customer service was rude.",
        "Disappointing {product}. {quality} doesn't match the description at all.",
        "Poor {product} with substandard {quality}. Would not recommend.",
    ]

    neutral_templates = [
        "Decent {product} with acceptable {quality}. Nothing special but works.",
        "Average {product}. {quality} is okay for the price point.",
        "Standard {product} with typical {quality}. Does what it's supposed to do.",
        "Regular {product} with normal {quality}. Not great, not terrible.",
        "Acceptable {product}. {quality} meets basic expectations.",
    ]

    products = ['laptop', 'smartphone', 'headphones', 'camera', 'tablet', 'watch', 'speaker', 'keyboard']
    qualities = ['build quality', 'performance', 'design', 'functionality', 'durability', 'value']

    reviews = []
    sentiments = []

    # Generate balanced dataset
    for sentiment, templates in [('positive', positive_templates),
                               ('negative', negative_templates),
                               ('neutral', neutral_templates)]:
        for _ in range(200):
            template = np.random.choice(templates)
            product = np.random.choice(products)
            quality = np.random.choice(qualities)

            review = template.format(product=product, quality=quality)
            reviews.append(review)
            sentiments.append(sentiment)

    return pd.DataFrame({'review': reviews, 'sentiment': sentiments})

def generate_news_articles():
    """Generate realistic news article headlines with categories"""

    categories = {
        'technology': [
            'New AI breakthrough announced by researchers at leading university',
            'Tech giant releases innovative smartphone with advanced features',
            'Cybersecurity experts warn about emerging online threats',
            'Revolutionary software update improves user experience significantly',
            'Major technology company reports record quarterly earnings',
        ],
        'business': [
            'Stock market reaches new record high amid economic optimism',
            'Major corporation announces expansion into international markets',
            'Economic indicators show steady growth in manufacturing sector',
            'Business leaders discuss strategies for post-pandemic recovery',
            'Consumer spending increases during holiday shopping season',
        ],
        'sports': [
            'Championship game draws record television audience worldwide',
            'Professional athlete signs record-breaking contract extension',
            'Olympic preparations continue with new training facilities',
            'Local team advances to playoffs after impressive season',
            'Sports league announces new safety protocols for players',
        ],
        'health': [
            'Medical researchers publish groundbreaking study on treatment',
            'Health officials recommend new guidelines for public safety',
            'Clinical trial shows promising results for experimental therapy',
            'Healthcare workers receive recognition for exceptional service',
            'Public health campaign aims to increase awareness of prevention',
        ]
    }

    articles = []
    article_categories = []

    for category, headlines in categories.items():
        for _ in range(150):
            base_headline = np.random.choice(headlines)
            # Add some variation to headlines
            variations = [base_headline,
                         base_headline.replace('new', 'latest'),
                         base_headline.replace('major', 'leading'),
                         f"Breaking: {base_headline.lower()}"]

            article = np.random.choice(variations)
            articles.append(article)
            article_categories.append(category)

    return pd.DataFrame({'headline': articles, 'category': article_categories})

def generate_social_media_posts():
    """Generate realistic social media posts with engagement metrics"""

    post_templates = [
        "Just tried the new {restaurant} downtown! {opinion} #foodie #local",
        "Working on {project} today. {feeling} about the progress! #work #productivity",
        "Beautiful {weather} perfect for {activity}! Anyone else enjoying this? #nature",
        "Finished reading {book}. {opinion} - highly recommend! #books #reading",
        "Attending {event} this weekend. {feeling} to see what's in store! #events",
    ]

    restaurants = ['pizza place', 'sushi bar', 'coffee shop', 'burger joint', 'taco truck']
    projects = ['web design', 'data analysis', 'writing project', 'art piece', 'presentation']
    weather = ['sunny weather', 'rainy day', 'snow day', 'spring morning', 'autumn colors']
    activities = ['hiking', 'cycling', 'photography', 'gardening', 'reading outside']
    books = ['mystery novel', 'sci-fi book', 'biography', 'cookbook', 'history book']
    events = ['concert', 'art show', 'farmers market', 'tech meetup', 'sports game']

    opinions = ['Absolutely loved it!', 'Pretty good overall', 'Not bad', 'Could be better', 'Amazing experience!']
    feelings = ['Excited', 'Nervous', 'Curious', 'Optimistic', 'Thrilled']

    posts = []
    engagement_scores = []

    for _ in range(400):
        template = np.random.choice(post_templates)

        # Fill template with random choices
        filled_post = template
        for placeholder, options in [
            ('restaurant', restaurants), ('project', projects), ('weather', weather),
            ('activity', activities), ('book', books), ('event', events),
            ('opinion', opinions), ('feeling', feelings)
        ]:
            if f'{{{placeholder}}}' in filled_post:
                filled_post = filled_post.replace(f'{{{placeholder}}}', np.random.choice(options))

        posts.append(filled_post)
        # Generate engagement score (likes + shares + comments)
        engagement_scores.append(np.random.randint(0, 1000))

    return pd.DataFrame({'post': posts, 'engagement': engagement_scores})

# Generate datasets
print("Creating customer reviews dataset...")
reviews_df = generate_customer_reviews()
print(f"✅ Generated {len(reviews_df)} customer reviews")

print("Creating news articles dataset...")
news_df = generate_news_articles()
print(f"✅ Generated {len(news_df)} news articles")

print("Creating social media posts dataset...")
social_df = generate_social_media_posts()
print(f"✅ Generated {len(social_df)} social media posts")

print("\n📊 Dataset Overview:")
print(f"Customer Reviews: {reviews_df['sentiment'].value_counts().to_dict()}")
print(f"News Categories: {news_df['category'].value_counts().to_dict()}")
print(f"Social Media Posts: Engagement range {social_df['engagement'].min()}-{social_df['engagement'].max()}")

# CHALLENGE 2: ADVANCED TEXT PREPROCESSING PIPELINE
print("\n" + "=" * 70)
print("🧹 CHALLENGE 2: ADVANCED TEXT PREPROCESSING & NORMALIZATION")
print("=" * 70)

class AdvancedTextPreprocessor:
    """Comprehensive text preprocessing pipeline for NLP tasks"""

    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Try to load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.spacy_available = True
        except OSError:
            print("⚠️  spaCy model not available. Install with: python -m spacy download en_core_web_sm")
            self.spacy_available = False

    def clean_text(self, text):
        """Basic text cleaning and normalization"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\\S+@\\S+', '', text)

        # Remove hashtags and mentions (keep the word part)
        text = re.sub(r'#(\\w+)', r'\\1', text)
        text = re.sub(r'@(\\w+)', r'\\1', text)

        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text).strip()

        return text

    def remove_punctuation(self, text, keep_contractions=True):
        """Remove punctuation with option to preserve contractions"""
        if keep_contractions:
            # Keep apostrophes in contractions
            text = re.sub(r"[^\\w\\s']", '', text)
        else:
            text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def tokenize_advanced(self, text):
        """Advanced tokenization with multiple approaches"""
        results = {}

        # NLTK tokenization
        results['nltk_tokens'] = word_tokenize(text)

        # Simple split tokenization
        results['simple_tokens'] = text.split()

        # spaCy tokenization (if available)
        if self.spacy_available:
            doc = self.nlp(text)
            results['spacy_tokens'] = [token.text for token in doc]
            results['spacy_lemmas'] = [token.lemma_ for token in doc]
            results['spacy_pos'] = [(token.text, token.pos_) for token in doc]

        return results

    def extract_features(self, text):
        """Extract various text features for analysis"""
        features = {}

        # Basic features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])

        # Advanced features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_count'] = sum(1 for c in text if c.isdigit())

        # Sentiment with TextBlob
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity

        return features

    def preprocess_pipeline(self, text, operations=['clean', 'punctuation', 'tokenize', 'stopwords', 'stem']):
        """Complete preprocessing pipeline with configurable operations"""
        result = {'original': text}

        # Start with original text
        processed_text = text

        if 'clean' in operations:
            processed_text = self.clean_text(processed_text)
            result['cleaned'] = processed_text

        if 'punctuation' in operations:
            processed_text = self.remove_punctuation(processed_text)
            result['no_punctuation'] = processed_text

        if 'tokenize' in operations:
            tokens = word_tokenize(processed_text)
            result['tokens'] = tokens
        else:
            tokens = processed_text.split()

        if 'stopwords' in operations:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
            result['no_stopwords'] = tokens

        if 'stem' in operations:
            tokens = [self.stemmer.stem(token) for token in tokens]
            result['stemmed'] = tokens

        if 'lemma' in operations and self.spacy_available:
            doc = self.nlp(' '.join(tokens))
            tokens = [token.lemma_ for token in doc]
            result['lemmatized'] = tokens

        result['final_tokens'] = tokens
        result['final_text'] = ' '.join(tokens)

        return result

# Initialize preprocessor
preprocessor = AdvancedTextPreprocessor()

# Test preprocessing on sample data
print("🧪 Testing Text Preprocessing Pipeline...")

sample_reviews = reviews_df['review'].head(3).tolist()
for i, review in enumerate(sample_reviews, 1):
    print(f"\\n--- Sample Review {i} ---")
    print(f"Original: {review}")

    # Process with different configurations
    basic_result = preprocessor.preprocess_pipeline(review, ['clean', 'punctuation', 'stopwords'])
    advanced_result = preprocessor.preprocess_pipeline(review, ['clean', 'punctuation', 'tokenize', 'stopwords', 'stem'])

    print(f"Basic Processing: {basic_result['final_text']}")
    print(f"Advanced Processing: {advanced_result['final_text']}")

    # Extract features
    features = preprocessor.extract_features(review)
    print(f"Features: Words={features['word_count']}, Sentiment={features['polarity']:.2f}, Subjectivity={features['subjectivity']:.2f}")

# CHALLENGE 3: SENTIMENT ANALYSIS AND TEXT CLASSIFICATION
print("\n" + "=" * 70)
print("😊 CHALLENGE 3: SENTIMENT ANALYSIS & TEXT CLASSIFICATION")
print("=" * 70)

# Prepare reviews data for classification
print("Preparing data for sentiment classification...")

# Clean and preprocess reviews
reviews_df['cleaned_review'] = reviews_df['review'].apply(
    lambda x: preprocessor.preprocess_pipeline(x, ['clean', 'punctuation'])['final_text']
)

# Feature extraction strategies
print("\\n🔤 Testing Different Feature Extraction Methods...")

# Method 1: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.95
)

X_tfidf = tfidf_vectorizer.fit_transform(reviews_df['cleaned_review'])
print(f"TF-IDF Features: {X_tfidf.shape}")

# Method 2: Count Vectorization
count_vectorizer = CountVectorizer(
    max_features=3000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=2
)

X_count = count_vectorizer.fit_transform(reviews_df['cleaned_review'])
print(f"Count Features: {X_count.shape}")

# Method 3: Feature engineering from text properties
feature_engineering_df = pd.DataFrame()
for idx, review in enumerate(reviews_df['review']):
    features = preprocessor.extract_features(review)
    for key, value in features.items():
        feature_engineering_df.loc[idx, key] = value

print(f"Engineered Features: {feature_engineering_df.shape}")

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(reviews_df['sentiment'])
sentiment_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"Sentiment Mapping: {sentiment_mapping}")

# Split data for training
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Test multiple classification algorithms
print("\\n🤖 Comparing Classification Algorithms...")

classifiers = {
    'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(kernel='linear', random_state=42)
}

results = {}

for name, classifier in classifiers.items():
    # Train classifier
    classifier.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"{name}: {accuracy:.3f} accuracy")

# Select best classifier
best_classifier_name = max(results, key=results.get)
best_classifier = classifiers[best_classifier_name]
print(f"\\n🏆 Best Classifier: {best_classifier_name} ({results[best_classifier_name]:.3f})")

# Detailed evaluation of best classifier
y_pred_best = best_classifier.predict(X_test_tfidf)
print("\\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# CHALLENGE 4: TOPIC MODELING AND DOCUMENT CLUSTERING
print("\n" + "=" * 70)
print("📋 CHALLENGE 4: TOPIC MODELING & DOCUMENT CLUSTERING")
print("=" * 70)

# Combine all text data for topic modeling
all_texts = []
text_sources = []

# Add reviews (use neutral and positive for topic modeling)
neutral_positive_reviews = reviews_df[reviews_df['sentiment'].isin(['neutral', 'positive'])]
for review in neutral_positive_reviews['cleaned_review']:
    all_texts.append(review)
    text_sources.append('review')

# Add news headlines
for headline in news_df['headline']:
    clean_headline = preprocessor.preprocess_pipeline(headline, ['clean'])['final_text']
    all_texts.append(clean_headline)
    text_sources.append('news')

# Add social media posts (clean hashtags)
for post in social_df['post']:
    clean_post = preprocessor.preprocess_pipeline(post, ['clean'])['final_text']
    all_texts.append(clean_post)
    text_sources.append('social')

print(f"Total documents for topic modeling: {len(all_texts)}")
print(f"Source distribution: {Counter(text_sources)}")

# Prepare documents for topic modeling
tfidf_topic = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2)
)

X_topic = tfidf_topic.fit_transform(all_texts)
feature_names = tfidf_topic.get_feature_names_out()

# Latent Dirichlet Allocation (LDA)
print("\\n🎯 Performing Topic Modeling with LDA...")

n_topics = 8
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20,
    learning_method='online'
)

lda_transform = lda.fit_transform(X_topic)

# Display top topics
def display_topics(model, feature_names, no_top_words=10):
    \"\"\"Display top words for each topic\"\"\"
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[::-1][:no_top_words]]
        topics.append(top_words)
        print(f"Topic {topic_idx}: {' | '.join(top_words)}")
    return topics

print("📌 LDA Topics:")
lda_topics = display_topics(lda, feature_names, 8)

# Non-negative Matrix Factorization (NMF)
print("\\n🎯 Performing Topic Modeling with NMF...")

nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
nmf_transform = nmf.fit_transform(X_topic)

print("📌 NMF Topics:")
nmf_topics = display_topics(nmf, feature_names, 8)

# Document clustering with K-Means
print("\\n🎯 Performing Document Clustering...")

# Use dimensionality reduction for better clustering
svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X_topic)

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_reduced)

print(f"Cluster distribution: {Counter(cluster_labels)}")

# Analyze clusters by source
cluster_analysis = pd.DataFrame({
    'source': text_sources,
    'cluster': cluster_labels,
    'text': all_texts
})

print("\\n📊 Cluster Analysis by Source:")
for cluster_id in range(6):
    cluster_data = cluster_analysis[cluster_analysis['cluster'] == cluster_id]
    source_dist = cluster_data['source'].value_counts()
    print(f"Cluster {cluster_id}: {dict(source_dist)} - Sample: '{cluster_data['text'].iloc[0][:80]}...'")

# CHALLENGE 5: NAMED ENTITY RECOGNITION AND INFORMATION EXTRACTION
print("\n" + "=" * 70)
print("🏷️  CHALLENGE 5: NAMED ENTITY RECOGNITION & INFORMATION EXTRACTION")
print("=" * 70)

def extract_entities_nltk(text):
    \"\"\"Extract named entities using NLTK\"\"\"
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Named entity recognition
    entities = ne_chunk(pos_tags)

    named_entities = []
    for entity in entities:
        if hasattr(entity, 'label'):
            entity_name = ' '.join([token for token, pos in entity.leaves()])
            entity_type = entity.label()
            named_entities.append((entity_name, entity_type))

    return named_entities

def extract_entities_spacy(text, nlp_model):
    \"\"\"Extract named entities using spaCy\"\"\"
    if not nlp_model:
        return []

    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_patterns(text):
    \"\"\"Extract common patterns like emails, phones, URLs\"\"\"
    patterns = {}

    # Email addresses
    email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
    patterns['emails'] = re.findall(email_pattern, text)

    # Phone numbers (simple US format)
    phone_pattern = r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b'
    patterns['phones'] = re.findall(phone_pattern, text)

    # URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    patterns['urls'] = re.findall(url_pattern, text)

    # Dates (simple format)
    date_pattern = r'\\b\\d{1,2}/\\d{1,2}/\\d{4}\\b'
    patterns['dates'] = re.findall(date_pattern, text)

    return patterns

# Test entity extraction on news headlines
print("🔍 Testing Named Entity Recognition...")

sample_texts = [
    "Apple Inc. reported strong quarterly earnings. CEO Tim Cook will speak at the conference in New York on March 15, 2024.",
    "The Federal Reserve announced interest rate changes affecting Wall Street and major banks like JPMorgan Chase.",
    "Microsoft's partnership with OpenAI continues to advance artificial intelligence research in Seattle and San Francisco."
]

for i, text in enumerate(sample_texts, 1):
    print(f"\\n--- Sample Text {i} ---")
    print(f"Text: {text}")

    # NLTK entities
    nltk_entities = extract_entities_nltk(text)
    print(f"NLTK Entities: {nltk_entities}")

    # spaCy entities (if available)
    if preprocessor.spacy_available:
        spacy_entities = extract_entities_spacy(text, preprocessor.nlp)
        print(f"spaCy Entities: {spacy_entities}")

    # Pattern extraction
    patterns = extract_patterns(text)
    if any(patterns.values()):
        print(f"Patterns: {patterns}")

# CHALLENGE 6: TEXT SIMILARITY AND DOCUMENT MATCHING
print("\n" + "=" * 70)
print("🔗 CHALLENGE 6: TEXT SIMILARITY & DOCUMENT MATCHING")
print("=" * 70)

class TextSimilarityAnalyzer:
    \"\"\"Comprehensive text similarity analysis toolkit\"\"\"

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.is_fitted = False

    def fit(self, documents):
        \"\"\"Fit the vectorizer on document collection\"\"\"
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.documents = documents
        self.is_fitted = True
        return self

    def cosine_similarity_analysis(self, query_doc, top_k=5):
        \"\"\"Find most similar documents using cosine similarity\"\"\"
        if not self.is_fitted:
            raise ValueError("Analyzer must be fitted first")

        query_vector = self.tfidf_vectorizer.transform([query_doc])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top k most similar documents
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'index': idx,
                'similarity': similarities[idx],
                'document': self.documents[idx][:100] + '...' if len(self.documents[idx]) > 100 else self.documents[idx]
            })

        return results

    def jaccard_similarity(self, doc1, doc2):
        \"\"\"Calculate Jaccard similarity between two documents\"\"\"
        tokens1 = set(word_tokenize(doc1.lower()))
        tokens2 = set(word_tokenize(doc2.lower()))

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0

    def sequence_similarity(self, doc1, doc2):
        \"\"\"Calculate sequence similarity using difflib\"\"\"
        return SequenceMatcher(None, doc1, doc2).ratio()

# Test similarity analysis
print("🧪 Testing Text Similarity Analysis...")

# Use news headlines for similarity testing
similarity_analyzer = TextSimilarityAnalyzer()
similarity_analyzer.fit(news_df['headline'].tolist())

query_examples = [
    "Technology company announces new artificial intelligence breakthrough",
    "Stock market shows strong performance with record gains",
    "Healthcare workers receive recognition for outstanding service"
]

for query in query_examples:
    print(f"\\n🔍 Query: '{query}'")
    similar_docs = similarity_analyzer.cosine_similarity_analysis(query, top_k=3)

    for i, result in enumerate(similar_docs, 1):
        print(f"  {i}. Similarity: {result['similarity']:.3f} | {result['document']}")

# CHALLENGE 7: ADVANCED NLP PIPELINE AND BUSINESS APPLICATIONS
print("\n" + "=" * 70)
print("💼 CHALLENGE 7: BUSINESS NLP PIPELINE & REAL-WORLD APPLICATIONS")
print("=" * 70)

class BusinessNLPPipeline:
    \"\"\"Production-ready NLP pipeline for business applications\"\"\"

    def __init__(self):
        self.preprocessor = AdvancedTextPreprocessor()
        self.sentiment_model = None
        self.topic_model = None
        self.similarity_analyzer = None
        self.is_trained = False

    def train_pipeline(self, training_data):
        \"\"\"Train complete NLP pipeline on business data\"\"\"
        print("🏗️  Training Business NLP Pipeline...")

        # Prepare data
        texts = training_data['text'].tolist()
        labels = training_data.get('labels', None)

        # Train sentiment analysis model
        if labels is not None:
            print("  📊 Training sentiment classifier...")
            tfidf_vec = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
            X_features = tfidf_vec.fit_transform(texts)

            self.sentiment_model = {
                'vectorizer': tfidf_vec,
                'classifier': LogisticRegression(random_state=42)
            }

            label_enc = LabelEncoder()
            y_encoded = label_enc.fit_transform(labels)
            self.sentiment_model['label_encoder'] = label_enc
            self.sentiment_model['classifier'].fit(X_features, y_encoded)

        # Train topic model
        print("  📋 Training topic model...")
        topic_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', min_df=2)
        X_topic_features = topic_vectorizer.fit_transform(texts)

        self.topic_model = {
            'vectorizer': topic_vectorizer,
            'model': LatentDirichletAllocation(n_components=5, random_state=42)
        }
        self.topic_model['model'].fit(X_topic_features)

        # Train similarity analyzer
        print("  🔗 Training similarity analyzer...")
        self.similarity_analyzer = TextSimilarityAnalyzer()
        self.similarity_analyzer.fit(texts)

        self.is_trained = True
        print("✅ Pipeline training complete!")

    def analyze_document(self, text):
        \"\"\"Comprehensive analysis of a single document\"\"\"
        if not self.is_trained:
            raise ValueError("Pipeline must be trained first")

        results = {'original_text': text}

        # Text preprocessing and feature extraction
        preprocessing_result = self.preprocessor.preprocess_pipeline(text)
        results['preprocessed'] = preprocessing_result['final_text']
        results['features'] = self.preprocessor.extract_features(text)

        # Sentiment analysis
        if self.sentiment_model:
            text_vector = self.sentiment_model['vectorizer'].transform([text])
            sentiment_pred = self.sentiment_model['classifier'].predict(text_vector)[0]
            sentiment_proba = self.sentiment_model['classifier'].predict_proba(text_vector)[0]

            results['sentiment'] = {
                'predicted_class': self.sentiment_model['label_encoder'].inverse_transform([sentiment_pred])[0],
                'confidence': float(max(sentiment_proba)),
                'probabilities': dict(zip(
                    self.sentiment_model['label_encoder'].classes_,
                    sentiment_proba
                ))
            }

        # Topic analysis
        if self.topic_model:
            topic_vector = self.topic_model['vectorizer'].transform([text])
            topic_dist = self.topic_model['model'].transform(topic_vector)[0]
            dominant_topic = topic_dist.argmax()

            results['topics'] = {
                'dominant_topic': int(dominant_topic),
                'topic_confidence': float(topic_dist[dominant_topic]),
                'topic_distribution': {f'topic_{i}': float(prob) for i, prob in enumerate(topic_dist)}
            }

        # Find similar documents
        if self.similarity_analyzer:
            similar_docs = self.similarity_analyzer.cosine_similarity_analysis(text, top_k=3)
            results['similar_documents'] = similar_docs

        return results

    def batch_analyze(self, documents):
        \"\"\"Analyze multiple documents efficiently\"\"\"
        results = []
        for doc in documents:
            try:
                analysis = self.analyze_document(doc)
                results.append(analysis)
            except Exception as e:
                results.append({'error': str(e), 'original_text': doc})
        return results

    def generate_insights_report(self, documents):
        \"\"\"Generate business insights from document collection\"\"\"
        analyses = self.batch_analyze(documents)

        # Aggregate insights
        insights = {
            'total_documents': len(analyses),
            'sentiment_distribution': {},
            'topic_distribution': {},
            'avg_text_length': 0,
            'key_themes': []
        }

        # Calculate aggregated metrics
        sentiments = []
        topics = []
        text_lengths = []

        for analysis in analyses:
            if 'error' not in analysis:
                if 'sentiment' in analysis:
                    sentiments.append(analysis['sentiment']['predicted_class'])
                if 'topics' in analysis:
                    topics.append(analysis['topics']['dominant_topic'])
                text_lengths.append(analysis['features']['word_count'])

        # Sentiment distribution
        if sentiments:
            sentiment_counts = Counter(sentiments)
            insights['sentiment_distribution'] = dict(sentiment_counts)

        # Topic distribution
        if topics:
            topic_counts = Counter(topics)
            insights['topic_distribution'] = dict(topic_counts)

        # Average text length
        if text_lengths:
            insights['avg_text_length'] = np.mean(text_lengths)

        return insights

# Train and test business pipeline
print("🏢 Testing Business NLP Pipeline...")

# Prepare training data combining all sources
business_training_data = pd.DataFrame({
    'text': list(reviews_df['review']) + list(news_df['headline']) + list(social_df['post'][:100]),
    'labels': list(reviews_df['sentiment']) + ['news'] * len(news_df) + ['social'] * 100
})

# Initialize and train pipeline
nlp_pipeline = BusinessNLPPipeline()
nlp_pipeline.train_pipeline(business_training_data)

# Test on sample business documents
test_documents = [
    "Our customer service team received excellent feedback this quarter with 95% satisfaction ratings.",
    "The new product launch exceeded expectations with strong sales performance across all regions.",
    "Market research indicates growing demand for sustainable technology solutions in enterprise sector."
]

print("\\n📈 Business Document Analysis Results:")
for i, doc in enumerate(test_documents, 1):
    print(f"\\n--- Document {i} ---")
    analysis = nlp_pipeline.analyze_document(doc)

    print(f"Text: {doc}")
    if 'sentiment' in analysis:
        sent_info = analysis['sentiment']
        print(f"Sentiment: {sent_info['predicted_class']} (confidence: {sent_info['confidence']:.3f})")

    if 'topics' in analysis:
        topic_info = analysis['topics']
        print(f"Dominant Topic: {topic_info['dominant_topic']} (confidence: {topic_info['topic_confidence']:.3f})")

    print(f"Text Features: {analysis['features']['word_count']} words, Polarity: {analysis['features']['polarity']:.2f}")

# Generate business insights report
print("\\n📊 Generating Business Insights Report...")
insights = nlp_pipeline.generate_insights_report(test_documents + list(reviews_df['review'][:20]))

print("🎯 Business Insights Summary:")
print(f"• Total documents analyzed: {insights['total_documents']}")
print(f"• Sentiment distribution: {insights['sentiment_distribution']}")
print(f"• Topic distribution: {insights['topic_distribution']}")
print(f"• Average text length: {insights['avg_text_length']:.1f} words")

print("\\n" + "=" * 70)
print("🎉 CONGRATULATIONS! NLP AND TEXT ANALYTICS MASTERY COMPLETE!")
print("=" * 70)

print("\\n🏆 You have successfully mastered:")
print("• Advanced text preprocessing and normalization techniques")
print("• Multi-algorithm sentiment analysis and text classification")
print("• Topic modeling and document clustering with LDA and NMF")
print("• Named entity recognition and information extraction")
print("• Text similarity analysis and document matching")
print("• Production-ready business NLP pipeline development")
print("• Real-world text analytics applications for business insights")

print("\\n💼 Business Applications Mastered:")
print("• Customer feedback analysis and sentiment monitoring")
print("• Content categorization and automated tagging")
print("• Document similarity and recommendation systems")
print("• Information extraction from unstructured text")
print("• Market research and social media analytics")
print("• Competitive intelligence and brand monitoring")

print("\\n🚀 Ready for Advanced Applications:")
print("• Deep learning NLP with transformers and BERT")
print("• Multi-language text processing and translation")
print("• Real-time text streaming and analysis")
print("• Custom domain-specific NLP model development")
print("• Integration with business intelligence platforms")

print("\\n📈 Business Impact & ROI:")
print("• Automated content analysis saves 60-80% manual effort")
print("• Real-time sentiment monitoring improves customer satisfaction by 25-40%")
print("• Intelligent document processing reduces operational costs by 30-50%")
print("• Competitive analysis provides strategic insights worth millions in market positioning")

```

### Success Criteria

- Implement comprehensive text preprocessing and normalization pipelines
- Build and compare multiple sentiment analysis and text classification models
- Master topic modeling techniques (LDA, NMF) for document understanding
- Develop named entity recognition and information extraction systems
- Create text similarity and document matching capabilities
- Build production-ready business NLP pipelines with real-world applications

### Learning Objectives

- Understand modern NLP techniques and their business applications
- Master text preprocessing, feature engineering, and vectorization methods
- Learn advanced classification algorithms for text data
- Practice topic modeling and unsupervised document analysis
- Develop skills in information extraction and entity recognition
- Build comprehensive NLP systems for business intelligence and automation

### Business Impact & Real-World Applications

**🏢 Enterprise Use Cases:**

- **Customer Intelligence**: Analyze customer feedback, reviews, and support tickets to identify satisfaction trends, pain points, and improvement opportunities
- **Market Research**: Process social media data, news articles, and industry reports to track brand sentiment, competitive positioning, and market trends
- **Content Management**: Automatically categorize, tag, and route documents, emails, and communications for improved organizational efficiency
- **Risk Management**: Monitor regulatory documents, news, and communications to identify compliance risks and business threats

**💰 ROI Considerations:**

- NLP automation can reduce manual content analysis costs by 60-80% while improving accuracy and consistency
- Real-time sentiment monitoring enables proactive customer service, potentially improving satisfaction scores by 25-40%
- Intelligent document processing can cut operational overhead by 30-50% in content-heavy industries
- Business stakeholders should expect 2-4 month implementation cycles for production NLP systems with measurable ROI within 6-12 months

**📊 Key Success Metrics:**

- Processing speed: 1000+ documents per minute for batch analysis
- Classification accuracy: >90% for domain-specific sentiment and category classification
- Cost reduction: 50-70% decrease in manual content review and categorization effort
- Response time: Real-time analysis enabling immediate business decision making

---

_Pro tip: The most valuable NLP systems solve specific business problems rather than showcasing technical complexity. Always start with clear business objectives and measurable success criteria!_
