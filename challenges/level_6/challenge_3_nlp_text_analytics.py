"""
Level 6 - Challenge 3: NLP & Text Analytics
==========================================

Master Natural Language Processing and text analysis techniques.
This challenge covers text preprocessing, sentiment analysis, topic modeling, and text classification.

Learning Objectives:
- Master text preprocessing and feature extraction
- Implement sentiment analysis with multiple approaches
- Learn topic modeling techniques (LDA, NMF)
- Build text classification systems
- Explore advanced NLP techniques and word embeddings

Required Libraries: pandas, numpy, matplotlib, scikit-learn, nltk (optional)
"""

import re
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# Simple text preprocessing without external dependencies
def simple_preprocess_text(text: str) -> str:
    """
    Simple text preprocessing without external libraries.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


# Simple stopwords list
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "this",
    "these",
    "those",
    "am",
    "been",
    "being",
    "have",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "should",
    "could",
    "ought",
    "can",
    "may",
    "might",
    "must",
    "shall",
    "will",
    "get",
    "got",
    "getting",
}


def remove_stopwords(text: str) -> str:
    """Remove common stopwords from text."""
    words = text.split()
    return " ".join([word for word in words if word not in STOPWORDS and len(word) > 2])


def create_text_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Create comprehensive text datasets for NLP analysis.

    Returns:
        Dictionary containing various text datasets for NLP tasks
    """
    print("📝 Creating NLP & Text Analytics Datasets...")

    datasets = {}
    rng = np.random.default_rng(42)

    # 1. Movie Reviews Dataset (Sentiment Analysis)
    print("Creating movie reviews dataset...")

    # Positive movie review templates
    positive_templates = [
        "This movie was absolutely {adj1} and {adj2}. The {noun1} was {adj3} and the {noun2} kept me {verb1}.",
        "Amazing {noun1}! The director did a {adj1} job with the {noun2}. Highly {verb1} this film.",
        "One of the {adj1} movies I've ever seen. The {noun1} was {adj2} and the story was {adj3}.",
        "Fantastic {noun1} and {adj1} {noun2}. This movie will {verb1} you {adj2}.",
        "Brilliant {noun1}! The {adj1} {noun2} and {adj2} direction make this a must-watch film.",
        "Incredible movie with {adj1} {noun1} and {adj2} {noun2}. Absolutely {verb1} it!",
        "Outstanding {noun1} and {adj1} performances. This film {verb1} all expectations.",
        "Spectacular {noun1} with {adj1} visuals and {adj2} storytelling. {verb1} recommend!",
        "Excellent movie that {verb1} me completely. The {noun1} was {adj1} and {adj2}.",
        "Beautiful {noun1} and {adj1} {noun2}. This film is truly {adj2} and {verb1}.",
    ]

    positive_words = {
        "adj1": [
            "amazing",
            "brilliant",
            "fantastic",
            "excellent",
            "outstanding",
            "incredible",
            "wonderful",
            "spectacular",
        ],
        "adj2": [
            "engaging",
            "captivating",
            "thrilling",
            "inspiring",
            "moving",
            "emotional",
            "powerful",
            "beautiful",
        ],
        "adj3": [
            "superb",
            "perfect",
            "flawless",
            "remarkable",
            "exceptional",
            "magnificent",
            "stunning",
            "breathtaking",
        ],
        "noun1": [
            "acting",
            "cinematography",
            "direction",
            "storyline",
            "script",
            "performance",
            "plot",
            "narrative",
        ],
        "noun2": [
            "dialogue",
            "characters",
            "music",
            "effects",
            "scenes",
            "ending",
            "development",
            "pacing",
        ],
        "verb1": [
            "entertained",
            "engaged",
            "amazed",
            "impressed",
            "surprised",
            "moved",
            "inspired",
            "captivated",
        ],
    }

    # Negative movie review templates
    negative_templates = [
        "This movie was {adj1} and {adj2}. The {noun1} was {adj3} and I felt {verb1}.",
        "Terrible {noun1}! The director {verb1} to deliver a {adj1} story. {adj2} waste of time.",
        "One of the {adj1} movies I've ever seen. The {noun1} was {adj2} and the {noun2} was {adj3}.",
        "Awful {noun1} and {adj1} {noun2}. This movie will {verb1} you {adj2}.",
        "Poor {noun1}! The {adj1} {noun2} and {adj2} direction ruined this film completely.",
        "Disappointing movie with {adj1} {noun1} and {adj2} {noun2}. Definitely {verb1} it.",
        "Horrible {noun1} and {adj1} performances. This film {verb1} my expectations.",
        "Dreadful {noun1} with {adj1} visuals and {adj2} storytelling. Cannot {verb1}!",
        "Bad movie that {verb1} me completely. The {noun1} was {adj1} and {adj2}.",
        "Weak {noun1} and {adj1} {noun2}. This film is truly {adj2} and {verb1}.",
    ]

    negative_words = {
        "adj1": [
            "terrible",
            "awful",
            "horrible",
            "bad",
            "poor",
            "dreadful",
            "disappointing",
            "boring",
        ],
        "adj2": [
            "confusing",
            "predictable",
            "slow",
            "annoying",
            "frustrating",
            "pointless",
            "mediocre",
            "bland",
        ],
        "adj3": [
            "weak",
            "terrible",
            "unconvincing",
            "shallow",
            "poorly written",
            "cliched",
            "forced",
            "unrealistic",
        ],
        "noun1": [
            "acting",
            "plot",
            "direction",
            "storyline",
            "script",
            "dialogue",
            "characters",
            "pacing",
        ],
        "noun2": [
            "ending",
            "music",
            "effects",
            "scenes",
            "development",
            "execution",
            "delivery",
            "performance",
        ],
        "verb1": [
            "bored",
            "disappointed",
            "confused",
            "frustrated",
            "annoyed",
            "failed",
            "struggled",
            "disliked",
        ],
    }

    # Generate movie reviews
    reviews = []
    labels = []

    # Positive reviews
    for _ in range(1000):
        template = rng.choice(positive_templates)
        filled_review = template
        for placeholder, words in positive_words.items():
            filled_review = filled_review.replace(
                "{" + placeholder + "}", rng.choice(words)
            )

        # Add some noise and variety
        extra_positive = rng.choice(
            ["Great!", "Love it!", "Perfect!", "Awesome!", "Beautiful!", ""]
        )
        review = f"{filled_review} {extra_positive}".strip()

        reviews.append(review)
        labels.append(1)  # Positive

    # Negative reviews
    for _ in range(1000):
        template = rng.choice(negative_templates)
        filled_review = template
        for placeholder, words in negative_words.items():
            filled_review = filled_review.replace(
                "{" + placeholder + "}", rng.choice(words)
            )

        # Add some noise and variety
        extra_negative = rng.choice(
            ["Hate it!", "Avoid!", "Terrible!", "Waste of time!", "Skip this!", ""]
        )
        review = f"{filled_review} {extra_negative}".strip()

        reviews.append(review)
        labels.append(0)  # Negative

    # Shuffle the data
    combined = list(zip(reviews, labels))
    rng.shuffle(combined)
    reviews, labels = zip(*combined)

    movie_df = pd.DataFrame({"review": reviews, "sentiment": labels})

    datasets["movie_reviews"] = {
        "data": movie_df,
        "text_column": "review",
        "target_column": "sentiment",
        "task": "sentiment_analysis",
        "description": "Movie reviews for sentiment classification",
    }

    # 2. News Articles Dataset (Topic Classification)
    print("Creating news articles dataset...")

    news_topics = {
        "technology": {
            "keywords": [
                "software",
                "computer",
                "internet",
                "digital",
                "app",
                "data",
                "artificial intelligence",
                "machine learning",
                "blockchain",
                "cryptocurrency",
                "startup",
                "innovation",
                "tech company",
            ],
            "templates": [
                "A new {keyword1} company has developed innovative {keyword2} technology for {keyword3} applications.",
                "Researchers announce breakthrough in {keyword1} using advanced {keyword2} and {keyword3} methods.",
                "Major tech giant invests billions in {keyword1} development, focusing on {keyword2} and {keyword3}.",
                "Startup creates revolutionary {keyword1} platform that transforms {keyword2} and {keyword3} industries.",
                "Scientists develop cutting-edge {keyword1} solution for {keyword2} challenges in {keyword3} sector.",
            ],
        },
        "sports": {
            "keywords": [
                "football",
                "basketball",
                "soccer",
                "baseball",
                "tennis",
                "championship",
                "tournament",
                "player",
                "team",
                "coach",
                "victory",
                "defeat",
                "stadium",
                "match",
                "game",
            ],
            "templates": [
                "The {keyword1} {keyword2} won yesterday's crucial {keyword3} match against their rivals.",
                "Championship {keyword1} tournament features top {keyword2} competing for the {keyword3} title.",
                "Star {keyword1} player signs record-breaking contract with {keyword2} team for {keyword3} season.",
                "Coach announces new strategy for upcoming {keyword1} {keyword2} to improve {keyword3} performance.",
                "Historic {keyword1} {keyword2} takes place at the new {keyword3} stadium this weekend.",
            ],
        },
        "politics": {
            "keywords": [
                "government",
                "election",
                "policy",
                "parliament",
                "congress",
                "senator",
                "representative",
                "president",
                "minister",
                "legislation",
                "vote",
                "campaign",
                "democracy",
                "constitution",
            ],
            "templates": [
                "The {keyword1} announces new {keyword2} aimed at improving {keyword3} for citizens nationwide.",
                "Congressional {keyword1} debate focuses on controversial {keyword2} affecting {keyword3} policies.",
                "Presidential {keyword1} campaign promises significant changes to {keyword2} and {keyword3} systems.",
                "Parliament votes on important {keyword1} legislation regarding {keyword2} and {keyword3} reforms.",
                "Senator proposes bipartisan {keyword1} bill to address {keyword2} concerns in {keyword3} sector.",
            ],
        },
        "health": {
            "keywords": [
                "medical",
                "hospital",
                "doctor",
                "patient",
                "treatment",
                "vaccine",
                "disease",
                "research",
                "clinical",
                "pharmaceutical",
                "therapy",
                "diagnosis",
                "healthcare",
                "medicine",
                "wellness",
            ],
            "templates": [
                "Medical researchers develop new {keyword1} {keyword2} for treating {keyword3} in patients.",
                "Hospital introduces innovative {keyword1} program to improve {keyword2} care and {keyword3} outcomes.",
                "Clinical trial shows promising results for {keyword1} {keyword2} in {keyword3} treatment protocols.",
                "Healthcare system implements new {keyword1} technology to enhance {keyword2} and {keyword3} services.",
                "Pharmaceutical company receives approval for {keyword1} {keyword2} targeting {keyword3} conditions.",
            ],
        },
    }

    # Generate news articles
    articles = []
    categories = []

    for category, info in news_topics.items():
        for _ in range(400):  # 400 articles per category
            template = rng.choice(info["templates"])
            keywords = rng.choice(info["keywords"], size=3, replace=True)

            article = template.format(
                keyword1=keywords[0], keyword2=keywords[1], keyword3=keywords[2]
            )

            # Add some additional sentences for variety
            extra_sentences = [
                f"This development represents a significant advancement in {keywords[0]} research.",
                f"Experts believe this will have major implications for the {keywords[1]} industry.",
                f"The announcement has generated considerable interest among {keywords[2]} professionals.",
                f"Industry leaders praise the innovative approach to {keywords[0]} challenges.",
                f"This breakthrough could revolutionize how we approach {keywords[1]} in the future.",
            ]

            article += " " + rng.choice(extra_sentences)

            articles.append(article)
            categories.append(category)

    # Shuffle the data
    combined = list(zip(articles, categories))
    rng.shuffle(combined)
    articles, categories = zip(*combined)

    news_df = pd.DataFrame({"article": articles, "category": categories})

    datasets["news_articles"] = {
        "data": news_df,
        "text_column": "article",
        "target_column": "category",
        "task": "text_classification",
        "description": "News articles for topic classification",
    }

    # 3. Customer Reviews Dataset (Multi-class Sentiment)
    print("Creating customer reviews dataset...")

    # Rating-specific review templates
    rating_templates = {
        5: [  # Excellent
            "Absolutely {adj1}! This {product} exceeded all my expectations. The {feature1} is {adj2} and the {feature2} works {adv1}.",
            "Outstanding {product}! Love the {feature1} and {feature2}. {adj1} quality and {adj2} design. Highly recommend!",
            "Perfect {product} with {adj1} {feature1} and {adj2} {feature2}. Everything works {adv1}. Five stars!",
            "Exceptional {product}! The {feature1} is {adj1} and {feature2} is {adj2}. {adv1} worth the price!",
        ],
        4: [  # Good
            "Great {product} overall. The {feature1} is {adj1} but the {feature2} could be {adj2}. Still recommend it.",
            "Good {product} with {adj1} {feature1}. The {feature2} works {adv1} most of the time. Happy with purchase.",
            "Solid {product}. Love the {feature1} though the {feature2} is {adj1}. {adj2} value for money.",
            "Nice {product} with {adj1} {feature1}. {feature2} is {adj2} but not perfect. Worth buying.",
        ],
        3: [  # Average
            "Average {product}. The {feature1} is {adj1} and {feature2} is {adj2}. Nothing special but works fine.",
            "Decent {product}. {feature1} is {adj1} but {feature2} could be better. {adj2} for the price.",
            "Okay {product}. The {feature1} works {adv1} but {feature2} is {adj1}. {adj2} overall experience.",
            "Fair {product}. {feature1} is {adj1}, {feature2} is {adj2}. Gets the job done adequately.",
        ],
        2: [  # Poor
            "Disappointing {product}. The {feature1} is {adj1} and {feature2} doesn't work {adv1}. {adj2} quality.",
            "Poor {product}. {feature1} is {adj1} and {feature2} is {adj2}. Not worth the money spent.",
            "Below average {product}. The {feature1} is {adj1} but {feature2} is {adj2}. Many issues encountered.",
            "Unsatisfactory {product}. {feature1} works {adv1} and {feature2} is {adj1}. {adj2} experience overall.",
        ],
        1: [  # Terrible
            "Terrible {product}! The {feature1} is {adj1} and {feature2} is completely {adj2}. Total waste of money!",
            "Awful {product}. {feature1} doesn't work at all and {feature2} is {adj1}. {adj2} quality. Avoid!",
            "Horrible {product}! Both {feature1} and {feature2} are {adj1}. {adj2} design and construction.",
            "Dreadful {product}. The {feature1} is {adj1}, {feature2} is {adj2}. Completely {adv1} disappointed.",
        ],
    }

    rating_words = {
        5: {  # Excellent
            "adj1": [
                "amazing",
                "fantastic",
                "excellent",
                "outstanding",
                "perfect",
                "brilliant",
                "superb",
            ],
            "adj2": [
                "flawless",
                "incredible",
                "wonderful",
                "magnificent",
                "exceptional",
                "remarkable",
            ],
            "adv1": [
                "perfectly",
                "flawlessly",
                "beautifully",
                "smoothly",
                "excellently",
                "wonderfully",
            ],
        },
        4: {  # Good
            "adj1": [
                "good",
                "nice",
                "solid",
                "decent",
                "fine",
                "satisfactory",
                "acceptable",
            ],
            "adj2": [
                "better",
                "improved",
                "enhanced",
                "upgraded",
                "refined",
                "optimized",
            ],
            "adv1": [
                "well",
                "nicely",
                "properly",
                "adequately",
                "satisfactorily",
                "reasonably",
            ],
        },
        3: {  # Average
            "adj1": [
                "okay",
                "average",
                "decent",
                "fair",
                "reasonable",
                "acceptable",
                "moderate",
            ],
            "adj2": [
                "adequate",
                "sufficient",
                "passable",
                "tolerable",
                "mediocre",
                "standard",
            ],
            "adv1": [
                "adequately",
                "sufficiently",
                "reasonably",
                "moderately",
                "fairly",
                "decently",
            ],
        },
        2: {  # Poor
            "adj1": [
                "poor",
                "bad",
                "disappointing",
                "unsatisfactory",
                "subpar",
                "inadequate",
            ],
            "adj2": [
                "problematic",
                "faulty",
                "defective",
                "unreliable",
                "inconsistent",
                "flawed",
            ],
            "adv1": [
                "poorly",
                "badly",
                "inadequately",
                "unsatisfactorily",
                "inconsistently",
            ],
        },
        1: {  # Terrible
            "adj1": [
                "terrible",
                "awful",
                "horrible",
                "dreadful",
                "useless",
                "broken",
                "defective",
            ],
            "adj2": [
                "unusable",
                "worthless",
                "garbage",
                "junk",
                "defective",
                "broken",
                "faulty",
            ],
            "adv1": [
                "horribly",
                "terribly",
                "completely",
                "utterly",
                "absolutely",
                "totally",
            ],
        },
    }

    products = [
        "laptop",
        "phone",
        "headphones",
        "camera",
        "tablet",
        "speaker",
        "watch",
        "keyboard",
    ]
    features = [
        "battery",
        "screen",
        "sound",
        "design",
        "performance",
        "build quality",
        "interface",
        "connectivity",
    ]

    # Generate customer reviews
    reviews = []
    ratings = []

    for rating in [1, 2, 3, 4, 5]:
        for _ in range(400):  # 400 reviews per rating
            template = rng.choice(rating_templates[rating])
            product = rng.choice(products)
            feature1, feature2 = rng.choice(features, size=2, replace=False)

            words = rating_words[rating]
            review = template.format(
                product=product,
                feature1=feature1,
                feature2=feature2,
                adj1=rng.choice(words["adj1"]),
                adj2=rng.choice(words["adj2"]),
                adv1=rng.choice(words["adv1"]),
            )

            reviews.append(review)
            ratings.append(rating)

    # Shuffle the data
    combined = list(zip(reviews, ratings))
    rng.shuffle(combined)
    reviews, ratings = zip(*combined)

    customer_df = pd.DataFrame({"review": reviews, "rating": ratings})

    datasets["customer_reviews"] = {
        "data": customer_df,
        "text_column": "review",
        "target_column": "rating",
        "task": "multiclass_sentiment",
        "description": "Customer reviews with 1-5 star ratings",
    }

    print(f"Created {len(datasets)} text datasets for NLP analysis")
    return datasets


def text_preprocessing_pipeline(
    texts: List[str], remove_stopwords_flag: bool = True
) -> List[str]:
    """
    Apply comprehensive text preprocessing pipeline.
    """
    print("\n🔧 Text Preprocessing Pipeline")
    print("=" * 40)

    processed_texts = []

    print(f"• Processing {len(texts)} texts...")

    for text in texts:
        # Basic preprocessing
        processed = simple_preprocess_text(text)

        # Remove stopwords if requested
        if remove_stopwords_flag:
            processed = remove_stopwords(processed)

        processed_texts.append(processed)

    # Show preprocessing statistics
    original_lengths = [len(text.split()) for text in texts]
    processed_lengths = [len(text.split()) for text in processed_texts]

    print(f"• Average words before preprocessing: {np.mean(original_lengths):.1f}")
    print(f"• Average words after preprocessing: {np.mean(processed_lengths):.1f}")
    print(
        f"• Reduction: {(1 - np.mean(processed_lengths)/np.mean(original_lengths))*100:.1f}%"
    )

    return processed_texts


def extract_text_features(
    texts: List[str], method: str = "tfidf", max_features: int = 1000
):
    """
    Extract features from text using different methods.
    """
    print(f"\n📊 Feature Extraction: {method.upper()}")
    print("=" * 40)

    if method == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
        )
    elif method == "count":
        vectorizer = CountVectorizer(
            max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.8
        )
    else:
        raise ValueError("Method must be 'tfidf' or 'count'")

    # Fit and transform texts
    X = vectorizer.fit_transform(texts)

    print(f"• Feature matrix shape: {X.shape}")
    print(f"• Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"• Sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1]))*100:.1f}%")

    # Show top features by variance
    if hasattr(vectorizer, "idf_"):
        # For TF-IDF, show words with highest IDF (most discriminative)
        feature_names = vectorizer.get_feature_names_out()
        idf_scores = vectorizer.idf_
        top_indices = np.argsort(idf_scores)[-10:]
        print("• Top discriminative features:", [feature_names[i] for i in top_indices])

    return X, vectorizer


def sentiment_analysis(
    data: pd.DataFrame, text_col: str, target_col: str
) -> Dict[str, Any]:
    """
    Perform comprehensive sentiment analysis.
    """
    print("\n❤️ Sentiment Analysis")
    print("=" * 40)

    texts = data[text_col].tolist()
    labels = data[target_col].values

    # Preprocess texts
    processed_texts = text_preprocessing_pipeline(texts)

    # Extract features
    X_tfidf, tfidf_vectorizer = extract_text_features(
        processed_texts, method="tfidf", max_features=5000
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\n• Training samples: {X_train.shape[0]}")
    print(f"• Testing samples: {X_test.shape[0]}")

    # Train multiple classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="linear", random_state=42),
    }

    results = {}

    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")

        # Train classifier
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Store results
        results[name] = {
            "classifier": classifier,
            "accuracy": accuracy,
            "predictions": y_pred,
            "y_test": y_test,
        }

        print(f"• {name} Accuracy: {accuracy:.3f}")

    # Visualize results
    plt.figure(figsize=(12, 8))

    # Accuracy comparison
    plt.subplot(2, 2, 1)
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in names]

    plt.bar(
        names,
        accuracies,
        alpha=0.7,
        color=["skyblue", "lightcoral", "lightgreen", "gold"],
    )
    plt.title("Sentiment Analysis Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Feature importance for Logistic Regression
    plt.subplot(2, 2, 2)
    if "Logistic Regression" in results:
        lr_model = results["Logistic Regression"]["classifier"]
        feature_names = tfidf_vectorizer.get_feature_names_out()

        if hasattr(lr_model, "coef_"):
            coefficients = (
                lr_model.coef_[0] if len(lr_model.coef_.shape) > 1 else lr_model.coef_
            )

            # Get top positive and negative features
            top_positive_idx = np.argsort(coefficients)[-10:]
            top_negative_idx = np.argsort(coefficients)[:10]

            top_features = [feature_names[i] for i in top_positive_idx] + [
                feature_names[i] for i in top_negative_idx
            ]
            top_coeffs = list(coefficients[top_positive_idx]) + list(
                coefficients[top_negative_idx]
            )

            colors = ["green" if c > 0 else "red" for c in top_coeffs]

            plt.barh(range(len(top_features)), top_coeffs, color=colors, alpha=0.7)
            plt.yticks(range(len(top_features)), top_features)
            plt.title("Top Features (Logistic Regression)")
            plt.xlabel("Coefficient Value")

    # Confusion matrix for best model
    plt.subplot(2, 2, 3)
    best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_results = results[best_model_name]

    cm = confusion_matrix(best_results["y_test"], best_results["predictions"])

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

    # Prediction distribution
    plt.subplot(2, 2, 4)
    unique_labels, counts = np.unique(best_results["predictions"], return_counts=True)
    plt.pie(
        counts, labels=[f"Class {label}" for label in unique_labels], autopct="%1.1f%%"
    )
    plt.title(f"Prediction Distribution - {best_model_name}")

    plt.tight_layout()
    plt.show()

    results["vectorizer"] = tfidf_vectorizer
    results["processed_texts"] = processed_texts

    return results


def topic_modeling(
    texts: List[str], n_topics: int = 5, method: str = "lda"
) -> Dict[str, Any]:
    """
    Perform topic modeling on text data.
    """
    print(f"\n🎯 Topic Modeling: {method.upper()}")
    print("=" * 40)

    # Preprocess texts
    processed_texts = text_preprocessing_pipeline(texts, remove_stopwords_flag=True)

    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=5,  # Ignore terms that appear in less than 5 documents
        max_df=0.7,  # Ignore terms that appear in more than 70% of documents
        ngram_range=(1, 1),  # Only unigrams for topic modeling
    )

    X = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()

    print(f"• Documents: {X.shape[0]}")
    print(f"• Vocabulary size: {X.shape[1]}")

    # Apply topic modeling
    if method == "lda":
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
        )
    elif method == "nmf":
        model = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=200,
            alpha_W=0.1,
            alpha_H=0.1,
        )
    else:
        raise ValueError("Method must be 'lda' or 'nmf'")

    # Fit model
    model.fit(X)

    # Get topic-word distributions
    if method == "lda":
        topic_word_dist = model.components_
    else:  # NMF
        topic_word_dist = model.components_

    # Extract top words for each topic
    n_top_words = 10
    topics = []

    print(f"\n📋 Discovered Topics:")
    for topic_idx, topic in enumerate(topic_word_dist):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]

        topics.append({"words": top_words, "weights": top_weights})

        print(f"\nTopic {topic_idx + 1}: {', '.join(top_words[:5])}")
        print(f"  Full: {', '.join(top_words)}")

    # Get document-topic distributions
    doc_topic_dist = model.transform(X)

    # Assign dominant topic to each document
    dominant_topics = np.argmax(doc_topic_dist, axis=1)

    # Visualize topics
    plt.figure(figsize=(15, 10))

    # Topic word distributions
    for i in range(min(n_topics, 6)):  # Show up to 6 topics
        plt.subplot(2, 3, i + 1)

        if i < len(topics):
            words = topics[i]["words"][:8]  # Top 8 words
            weights = topics[i]["weights"][:8]

            plt.barh(range(len(words)), weights, alpha=0.7)
            plt.yticks(range(len(words)), words)
            plt.title(f"Topic {i + 1}")
            plt.xlabel("Weight")
            plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

    # Topic distribution visualization
    plt.figure(figsize=(12, 6))

    # Document distribution across topics
    plt.subplot(1, 2, 1)
    unique_topics, topic_counts = np.unique(dominant_topics, return_counts=True)
    plt.bar(unique_topics + 1, topic_counts, alpha=0.7, color="skyblue")
    plt.xlabel("Topic")
    plt.ylabel("Number of Documents")
    plt.title("Document Distribution Across Topics")
    plt.xticks(unique_topics + 1)
    plt.grid(True, alpha=0.3)

    # Average topic coherence (document-topic strength)
    plt.subplot(1, 2, 2)
    topic_strengths = []
    for topic_idx in range(n_topics):
        # Get documents where this topic is dominant
        topic_docs = doc_topic_dist[dominant_topics == topic_idx]
        if len(topic_docs) > 0:
            # Average strength of this topic in its dominant documents
            avg_strength = np.mean(topic_docs[:, topic_idx])
            topic_strengths.append(avg_strength)
        else:
            topic_strengths.append(0)

    plt.bar(range(1, n_topics + 1), topic_strengths, alpha=0.7, color="lightcoral")
    plt.xlabel("Topic")
    plt.ylabel("Average Topic Strength")
    plt.title("Topic Coherence")
    plt.xticks(range(1, n_topics + 1))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "model": model,
        "vectorizer": vectorizer,
        "topics": topics,
        "doc_topic_dist": doc_topic_dist,
        "dominant_topics": dominant_topics,
        "feature_names": feature_names,
    }


def text_classification(
    data: pd.DataFrame, text_col: str, target_col: str
) -> Dict[str, Any]:
    """
    Perform multi-class text classification.
    """
    print("\n📚 Text Classification")
    print("=" * 40)

    texts = data[text_col].tolist()
    labels = data[target_col].values

    # Show class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"• Classes: {len(unique_labels)}")
    print(f"• Samples per class: {dict(zip(unique_labels, counts))}")

    # Preprocess texts
    processed_texts = text_preprocessing_pipeline(texts)

    # Extract features
    X, vectorizer = extract_text_features(
        processed_texts, method="tfidf", max_features=3000
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")

        # Train classifier
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test)

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
        target_names_str = [str(label) for label in unique_labels]
        print(classification_report(y_test, y_pred, target_names=target_names_str))

    # Visualize results
    plt.figure(figsize=(15, 10))

    # Accuracy comparison
    plt.subplot(2, 3, 1)
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in names]

    plt.bar(names, accuracies, alpha=0.7, color=["skyblue", "lightcoral", "lightgreen"])
    plt.title("Classification Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Confusion matrices for each classifier
    for idx, (name, result) in enumerate(results.items()):
        plt.subplot(2, 3, idx + 2)

        cm = confusion_matrix(result["y_test"], result["predictions"])

        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.colorbar()

        # Add class labels
        tick_marks = np.arange(len(unique_labels))
        plt.xticks(tick_marks, unique_labels, rotation=45)
        plt.yticks(tick_marks, unique_labels)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

    # Class distribution
    plt.subplot(2, 3, 6)
    plt.bar(unique_labels, counts, alpha=0.7, color="gold")
    plt.title("Class Distribution in Dataset")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    results["vectorizer"] = vectorizer
    results["processed_texts"] = processed_texts
    results["unique_labels"] = unique_labels

    return results


def run_nlp_challenges() -> None:
    """
    Run all NLP and text analytics challenges.
    """
    print("🚀 Starting Level 6 Challenge 3: NLP & Text Analytics")
    print("=" * 60)

    try:
        # Challenge 1: Create text datasets
        print("\n" + "=" * 50)
        print("CHALLENGE 1: Text Dataset Creation")
        print("=" * 50)

        datasets = create_text_datasets()

        print(f"\n✅ Created {len(datasets)} text datasets:")
        for name, info in datasets.items():
            data = info["data"]
            text_col = info["text_column"]
            target_col = info["target_column"]
            task = info["task"]

            print(f"• {name}: {len(data)} samples")
            print(f"  Task: {task}")
            print(f"  Text column: {text_col}")
            print(f"  Target: {target_col}")

            if task == "sentiment_analysis":
                pos_count = (data[target_col] == 1).sum()
                print(f"  Positive: {pos_count}, Negative: {len(data) - pos_count}")
            elif task == "text_classification" or task == "multiclass_sentiment":
                class_dist = data[target_col].value_counts()
                print(f"  Classes: {dict(class_dist)}")

        # Challenge 2: Sentiment Analysis
        print("\n" + "=" * 50)
        print("CHALLENGE 2: Sentiment Analysis")
        print("=" * 50)

        movie_data = datasets["movie_reviews"]
        sentiment_results = sentiment_analysis(
            movie_data["data"], movie_data["text_column"], movie_data["target_column"]
        )

        print("\n✅ Sentiment Analysis Complete")
        # Find best model accuracy from results
        accuracies = []
        for key, result in sentiment_results.items():
            if isinstance(result, dict) and "accuracy" in result:
                accuracies.append(result["accuracy"])

        if accuracies:
            print(f"• Best model accuracy: {max(accuracies):.3f}")
        else:
            print("• Sentiment analysis models trained successfully")

        # Challenge 3: Topic Modeling
        print("\n" + "=" * 50)
        print("CHALLENGE 3: Topic Modeling")
        print("=" * 50)

        # Use news articles for topic modeling
        news_data = datasets["news_articles"]
        articles = news_data["data"][news_data["text_column"]].tolist()

        # LDA Topic Modeling
        print("\nApplying LDA Topic Modeling...")
        lda_results = topic_modeling(articles, n_topics=4, method="lda")

        print("\n✅ LDA Topic Modeling Complete")
        print(f"• Discovered {len(lda_results['topics'])} topics")

        # NMF Topic Modeling for comparison
        print("\nApplying NMF Topic Modeling...")
        nmf_results = topic_modeling(articles, n_topics=4, method="nmf")

        print("\n✅ NMF Topic Modeling Complete")
        print(f"• Discovered {len(nmf_results['topics'])} topics")

        # Challenge 4: Multi-class Text Classification
        print("\n" + "=" * 50)
        print("CHALLENGE 4: Multi-class Text Classification")
        print("=" * 50)

        # News classification
        print("News Article Classification:")
        news_classification_results = text_classification(
            news_data["data"], news_data["text_column"], news_data["target_column"]
        )

        # Customer review rating prediction
        print("\nCustomer Review Rating Prediction:")
        customer_data = datasets["customer_reviews"]
        customer_classification_results = text_classification(
            customer_data["data"],
            customer_data["text_column"],
            customer_data["target_column"],
        )

        print("\n" + "🎉" * 20)
        print("LEVEL 6 CHALLENGE 3 COMPLETE!")
        print("🎉" * 20)

        print("\n📚 What You've Learned:")
        print("• Text preprocessing and feature extraction techniques")
        print("• Sentiment analysis with multiple machine learning approaches")
        print("• Topic modeling using LDA and NMF methods")
        print("• Multi-class text classification for various domains")
        print("• Performance evaluation for NLP tasks")

        print("\n🚀 Next Steps:")
        print("• Explore word embeddings and neural language models")
        print("• Learn advanced NLP techniques (Named Entity Recognition, POS tagging)")
        print("• Study transformer models and attention mechanisms")
        print("• Apply to real-world text mining and document analysis")
        print("• Move to Level 6 Challenge 4: Computer Vision")

        return datasets

    except Exception as e:
        print(f"❌ Error in NLP challenges: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete NLP and text analytics challenge
    datasets = run_nlp_challenges()

    if datasets:
        print("\n" + "=" * 60)
        print("NLP & TEXT ANALYTICS CHALLENGE SUMMARY")
        print("=" * 60)

        print("\nDatasets Created:")
        for name, info in datasets.items():
            data = info["data"]
            task = info["task"]
            print(f"• {name}: {len(data)} samples ({task})")

        print("\nKey NLP Concepts Covered:")
        concepts = [
            "Text preprocessing and cleaning techniques",
            "Feature extraction (TF-IDF, Count Vectorization)",
            "Sentiment analysis and polarity detection",
            "Topic modeling and document clustering",
            "Multi-class text classification",
            "Model evaluation for NLP tasks",
        ]

        for i, concept in enumerate(concepts, 1):
            print(f"{i}. {concept}")

        print("\n✨ Ready for Level 6 Challenge 4: Computer Vision!")
