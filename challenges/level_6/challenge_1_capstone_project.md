# Level 6: Data Science Master

## Challenge 1: Capstone Project - Real-World Data Science Pipeline

Welcome to the ultimate challenge! Build an end-to-end data science solution combining multiple domains and advanced techniques.

### Objective
Create a comprehensive data science project that demonstrates mastery across all domains: data engineering, analysis, machine learning, NLP, time series, and deployment.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Time series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Text processing
import re
import nltk
try:
    from textblob import TextBlob
except ImportError:
    print("TextBlob not available, using basic text processing")
    TextBlob = None

# Deep learning (if available)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available, using classical ML only")
    TF_AVAILABLE = False

print("🏆 CAPSTONE PROJECT: E-Commerce Intelligence Platform")
print("====================================================")
print("Building a complete data science solution for business intelligence")

# Your tasks:
# 1. DATA ENGINEERING - Create Realistic Multi-Domain Dataset
print("\n=== DATA ENGINEERING & SIMULATION ===")

np.random.seed(42)

# Simulate 2 years of e-commerce data
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

print("🏗️ Generating realistic e-commerce dataset...")

# Product catalog
products = {
    'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera'],
    'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket'],
    'Home': ['Chair', 'Table', 'Lamp', 'Curtains', 'Rug'],
    'Sports': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Bicycle', 'Tennis Racket'],
    'Books': ['Fiction', 'Non-Fiction', 'Textbook', 'Comics', 'Biography']
}

# Generate transaction data
transactions = []
customer_reviews = []
customer_id = 1000

for date in date_range:
    # Seasonal effects
    month = date.month
    is_holiday = month in [11, 12, 1]  # Holiday season
    is_summer = month in [6, 7, 8]    # Summer season
    
    # Daily transaction volume (with seasonality)
    base_volume = 100
    if is_holiday:
        daily_volume = int(base_volume * np.random.uniform(1.5, 2.5))
    elif is_summer:
        daily_volume = int(base_volume * np.random.uniform(1.2, 1.8))
    else:
        daily_volume = int(base_volume * np.random.uniform(0.8, 1.2))
    
    # Weekend effect
    if date.weekday() >= 5:  # Weekend
        daily_volume = int(daily_volume * 1.3)
    
    for _ in range(daily_volume):
        # Customer demographics
        age = np.random.normal(35, 15)
        age = max(18, min(80, age))  # Clip to reasonable range
        
        gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
        
        # Location bias
        regions = ['North', 'South', 'East', 'West', 'Central']
        region_weights = [0.25, 0.20, 0.22, 0.18, 0.15]
        region = np.random.choice(regions, p=region_weights)
        
        # Product selection (with category preferences by demographics)
        category_prefs = {
            'Electronics': 1.0 if age < 40 else 0.6,
            'Clothing': 1.2 if gender == 'F' else 0.8,
            'Home': 1.0 if age > 30 else 0.5,
            'Sports': 1.0 if age < 50 else 0.4,
            'Books': 0.8 + (age - 30) * 0.01
        }
        
        # Select category based on preferences
        categories = list(category_prefs.keys())
        weights = [max(0.1, category_prefs[cat]) for cat in categories]
        weights = [w/sum(weights) for w in weights]  # Normalize
        
        category = np.random.choice(categories, p=weights)
        product = np.random.choice(products[category])
        
        # Price modeling
        base_prices = {
            'Electronics': np.random.uniform(200, 1500),
            'Clothing': np.random.uniform(20, 200),
            'Home': np.random.uniform(50, 800),
            'Sports': np.random.uniform(30, 500),
            'Books': np.random.uniform(10, 50)
        }
        
        price = base_prices[category]
        
        # Discount effects
        if is_holiday:
            discount = np.random.uniform(0.1, 0.4)
            price *= (1 - discount)
        
        # Customer satisfaction modeling
        satisfaction_base = 4.0  # Base satisfaction
        satisfaction = satisfaction_base + np.random.normal(0, 0.8)
        satisfaction = max(1, min(5, satisfaction))
        
        # Purchase decision based on price sensitivity
        price_sensitivity = np.random.uniform(0.5, 1.5)
        purchase_probability = 1 / (1 + np.exp((price - 300) / 200 * price_sensitivity))
        
        if np.random.random() < purchase_probability:
            transaction = {
                'transaction_id': len(transactions) + 1,
                'date': date,
                'customer_id': customer_id,
                'age': int(age),
                'gender': gender,
                'region': region,
                'category': category,
                'product': product,
                'price': round(price, 2),
                'satisfaction': round(satisfaction, 1),
                'quantity': np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
            }
            
            transactions.append(transaction)
            
            # Generate customer review text
            if np.random.random() < 0.3:  # 30% leave reviews
                sentiment_words = {
                    1: ['terrible', 'awful', 'worst', 'horrible', 'disappointing'],
                    2: ['bad', 'poor', 'unsatisfied', 'below expectations'],
                    3: ['okay', 'average', 'acceptable', 'decent'],
                    4: ['good', 'satisfied', 'nice', 'pleased', 'happy'],
                    5: ['excellent', 'amazing', 'fantastic', 'outstanding', 'perfect']
                }
                
                rating = int(round(satisfaction))
                sentiment = np.random.choice(sentiment_words[rating])
                
                review_templates = [
                    f"The {product.lower()} was {sentiment}. {'Would recommend!' if rating >= 4 else 'Not satisfied.'}",
                    f"{sentiment.capitalize()} quality {product.lower()}. {'Great purchase' if rating >= 4 else 'Could be better'}.",
                    f"I found this {product.lower()} to be {sentiment}. {'Will buy again' if rating >= 4 else 'Looking for alternatives'}."
                ]
                
                review_text = np.random.choice(review_templates)
                
                review = {
                    'transaction_id': transaction['transaction_id'],
                    'customer_id': customer_id,
                    'product': product,
                    'rating': rating,
                    'review_text': review_text,
                    'review_date': date + timedelta(days=np.random.randint(1, 30))
                }
                
                customer_reviews.append(review)
        
        customer_id += 1

# Convert to DataFrames
df_transactions = pd.DataFrame(transactions)
df_reviews = pd.DataFrame(customer_reviews)

print(f"✅ Generated {len(transactions):,} transactions")
print(f"✅ Generated {len(customer_reviews):,} customer reviews")
print(f"📊 Dataset covers {len(date_range)} days from {start_date.date()} to {end_date.date()}")

# Display sample data
print("\n📋 Sample Transaction Data:")
print(df_transactions.head())
print(f"\nTransaction Data Shape: {df_transactions.shape}")
print(f"Transaction Data Columns: {list(df_transactions.columns)}")

print("\n💬 Sample Review Data:")
print(df_reviews.head())
print(f"\nReview Data Shape: {df_reviews.shape}")

# 2. EXPLORATORY DATA ANALYSIS
print("\n=== EXPLORATORY DATA ANALYSIS ===")

# Business metrics
total_revenue = (df_transactions['price'] * df_transactions['quantity']).sum()
avg_order_value = (df_transactions['price'] * df_transactions['quantity']).mean()
unique_customers = df_transactions['customer_id'].nunique()

print(f"💰 Total Revenue: ${total_revenue:,.2f}")
print(f"🛒 Average Order Value: ${avg_order_value:.2f}")
print(f"👥 Unique Customers: {unique_customers:,}")

# Time series analysis
daily_revenue = df_transactions.groupby('date').apply(
    lambda x: (x['price'] * x['quantity']).sum()
).reset_index()
daily_revenue.columns = ['date', 'revenue']

# Seasonal decomposition
print("\n📈 Time Series Analysis:")
ts_data = daily_revenue.set_index('date')['revenue']
decomposition = seasonal_decompose(ts_data, model='additive', period=30)

fig, axes = plt.subplots(4, 1, figsize=(15, 12))
fig.suptitle('Revenue Time Series Decomposition', fontsize=16)

decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')

plt.tight_layout()
plt.show()

# Customer segmentation analysis
print("\n👥 Customer Segmentation:")
customer_metrics = df_transactions.groupby('customer_id').agg({
    'price': ['count', 'sum', 'mean'],
    'quantity': 'sum',
    'satisfaction': 'mean'
}).round(2)

customer_metrics.columns = ['transaction_count', 'total_spent', 'avg_order_value', 'total_items', 'avg_satisfaction']
customer_metrics = customer_metrics.reset_index()

# RFM Analysis (Recency, Frequency, Monetary)
last_date = df_transactions['date'].max()
rfm_data = df_transactions.groupby('customer_id').agg({
    'date': lambda x: (last_date - x.max()).days,  # Recency
    'transaction_id': 'count',  # Frequency
    'price': 'sum'  # Monetary
}).reset_index()
rfm_data.columns = ['customer_id', 'recency', 'frequency', 'monetary']

print(f"📊 Customer Metrics Summary:")
print(customer_metrics.describe())

# 3. PREDICTIVE MODELING
print("\n=== PREDICTIVE MODELING ===")

# Customer Lifetime Value Prediction
print("💎 Building Customer Lifetime Value Model:")

# Prepare features for CLV prediction
clv_features = df_transactions.groupby('customer_id').agg({
    'price': ['sum', 'mean', 'std'],
    'quantity': ['sum', 'mean'],
    'satisfaction': 'mean',
    'transaction_id': 'count',
    'date': ['min', 'max']
}).reset_index()

# Flatten column names
clv_features.columns = ['customer_id', 'total_spent', 'avg_price', 'price_std', 
                       'total_quantity', 'avg_quantity', 'avg_satisfaction', 
                       'purchase_frequency', 'first_purchase', 'last_purchase']

# Fill NaN values
clv_features['price_std'] = clv_features['price_std'].fillna(0)

# Customer lifetime (days)
clv_features['customer_lifetime_days'] = (clv_features['last_purchase'] - clv_features['first_purchase']).dt.days
clv_features['customer_lifetime_days'] = clv_features['customer_lifetime_days'].fillna(0)

# Target: Future value (simplified as current total_spent)
target = clv_features['total_spent']

# Features for modeling
feature_cols = ['avg_price', 'price_std', 'total_quantity', 'avg_quantity', 
               'avg_satisfaction', 'purchase_frequency', 'customer_lifetime_days']
X_clv = clv_features[feature_cols]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_clv, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train CLV model
clv_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
clv_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_clv = clv_model.predict(X_test_scaled)

# Evaluate
clv_mae = mean_absolute_error(y_test, y_pred_clv)
clv_r2 = r2_score(y_test, y_pred_clv)

print(f"✅ CLV Model Performance:")
print(f"   Mean Absolute Error: ${clv_mae:.2f}")
print(f"   R² Score: {clv_r2:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': clv_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"🔍 Top CLV Predictors:")
for _, row in feature_importance.head().iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

# 4. SENTIMENT ANALYSIS & NLP
print("\n=== SENTIMENT ANALYSIS & NLP ===")

if len(df_reviews) > 0:
    print("💬 Analyzing Customer Reviews:")
    
    # Text preprocessing
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    df_reviews['clean_text'] = df_reviews['review_text'].apply(clean_text)
    
    # Basic sentiment analysis using ratings as ground truth
    df_reviews['sentiment'] = df_reviews['rating'].apply(
        lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
    )
    
    print(f"📊 Sentiment Distribution:")
    sentiment_dist = df_reviews['sentiment'].value_counts()
    print(sentiment_dist)
    
    # Build sentiment classifier
    if len(df_reviews) >= 100:  # Need minimum data for modeling
        # Prepare data for sentiment modeling
        X_text = df_reviews['clean_text']
        y_sentiment = df_reviews['sentiment']
        
        # Create text processing pipeline
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        # Train sentiment model
        X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
            X_text, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
        )
        
        text_pipeline.fit(X_train_text, y_train_text)
        
        # Evaluate
        text_accuracy = text_pipeline.score(X_test_text, y_test_text)
        print(f"✅ Sentiment Classification Accuracy: {text_accuracy:.3f}")
        
        # Analyze sentiment by product category
        review_category_sentiment = df_reviews.merge(
            df_transactions[['transaction_id', 'category']], 
            on='transaction_id', 
            how='left'
        )
        
        sentiment_by_category = review_category_sentiment.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
        print(f"\n📊 Sentiment by Category:")
        print(sentiment_by_category)

# 5. TIME SERIES FORECASTING
print("\n=== TIME SERIES FORECASTING ===")

print("🔮 Building Revenue Forecasting Model:")

# Prepare daily revenue data
daily_revenue_ts = daily_revenue.set_index('date')['revenue']

# Split into train/test
train_size = int(len(daily_revenue_ts) * 0.8)
train_data = daily_revenue_ts[:train_size]
test_data = daily_revenue_ts[train_size:]

print(f"📊 Training on {len(train_data)} days, testing on {len(test_data)} days")

# ARIMA Model
try:
    arima_model = ARIMA(train_data, order=(1, 1, 1))
    arima_fitted = arima_model.fit()
    
    # Forecast
    arima_forecast = arima_fitted.forecast(steps=len(test_data))
    arima_mae = mean_absolute_error(test_data, arima_forecast)
    
    print(f"✅ ARIMA Model MAE: ${arima_mae:,.2f}")
    
except Exception as e:
    print(f"⚠️ ARIMA modeling failed: {e}")
    arima_forecast = None

# Exponential Smoothing
try:
    exp_smooth_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=30)
    exp_smooth_fitted = exp_smooth_model.fit()
    
    exp_smooth_forecast = exp_smooth_fitted.forecast(len(test_data))
    exp_smooth_mae = mean_absolute_error(test_data, exp_smooth_forecast)
    
    print(f"✅ Exponential Smoothing MAE: ${exp_smooth_mae:,.2f}")
    
except Exception as e:
    print(f"⚠️ Exponential Smoothing failed: {e}")
    exp_smooth_forecast = None

# Visualize forecasts
plt.figure(figsize=(15, 8))
plt.plot(train_data.index, train_data.values, label='Training Data', color='blue')
plt.plot(test_data.index, test_data.values, label='Actual', color='green', alpha=0.7)

if arima_forecast is not None:
    plt.plot(test_data.index, arima_forecast, label='ARIMA Forecast', color='red', linestyle='--')

if exp_smooth_forecast is not None:
    plt.plot(test_data.index, exp_smooth_forecast, label='Exponential Smoothing', color='orange', linestyle='--')

plt.title('Revenue Forecasting Results')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. BUSINESS INTELLIGENCE DASHBOARD
print("\n=== BUSINESS INTELLIGENCE INSIGHTS ===")

# Key business metrics
print("📊 Executive Summary Dashboard:")
print("=" * 50)

# Revenue metrics
monthly_revenue = df_transactions.groupby(df_transactions['date'].dt.to_period('M')).apply(
    lambda x: (x['price'] * x['quantity']).sum()
)
revenue_growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[0]) / monthly_revenue.iloc[0] * 100)

print(f"💰 Total Revenue: ${total_revenue:,.2f}")
print(f"📈 Revenue Growth: {revenue_growth:.1f}% over period")
print(f"🛒 Average Order Value: ${avg_order_value:.2f}")
print(f"👥 Total Customers: {unique_customers:,}")
print(f"🔁 Avg Transactions per Customer: {len(df_transactions)/unique_customers:.1f}")

# Top performing categories
category_performance = df_transactions.groupby('category').agg({
    'price': lambda x: (x * df_transactions.loc[x.index, 'quantity']).sum(),
    'transaction_id': 'count'
}).round(2)
category_performance.columns = ['revenue', 'transactions']
category_performance = category_performance.sort_values('revenue', ascending=False)

print(f"\n🏆 Top Performing Categories:")
for cat, row in category_performance.iterrows():
    print(f"   {cat}: ${row['revenue']:,.2f} ({row['transactions']} transactions)")

# Regional analysis
regional_performance = df_transactions.groupby('region').agg({
    'price': lambda x: (x * df_transactions.loc[x.index, 'quantity']).sum(),
    'customer_id': 'nunique',
    'satisfaction': 'mean'
}).round(2)
regional_performance.columns = ['revenue', 'customers', 'avg_satisfaction']

print(f"\n🌍 Regional Performance:")
for region, row in regional_performance.iterrows():
    print(f"   {region}: ${row['revenue']:,.2f} | {row['customers']} customers | {row['avg_satisfaction']:.1f}★ satisfaction")

# Customer insights
high_value_customers = customer_metrics.nlargest(5, 'total_spent')
print(f"\n💎 Top 5 Customers by Value:")
for _, customer in high_value_customers.iterrows():
    print(f"   Customer {customer['customer_id']}: ${customer['total_spent']:.2f} ({customer['transaction_count']} orders)")

# 7. MODEL DEPLOYMENT PREPARATION
print("\n=== MODEL DEPLOYMENT PREPARATION ===")

print("🚀 Preparing Models for Production:")

# Save trained models
import os
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

try:
    # Save CLV model and scaler
    joblib.dump(clv_model, f'{model_dir}/clv_model.joblib')
    joblib.dump(scaler, f'{model_dir}/clv_scaler.joblib')
    
    if len(df_reviews) >= 100:
        joblib.dump(text_pipeline, f'{model_dir}/sentiment_model.joblib')
    
    print("✅ Models saved successfully")
    
    # Create model metadata
    model_metadata = {
        'clv_model': {
            'type': 'GradientBoostingRegressor',
            'performance': {'mae': clv_mae, 'r2': clv_r2},
            'features': feature_cols,
            'training_date': datetime.now().isoformat()
        }
    }
    
    if len(df_reviews) >= 100:
        model_metadata['sentiment_model'] = {
            'type': 'MultinomialNB + TfIdf',
            'performance': {'accuracy': text_accuracy},
            'training_date': datetime.now().isoformat()
        }
    
    # Save datasets for future use
    df_transactions.to_csv('data/processed/transactions.csv', index=False)
    df_reviews.to_csv('data/processed/reviews.csv', index=False)
    daily_revenue.to_csv('data/processed/daily_revenue.csv', index=False)
    
    print("✅ Data and metadata saved")
    
except Exception as e:
    print(f"⚠️ Error saving models: {e}")

# 8. ACTIONABLE BUSINESS RECOMMENDATIONS
print("\n=== ACTIONABLE BUSINESS RECOMMENDATIONS ===")

print("💡 Strategic Recommendations:")
print("=" * 40)

# Revenue recommendations
best_category = category_performance.index[0]
worst_category = category_performance.index[-1]
print(f"1. 📈 REVENUE OPTIMIZATION")
print(f"   • Focus marketing on {best_category} (top performer: ${category_performance.loc[best_category, 'revenue']:,.2f})")
print(f"   • Investigate {worst_category} category performance (${category_performance.loc[worst_category, 'revenue']:,.2f})")

# Customer recommendations
avg_satisfaction = df_transactions['satisfaction'].mean()
print(f"\n2. 👥 CUSTOMER EXPERIENCE")
print(f"   • Current satisfaction: {avg_satisfaction:.1f}/5.0")
if avg_satisfaction < 4.0:
    print(f"   • Priority: Improve satisfaction (target: 4.0+)")
    print(f"   • Focus on quality and service improvements")

# Seasonal recommendations
peak_month = monthly_revenue.idxmax()
print(f"\n3. 📅 SEASONAL STRATEGY")
print(f"   • Peak month: {peak_month} (${monthly_revenue.max():,.2f})")
print(f"   • Prepare inventory and marketing for seasonal peaks")
print(f"   • Consider off-season promotions to balance demand")

# Regional recommendations
best_region = regional_performance['revenue'].idxmax()
underperform_region = regional_performance['revenue'].idxmin()
print(f"\n4. 🌍 REGIONAL EXPANSION")
print(f"   • Strongest market: {best_region} (${regional_performance.loc[best_region, 'revenue']:,.2f})")
print(f"   • Growth opportunity: {underperform_region} (${regional_performance.loc[underperform_region, 'revenue']:,.2f})")

print(f"\n🎯 NEXT STEPS:")
print(f"   1. Implement real-time CLV scoring for customer prioritization")
print(f"   2. Deploy sentiment monitoring for product quality alerts")
print(f"   3. Automate revenue forecasting for inventory planning")
print(f"   4. A/B test pricing strategies in different regions")
print(f"   5. Create automated alerts for satisfaction drops")

print(f"\n" + "="*60)
print(f"🏆 CONGRATULATIONS! CAPSTONE PROJECT COMPLETED!")
print(f"="*60)
print(f"You've successfully built a comprehensive data science solution featuring:")
print(f"✅ End-to-end data pipeline (generation to deployment)")
print(f"✅ Advanced analytics and business intelligence")
print(f"✅ Machine learning models (regression, classification, NLP)")
print(f"✅ Time series forecasting")
print(f"✅ Customer segmentation and lifetime value prediction")
print(f"✅ Sentiment analysis and text mining")
print(f"✅ Production-ready model deployment")
print(f"✅ Actionable business recommendations")
print(f"\n🎓 YOU ARE NOW A DATA SCIENCE MASTER! 🎓")
print(f"Ready to tackle real-world data challenges!")
```

### Success Criteria
- Build end-to-end data science pipeline from data generation to deployment
- Implement multiple ML domains: supervised learning, NLP, time series
- Create comprehensive business intelligence dashboard
- Generate actionable business insights and recommendations
- Prepare models for production deployment
- Demonstrate mastery of the full data science workflow

### Learning Objectives
- Master complete data science project lifecycle
- Integrate multiple domains (ML, NLP, time series, BI)
- Develop business acumen and strategic thinking
- Learn production deployment considerations
- Practice stakeholder communication through insights
- Gain experience with real-world complexity and constraints

### Final Achievement
Congratulations! You have completed the Data Science Sandbox journey and achieved mastery across all core domains. You're now equipped to tackle complex, real-world data science challenges and drive business value through data-driven insights.

---

*🏆 Master's Note: The best data scientists combine technical excellence with business understanding. Your ability to translate complex analyses into actionable insights is what will set you apart in the field. Keep learning, keep experimenting, and keep making an impact!*