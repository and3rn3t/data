"""
Generate sample datasets for Level 2 challenges
"""

import os

import numpy as np
import pandas as pd

# Use modern NumPy random generator for better practices
rng = np.random.default_rng(42)

# Ensure datasets directory exists
datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(datasets_dir, exist_ok=True)


def create_messy_sales_dataset() -> pd.DataFrame:
    """Create a messy sales dataset for data cleaning challenges"""

    n_samples = 1000

    # Generate base data
    data = {
        "customer_id": range(1, n_samples + 1),
        "customer_name": [f"Customer {i}" for i in range(1, n_samples + 1)],
        "customer_age": rng.normal(35, 12, n_samples),
        "email": [f"customer{i}@email.com" for i in range(1, n_samples + 1)],
        "phone": [f"+1-555-{str(i).zfill(4)}" for i in range(1, n_samples + 1)],
        "registration_date": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
        "country": rng.choice(["USA", "UK", "Canada", "Germany", "France"], n_samples),
        "sales": rng.gamma(2, 100, n_samples),
        "product_category": rng.choice(
            ["Electronics", "Clothing", "Books", "Home", "Sports"], n_samples
        ),
        "customer_satisfaction": rng.choice(
            [1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.2, 0.4, 0.25]
        ),
        "is_premium": rng.choice([True, False], n_samples, p=[0.3, 0.7]),
    }

    df = pd.DataFrame(data)

    # Introduce missing values
    missing_indices = rng.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[missing_indices[:50], "customer_age"] = np.nan
    df.loc[missing_indices[50:80], "customer_satisfaction"] = np.nan
    df.loc[missing_indices[80:], "phone"] = np.nan

    # Introduce outliers
    outlier_indices = rng.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices[:10], "sales"] = (
        df["sales"].max() * 10
    )  # Extreme high values
    df.loc[outlier_indices[10:15], "customer_age"] = -5  # Invalid ages
    df.loc[outlier_indices[15:], "sales"] = -100  # Negative sales

    # Introduce data inconsistencies
    df.loc[df.index[:10], "email"] = "invalid_email_format"
    df.loc[df.index[10:20], "country"] = "UNKNOWN"

    return df


def create_ecommerce_analytics_dataset() -> pd.DataFrame:
    """Create e-commerce dataset for statistical analysis"""

    n_samples = 2000

    # Generate correlated data
    age = rng.normal(35, 15, n_samples)
    age = np.clip(age, 18, 80)  # Realistic age range

    # Time on site correlated with age (younger users spend more time)
    time_on_site = 10 + (45 - age) * 0.3 + rng.normal(0, 5, n_samples)
    time_on_site = np.clip(time_on_site, 1, 120)

    # Pages visited correlated with time on site
    pages_visited = 1 + time_on_site * 0.2 + rng.poisson(2, n_samples)
    pages_visited = np.clip(pages_visited, 1, 50)

    # Purchase amount correlated with time and pages
    purchase_amount = time_on_site * 2 + pages_visited * 5 + rng.gamma(2, 20, n_samples)

    # Customer satisfaction correlated with purchase amount
    satisfaction_base = 5 + (purchase_amount / 100) + rng.normal(0, 1.5, n_samples)
    customer_satisfaction = np.clip(satisfaction_base, 1, 10)

    # Premium status (higher income users more likely to be premium)
    premium_prob = np.clip((purchase_amount / 200), 0.1, 0.8)
    is_premium: np.ndarray = rng.binomial(1, premium_prob, n_samples).astype(bool)

    data = {
        "user_id": range(1, n_samples + 1),
        "customer_age": age.round(0).astype(int),
        "time_on_site": time_on_site.round(1),
        "pages_visited": pages_visited.astype(int),
        "purchase_amount": purchase_amount.round(2),
        "customer_satisfaction": customer_satisfaction.round(1),
        "is_premium": is_premium,
        "session_date": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
        "device_type": rng.choice(
            ["Desktop", "Mobile", "Tablet"], n_samples, p=[0.5, 0.4, 0.1]
        ),
        "traffic_source": rng.choice(
            ["Organic", "Paid", "Social", "Direct"], n_samples, p=[0.3, 0.25, 0.25, 0.2]
        ),
        "country": rng.choice(
            ["USA", "UK", "Canada", "Germany", "France", "Australia"], n_samples
        ),
        "browser": rng.choice(
            ["Chrome", "Firefox", "Safari", "Edge"], n_samples, p=[0.6, 0.2, 0.15, 0.05]
        ),
    }

    return pd.DataFrame(data)


def create_customer_survey_dataset() -> pd.DataFrame:
    """Create customer survey dataset for advanced analysis"""
    rng = np.random.default_rng(42)

    n_samples = 1500

    # Different customer segments
    segments = ["Budget", "Premium", "Enterprise"]
    segment_probs = [0.5, 0.3, 0.2]
    customer_segment = rng.choice(segments, n_samples, p=segment_probs)

    # Age varies by segment
    age_mapping = {"Budget": (22, 8), "Premium": (35, 10), "Enterprise": (45, 12)}
    ages = []
    for segment in customer_segment:
        mean_age, std_age = age_mapping[segment]
        age = rng.normal(mean_age, std_age)
        ages.append(max(18, min(65, age)))

    # Income correlated with segment and age
    income_base = {"Budget": 35000, "Premium": 75000, "Enterprise": 120000}
    incomes = []
    for i, segment in enumerate(customer_segment):
        base = income_base[segment]
        age_factor = (ages[i] - 25) * 1000  # Older = higher income
        income = base + age_factor + rng.normal(0, base * 0.2)
        incomes.append(max(20000, income))

    # Satisfaction varies by segment
    satisfaction_mapping = {"Budget": 6.5, "Premium": 8.0, "Enterprise": 7.5}
    satisfaction_scores = []
    for segment in customer_segment:
        base_satisfaction = satisfaction_mapping[segment]
        score = rng.normal(base_satisfaction, 1.5)
        satisfaction_scores.append(max(1, min(10, score)))

    data = {
        "customer_id": [f"CUST_{str(i).zfill(6)}" for i in range(1, n_samples + 1)],
        "age": np.array(ages).round(0).astype(int),
        "annual_income": np.array(incomes).round(0).astype(int),
        "customer_segment": customer_segment,
        "satisfaction_score": np.array(satisfaction_scores).round(1),
        "years_as_customer": rng.exponential(2, n_samples).round(1),
        "monthly_spend": rng.gamma(2, 50, n_samples).round(2),
        "support_tickets_last_year": rng.poisson(2, n_samples),
        "product_usage_hours_weekly": rng.gamma(1.5, 8, n_samples).round(1),
        "referral_count": rng.poisson(1, n_samples),
        "churn_risk_score": rng.beta(2, 5, n_samples).round(3),
        "last_purchase_days_ago": rng.exponential(30, n_samples).round(0).astype(int),
        "preferred_contact": rng.choice(
            ["Email", "Phone", "Chat", "None"], n_samples, p=[0.4, 0.3, 0.2, 0.1]
        ),
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("🔧 Generating Level 2 datasets...")

    # Create datasets
    print("📊 Creating messy sales dataset...")
    messy_sales = create_messy_sales_dataset()
    messy_sales.to_csv(os.path.join(datasets_dir, "messy_sales_data.csv"), index=False)
    print(
        f"   ✅ Saved: {len(messy_sales)} records with intentional data quality issues"
    )

    print("📈 Creating e-commerce analytics dataset...")
    ecommerce_data = create_ecommerce_analytics_dataset()
    ecommerce_data.to_csv(
        os.path.join(datasets_dir, "ecommerce_analytics.csv"), index=False
    )
    print(f"   ✅ Saved: {len(ecommerce_data)} records for statistical analysis")

    print("📋 Creating customer survey dataset...")
    survey_data = create_customer_survey_dataset()
    survey_data.to_csv(
        os.path.join(datasets_dir, "customer_survey_data.csv"), index=False
    )
    print(f"   ✅ Saved: {len(survey_data)} records for advanced analytics")

    print("\n🎯 Level 2 datasets ready for challenges!")
    print("\nDatasets created:")
    print("• messy_sales_data.csv - For data cleaning challenges")
    print("• ecommerce_analytics.csv - For statistical analysis")
    print("• customer_survey_data.csv - For advanced analytics")
