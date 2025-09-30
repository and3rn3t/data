# Level 2: Analytics Apprentice Challenges

## Challenge 1: Advanced Data Cleaning

Welcome to Level 2! Now we'll tackle messier, more realistic data cleaning scenarios.

### Objective

Master advanced data cleaning techniques including handling outliers, data validation, and complex missing value strategies.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create a messy dataset for practice
np.random.seed(42)
n_samples = 1000

messy_data = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'name': ['Customer ' + str(i) if i % 10 != 0 else np.nan for i in range(1, n_samples + 1)],
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.lognormal(10, 1, n_samples),
    'email': ['user' + str(i) + '@email.com' if i % 15 != 0 else 'invalid_email' for i in range(1, n_samples + 1)],
    'phone': [f'+1-555-{str(i).zfill(4)}' if i % 20 != 0 else np.nan for i in range(1, n_samples + 1)],
    'purchase_amount': np.random.gamma(2, 50, n_samples)
})

# Add some extreme outliers
messy_data.loc[messy_data.index[:10], 'income'] = messy_data['income'].max() * 10
messy_data.loc[messy_data.index[10:20], 'age'] = -5  # Invalid ages
messy_data.loc[messy_data.index[20:25], 'purchase_amount'] = -100  # Negative purchases

print("Original dataset info:")
print(messy_data.info())
print("\nFirst few rows:")
print(messy_data.head())

# Your tasks:
# 1. Identify and handle missing values with different strategies
print("\n=== MISSING VALUES ANALYSIS ===")
missing_summary = messy_data.isnull().sum()
missing_percentage = (missing_summary / len(messy_data)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_summary,
    'Missing_Percentage': missing_percentage
})
print(missing_df[missing_df['Missing_Count'] > 0])

# Handle missing names by forward filling
messy_data['name'] = messy_data['name'].fillna(method='ffill')

# Handle missing phones with a placeholder
messy_data['phone'] = messy_data['phone'].fillna('Not Provided')

# 2. Data validation and cleaning
print("\n=== DATA VALIDATION ===")

# Fix invalid ages
messy_data['age'] = messy_data['age'].clip(lower=0, upper=120)
print(f"Age range after cleaning: {messy_data['age'].min():.1f} - {messy_data['age'].max():.1f}")

# Fix negative purchase amounts
messy_data['purchase_amount'] = messy_data['purchase_amount'].clip(lower=0)
print(f"Purchase amount range: ${messy_data['purchase_amount'].min():.2f} - ${messy_data['purchase_amount'].max():.2f}")

# Validate email format
def is_valid_email(email):
    return '@' in email and '.' in email.split('@')[1]

messy_data['email_valid'] = messy_data['email'].apply(is_valid_email)
print(f"Valid emails: {messy_data['email_valid'].sum()} out of {len(messy_data)}")

# 3. Outlier detection and handling
print("\n=== OUTLIER DETECTION ===")

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

income_outliers = detect_outliers_iqr(messy_data, 'income')
print(f"Income outliers detected: {len(income_outliers)}")

# Z-score method for outlier detection
def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores > threshold]

purchase_outliers = detect_outliers_zscore(messy_data, 'purchase_amount')
print(f"Purchase amount outliers (z-score > 3): {len(purchase_outliers)}")

# 4. Advanced cleaning techniques
print("\n=== ADVANCED CLEANING ===")

# Cap extreme income values at 99th percentile
income_cap = messy_data['income'].quantile(0.99)
messy_data['income_capped'] = messy_data['income'].clip(upper=income_cap)
print(f"Income capped at: ${income_cap:.2f}")

# Create cleaned dataset
cleaned_data = messy_data[
    (messy_data['email_valid'] == True) &
    (messy_data['age'] >= 18) &
    (messy_data['age'] <= 80) &
    (messy_data['purchase_amount'] > 0)
].copy()

print(f"\nDataset size after cleaning: {len(cleaned_data)} rows (from {len(messy_data)})")
print(f"Data quality improved: {len(cleaned_data)/len(messy_data)*100:.1f}% retained")

# 5. Data quality report
print("\n=== DATA QUALITY REPORT ===")
quality_metrics = {
    'Total_Records': len(messy_data),
    'Clean_Records': len(cleaned_data),
    'Data_Retention_Rate': f"{len(cleaned_data)/len(messy_data)*100:.1f}%",
    'Missing_Values_Fixed': missing_summary.sum(),
    'Outliers_Handled': len(income_outliers) + len(purchase_outliers),
    'Invalid_Emails_Found': (~messy_data['email_valid']).sum()
}

for metric, value in quality_metrics.items():
    print(f"{metric}: {value}")
```

### Success Criteria

- Implement multiple missing value handling strategies
- Validate and clean data using business rules
- Detect and handle outliers using statistical methods
- Create a comprehensive data quality report
- Retain maximum valid data while ensuring quality

### Learning Objectives

- Master advanced pandas cleaning functions
- Understand different outlier detection methods
- Learn data validation techniques
- Practice quality assessment and reporting

---

*Tip: Always understand your data domain before cleaning - what constitutes an outlier depends on business context!*
