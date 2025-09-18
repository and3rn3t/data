# Level 2: Analytics Apprentice - Data Cleaning Challenge

## Challenge 3: Cleaning Messy Data

Real-world data is rarely clean! This challenge teaches you essential data cleaning skills that every data scientist needs.

### Objective
Clean and prepare the sales dataset for analysis by handling missing values, outliers, and data inconsistencies.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../data/datasets/sample_sales.csv')

print("🔍 Initial Data Quality Assessment:")
print(f"Dataset shape: {df.shape}")
print(f"\nMissing values:")
print(df.isnull().sum())

# Task 1: Handle missing values in customer_age
print(f"\n📊 Customer age statistics (before cleaning):")
print(df['customer_age'].describe())

# Strategy 1: Fill with median (robust to outliers)
median_age = df['customer_age'].median()
df['customer_age_filled'] = df['customer_age'].fillna(median_age)

# Task 2: Handle missing values in customer_satisfaction
print(f"\n😊 Customer satisfaction distribution (before cleaning):")
print(df['customer_satisfaction'].value_counts().sort_index())

# Strategy 2: Fill with mode (most common value)
mode_satisfaction = df['customer_satisfaction'].mode()[0]
df['customer_satisfaction_filled'] = df['customer_satisfaction'].fillna(mode_satisfaction)

# Task 3: Detect and handle outliers in sales amounts
Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n💰 Sales outlier detection:")
print(f"Lower bound: ${lower_bound:.2f}")
print(f"Upper bound: ${upper_bound:.2f}")

outliers = df[(df['sales'] < lower_bound) | (df['sales'] > upper_bound)]
print(f"Found {len(outliers)} potential outliers")

# Visualize outliers
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot(df['sales'])
plt.title('Sales Distribution (with outliers)')
plt.ylabel('Sales Amount')

plt.subplot(1, 2, 2)
plt.hist(df['sales'], bins=30, alpha=0.7)
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower bound')
plt.axvline(upper_bound, color='red', linestyle='--', label='Upper bound')
plt.title('Sales Histogram')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# Task 4: Create cleaned dataset
df_cleaned = df.copy()
df_cleaned['customer_age'] = df_cleaned['customer_age_filled']
df_cleaned['customer_satisfaction'] = df_cleaned['customer_satisfaction_filled']

# Remove extreme outliers (optional - be careful!)
df_cleaned = df_cleaned[(df_cleaned['sales'] >= lower_bound) & (df_cleaned['sales'] <= upper_bound)]

print(f"\n✅ Data cleaning summary:")
print(f"Original records: {len(df)}")
print(f"Cleaned records: {len(df_cleaned)}")
print(f"Records removed: {len(df) - len(df_cleaned)}")
print(f"Remaining missing values: {df_cleaned.isnull().sum().sum()}")
```

### Success Criteria
- Handle all missing values appropriately
- Identify and address outliers
- Create a cleaned dataset ready for analysis
- Document your cleaning decisions

### Learning Objectives
- Master different strategies for handling missing data
- Learn outlier detection and treatment techniques
- Understand the impact of data cleaning on analysis
- Practice data quality assessment

### Reflection Questions
1. Why did you choose median over mean for filling missing ages?
2. Should you always remove outliers? What are the trade-offs?
3. How does data cleaning affect your analysis results?
4. What other data quality issues might you encounter?

### Next Steps
Excellent work on data cleaning! Next, you'll learn statistical analysis techniques to extract insights from your clean data.

---

*Pro Tip: Always document your data cleaning steps - future you will thank you! And remember: sometimes outliers contain the most interesting insights.*