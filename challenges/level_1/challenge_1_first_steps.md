# Level 1: Data Explorer Challenges

## Challenge 1: First Steps with Data

Welcome to your data science journey! In this challenge, you'll learn the basics of working with data using Python and pandas.

### Objective

Load a dataset and perform basic exploration to understand its structure and contents.

### Instructions

```python
import pandas as pd
import numpy as np

# Load the sample dataset
df = pd.read_csv('../data/datasets/sample_sales.csv')

# Convert date column to datetime for proper handling
df['date'] = pd.to_datetime(df['date'])

print("🎯 Welcome to Data Science! Let's explore our first dataset...")

# Your tasks:
# 1. Display the first 5 rows of the dataset
print("\n📊 First 5 rows:")
print(df.head())

# 2. Show basic information about the dataset
print("\n📋 Dataset info:")
print(df.info())
print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# 3. Calculate basic statistics
print("\n📈 Basic statistics:")
print(df.describe())

# 4. Check for missing values (data quality check!)
print("\n🔍 Missing values:")
missing_summary = df.isnull().sum()
total_missing = missing_summary.sum()
print(missing_summary)
print(f"\nTotal missing values: {total_missing}")

if total_missing > 0:
    print("💡 Tip: Missing values are common in real-world data!")
else:
    print("✅ Great! This dataset has no missing values.")

# 5. Show unique values in categorical columns
print("\n🏷️ Categories in our data:")
for col in df.select_dtypes(include=['object']).columns:
    unique_count = df[col].nunique()
    print(f"\n{col.title()}:")
    print(f"  • {unique_count} unique values: {list(df[col].unique())}")

# 6. Quick insights about our sales data
print(f"\n💰 Sales Insights:")
print(f"  • Total sales amount: ${df['sales'].sum():,.2f}")
print(f"  • Average sale: ${df['sales'].mean():.2f}")
print(f"  • Highest sale: ${df['sales'].max():,.2f}")
print(f"  • Date range: {df['date'].min().date()} to {df['date'].max().date()}")

print(f"\n🎉 Congratulations! You've successfully explored your first dataset!")
```

### Success Criteria

- Display dataset structure and summary statistics
- Identify and report missing values
- Show understanding of different data types

### Learning Objectives

- Understand pandas DataFrame structure
- Learn basic data exploration techniques
- Identify data quality issues

### Next Steps

Once you complete this challenge, you'll unlock visualization challenges where you'll learn to create charts and graphs from your data!

---

_Tip: The `describe()` method is your friend for getting a quick overview of numerical data!_
