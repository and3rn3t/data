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

# Your tasks:
# 1. Display the first 5 rows of the dataset
print("First 5 rows:")
print(df.head())

# 2. Show basic information about the dataset
print("\nDataset info:")
print(df.info())

# 3. Calculate basic statistics
print("\nBasic statistics:")
print(df.describe())

# 4. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Show unique values in categorical columns
for col in df.select_dtypes(include=['object']).columns:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())
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

*Tip: The `describe()` method is your friend for getting a quick overview of numerical data!*