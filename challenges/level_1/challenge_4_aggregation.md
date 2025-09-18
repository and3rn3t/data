# Level 1: Data Explorer - Advanced Challenge

## Challenge 4: Data Aggregation and Grouping

Time to learn one of the most powerful features of data analysis - grouping and aggregation!

### Objective
Master the art of grouping data and calculating aggregate statistics to discover patterns.

### Instructions

```python
import pandas as pd
import numpy as np

# Load the sales dataset
df = pd.read_csv('../data/datasets/sample_sales.csv')

# Your tasks:
# 1. Group by categorical columns and calculate basic statistics
if 'category' in df.columns and 'amount' in df.columns:
    category_stats = df.groupby('category')['amount'].agg(['count', 'sum', 'mean', 'std'])
    print("Category statistics:")
    print(category_stats)

# 2. Multiple grouping
if 'category' in df.columns and 'region' in df.columns:
    multi_group = df.groupby(['category', 'region'])['amount'].sum()
    print("\nSales by category and region:")
    print(multi_group)

# 3. Custom aggregation functions
def coefficient_of_variation(x):
    return x.std() / x.mean() if x.mean() != 0 else 0

if 'category' in df.columns:
    custom_agg = df.groupby('category')['amount'].agg([
        'mean', 
        'std',
        ('cv', coefficient_of_variation)
    ])
    print("\nCustom aggregation with coefficient of variation:")
    print(custom_agg)

# 4. Pivot tables
if 'category' in df.columns and 'month' in df.columns:
    pivot = pd.pivot_table(df, 
                          values='amount', 
                          index='category', 
                          columns='month', 
                          aggfunc='sum', 
                          fill_value=0)
    print("\nPivot table - Sales by category and month:")
    print(pivot)

# 5. Time-based grouping (if date column exists)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    monthly_sales = df.set_index('date').resample('M')['amount'].sum()
    print("\nMonthly sales trend:")
    print(monthly_sales)
```

### Success Criteria
- Group data by single and multiple columns
- Calculate various aggregate statistics
- Create custom aggregation functions
- Build pivot tables
- Perform time-based aggregations

### Learning Objectives
- Master pandas groupby operations
- Understand pivot tables
- Learn custom aggregation functions
- Practice time series resampling

---

*Tip: Use `pd.pivot_table()` for complex cross-tabulations!*