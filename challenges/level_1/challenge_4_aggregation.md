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

# Convert date to datetime for time-based analysis
df['date'] = pd.to_datetime(df['date'])

print("📊 Learning Data Aggregation - Let's find patterns in our data!")

# Your tasks:
# 1. Group by categorical columns and calculate basic statistics
print("\n🏷️ Sales by Product Category:")
category_stats = df.groupby('category')['sales'].agg(['count', 'sum', 'mean', 'std'])
category_stats.columns = ['Number of Sales', 'Total Sales', 'Average Sale', 'Standard Deviation']
print(category_stats.round(2))

# 2. Multiple grouping - sales by category AND region
print("\n🌍 Sales by Category and Region:")
multi_group = df.groupby(['category', 'region'])['sales'].sum().unstack(fill_value=0)
print(multi_group)

# Find the best performing combinations
print(f"\n🏆 Top performing category-region combination:")
best_combo = multi_group.stack().idxmax()
best_value = multi_group.stack().max()
print(f"   {best_combo[0]} in {best_combo[1]}: ${best_value:,.2f}")

# 3. Time-based grouping - monthly sales trends
print("\n📅 Monthly Sales Trends:")
df['month'] = df['date'].dt.to_period('M')
monthly_sales = df.groupby('month')['sales'].sum()
print(monthly_sales)

# 3. Custom aggregation functions
def coefficient_of_variation(x):
    return x.std() / x.mean() if x.mean() != 0 else 0

if 'category' in df.columns:
    custom_agg = df.groupby('category')['sales'].agg([
        'mean',
        'std',
        ('cv', coefficient_of_variation)
    ])
    print("\nCustom aggregation with coefficient of variation:")
    print(custom_agg)

# 4. Pivot tables
if 'category' in df.columns and 'month' in df.columns:
    pivot = pd.pivot_table(df,
                          values='sales',
                          index='category',
                          columns='month',
                          aggfunc='sum',
                          fill_value=0)
    print("\nPivot table - Sales by category and month:")
    print(pivot)

# 5. Time-based grouping (if date column exists)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    monthly_sales = df.set_index('date').resample('M')['sales'].sum()
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

_Tip: Use `pd.pivot_table()` for complex cross-tabulations!_
