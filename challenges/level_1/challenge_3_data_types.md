# Level 1: Data Explorer - Additional Challenges

## Challenge 3: Data Types and Structures

Welcome to your third challenge! Now you'll explore different data types and learn to work with various data structures.

### Objective

Understand different data types and practice working with lists, dictionaries, and pandas data structures.

### Instructions

```python
import pandas as pd
import numpy as np

# Load multiple datasets
sales_df = pd.read_csv('../data/datasets/sample_sales.csv')
iris_df = pd.read_csv('../data/datasets/iris.csv')

# Your tasks:
# 1. Examine data types in both datasets
print("Sales dataset data types:")
print(sales_df.dtypes)
print("\nIris dataset data types:")
print(iris_df.dtypes)

# 2. Convert data types where appropriate
# Convert date columns to datetime (if exists)
if 'date' in sales_df.columns:
    sales_df['date'] = pd.to_datetime(sales_df['date'])

# 3. Practice with Python data structures
products = ['laptops', 'phones', 'tablets', 'accessories']
prices = [1200, 800, 400, 50]

# Create dictionary from lists
product_prices = dict(zip(products, prices))
print(f"\nProduct prices: {product_prices}")

# 4. Work with pandas Series
price_series = pd.Series(prices, index=products)
print(f"\nPrice series:\n{price_series}")

# 5. Filter data based on conditions
# Use the correct column name 'sales' instead of 'price'
expensive_items = sales_df[sales_df['sales'] > sales_df['sales'].mean()]
print(f"\nExpensive items (above average): {len(expensive_items)} out of {len(sales_df)}")

# 6. Work with dates (datetime objects)
sales_df['date'] = pd.to_datetime(sales_df['date'])
print(f"\nDate range: {sales_df['date'].min().date()} to {sales_df['date'].max().date()}")

# Extract components from dates
sales_df['year'] = sales_df['date'].dt.year
sales_df['month'] = sales_df['date'].dt.month
sales_df['day_name'] = sales_df['date'].dt.day_name()

print(f"\nSales by day of week:")
day_sales = sales_df.groupby('day_name')['sales'].mean()
print(day_sales.round(2))
```

### Success Criteria

- Successfully examine and understand data types
- Convert data types appropriately
- Create and manipulate Python data structures
- Filter data using conditions

### Learning Objectives

- Master pandas data types
- Understand Python data structures
- Learn data type conversions
- Practice conditional filtering

---

_Tip: Use `df.astype()` to convert data types explicitly!_
