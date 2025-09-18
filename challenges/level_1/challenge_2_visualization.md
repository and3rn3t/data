# Level 1: Basic Data Visualization Challenge

## Challenge 2: Creating Your First Charts

Now that you can explore data, let's learn to visualize it! Data visualization is crucial for understanding patterns and communicating insights.

### Objective
Create basic charts to visualize the sales dataset and discover patterns.

### Instructions

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../data/datasets/sample_sales.csv')

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Your tasks:

# 1. Create a histogram of sales amounts
plt.figure(figsize=(10, 6))
plt.hist(df['sales'], bins=20, alpha=0.7, edgecolor='black')
plt.title('Distribution of Sales Amounts')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# 2. Create a bar chart of sales by region
plt.figure(figsize=(10, 6))
sales_by_region = df.groupby('region')['sales'].sum()
plt.bar(sales_by_region.index, sales_by_region.values)
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Create a scatter plot of quantity vs sales
plt.figure(figsize=(10, 6))
plt.scatter(df['quantity'], df['sales'], alpha=0.6)
plt.title('Quantity vs Sales Amount')
plt.xlabel('Quantity Sold')
plt.ylabel('Sales Amount')
plt.grid(True, alpha=0.3)
plt.show()

# 4. Create a box plot to compare sales across different categories
plt.figure(figsize=(12, 6))
df.boxplot(column='sales', by='category', ax=plt.gca())
plt.title('Sales Distribution by Product Category')
plt.suptitle('')  # Remove default title
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. BONUS: Create a correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numeric Variables')
plt.tight_layout()
plt.show()
```

### Success Criteria
- Create at least 4 different types of charts
- Add appropriate titles and labels to all charts
- Interpret what each visualization reveals about the data

### Learning Objectives
- Master basic matplotlib and seaborn functionality
- Choose appropriate chart types for different data relationships
- Make visualizations clear and informative

### Reflection Questions
1. Which region has the highest total sales?
2. Is there a relationship between quantity and sales amount?
3. Which product category shows the most variation in sales?
4. What correlations do you notice between numeric variables?

### Next Steps
Great job on visualization! Next, you'll learn data cleaning techniques to handle missing values and outliers.

---

*Pro Tip: Always add titles and labels to your charts - they make your visualizations much more professional and easier to understand!*