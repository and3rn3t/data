# Level 3: Visualization Virtuoso

## Challenge 1: Master Data Visualization

Welcome to the art of data storytelling! Create compelling visualizations that reveal insights.

### Objective
Master advanced data visualization techniques using matplotlib, seaborn, and plotly.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create comprehensive dataset
np.random.seed(42)
n_samples = 1000

# Simulate company sales data
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples, p=[0.3, 0.25, 0.25, 0.2]),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_samples),
    'sales_amount': np.random.gamma(2, 500, n_samples),
    'profit_margin': np.random.normal(0.15, 0.05, n_samples),
    'customer_satisfaction': np.random.normal(4.2, 0.8, n_samples),
    'marketing_spend': np.random.exponential(100, n_samples),
    'season': ['Winter' if m in [12, 1, 2] else 
              'Spring' if m in [3, 4, 5] else
              'Summer' if m in [6, 7, 8] else 'Fall'
              for m in pd.to_datetime(data['date'] if 'date' in locals() else pd.date_range('2023-01-01', periods=n_samples, freq='D')).month]
})

# Add some realistic relationships
data['profit_amount'] = data['sales_amount'] * data['profit_margin']
data['month'] = pd.to_datetime(data['date']).dt.month
data['quarter'] = pd.to_datetime(data['date']).dt.quarter

print("Dataset ready for visualization!")
print(data.head())

# Your tasks:
# 1. STATISTICAL VISUALIZATIONS
print("\n=== STATISTICAL VISUALIZATIONS ===")

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution Analysis', fontsize=16, fontweight='bold')

# Histogram with KDE
axes[0,0].hist(data['sales_amount'], bins=50, alpha=0.7, density=True, color='skyblue')
axes[0,0].set_title('Sales Amount Distribution')
axes[0,0].set_xlabel('Sales Amount ($)')
axes[0,0].set_ylabel('Density')

# Box plot by category
data.boxplot(column='sales_amount', by='product_category', ax=axes[0,1])
axes[0,1].set_title('Sales by Product Category')
axes[0,1].set_xlabel('Product Category')

# Violin plot for profit margin
sns.violinplot(data=data, x='region', y='profit_margin', ax=axes[1,0])
axes[1,0].set_title('Profit Margin Distribution by Region')
axes[1,0].tick_params(axis='x', rotation=45)

# Correlation heatmap
numeric_cols = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
axes[1,1].set_title('Correlation Matrix')

plt.tight_layout()
plt.show()

# 2. TIME SERIES VISUALIZATIONS
print("\n=== TIME SERIES VISUALIZATIONS ===")

# Advanced time series plot
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')

# Daily sales trend with moving average
daily_sales = data.groupby('date')['sales_amount'].sum().reset_index()
daily_sales['moving_avg_7'] = daily_sales['sales_amount'].rolling(window=7).mean()
daily_sales['moving_avg_30'] = daily_sales['sales_amount'].rolling(window=30).mean()

axes[0].plot(daily_sales['date'], daily_sales['sales_amount'], alpha=0.3, label='Daily Sales', color='gray')
axes[0].plot(daily_sales['date'], daily_sales['moving_avg_7'], label='7-day MA', color='blue')
axes[0].plot(daily_sales['date'], daily_sales['moving_avg_30'], label='30-day MA', color='red')
axes[0].set_title('Daily Sales with Moving Averages')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Monthly sales by region
monthly_regional = data.groupby([pd.Grouper(key='date', freq='M'), 'region'])['sales_amount'].sum().reset_index()
for region in data['region'].unique():
    region_data = monthly_regional[monthly_regional['region'] == region]
    axes[1].plot(region_data['date'], region_data['sales_amount'], marker='o', label=region)
axes[1].set_title('Monthly Sales by Region')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

# Seasonal decomposition simulation
monthly_sales = data.groupby(pd.Grouper(key='date', freq='M'))['sales_amount'].sum()
axes[2].plot(monthly_sales.index, monthly_sales.values, marker='o', color='green')
axes[2].set_title('Monthly Sales Trend')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 3. ADVANCED STATISTICAL PLOTS
print("\n=== ADVANCED STATISTICAL PLOTS ===")

# Pair plot for key variables
key_vars = ['sales_amount', 'profit_margin', 'customer_satisfaction', 'marketing_spend']
plt.figure(figsize=(12, 10))
sns.pairplot(data[key_vars + ['region']], hue='region', diag_kind='kde')
plt.suptitle('Pair Plot Analysis', fontsize=16, y=1.02)
plt.show()

# 4. INTERACTIVE PLOTLY VISUALIZATIONS
print("\n=== INTERACTIVE VISUALIZATIONS ===")

# Interactive scatter plot
fig_scatter = px.scatter(data, 
                        x='marketing_spend', 
                        y='sales_amount',
                        color='region',
                        size='profit_amount',
                        hover_data=['customer_satisfaction', 'product_category'],
                        title='Sales vs Marketing Spend by Region')
fig_scatter.show()

# Interactive time series with range selector
monthly_data = data.groupby([pd.Grouper(key='date', freq='M'), 'region']).agg({
    'sales_amount': 'sum',
    'profit_amount': 'sum'
}).reset_index()

fig_time = go.Figure()

for region in data['region'].unique():
    region_data = monthly_data[monthly_data['region'] == region]
    fig_time.add_trace(go.Scatter(
        x=region_data['date'],
        y=region_data['sales_amount'],
        mode='lines+markers',
        name=region
    ))

fig_time.update_layout(
    title='Interactive Monthly Sales by Region',
    xaxis_title='Date',
    yaxis_title='Sales Amount ($)',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=3, label='3M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible=True),
        type='date'
    )
)
fig_time.show()

# 5. DASHBOARD-STYLE SUBPLOTS
print("\n=== DASHBOARD VISUALIZATION ===")

# Create subplot dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sales by Category', 'Regional Performance', 
                   'Profit vs Satisfaction', 'Seasonal Trends'),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "bar"}]]
)

# Sales by category
category_sales = data.groupby('product_category')['sales_amount'].sum().sort_values(ascending=False)
fig.add_trace(
    go.Bar(x=category_sales.index, y=category_sales.values, name='Sales by Category'),
    row=1, col=1
)

# Regional performance
regional_profit = data.groupby('region')['profit_amount'].sum()
fig.add_trace(
    go.Bar(x=regional_profit.index, y=regional_profit.values, name='Regional Profit'),
    row=1, col=2
)

# Profit vs satisfaction scatter
fig.add_trace(
    go.Scatter(x=data['customer_satisfaction'], y=data['profit_amount'],
              mode='markers', name='Profit vs Satisfaction', opacity=0.6),
    row=2, col=1
)

# Seasonal trends
seasonal_sales = data.groupby('season')['sales_amount'].sum()
fig.add_trace(
    go.Bar(x=seasonal_sales.index, y=seasonal_sales.values, name='Seasonal Sales'),
    row=2, col=2
)

fig.update_layout(
    height=800,
    title_text="Sales Performance Dashboard",
    showlegend=False
)
fig.show()

# 6. STATISTICAL ANNOTATION AND INSIGHTS
print("\n=== INSIGHTS SUMMARY ===")

insights = []

# Top performing category
top_category = data.groupby('product_category')['sales_amount'].sum().idxmax()
top_category_sales = data.groupby('product_category')['sales_amount'].sum().max()
insights.append(f"📊 Top performing category: {top_category} (${top_category_sales:,.0f})")

# Best region
best_region = data.groupby('region')['profit_amount'].sum().idxmax()
best_region_profit = data.groupby('region')['profit_amount'].sum().max()
insights.append(f"🌍 Most profitable region: {best_region} (${best_region_profit:,.0f})")

# Correlation insight
corr_marketing_sales = data['marketing_spend'].corr(data['sales_amount'])
insights.append(f"📈 Marketing-Sales correlation: {corr_marketing_sales:.3f}")

# Seasonal insight
best_season = data.groupby('season')['sales_amount'].sum().idxmax()
insights.append(f"🗓️ Best performing season: {best_season}")

for insight in insights:
    print(insight)

print("\n✅ Visualization challenge completed!")
print("You've created comprehensive visualizations covering:")
print("• Statistical distributions and relationships")
print("• Time series analysis with trends")
print("• Interactive plots with plotly")
print("• Dashboard-style multi-panel displays")
print("• Data-driven insights extraction")
```

### Success Criteria
- Create diverse visualization types (statistical, time series, interactive)
- Use appropriate chart types for different data relationships
- Apply good design principles (colors, labels, titles)
- Build interactive dashboards
- Extract and communicate insights from visualizations

### Learning Objectives
- Master matplotlib and seaborn for static plots
- Learn plotly for interactive visualizations
- Understand when to use different chart types
- Practice visual design principles
- Develop data storytelling skills

---

*Tip: The best visualization is the one that makes your data story crystal clear to the audience!*