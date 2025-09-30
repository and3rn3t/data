# Level 3: Visualization Virtuoso

## Challenge 2: Interactive Dashboards with Streamlit

Transform your visualizations into interactive web applications! Build a real-time dashboard that users can explore.

### Objective

Create interactive dashboards using Streamlit with dynamic filtering, real-time updates, and user controls.

### Instructions

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from datetime import datetime, timedelta

# Configure Streamlit page
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
.stSelectbox > div > div {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# Generate sample data for dashboard
@st.cache_data
def load_dashboard_data():
    """Generate comprehensive sales data for dashboard"""
    np.random.seed(42)

    # Date range for the last 2 years
    start_date = datetime.now() - timedelta(days=730)
    dates = [start_date + timedelta(days=i) for i in range(730)]

    n_records = len(dates) * 5  # 5 records per day on average

    data = pd.DataFrame({
        'date': np.random.choice(dates, n_records),
        'region': np.random.choice(['North America', 'Europe', 'Asia Pacific', 'Latin America'],
                                 n_records, p=[0.4, 0.3, 0.2, 0.1]),
        'product_category': np.random.choice(['Electronics', 'Software', 'Hardware', 'Services', 'Accessories'],
                                           n_records, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'sales_rep': np.random.choice([f'Rep_{i}' for i in range(1, 21)], n_records),
        'customer_type': np.random.choice(['Enterprise', 'SMB', 'Individual'], n_records, p=[0.2, 0.5, 0.3]),
        'sales_amount': np.random.gamma(2, 1000, n_records),
        'profit_margin': np.random.normal(0.25, 0.08, n_records),
        'customer_satisfaction': np.random.normal(4.3, 0.7, n_records),
        'deal_size': np.random.choice(['Small', 'Medium', 'Large', 'Enterprise'],
                                    n_records, p=[0.4, 0.35, 0.2, 0.05])
    })

    # Calculate derived metrics
    data['profit_amount'] = data['sales_amount'] * data['profit_margin']
    data['month'] = pd.to_datetime(data['date']).dt.to_period('M')
    data['quarter'] = pd.to_datetime(data['date']).dt.to_period('Q')
    data['weekday'] = pd.to_datetime(data['date']).dt.day_name()

    # Clean up satisfaction scores
    data['customer_satisfaction'] = np.clip(data['customer_satisfaction'], 1, 5)

    return data

# Load data
data = load_dashboard_data()

# Dashboard Header
st.markdown('<h1 class="main-header">📊 Sales Performance Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Controls
st.sidebar.header("🎛️ Dashboard Controls")

# Date range selector
min_date = data['date'].min()
max_date = data['date'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Multi-select filters
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=data['region'].unique(),
    default=data['region'].unique()
)

selected_categories = st.sidebar.multiselect(
    "Select Product Categories",
    options=data['product_category'].unique(),
    default=data['product_category'].unique()
)

selected_customer_types = st.sidebar.multiselect(
    "Select Customer Types",
    options=data['customer_type'].unique(),
    default=data['customer_type'].unique()
)

# Metric selector for charts
chart_metric = st.sidebar.selectbox(
    "Primary Metric for Analysis",
    options=['sales_amount', 'profit_amount', 'customer_satisfaction'],
    format_func=lambda x: {
        'sales_amount': 'Sales Amount ($)',
        'profit_amount': 'Profit Amount ($)',
        'customer_satisfaction': 'Customer Satisfaction'
    }[x]
)

# Filter data based on selections
if len(date_range) == 2:
    filtered_data = data[
        (pd.to_datetime(data['date']).dt.date >= date_range[0]) &
        (pd.to_datetime(data['date']).dt.date <= date_range[1]) &
        (data['region'].isin(selected_regions)) &
        (data['product_category'].isin(selected_categories)) &
        (data['customer_type'].isin(selected_customer_types))
    ]
else:
    filtered_data = data[
        (data['region'].isin(selected_regions)) &
        (data['product_category'].isin(selected_categories)) &
        (data['customer_type'].isin(selected_customer_types))
    ]

# Key Metrics Row
st.subheader("📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = filtered_data['sales_amount'].sum()
    st.metric(
        label="Total Sales",
        value=f"${total_sales:,.0f}",
        delta=f"{len(filtered_data)} transactions"
    )

with col2:
    total_profit = filtered_data['profit_amount'].sum()
    avg_margin = filtered_data['profit_margin'].mean()
    st.metric(
        label="Total Profit",
        value=f"${total_profit:,.0f}",
        delta=f"{avg_margin:.1%} avg margin"
    )

with col3:
    avg_satisfaction = filtered_data['customer_satisfaction'].mean()
    satisfaction_trend = "📈" if avg_satisfaction > 4.0 else "📉"
    st.metric(
        label="Avg Customer Satisfaction",
        value=f"{avg_satisfaction:.2f}/5.0",
        delta=satisfaction_trend
    )

with col4:
    avg_deal_size = filtered_data['sales_amount'].mean()
    total_customers = filtered_data['customer_type'].nunique()
    st.metric(
        label="Avg Deal Size",
        value=f"${avg_deal_size:,.0f}",
        delta=f"{total_customers} customer segments"
    )

st.markdown("---")

# Main Charts Section
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Sales Trends Over Time")

    # Time series chart
    daily_metrics = filtered_data.groupby('date').agg({
        'sales_amount': 'sum',
        'profit_amount': 'sum',
        'customer_satisfaction': 'mean'
    }).reset_index()

    fig_time = px.line(
        daily_metrics,
        x='date',
        y=chart_metric,
        title=f"Daily {chart_metric.replace('_', ' ').title()}",
        color_discrete_sequence=['#1f77b4']
    )
    fig_time.update_layout(height=400)
    st.plotly_chart(fig_time, use_container_width=True)

with col_right:
    st.subheader("🌍 Regional Performance")

    # Regional breakdown
    regional_data = filtered_data.groupby('region').agg({
        'sales_amount': 'sum',
        'profit_amount': 'sum',
        'customer_satisfaction': 'mean'
    }).reset_index()

    fig_region = px.bar(
        regional_data,
        x='region',
        y=chart_metric,
        title=f"Regional {chart_metric.replace('_', ' ').title()}",
        color=chart_metric,
        color_continuous_scale='Blues'
    )
    fig_region.update_layout(height=400)
    st.plotly_chart(fig_region, use_container_width=True)

# Product Analysis Section
st.subheader("🏷️ Product Category Analysis")

col_prod1, col_prod2 = st.columns(2)

with col_prod1:
    # Product category pie chart
    category_sales = filtered_data.groupby('product_category')['sales_amount'].sum()

    fig_pie = px.pie(
        values=category_sales.values,
        names=category_sales.index,
        title="Sales Distribution by Product Category"
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_prod2:
    # Customer type vs product category heatmap
    heatmap_data = filtered_data.pivot_table(
        values='sales_amount',
        index='product_category',
        columns='customer_type',
        aggfunc='sum',
        fill_value=0
    )

    fig_heat = px.imshow(
        heatmap_data,
        title="Sales Heatmap: Product vs Customer Type",
        color_continuous_scale='RdYlBu_r',
        aspect='auto'
    )
    fig_heat.update_layout(height=400)
    st.plotly_chart(fig_heat, use_container_width=True)

# Advanced Analytics Section
st.subheader("🔍 Advanced Analytics")

col_adv1, col_adv2 = st.columns(2)

with col_adv1:
    # Correlation analysis
    numeric_cols = ['sales_amount', 'profit_amount', 'customer_satisfaction']
    corr_matrix = filtered_data[numeric_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        title="Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with col_adv2:
    # Sales rep performance
    rep_performance = filtered_data.groupby('sales_rep').agg({
        'sales_amount': 'sum',
        'profit_amount': 'sum',
        'customer_satisfaction': 'mean'
    }).reset_index().sort_values('sales_amount', ascending=False).head(10)

    fig_reps = px.scatter(
        rep_performance,
        x='sales_amount',
        y='customer_satisfaction',
        size='profit_amount',
        hover_data=['sales_rep'],
        title="Top 10 Sales Rep Performance",
        color='profit_amount',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_reps, use_container_width=True)

# Interactive Data Table
st.subheader("📋 Detailed Data View")

# Add filters for data table
col_table1, col_table2, col_table3 = st.columns(3)

with col_table1:
    sort_by = st.selectbox(
        "Sort by",
        options=['date', 'sales_amount', 'profit_amount', 'customer_satisfaction'],
        index=1
    )

with col_table2:
    sort_order = st.selectbox("Order", options=['Descending', 'Ascending'])

with col_table3:
    show_records = st.selectbox("Show records", options=[10, 25, 50, 100], index=1)

# Display filtered and sorted data
display_data = filtered_data.sort_values(
    sort_by,
    ascending=(sort_order == 'Ascending')
).head(show_records)

st.dataframe(
    display_data.round(2),
    use_container_width=True,
    height=300
)

# Download button for filtered data
csv = filtered_data.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name=f"sales_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# Real-time updates simulation
st.subheader("🔄 Real-Time Updates")

if st.button("🔄 Refresh Data", help="Simulate new data arrival"):
    st.cache_data.clear()
    st.experimental_rerun()

# Usage Instructions
st.markdown("---")
st.markdown("""
### 🎯 Dashboard Usage Guide

**Interactive Features:**
- 📅 **Date Range**: Filter data by specific time periods
- 🌍 **Multi-Select Filters**: Choose regions, categories, and customer types
- 📊 **Dynamic Charts**: All visualizations update based on your selections
- 🔄 **Real-Time**: Refresh button simulates live data updates
- 📥 **Export**: Download filtered data for further analysis

**Chart Types Demonstrated:**
- Time series line charts with trend analysis
- Regional bar charts with color coding
- Pie charts for distribution analysis
- Heatmaps for correlation and cross-category analysis
- Scatter plots for performance comparison
- Interactive data tables with sorting

**Best Practices Applied:**
- Responsive layout that works on different screen sizes
- Consistent color schemes and styling
- Clear labeling and tooltips
- Efficient data caching for performance
- User-friendly controls and feedback
""")

print("✅ Interactive dashboard challenge completed!")
print("You've built a comprehensive Streamlit dashboard featuring:")
print("• Multi-level filtering and interactivity")
print("• Real-time data updates and refresh capability")
print("• Professional styling and responsive design")
print("• Multiple chart types and analytics views")
print("• Data export and download functionality")
```

### Success Criteria

- Build a fully interactive multi-page dashboard
- Implement dynamic filtering across multiple dimensions
- Create responsive layouts that work on different devices
- Add real-time update capabilities
- Provide data export functionality
- Apply professional styling and UX principles

### Learning Objectives

- Master Streamlit for web dashboard creation
- Understand interactive widget patterns and state management
- Learn responsive design principles for data applications
- Practice real-time data visualization techniques
- Develop professional dashboard UX/UI skills

---

_Pro tip: Great dashboards tell a story - design your layout to guide users through insights naturally!_
