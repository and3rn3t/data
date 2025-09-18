# Level 2: Analytics Apprentice 

## Challenge 2: Statistical Analysis Deep Dive

Time to become a statistics guru! Explore distributions, hypothesis testing, and correlation analysis.

### Objective
Master essential statistical analysis techniques for data science.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load or create dataset
np.random.seed(42)
n_samples = 500

# Simulate e-commerce data
data = pd.DataFrame({
    'customer_age': np.random.normal(35, 12, n_samples),
    'time_on_site': np.random.exponential(5, n_samples),
    'pages_visited': np.random.poisson(3, n_samples),
    'purchase_amount': np.random.gamma(2, 30, n_samples),
    'customer_satisfaction': np.random.normal(7, 1.5, n_samples),
    'is_premium': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})

# Add some correlations
data['purchase_amount'] += data['time_on_site'] * 5  # More time = more spending
data['customer_satisfaction'] += data['purchase_amount'] * 0.01  # More spending = higher satisfaction

print("Dataset overview:")
print(data.describe())

# Your tasks:
# 1. Descriptive Statistics Analysis
print("\n=== DESCRIPTIVE STATISTICS ===")

# Calculate comprehensive statistics
def comprehensive_stats(series):
    stats_dict = {
        'count': series.count(),
        'mean': series.mean(),
        'median': series.median(),
        'mode': series.mode().iloc[0] if len(series.mode()) > 0 else np.nan,
        'std': series.std(),
        'var': series.var(),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
        'min': series.min(),
        'max': series.max(),
        'range': series.max() - series.min(),
        'q1': series.quantile(0.25),
        'q3': series.quantile(0.75),
        'iqr': series.quantile(0.75) - series.quantile(0.25)
    }
    return pd.Series(stats_dict)

for col in data.select_dtypes(include=[np.number]).columns:
    if col != 'is_premium':  # Skip binary variable
        print(f"\n{col.upper()} Statistics:")
        stats_summary = comprehensive_stats(data[col])
        print(stats_summary.round(2))

# 2. Distribution Analysis
print("\n=== DISTRIBUTION ANALYSIS ===")

# Test for normality
def test_normality(data, column):
    stat, p_value = stats.shapiro(data[column].sample(min(5000, len(data))))  # Shapiro-Wilk test
    print(f"{column} - Normality test p-value: {p_value:.6f}")
    if p_value > 0.05:
        print(f"  → {column} appears to be normally distributed")
    else:
        print(f"  → {column} is not normally distributed")
    return p_value > 0.05

for col in ['customer_age', 'purchase_amount', 'customer_satisfaction']:
    test_normality(data, col)

# 3. Correlation Analysis
print("\n=== CORRELATION ANALYSIS ===")

# Calculate different types of correlations
corr_pearson = data.corr(method='pearson')
corr_spearman = data.corr(method='spearman')

print("Pearson Correlation Matrix:")
print(corr_pearson.round(3))

print("\nSpearman Correlation Matrix:")
print(corr_spearman.round(3))

# Find strongest correlations
def find_strong_correlations(corr_matrix, threshold=0.5):
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                strong_corr.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    return pd.DataFrame(strong_corr)

strong_corr = find_strong_correlations(corr_pearson)
if len(strong_corr) > 0:
    print(f"\nStrong correlations (|r| > 0.5):")
    print(strong_corr)

# 4. Hypothesis Testing
print("\n=== HYPOTHESIS TESTING ===")

# T-test: Compare purchase amounts between premium and regular customers
premium_customers = data[data['is_premium'] == 1]['purchase_amount']
regular_customers = data[data['is_premium'] == 0]['purchase_amount']

t_stat, p_value = stats.ttest_ind(premium_customers, regular_customers)
print(f"T-test: Premium vs Regular Customers Purchase Amount")
print(f"  Premium customers mean: ${premium_customers.mean():.2f}")
print(f"  Regular customers mean: ${regular_customers.mean():.2f}")
print(f"  T-statistic: {t_stat:.3f}")
print(f"  P-value: {p_value:.6f}")

if p_value < 0.05:
    print("  → Statistically significant difference (p < 0.05)")
else:
    print("  → No statistically significant difference (p >= 0.05)")

# Chi-square test for independence
print(f"\nChi-square test: Premium status vs High satisfaction")
high_satisfaction = data['customer_satisfaction'] > data['customer_satisfaction'].median()
contingency_table = pd.crosstab(data['is_premium'], high_satisfaction)
chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

print("Contingency table:")
print(contingency_table)
print(f"Chi-square statistic: {chi2:.3f}")
print(f"P-value: {p_val:.6f}")

# 5. Advanced Statistical Analysis
print("\n=== ADVANCED ANALYSIS ===")

# ANOVA test: Compare customer satisfaction across age groups
data['age_group'] = pd.cut(data['customer_age'], 
                          bins=[0, 25, 35, 50, 100], 
                          labels=['18-25', '26-35', '36-50', '50+'])

age_groups = [group['customer_satisfaction'].values for name, group in data.groupby('age_group')]
f_stat, p_val = stats.f_oneway(*age_groups)

print(f"ANOVA: Customer satisfaction across age groups")
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {p_val:.6f}")

# Confidence intervals
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin_error = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin_error, mean + margin_error

ci_purchase = confidence_interval(data['purchase_amount'])
print(f"\n95% Confidence Interval for Purchase Amount:")
print(f"  ${ci_purchase[0]:.2f} - ${ci_purchase[1]:.2f}")

# Effect size calculation (Cohen's d)
def cohens_d(group1, group2):
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                         (len(group2) - 1) * np.var(group2, ddof=1)) / 
                        (len(group1) + len(group2) - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

effect_size = cohens_d(premium_customers, regular_customers)
print(f"\nCohen's d (effect size): {effect_size:.3f}")
if abs(effect_size) < 0.2:
    effect_interpretation = "small"
elif abs(effect_size) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"Effect size interpretation: {effect_interpretation}")
```

### Success Criteria
- Calculate comprehensive descriptive statistics
- Test data distributions for normality
- Perform correlation analysis using multiple methods
- Conduct appropriate hypothesis tests
- Calculate confidence intervals and effect sizes

### Learning Objectives
- Master descriptive statistics
- Understand probability distributions
- Learn hypothesis testing framework
- Practice correlation vs causation concepts
- Calculate statistical significance and effect sizes

---

*Tip: Always check your assumptions before applying statistical tests!*