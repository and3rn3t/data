# Level 6: Advanced Analytics Expert

## Challenge 4: Advanced Statistical Analysis and Experimental Design

Master sophisticated statistical methods, experimental design principles, causal inference techniques, and advanced hypothesis testing for data-driven decision making.

### Objective

Learn comprehensive statistical analysis approaches including experimental design, A/B testing, causal inference, Bayesian methods, and advanced statistical modeling techniques.

### Instructions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from scipy.stats import ttest_ind, ttest_rel, f_oneway, kruskal
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import normaltest, shapiro, anderson, jarque_bera
from scipy.stats import levene, bartlett, fligner

# Advanced statistical methods
from statsmodels.stats.power import ttest_power, ztost_power
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Experimental design and causal inference
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Bayesian statistics
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    try:
        import pymc3 as pm
        import arviz as az
        PYMC_AVAILABLE = True
    except ImportError:
        PYMC_AVAILABLE = False

# Bootstrap and resampling
from sklearn.utils import resample
import itertools
from collections import defaultdict

print("📊 Advanced Statistical Analysis and Experimental Design")
print("=" * 52)

# Set random seed for reproducibility
np.random.seed(42)

print("🧪 Creating Comprehensive Experimental Datasets...")

# CHALLENGE 1: EXPERIMENTAL DESIGN DATA GENERATION
print("\n" + "=" * 60)
print("🔬 CHALLENGE 1: EXPERIMENTAL DESIGN & A/B TESTING DATA")
print("=" * 60)

def generate_experimental_datasets():
    """Generate realistic experimental and observational datasets"""

    datasets = {}

    # Dataset 1: A/B Testing for Website Conversion
    print("Creating A/B testing dataset...")

    n_users = 5000

    # Control group (A)
    n_control = n_users // 2

    # Treatment group (B)
    n_treatment = n_users - n_control

    # User characteristics (affecting baseline conversion)
    np.random.seed(42)

    # Control group
    control_data = {
        'user_id': range(n_control),
        'group': ['control'] * n_control,
        'age': np.random.normal(35, 12, n_control),
        'session_duration': np.random.exponential(5, n_control),  # minutes
        'previous_purchases': np.random.poisson(2, n_control),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_control, p=[0.6, 0.3, 0.1])
    }

    # Treatment group
    treatment_data = {
        'user_id': range(n_control, n_users),
        'group': ['treatment'] * n_treatment,
        'age': np.random.normal(35, 12, n_treatment),
        'session_duration': np.random.exponential(5, n_treatment),
        'previous_purchases': np.random.poisson(2, n_treatment),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_treatment, p=[0.6, 0.3, 0.1])
    }

    # Combine groups
    ab_test_data = pd.DataFrame({
        'user_id': list(control_data['user_id']) + list(treatment_data['user_id']),
        'group': control_data['group'] + treatment_data['group'],
        'age': np.concatenate([control_data['age'], treatment_data['age']]),
        'session_duration': np.concatenate([control_data['session_duration'], treatment_data['session_duration']]),
        'previous_purchases': np.concatenate([control_data['previous_purchases'], treatment_data['previous_purchases']]),
        'device_type': control_data['device_type'] + treatment_data['device_type']
    })

    # Generate conversion probabilities based on features
    def calculate_conversion_probability(row):
        base_prob = 0.05  # 5% baseline conversion

        # Age effect (peak conversion around 30-40)
        age_factor = 1 + 0.3 * np.exp(-((row['age'] - 35) / 15) ** 2)

        # Session duration effect (longer sessions more likely to convert)
        duration_factor = 1 + 0.5 * (1 - np.exp(-row['session_duration'] / 10))

        # Previous purchases effect (loyal customers convert more)
        purchase_factor = 1 + 0.2 * np.log1p(row['previous_purchases'])

        # Device effect
        device_factors = {'mobile': 0.8, 'desktop': 1.2, 'tablet': 1.0}
        device_factor = device_factors[row['device_type']]

        # Treatment effect (15% relative improvement)
        treatment_effect = 1.15 if row['group'] == 'treatment' else 1.0

        return base_prob * age_factor * duration_factor * purchase_factor * device_factor * treatment_effect

    ab_test_data['conversion_prob'] = ab_test_data.apply(calculate_conversion_probability, axis=1)
    ab_test_data['converted'] = np.random.binomial(1, ab_test_data['conversion_prob'])

    # Add noise and edge cases
    ab_test_data['revenue'] = np.where(
        ab_test_data['converted'] == 1,
        np.random.lognormal(4, 0.5, len(ab_test_data)),  # $50-200 average
        0
    )

    datasets['ab_test'] = ab_test_data

    # Dataset 2: Multi-armed Bandit Experiment
    print("Creating multi-armed bandit dataset...")

    n_trials = 3000
    n_arms = 4

    # True conversion rates for each arm
    true_rates = [0.12, 0.15, 0.18, 0.14]  # Arm 2 is best

    # Simulate bandit experiment with epsilon-greedy strategy
    epsilon = 0.1
    trials_data = []
    arm_counts = np.zeros(n_arms)
    arm_successes = np.zeros(n_arms)

    for trial in range(n_trials):
        # Epsilon-greedy arm selection
        if np.random.random() < epsilon or trial < n_arms:
            # Explore: random arm
            chosen_arm = np.random.randint(n_arms)
        else:
            # Exploit: best arm so far
            estimated_rates = arm_successes / (arm_counts + 1e-8)
            chosen_arm = np.argmax(estimated_rates)

        # Simulate outcome
        success = np.random.binomial(1, true_rates[chosen_arm])

        # Update counts
        arm_counts[chosen_arm] += 1
        arm_successes[chosen_arm] += success

        # Record trial
        trials_data.append({
            'trial': trial,
            'arm': chosen_arm,
            'success': success,
            'cumulative_successes': arm_successes.copy(),
            'cumulative_trials': arm_counts.copy()
        })

    bandit_df = pd.DataFrame(trials_data)
    datasets['bandit'] = bandit_df

    # Dataset 3: Observational Study with Confounders
    print("Creating observational study dataset...")

    n_subjects = 2000

    # Confounding variables
    age = np.random.normal(45, 15, n_subjects)
    income = np.random.lognormal(10.5, 0.7, n_subjects)
    education = np.random.choice(['high_school', 'college', 'graduate'],
                                n_subjects, p=[0.4, 0.4, 0.2])

    # Treatment assignment (not random - depends on confounders)
    treatment_prob = (
        0.2 +  # Base probability
        0.3 * (age > 40) +  # Older people more likely to get treatment
        0.2 * (income > np.median(income)) +  # Higher income more likely
        0.3 * (education == 'graduate')  # Graduate education more likely
    )

    treatment_prob = np.clip(treatment_prob, 0, 1)
    treatment = np.random.binomial(1, treatment_prob)

    # Outcome variable (affected by treatment and confounders)
    outcome_mean = (
        50 +  # Baseline
        10 * treatment +  # Treatment effect
        0.3 * age +  # Age effect
        0.00001 * income +  # Income effect
        5 * (education == 'college') +
        10 * (education == 'graduate') +
        np.random.normal(0, 10, n_subjects)  # Noise
    )

    observational_df = pd.DataFrame({
        'subject_id': range(n_subjects),
        'age': age,
        'income': income,
        'education': education,
        'treatment': treatment,
        'outcome': outcome_mean
    })

    datasets['observational'] = observational_df

    # Dataset 4: Factorial Experiment Design
    print("Creating factorial experiment dataset...")

    # 2x2x3 factorial design
    factor_a = ['low', 'high']  # Temperature
    factor_b = ['slow', 'fast']  # Speed
    factor_c = ['type1', 'type2', 'type3']  # Material

    # Generate all combinations
    combinations = list(itertools.product(factor_a, factor_b, factor_c))

    # Replications per combination
    n_reps = 10

    factorial_data = []

    for combo in combinations:
        temp, speed, material = combo

        for rep in range(n_reps):
            # Main effects
            temp_effect = 5 if temp == 'high' else 0
            speed_effect = 3 if speed == 'fast' else 0
            material_effects = {'type1': 0, 'type2': 2, 'type3': 4}
            material_effect = material_effects[material]

            # Interaction effects
            temp_speed_interaction = 2 if (temp == 'high' and speed == 'fast') else 0
            temp_material_interaction = 3 if (temp == 'high' and material == 'type3') else 0

            # Response variable
            response = (
                20 +  # Baseline
                temp_effect +
                speed_effect +
                material_effect +
                temp_speed_interaction +
                temp_material_interaction +
                np.random.normal(0, 2)  # Error term
            )

            factorial_data.append({
                'temperature': temp,
                'speed': speed,
                'material': material,
                'replication': rep,
                'response': response
            })

    factorial_df = pd.DataFrame(factorial_data)
    datasets['factorial'] = factorial_df

    return datasets

# Generate all experimental datasets
experimental_datasets = generate_experimental_datasets()

print(f"\nGenerated {len(experimental_datasets)} experimental datasets:")
for name, data in experimental_datasets.items():
    print(f"  • {name.replace('_', ' ').title()}: {len(data)} observations")

# CHALLENGE 2: HYPOTHESIS TESTING AND POWER ANALYSIS
print("\n" + "=" * 60)
print("🧮 CHALLENGE 2: ADVANCED HYPOTHESIS TESTING")
print("=" * 60)

def comprehensive_hypothesis_testing(group_a, group_b, alpha=0.05, test_type='auto'):
    """Comprehensive hypothesis testing with multiple methods"""

    print(f"Comparing groups: n1={len(group_a)}, n2={len(group_b)}")
    print(f"Group A: mean={np.mean(group_a):.3f}, std={np.std(group_a):.3f}")
    print(f"Group B: mean={np.mean(group_b):.3f}, std={np.std(group_b):.3f}")

    results = {}

    # 1. Normality Tests
    print("\n📊 Normality Testing:")

    # Shapiro-Wilk test (small samples)
    if len(group_a) <= 5000:
        shapiro_a = shapiro(group_a)
        shapiro_b = shapiro(group_b)
        print(f"Shapiro-Wilk Group A: W={shapiro_a.statistic:.4f}, p={shapiro_a.pvalue:.4f}")
        print(f"Shapiro-Wilk Group B: W={shapiro_b.statistic:.4f}, p={shapiro_b.pvalue:.4f}")

        results['normality'] = {
            'shapiro_a': {'statistic': shapiro_a.statistic, 'pvalue': shapiro_a.pvalue},
            'shapiro_b': {'statistic': shapiro_b.statistic, 'pvalue': shapiro_b.pvalue}
        }

    # Anderson-Darling test
    anderson_a = anderson(group_a)
    anderson_b = anderson(group_b)
    print(f"Anderson-Darling Group A: statistic={anderson_a.statistic:.4f}")
    print(f"Anderson-Darling Group B: statistic={anderson_b.statistic:.4f}")

    # 2. Variance Equality Tests
    print("\n📈 Variance Equality Testing:")

    # Levene's test (robust to non-normality)
    levene_stat, levene_p = levene(group_a, group_b)
    print(f"Levene's test: statistic={levene_stat:.4f}, p={levene_p:.4f}")

    # Bartlett's test (assumes normality)
    bartlett_stat, bartlett_p = bartlett(group_a, group_b)
    print(f"Bartlett's test: statistic={bartlett_stat:.4f}, p={bartlett_p:.4f}")

    results['variance_tests'] = {
        'levene': {'statistic': levene_stat, 'pvalue': levene_p},
        'bartlett': {'statistic': bartlett_stat, 'pvalue': bartlett_p}
    }

    # 3. Mean Comparison Tests
    print("\n🎯 Mean Comparison Tests:")

    # T-test (parametric)
    equal_var = levene_p > alpha  # Use equal variance if Levene's test is not significant
    ttest_stat, ttest_p = ttest_ind(group_a, group_b, equal_var=equal_var)
    print(f"Independent t-test: t={ttest_stat:.4f}, p={ttest_p:.4f}")

    # Mann-Whitney U test (non-parametric)
    mw_stat, mw_p = mannwhitneyu(group_a, group_b, alternative='two-sided')
    print(f"Mann-Whitney U test: U={mw_stat:.4f}, p={mw_p:.4f}")

    # Bootstrap confidence interval for difference in means
    def bootstrap_mean_diff(a, b, n_bootstrap=1000):
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            boot_a = resample(a, n_samples=len(a))
            boot_b = resample(b, n_samples=len(b))
            bootstrap_diffs.append(np.mean(boot_b) - np.mean(boot_a))
        return np.array(bootstrap_diffs)

    bootstrap_diffs = bootstrap_mean_diff(group_a, group_b)
    ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)

    print(f"Bootstrap 95% CI for mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

    results['mean_tests'] = {
        'ttest': {'statistic': ttest_stat, 'pvalue': ttest_p, 'equal_var': equal_var},
        'mannwhitney': {'statistic': mw_stat, 'pvalue': mw_p},
        'bootstrap_ci': {'lower': ci_lower, 'upper': ci_upper}
    }

    # 4. Effect Size Calculations
    print("\n📏 Effect Size Measures:")

    # Cohen's d
    pooled_std = np.sqrt(((len(group_a) - 1) * np.var(group_a, ddof=1) +
                         (len(group_b) - 1) * np.var(group_b, ddof=1)) /
                        (len(group_a) + len(group_b) - 2))
    cohens_d = (np.mean(group_b) - np.mean(group_a)) / pooled_std
    print(f"Cohen's d: {cohens_d:.4f}")

    # Glass's delta (using control group standard deviation)
    glass_delta = (np.mean(group_b) - np.mean(group_a)) / np.std(group_a, ddof=1)
    print(f"Glass's delta: {glass_delta:.4f}")

    # Hedge's g (bias-corrected Cohen's d)
    correction_factor = 1 - (3 / (4 * (len(group_a) + len(group_b) - 2) - 1))
    hedges_g = cohens_d * correction_factor
    print(f"Hedge's g: {hedges_g:.4f}")

    results['effect_sizes'] = {
        'cohens_d': cohens_d,
        'glass_delta': glass_delta,
        'hedges_g': hedges_g
    }

    return results

# Test A/B experiment data
print("\n🧪 A/B Test Analysis:")
ab_data = experimental_datasets['ab_test']

control_conversion = ab_data[ab_data['group'] == 'control']['converted']
treatment_conversion = ab_data[ab_data['group'] == 'treatment']['converted']

# Proportion test for conversion rates
n_control = len(control_conversion)
n_treatment = len(treatment_conversion)
successes_control = control_conversion.sum()
successes_treatment = treatment_conversion.sum()

print(f"\nConversion Rate Analysis:")
print(f"Control: {successes_control}/{n_control} = {successes_control/n_control:.3%}")
print(f"Treatment: {successes_treatment}/{n_treatment} = {successes_treatment/n_treatment:.3%}")

# Proportion z-test
counts = np.array([successes_treatment, successes_control])
nobs = np.array([n_treatment, n_control])
z_stat, p_value = proportions_ztest(counts, nobs)

print(f"Proportions z-test: z={z_stat:.4f}, p={p_value:.4f}")

# Confidence interval for difference in proportions
p1 = successes_treatment / n_treatment
p2 = successes_control / n_control
p_diff = p1 - p2

# Standard error for difference
se_diff = np.sqrt(p1*(1-p1)/n_treatment + p2*(1-p2)/n_control)
ci_lower = p_diff - 1.96 * se_diff
ci_upper = p_diff + 1.96 * se_diff

print(f"95% CI for conversion rate difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Revenue analysis
control_revenue = ab_data[ab_data['group'] == 'control']['revenue']
treatment_revenue = ab_data[ab_data['group'] == 'treatment']['revenue']

print(f"\nRevenue Analysis:")
revenue_test_results = comprehensive_hypothesis_testing(control_revenue, treatment_revenue)

# CHALLENGE 3: EXPERIMENTAL DESIGN AND ANOVA
print("\n" + "=" * 60)
print("🔬 CHALLENGE 3: EXPERIMENTAL DESIGN & ANOVA")
print("=" * 60)

def factorial_anova_analysis(data, response_var, factors):
    """Comprehensive factorial ANOVA analysis"""

    print(f"Factorial ANOVA: {response_var} ~ {' + '.join(factors)}")
    print(f"Data shape: {data.shape}")

    # Create formula for ANOVA
    formula = f"{response_var} ~ " + " * ".join(factors)

    # Fit model
    model = ols(formula, data=data).fit()

    # ANOVA table
    anova_table = anova_lm(model, typ=2)  # Type II SS
    print(f"\nANOVA Table:")
    print(anova_table)

    # Model diagnostics
    print(f"\nModel Diagnostics:")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"F-statistic p-value: {model.f_pvalue:.4f}")

    # Residual analysis
    residuals = model.resid
    fitted_values = model.fittedvalues

    print(f"\nResidual Analysis:")
    print(f"Mean residual: {np.mean(residuals):.6f}")
    print(f"Std residual: {np.std(residuals):.4f}")

    # Normality test on residuals
    shapiro_stat, shapiro_p = shapiro(residuals)
    print(f"Shapiro-Wilk test on residuals: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

    # Homoscedasticity tests
    # Breusch-Pagan test
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    print(f"Breusch-Pagan test: LM={bp_stat:.4f}, p={bp_p:.4f}")

    # Post-hoc analysis for significant factors
    significant_factors = []
    for factor in anova_table.index:
        if factor != 'Residual' and anova_table.loc[factor, 'PR(>F)'] < 0.05:
            significant_factors.append(factor)

    print(f"\nSignificant factors (p < 0.05): {significant_factors}")

    # Effect sizes (Eta-squared)
    ss_total = anova_table['sum_sq'].sum()
    effect_sizes = {}

    for factor in anova_table.index:
        if factor != 'Residual':
            eta_squared = anova_table.loc[factor, 'sum_sq'] / ss_total
            effect_sizes[factor] = eta_squared
            print(f"Eta-squared for {factor}: {eta_squared:.4f}")

    return model, anova_table, effect_sizes, residuals, fitted_values

# Analyze factorial experiment
print("\n🧪 Factorial Experiment Analysis:")
factorial_data = experimental_datasets['factorial']

factorial_model, factorial_anova, factorial_effects, factorial_residuals, factorial_fitted = factorial_anova_analysis(
    factorial_data, 'response', ['temperature', 'speed', 'material']
)

# CHALLENGE 4: CAUSAL INFERENCE METHODS
print("\n" + "=" * 60)
print("🔗 CHALLENGE 4: CAUSAL INFERENCE ANALYSIS")
print("=" * 60)

def causal_inference_analysis(data, treatment_col, outcome_col, confounders):
    """Advanced causal inference methods"""

    print(f"Causal Analysis: Effect of {treatment_col} on {outcome_col}")
    print(f"Confounders: {confounders}")

    results = {}

    # 1. Naive comparison (biased)
    treated = data[data[treatment_col] == 1][outcome_col]
    control = data[data[treatment_col] == 0][outcome_col]

    naive_effect = np.mean(treated) - np.mean(control)
    print(f"\n1. Naive Treatment Effect: {naive_effect:.4f}")

    results['naive_effect'] = naive_effect

    # 2. Regression Adjustment
    print(f"\n2. Regression Adjustment:")

    # Prepare features
    X = data[confounders + [treatment_col]]
    y = data[outcome_col]

    # Fit regression model
    reg_model = LinearRegression()
    reg_model.fit(X, y)

    # Treatment coefficient is the adjusted effect
    treatment_idx = X.columns.get_loc(treatment_col)
    regression_effect = reg_model.coef_[treatment_idx]

    print(f"Regression-adjusted effect: {regression_effect:.4f}")

    # R-squared for model fit
    r2 = reg_model.score(X, y)
    print(f"Model R-squared: {r2:.4f}")

    results['regression_effect'] = regression_effect
    results['regression_r2'] = r2

    # 3. Propensity Score Matching
    print(f"\n3. Propensity Score Analysis:")

    # Fit propensity score model
    X_confounders = data[confounders]
    treatment = data[treatment_col]

    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X_confounders, treatment)

    # Calculate propensity scores
    propensity_scores = ps_model.predict_proba(X_confounders)[:, 1]
    data_with_ps = data.copy()
    data_with_ps['propensity_score'] = propensity_scores

    print(f"Propensity score range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")

    # Simple matching (1:1 nearest neighbor)
    treated_indices = data_with_ps[data_with_ps[treatment_col] == 1].index
    control_indices = data_with_ps[data_with_ps[treatment_col] == 0].index

    matched_pairs = []
    used_controls = set()

    for treated_idx in treated_indices:
        treated_ps = data_with_ps.loc[treated_idx, 'propensity_score']

        # Find closest control unit
        best_control = None
        best_distance = float('inf')

        for control_idx in control_indices:
            if control_idx not in used_controls:
                control_ps = data_with_ps.loc[control_idx, 'propensity_score']
                distance = abs(treated_ps - control_ps)

                if distance < best_distance:
                    best_distance = distance
                    best_control = control_idx

        if best_control is not None and best_distance < 0.1:  # Caliper
            matched_pairs.append((treated_idx, best_control))
            used_controls.add(best_control)

    print(f"Successful matches: {len(matched_pairs)}")

    if matched_pairs:
        # Calculate treatment effect on matched sample
        matched_treated_outcomes = [data_with_ps.loc[pair[0], outcome_col] for pair in matched_pairs]
        matched_control_outcomes = [data_with_ps.loc[pair[1], outcome_col] for pair in matched_pairs]

        ps_effect = np.mean(matched_treated_outcomes) - np.mean(matched_control_outcomes)
        print(f"Propensity score matching effect: {ps_effect:.4f}")

        results['ps_effect'] = ps_effect
        results['n_matched'] = len(matched_pairs)

    # 4. Instrumental Variables (if available)
    # For demonstration, we'll use a simulated instrument
    print(f"\n4. Instrumental Variables Analysis:")

    # Create a simulated instrument (weakly related to treatment, not directly to outcome)
    np.random.seed(42)
    instrument = np.random.normal(0, 1, len(data))

    # Make instrument correlated with treatment but not outcome (except through treatment)
    treatment_with_instrument = treatment + 0.3 * instrument + np.random.normal(0, 0.1, len(data))

    # Two-stage least squares (2SLS)
    # First stage: regress treatment on instrument and confounders
    X_first_stage = np.column_stack([instrument, X_confounders])
    first_stage_model = LinearRegression()
    first_stage_model.fit(X_first_stage, treatment)

    # Predicted treatment values
    treatment_hat = first_stage_model.predict(X_first_stage)

    # Check instrument strength (F-statistic from first stage)
    first_stage_r2 = first_stage_model.score(X_first_stage, treatment)
    print(f"First stage R-squared: {first_stage_r2:.4f}")

    # Second stage: regress outcome on predicted treatment and confounders
    X_second_stage = np.column_stack([treatment_hat, X_confounders])
    second_stage_model = LinearRegression()
    second_stage_model.fit(X_second_stage, y)

    iv_effect = second_stage_model.coef_[0]
    print(f"Instrumental variables effect: {iv_effect:.4f}")

    results['iv_effect'] = iv_effect
    results['first_stage_r2'] = first_stage_r2

    return results, propensity_scores

# Analyze observational study
print("\n📊 Observational Study Causal Analysis:")
obs_data = experimental_datasets['observational']

causal_results, prop_scores = causal_inference_analysis(
    obs_data, 'treatment', 'outcome',
    ['age', 'income', 'education']
)

# CHALLENGE 5: BAYESIAN STATISTICAL ANALYSIS
print("\n" + "=" * 60)
print("🎲 CHALLENGE 5: BAYESIAN STATISTICAL METHODS")
print("=" * 60)

if PYMC_AVAILABLE:
    def bayesian_ab_test_analysis(control_data, treatment_data):
        """Bayesian A/B test analysis using PyMC"""

        print("🎯 Bayesian A/B Test Analysis")

        # Data preparation
        n_control = len(control_data)
        n_treatment = len(treatment_data)
        successes_control = control_data.sum()
        successes_treatment = treatment_data.sum()

        print(f"Control: {successes_control}/{n_control}")
        print(f"Treatment: {successes_treatment}/{n_treatment}")

        # Bayesian model
        with pm.Model() as model:
            # Priors for conversion rates (Beta distribution)
            p_control = pm.Beta('p_control', alpha=1, beta=1)
            p_treatment = pm.Beta('p_treatment', alpha=1, beta=1)

            # Likelihood
            obs_control = pm.Binomial('obs_control', n=n_control, p=p_control,
                                     observed=successes_control)
            obs_treatment = pm.Binomial('obs_treatment', n=n_treatment, p=p_treatment,
                                       observed=successes_treatment)

            # Derived quantities
            delta = pm.Deterministic('delta', p_treatment - p_control)
            relative_lift = pm.Deterministic('relative_lift',
                                           (p_treatment - p_control) / p_control)

            # Sample from posterior
            trace = pm.sample(2000, tune=1000, random_seed=42,
                             return_inferencedata=True, chains=2)

        # Posterior analysis
        posterior_summary = az.summary(trace)
        print("\nPosterior Summary:")
        print(posterior_summary)

        # Calculate probabilities
        delta_samples = trace.posterior['delta'].values.flatten()
        prob_positive = (delta_samples > 0).mean()
        prob_significant = (np.abs(delta_samples) > 0.01).mean()  # >1% difference

        print(f"\nProbability that treatment > control: {prob_positive:.3f}")
        print(f"Probability of meaningful difference (>1%): {prob_significant:.3f}")

        # Credible intervals
        delta_ci = np.percentile(delta_samples, [2.5, 97.5])
        print(f"95% Credible interval for difference: [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]")

        return trace, posterior_summary, prob_positive

    # Run Bayesian A/B test
    print("\n🎲 Bayesian A/B Test:")
    try:
        bayesian_trace, bayesian_summary, prob_positive = bayesian_ab_test_analysis(
            control_conversion, treatment_conversion
        )
        print("✅ Bayesian analysis completed successfully")
    except Exception as e:
        print(f"⚠️ Bayesian analysis failed: {e}")
        bayesian_trace = None

else:
    print("⚠️ PyMC not available, skipping Bayesian analysis")
    bayesian_trace = None

# CHALLENGE 6: POWER ANALYSIS AND SAMPLE SIZE CALCULATION
print("\n" + "=" * 60)
print("⚡ CHALLENGE 6: POWER ANALYSIS & SAMPLE SIZE")
print("=" * 60)

def comprehensive_power_analysis(effect_size, alpha=0.05, power=0.8):
    """Comprehensive power analysis for different test types"""

    print(f"Power Analysis Parameters:")
    print(f"Effect size: {effect_size}")
    print(f"Alpha: {alpha}")
    print(f"Desired power: {power}")

    results = {}

    # Two-sample t-test
    try:
        n_ttest = int(np.ceil(ttest_power(effect_size, power, alpha, alternative='two-sided')))
        print(f"\nTwo-sample t-test required sample size: {n_ttest} per group")
        results['ttest_n'] = n_ttest
    except:
        print("Could not calculate t-test sample size")

    # Proportion test (approximation)
    def proportion_sample_size(p1, p2, alpha=0.05, power=0.8):
        """Calculate sample size for proportion test"""
        from scipy.stats import norm

        # Two-sided test
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)

        p_avg = (p1 + p2) / 2

        n = ((z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2

        return int(np.ceil(n))

    # Example for conversion rate test
    p_control = 0.10  # 10% baseline conversion
    p_treatment = p_control * (1 + effect_size)  # Relative improvement

    if p_treatment <= 1.0:
        n_prop = proportion_sample_size(p_control, p_treatment, alpha, power)
        print(f"Proportion test required sample size: {n_prop} per group")
        print(f"  (for {p_control:.1%} vs {p_treatment:.1%} conversion rates)")
        results['proportion_n'] = n_prop

    # Power curves
    sample_sizes = np.arange(10, 500, 10)
    powers = []

    for n in sample_sizes:
        try:
            power_calc = ttest_power(effect_size, None, alpha, alternative='two-sided')
            # Approximate power calculation (not exact but illustrative)
            approx_power = 1 - stats.t.cdf(
                stats.t.ppf(1 - alpha/2, 2*n - 2) - effect_size * np.sqrt(n/2),
                2*n - 2
            )
            powers.append(approx_power)
        except:
            powers.append(0)

    results['power_curve'] = {'sample_sizes': sample_sizes, 'powers': powers}

    return results

# Perform power analysis
print("\n⚡ Power Analysis for Different Effect Sizes:")
effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large effects

power_results = {}
for es in effect_sizes:
    print(f"\n--- Effect Size: {es} ---")
    power_results[es] = comprehensive_power_analysis(es)

# CHALLENGE 7: COMPREHENSIVE VISUALIZATION
print("\n" + "=" * 60)
print("📊 CHALLENGE 7: STATISTICAL ANALYSIS VISUALIZATION")
print("=" * 60)

# Create comprehensive visualization
fig, axes = plt.subplots(4, 4, figsize=(24, 20))
fig.suptitle('Advanced Statistical Analysis and Experimental Design', fontsize=16, fontweight='bold')

# Plot 1: A/B Test Results
ax = axes[0, 0]
groups = ['Control', 'Treatment']
conversion_rates = [successes_control/n_control, successes_treatment/n_treatment]
errors = [np.sqrt(p*(1-p)/n) for p, n in zip(conversion_rates, [n_control, n_treatment])]

bars = ax.bar(groups, conversion_rates, yerr=errors, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
ax.set_ylabel('Conversion Rate')
ax.set_title('A/B Test Results')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, rate in zip(bars, conversion_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
           f'{rate:.3%}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Revenue Distribution by Group
ax = axes[0, 1]
ax.boxplot([control_revenue, treatment_revenue], labels=['Control', 'Treatment'])
ax.set_ylabel('Revenue ($)')
ax.set_title('Revenue Distribution by Group')
ax.grid(True, alpha=0.3)

# Plot 3: Factorial Experiment Main Effects
ax = axes[0, 2]
factorial_means = factorial_data.groupby(['temperature', 'speed'])['response'].mean().unstack()
sns.heatmap(factorial_means, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax)
ax.set_title('Factorial Design: Temperature × Speed')

# Plot 4: ANOVA Residuals
ax = axes[0, 3]
ax.scatter(factorial_fitted, factorial_residuals, alpha=0.6)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Fitted (ANOVA)')
ax.grid(True, alpha=0.3)

# Plot 5: Propensity Score Distribution
ax = axes[1, 0]
treated_ps = prop_scores[obs_data['treatment'] == 1]
control_ps = prop_scores[obs_data['treatment'] == 0]

ax.hist(control_ps, bins=30, alpha=0.6, label='Control', color='skyblue')
ax.hist(treated_ps, bins=30, alpha=0.6, label='Treated', color='lightcoral')
ax.set_xlabel('Propensity Score')
ax.set_ylabel('Frequency')
ax.set_title('Propensity Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Effect Size Comparison
ax = axes[1, 1]
methods = ['Naive', 'Regression', 'Prop. Score', 'IV']
effects = [
    causal_results['naive_effect'],
    causal_results['regression_effect'],
    causal_results.get('ps_effect', 0),
    causal_results['iv_effect']
]

colors = ['red', 'blue', 'green', 'orange']
bars = ax.bar(methods, effects, color=colors, alpha=0.7)
ax.axhline(y=10, color='black', linestyle='--', alpha=0.5, label='True Effect')
ax.set_ylabel('Treatment Effect')
ax.set_title('Causal Inference Methods Comparison')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, effect in zip(bars, effects):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
           f'{effect:.2f}', ha='center', va='bottom', fontweight='bold')

# Plot 7: Power Curves
ax = axes[1, 2]
for es in effect_sizes:
    if es in power_results:
        sample_sizes = power_results[es]['power_curve']['sample_sizes']
        powers = power_results[es]['power_curve']['powers']
        ax.plot(sample_sizes, powers, label=f'Effect Size = {es}', alpha=0.8)

ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Power')
ax.set_xlabel('Sample Size per Group')
ax.set_ylabel('Statistical Power')
ax.set_title('Power Curves by Effect Size')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 8: Bandit Performance
ax = axes[1, 3]
bandit_data = experimental_datasets['bandit']
# Calculate cumulative regret
optimal_arm = 2  # Arm with highest true rate (0.18)
true_optimal_rate = 0.18

cumulative_regret = []
cumulative_optimal_reward = 0
cumulative_actual_reward = 0

for i in range(len(bandit_data)):
    chosen_arm = bandit_data.iloc[i]['arm']
    success = bandit_data.iloc[i]['success']

    cumulative_optimal_reward += true_optimal_rate
    cumulative_actual_reward += success

    regret = cumulative_optimal_reward - cumulative_actual_reward
    cumulative_regret.append(regret)

ax.plot(range(len(cumulative_regret)), cumulative_regret, alpha=0.8)
ax.set_xlabel('Trial')
ax.set_ylabel('Cumulative Regret')
ax.set_title('Multi-Armed Bandit: Cumulative Regret')
ax.grid(True, alpha=0.3)

# Plot 9-12: Statistical Test Visualizations
test_results = [
    ('Normality', 'Shapiro-Wilk p-values'),
    ('Variance Equality', 'Levene Test p-values'),
    ('Mean Comparison', 'T-test vs Mann-Whitney'),
    ('Effect Sizes', 'Cohen\'s d Distribution')
]

# Plot 9: Sample Size Requirements
ax = axes[2, 0]
effect_sizes_plot = list(power_results.keys())
sample_sizes_ttest = [power_results[es].get('ttest_n', 0) for es in effect_sizes_plot]
sample_sizes_prop = [power_results[es].get('proportion_n', 0) for es in effect_sizes_plot]

x_pos = np.arange(len(effect_sizes_plot))
width = 0.35

ax.bar(x_pos - width/2, sample_sizes_ttest, width, label='T-test', alpha=0.7)
ax.bar(x_pos + width/2, sample_sizes_prop, width, label='Proportion test', alpha=0.7)
ax.set_xlabel('Effect Size')
ax.set_ylabel('Required Sample Size per Group')
ax.set_title('Sample Size Requirements')
ax.set_xticks(x_pos)
ax.set_xticklabels(effect_sizes_plot)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 10: Bootstrap Distribution
ax = axes[2, 1]
# Generate bootstrap distribution for demonstration
n_bootstrap = 1000
bootstrap_means = []
sample_data = control_revenue.values

for _ in range(n_bootstrap):
    bootstrap_sample = resample(sample_data, n_samples=len(sample_data))
    bootstrap_means.append(np.mean(bootstrap_sample))

ax.hist(bootstrap_means, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(sample_data), color='red', linestyle='--',
          linewidth=2, label='Original Mean')
ax.set_xlabel('Bootstrap Sample Mean')
ax.set_ylabel('Frequency')
ax.set_title('Bootstrap Distribution of Sample Mean')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 11: Factorial Interaction Plot
ax = axes[2, 2]
interaction_data = factorial_data.groupby(['temperature', 'material'])['response'].mean().unstack()
for material in interaction_data.columns:
    ax.plot(['low', 'high'], interaction_data[material], marker='o',
           label=material, linewidth=2, alpha=0.8)

ax.set_xlabel('Temperature')
ax.set_ylabel('Mean Response')
ax.set_title('Temperature × Material Interaction')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 12: Bayesian Posterior (if available)
ax = axes[2, 3]
if PYMC_AVAILABLE and bayesian_trace is not None:
    try:
        delta_samples = bayesian_trace.posterior['delta'].values.flatten()
        ax.hist(delta_samples, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No Effect')
        ax.axvline(np.mean(delta_samples), color='green', linestyle='--',
                  linewidth=2, label='Posterior Mean')
        ax.set_xlabel('Treatment Effect (Delta)')
        ax.set_ylabel('Posterior Density')
        ax.set_title('Bayesian Posterior Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, 'Bayesian Analysis\nNot Available',
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
else:
    ax.text(0.5, 0.5, 'Bayesian Analysis\nNot Available',
           ha='center', va='center', fontsize=12, transform=ax.transAxes)

# Plot 13: Confidence Intervals Comparison
ax = axes[3, 0]
methods = ['Frequentist\n95% CI', 'Bootstrap\n95% CI', 'Bayesian\n95% CrI']

# Use A/B test results
freq_ci = [ci_lower, ci_upper]
bootstrap_ci = revenue_test_results['mean_tests']['bootstrap_ci']
bootstrap_ci_vals = [bootstrap_ci['lower'], bootstrap_ci['upper']]

# Mock Bayesian CI for demonstration
if bayesian_trace is not None:
    try:
        delta_samples = bayesian_trace.posterior['delta'].values.flatten()
        bayesian_ci = np.percentile(delta_samples, [2.5, 97.5])
    except:
        bayesian_ci = [0, 0]
else:
    bayesian_ci = [0, 0]

cis = [freq_ci, bootstrap_ci_vals, bayesian_ci]
centers = [np.mean(ci) for ci in cis]
errors = [[center - ci[0], ci[1] - center] for center, ci in zip(centers, cis)]

ax.errorbar(range(len(methods)), centers,
           yerr=np.array(errors).T, fmt='o', capsize=8, capthick=2)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods)
ax.set_ylabel('Effect Size')
ax.set_title('Confidence Intervals Comparison')
ax.grid(True, alpha=0.3)

# Plot 14: Type I and Type II Error Illustration
ax = axes[3, 1]
x = np.linspace(-4, 4, 1000)
null_dist = stats.norm.pdf(x, 0, 1)
alt_dist = stats.norm.pdf(x, 2, 1)  # Effect size = 2

ax.plot(x, null_dist, label='Null Hypothesis', alpha=0.8)
ax.plot(x, alt_dist, label='Alternative Hypothesis', alpha=0.8)

# Critical value for alpha = 0.05
critical_value = stats.norm.ppf(0.975)
ax.axvline(critical_value, color='red', linestyle='--', alpha=0.7, label='Critical Value')

# Fill Type I error area
type_i_x = x[x >= critical_value]
type_i_y = null_dist[x >= critical_value]
ax.fill_between(type_i_x, type_i_y, alpha=0.3, color='red', label='Type I Error')

# Fill Type II error area
type_ii_x = x[x <= critical_value]
type_ii_y = alt_dist[x <= critical_value]
ax.fill_between(type_ii_x, type_ii_y, alpha=0.3, color='blue', label='Type II Error')

ax.set_xlabel('Test Statistic')
ax.set_ylabel('Probability Density')
ax.set_title('Type I and Type II Errors')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 15: P-value Distribution Under Null
ax = axes[3, 2]
# Simulate p-values under null hypothesis
n_simulations = 1000
p_values = []

for _ in range(n_simulations):
    # Generate random data under null
    group1 = np.random.normal(0, 1, 30)
    group2 = np.random.normal(0, 1, 30)
    _, p_val = ttest_ind(group1, group2)
    p_values.append(p_val)

ax.hist(p_values, bins=50, alpha=0.7, edgecolor='black')
ax.axhline(y=n_simulations/20, color='red', linestyle='--',
          alpha=0.7, label='Expected (Uniform)')
ax.set_xlabel('P-value')
ax.set_ylabel('Frequency')
ax.set_title('P-value Distribution Under Null')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 16: Effect Size Interpretation
ax = axes[3, 3]
effect_categories = ['Small\n(0.2)', 'Medium\n(0.5)', 'Large\n(0.8)', 'Very Large\n(1.2)']
cohen_benchmarks = [0.2, 0.5, 0.8, 1.2]
colors = ['lightblue', 'lightgreen', 'orange', 'red']

bars = ax.bar(effect_categories, cohen_benchmarks, color=colors, alpha=0.7)
ax.set_ylabel("Cohen's d")
ax.set_title("Effect Size Interpretation (Cohen's Guidelines)")
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, cohen_benchmarks):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{value}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("📊 STATISTICAL ANALYSIS INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

print("🔍 Key Findings:")
print(f"\n1. A/B Test Results:")
print(f"   • Control conversion: {successes_control/n_control:.3%}")
print(f"   • Treatment conversion: {successes_treatment/n_treatment:.3%}")
print(f"   • Relative improvement: {((successes_treatment/n_treatment)/(successes_control/n_control) - 1)*100:.1f}%")
print(f"   • Statistical significance: p = {p_value:.4f}")

print(f"\n2. Causal Inference Analysis:")
print(f"   • Naive effect estimate: {causal_results['naive_effect']:.2f}")
print(f"   • Regression-adjusted effect: {causal_results['regression_effect']:.2f}")
if 'ps_effect' in causal_results:
    print(f"   • Propensity score effect: {causal_results['ps_effect']:.2f}")
print(f"   • Instrumental variables effect: {causal_results['iv_effect']:.2f}")

print(f"\n3. Factorial Experiment:")
significant_effects = [factor for factor, p_val in factorial_anova['PR(>F)'].items()
                      if factor != 'Residual' and p_val < 0.05]
print(f"   • Significant factors: {significant_effects}")
print(f"   • Model R-squared: {factorial_model.rsquared:.3f}")

print(f"\n🎯 Statistical Method Recommendations:")
print("\nExperimental Design:")
print("• Use randomized controlled trials when possible")
print("• Implement proper sample size calculations before experiments")
print("• Consider factorial designs for multiple factor analysis")
print("• Plan for adequate power (typically 80% or higher)")

print("\nHypothesis Testing:")
print("• Always check assumptions (normality, equal variance)")
print("• Use appropriate tests for data type and distribution")
print("• Report effect sizes alongside significance tests")
print("• Consider multiple comparison corrections when needed")

print("\nCausal Inference:")
print("• Use randomized experiments for strongest causal claims")
print("• Apply multiple methods to observational data")
print("• Check for unmeasured confounders and selection bias")
print("• Validate results with domain expertise")

print(f"\n📈 Best Practices for Experimental Design:")
print("\n1. Pre-experimental Phase:")
print("   • Define clear hypotheses and success metrics")
print("   • Calculate required sample sizes")
print("   • Plan randomization and blocking strategies")
print("   • Consider potential confounders and covariates")

print("\n2. During Experiment:")
print("   • Monitor for violations of assumptions")
print("   • Check for differential attrition")
print("   • Maintain experimental integrity")
print("   • Document any deviations from protocol")

print("\n3. Post-experimental Analysis:")
print("   • Conduct comprehensive assumption checking")
print("   • Report both statistical and practical significance")
print("   • Perform sensitivity analyses")
print("   • Consider external validity and generalizability")

print(f"\n🔧 Advanced Considerations:")
print("\n1. Multiple Testing:")
print("   • Use Bonferroni, FDR, or other corrections")
print("   • Pre-specify primary and secondary endpoints")
print("   • Consider hierarchical testing strategies")

print("\n2. Power and Sample Size:")
print("   • Account for expected effect sizes")
print("   • Consider practical constraints and costs")
print("   • Plan for adequate follow-up periods")
print("   • Use adaptive designs when appropriate")

print("\n3. Causal Inference:")
print("   • Implement proper identification strategies")
print("   • Use instrumental variables when available")
print("   • Consider natural experiments and discontinuities")
print("   • Apply machine learning for confounder selection")

if PYMC_AVAILABLE and bayesian_trace is not None:
    print(f"\n4. Bayesian Methods:")
    print(f"   • Probability treatment is better: {prob_positive:.1%}")
    print("   • Provides full uncertainty quantification")
    print("   • Allows for prior knowledge incorporation")
    print("   • Enables sequential testing and early stopping")

print(f"\n🎖️ Quality Assurance Checklist:")
print("✓ Assumptions validated before analysis")
print("✓ Appropriate statistical methods selected")
print("✓ Effect sizes reported with confidence intervals")
print("✓ Multiple testing corrections applied when needed")
print("✓ Practical significance considered alongside statistical")
print("✓ Results reproducible with documented code")
print("✓ Limitations and assumptions clearly stated")

print("\n✅ Advanced Statistical Analysis and Experimental Design Challenge Completed!")
print("What you've mastered:")
print("• Comprehensive experimental design principles")
print("• Advanced hypothesis testing and assumption checking")
print("• Sophisticated A/B testing and multi-armed bandits")
print("• Causal inference methods and propensity score analysis")
print("• Factorial ANOVA and interaction analysis")
print("• Bayesian statistical methods (if PyMC available)")
print("• Power analysis and sample size calculations")
print("• Production-ready statistical analysis frameworks")

print(f"\n📊 You are now an Advanced Statistical Analysis Expert! Ready for Level 7!")
```
