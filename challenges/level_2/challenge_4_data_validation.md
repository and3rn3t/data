# Level 2: Analytics Apprentice

## Challenge 4: Data Validation with Pandera

Master modern data validation techniques using Pandera to ensure data quality and consistency.

### Objective

Learn to create data schemas, validate datasets, and implement automated data quality checks using modern tools.

### Instructions

```python
import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import warnings
warnings.filterwarnings('ignore')

# Create sample e-commerce dataset
np.random.seed(42)
n_samples = 1000

# Generate realistic e-commerce data
data = pd.DataFrame({
    'user_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 80, n_samples),
    'email': [f'user{i}@example.com' for i in range(1, n_samples + 1)],
    'registration_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
    'country': np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France'], n_samples),
    'purchase_amount': np.random.gamma(2, 50, n_samples),
    'session_duration_minutes': np.random.exponential(15, n_samples),
    'pages_viewed': np.random.poisson(5, n_samples),
    'is_premium': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
    'customer_satisfaction': np.random.uniform(1, 10, n_samples)
})

print("Original dataset info:")
print(data.info())
print(f"\nFirst few rows:")
print(data.head())

# Task 1: Define comprehensive data schema
print("\n=== DEFINING DATA SCHEMA ===")

# Create Pandera schema with comprehensive validation rules
user_schema = DataFrameSchema({
    'user_id': Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.less_than_or_equal_to(100000)
        ],
        nullable=False,
        unique=True,
        description="Unique user identifier"
    ),
    'age': Column(
        int,
        checks=[
            Check.greater_than_or_equal_to(18),
            Check.less_than_or_equal_to(120)
        ],
        nullable=False,
        description="User age in years"
    ),
    'email': Column(
        str,
        checks=[
            Check.str_contains('@'),
            Check.str_contains('\.com|\.org|\.edu'),
            Check.str_length(min_val=5, max_val=50)
        ],
        nullable=False,
        unique=True,
        description="User email address"
    ),
    'registration_date': Column(
        'datetime64[ns]',
        checks=[
            Check.greater_than_or_equal_to(pd.Timestamp('2020-01-01')),
            Check.less_than_or_equal_to(pd.Timestamp('2025-12-31'))
        ],
        nullable=False,
        description="User registration date"
    ),
    'country': Column(
        str,
        checks=[
            Check.isin(['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'Japan'])
        ],
        nullable=False,
        description="User country"
    ),
    'purchase_amount': Column(
        float,
        checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than(10000)  # Reasonable upper limit
        ],
        nullable=False,
        description="Purchase amount in USD"
    ),
    'session_duration_minutes': Column(
        float,
        checks=[
            Check.greater_than(0),
            Check.less_than(480)  # Max 8 hours
        ],
        nullable=False,
        description="Session duration in minutes"
    ),
    'pages_viewed': Column(
        int,
        checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than(1000)  # Reasonable upper limit
        ],
        nullable=False,
        description="Number of pages viewed"
    ),
    'is_premium': Column(
        bool,
        nullable=False,
        description="Premium membership status"
    ),
    'customer_satisfaction': Column(
        float,
        checks=[
            Check.greater_than_or_equal_to(1),
            Check.less_than_or_equal_to(10)
        ],
        nullable=False,
        description="Customer satisfaction rating (1-10)"
    )
})

print("✅ Schema defined with comprehensive validation rules")

# Task 2: Validate clean dataset
print("\n=== VALIDATING CLEAN DATASET ===")

try:
    validated_data = user_schema.validate(data)
    print("✅ Dataset validation successful!")
    print(f"Validated {len(validated_data)} records")
except pa.errors.SchemaError as e:
    print("❌ Schema validation failed:")
    print(e)

# Task 3: Create corrupted data and test validation
print("\n=== TESTING WITH CORRUPTED DATA ===")

# Create intentionally corrupted dataset
corrupted_data = data.copy()
corrupted_data.loc[0, 'age'] = -5  # Invalid age
corrupted_data.loc[1, 'email'] = 'invalid_email'  # Invalid email
corrupted_data.loc[2, 'purchase_amount'] = -100  # Negative purchase
corrupted_data.loc[3, 'customer_satisfaction'] = 15  # Out of range rating
corrupted_data.loc[4, 'country'] = 'INVALID_COUNTRY'  # Invalid country

print("Corrupted data sample:")
print(corrupted_data.head())

try:
    user_schema.validate(corrupted_data)
    print("✅ Validation passed (unexpected!)")
except pa.errors.SchemaError as e:
    print("❌ Validation failed as expected:")
    print(f"Error details: {str(e)[:200]}...")

    # Get detailed error information
    try:
        user_schema.validate(corrupted_data, lazy=True)
    except pa.errors.SchemaErrors as e:
        print(f"\nDetailed validation errors ({len(e.schema_errors)} total):")
        for i, error in enumerate(e.schema_errors[:5]):  # Show first 5 errors
            print(f"{i+1}. {error}")

# Task 4: Custom validation functions
print("\n=== CUSTOM VALIDATION FUNCTIONS ===")

def validate_email_domain(email_series):
    """Custom validator for email domains"""
    valid_domains = ['example.com', 'test.org', 'company.edu']
    return email_series.str.split('@').str[1].isin(valid_domains)

def validate_purchase_consistency(dataframe):
    """Custom validator for business logic"""
    # Premium users should have higher average purchases
    premium_avg = dataframe[dataframe['is_premium']]['purchase_amount'].mean()
    regular_avg = dataframe[~dataframe['is_premium']]['purchase_amount'].mean()
    return premium_avg > regular_avg

# Enhanced schema with custom validations
enhanced_schema = DataFrameSchema({
    'user_id': Column(int, checks=Check.greater_than(0), unique=True),
    'age': Column(int, checks=Check.in_range(18, 120)),
    'email': Column(str, checks=[
        Check.str_contains('@'),
        Check(validate_email_domain, element_wise=False, error="Invalid email domain")
    ]),
    'registration_date': Column('datetime64[ns]'),
    'country': Column(str, checks=Check.isin(['USA', 'UK', 'Canada', 'Germany', 'France'])),
    'purchase_amount': Column(float, checks=Check.greater_than_or_equal_to(0)),
    'session_duration_minutes': Column(float, checks=Check.greater_than(0)),
    'pages_viewed': Column(int, checks=Check.greater_than_or_equal_to(0)),
    'is_premium': Column(bool),
    'customer_satisfaction': Column(float, checks=Check.in_range(1, 10))
}, checks=[
    Check(validate_purchase_consistency, error="Premium users should have higher average purchases")
])

# Task 5: Data quality monitoring
print("\n=== DATA QUALITY MONITORING ===")

def generate_data_quality_report(dataframe, schema):
    """Generate comprehensive data quality report"""
    report = {
        'total_records': len(dataframe),
        'total_columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().sum(),
        'duplicate_records': dataframe.duplicated().sum(),
        'schema_compliance': True
    }

    try:
        schema.validate(dataframe)
        report['validation_errors'] = 0
    except pa.errors.SchemaError as e:
        report['schema_compliance'] = False
        report['validation_errors'] = 1
    except pa.errors.SchemaErrors as e:
        report['schema_compliance'] = False
        report['validation_errors'] = len(e.schema_errors)

    # Calculate data quality score (0-100)
    quality_score = 100
    if report['missing_values'] > 0:
        quality_score -= (report['missing_values'] / (report['total_records'] * report['total_columns'])) * 20
    if report['duplicate_records'] > 0:
        quality_score -= (report['duplicate_records'] / report['total_records']) * 10
    if not report['schema_compliance']:
        quality_score -= min(report['validation_errors'] * 5, 30)

    report['quality_score'] = max(quality_score, 0)

    return report

# Generate reports for both datasets
clean_report = generate_data_quality_report(data, user_schema)
corrupted_report = generate_data_quality_report(corrupted_data, user_schema)

print("📊 Data Quality Report - Clean Dataset:")
for key, value in clean_report.items():
    print(f"  {key}: {value}")

print(f"\n📊 Data Quality Report - Corrupted Dataset:")
for key, value in corrupted_report.items():
    print(f"  {key}: {value}")

# Task 6: Automated data pipeline with validation
print("\n=== AUTOMATED DATA PIPELINE ===")

class DataValidationPipeline:
    """Data validation pipeline class"""

    def __init__(self, schema):
        self.schema = schema
        self.validation_history = []

    def validate_and_process(self, dataframe, name="dataset"):
        """Validate dataset and log results"""
        start_time = pd.Timestamp.now()

        try:
            validated_df = self.schema.validate(dataframe)
            status = "SUCCESS"
            errors = 0
            result_df = validated_df
        except pa.errors.SchemaErrors as e:
            status = "FAILED"
            errors = len(e.schema_errors)
            result_df = None
        except pa.errors.SchemaError as e:
            status = "FAILED"
            errors = 1
            result_df = None

        # Log validation result
        log_entry = {
            'timestamp': start_time,
            'dataset_name': name,
            'records': len(dataframe),
            'status': status,
            'errors': errors,
            'processing_time': (pd.Timestamp.now() - start_time).total_seconds()
        }

        self.validation_history.append(log_entry)

        print(f"Pipeline result for '{name}':")
        print(f"  Status: {status}")
        print(f"  Records processed: {len(dataframe)}")
        print(f"  Errors: {errors}")
        print(f"  Processing time: {log_entry['processing_time']:.3f} seconds")

        return result_df, log_entry

# Create and test pipeline
pipeline = DataValidationPipeline(user_schema)

# Test with different datasets
clean_result, clean_log = pipeline.validate_and_process(data, "clean_dataset")
corrupted_result, corrupted_log = pipeline.validate_and_process(corrupted_data, "corrupted_dataset")

print(f"\n📈 Pipeline History:")
history_df = pd.DataFrame(pipeline.validation_history)
print(history_df)
```

### Success Criteria

- Define comprehensive data schemas with Pandera
- Implement custom validation functions
- Create automated data quality monitoring
- Build a reusable data validation pipeline
- Generate detailed data quality reports

### Learning Objectives

- Master modern data validation techniques
- Learn to create robust data schemas
- Understand automated data quality monitoring
- Practice building reusable data pipelines
- Implement custom business logic validation

---

_Pro tip: Data validation should be the first step in any data science pipeline - catch issues early!_

_Pro tip: Data validation should be the first step in any data science pipeline - catch issues early!_
