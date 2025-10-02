"""
Data Engineering & Pipeline Tools Integration

Provides integration with modern data engineering tools for building
production-ready data pipelines and workflows.
"""

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Union

import pandas as pd


class DataPipelineBuilder:
    """
    Integration class for data engineering and pipeline tools.

    Supports Great Expectations for data validation, workflow orchestration tools,
    and data quality monitoring with educational examples.
    """

    def __init__(self):
        """Initialize the data pipeline builder."""
        self.great_expectations_available = self._check_great_expectations()
        self.prefect_available = self._check_prefect()
        self.pandera_available = self._check_pandera()
        self.sqlmodel_available = self._check_sqlmodel()

        self.validation_results = []
        self.pipeline_runs = []

    def _check_great_expectations(self) -> bool:
        """Check if Great Expectations is available."""
        try:
            import great_expectations as ge
            from great_expectations.core import ExpectationSuite

            self.ge = ge
            self.ExpectationSuite = ExpectationSuite
            return True
        except ImportError:
            return False

    def _check_prefect(self) -> bool:
        """Check if Prefect is available."""
        try:
            import prefect
            from prefect import flow, task

            self.prefect = prefect
            self.flow = flow
            self.task = task
            return True
        except ImportError:
            return False

    def _check_pandera(self) -> bool:
        """Check if Pandera is available."""
        try:
            import pandera as pa

            self.pa = pa
            return True
        except ImportError:
            return False

    def _check_sqlmodel(self) -> bool:
        """Check if SQLModel is available."""
        try:
            import sqlmodel

            self.sqlmodel = sqlmodel
            return True
        except ImportError:
            return False

    def create_data_validation_suite(
        self, data: pd.DataFrame, suite_name: str = "default_suite"
    ) -> Dict[str, Any]:
        """
        Create a comprehensive data validation suite.

        Args:
            data: DataFrame to create expectations for
            suite_name: Name for the expectation suite

        Returns:
            Dictionary with validation suite and results
        """
        results = {"suite_name": suite_name, "expectations": [], "method": "fallback"}

        if self.great_expectations_available:
            results.update(self._create_ge_suite(data, suite_name))
        elif self.pandera_available:
            results.update(self._create_pandera_schema(data, suite_name))
        else:
            results.update(self._create_basic_validation(data, suite_name))

        return results

    def _create_ge_suite(self, data, suite_name):
        """Create validation suite using Great Expectations."""
        try:
            # Convert pandas DataFrame to GE DataFrame
            ge_df = self.ge.from_pandas(data)

            expectations = []

            # Basic expectations for all columns
            for column in data.columns:
                # Expect column to exist
                expectation = f"expect_column_to_exist('{column}')"
                expectations.append(expectation)
                ge_df.expect_column_to_exist(column)

                # Expect no null values for non-nullable columns
                null_count = data[column].isnull().sum()
                if null_count == 0:
                    expectation = f"expect_column_values_to_not_be_null('{column}')"
                    expectations.append(expectation)
                    ge_df.expect_column_values_to_not_be_null(column)

                # Type-specific expectations
                if data[column].dtype in ["int64", "float64"]:
                    # Numeric columns
                    min_val = data[column].min()
                    max_val = data[column].max()

                    expectation = f"expect_column_values_to_be_between('{column}', {min_val}, {max_val})"
                    expectations.append(expectation)
                    ge_df.expect_column_values_to_be_between(column, min_val, max_val)

                elif data[column].dtype == "object":
                    # String columns
                    unique_values = data[column].nunique()
                    total_values = len(data[column])

                    if unique_values < total_values * 0.1:  # Categorical-like
                        expectation = f"expect_column_distinct_values_to_be_in_set('{column}', {list(data[column].unique())[:10]})"
                        expectations.append(expectation)

            # Dataset-level expectations
            row_count = len(data)
            expectations.append(
                f"expect_table_row_count_to_be_between({int(row_count * 0.9)}, {int(row_count * 1.1)})"
            )
            ge_df.expect_table_row_count_to_be_between(
                int(row_count * 0.9), int(row_count * 1.1)
            )

            column_count = len(data.columns)
            expectations.append(f"expect_table_column_count_to_equal({column_count})")
            ge_df.expect_table_column_count_to_equal(column_count)

            # Get validation results
            validation_results = ge_df.validate()

            return {
                "method": "great_expectations",
                "expectations": expectations,
                "validation_results": {
                    "success": validation_results.success,
                    "statistics": validation_results.statistics,
                    "results_count": len(validation_results.results),
                },
                "ge_dataframe": ge_df,
            }

        except Exception as e:
            print(f"Great Expectations validation failed: {e}")
            return self._create_basic_validation(data, suite_name)

    def _create_pandera_schema(self, data, suite_name):
        """Create validation schema using Pandera."""
        try:
            # Infer schema from data
            schema = self.pa.infer_schema(data)

            # Validate data against schema
            validated_data = schema.validate(data)

            expectations = []
            for column, column_schema in schema.columns.items():
                expectations.append(f"Column '{column}': {column_schema.dtype}")
                if column_schema.nullable is False:
                    expectations.append(f"Column '{column}': not nullable")
                if hasattr(column_schema, "checks") and column_schema.checks:
                    for check in column_schema.checks:
                        expectations.append(f"Column '{column}': {check}")

            return {
                "method": "pandera",
                "expectations": expectations,
                "schema": schema,
                "validation_success": True,
            }

        except Exception as e:
            print(f"Pandera validation failed: {e}")
            return self._create_basic_validation(data, suite_name)

    def _create_basic_validation(self, data, suite_name):
        """Create basic validation using pandas."""
        expectations = []
        issues = []

        # Basic data quality checks
        total_rows = len(data)
        total_cols = len(data.columns)

        expectations.append(f"Dataset has {total_rows} rows and {total_cols} columns")

        # Check for missing values
        missing_data = data.isnull().sum()
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                missing_pct = (missing_count / total_rows) * 100
                expectations.append(
                    f"Column '{col}': {missing_count} missing values ({missing_pct:.1f}%)"
                )
                if missing_pct > 50:
                    issues.append(
                        f"High missing data in column '{col}': {missing_pct:.1f}%"
                    )

        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / total_rows) * 100
            expectations.append(
                f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)"
            )
            if duplicate_pct > 10:
                issues.append(f"High duplicate rate: {duplicate_pct:.1f}%")

        # Data type analysis
        for col in data.columns:
            dtype = data[col].dtype
            expectations.append(f"Column '{col}': {dtype}")

            if dtype in ["int64", "float64"]:
                min_val = data[col].min()
                max_val = data[col].max()
                expectations.append(f"Column '{col}': range [{min_val}, {max_val}]")

        return {
            "method": "basic_validation",
            "expectations": expectations,
            "issues": issues,
            "validation_success": len(issues) == 0,
        }

    def create_data_pipeline(
        self,
        pipeline_name: str,
        data_source: Union[pd.DataFrame, str],
        transformations: List[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Create a data processing pipeline.

        Args:
            pipeline_name: Name for the pipeline
            data_source: Data source (DataFrame or file path)
            transformations: List of transformation functions

        Returns:
            Dictionary with pipeline results
        """
        if transformations is None:
            transformations = [
                self._clean_missing_values,
                self._remove_duplicates,
                self._standardize_column_names,
            ]

        pipeline_result = {
            "pipeline_name": pipeline_name,
            "start_time": datetime.now(),
            "transformations": [],
            "success": True,
            "error": None,
        }

        try:
            # Load data if path provided
            if isinstance(data_source, str):
                if data_source.endswith(".csv"):
                    data = pd.read_csv(data_source)
                elif data_source.endswith(".parquet"):
                    data = pd.read_parquet(data_source)
                else:
                    raise ValueError(f"Unsupported file format: {data_source}")
            else:
                # Handle both Pandas and Polars DataFrames
                try:
                    import polars as pl

                    if isinstance(data_source, pl.DataFrame):
                        # Convert Polars to Pandas for pipeline processing
                        data = data_source.to_pandas()
                    else:
                        data = data_source.copy()
                except ImportError:
                    # Polars not available, assume pandas DataFrame
                    data = data_source.copy()

            original_shape = data.shape

            # Apply transformations
            for i, transform_func in enumerate(transformations):
                try:
                    transform_name = getattr(
                        transform_func, "__name__", f"transformation_{i}"
                    )
                    print(f"  Applying {transform_name}...")

                    data = transform_func(data)

                    pipeline_result["transformations"].append(
                        {
                            "name": transform_name,
                            "success": True,
                            "data_shape": data.shape,
                        }
                    )

                except Exception as e:
                    pipeline_result["transformations"].append(
                        {"name": transform_name, "success": False, "error": str(e)}
                    )
                    print(f"  ⚠️ {transform_name} failed: {e}")

            pipeline_result.update(
                {
                    "end_time": datetime.now(),
                    "original_shape": original_shape,
                    "final_shape": data.shape,
                    "processed_data": data,
                }
            )

        except Exception as e:
            pipeline_result.update(
                {"success": False, "error": str(e), "end_time": datetime.now()}
            )

        # Calculate duration
        if "end_time" in pipeline_result:
            duration = pipeline_result["end_time"] - pipeline_result["start_time"]
            pipeline_result["duration_seconds"] = duration.total_seconds()

        # Store pipeline run
        self.pipeline_runs.append(pipeline_result)

        return pipeline_result

    def _clean_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean missing values in the dataset."""
        for col in data.columns:
            if data[col].dtype in ["int64", "float64"]:
                # Fill numeric columns with median
                data[col] = data[col].fillna(data[col].median())
            else:
                # Fill categorical columns with mode
                mode_value = data[col].mode()
                if len(mode_value) > 0:
                    data[col] = data[col].fillna(mode_value[0])
                else:
                    data[col] = data[col].fillna("unknown")

        return data

    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_count = len(data)
        data = data.drop_duplicates()
        removed_count = original_count - len(data)
        if removed_count > 0:
            print(f"    Removed {removed_count} duplicate rows")
        return data

    def _standardize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        new_columns = {}
        for col in data.columns:
            new_col = col.lower().replace(" ", "_").replace("-", "_")
            # Remove special characters
            new_col = "".join(char for char in new_col if char.isalnum() or char == "_")
            new_columns[col] = new_col

        data = data.rename(columns=new_columns)
        return data

    def create_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary with data quality metrics
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_overview": {},
            "column_analysis": {},
            "data_quality_score": 0,
            "recommendations": [],
        }

        # Dataset overview
        total_rows = len(data)
        total_cols = len(data.columns)
        total_cells = total_rows * total_cols
        missing_cells = data.isnull().sum().sum()

        report["dataset_overview"] = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": (
                (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            ),
            "duplicate_rows": data.duplicated().sum(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        # Column-level analysis
        quality_scores = []

        for col in data.columns:
            col_analysis = {
                "dtype": str(data[col].dtype),
                "missing_count": data[col].isnull().sum(),
                "missing_percentage": (data[col].isnull().sum() / total_rows) * 100,
                "unique_values": data[col].nunique(),
                "uniqueness_ratio": (
                    data[col].nunique() / total_rows if total_rows > 0 else 0
                ),
            }

            # Type-specific analysis
            if data[col].dtype in ["int64", "float64"]:
                col_analysis.update(
                    {
                        "min": data[col].min(),
                        "max": data[col].max(),
                        "mean": data[col].mean(),
                        "std": data[col].std(),
                        "outliers_iqr": self._count_outliers_iqr(data[col]),
                    }
                )

            # Column quality score (0-100)
            col_quality = 100
            if col_analysis["missing_percentage"] > 0:
                col_quality -= min(
                    col_analysis["missing_percentage"], 50
                )  # Max 50 point deduction
            if col_analysis["uniqueness_ratio"] == 0:
                col_quality -= 20  # No variation
            elif col_analysis["uniqueness_ratio"] == 1 and total_rows > 10:
                col_quality -= 10  # Too unique (possible ID column)

            col_analysis["quality_score"] = max(col_quality, 0)
            quality_scores.append(col_analysis["quality_score"])

            report["column_analysis"][col] = col_analysis

        # Overall quality score
        report["data_quality_score"] = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0
        )

        # Generate recommendations
        report["recommendations"] = self._generate_quality_recommendations(report)

        return report

    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        if series.dtype not in ["int64", "float64"]:
            return 0

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        return outliers

    def _generate_quality_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []

        # Missing data recommendations
        if report["dataset_overview"]["missing_percentage"] > 5:
            recommendations.append(
                "🔍 High missing data detected - consider imputation strategies"
            )

        # Duplicate recommendations
        if report["dataset_overview"]["duplicate_rows"] > 0:
            recommendations.append("🔄 Remove duplicate rows to improve data quality")

        # Column-specific recommendations
        for col, analysis in report["column_analysis"].items():
            if analysis["missing_percentage"] > 20:
                recommendations.append(
                    f"📊 Column '{col}': High missing rate ({analysis['missing_percentage']:.1f}%)"
                )

            if (
                analysis["uniqueness_ratio"] == 1
                and len(report["column_analysis"]) > 10
            ):
                recommendations.append(
                    f"🔑 Column '{col}': Appears to be an identifier - consider removing for modeling"
                )

            if analysis["uniqueness_ratio"] == 0:
                recommendations.append(
                    f"📈 Column '{col}': No variation detected - consider removing"
                )

        # Memory optimization
        if report["dataset_overview"]["memory_usage_mb"] > 100:
            recommendations.append("💾 Large dataset - consider data type optimization")

        if not recommendations:
            recommendations.append(
                "✅ Data quality looks good - no major issues detected"
            )

        return recommendations

    def get_tool_comparison(self) -> Dict[str, Any]:
        """Get comparison of available data engineering tools."""
        comparison = {
            "great_expectations": {
                "status": (
                    "✅ Available"
                    if self.great_expectations_available
                    else "❌ Not installed"
                ),
                "install_cmd": "pip install great-expectations",
                "strengths": [
                    "🎯 Comprehensive data validation framework",
                    "📊 Rich expectation library",
                    "🔍 Data profiling and documentation",
                    "📈 Integration with data catalogs",
                    "🚀 Production-ready data testing",
                ],
                "use_cases": [
                    "Data validation in production pipelines",
                    "Data quality monitoring",
                    "Automated data testing",
                    "Data documentation and profiling",
                ],
            },
            "pandera": {
                "status": (
                    "✅ Available" if self.pandera_available else "❌ Not installed"
                ),
                "install_cmd": "pip install pandera",
                "strengths": [
                    "🐼 Pandas-native data validation",
                    "🔧 Type hints for data schemas",
                    "⚡ Lightweight and fast",
                    "🧪 Statistical hypothesis testing",
                    "🎯 Schema inference",
                ],
                "use_cases": [
                    "Pandas DataFrame validation",
                    "Data contract enforcement",
                    "Statistical data testing",
                    "Schema documentation",
                ],
            },
            "prefect": {
                "status": (
                    "✅ Available" if self.prefect_available else "❌ Not installed"
                ),
                "install_cmd": "pip install prefect",
                "strengths": [
                    "🌊 Modern workflow orchestration",
                    "📊 Beautiful UI and monitoring",
                    "🔄 Dynamic workflows",
                    "☁️ Cloud and on-premise options",
                    "🎯 Python-first design",
                ],
                "use_cases": [
                    "Data pipeline orchestration",
                    "ML workflow automation",
                    "Scheduled data processing",
                    "Complex workflow management",
                ],
            },
        }

        return comparison

    def export_pipeline_history(self, output_file: str = "pipeline_history.json"):
        """Export pipeline run history to JSON."""
        with open(output_file, "w") as f:
            json.dump(
                {
                    "pipeline_runs": self.pipeline_runs,
                    "validation_results": self.validation_results,
                    "export_timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=str,
            )

        print(f"📁 Pipeline history exported to {output_file}")
        return output_file
