"""
Model Explainability Integration

Provides integration with SHAP, LIME, and other model explanation tools
for understanding machine learning model predictions.
"""

from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ModelExplainer:
    """
    Integration class for model explainability tools.

    Supports SHAP, LIME, and other explanation methods with educational
    examples and visualizations.
    """

    def __init__(self):
        """Initialize the model explainer with available libraries."""
        self.shap_available = self._check_shap()
        self.lime_available = self._check_lime()
        self.yellowbrick_available = self._check_yellowbrick()

    def _check_shap(self) -> bool:
        """Check if SHAP is available."""
        try:
            import shap

            self.shap = shap
            return True
        except ImportError:
            return False

    def _check_lime(self) -> bool:
        """Check if LIME is available."""
        try:
            import lime
            import lime.lime_tabular

            self.lime = lime
            return True
        except ImportError:
            return False

    def _check_yellowbrick(self) -> bool:
        """Check if Yellowbrick is available."""
        try:
            import yellowbrick
            from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
            from yellowbrick.regressor import PredictionError, ResidualsPlot

            self.yellowbrick = yellowbrick
            self.ConfusionMatrix = ConfusionMatrix
            self.ClassificationReport = ClassificationReport
            self.ResidualsPlot = ResidualsPlot
            self.PredictionError = PredictionError
            return True
        except ImportError:
            return False

    def explain_prediction(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_explain: Union[pd.DataFrame, np.ndarray],
        method: str = "auto",
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Explain model predictions using available explainability tools.

        Args:
            model: Trained model with predict method
            X_train: Training data for context
            X_explain: Data points to explain
            method: Explanation method ("shap", "lime", "auto")
            feature_names: Optional feature names

        Returns:
            Dictionary with explanations and visualizations
        """
        results = {"method_used": method, "explanations": {}}

        # Determine feature names
        if feature_names is None:
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_train_np = X_train.values
        else:
            X_train_np = X_train

        if isinstance(X_explain, pd.DataFrame):
            X_explain_np = X_explain.values
        else:
            X_explain_np = X_explain

        # Auto-select method based on availability
        if method == "auto":
            if self.shap_available:
                method = "shap"
            elif self.lime_available:
                method = "lime"
            else:
                method = "simple"

        # SHAP explanations
        if method == "shap" and self.shap_available:
            results.update(
                self._explain_with_shap(model, X_train_np, X_explain_np, feature_names)
            )

        # LIME explanations
        elif method == "lime" and self.lime_available:
            results.update(
                self._explain_with_lime(model, X_train_np, X_explain_np, feature_names)
            )

        # Simple feature importance fallback
        else:
            results.update(
                self._explain_simple(model, X_train_np, X_explain_np, feature_names)
            )

        return results

    def _explain_with_shap(self, model, X_train, X_explain, feature_names):
        """Explain using SHAP."""
        try:
            # Create SHAP explainer
            explainer = self.shap.Explainer(model.predict, X_train)
            shap_values = explainer(X_explain)

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Summary plot
            plt.sca(axes[0, 0])
            self.shap.plots.bar(shap_values, show=False)
            axes[0, 0].set_title("SHAP Feature Importance")

            # Waterfall plot for first instance
            if len(X_explain) > 0:
                plt.sca(axes[0, 1])
                self.shap.plots.waterfall(shap_values[0], show=False)
                axes[0, 1].set_title("SHAP Waterfall (First Instance)")

            # Beeswarm plot
            if len(X_explain) > 1:
                plt.sca(axes[1, 0])
                self.shap.plots.beeswarm(shap_values, show=False)
                axes[1, 0].set_title("SHAP Beeswarm Plot")

            # Feature importance
            plt.sca(axes[1, 1])
            feature_importance = np.abs(shap_values.values).mean(0)
            plt.barh(feature_names, feature_importance)
            plt.title("Mean |SHAP Values|")
            plt.xlabel("Mean |SHAP Value|")

            plt.tight_layout()

            return {
                "method_used": "shap",
                "shap_values": shap_values.values,
                "feature_importance": dict(zip(feature_names, feature_importance)),
                "explanations": {
                    "summary": "SHAP values show how each feature contributes to predictions",
                    "interpretation": self._interpret_shap_values(
                        shap_values, feature_names
                    ),
                },
                "plot_created": True,
            }

        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._explain_simple(model, X_train, X_explain, feature_names)

    def _explain_with_lime(self, model, X_train, X_explain, feature_names):
        """Explain using LIME."""
        try:
            # Create LIME explainer
            explainer = self.lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                mode=(
                    "classification"
                    if hasattr(model, "predict_proba")
                    else "regression"
                ),
            )

            # Explain first instance
            instance_idx = 0
            explanation = explainer.explain_instance(
                X_explain[instance_idx],
                (
                    model.predict_proba
                    if hasattr(model, "predict_proba")
                    else model.predict
                ),
                num_features=min(10, len(feature_names)),
            )

            # Create visualization
            fig = explanation.as_pyplot_figure()
            fig.suptitle(f"LIME Explanation for Instance {instance_idx}")

            # Extract feature importance
            feature_importance = dict(explanation.as_list())

            return {
                "method_used": "lime",
                "lime_explanation": explanation,
                "feature_importance": feature_importance,
                "explanations": {
                    "summary": "LIME shows local explanations around individual predictions",
                    "interpretation": self._interpret_lime_explanation(explanation),
                },
                "plot_created": True,
            }

        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return self._explain_simple(model, X_train, X_explain, feature_names)

    def _explain_simple(self, model, X_train, X_explain, feature_names):
        """Simple explanation using permutation importance."""
        try:
            from sklearn.inspection import permutation_importance

            # Get baseline predictions
            baseline_preds = model.predict(X_explain)

            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_train, model.predict(X_train), n_repeats=5, random_state=42
            )

            feature_importance = dict(
                zip(feature_names, perm_importance.importances_mean)
            )

            # Create simple plot
            plt.figure(figsize=(10, 6))
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )
            features, importance = zip(*sorted_features[:10])  # Top 10 features

            plt.barh(features, importance)
            plt.title("Permutation Feature Importance")
            plt.xlabel("Importance Score")

            return {
                "method_used": "permutation_importance",
                "feature_importance": feature_importance,
                "baseline_predictions": baseline_preds,
                "explanations": {
                    "summary": "Permutation importance shows how much performance drops when features are shuffled",
                    "interpretation": f"Top features: {', '.join(features[:3])}",
                },
                "plot_created": True,
            }

        except Exception as e:
            print(f"Simple explanation failed: {e}")
            return {
                "method_used": "failed",
                "error": str(e),
                "feature_importance": {},
                "explanations": {
                    "summary": "Explanation failed - model may not be compatible"
                },
            }

    def _interpret_shap_values(self, shap_values, feature_names):
        """Provide interpretation of SHAP values."""
        mean_importance = np.abs(shap_values.values).mean(0)
        top_features = np.argsort(mean_importance)[-3:][::-1]

        interpretation = f"Top 3 most important features: {', '.join([feature_names[i] for i in top_features])}"
        return interpretation

    def _interpret_lime_explanation(self, explanation):
        """Provide interpretation of LIME explanation."""
        feature_scores = explanation.as_list()
        top_positive = [f for f, score in feature_scores if score > 0][:2]
        top_negative = [f for f, score in feature_scores if score < 0][:2]

        interpretation = []
        if top_positive:
            interpretation.append(
                f"Features supporting prediction: {', '.join(top_positive)}"
            )
        if top_negative:
            interpretation.append(
                f"Features opposing prediction: {', '.join(top_negative)}"
            )

        return "; ".join(interpretation)

    def create_model_evaluation_report(
        self,
        model: Any,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        task_type: str = "auto",
    ) -> Dict[str, Any]:
        """
        Create comprehensive model evaluation report with visualizations.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            task_type: "classification", "regression", or "auto"

        Returns:
            Dictionary with evaluation results and visualizations
        """
        results = {"task_type": task_type}

        # Auto-detect task type
        if task_type == "auto":
            if hasattr(model, "predict_proba"):
                task_type = "classification"
            else:
                # Check if target is continuous or discrete
                unique_values = len(np.unique(y_test))
                if unique_values > 10:
                    task_type = "regression"
                else:
                    task_type = "classification"
            results["task_type"] = task_type

        # Get predictions
        y_pred = model.predict(X_test)

        if task_type == "classification":
            results.update(
                self._create_classification_report(model, X_test, y_test, y_pred)
            )
        else:
            results.update(
                self._create_regression_report(model, X_test, y_test, y_pred)
            )

        return results

    def _create_classification_report(self, model, X_test, y_test, y_pred):
        """Create classification evaluation report."""
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )

        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # Create visualizations with Yellowbrick if available
        if self.yellowbrick_available:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Confusion Matrix
                cm = self.ConfusionMatrix(model, ax=axes[0])
                cm.fit(X_test, y_test)
                cm.score(X_test, y_test)

                # Classification Report
                cr = self.ClassificationReport(model, ax=axes[1])
                cr.fit(X_test, y_test)
                cr.score(X_test, y_test)

                plt.tight_layout()
                results["yellowbrick_plots"] = True

            except Exception as e:
                print(f"Yellowbrick visualization failed: {e}")

        return results

    def _create_regression_report(self, model, X_test, y_test, y_pred):
        """Create regression evaluation report."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        results = {
            "r2_score": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
        }

        # Create visualizations with Yellowbrick if available
        if self.yellowbrick_available:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Residuals Plot
                residuals = self.ResidualsPlot(model, ax=axes[0])
                residuals.fit(X_test, y_test)
                residuals.score(X_test, y_test)

                # Prediction Error
                pred_error = self.PredictionError(model, ax=axes[1])
                pred_error.fit(X_test, y_test)
                pred_error.score(X_test, y_test)

                plt.tight_layout()
                results["yellowbrick_plots"] = True

            except Exception as e:
                print(f"Yellowbrick visualization failed: {e}")

        return results

    def get_tool_comparison(self) -> Dict[str, Any]:
        """Get comparison of available explainability tools."""
        comparison = {
            "shap": {
                "status": "✅ Available" if self.shap_available else "❌ Not installed",
                "install_cmd": "pip install shap",
                "strengths": [
                    "🎯 Unified framework for model explanations",
                    "📊 Game-theoretic foundation (Shapley values)",
                    "🔍 Both global and local explanations",
                    "📈 Great visualizations",
                    "🤖 Works with many ML frameworks",
                ],
                "use_cases": [
                    "Understanding feature contributions",
                    "Model debugging and validation",
                    "Regulatory compliance and fairness",
                    "Feature selection insights",
                ],
            },
            "lime": {
                "status": "✅ Available" if self.lime_available else "❌ Not installed",
                "install_cmd": "pip install lime",
                "strengths": [
                    "🎯 Local interpretable explanations",
                    "🔍 Model-agnostic approach",
                    "📱 Works with text, images, and tabular data",
                    "💡 Intuitive perturbation-based explanations",
                ],
                "use_cases": [
                    "Explaining individual predictions",
                    "Understanding decision boundaries",
                    "Building trust in model predictions",
                    "Debugging specific instances",
                ],
            },
            "yellowbrick": {
                "status": (
                    "✅ Available" if self.yellowbrick_available else "❌ Not installed"
                ),
                "install_cmd": "pip install yellowbrick",
                "strengths": [
                    "📊 Beautiful ML visualizations",
                    "🛠️ Scikit-learn integration",
                    "📈 Model evaluation and diagnostics",
                    "🎨 Professional-quality plots",
                ],
                "use_cases": [
                    "Model evaluation and comparison",
                    "Feature visualization and selection",
                    "Hyperparameter tuning guidance",
                    "Publication-ready visualizations",
                ],
            },
        }

        return comparison
