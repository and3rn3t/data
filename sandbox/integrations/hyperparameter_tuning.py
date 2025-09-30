"""
Hyperparameter Tuning Integration

Provides integration with Optuna, Hyperopt, and other hyperparameter optimization
libraries for automated machine learning model tuning.
"""

import json
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np


class HyperparameterOptimizer:
    """
    Integration class for hyperparameter optimization tools.

    Supports Optuna, Hyperopt, and grid search with educational examples
    and visualization of optimization results.
    """

    def __init__(self):
        """Initialize the hyperparameter optimizer with available libraries."""
        self.optuna_available = self._check_optuna()
        self.hyperopt_available = self._check_hyperopt()
        self.skopt_available = self._check_skopt()

        self.optimization_history = []

    def _check_optuna(self) -> bool:
        """Check if Optuna is available."""
        try:
            import optuna

            self.optuna = optuna
            # Suppress Optuna logging
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            return True
        except ImportError:
            return False

    def _check_hyperopt(self) -> bool:
        """Check if Hyperopt is available."""
        try:
            import hyperopt

            self.hyperopt = hyperopt
            return True
        except ImportError:
            return False

    def _check_skopt(self) -> bool:
        """Check if Scikit-Optimize is available."""
        try:
            import skopt

            self.skopt = skopt
            return True
        except ImportError:
            return False

    def optimize_model(
        self,
        objective_function: Callable,
        param_space: Dict[str, Any],
        n_trials: int = 100,
        method: str = "auto",
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using the specified method.

        Args:
            objective_function: Function to minimize (should return a score)
            param_space: Parameter space definition
            n_trials: Number of optimization trials
            method: Optimization method ("optuna", "hyperopt", "grid", "auto")
            study_name: Optional name for the optimization study

        Returns:
            Dictionary with optimization results
        """
        if study_name is None:
            study_name = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Auto-select method based on availability
        if method == "auto":
            if self.optuna_available:
                method = "optuna"
            elif self.hyperopt_available:
                method = "hyperopt"
            elif self.skopt_available:
                method = "skopt"
            else:
                method = "grid"

        # Perform optimization
        if method == "optuna" and self.optuna_available:
            results = self._optimize_with_optuna(
                objective_function, param_space, n_trials, study_name
            )
        elif method == "hyperopt" and self.hyperopt_available:
            results = self._optimize_with_hyperopt(
                objective_function, param_space, n_trials, study_name
            )
        elif method == "skopt" and self.skopt_available:
            results = self._optimize_with_skopt(
                objective_function, param_space, n_trials, study_name
            )
        else:
            results = self._optimize_with_grid_search(
                objective_function, param_space, study_name
            )

        # Store optimization history
        self.optimization_history.append(
            {
                "study_name": study_name,
                "method": method,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return results

    def _optimize_with_optuna(
        self, objective_function, param_space, n_trials, study_name
    ):
        """Optimize using Optuna."""
        try:
            # Create study
            study = self.optuna.create_study(
                direction="minimize",
                study_name=study_name,
                sampler=self.optuna.samplers.TPESampler(seed=42),
            )

            # Define objective wrapper
            def optuna_objective(trial):
                params = {}
                for param_name, param_config in param_space.items():
                    if param_config["type"] == "float":
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            log=param_config.get("log", False),
                        )
                    elif param_config["type"] == "int":
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            log=param_config.get("log", False),
                        )
                    elif param_config["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config["choices"]
                        )

                return objective_function(params)

            # Optimize
            study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=True)

            # Create visualizations if available
            visualizations = {}
            try:
                import matplotlib.pyplot as plt
                import optuna.visualization as vis

                # Optimization history
                fig = vis.matplotlib.plot_optimization_history(study)
                plt.title("Optuna Optimization History")
                visualizations["optimization_history"] = True

                # Parameter importance
                if len(study.trials) > 10:
                    fig = vis.matplotlib.plot_param_importances(study)
                    plt.title("Parameter Importance")
                    visualizations["param_importance"] = True

                # Parallel coordinate plot
                if len(param_space) > 1:
                    fig = vis.matplotlib.plot_parallel_coordinate(study)
                    plt.title("Parameter Relationships")
                    visualizations["parallel_coordinate"] = True

            except ImportError:
                print("Optuna visualization requires additional packages")
            except Exception as e:
                print(f"Visualization failed: {e}")

            return {
                "method": "optuna",
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "study": study,
                "visualizations": visualizations,
                "trials_dataframe": (
                    study.trials_dataframe().to_dict("records")
                    if hasattr(study, "trials_dataframe")
                    else []
                ),
            }

        except Exception as e:
            print(f"Optuna optimization failed: {e}")
            return self._optimize_with_grid_search(
                objective_function, param_space, study_name
            )

    def _optimize_with_hyperopt(
        self, objective_function, param_space, n_trials, study_name
    ):
        """Optimize using Hyperopt."""
        try:
            from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

            # Convert parameter space
            hp_space = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "float":
                    if param_config.get("log", False):
                        hp_space[param_name] = hp.lognormal(
                            param_name,
                            np.log((param_config["low"] + param_config["high"]) / 2),
                            1,
                        )
                    else:
                        hp_space[param_name] = hp.uniform(
                            param_name, param_config["low"], param_config["high"]
                        )
                elif param_config["type"] == "int":
                    hp_space[param_name] = hp.choice(
                        param_name,
                        list(range(param_config["low"], param_config["high"] + 1)),
                    )
                elif param_config["type"] == "categorical":
                    hp_space[param_name] = hp.choice(
                        param_name, param_config["choices"]
                    )

            # Define objective wrapper
            def hyperopt_objective(params):
                score = objective_function(params)
                return {"loss": score, "status": STATUS_OK}

            # Optimize
            trials = Trials()
            best = fmin(
                fn=hyperopt_objective,
                space=hp_space,
                algo=tpe.suggest,
                max_evals=n_trials,
                trials=trials,
                rstate=np.random.RandomState(42),
            )

            return {
                "method": "hyperopt",
                "best_params": best,
                "best_value": min([trial["result"]["loss"] for trial in trials.trials]),
                "n_trials": len(trials.trials),
                "trials": trials,
                "trials_list": [trial["result"]["loss"] for trial in trials.trials],
            }

        except Exception as e:
            print(f"Hyperopt optimization failed: {e}")
            return self._optimize_with_grid_search(
                objective_function, param_space, study_name
            )

    def _optimize_with_skopt(
        self, objective_function, param_space, n_trials, study_name
    ):
        """Optimize using Scikit-Optimize."""
        try:
            from skopt import gp_minimize
            from skopt.space import Categorical, Integer, Real
            from skopt.utils import use_named_args

            # Convert parameter space
            dimensions = []
            param_names = []

            for param_name, param_config in param_space.items():
                param_names.append(param_name)
                if param_config["type"] == "float":
                    dimensions.append(
                        Real(
                            param_config["low"],
                            param_config["high"],
                            prior=(
                                "log-uniform"
                                if param_config.get("log", False)
                                else "uniform"
                            ),
                            name=param_name,
                        )
                    )
                elif param_config["type"] == "int":
                    dimensions.append(
                        Integer(
                            param_config["low"], param_config["high"], name=param_name
                        )
                    )
                elif param_config["type"] == "categorical":
                    dimensions.append(
                        Categorical(param_config["choices"], name=param_name)
                    )

            # Define objective wrapper
            @use_named_args(dimensions)
            def skopt_objective(**params):
                return objective_function(params)

            # Optimize
            result = gp_minimize(
                func=skopt_objective,
                dimensions=dimensions,
                n_calls=n_trials,
                random_state=42,
                acq_func="EI",  # Expected Improvement
            )

            # Extract best parameters
            best_params = {}
            for i, param_name in enumerate(param_names):
                best_params[param_name] = result.x[i]

            return {
                "method": "scikit-optimize",
                "best_params": best_params,
                "best_value": result.fun,
                "n_trials": len(result.func_vals),
                "result": result,
                "convergence": result.func_vals,
            }

        except Exception as e:
            print(f"Scikit-Optimize optimization failed: {e}")
            return self._optimize_with_grid_search(
                objective_function, param_space, study_name
            )

    def _optimize_with_grid_search(self, objective_function, param_space, study_name):
        """Fallback grid search optimization."""
        try:
            from sklearn.model_selection import ParameterGrid

            # Convert parameter space for grid search
            grid_space = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "float":
                    grid_space[param_name] = np.linspace(
                        param_config["low"],
                        param_config["high"],
                        5,  # 5 points for float parameters
                    ).tolist()
                elif param_config["type"] == "int":
                    grid_space[param_name] = list(
                        range(
                            param_config["low"],
                            min(
                                param_config["high"] + 1, param_config["low"] + 5
                            ),  # Max 5 points
                        )
                    )
                elif param_config["type"] == "categorical":
                    grid_space[param_name] = param_config["choices"]

            # Generate parameter grid
            param_grid = list(ParameterGrid(grid_space))

            # Evaluate all combinations
            results = []
            best_score = float("inf")
            best_params = None

            for params in param_grid[:50]:  # Limit to 50 combinations
                score = objective_function(params)
                results.append({"params": params, "score": score})

                if score < best_score:
                    best_score = score
                    best_params = params

            return {
                "method": "grid_search",
                "best_params": best_params,
                "best_value": best_score,
                "n_trials": len(results),
                "all_results": results,
            }

        except Exception as e:
            print(f"Grid search optimization failed: {e}")
            return {
                "method": "failed",
                "error": str(e),
                "best_params": {},
                "best_value": None,
                "n_trials": 0,
            }

    def create_optimization_examples(self) -> Dict[str, Any]:
        """
        Create educational examples of hyperparameter optimization.

        Returns:
            Dictionary with examples and explanations
        """
        examples = {
            "parameter_space_examples": {
                "random_forest": {
                    "n_estimators": {"type": "int", "low": 50, "high": 500},
                    "max_depth": {"type": "int", "low": 3, "high": 20},
                    "min_samples_split": {"type": "int", "low": 2, "high": 20},
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
                },
                "xgboost": {
                    "n_estimators": {"type": "int", "low": 100, "high": 1000},
                    "max_depth": {"type": "int", "low": 3, "high": 10},
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.3,
                        "log": True,
                    },
                    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                },
                "neural_network": {
                    "learning_rate": {
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-1,
                        "log": True,
                    },
                    "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                    "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5},
                    "optimizer": {
                        "type": "categorical",
                        "choices": ["adam", "sgd", "rmsprop"],
                    },
                },
            },
            "objective_function_example": '''
def objective_function(params):
    """
    Example objective function for hyperparameter optimization.
    Should return a value to minimize (e.g., validation error).
    """
    model = RandomForestClassifier(**params, random_state=42)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Return negative accuracy (since we want to minimize)
    return -cv_scores.mean()
''',
            "usage_example": """
# Initialize optimizer
optimizer = HyperparameterOptimizer()

# Define parameter space
param_space = {
    "n_estimators": {"type": "int", "low": 50, "high": 500},
    "max_depth": {"type": "int", "low": 3, "high": 20}
}

# Run optimization
results = optimizer.optimize_model(
    objective_function=my_objective,
    param_space=param_space,
    n_trials=100,
    method="optuna"
)

print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_value']}")
""",
        }

        return examples

    def get_tool_comparison(self) -> Dict[str, Any]:
        """Get comparison of available optimization tools."""
        comparison = {
            "optuna": {
                "status": (
                    "✅ Available" if self.optuna_available else "❌ Not installed"
                ),
                "install_cmd": "pip install optuna",
                "strengths": [
                    "🎯 State-of-the-art optimization algorithms",
                    "📊 Excellent visualization tools",
                    "🔄 Supports pruning of unpromising trials",
                    "📈 Built-in study management",
                    "🤖 Easy integration with ML frameworks",
                ],
                "algorithms": ["TPE", "CMA-ES", "Random", "Grid"],
                "use_cases": [
                    "Deep learning hyperparameter tuning",
                    "AutoML pipeline optimization",
                    "Multi-objective optimization",
                    "Production ML system tuning",
                ],
            },
            "hyperopt": {
                "status": (
                    "✅ Available" if self.hyperopt_available else "❌ Not installed"
                ),
                "install_cmd": "pip install hyperopt",
                "strengths": [
                    "🧠 Bayesian optimization focus",
                    "🔧 Flexible parameter space definition",
                    "📊 Tree-structured Parzen Estimator (TPE)",
                    "⚡ Good for complex search spaces",
                ],
                "algorithms": ["TPE", "Random", "Adaptive TPE"],
                "use_cases": [
                    "Complex nested parameter spaces",
                    "Bayesian hyperparameter optimization",
                    "Research and experimentation",
                    "Custom optimization workflows",
                ],
            },
            "scikit_optimize": {
                "status": (
                    "✅ Available" if self.skopt_available else "❌ Not installed"
                ),
                "install_cmd": "pip install scikit-optimize",
                "strengths": [
                    "🎯 Gaussian Process based optimization",
                    "📊 Strong theoretical foundation",
                    "🔧 Scikit-learn integration",
                    "📈 Acquisition function variety",
                ],
                "algorithms": ["GP", "RF", "ET", "GBRT"],
                "use_cases": [
                    "Small to medium search spaces",
                    "Gaussian Process modeling",
                    "Sequential model-based optimization",
                    "Academic research",
                ],
            },
        }

        return comparison

    def export_optimization_history(
        self, output_file: str = "optimization_history.json"
    ):
        """
        Export optimization history to a JSON file.

        Args:
            output_file: Path to save the history
        """
        with open(output_file, "w") as f:
            json.dump(self.optimization_history, f, indent=2, default=str)

        print(f"📁 Optimization history exported to {output_file}")
        return output_file
