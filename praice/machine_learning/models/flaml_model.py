from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from praice.machine_learning.base import BaseAutoMLModel
from praice.machine_learning.data.preprocessing import (
    FeaturePreprocessor,
    FeatureSelector,
)


class FLAMLModel(BaseAutoMLModel):
    """
    FLAML AutoML model implementation for financial time series prediction.
    """

    def __init__(
        self,
        config: Optional[Dict[str, any]] = None,
        preprocessor: Optional[FeaturePreprocessor] = None,
        feature_selector: Optional[FeatureSelector] = None,
    ):
        """
        Initialize the FLAML model.

        Args:
            config: Configuration dictionary containing FLAML parameters
            preprocessor: Optional custom feature preprocessor
            feature_selector: Optional custom feature selector
        """
        super().__init__()

        # Initialize FLAML AutoML
        self.automl = AutoML(**config) if config else AutoML()

        # Set up data preprocessing
        self.preprocessor = preprocessor or FeaturePreprocessor()
        self.feature_selector = feature_selector or FeatureSelector()

        # Store training metadata
        self.models = {}
        self.feature_importance = {}
        self.best_pipeline = {}
        self.training_summary = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and targets from input DataFrame.

        Args:
            df: Input DataFrame containing features and targets

        Returns:
            Tuple of (X, y) containing processed features and targets
        """
        # Preprocess features and get targets
        X, y = self.preprocessor.fit_transform(df)

        # Select features
        X = self.feature_selector.fit_transform(X)

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        task: str = "regression",
        time_budget: int = 3600,
        metric: str = "rmse",
        estimator_list: Optional[List[str]] = None,
        ensemble: bool = True,
        **kwargs,
    ) -> None:
        """
        Train the FLAML model.

        Args:
            X: Feature DataFrame
            y: Target DataFrame
            task: Type of ML task ('regression' or 'classification')
            time_budget: Time budget in seconds
            metric: Metric to optimize
            estimator_list: List of estimators to try
            ensemble: Whether to use ensemble learning
            **kwargs: Additional arguments passed to FLAML
        """
        # Set default estimators if not provided
        if estimator_list is None:
            estimator_list = ["xgboost", "rf", "lgbm", "extra_tree", "catboost"]

        # Configure FLAML
        automl_settings = {
            "time_budget": time_budget,
            "metric": metric,
            "task": task,
            "estimator_list": estimator_list,
            "ensemble": ensemble,
            **kwargs,
        }

        # Train models for each target
        self.models = {}
        self.feature_importance = {}
        self.best_pipeline = {}
        self.training_summary = {}

        for target in y.columns:
            # Initialize new AutoML instance for each target
            self.models[target] = AutoML()

            # Fit the model
            self.models[target].fit(X_train=X, y_train=y[target], **automl_settings)

            # Store results
            self.best_pipeline[target] = self.models[target].best_config
            self.training_summary[target] = {
                "best_estimator": self.models[target].best_estimator,
                "best_loss": self.models[target].best_loss,
                "best_iteration": self.models[target].best_iteration,
                # "time_to_best": self.models[target].time_to_best,
            }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame containing predictions for all targets
        """
        if not self.models:
            raise ValueError("Model must be trained before making predictions.")

        # Process features
        X = self.preprocessor.transform(X)
        X = self.feature_selector.transform(X)

        # Make predictions for each target
        predictions = {}
        for target, model in self.models.items():
            predictions[target] = model.predict(X)

        return pd.DataFrame(predictions, index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance.

        Args:
            X: Feature DataFrame
            y: Target DataFrame

        Returns:
            Dictionary containing evaluation metrics for each target
        """
        predictions = self.predict(X)

        # Calculate metrics for each target
        metrics = {}
        for target in y.columns:
            y_true = y[target]
            y_pred = predictions[target]

            metrics[target] = {
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2": r2_score(y_true, y_pred),
                "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            }

        return metrics

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance scores for each target.

        Returns:
            Dictionary mapping target names to feature importance DataFrames
        """
        if not self.feature_importance:
            raise ValueError("Model must be trained before getting feature importance.")

        return self.feature_importance

    def get_best_pipeline(self) -> Dict[str, Dict]:
        """
        Get the best pipeline configuration for each target.

        Returns:
            Dictionary mapping target names to pipeline configurations
        """
        if not self.best_pipeline:
            raise ValueError(
                "Model must be trained before getting pipeline information."
            )

        return self.best_pipeline

    def get_training_summary(self) -> Dict[str, Dict]:
        """
        Get training summary information for each target.

        Returns:
            Dictionary containing training summaries for each target
        """
        if not self.training_summary:
            raise ValueError("Model must be trained before getting training summary.")

        return self.training_summary

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        model_data = {
            "models": self.models,
            "preprocessor": self.preprocessor,
            "feature_selector": self.feature_selector,
            "feature_importance": self.feature_importance,
            "best_pipeline": self.best_pipeline,
            "training_summary": self.training_summary,
            "config": self.config,
        }

        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: str) -> "FLAMLModel":
        """
        Load a model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded FLAMLModel instance
        """
        model_data = joblib.load(path)

        # Create new instance
        instance = cls(config=model_data["config"])

        # Restore state
        instance.models = model_data["models"]
        instance.preprocessor = model_data["preprocessor"]
        instance.feature_selector = model_data["feature_selector"]
        instance.feature_importance = model_data["feature_importance"]
        instance.best_pipeline = model_data["best_pipeline"]
        instance.training_summary = model_data["training_summary"]

        return instance
