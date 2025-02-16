# praice/machine_learning/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator


class BaseMLModel(ABC):
    """Abstract base class for all ML models in the system."""

    def __init__(self):
        self.model: Optional[BaseEstimator] = None
        self.feature_columns: Optional[list] = None
        self.target_columns: Optional[list] = None

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and targets from input DataFrame."""
        pass

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions."""
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseMLModel":
        """Load model from disk."""
        pass


class BaseAutoMLModel(BaseMLModel):
    """Base class for AutoML models."""

    @abstractmethod
    def get_best_pipeline(self) -> Dict[str, Any]:
        """Get the best pipeline found during training."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        pass
