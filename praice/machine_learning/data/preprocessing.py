from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler


class FeaturePreprocessor:
    """
    A class for preprocessing financial time series data.

    This preprocessor handles:
    - Feature selection
    - Missing value imputation
    - Outlier treatment
    - Feature scaling
    - Technical indicator processing
    """

    def __init__(
        self,
        target_columns: List[str] = None,
        scale_features: bool = True,
        handle_outliers: bool = True,
        handle_missing: bool = True,
        scaler_type: str = "robust",
        missing_strategy: str = "median",
    ):
        """
        Initialize the preprocessor.

        Args:
            target_columns: List of column names to be used as targets
            scale_features: Whether to scale features
            handle_outliers: Whether to handle outliers
            handle_missing: Whether to handle missing values
            scaler_type: Type of scaler to use ('robust' or 'standard')
            missing_strategy: Strategy for handling missing values ('mean', 'median', 'most_frequent')
        """
        self.target_columns = target_columns or ["future_close_21d", "future_close_63d"]
        self.scale_features = scale_features
        self.handle_outliers = handle_outliers
        self.handle_missing = handle_missing
        self.scaler_type = scaler_type
        self.missing_strategy = missing_strategy

        # Initialize transformers
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.imputer = SimpleImputer(strategy=missing_strategy)

        # Store column information
        self.feature_columns = None
        self.technical_columns = None
        self.candlestick_columns = None
        self.return_columns = None
        self.price_columns = None

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """Identify different types of columns in the dataset."""
        self.technical_columns = [
            col
            for col in df.columns
            if not col.startswith(
                ("CDL", "simple_return", "log_return", "previous_close", "future_close")
            )
        ]
        self.candlestick_columns = [col for col in df.columns if col.startswith("CDL")]
        self.return_columns = [
            col for col in df.columns if col.startswith(("simple_return", "log_return"))
        ]
        self.price_columns = [
            col for col in df.columns if col.startswith("previous_close")
        ]

        # All features except targets
        self.feature_columns = [
            col for col in df.columns if col not in self.target_columns
        ]

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if not self.handle_missing:
            return df

        # For technical indicators, use imputer
        technical_df = df[self.technical_columns]
        if technical_df.isnull().any().any():
            technical_df = pd.DataFrame(
                self.imputer.fit_transform(technical_df),
                columns=technical_df.columns,
                index=technical_df.index,
            )

        # For other columns, forward fill then backward fill
        other_cols = [col for col in df.columns if col not in self.technical_columns]
        other_df = df[other_cols].fillna(method="ffill").fillna(method="bfill")

        return pd.concat([technical_df, other_df], axis=1)

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using winsorization."""
        if not self.handle_outliers:
            return df

        def winsorize(series, limits=(0.05, 0.05)):
            lower = series.quantile(limits[0])
            upper = series.quantile(1 - limits[1])
            return series.clip(lower=lower, upper=upper)

        # Winsorize technical indicators and returns
        cols_to_winsorize = self.technical_columns + self.return_columns
        df_clean = df.copy()

        for col in cols_to_winsorize:
            df_clean[col] = winsorize(df[col])

        return df_clean

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using the specified scaler."""
        if not self.scale_features:
            return df

        # Scale technical indicators and returns
        cols_to_scale = self.technical_columns + self.return_columns
        scaled_data = self.scaler.fit_transform(df[cols_to_scale])

        df_scaled = df.copy()
        df_scaled[cols_to_scale] = scaled_data

        return df_scaled

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the preprocessor and transform the data.

        Args:
            df: Input DataFrame containing features and targets

        Returns:
            Tuple of (X, y) where X contains features and y contains targets
        """
        # Identify column types
        self._identify_column_types(df)

        # Split features and targets
        X = df[self.feature_columns].copy()
        y = df[self.target_columns].copy()

        # Process features
        X = self._handle_missing_values(X)
        X = self._handle_outliers(X)
        X = self._scale_features(X)

        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted preprocessor.

        Args:
            df: Input DataFrame containing features

        Returns:
            Transformed DataFrame
        """
        if self.feature_columns is None:
            raise NotFittedError(
                "Preprocessor must be fitted before calling transform."
            )

        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        X = df[self.feature_columns].copy()

        # Apply transformations
        X = self._handle_missing_values(X)
        X = self._handle_outliers(X)

        if self.scale_features:
            cols_to_scale = self.technical_columns + self.return_columns
            X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])

        return X


class FeatureSelector:
    """
    A class for selecting relevant features based on various criteria.
    """

    def __init__(
        self, correlation_threshold: float = 0.95, missing_threshold: float = 0.5
    ):
        """
        Initialize the feature selector.

        Args:
            correlation_threshold: Maximum correlation allowed between features
            missing_threshold: Maximum ratio of missing values allowed
        """
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        self.selected_features = None

    def _remove_high_missing(self, df: pd.DataFrame) -> List[str]:
        """Remove features with too many missing values."""
        missing_ratio = df.isnull().sum() / len(df)
        return list(missing_ratio[missing_ratio < self.missing_threshold].index)

    def _remove_high_correlation(self, df: pd.DataFrame) -> List[str]:
        """Remove highly correlated features."""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]

        return [col for col in df.columns if col not in to_drop]

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the feature selector.

        Args:
            df: Input DataFrame
        """
        # Remove features with too many missing values
        features = self._remove_high_missing(df)

        # Remove highly correlated features
        features = self._remove_high_correlation(df[features])

        self.selected_features = features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with selected features
        """
        if self.selected_features is None:
            raise NotFittedError(
                "Feature selector must be fitted before calling transform."
            )

        return df[self.selected_features]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the selector and transform the data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with selected features
        """
        self.fit(df)
        return self.transform(df)
