from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from praice.data_handling.db_ops.crud import get_symbol
from praice.data_handling.models import HistoricalPrice1D, TechnicalAnalysis


class PriceAndTechnicalAnalysisExtractor:
    """
    A class to extract historical prices and technical analysis data for a given symbol.

    Attributes:
    -----------
    symbol : str
        The symbol for which the data is to be extracted.

    Methods:
    --------
    get_data() -> pd.DataFrame:
        Retrieves historical prices and technical analysis data for the symbol,
        merges them into a single DataFrame, and returns the result.
    """

    def __init__(self, symbol: str):
        self.symbol = get_symbol(symbol)

    def get_data(self) -> pd.DataFrame:
        """
        Retrieve and process historical price and technical analysis data for a given symbol.

        Returns:
            pd.DataFrame: A DataFrame containing the processed historical price and technical analysis data.
        """
        # Query historical prices
        prices_query = HistoricalPrice1D.select().where(
            HistoricalPrice1D.symbol == self.symbol
        )

        # Query technical analysis
        technical_query = TechnicalAnalysis.select().where(
            TechnicalAnalysis.symbol == self.symbol
        )

        # Convert to DataFrame
        prices_df = pd.DataFrame(list(prices_query.dicts())).sort_values(
            "date", ignore_index=True
        )
        technical_df = pd.DataFrame(list(technical_query.dicts())).sort_values(
            "date", ignore_index=True
        )

        # Normalize JSON fields
        technical_indicators_df = pd.json_normalize(
            technical_df["technical_indicators"]
        )
        candlestick_patterns_df = pd.json_normalize(
            technical_df["candlestick_patterns"]
        )

        # Merge DataFrames
        final_df = prices_df.merge(
            technical_df, on=["symbol", "date"], suffixes=("_price", "_tech")
        )
        final_df = final_df.join(technical_indicators_df).join(candlestick_patterns_df)

        # Drop extra columns
        final_df = final_df.drop(
            columns=[
                "id",
                "technical_indicators",
                "candlestick_patterns",
                "symbol",
                "timeframe",
                "dividends",
                "stock_splits",
            ],
            errors="ignore",
        )

        # Set date as index
        final_df.set_index("date", inplace=True)

        # Convert object columns to float
        object_cols = final_df.select_dtypes(include="object").columns
        final_df[object_cols] = final_df[object_cols].astype(float)

        return final_df


class ReturnsCalculator:
    """
    A class to calculate simple and logarithmic returns over specified rolling windows.

    Attributes:
    -----------
    df : pd.DataFrame
        The input dataframe containing the 'close' prices.
    rolling_windows : list, optional
        A list of integers representing the rolling window periods in days.
        Defaults to [5, 10, 21, 63, 126, 252] which correspond to ~1 week, 2 weeks, 1 month, 3 months, 6 months, and 1 year.

    Methods:
    --------
    calculate_returns() -> pd.DataFrame:
        Calculates the simple and logarithmic returns for the specified rolling windows and adds them as new columns to the dataframe.
    """

    def __init__(self, df: pd.DataFrame, rolling_windows: Optional[list] = None):
        self.df = df.copy()
        self.rolling_windows = (
            rolling_windows if rolling_windows else [5, 10, 21, 63, 126, 252]
        )  # ~1w, 2w, 1m, 3m, 6m, 1y

    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate simple and logarithmic returns for specified rolling windows.

        This method calculates both simple and logarithmic returns for each rolling
        window specified in the `rolling_windows` attribute. The results are added
        as new columns to the DataFrame `df`.

        Simple returns are calculated using the percentage change in the 'close'
        prices over the specified window. Logarithmic returns are calculated using
        the natural logarithm of the ratio of 'close' prices over the specified window.

        Returns:
            pd.DataFrame: The DataFrame `df` with additional columns for simple and
            logarithmic returns for each rolling window.
        """
        for window in self.rolling_windows:
            self.df[f"simple_return_{window}d"] = self.df["close"].pct_change(window)
            log_returns = np.log(
                (self.df["close"] / self.df["close"].shift(window))
                .fillna(1)
                .astype(float)
                .values
            )
            log_returns[:window] = np.nan
            self.df[f"log_return_{window}d"] = log_returns

        return self.df


class FuturePricesAdder:
    """
    A class used to add future prices to a DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame containing historical price data.
    future_periods : list, optional
        A list of integers representing the future periods in days.
        Defaults to [21, 63] which correspond to ~1 month and 3 months.
    """

    def __init__(self, df: pd.DataFrame, future_periods: Optional[list] = None):
        self.df = df.copy()
        self.future_periods = future_periods if future_periods else [21, 63]  # ~1m, 3m

    def add_future_prices(self) -> pd.DataFrame:
        """
        Adds future prices to the DataFrame.

        This method adds new columns to the DataFrame for each period in
        `self.future_periods`. Each new column contains the closing prices
        shifted by the specified period into the future.

        Returns:
            pd.DataFrame: The DataFrame with the added future price columns.
        """
        for period in self.future_periods:
            self.df[f"future_close_{period}d"] = self.df["close"].shift(-period)
        return self.df


class PreviousPricesAdder:
    """
    A class used to add previous close prices to a DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame containing historical price data.
    n_periods : int
        The number of previous periods to add as columns.
    """

    def __init__(self, df: pd.DataFrame, n_periods: int = 21):
        self.df = df.copy()
        self.n_periods = n_periods

    def add_previous_prices(self) -> pd.DataFrame:
        """
        Adds previous close prices to the DataFrame.

        This method adds new columns to the DataFrame for each of the previous
        periods specified by `self.n_periods`. Each new column contains the closing
        prices shifted by the specified period into the past.

        Returns:
            pd.DataFrame: The DataFrame with the added previous price columns.
        """
        for period in range(1, self.n_periods + 1):
            self.df[f"previous_close_{period}d"] = self.df["close"].shift(period)
        return self.df


def prepare_price_data(
    symbol: str, drop_null_features: bool = True, drop_null_targets: bool = False
) -> pd.DataFrame:
    """
    Prepares price data for a given symbol by extracting data, calculating returns,
    and adding future prices (targets).

    Parameters:
        symbol (str): The symbol for which to prepare the price data.
        drop_null_features (bool): If True, drop rows with null values after calculating returns. Default is True.
        drop_null_targets (bool): If True, drop rows with null values after adding future prices. Default is False.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared price data.
    """
    # Extract data
    df = PriceAndTechnicalAnalysisExtractor(symbol).get_data()

    # Calculate returns
    df = ReturnsCalculator(df).calculate_returns()

    # Add previous prices
    df = PreviousPricesAdder(df).add_previous_prices()
    if drop_null_features:
        df = df.dropna()

    # Add future prices
    df = FuturePricesAdder(df).add_future_prices()
    if drop_null_targets:
        df = df.dropna()

    return df


def get_train_and_test_data(
    symbol: str,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    train_size: Optional[float] = None,
    test_size: Optional[float] = None,
    split_date: Optional[str] = None,
    skip_start_pct: Optional[float] = None,
    skip_end_pct: Optional[float] = None,
    drop_null_features: bool = True,
    drop_null_targets: bool = True,
) -> tuple:
    """
    Get train and test data for a given symbol.

    Parameters:
        symbol (str): The symbol for which to get train and test data.
        train_start (str, optional): The start date for the training set. Format: 'YYYY-MM-DD'.
        train_end (str, optional): The end date for the training set. Format: 'YYYY-MM-DD'.
        test_start (str, optional): The start date for the test set. Format: 'YYYY-MM-DD'.
        test_end (str, optional): The end date for the test set. Format: 'YYYY-MM-DD'.
        train_size (float, optional): The proportion of the dataset to include in the train split. Should be between 0.0 and 1.0.
        test_size (float, optional): The proportion of the dataset to include in the test split. Should be between 0.0 and 1.0.
        split_date (str, optional): The date to split the data into train and test sets. Format: 'YYYY-MM-DD'.
        skip_start_pct (float, optional): Percentage of data to skip from the start. Should be between 0.0 and 1.0.
        skip_end_pct (float, optional): Percentage of data to skip from the end. Should be between 0.0 and 1.0.
        drop_null_features (bool): If True, drop rows with null values after calculating returns. Default is True.
        drop_null_targets (bool): If True, drop rows with null values after adding future prices. Default is True.

    Returns:
        tuple: A tuple containing the train and test DataFrames.

    Raises:
        ValueError: If invalid combination of parameters is provided or if parameters are out of valid range.
    """
    # Validate input parameters
    if split_date:
        if any([train_start, train_end, test_start, test_end, train_size, test_size]):
            raise ValueError(
                "When split_date is provided, other splitting parameters should not be used."
            )
    elif train_start and train_end:
        if any([train_size, test_size]):
            raise ValueError(
                "Cannot use train_size or test_size with train_start/train_end."
            )
        if not test_start or not test_end:
            raise ValueError(
                "Both test_start and test_end must be provided with train_start/train_end."
            )
    elif test_start and test_end:
        if any([train_size, test_size]):
            raise ValueError(
                "Cannot use train_size or test_size with test_start/test_end."
            )
        if not train_start or not train_end:
            raise ValueError(
                "Both train_start and train_end must be provided with test_start/test_end."
            )
    elif train_size or test_size:
        if train_size and test_size:
            raise ValueError("Provide either train_size or test_size, not both.")
        if any([train_start, train_end, test_start, test_end]):
            raise ValueError(
                "Cannot use train_start/train_end/test_start/test_end with train_size/test_size."
            )
        if train_size and (train_size <= 0 or train_size >= 1):
            raise ValueError("train_size must be between 0 and 1.")
        if test_size and (test_size <= 0 or test_size >= 1):
            raise ValueError("test_size must be between 0 and 1.")
    else:
        raise ValueError(
            "Must provide either split_date, train_start/train_end with test_start/test_end, "
            "or one of train_size/test_size."
        )

    # Validate skip percentages
    if skip_start_pct is not None and (skip_start_pct < 0 or skip_start_pct >= 1):
        raise ValueError("skip_start_pct must be between 0 and 1")
    if skip_end_pct is not None and (skip_end_pct < 0 or skip_end_pct >= 1):
        raise ValueError("skip_end_pct must be between 0 and 1")
    if (
        skip_start_pct is not None
        and skip_end_pct is not None
        and (skip_start_pct + skip_end_pct >= 1)
    ):
        raise ValueError("sum of skip_start_pct and skip_end_pct must be less than 1")

    # Get the data
    df = prepare_price_data(
        symbol=symbol,
        drop_null_features=drop_null_features,
        drop_null_targets=drop_null_targets,
    )

    # Apply skip percentages if provided
    if skip_start_pct is not None:
        start_idx = int(len(df) * skip_start_pct)
        df = df.iloc[start_idx:]
    if skip_end_pct is not None:
        end_idx = int(len(df) * (1 - skip_end_pct))
        df = df.iloc[:end_idx]

    # Split the data
    if split_date:
        split_date = datetime.strptime(split_date, "%Y-%m-%d").date()
        train_df = df.loc[df.index <= split_date]
        test_df = df.loc[df.index > split_date]
    elif train_start and train_end:
        train_df = df.loc[train_start:train_end]
        test_df = df.loc[test_start:test_end]
    else:
        # Calculate the complementary size if only one is provided
        if train_size:
            test_size = 1 - train_size
            split_idx = int(len(df) * train_size)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:  # test_size is provided
            train_size = 1 - test_size
            split_idx = int(len(df) * train_size)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

    if train_df.empty or test_df.empty:
        raise ValueError("One or both of the resulting DataFrames is empty.")

    return train_df, test_df
