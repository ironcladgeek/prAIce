from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate common regression metrics.

    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        sample_weight: Optional sample weights

    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
        "rmse": np.sqrt(
            mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        ),
        "r2": r2_score(y_true, y_pred, sample_weight=sample_weight),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "median_ae": np.median(np.abs(y_true - y_pred)),
        "max_error": np.max(np.abs(y_true - y_pred)),
        "std_error": np.std(y_true - y_pred),
    }

    return metrics


def calculate_financial_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate financial-specific metrics.

    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        returns: Optional array of actual returns
        risk_free_rate: Annual risk-free rate (default: 0.0)

    Returns:
        Dictionary containing calculated metrics
    """
    # Direction accuracy
    direction = np.sign(y_pred - np.roll(y_true, 1))
    direction_accuracy = (
        np.mean(direction[1:] == np.sign(returns[1:])) if returns is not None else 0
    )

    # Calculate trading metrics if returns are provided
    if returns is not None:
        # Generate trading signals (-1 for sell, 1 for buy)
        signals = np.where(direction > 0, 1, -1)

        # Calculate strategy returns
        strategy_returns = (
            signals[:-1] * returns[1:]
        )  # Shift signals by 1 to avoid look-ahead bias

        # Daily metrics
        daily_return = np.mean(strategy_returns)
        daily_volatility = np.std(strategy_returns)
        daily_downside_volatility = np.std(strategy_returns[strategy_returns < 0])

        # Annualized metrics (assuming 252 trading days)
        annual_return = (1 + daily_return) ** 252 - 1
        annual_volatility = daily_volatility * np.sqrt(252)
        annual_downside_volatility = daily_downside_volatility * np.sqrt(252)

        # Sharpe ratio
        excess_returns = strategy_returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

        # Sortino ratio
        sortino = (
            np.sqrt(252) * np.mean(excess_returns) / annual_downside_volatility
            if annual_downside_volatility != 0
            else np.inf
        )

        # Maximum drawdown
        cum_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(strategy_returns, 5)
        cvar_95 = np.mean(strategy_returns[strategy_returns <= var_95])

        # Win rate and profit metrics
        winning_trades = strategy_returns > 0
        win_rate = np.mean(winning_trades)
        avg_win = (
            np.mean(strategy_returns[winning_trades]) if any(winning_trades) else 0
        )
        avg_loss = (
            np.mean(strategy_returns[~winning_trades]) if any(~winning_trades) else 0
        )
        profit_factor = (
            abs(
                np.sum(strategy_returns[winning_trades])
                / np.sum(strategy_returns[~winning_trades])
            )
            if any(~winning_trades)
            else np.inf
        )

    else:
        annual_return = annual_volatility = sharpe = sortino = calmar = 0
        max_drawdown = var_95 = cvar_95 = win_rate = profit_factor = 0
        avg_win = avg_loss = 0

    metrics = {
        "direction_accuracy": direction_accuracy,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }

    return metrics


def calculate_rolling_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: Optional[np.ndarray] = None,
    window_size: int = 252,
    min_periods: int = 63,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate rolling metrics over time.

    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        returns: Optional array of actual returns
        window_size: Size of rolling window (default: 252 days)
        min_periods: Minimum number of observations required (default: 63 days)
        risk_free_rate: Annual risk-free rate (default: 0.0)

    Returns:
        DataFrame containing rolling metrics
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    if returns is not None:
        df["returns"] = returns

    # Statistical metrics
    def rolling_rmse(x):
        return np.sqrt(mean_squared_error(x["y_true"], x["y_pred"]))

    def rolling_mae(x):
        return mean_absolute_error(x["y_true"], x["y_pred"])

    def rolling_r2(x):
        return r2_score(x["y_true"], x["y_pred"])

    def rolling_mape(x):
        return np.mean(np.abs((x["y_true"] - x["y_pred"]) / x["y_true"])) * 100

    # Financial metrics
    def rolling_sharpe(x):
        if "returns" not in x.columns:
            return 0
        ret = x["returns"].values
        excess_ret = ret - risk_free_rate / 252
        return np.sqrt(252) * np.mean(excess_ret) / np.std(excess_ret)

    def rolling_sortino(x):
        if "returns" not in x.columns:
            return 0
        ret = x["returns"].values
        excess_ret = ret - risk_free_rate / 252
        downside_std = np.std(excess_ret[excess_ret < 0])
        return (
            np.sqrt(252) * np.mean(excess_ret) / downside_std
            if downside_std != 0
            else np.inf
        )

    def rolling_max_drawdown(x):
        if "returns" not in x.columns:
            return 0
        cum_rets = np.cumprod(1 + x["returns"].values)
        rolling_max = np.maximum.accumulate(cum_rets)
        drawdowns = (cum_rets - rolling_max) / rolling_max
        return np.min(drawdowns)

    # Calculate rolling metrics
    rolling_metrics = pd.DataFrame(
        {
            "rolling_rmse": df.rolling(
                window=window_size, min_periods=min_periods
            ).apply(rolling_rmse),
            "rolling_mae": df.rolling(
                window=window_size, min_periods=min_periods
            ).apply(rolling_mae),
            "rolling_r2": df.rolling(window=window_size, min_periods=min_periods).apply(
                rolling_r2
            ),
            "rolling_mape": df.rolling(
                window=window_size, min_periods=min_periods
            ).apply(rolling_mape),
        }
    )

    # Add financial metrics if returns are provided
    if returns is not None:
        financial_metrics = pd.DataFrame(
            {
                "rolling_sharpe": df.rolling(
                    window=window_size, min_periods=min_periods
                ).apply(rolling_sharpe),
                "rolling_sortino": df.rolling(
                    window=window_size, min_periods=min_periods
                ).apply(rolling_sortino),
                "rolling_max_drawdown": df.rolling(
                    window=window_size, min_periods=min_periods
                ).apply(rolling_max_drawdown),
            }
        )
        rolling_metrics = pd.concat([rolling_metrics, financial_metrics], axis=1)

    return rolling_metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: Optional[np.ndarray] = None,
    calculate_rolling: bool = True,
    window_size: int = 252,
    min_periods: int = 63,
    sample_weight: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    target_idx: int = 0,  # Index of target to use for financial metrics
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Comprehensive evaluation of predictions.

    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        returns: Optional array of actual returns
        calculate_rolling: Whether to calculate rolling metrics
        window_size: Size of rolling window for rolling metrics
        min_periods: Minimum periods for rolling metrics
        sample_weight: Optional sample weights for regression metrics
        risk_free_rate: Annual risk-free rate for financial metrics

    Returns:
        Dictionary containing all calculated metrics and rolling metrics if requested
    """
    # Handle multiple targets for regression metrics
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        regression_metrics = {}
        for i in range(y_true.shape[1]):
            target_metrics = calculate_regression_metrics(
                y_true=y_true[:, i], y_pred=y_pred[:, i], sample_weight=sample_weight
            )
            regression_metrics[f"target_{i}"] = target_metrics
    else:
        regression_metrics = calculate_regression_metrics(
            y_true=y_true.ravel(), y_pred=y_pred.ravel(), sample_weight=sample_weight
        )

    # Calculate financial metrics if returns are provided
    # Use only the specified target for financial metrics
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_financial = y_true[:, target_idx]
        y_pred_financial = y_pred[:, target_idx]
    else:
        y_true_financial = y_true.ravel()
        y_pred_financial = y_pred.ravel()

    financial_metrics = calculate_financial_metrics(
        y_true=y_true_financial,
        y_pred=y_pred_financial,
        returns=returns,
        risk_free_rate=risk_free_rate,
    )

    # Combine all metrics
    all_metrics = {
        "regression_metrics": regression_metrics,
        "financial_metrics": financial_metrics,
    }

    # Add rolling metrics if requested
    if calculate_rolling:
        rolling_metrics = calculate_rolling_metrics(
            y_true=y_true,
            y_pred=y_pred,
            returns=returns,
            window_size=window_size,
            min_periods=min_periods,
            risk_free_rate=risk_free_rate,
        )
        all_metrics["rolling_metrics"] = rolling_metrics

    return all_metrics
