"""
Model validation metrics & Quantitative Backtesting Engine.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelEvaluator:
    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        actual = np.asarray(actual).flatten()
        predicted = np.asarray(predicted).flatten()

        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual = actual[mask]
        predicted = predicted[mask]

        if len(actual) == 0:
            return {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "directional_accuracy": 0.0}

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100

        if len(actual) > 1:
            actual_dir = np.sign(np.diff(actual))
            pred_dir = np.sign(np.diff(predicted))
            hit_ratio = np.mean(actual_dir == pred_dir) * 100
        else:
            hit_ratio = 50.0

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "directional_accuracy": float(hit_ratio)
        }

    @staticmethod
    def backtest_strategy(
        prices: pd.Series,
        predicted_prices: pd.Series,
        threshold: float = 0.003,
        initial_capital: float = 100_000_000.0,
        fee_rate: float = 0.0015
    ) -> Dict[str, Any]:
        common_idx = prices.index.intersection(predicted_prices.index)
        actual = prices.loc[common_idx]
        preds = predicted_prices.loc[common_idx]

        expected_return = (preds - actual) / actual
        signals = np.where(expected_return > threshold, 1, 0)
        signals = pd.Series(signals, index=common_idx).shift(1).fillna(0)

        daily_returns = actual.pct_change().fillna(0)
        position_changes = signals.diff().abs().fillna(0)
        strategy_returns = (signals * daily_returns) - (position_changes * fee_rate)

        benchmark_equity = (1 + daily_returns).cumprod() * initial_capital
        strategy_equity = (1 + strategy_returns).cumprod() * initial_capital

        total_return_strategy = (strategy_equity.iloc[-1] / initial_capital - 1) * 100
        total_return_benchmark = (benchmark_equity.iloc[-1] / initial_capital - 1) * 100

        rf_daily = 0.03 / 252
        excess_returns = strategy_returns - rf_daily
        sharpe = (excess_returns.mean() / (strategy_returns.std() + 1e-9)) * np.sqrt(252)

        rolling_max = strategy_equity.cummax()
        drawdown = (strategy_equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        return {
            "initial_capital": initial_capital,
            "final_equity_strategy": float(strategy_equity.iloc[-1]),
            "final_equity_benchmark": float(benchmark_equity.iloc[-1]),
            "total_return_strategy_pct": float(total_return_strategy),
            "total_return_benchmark_pct": float(total_return_benchmark),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_drawdown),
            "equity_df": pd.DataFrame({
                "Strategy": strategy_equity,
                "Buy_and_Hold": benchmark_equity
            }, index=common_idx)
        }
