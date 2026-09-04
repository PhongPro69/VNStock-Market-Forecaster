"""
Baseline models for stock price forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class MovingAverageForecaster:
    """Predicts price using rolling window mean."""

    def __init__(self, window: int = 10):
        self.window = window
        self.history: Optional[pd.Series] = None

    def fit(self, y_train: pd.Series) -> "MovingAverageForecaster":
        self.history = y_train
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.history is None:
            raise ValueError("Model must be fitted first.")
        val = float(self.history.tail(self.window).mean())
        return np.full(shape=(steps,), fill_value=val)


class ARIMABaseline:
    """Standard Auto-Regressive Integrated Moving Average (ARIMA) baseline."""

    def __init__(self, order: tuple = (2, 1, 2)):
        self.order = order
        self.model_res = None

    def fit(self, y_train: pd.Series) -> "ARIMABaseline":
        from statsmodels.tsa.arima.model import ARIMA
        try:
            model = ARIMA(y_train, order=self.order)
            self.model_res = model.fit()
        except Exception:
            model = ARIMA(y_train, order=(1, 1, 1))
            self.model_res = model.fit()
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.model_res is None:
            raise ValueError("Model must be fitted first.")
        forecast = self.model_res.forecast(steps=steps)
        return np.asarray(forecast)
