"""
Feature engineering module for time series analysis and stock market forecasting.
Includes trend indicators, momentum oscillators, volatility bands, and lag features.
"""

from __future__ import annotations

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Calculates quantitative financial indicators and prepares structured feature matrices."""

    def __init__(
        self,
        sma_windows: Tuple[int, ...] = (10, 20, 50, 200),
        ema_windows: Tuple[int, ...] = (12, 26),
        rsi_window: int = 14,
        bollinger_window: int = 20,
        bollinger_std: float = 2.0,
        lag_days: Tuple[int, ...] = (1, 2, 3, 5, 10),
        forecast_horizon: int = 1
    ):
        self.sma_windows = sma_windows
        self.ema_windows = ema_windows
        self.rsi_window = rsi_window
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std
        self.lag_days = lag_days
        self.forecast_horizon = forecast_horizon

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # 1. Moving Averages
        for w in self.sma_windows:
            data[f"sma_{w}"] = close.rolling(window=w).mean()
            data[f"dist_sma_{w}"] = (close - data[f"sma_{w}"]) / data[f"sma_{w}"]

        for w in self.ema_windows:
            data[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

        # 2. RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        data[f"rsi_{self.rsi_window}"] = 100 - (100 / (1 + rs))

        # 3. MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        data["macd"] = ema_12 - ema_26
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]

        # 4. Bollinger Bands
        bb_mid = close.rolling(window=self.bollinger_window).mean()
        bb_std = close.rolling(window=self.bollinger_window).std()
        data["bb_upper"] = bb_mid + (bb_std * self.bollinger_std)
        data["bb_lower"] = bb_mid - (bb_std * self.bollinger_std)
        data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / bb_mid

        # 5. Volatility & ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr_14"] = tr.rolling(window=14).mean()
        data["rolling_vol_7d"] = close.pct_change().rolling(7).std() * np.sqrt(252)
        data["rolling_vol_30d"] = close.pct_change().rolling(30).std() * np.sqrt(252)

        # 6. Volume
        data["volume_sma_20"] = volume.rolling(20).mean()
        data["volume_ratio"] = volume / (data["volume_sma_20"] + 1e-9)

        # 7. Returns & Lags
        data["return_1d"] = close.pct_change(1)
        data["return_5d"] = close.pct_change(5)
        for lag in self.lag_days:
            data[f"lag_close_{lag}"] = close.shift(lag)
            data[f"lag_return_{lag}"] = data["return_1d"].shift(lag)

        return data

    def prepare_modeling_data(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        predict_return: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        enriched = self.add_technical_indicators(df)

        if predict_return:
            target = (enriched[target_col].shift(-self.forecast_horizon) - enriched[target_col]) / enriched[target_col]
        else:
            target = enriched[target_col].shift(-self.forecast_horizon)

        target.name = "target"
        exclude_cols = ["target", "open", "high", "low", "close", "volume"]
        raw_feature_cols = [col for col in enriched.columns if col not in exclude_cols]

        # Prevent dropping all rows when long window indicators (e.g. SMA200) contain all NaNs
        min_valid_threshold = max(20, int(len(enriched) * 0.4))
        feature_cols = [c for c in raw_feature_cols if enriched[c].notna().sum() >= min_valid_threshold]

        combined = pd.concat([enriched[feature_cols], target], axis=1).dropna()
        X = combined[feature_cols]
        y = combined["target"]

        return X, y, feature_cols
