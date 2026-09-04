"""
Machine Learning ensemble models (XGBoost, Random Forest, Ridge).
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class TreeEnsembleForecaster:
    """Gradient Boosting & Random Forest forecaster."""

    def __init__(self, model_type: str = "xgboost", **hyperparams):
        self.model_type = model_type.lower()
        self.hyperparams = hyperparams
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
                params = {
                    "n_estimators": 150,
                    "max_depth": 5,
                    "learning_rate": 0.03,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1
                }
                params.update(self.hyperparams)
                return XGBRegressor(**params)
            except ImportError:
                return GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)

        elif self.model_type == "random_forest":
            params = {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 5,
                "random_state": 42,
                "n_jobs": -1
            }
            params.update(self.hyperparams)
            return RandomForestRegressor(**params)

        elif self.model_type == "ridge":
            return Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TreeEnsembleForecaster":
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return pd.DataFrame({
                "feature": self.feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False).head(top_n)
        return pd.DataFrame()
