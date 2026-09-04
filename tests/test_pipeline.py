"""
Unit tests for data loader, feature engineering, database, quant engine and models.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import VNStockDataLoader
from src.feature_engineering import FeatureEngineer
from src.models.ml_forecaster import TreeEnsembleForecaster
from src.models.evaluator import ModelEvaluator
from src.database import DatabaseManager
from src.quant_engine import QuantEngine


@pytest.fixture
def sample_stock_df():
    loader = VNStockDataLoader()
    return loader._generate_synthetic_stock_data("FPT", "2023-01-01", "2023-12-31")


def test_data_loader_structure(sample_stock_df):
    assert not sample_stock_df.empty
    expected_cols = ["open", "high", "low", "close", "volume"]
    for col in expected_cols:
        assert col in sample_stock_df.columns
    assert isinstance(sample_stock_df.index, pd.DatetimeIndex)


def test_feature_engineering(sample_stock_df):
    fe = FeatureEngineer()
    enriched = fe.add_technical_indicators(sample_stock_df)
    assert "rsi_14" in enriched.columns
    assert "macd" in enriched.columns
    assert "bb_upper" in enriched.columns
    assert "lag_close_1" in enriched.columns

    X, y, cols = fe.prepare_modeling_data(sample_stock_df)
    assert len(X) == len(y)
    assert len(X) > 0


def test_database_manager(sample_stock_df):
    db = DatabaseManager()
    stats = db.get_warehouse_stats()
    assert stats["total_companies"] >= 20

    inserted = db.upsert_ohlcv("TEST", sample_stock_df)
    df_db = db.get_ohlcv("TEST")
    assert not df_db.empty


def test_quant_engine(sample_stock_df):
    fe = FeatureEngineer()
    df_tech = fe.add_technical_indicators(sample_stock_df)
    score = QuantEngine.calculate_confluence_score(df_tech)
    assert 0 <= score["score"] <= 100
    assert "rating" in score

    mc = QuantEngine.simulate_monte_carlo(
        current_price=100.0,
        historical_returns=sample_stock_df["close"].pct_change(),
        forecast_days=7,
        n_simulations=20
    )
    assert mc["paths"].shape == (8, 20)
    assert mc["p50"].shape == (8,)


def test_models_fit_predict(sample_stock_df):
    fe = FeatureEngineer()
    X, y, _ = fe.prepare_modeling_data(sample_stock_df)

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    rf = TreeEnsembleForecaster(model_type="random_forest", n_estimators=20)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert len(preds) == len(y_test)

    metrics = ModelEvaluator.calculate_metrics(y_test.values, preds)
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert 0 <= metrics["directional_accuracy"] <= 100


def test_backtest_strategy(sample_stock_df):
    prices = sample_stock_df["close"].tail(50)
    fake_preds = prices * (1 + np.random.normal(0, 0.01, size=len(prices)))
    bt = ModelEvaluator.backtest_strategy(prices, fake_preds, initial_capital=100_000_000)
    assert "final_equity_strategy" in bt
    assert "sharpe_ratio" in bt