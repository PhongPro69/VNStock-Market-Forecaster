"""
Quantitative Analysis Engine.
Implements Monte Carlo Stochastic Projections, Technical Confluence Scoring,
and Portfolio Risk Attribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class QuantEngine:
    """Quantitative computational engine for market simulation and signal synthesis."""

    @staticmethod
    def simulate_monte_carlo(
        current_price: float,
        historical_returns: pd.Series,
        forecast_days: int = 14,
        n_simulations: int = 100,
        expected_drift_pct: float | None = None
    ) -> Dict[str, Any]:
        """
        Execute Geometric Brownian Motion (GBM) Monte Carlo simulation.
        
        Parameters:
            current_price: Latest asset closing price.
            historical_returns: Series of daily log returns.
            forecast_days: Time horizon in business days.
            n_simulations: Number of stochastic paths.
            expected_drift_pct: Optional total return percentage drift estimated by ML model.
        """
        clean_ret = historical_returns.dropna()
        if len(clean_ret) < 20:
            sigma = 0.015
            mu = 0.0005
        else:
            sigma = float(clean_ret.std())
            mu = float(clean_ret.mean())

        # If an external ML drift is supplied, calibrate daily drift
        if expected_drift_pct is not None and forecast_days > 0:
            mu = (expected_drift_pct / 100.0) / forecast_days

        dt = 1.0
        drift = (mu - 0.5 * (sigma ** 2)) * dt
        volatility = sigma * np.sqrt(dt)

        np.random.seed(42)
        shock = np.random.normal(0, 1, (forecast_days, n_simulations))
        daily_log_returns = drift + volatility * shock

        # Compound price paths
        price_paths = np.zeros((forecast_days + 1, n_simulations))
        price_paths[0] = current_price

        for t in range(1, forecast_days + 1):
            price_paths[t] = price_paths[t - 1] * np.exp(daily_log_returns[t - 1])

        # Statistical percentiles
        p10 = np.percentile(price_paths, 10, axis=1)
        p50 = np.percentile(price_paths, 50, axis=1)
        p90 = np.percentile(price_paths, 90, axis=1)

        return {
            "paths": price_paths,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "daily_volatility": sigma,
            "annualized_volatility": sigma * np.sqrt(252),
            "expected_p50_target": p50[-1],
            "bull_p90_target": p90[-1],
            "bear_p10_target": p10[-1]
        }

    @staticmethod
    def calculate_confluence_score(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Synthesize multiple indicators into a composite 0-100 Technical Confluence Score.
        Weights: Trend (40%), Momentum (35%), Volume/Volatility (25%).
        """
        if df.empty or len(df) < 50:
            return {
                "score": 50,
                "rating": "Neutral",
                "color": "#94a3b8",
                "components": []
            }

        latest = df.iloc[-1]
        close = latest.get("close", 0.0)
        components: List[Dict[str, Any]] = []
        score = 0.0

        # --- 1. Trend Indicators (Max 40 pts) ---
        trend_score = 0.0
        sma_20 = latest.get("sma_20", close)
        sma_50 = latest.get("sma_50", close)
        sma_200 = latest.get("sma_200", close)

        if close > sma_20:
            trend_score += 15
            components.append({"factor": "Price > SMA 20", "status": "Bullish", "pts": "+15"})
        else:
            components.append({"factor": "Price < SMA 20", "status": "Bearish", "pts": "0"})

        if close > sma_50:
            trend_score += 15
            components.append({"factor": "Price > SMA 50", "status": "Bullish", "pts": "+15"})
        else:
            components.append({"factor": "Price < SMA 50", "status": "Bearish", "pts": "0"})

        if close > sma_200:
            trend_score += 10
            components.append({"factor": "Macro Bull (Price > SMA 200)", "status": "Bullish", "pts": "+10"})
        else:
            components.append({"factor": "Macro Bear (Price < SMA 200)", "status": "Bearish", "pts": "0"})

        score += trend_score

        # --- 2. Momentum Indicators (Max 35 pts) ---
        momentum_score = 0.0
        rsi = latest.get("rsi_14", 50.0)
        macd = latest.get("macd", 0.0)
        macd_sig = latest.get("macd_signal", 0.0)
        macd_hist = latest.get("macd_hist", 0.0)

        if 40 <= rsi <= 68:
            momentum_score += 15
            components.append({"factor": "RSI in Optimal Bull Zone (40-68)", "status": "Strong", "pts": "+15"})
        elif rsi < 30:
            momentum_score += 10
            components.append({"factor": "RSI Deep Oversold (<30)", "status": "Rebound Potential", "pts": "+10"})
        elif rsi > 70:
            momentum_score += 5
            components.append({"factor": "RSI Overbought (>70)", "status": "Caution", "pts": "+5"})

        if macd > macd_sig:
            momentum_score += 10
            components.append({"factor": "MACD > Signal Line", "status": "Bullish", "pts": "+10"})
        if macd_hist > 0:
            momentum_score += 10
            components.append({"factor": "Positive MACD Histogram", "status": "Expanding", "pts": "+10"})

        score += momentum_score

        # --- 3. Volatility & Volume (Max 25 pts) ---
        vol_score = 0.0
        volume = latest.get("volume", 0.0)
        vol_ma = df["volume"].tail(20).mean() if "volume" in df else volume

        if volume > vol_ma:
            vol_score += 15
            components.append({"factor": "Volume > 20D Average", "status": "Accumulation", "pts": "+15"})
        else:
            components.append({"factor": "Volume < 20D Average", "status": "Low Liquidity", "pts": "0"})

        bb_mid = (latest.get("bb_upper", close) + latest.get("bb_lower", close)) / 2.0
        if close > bb_mid:
            vol_score += 10
            components.append({"factor": "Trading in Upper Bollinger Band", "status": "Bullish", "pts": "+10"})

        score += vol_score
        final_score = int(min(max(score, 5), 98))

        if final_score >= 75:
            rating = "Strong Bullish"
            color = "#10b981"
        elif final_score >= 55:
            rating = "Moderate Accumulate"
            color = "#34d399"
        elif final_score >= 40:
            rating = "Consolidation / Neutral"
            color = "#94a3b8"
        elif final_score >= 25:
            rating = "Defensive Stance"
            color = "#fb923c"
        else:
            rating = "High Risk / Bearish"
            color = "#f43f5e"

        return {
            "score": final_score,
            "rating": rating,
            "color": color,
            "components": components
        }