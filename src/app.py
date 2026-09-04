"""
VN-Quant Terminal: Full-Width Enterprise Quantitative Analytics.
Engineered for the Vietnam Stock Market (VN30 / HOSE).
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import VNStockDataLoader, VN30_TICKERS, TICKER_NAMES
from src.database import DatabaseManager, VN30_PROFILES
from src.feature_engineering import FeatureEngineer
from src.models.ml_forecaster import TreeEnsembleForecaster
from src.models.baseline import ARIMABaseline
from src.models.evaluator import ModelEvaluator
from src.quant_engine import QuantEngine

# --- Streamlit Setup: Full-width, collapsed sidebar ---
st.set_page_config(
    page_title="VN-Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- TradingView Dark Terminal Styling (Complete Header Removal & High Contrast) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #f1f5f9;
    }
    .stApp {
        background-color: #11141d;
    }

    /* 1. COMPLETELY HIDE STREAMLIT TOP BAR (ELIMINATES WHITE STRIP & DEPLOY BUTTON) */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    .stDeployButton, [data-testid="stDeployButton"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
    }

    /* 2. Container full-width padding */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* 3. Top Command Hub Bar */
    .top-command-bar {
        background: linear-gradient(180deg, #191e2b 0%, #141822 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    /* 4. Terminal Cards */
    .term-card {
        background-color: #171b26;
        border: 1px solid #283042;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }
    .term-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .term-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
    }
    .term-sub {
        font-size: 0.78rem;
        color: #cbd5e1;
        margin-top: 5px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* 5. 52-Week Range Bar */
    .range-track {
        width: 100%;
        height: 7px;
        background-color: #2d3748;
        border-radius: 4px;
        position: relative;
        margin-top: 8px;
    }
    .range-fill {
        height: 100%;
        background: linear-gradient(90deg, #f43f5e 0%, #fbbf24 50%, #10b981 100%);
        border-radius: 4px;
    }
    .range-marker {
        position: absolute;
        top: -3px;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background-color: #ffffff;
        box-shadow: 0 0 8px rgba(255,255,255,0.9);
        transform: translateX(-50%);
    }

    /* 6. Tabs High Contrast */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #141824;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #283042;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #94a3b8;
        background: transparent;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
    }

    /* 7. Input and select widgets */
    .stSelectbox > div > div {
        background-color: #171b26 !important;
        border: 1px solid #2d3748 !important;
        color: #ffffff !important;
    }
    input {
        color: #ffffff !important;
        background-color: #171b26 !important;
        border: 1px solid #2d3748 !important;
    }
    input::placeholder {
        color: #64748b !important;
    }

    /* 8. Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .badge-bull { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-bear { background-color: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.4); }
    .badge-neutral { background-color: rgba(148, 163, 184, 0.2); color: #e2e8f0; border: 1px solid rgba(148, 163, 184, 0.4); }
    .badge-info { background-color: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
</style>
""", unsafe_allow_html=True)


# --- Initialize Database & State ---
db = DatabaseManager()
db_stats = db.get_warehouse_stats()

SECTOR_MAP = {
    "Technology": ["FPT"],
    "Banking": ["VCB", "TCB", "MBB", "ACB", "VPB", "CTG", "BID", "STB"],
    "Steel & Materials": ["HPG", "MSN", "VNM", "SAB"],
    "Real Estate & Retail": ["VHM", "VIC", "MWG", "VRE", "SSI", "GAS", "PLX"]
}

if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = "FPT"
if "timeframe" not in st.session_state:
    st.session_state.timeframe = "1Y"
if "active_sector" not in st.session_state:
    st.session_state.active_sector = "Technology"


# =============================================================================
# TOP NAVIGATION & COMMAND HUB
# =============================================================================
st.markdown("""
<div class="top-command-bar">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
                VN·QUANT <span style="font-size: 0.75rem; color: #38bdf8; font-weight: 700; padding: 3px 8px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 4px;">PRO TERMINAL</span>
            </div>
            <span class="badge badge-bull">● HOSE LIVE</span>
            <span class="badge badge-info">SQL Warehouse: Connected ({total_bars:,} bars)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <a href="https://github.com/PhongPro69" target="_blank" style="text-decoration: none;">
                <span class="badge" style="background: rgba(37, 99, 235, 0.25); color: #93c5fd; border: 1px solid rgba(96, 165, 250, 0.5); padding: 4px 12px; font-weight: 700; font-size: 0.78rem;">
                    👨‍💻 Vũ Thanh Phong (@PhongPro69)
                </span>
            </a>
            <div style="font-size: 0.78rem; color: #cbd5e1;">
                Storage: <strong style="color: #38bdf8;">{dialect}</strong> &nbsp;·&nbsp; Auto-Sync: Active
            </div>
        </div>
    </div>
</div>
""".format(total_bars=db_stats['total_bars'], dialect=db_stats['dialect']), unsafe_allow_html=True)

# 3-Column Central Control
nav_col1, nav_col2, nav_col3 = st.columns([1.2, 2.8, 1.5])

with nav_col1:
    st.markdown("<div class='term-label'>1. INDUSTRY SECTOR</div>", unsafe_allow_html=True)
    selected_sec = st.selectbox(
        "Select Sector",
        list(SECTOR_MAP.keys()),
        index=list(SECTOR_MAP.keys()).index(st.session_state.active_sector),
        label_visibility="collapsed"
    )
    if selected_sec != st.session_state.active_sector:
        st.session_state.active_sector = selected_sec
        if st.session_state.active_ticker not in SECTOR_MAP[selected_sec]:
            st.session_state.active_ticker = SECTOR_MAP[selected_sec][0]
        st.rerun()

with nav_col2:
    st.markdown("<div class='term-label'>2. VN30 ASSET (CLICK TO SWITCH)</div>", unsafe_allow_html=True)
    tickers_in_sec = SECTOR_MAP[st.session_state.active_sector]
    chip_cols = st.columns(min(len(tickers_in_sec), 8))
    for i, t in enumerate(tickers_in_sec):
        is_active = (st.session_state.active_ticker == t)
        if chip_cols[i].button(
            t,
            key=f"chip_{t}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.active_ticker = t
            st.rerun()

with nav_col3:
    st.markdown("<div class='term-label'>3. TIMEFRAME & SEARCH</div>", unsafe_allow_html=True)
    tf_c1, tf_c2 = st.columns([2, 1.2])
    with tf_c1:
        custom_input = st.text_input(
            "Symbol",
            placeholder="Search (DGC, DIG...)",
            label_visibility="collapsed"
        ).strip().upper()
        if custom_input and custom_input != st.session_state.active_ticker:
            st.session_state.active_ticker = custom_input
            st.rerun()
    with tf_c2:
        tf_options = ["3M", "6M", "1Y", "3Y"]
        new_tf = st.selectbox(
            "Timeframe",
            tf_options,
            index=tf_options.index(st.session_state.timeframe),
            label_visibility="collapsed"
        )
        if new_tf != st.session_state.timeframe:
            st.session_state.timeframe = new_tf
            st.rerun()

ticker = st.session_state.active_ticker

# Collapsible Model Configuration
with st.expander("⚙️ Predictive Engine & Model Hyperparameters (Click to expand)", expanded=False):
    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        model_choice = st.selectbox(
            "Forecasting Model",
            ["XGBoost Regressor (Recommended)", "Random Forest Ensemble", "Ridge Linear Regressor", "ARIMA(2,1,2) Baseline"]
        )
        model_key = "xgboost" if "XGBoost" in model_choice else "random_forest" if "Random" in model_choice else "ridge" if "Ridge" in model_choice else "arima"
    with exp_c2:
        forecast_days = st.slider("Forecast Horizon (Trading Days Ahead)", min_value=3, max_value=30, value=14, step=1)
    with exp_c3:
        test_split = st.slider("Out-of-Sample Test Split", min_value=0.10, max_value=0.35, value=0.20, step=0.05)


# =============================================================================
# DATA PIPELINE & QUANTITATIVE COMPUTATION
# =============================================================================
# Always fetch at least 3 years to ensure SMA200 and long features never suffer from lookback starvation
end_dt = datetime.now()
lookback_start_dt = end_dt - timedelta(days=365 * 3)

# Filter start date for chart visualization
days_map = {"3M": 90, "6M": 180, "1Y": 365, "3Y": 365 * 3}
vis_start_dt = end_dt - timedelta(days=days_map.get(st.session_state.timeframe, 365))

loader = VNStockDataLoader()
with st.spinner(f"Ingesting {ticker} into SQL Warehouse..."):
    try:
        raw_df = loader.fetch_data(
            ticker,
            start_date=lookback_start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d")
        )
    except Exception as exc:
        st.error(f"Data ingestion error for {ticker}: {exc}")
        st.stop()

if raw_df.empty or len(raw_df) < 30:
    st.warning(f"Insufficient historical data for {ticker}.")
    st.stop()

# Feature Engineering on full history
fe = FeatureEngineer()
df_tech = fe.add_technical_indicators(raw_df)

# Sliced view for charts according to selected timeframe
df_plot = df_tech[df_tech.index >= vis_start_dt]
if df_plot.empty or len(df_plot) < 10:
    df_plot = df_tech.tail(60)

# Metadata
company_name = TICKER_NAMES.get(ticker, f"{ticker} Corporation")
sector_tag = next((sec for sec, t_list in SECTOR_MAP.items() if ticker in t_list), "Equities")

# Latest price metrics
latest_close = float(raw_df["close"].iloc[-1])
prev_close = float(raw_df["close"].iloc[-2]) if len(raw_df) > 1 else latest_close
change_val = latest_close - prev_close
change_pct = (change_val / prev_close) * 100
volume_latest = float(raw_df["volume"].iloc[-1])

high_52w = float(raw_df["high"].tail(252).max()) if len(raw_df) >= 20 else float(raw_df["high"].max())
low_52w = float(raw_df["low"].tail(252).min()) if len(raw_df) >= 20 else float(raw_df["low"].min())
range_pct = max(0.0, min(100.0, ((latest_close - low_52w) / (high_52w - low_52w + 1e-6)) * 100))

# Technical Confluence Score
confluence = QuantEngine.calculate_confluence_score(df_tech)


# =============================================================================
# STOCK BANNER & KEY PERFORMANCE METRICS
# =============================================================================
delta_color = "#10b981" if change_val >= 0 else "#f43f5e"
delta_sign = "+" if change_val >= 0 else ""
trend_badge = f'<span class="badge badge-bull">▲ BULLISH (+{change_pct:.2f}%)</span>' if change_val > 0 else f'<span class="badge badge-bear">▼ BEARISH ({change_pct:.2f}%)</span>' if change_val < 0 else '<span class="badge badge-neutral">● FLAT</span>'

header_col, price_col = st.columns([3, 1.2])

with header_col:
    st.markdown(f"""
    <div style="display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 2.4rem; font-weight: 800; color: #ffffff; letter-spacing: -0.03em;">
            {ticker}
        </span>
        <span style="font-size: 1.15rem; font-weight: 600; color: #f1f5f9;">
            {company_name}
        </span>
        <span class="badge badge-neutral">{sector_tag}</span>
        {trend_badge}
    </div>
    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
        HOSE / HNX Financial Series &nbsp;·&nbsp; {df_plot.index[0].strftime('%d/%m/%Y')} – {df_plot.index[-1].strftime('%d/%m/%Y')} &nbsp;·&nbsp; <strong style="color: #f1f5f9;">{len(df_plot):,} Visualized Sessions</strong>
    </div>
    """, unsafe_allow_html=True)

with price_col:
    st.markdown(f"""
    <div style="text-align: right;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.4rem; font-weight: 800; color: {delta_color}; line-height: 1;">
            {latest_close:,.0f} <span style="font-size: 1rem; color: #94a3b8; font-weight: 500;">VND</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 700; color: {delta_color}; margin-top: 4px;">
            {delta_sign}{change_val:,.0f} VND ({delta_sign}{change_pct:.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# 5 KPI Cards Strip
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="term-card">
        <div class="term-label">LATEST CLOSE</div>
        <div class="term-value">{latest_close:,.0f} <span style="font-size: 0.75rem; color: #94a3b8;">VND</span></div>
        <div class="term-sub" style="color: {delta_color}; font-weight: 600;">{delta_sign}{change_pct:.2f}% Session Delta</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="term-card">
        <div class="term-label">SESSION VOLUME</div>
        <div class="term-value">{volume_latest:,.0f}</div>
        <div class="term-sub">Shares Transacted</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="term-card">
        <div class="term-label">52-WEEK RANGE</div>
        <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #f1f5f9; font-weight: 600;">
            <span>{low_52w:,.0f}</span>
            <span>{high_52w:,.0f}</span>
        </div>
        <div class="range-track">
            <div class="range-fill" style="width: {range_pct}%;"></div>
            <div class="range-marker" style="left: {range_pct}%;"></div>
        </div>
        <div class="term-sub" style="justify-content: flex-end;">Position: <strong>{range_pct:.0f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    rsi_val = float(df_tech["rsi_14"].dropna().iloc[-1]) if "rsi_14" in df_tech.columns else 50.0
    rsi_col = "#10b981" if 40 <= rsi_val <= 65 else "#f43f5e" if rsi_val > 70 else "#38bdf8"
    st.markdown(f"""
    <div class="term-card">
        <div class="term-label">RSI (14) MOMENTUM</div>
        <div class="term-value" style="color: {rsi_col};">{rsi_val:.1f}</div>
        <div class="term-sub" style="color: {rsi_col}; font-weight: 600;">{'Oversold (<30)' if rsi_val < 30 else 'Overbought (>70)' if rsi_val > 70 else 'Healthy Momentum'}</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="term-card" style="border-left: 3px solid {confluence['color']};">
        <div class="term-label">TECHNICAL CONFLUENCE</div>
        <div class="term-value" style="color: {confluence['color']};">{confluence['score']}<span style="font-size: 0.85rem; color: #94a3b8;">/100</span></div>
        <div class="term-sub" style="font-weight: 700; color: {confluence['color']};">{confluence['rating']}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB NAVIGATION (FULL 100% SCREEN WIDTH)
# =============================================================================
tab_chart, tab_forecast, tab_backtest, tab_db = st.tabs([
    "Interactive Candlestick & Indicators",
    "AI Forecast & Monte Carlo Stochastic Cones",
    "Quantitative Strategy & Risk Drawdown",
    "SQL Warehouse & Feature Drivers"
])

PLOT_THEME = dict(
    paper_bgcolor="#11141d",
    plot_bgcolor="#11141d",
    font=dict(family="Inter", color="#cbd5e1", size=11),
    hoverlabel=dict(bgcolor="#1c2230", bordercolor="#3b4252", font=dict(family="JetBrains Mono", color="#ffffff")),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(20, 24, 36, 0.95)",
        bordercolor="#283042",
        borderwidth=1,
        font=dict(color="#f1f5f9", size=11)
    ),
    margin=dict(l=10, r=10, t=10, b=10)
)


# =============================================================================
# TAB 1: INTERACTIVE CHART & TECHNICAL OVERLAYS
# =============================================================================
with tab_chart:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.60, 0.20, 0.20]
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name="OHLC Price",
            increasing_line_color="#10b981",
            decreasing_line_color="#f43f5e",
            increasing_fillcolor="#10b981",
            decreasing_fillcolor="#f43f5e"
        ),
        row=1, col=1
    )

    # SMA Overlays
    if "sma_20" in df_plot:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["sma_20"], line=dict(color="#fbbf24", width=1.5), name="SMA 20"), row=1, col=1)
    if "sma_50" in df_plot:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["sma_50"], line=dict(color="#38bdf8", width=1.5), name="SMA 50"), row=1, col=1)
    if "sma_200" in df_plot and df_plot["sma_200"].notna().sum() > 5:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["sma_200"], line=dict(color="#c084fc", width=1.8), name="SMA 200"), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in df_plot and "bb_lower" in df_plot:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["bb_upper"], line=dict(color="#64748b", width=1, dash="dot"), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["bb_lower"], line=dict(color="#64748b", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(100, 116, 139, 0.12)", name="BB Lower"), row=1, col=1)

    # 2. Volume Subplot
    vol_colors = ["#10b981" if c >= o else "#f43f5e" for c, o in zip(df_plot["close"], df_plot["open"])]
    fig.add_trace(
        go.Bar(x=df_plot.index, y=df_plot["volume"], marker_color=vol_colors, opacity=0.8, name="Volume"),
        row=2, col=1
    )

    # 3. MACD Subplot
    if "macd" in df_plot and "macd_signal" in df_plot:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["macd"], line=dict(color="#38bdf8", width=1.5), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["macd_signal"], line=dict(color="#fb923c", width=1.5), name="Signal"), row=3, col=1)
        hist_colors = ["#10b981" if h >= 0 else "#f43f5e" for h in df_plot["macd_hist"]]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["macd_hist"], marker_color=hist_colors, name="MACD Hist"), row=3, col=1)

    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        **PLOT_THEME
    )
    fig.update_xaxes(gridcolor="#1e2433", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2433", zeroline=False)

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 2: AI FORECASTING & MONTE CARLO PROJECTIONS
# =============================================================================
with tab_forecast:
    st.markdown("#### Out-of-Sample Machine Learning & Stochastic Projections")
    st.caption("Trained strictly on historical chronological bars. Zero lookahead data leakage.")

    X, y, feature_cols = fe.prepare_modeling_data(raw_df, target_col="close")

    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model_key == "arima":
        forecaster = ARIMABaseline()
        forecaster.fit(y_train)
        preds_test = forecaster.predict(steps=len(y_test))
    else:
        forecaster = TreeEnsembleForecaster(model_type=model_key)
        forecaster.fit(X_train, y_train)
        preds_test = forecaster.predict(X_test)

    preds_series = pd.Series(preds_test, index=y_test.index)
    metrics = ModelEvaluator.calculate_metrics(y_test.values, preds_test)

    # 4 Validation Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""
    <div class="term-card">
        <div class="term-label">MEAN ABSOLUTE ERROR (MAE)</div>
        <div class="term-value">{metrics['mae']:,.0f} <span style="font-size: 0.75rem; color: #94a3b8;">VND</span></div>
        <div class="term-sub">Average Absolute Price Error</div>
    </div>
    """, unsafe_allow_html=True)

    m2.markdown(f"""
    <div class="term-card">
        <div class="term-label">ROOT MEAN SQ. ERROR (RMSE)</div>
        <div class="term-value">{metrics['rmse']:,.0f} <span style="font-size: 0.75rem; color: #94a3b8;">VND</span></div>
        <div class="term-sub">Penalizes High Deviation Sessions</div>
    </div>
    """, unsafe_allow_html=True)

    mape_col = "#10b981" if metrics["mape"] < 3.0 else "#fbbf24" if metrics["mape"] < 5.0 else "#f43f5e"
    m3.markdown(f"""
    <div class="term-card">
        <div class="term-label">PERCENT ERROR (MAPE)</div>
        <div class="term-value" style="color: {mape_col};">{metrics['mape']:.2f}%</div>
        <div class="term-sub">{'Strong Generalization (<3%)' if metrics['mape'] < 3 else 'Acceptable Accuracy'}</div>
    </div>
    """, unsafe_allow_html=True)

    hit_col = "#10b981" if metrics["directional_accuracy"] >= 55.0 else "#94a3b8"
    m4.markdown(f"""
    <div class="term-card">
        <div class="term-label">DIRECTIONAL HIT RATIO</div>
        <div class="term-value" style="color: {hit_col};">{metrics['directional_accuracy']:.1f}%</div>
        <div class="term-sub">Up/Down Movement Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    # Out-of-Sample Chart
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(x=y_train.index[-60:], y=y_train.iloc[-60:], line=dict(color="#64748b", width=1.5), name="Historical Train"))
    fig_eval.add_trace(go.Scatter(x=y_test.index, y=y_test, line=dict(color="#ffffff", width=2.2), name="Actual Price (Ground Truth)"))
    fig_eval.add_trace(go.Scatter(x=y_test.index, y=preds_series, line=dict(color="#38bdf8", width=2.2, dash="dash"), name=f"AI Forecast ({model_choice})"))

    fig_eval.update_layout(
        title="Out-of-Sample Test Set: Predicted vs Ground Truth",
        height=380,
        **PLOT_THEME
    )
    fig_eval.update_xaxes(gridcolor="#1e2433")
    fig_eval.update_yaxes(gridcolor="#1e2433")
    st.plotly_chart(fig_eval, use_container_width=True)

    # Forward Multi-Day Monte Carlo Fan Chart
    st.markdown(f"#### Next {forecast_days} Sessions: Monte Carlo Stochastic Fan Chart")
    st.caption("100 Stochastic Geometric Brownian Motion paths with P10 (Bearish), P50 (Median Expected), and P90 (Bullish) probability cones.")

    curr_features = X.iloc[-1:].copy()
    ml_next_pred = float(forecaster.predict(steps=1)[0] if model_key == "arima" else forecaster.predict(curr_features)[0])
    expected_drift_pct = ((ml_next_pred - latest_close) / latest_close) * 100

    log_returns = raw_df["close"].pct_change().dropna()
    mc_result = QuantEngine.simulate_monte_carlo(
        current_price=latest_close,
        historical_returns=log_returns,
        forecast_days=forecast_days,
        n_simulations=100,
        expected_drift_pct=expected_drift_pct
    )

    future_dates = pd.date_range(start=raw_df.index[-1] + timedelta(days=1), periods=forecast_days * 2, freq="B")[:forecast_days]
    future_dates_all = [raw_df.index[-1]] + list(future_dates)

    col_fc_left, col_fc_right = st.columns([1.2, 2.8])

    with col_fc_left:
        p50_change = ((mc_result["expected_p50_target"] - latest_close) / latest_close) * 100
        p90_change = ((mc_result["bull_p90_target"] - latest_close) / latest_close) * 100
        p10_change = ((mc_result["bear_p10_target"] - latest_close) / latest_close) * 100

        st.markdown(f"""
        <div class="term-card">
            <div class="term-label">STOCHASTIC TARGETS ({forecast_days}D)</div>
            <div style="margin-top: 10px;">
                <div style="font-size: 0.75rem; color: #34d399; font-weight: 700;">P90 (Bullish Upper Band):</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: #34d399; font-weight: 800;">
                    {mc_result['bull_p90_target']:,.0f} VND <span style="font-size: 0.8rem;">(+{p90_change:.1f}%)</span>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 700;">P50 (Median Expected):</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: #38bdf8; font-weight: 800;">
                    {mc_result['expected_p50_target']:,.0f} VND <span style="font-size: 0.8rem;">({'+' if p50_change >= 0 else ''}{p50_change:.1f}%)</span>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <div style="font-size: 0.75rem; color: #fb7185; font-weight: 700;">P10 (Bearish Lower Band):</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; color: #fb7185; font-weight: 800;">
                    {mc_result['bear_p10_target']:,.0f} VND <span style="font-size: 0.8rem;">({p10_change:.1f}%)</span>
                </div>
            </div>
            <div style="margin-top: 14px; font-size: 0.75rem; color: #cbd5e1;">
                Annualized Volatility: <strong style="color: #38bdf8;">{mc_result['annualized_volatility']*100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        db.log_forecast(
            ticker=ticker,
            model_name=model_choice,
            horizon_days=forecast_days,
            current_price=latest_close,
            target_price=mc_result["expected_p50_target"],
            expected_return_pct=p50_change,
            signal="BULLISH" if p50_change > 1.5 else "BEARISH" if p50_change < -1.5 else "NEUTRAL",
            mape=metrics["mape"],
            directional_accuracy=metrics["directional_accuracy"]
        )

    with col_fc_right:
        fig_mc = go.Figure()

        fig_mc.add_trace(go.Scatter(
            x=raw_df.index[-30:],
            y=raw_df["close"].tail(30),
            line=dict(color="#ffffff", width=2.4),
            name="Actual Price"
        ))

        for i in range(min(20, mc_result["paths"].shape[1])):
            fig_mc.add_trace(go.Scatter(
                x=future_dates_all,
                y=mc_result["paths"][:, i],
                line=dict(color="rgba(148, 163, 184, 0.15)", width=1),
                showlegend=False,
                hoverinfo="skip"
            ))

        fig_mc.add_trace(go.Scatter(
            x=future_dates_all,
            y=mc_result["p90"],
            line=dict(color="#10b981", width=2, dash="dot"),
            name="P90 Bull Band"
        ))
        fig_mc.add_trace(go.Scatter(
            x=future_dates_all,
            y=mc_result["p50"],
            line=dict(color="#38bdf8", width=2.8),
            name="P50 Expected Median"
        ))
        fig_mc.add_trace(go.Scatter(
            x=future_dates_all,
            y=mc_result["p10"],
            line=dict(color="#f43f5e", width=2, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.10)",
            name="P10 Bear Band"
        ))

        fig_mc.update_layout(
            height=340,
            **PLOT_THEME
        )
        fig_mc.update_xaxes(gridcolor="#1e2433")
        fig_mc.update_yaxes(gridcolor="#1e2433")
        st.plotly_chart(fig_mc, use_container_width=True)


# =============================================================================
# TAB 3: QUANTITATIVE STRATEGY & UNDERWATER DRAWDOWN
# =============================================================================
with tab_backtest:
    st.markdown("#### Algorithmic Momentum Execution & Portfolio Risk Backtesting")
    st.caption("Simulates execution with a 0.15% fee deduction per roundtrip trade vs Buy & Hold Benchmark.")

    bt = ModelEvaluator.backtest_strategy(
        prices=y_test,
        predicted_prices=preds_series,
        threshold=0.002,
        initial_capital=100_000_000.0,
        fee_rate=0.0015
    )

    strat_ret = bt["total_return_strategy_pct"]
    bench_ret = bt["total_return_benchmark_pct"]

    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(f"""
    <div class="term-card">
        <div class="term-label">AI STRATEGY RETURN</div>
        <div class="term-value" style="color: {'#10b981' if strat_ret >= 0 else '#f43f5e'};">
            {'+' if strat_ret >= 0 else ''}{strat_ret:.2f}%
        </div>
        <div class="term-sub">Final: {bt['final_equity_strategy']:,.0f} VND</div>
    </div>
    """, unsafe_allow_html=True)

    b2.markdown(f"""
    <div class="term-card">
        <div class="term-label">BUY & HOLD BENCHMARK</div>
        <div class="term-value" style="color: {'#10b981' if bench_ret >= 0 else '#f43f5e'};">
            {'+' if bench_ret >= 0 else ''}{bench_ret:.2f}%
        </div>
        <div class="term-sub">Final: {bt['final_equity_benchmark']:,.0f} VND</div>
    </div>
    """, unsafe_allow_html=True)

    b3.markdown(f"""
    <div class="term-card">
        <div class="term-label">ANNUALIZED SHARPE</div>
        <div class="term-value" style="color: {'#10b981' if bt['sharpe_ratio'] >= 1.0 else '#cbd5e1'};">
            {bt['sharpe_ratio']:.2f}
        </div>
        <div class="term-sub">Risk-Adjusted Return Metric</div>
    </div>
    """, unsafe_allow_html=True)

    b4.markdown(f"""
    <div class="term-card">
        <div class="term-label">MAXIMUM DRAWDOWN</div>
        <div class="term-value" style="color: #f43f5e;">
            {bt['max_drawdown_pct']:.2f}%
        </div>
        <div class="term-sub">Peak-to-Trough Capital Decline</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Equity Growth Curve
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=bt["equity_df"].index, y=bt["equity_df"]["Strategy"], line=dict(color="#10b981", width=2.4), name="AI Quantitative Strategy"))
    fig_equity.add_trace(go.Scatter(x=bt["equity_df"].index, y=bt["equity_df"]["Buy_and_Hold"], line=dict(color="#64748b", width=1.8, dash="dash"), name=f"Buy & Hold ({ticker})"))

    fig_equity.update_layout(
        title="Portfolio Cumulative Equity Curve (100,000,000 VND Initial Capital)",
        height=380,
        **PLOT_THEME
    )
    fig_equity.update_xaxes(gridcolor="#1e2433")
    fig_equity.update_yaxes(gridcolor="#1e2433")
    st.plotly_chart(fig_equity, use_container_width=True)

    # 2. Underwater Drawdown Curve
    st.markdown("##### Portfolio Underwater Drawdown Profile")
    st.caption("Illustrates drawdown magnitude and capital recovery periods throughout the test horizon.")

    strat_series = bt["equity_df"]["Strategy"]
    peak = strat_series.cummax()
    drawdown_pct = ((strat_series - peak) / peak) * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown_pct.index,
        y=drawdown_pct,
        line=dict(color="#f43f5e", width=1.6),
        fill="tozeroy",
        fillcolor="rgba(244, 63, 94, 0.20)",
        name="Strategy Drawdown %"
    ))
    fig_dd.update_layout(
        height=220,
        yaxis_title="Drawdown (%)",
        **PLOT_THEME
    )
    fig_dd.update_xaxes(gridcolor="#1e2433")
    fig_dd.update_yaxes(gridcolor="#1e2433")
    st.plotly_chart(fig_dd, use_container_width=True)


# =============================================================================
# TAB 4: SQL WAREHOUSE & FEATURE DRIVERS
# =============================================================================
with tab_db:
    st.markdown("#### SQL Data Warehouse Explorer & Feature Attribution")
    st.caption("Relational database schema status, cached rows, and tree feature importance weights.")

    col_db1, col_db2 = st.columns([1.3, 1.7])

    with col_db1:
        st.markdown("""
        <div class="term-card">
            <div class="term-label">WAREHOUSE ARCHITECTURE</div>
            <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 8px; line-height: 1.7;">
                • <strong>dim_company</strong>: Relational Dimension holding 24+ VN30 companies & sector profiles.<br>
                • <strong>fact_market_ohlcv</strong>: High-frequency daily OHLCV bars with unique <code>(ticker, trade_date)</code> constraints.<br>
                • <strong>fact_forecast_log</strong>: Audit trail recording model predictions, MAPE, and directional accuracy.
            </div>
            <div style="margin-top: 14px;">
                <span class="badge badge-neutral">SQLAlchemy 2.0 ORM</span>
                <span class="badge badge-bull">MSSQL / SQLite Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        df_db_sample = db.get_ohlcv(ticker).tail(5)
        if not df_db_sample.empty:
            st.markdown(f"<div class='term-label' style='margin-top: 10px;'>LATEST 5 CACHED BARS IN SQL ({ticker})</div>", unsafe_allow_html=True)
            st.dataframe(df_db_sample, use_container_width=True)

    with col_db2:
        if hasattr(forecaster, "get_feature_importances"):
            df_imp = forecaster.get_feature_importances(top_n=12)
            if not df_imp.empty:
                fig_imp = go.Figure(go.Bar(
                    x=df_imp["importance"][::-1],
                    y=df_imp["feature"][::-1],
                    orientation="h",
                    marker=dict(color="#38bdf8", opacity=0.85)
                ))
                fig_imp.update_layout(
                    title="Top 12 Quantitative Feature Drivers",
                    xaxis_title="Feature Importance",
                    height=360,
                    **PLOT_THEME
                )
                fig_imp.update_xaxes(gridcolor="#1e2433")
                fig_imp.update_yaxes(gridcolor="#1e2433")
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance not supported for baseline/linear models.")
        else:
            st.info("Selected model architecture does not expose Gini/Gain feature weights.")
