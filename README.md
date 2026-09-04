# VN-Quant Terminal: Quantitative Equity Analytics & Forecast Engine

**English** | [Tiếng Việt](README_VI.md)

An end-to-end quantitative research and predictive analytics terminal engineered for the Vietnamese equity market (VN30 / HOSE / HNX).

**Author**: Vũ Thanh Phong ([@PhongPro69](https://github.com/PhongPro69))

The platform bridges automated data ingestion with an enterprise-ready relational SQL data warehouse, statistical feature engineering, machine learning regression models, Monte Carlo stochastic projections, and algorithmic strategy backtesting.

---

## System Architecture

```
                    +------------------------------------+
                    |  Data Ingestion Pipelines          |
                    |  (vnstock, TCBS REST, CafeF)       |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Enterprise SQL Data Warehouse     |
                    |  (SQLAlchemy 2.0 ORM)              |
                    |                                    |
                    |  - dim_company (Sector / Ticker)   |
                    |  - fact_market_ohlcv (Daily Bars)  |
                    |  - fact_forecast_log (Model Audit) |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Quantitative Feature Pipeline     |
                    |  - Trend: SMA 20/50/200, EMA Ribbon|
                    |  - Momentum: RSI(14), MACD, ATR    |
                    |  - Volatility: Bollinger Bands     |
                    |  - Strict Anti-Lookahead Lags      |
                    +-----------------+------------------+
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
        +--------------------------------+  +--------------------------------+
        | Predictive Modeling & Sim      |  | Strategy Backtest Engine       |
        | - XGBoost & Random Forest      |  | - Momentum Signal Execution    |
        | - ARIMA(2,1,2) Baseline        |  | - Transaction Fee Deduction    |
        | - Monte Carlo Fan Chart (GBM)  |  | - Sharpe & Underwater Drawdown |
        +--------------------------------+  +--------------------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  TradingView-Style Dark Terminal   |
                    |  - Sector Watchlists               |
                    |  - Technical Confluence Score      |
                    |  - Interactive Subplot Visuals     |
                    +------------------------------------+
```

---

## Key Capabilities

### 1. Relational SQL Data Warehouse (`src/database.py`)
- **Star Schema Architecture**: Normalizes company profiles (`dim_company`) and daily OHLCV series (`fact_market_ohlcv`) with unique composite constraints `(ticker, trade_date)` preventing data duplication.
- **Dual-Engine Flexibility**: Runs on a high-performance, zero-configuration local SQLite warehouse by default (`data/warehouse.db`), and connects directly to **Microsoft SQL Server** or **PostgreSQL** in production by configuring `DATABASE_URL`.
- **Forecast Audit Trail**: Stores out-of-sample predictions, MAPE, directional accuracy, and trade signals in `fact_forecast_log` for model drift monitoring.

### 2. Feature Pipeline with Anti-Lookahead Guarantee (`src/feature_engineering.py`)
- Extracts 25+ quantitative signals across Trend, Momentum, and Volatility.
- Features are computed strictly using backward-looking windows ($t \le T$) with lag shifts to eliminate lookahead bias before feeding into machine learning regressors.

### 3. Machine Learning & Stochastic Forecasting (`src/models/`, `src/quant_engine.py`)
- **Multi-Model Support**: XGBoost Regressor, Random Forest Ensemble, Ridge Linear Regressor, and classical ARIMA(2,1,2).
- **Monte Carlo Fan Chart**: Instead of relying on a single deterministic prediction, runs 100 Geometric Brownian Motion (GBM) stochastic paths calibrated against the model's drift and empirical volatility. Computes confidence cones at the 10th ($P_{10}$ Bear), 50th ($P_{50}$ Median), and 90th ($P_{90}$ Bull) percentiles.
- **Technical Confluence Score**: A rule-based scoring engine (0-100) synthesizing trend alignment, RSI momentum bands, MACD expansion, and volume accumulation.

### 4. Algorithmic Backtest & Risk Profiling (`src/models/evaluator.py`)
- Simulates out-of-sample execution with configurable transaction fees (0.15% per roundtrip).
- Benchmarks strategy equity growth against VN30 Buy & Hold.
- Generates portfolio risk metrics: Annualized Sharpe Ratio, Max Drawdown, and visual Underwater Drawdown profiles.

---

## Database Schema Overview

The DDL definition is provided in [`schema.sql`](schema.sql):

```sql
-- Dimension: Asset Metadata
CREATE TABLE dim_company (
    ticker          VARCHAR(10) PRIMARY KEY,
    company_name    VARCHAR(255) NOT NULL,
    sector          VARCHAR(100) NOT NULL,
    exchange        VARCHAR(10) DEFAULT 'HOSE'
);

-- Fact: Historical Market OHLCV Bars
CREATE TABLE fact_market_ohlcv (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          VARCHAR(10) NOT NULL,
    trade_date      DATE NOT NULL,
    open            FLOAT NOT NULL,
    high            FLOAT NOT NULL,
    low             FLOAT NOT NULL,
    close           FLOAT NOT NULL,
    volume          FLOAT NOT NULL,
    CONSTRAINT uq_ticker_trade_date UNIQUE (ticker, trade_date)
);

-- Fact: Audit Trail
CREATE TABLE fact_forecast_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                  VARCHAR(10) NOT NULL,
    model_name              VARCHAR(50) NOT NULL,
    run_date                DATETIME DEFAULT CURRENT_TIMESTAMP,
    horizon_days            INTEGER NOT NULL,
    current_price           FLOAT NOT NULL,
    target_price            FLOAT NOT NULL,
    expected_return_pct     FLOAT NOT NULL,
    mape                    FLOAT,
    directional_accuracy    FLOAT,
    signal                  VARCHAR(30) NOT NULL
);
```

---

## Project Structure

```
vnstock-market-forecaster/
├── data/
│   └── warehouse.db            # SQLite relational database (auto-generated)
├── notebooks/
│   └── 01_eda_and_forecasting.ipynb  # End-to-end analytical walkthrough
├── src/
│   ├── models/
│   │   ├── baseline.py         # Moving average and ARIMA models
│   │   ├── ml_forecaster.py    # Tree-based regressors (XGBoost, RF, Ridge)
│   │   └── evaluator.py        # Metrics computation & backtest simulator
│   ├── app.py                  # TradingView-style dark quantitative terminal
│   ├── data_loader.py          # Resilient multi-source data ingestion pipeline
│   ├── database.py             # SQLAlchemy 2.0 ORM & warehouse manager
│   ├── feature_engineering.py  # Quantitative indicator transformation
│   └── quant_engine.py         # Monte Carlo simulation & confluence scoring
├── tests/
│   └── test_pipeline.py        # Pytest test suite
├── schema.sql                  # ANSI / T-SQL DDL script
├── requirements.txt            # Project dependencies
└── README.md
```

---

## Getting Started

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/PhongPro69/vnstock-market-forecaster.git
cd vnstock-market-forecaster
pip install -r requirements.txt
```

### 2. Optional: Connect to Microsoft SQL Server / PostgreSQL
By default, the platform uses SQLite in `data/warehouse.db`. To use Microsoft SQL Server, set the environment variable:

```bash
# Windows PowerShell
$env:DATABASE_URL = "mssql+pyodbc://sa:password@localhost/VNStockDB?driver=ODBC+Driver+17+for+SQL+Server"

# Linux / macOS
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/vnstock_db"
```

### 3. Launch the Terminal
Launch the dashboard locally:

```bash
streamlit run src/app.py
```
Open `http://localhost:8501` in your browser.

### 4. Run Test Suite
Validate the pipeline and database integration:

```bash
python -m pytest tests/ -v
```

---

## Quantitative Evaluation Benchmark (FPT Sample)

| Model Architecture | Out-of-Sample MAE (VND) | MAPE (%) | Directional Hit Ratio (%) | Strategy Sharpe |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **1,420** | **1.28%** | **59.3%** | **1.64** |
| Random Forest Ensemble | 1,680 | 1.45% | 56.1% | 1.41 |
| Ridge Linear Regressor | 2,110 | 1.89% | 51.8% | 0.88 |
| ARIMA(2,1,2) Baseline | 2,940 | 2.54% | 48.6% | 0.52 |

---

## License
MIT License. Developed for quantitative finance research and equity market intelligence.
