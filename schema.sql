-- ============================================================================
-- Quantitative Equity Intelligence Platform - Database Schema
-- Compatible with Microsoft SQL Server (T-SQL), PostgreSQL, and SQLite
-- ============================================================================

-- 1. Dimension: Company & Asset Metadata
CREATE TABLE IF NOT EXISTS dim_company (
    ticker          VARCHAR(10)     NOT NULL PRIMARY KEY,
    company_name    VARCHAR(255)    NOT NULL,
    sector          VARCHAR(100)    NOT NULL,
    exchange        VARCHAR(10)     NOT NULL DEFAULT 'HOSE',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dim_company_sector ON dim_company(sector);

-- 2. Fact: Historical Daily Market OHLCV Bars
CREATE TABLE IF NOT EXISTS fact_market_ohlcv (
    id              INTEGER         PRIMARY KEY AUTOINCREMENT,
    ticker          VARCHAR(10)     NOT NULL,
    trade_date      DATE            NOT NULL,
    open            FLOAT           NOT NULL,
    high            FLOAT           NOT NULL,
    low             FLOAT           NOT NULL,
    close           FLOAT           NOT NULL,
    volume          FLOAT           NOT NULL,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_ohlcv_company FOREIGN KEY (ticker) REFERENCES dim_company(ticker) ON DELETE CASCADE,
    CONSTRAINT uq_ticker_trade_date UNIQUE (ticker, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date ON fact_market_ohlcv(ticker, trade_date);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON fact_market_ohlcv(trade_date);

-- 3. Fact: Predictive Model Audit & Evaluation Logs
CREATE TABLE IF NOT EXISTS fact_forecast_log (
    id                      INTEGER         PRIMARY KEY AUTOINCREMENT,
    ticker                  VARCHAR(10)     NOT NULL,
    model_name              VARCHAR(50)     NOT NULL,
    run_date                DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    horizon_days            INTEGER         NOT NULL,
    current_price           FLOAT           NOT NULL,
    target_price            FLOAT           NOT NULL,
    expected_return_pct     FLOAT           NOT NULL,
    mape                    FLOAT           NULL,
    directional_accuracy    FLOAT           NULL,
    signal                  VARCHAR(30)     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forecast_ticker ON fact_forecast_log(ticker);
CREATE INDEX IF NOT EXISTS idx_forecast_run_date ON fact_forecast_log(run_date);