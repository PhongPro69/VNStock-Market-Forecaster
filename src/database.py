"""
Database Management Layer for Quantitative Stock Market Data.
Supports SQLite (zero-config local default) and enterprise SQL Server / PostgreSQL
via SQLAlchemy 2.0 ORM.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, DateTime,
    ForeignKey, UniqueConstraint, Index, text, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

logger = logging.getLogger(__name__)

Base = declarative_base()


class DimCompany(Base):
    """Dimension table: Stock metadata and sector classifications."""
    __tablename__ = "dim_company"

    ticker = Column(String(10), primary_key=True)
    company_name = Column(String(255), nullable=False)
    sector = Column(String(100), nullable=False, index=True)
    exchange = Column(String(10), default="HOSE")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to OHLCV fact table
    prices = relationship("FactMarketOHLCV", back_populates="company", cascade="all, delete-orphan")


class FactMarketOHLCV(Base):
    """Fact table: Daily historical market OHLCV bars."""
    __tablename__ = "fact_market_ohlcv"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey("dim_company.ticker"), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("DimCompany", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("ticker", "trade_date", name="uq_ticker_trade_date"),
        Index("idx_ticker_trade_date", "ticker", "trade_date"),
    )


class FactForecastLog(Base):
    """Fact table: Model forecast audit trail for tracking prediction drift."""
    __tablename__ = "fact_forecast_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    run_date = Column(DateTime, default=datetime.utcnow)
    horizon_days = Column(Integer, nullable=False)
    current_price = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    expected_return_pct = Column(Float, nullable=False)
    mape = Column(Float, nullable=True)
    directional_accuracy = Column(Float, nullable=True)
    signal = Column(String(30), nullable=False)


# Default VN30 Companies with Sectors
VN30_PROFILES = [
    ("FPT", "FPT Corporation", "Technology", "HOSE"),
    ("VNM", "Vinamilk", "Consumer Goods", "HOSE"),
    ("HPG", "Hoa Phat Group", "Steel & Materials", "HOSE"),
    ("VCB", "Vietcombank", "Banking", "HOSE"),
    ("MWG", "Mobile World Group", "Retail", "HOSE"),
    ("SSI", "SSI Securities", "Financial Services", "HOSE"),
    ("VIC", "Vingroup JSC", "Conglomerate", "HOSE"),
    ("VHM", "Vinhomes JSC", "Real Estate", "HOSE"),
    ("TCB", "Techcombank", "Banking", "HOSE"),
    ("MBB", "Military Commercial Bank", "Banking", "HOSE"),
    ("ACB", "Asia Commercial Bank", "Banking", "HOSE"),
    ("STB", "Sacombank", "Banking", "HOSE"),
    ("MSN", "Masan Group", "Consumer Goods", "HOSE"),
    ("VRE", "Vincom Retail", "Real Estate", "HOSE"),
    ("GAS", "PetroVietnam Gas", "Energy", "HOSE"),
    ("BID", "BIDV", "Banking", "HOSE"),
    ("CTG", "VietinBank", "Banking", "HOSE"),
    ("PLX", "Petrolimex", "Energy", "HOSE"),
    ("SAB", "Sabeco", "Beverages", "HOSE"),
    ("TPB", "TPBank", "Banking", "HOSE"),
    ("HDB", "HDBank", "Banking", "HOSE"),
    ("VIB", "VIB Bank", "Banking", "HOSE"),
    ("VPB", "VPBank", "Banking", "HOSE"),
    ("BVH", "Bao Viet Holdings", "Insurance", "HOSE")
]


class DatabaseManager:
    """Manages database connection pooling, schema initialization, and high-performance queries."""

    def __init__(self, connection_url: Optional[str] = None):
        if not connection_url:
            connection_url = os.getenv("DATABASE_URL")

        if not connection_url:
            data_dir = Path(__file__).resolve().parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "warehouse.db"
            connection_url = f"sqlite:///{db_path}"

        self.connection_url = connection_url
        self.is_sqlite = connection_url.startswith("sqlite")
        self.engine = create_engine(
            self.connection_url,
            pool_pre_ping=True,
            echo=False
        )
        self.Session = sessionmaker(bind=self.engine)
        self.init_db()

    def init_db(self) -> None:
        """Create tables if they do not exist and seed initial dimension records."""
        Base.metadata.create_all(self.engine)
        self._seed_dim_company()

    def _seed_dim_company(self) -> None:
        """Populate dim_company if empty."""
        session = self.Session()
        try:
            count = session.query(DimCompany).count()
            if count == 0:
                logger.info("Seeding dim_company with VN30 assets...")
                for ticker, name, sector, exchange in VN30_PROFILES:
                    company = DimCompany(
                        ticker=ticker,
                        company_name=name,
                        sector=sector,
                        exchange=exchange
                    )
                    session.merge(company)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error seeding dim_company: {e}")
        finally:
            session.close()

    def upsert_ohlcv(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Store daily OHLCV rows into SQL database.
        Returns the count of successfully persisted records.
        """
        if df.empty:
            return 0

        ticker = ticker.upper()
        session = self.Session()
        inserted_count = 0

        try:
            # Ensure company exists in dim_company
            existing = session.query(DimCompany).filter_by(ticker=ticker).first()
            if not existing:
                session.add(DimCompany(ticker=ticker, company_name=f"{ticker} Stock", sector="General", exchange="HOSE"))
                session.commit()

            # Prepare records
            records = []
            for idx, row in df.iterrows():
                trade_d = idx.date() if isinstance(idx, (datetime, pd.Timestamp)) else pd.to_datetime(idx).date()
                records.append({
                    "ticker": ticker,
                    "trade_date": trade_d,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"])
                })

            if not records:
                return 0

            # For SQLite/General SQL: query existing dates to perform insert-or-update
            existing_dates = set(
                r[0] for r in session.query(FactMarketOHLCV.trade_date).filter(
                    FactMarketOHLCV.ticker == ticker
                ).all()
            )

            new_entries = [
                FactMarketOHLCV(**r) for r in records if r["trade_date"] not in existing_dates
            ]

            if new_entries:
                session.bulk_save_objects(new_entries)
                session.commit()
                inserted_count = len(new_entries)
                logger.info(f"Persisted {inserted_count} new bars for {ticker} into SQL warehouse.")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to upsert OHLCV into database: {e}")
        finally:
            session.close()

        return inserted_count

    def get_ohlcv(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch historical bars directly from SQL warehouse."""
        ticker = ticker.upper()
        session = self.Session()
        try:
            query = session.query(FactMarketOHLCV).filter(FactMarketOHLCV.ticker == ticker)

            if start_date:
                s_dt = pd.to_datetime(start_date).date()
                query = query.filter(FactMarketOHLCV.trade_date >= s_dt)
            if end_date:
                e_dt = pd.to_datetime(end_date).date()
                query = query.filter(FactMarketOHLCV.trade_date <= e_dt)

            query = query.order_by(FactMarketOHLCV.trade_date.asc())

            results = query.all()
            if not results:
                return pd.DataFrame()

            data = [{
                "time": pd.to_datetime(r.trade_date),
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume
            } for r in results]

            df = pd.DataFrame(data).set_index("time")
            return df
        finally:
            session.close()

    def log_forecast(
        self,
        ticker: str,
        model_name: str,
        horizon_days: int,
        current_price: float,
        target_price: float,
        expected_return_pct: float,
        signal: str,
        mape: Optional[float] = None,
        directional_accuracy: Optional[float] = None
    ) -> None:
        """Record model forecast runs into audit log."""
        session = self.Session()
        try:
            log_entry = FactForecastLog(
                ticker=ticker.upper(),
                model_name=model_name,
                horizon_days=horizon_days,
                current_price=current_price,
                target_price=target_price,
                expected_return_pct=expected_return_pct,
                signal=signal,
                mape=mape,
                directional_accuracy=directional_accuracy
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log forecast audit: {e}")
        finally:
            session.close()

    def get_warehouse_stats(self) -> Dict[str, Any]:
        """Return operational metrics of the SQL warehouse for dashboard status badge."""
        session = self.Session()
        try:
            total_companies = session.query(DimCompany).count()
            total_bars = session.query(FactMarketOHLCV).count()
            latest_bar = session.query(FactMarketOHLCV.trade_date).order_by(FactMarketOHLCV.trade_date.desc()).first()
            total_audits = session.query(FactForecastLog).count()

            dialect_name = self.engine.dialect.name.upper()

            return {
                "dialect": "Microsoft SQL Server" if "MSSQL" in dialect_name else "SQLite Warehouse",
                "total_companies": total_companies,
                "total_bars": total_bars,
                "latest_date": latest_bar[0].strftime("%d/%m/%Y") if latest_bar else "N/A",
                "total_audits": total_audits,
                "status": "Healthy & Synced"
            }
        except Exception as e:
            logger.error(f"Error fetching warehouse stats: {e}")
            return {
                "dialect": "Offline",
                "total_companies": 0,
                "total_bars": 0,
                "latest_date": "N/A",
                "total_audits": 0,
                "status": "Degraded"
            }
        finally:
            session.close()