"""
database.py — SQLAlchemy models and DB session setup.
Uses SQLite by default; swap DATABASE_URL env var for PostgreSQL.
"""

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./solar_flare.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class SolarObservation(Base):
    """Raw observations ingested from uploaded CSV files."""

    __tablename__ = "solar_observations"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    soft_xray_flux = Column(Float, nullable=False)
    hard_xray_flux = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """Audit log of every prediction request."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    soft_xray_flux = Column(Float, nullable=False)
    hard_xray_flux = Column(Float, nullable=False)
    flare_probability = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    prediction = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
