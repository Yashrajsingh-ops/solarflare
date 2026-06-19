"""database.py — SQLAlchemy models and DB session setup."""

import os
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

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

    insight = relationship("PredictionInsight", back_populates="prediction_log", uselist=False)


class PredictionInsight(Base):
    """Explainability and impact metadata for a prediction log."""

    __tablename__ = "prediction_insights"

    id = Column(Integer, primary_key=True, index=True)
    prediction_log_id = Column(Integer, ForeignKey("prediction_logs.id"), unique=True, nullable=False)
    flare_class = Column(String(2), nullable=False)
    model_name = Column(String(32), nullable=False, default="heuristic")
    reasons_json = Column(Text, nullable=False)
    impact_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction_log = relationship("PredictionLog", back_populates="insight")


class FlareEvent(Base):
    """Historical flare event catalog used by the history and analytics views."""

    __tablename__ = "flare_events"

    id = Column(Integer, primary_key=True, index=True)
    prediction_log_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=False, index=True)
    flare_class = Column(String(2), nullable=False)
    flare_probability = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    event_time = Column(DateTime, default=datetime.utcnow, index=True)
    summary = Column(String(256), nullable=False)


class AlertHistory(Base):
    """Persisted alert records for the alert center and notification feed."""

    __tablename__ = "alert_history"

    id = Column(Integer, primary_key=True, index=True)
    prediction_log_id = Column(Integer, ForeignKey("prediction_logs.id"), nullable=True, index=True)
    alert_level = Column(String(16), nullable=False)
    message = Column(String(256), nullable=False)
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class UserActivity(Base):
    """Very small audit trail for uploads and prediction requests."""

    __tablename__ = "user_activity"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(32), nullable=False)
    entity_type = Column(String(32), nullable=False)
    entity_id = Column(String(64), nullable=True)
    details_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


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
