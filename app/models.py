from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/baby.db")
if DATABASE_URL.startswith("sqlite:///"):
    os.makedirs(os.path.dirname(DATABASE_URL[len("sqlite:///"):]) or ".", exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    type = Column(String, index=True)
    data = Column(JSON)
    source = Column(String, default="manual")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    source_path = Column(String)
    content = Column(Text)
    ingested_at = Column(DateTime, default=datetime.utcnow)

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True)
    content = Column(Text)
    page = Column(Integer, default=1)
    embedding = Column(JSON)

class BabyProfile(Base):
    """One row (id=1): name, birth date/time, birth measurements, free-text notes."""
    __tablename__ = "baby_profile"
    id = Column(Integer, primary_key=True)
    data = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Immunizations are stored as events with type "immunization"; the event timestamp is the date given.

Base.metadata.create_all(bind=engine)
