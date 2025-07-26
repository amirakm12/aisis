from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from typing import Generator
import redis
from app.config import settings

# SQLAlchemy setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.from_url(settings.redis_url, decode_responses=True)


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    content_hash = Column(String, unique=True, index=True)
    content_preview = Column(Text)
    metadata = Column(Text)  # JSON string
    upload_time = Column(DateTime, default=func.now())
    last_accessed = Column(DateTime, default=func.now())
    access_count = Column(Integer, default=0)
    is_processed = Column(Boolean, default=False)
    vector_store_id = Column(String, index=True)  # ChromaDB collection ID


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String, index=True)
    embedding_model = Column(String, nullable=False)
    chunk_metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=func.now())


class SearchQuery(Base):
    __tablename__ = "search_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    query_hash = Column(String, index=True)
    results_count = Column(Integer, default=0)
    response_time = Column(Float)  # in seconds
    timestamp = Column(DateTime, default=func.now())
    user_feedback = Column(Integer)  # 1-5 rating


class RAGSession(Base):
    __tablename__ = "rag_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    conversation_history = Column(Text)  # JSON string
    context_documents = Column(Text)  # JSON string of document IDs
    created_at = Column(DateTime, default=func.now())
    last_interaction = Column(DateTime, default=func.now())
    total_queries = Column(Integer, default=0)


# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Redis utilities
class CacheManager:
    @staticmethod
    def set_cache(key: str, value: str, expire: int = 3600):
        """Set cache with expiration time in seconds"""
        redis_client.setex(key, expire, value)
    
    @staticmethod
    def get_cache(key: str) -> str:
        """Get cache value"""
        return redis_client.get(key)
    
    @staticmethod
    def delete_cache(key: str):
        """Delete cache key"""
        redis_client.delete(key)
    
    @staticmethod
    def exists(key: str) -> bool:
        """Check if key exists in cache"""
        return redis_client.exists(key)


# Initialize database
create_tables()