"""
Configuration management for Advanced RAG System
"""

import os
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    ELASTICSEARCH = "elasticsearch"
    FAISS = "faiss"


class EmbeddingType(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    COHERE = "cohere"


class VectorStoreConfig(BaseModel):
    type: VectorStoreType = VectorStoreType.CHROMA
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "documents"
    dimension: int = 768
    distance_metric: str = "cosine"
    api_key: Optional[str] = None
    environment: Optional[str] = None
    index_name: Optional[str] = None
    namespace: Optional[str] = None


class EmbeddingConfig(BaseModel):
    type: EmbeddingType = EmbeddingType.SENTENCE_TRANSFORMERS
    model_name: str = "all-MiniLM-L6-v2"
    api_key: Optional[str] = None
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30


class RetrieverConfig(BaseModel):
    top_k: int = 10
    similarity_threshold: float = 0.7
    use_reranking: bool = True
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_weight: float = 0.5
    max_context_length: int = 4000


class AgentConfig(BaseModel):
    enable_query_agent: bool = True
    enable_document_agent: bool = True
    enable_orchestrator: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30


class ProcessingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    preserve_structure: bool = True
    extract_metadata: bool = True
    enable_ocr: bool = False


class CacheConfig(BaseModel):
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None


class MonitoringConfig(BaseModel):
    enable_metrics: bool = True
    metrics_retention_days: int = 30
    log_level: str = "INFO"
    enable_performance_tracking: bool = True


class RAGConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # Vector Store
        if os.getenv("VECTOR_STORE_TYPE"):
            config.vector_store.type = VectorStoreType(os.getenv("VECTOR_STORE_TYPE"))
        if os.getenv("VECTOR_STORE_HOST"):
            config.vector_store.host = os.getenv("VECTOR_STORE_HOST")
        if os.getenv("VECTOR_STORE_PORT"):
            config.vector_store.port = int(os.getenv("VECTOR_STORE_PORT"))
        if os.getenv("VECTOR_STORE_API_KEY"):
            config.vector_store.api_key = os.getenv("VECTOR_STORE_API_KEY")
        
        # Embedding
        if os.getenv("EMBEDDING_TYPE"):
            config.embedding.type = EmbeddingType(os.getenv("EMBEDDING_TYPE"))
        if os.getenv("EMBEDDING_MODEL"):
            config.embedding.model_name = os.getenv("EMBEDDING_MODEL")
        if os.getenv("EMBEDDING_API_KEY"):
            config.embedding.api_key = os.getenv("EMBEDDING_API_KEY")
        
        # LLM
        if os.getenv("LLM_PROVIDER"):
            config.llm.provider = os.getenv("LLM_PROVIDER")
        if os.getenv("LLM_MODEL"):
            config.llm.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_API_KEY"):
            config.llm.api_key = os.getenv("LLM_API_KEY")
        
        # Cache
        if os.getenv("REDIS_HOST"):
            config.cache.redis_host = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            config.cache.redis_port = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_PASSWORD"):
            config.cache.redis_password = os.getenv("REDIS_PASSWORD")
        
        return config

    @classmethod
    def from_file(cls, file_path: str) -> "RAGConfig":
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.model_dump()

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined configurations for different vector stores
VECTOR_STORE_CONFIGS = {
    VectorStoreType.CHROMA: {
        "host": "localhost",
        "port": 8000,
        "collection_name": "documents"
    },
    VectorStoreType.PINECONE: {
        "host": "api.pinecone.io",
        "port": 443,
        "environment": "us-west1-gcp"
    },
    VectorStoreType.WEAVIATE: {
        "host": "localhost",
        "port": 8080,
        "collection_name": "Documents"
    },
    VectorStoreType.QDRANT: {
        "host": "localhost",
        "port": 6333,
        "collection_name": "documents"
    },
    VectorStoreType.MILVUS: {
        "host": "localhost",
        "port": 19530,
        "collection_name": "documents"
    },
    VectorStoreType.ELASTICSEARCH: {
        "host": "localhost",
        "port": 9200,
        "index_name": "documents"
    }
} 