"""
Configuration management for Advanced RAG System
"""

import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class VectorStoreConfig(BaseModel):
    """Configuration for vector stores"""
    provider: str = Field(default="chromadb", description="Vector store provider")
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    collection_name: str = Field(default="rag_documents")
    embedding_dimension: int = Field(default=1536)
    similarity_metric: str = Field(default="cosine")

class EmbeddingConfig(BaseModel):
    """Configuration for embeddings"""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    model_type: str = Field(default="sentence_transformers")
    batch_size: int = Field(default=32)
    max_length: int = Field(default=512)
    normalize_embeddings: bool = Field(default=True)

class LLMConfig(BaseModel):
    """Configuration for Language Models"""
    provider: str = Field(default="openai")
    model_name: str = Field(default="gpt-3.5-turbo")
    api_key: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    timeout: int = Field(default=30)

class RetrieverConfig(BaseModel):
    """Configuration for retrievers"""
    top_k: int = Field(default=10)
    similarity_threshold: float = Field(default=0.7)
    enable_reranking: bool = Field(default=True)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    hybrid_search_alpha: float = Field(default=0.7)

class AgentConfig(BaseModel):
    """Configuration for AI Agents"""
    enable_query_expansion: bool = Field(default=True)
    enable_query_routing: bool = Field(default=True)
    enable_document_analysis: bool = Field(default=True)
    max_iterations: int = Field(default=5)
    confidence_threshold: float = Field(default=0.8)

class ProcessingConfig(BaseModel):
    """Configuration for document processing"""
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    enable_ocr: bool = Field(default=True)
    supported_formats: List[str] = Field(
        default=["pdf", "docx", "txt", "html", "md", "json"]
    )
    enable_multimodal: bool = Field(default=True)

class CacheConfig(BaseModel):
    """Configuration for caching"""
    enable_cache: bool = Field(default=True)
    cache_provider: str = Field(default="redis")
    cache_ttl: int = Field(default=3600)
    connection_params: Dict[str, Any] = Field(default_factory=dict)

class MonitoringConfig(BaseModel):
    """Configuration for monitoring and logging"""
    log_level: str = Field(default="INFO")
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=8080)
    enable_tracing: bool = Field(default=True)

class RAGConfig(BaseModel):
    """Main configuration class for Advanced RAG System"""
    
    # Core configurations
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # System configurations
    enable_async: bool = Field(default=True)
    max_concurrent_requests: int = Field(default=10)
    request_timeout: int = Field(default=60)
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables"""
        config = cls()
        
        # Vector Store Configuration
        if os.getenv("VECTOR_STORE_PROVIDER"):
            config.vector_store.provider = os.getenv("VECTOR_STORE_PROVIDER")
        
        if os.getenv("VECTOR_STORE_COLLECTION"):
            config.vector_store.collection_name = os.getenv("VECTOR_STORE_COLLECTION")
            
        # LLM Configuration
        if os.getenv("LLM_PROVIDER"):
            config.llm.provider = os.getenv("LLM_PROVIDER")
            
        if os.getenv("LLM_MODEL"):
            config.llm.model_name = os.getenv("LLM_MODEL")
            
        if os.getenv("OPENAI_API_KEY"):
            config.llm.api_key = os.getenv("OPENAI_API_KEY")
            
        # Embedding Configuration
        if os.getenv("EMBEDDING_MODEL"):
            config.embedding.model_name = os.getenv("EMBEDDING_MODEL")
            
        # Cache Configuration
        if os.getenv("REDIS_URL"):
            config.cache.connection_params["url"] = os.getenv("REDIS_URL")
            
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> "RAGConfig":
        """Load configuration from JSON file"""
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.dict()
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        import json
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Default configuration instance
default_config = RAGConfig.from_env()

# Vector store specific configurations
VECTOR_STORE_CONFIGS = {
    "chromadb": {
        "persist_directory": "./chroma_db",
        "client_settings": {}
    },
    "pinecone": {
        "api_key": os.getenv("PINECONE_API_KEY"),
        "environment": os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    },
    "weaviate": {
        "url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        "api_key": os.getenv("WEAVIATE_API_KEY")
    },
    "qdrant": {
        "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "api_key": os.getenv("QDRANT_API_KEY")
    },
    "milvus": {
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "port": int(os.getenv("MILVUS_PORT", "19530"))
    },
    "elasticsearch": {
        "hosts": [os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")],
        "api_key": os.getenv("ELASTICSEARCH_API_KEY")
    }
}