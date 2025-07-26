"""
Advanced RAG System with AI Agents & Vector Databases

A comprehensive Retrieval-Augmented Generation (RAG) system that combines
AI agents for intelligent processing with multiple vector database backends
for efficient information retrieval and generation.

Features:
- Multi-agent architecture for query processing and document analysis
- Support for multiple vector databases (ChromaDB, Pinecone, Weaviate, etc.)
- Hybrid retrieval combining semantic and keyword search
- Advanced document processing with multi-format support
- Intelligent caching and performance optimization
- Comprehensive monitoring and metrics collection
- Flexible configuration management
"""

__version__ = "1.0.0"
__author__ = "Advanced RAG System"

from .core.rag_engine import AdvancedRAGEngine
from .agents.query_agent import QueryAgent
from .agents.document_agent import DocumentAgent
from .agents.orchestrator_agent import OrchestratorAgent
from .vector_stores.vector_store_manager import VectorStoreManager
from .embeddings.embedding_manager import EmbeddingManager
from .retrievers.hybrid_retriever import HybridRetriever
from .processors.document_processor import DocumentProcessor
from .utils.config import RAGConfig

__all__ = [
    "AdvancedRAGEngine",
    "QueryAgent",
    "DocumentAgent",
    "OrchestratorAgent",
    "VectorStoreManager",
    "EmbeddingManager",
    "HybridRetriever",
    "DocumentProcessor",
    "RAGConfig"
] 