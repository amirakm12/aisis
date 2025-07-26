"""
Advanced RAG System with AI Agents & Vector Databases

This package provides a comprehensive RAG (Retrieval-Augmented Generation) system
with advanced features including:
- Multiple vector database support (ChromaDB, Pinecone, Weaviate, Qdrant, etc.)
- AI Agents for intelligent document processing and query routing
- Advanced embedding strategies and reranking
- Multi-modal document processing
- Real-time indexing and updates
- Distributed processing capabilities
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