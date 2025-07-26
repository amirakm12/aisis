"""
Advanced RAG Engine - Main orchestrator for the RAG system
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from loguru import logger

from ..utils.config import RAGConfig
from ..agents.orchestrator_agent import OrchestratorAgent
from ..agents.query_agent import QueryAgent
from ..agents.document_agent import DocumentAgent
from ..vector_stores.vector_store_manager import VectorStoreManager
from ..embeddings.embedding_manager import EmbeddingManager
from ..retrievers.hybrid_retriever import HybridRetriever
from ..processors.document_processor import DocumentProcessor
from ..utils.cache_manager import CacheManager
from ..utils.metrics_collector import MetricsCollector

@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_analysis: Dict[str, Any]
    retrieval_metadata: Dict[str, Any]
    processing_time: float

@dataclass
class DocumentIngestionResult:
    """Result from document ingestion"""
    success: bool
    document_id: str
    chunks_created: int
    embeddings_created: int
    metadata: Dict[str, Any]
    processing_time: float

class AdvancedRAGEngine:
    """
    Advanced RAG Engine with AI Agents and Vector Databases
    
    This is the main orchestrator that coordinates all components:
    - AI Agents for intelligent processing
    - Multiple vector database support
    - Advanced retrieval strategies
    - Multi-modal document processing
    - Real-time optimization
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the Advanced RAG Engine"""
        self.config = config or RAGConfig.from_env()
        self.is_initialized = False
        
        # Core components
        self.embedding_manager = None
        self.vector_store_manager = None
        self.document_processor = None
        self.hybrid_retriever = None
        
        # AI Agents
        self.orchestrator_agent = None
        self.query_agent = None
        self.document_agent = None
        
        # Utilities
        self.cache_manager = None
        self.metrics_collector = None
        
        logger.info("Advanced RAG Engine initialized with configuration")
    
    async def initialize(self):
        """Initialize all components asynchronously"""
        if self.is_initialized:
            return
            
        logger.info("Initializing Advanced RAG Engine components...")
        
        try:
            # Initialize core components
            self.embedding_manager = EmbeddingManager(self.config.embedding)
            await self.embedding_manager.initialize()
            
            self.vector_store_manager = VectorStoreManager(
                self.config.vector_store,
                self.embedding_manager
            )
            await self.vector_store_manager.initialize()
            
            self.document_processor = DocumentProcessor(self.config.processing)
            await self.document_processor.initialize()
            
            self.hybrid_retriever = HybridRetriever(
                self.config.retriever,
                self.vector_store_manager,
                self.embedding_manager
            )
            
            # Initialize AI Agents
            self.query_agent = QueryAgent(
                self.config.agent,
                self.config.llm,
                self.embedding_manager
            )
            await self.query_agent.initialize()
            
            self.document_agent = DocumentAgent(
                self.config.agent,
                self.config.llm,
                self.document_processor
            )
            await self.document_agent.initialize()
            
            self.orchestrator_agent = OrchestratorAgent(
                self.config.agent,
                self.config.llm,
                self.query_agent,
                self.document_agent,
                self.hybrid_retriever
            )
            await self.orchestrator_agent.initialize()
            
            # Initialize utilities
            if self.config.cache.enable_cache:
                self.cache_manager = CacheManager(self.config.cache)
                await self.cache_manager.initialize()
            
            if self.config.monitoring.enable_metrics:
                self.metrics_collector = MetricsCollector(self.config.monitoring)
                await self.metrics_collector.initialize()
            
            self.is_initialized = True
            logger.info("Advanced RAG Engine successfully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Engine: {str(e)}")
            raise
    
    async def query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Process a query through the advanced RAG system
        
        Args:
            query: The user query
            context: Additional context for the query
            filters: Filters to apply during retrieval
            
        Returns:
            RAGResponse with answer and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Check cache first
            if self.cache_manager:
                cached_response = await self.cache_manager.get_cached_response(
                    query, context, filters
                )
                if cached_response:
                    logger.info("Returning cached response")
                    return cached_response
            
            # Use orchestrator agent to process the query
            response = await self.orchestrator_agent.process_query(
                query=query,
                context=context or {},
                filters=filters or {}
            )
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Create RAG response
            rag_response = RAGResponse(
                answer=response["answer"],
                sources=response["sources"],
                confidence=response["confidence"],
                query_analysis=response["query_analysis"],
                retrieval_metadata=response["retrieval_metadata"],
                processing_time=processing_time
            )
            
            # Cache the response
            if self.cache_manager:
                await self.cache_manager.cache_response(
                    query, context, filters, rag_response
                )
            
            # Collect metrics
            if self.metrics_collector:
                await self.metrics_collector.record_query_metrics(
                    query, rag_response, processing_time
                )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return rag_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def ingest_document(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> DocumentIngestionResult:
        """
        Ingest a document into the RAG system
        
        Args:
            document_path: Path to the document
            metadata: Additional metadata for the document
            processing_options: Options for document processing
            
        Returns:
            DocumentIngestionResult with ingestion details
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use document agent to process the document
            result = await self.document_agent.process_document(
                document_path=document_path,
                metadata=metadata or {},
                options=processing_options or {}
            )
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Create ingestion result
            ingestion_result = DocumentIngestionResult(
                success=result["success"],
                document_id=result["document_id"],
                chunks_created=result["chunks_created"],
                embeddings_created=result["embeddings_created"],
                metadata=result["metadata"],
                processing_time=processing_time
            )
            
            # Collect metrics
            if self.metrics_collector:
                await self.metrics_collector.record_ingestion_metrics(
                    document_path, ingestion_result, processing_time
                )
            
            logger.info(f"Document ingested successfully in {processing_time:.2f}s")
            return ingestion_result
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            raise
    
    async def ingest_documents_batch(
        self,
        document_paths: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        processing_options: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 5
    ) -> List[DocumentIngestionResult]:
        """
        Ingest multiple documents concurrently
        
        Args:
            document_paths: List of document paths
            metadata_list: List of metadata for each document
            processing_options: Options for document processing
            max_concurrent: Maximum concurrent ingestions
            
        Returns:
            List of DocumentIngestionResult
        """
        if not self.is_initialized:
            await self.initialize()
        
        metadata_list = metadata_list or [{}] * len(document_paths)
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def ingest_single(doc_path: str, metadata: Dict[str, Any]):
            async with semaphore:
                return await self.ingest_document(
                    doc_path, metadata, processing_options
                )
        
        # Execute concurrent ingestions
        tasks = [
            ingest_single(doc_path, metadata)
            for doc_path, metadata in zip(document_paths, metadata_list)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to ingest {document_paths[i]}: {str(result)}")
                final_results.append(DocumentIngestionResult(
                    success=False,
                    document_id="",
                    chunks_created=0,
                    embeddings_created=0,
                    metadata={"error": str(result)},
                    processing_time=0.0
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def update_document(
        self,
        document_id: str,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentIngestionResult:
        """Update an existing document"""
        if not self.is_initialized:
            await self.initialize()
        
        # Remove old document
        await self.vector_store_manager.delete_document(document_id)
        
        # Ingest updated document
        return await self.ingest_document(document_path, metadata)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the system"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            await self.vector_store_manager.delete_document(document_id)
            logger.info(f"Document {document_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.is_initialized:
            await self.initialize()
        
        stats = {
            "initialized": self.is_initialized,
            "config": self.config.to_dict(),
            "vector_store_stats": await self.vector_store_manager.get_stats(),
            "embedding_stats": await self.embedding_manager.get_stats(),
        }
        
        if self.metrics_collector:
            stats["metrics"] = await self.metrics_collector.get_metrics()
        
        return stats
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Optimize system performance"""
        if not self.is_initialized:
            await self.initialize()
        
        optimization_results = {}
        
        # Optimize vector store
        if hasattr(self.vector_store_manager, 'optimize'):
            optimization_results["vector_store"] = await self.vector_store_manager.optimize()
        
        # Optimize embeddings cache
        if hasattr(self.embedding_manager, 'optimize_cache'):
            optimization_results["embeddings"] = await self.embedding_manager.optimize_cache()
        
        # Clear old cache entries
        if self.cache_manager:
            optimization_results["cache"] = await self.cache_manager.cleanup_expired()
        
        logger.info("System optimization completed")
        return optimization_results
    
    async def shutdown(self):
        """Gracefully shutdown the RAG engine"""
        logger.info("Shutting down Advanced RAG Engine...")
        
        try:
            if self.vector_store_manager:
                await self.vector_store_manager.close()
            
            if self.embedding_manager:
                await self.embedding_manager.close()
            
            if self.cache_manager:
                await self.cache_manager.close()
            
            if self.metrics_collector:
                await self.metrics_collector.close()
            
            self.is_initialized = False
            logger.info("Advanced RAG Engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            raise