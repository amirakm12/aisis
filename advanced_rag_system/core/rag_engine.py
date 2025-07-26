"""
Main RAG Engine - Central orchestrator for the Advanced RAG System
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from ..utils.config import RAGConfig
from ..agents.orchestrator_agent import OrchestratorAgent
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
    query_time: float
    total_tokens: int
    metadata: Dict[str, Any]


@dataclass
class DocumentIngestionResult:
    """Result of document ingestion"""
    document_id: str
    success: bool
    chunks_created: int
    processing_time: float
    error_message: Optional[str] = None


class AdvancedRAGEngine:
    """
    Main RAG Engine that orchestrates all components
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.initialized = False
        
        # Core components
        self.orchestrator_agent: Optional[OrchestratorAgent] = None
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.cache_manager: Optional[CacheManager] = None
        self.metrics_collector: Optional[MetricsCollector] = None
    
    async def initialize(self) -> None:
        """Initialize all components of the RAG system"""
        if self.initialized:
            return
        
        print("Initializing Advanced RAG System...")
        
        # Initialize components in order
        await self._initialize_embeddings()
        await self._initialize_vector_store()
        await self._initialize_retriever()
        await self._initialize_processors()
        await self._initialize_agents()
        await self._initialize_cache()
        await self._initialize_metrics()
        
        self.initialized = True
        print("Advanced RAG System initialized successfully!")
    
    async def _initialize_embeddings(self) -> None:
        """Initialize embedding manager"""
        self.embedding_manager = EmbeddingManager(self.config.embedding)
        await self.embedding_manager.initialize()
    
    async def _initialize_vector_store(self) -> None:
        """Initialize vector store manager"""
        self.vector_store_manager = VectorStoreManager(
            self.config.vector_store,
            self.embedding_manager
        )
        await self.vector_store_manager.initialize()
    
    async def _initialize_retriever(self) -> None:
        """Initialize hybrid retriever"""
        self.hybrid_retriever = HybridRetriever(
            self.vector_store_manager,
            self.config.retriever
        )
        await self.hybrid_retriever.initialize()
    
    async def _initialize_processors(self) -> None:
        """Initialize document processor"""
        self.document_processor = DocumentProcessor(self.config.processing)
    
    async def _initialize_agents(self) -> None:
        """Initialize AI agents"""
        if self.config.agent.enable_orchestrator:
            self.orchestrator_agent = OrchestratorAgent(
                self.hybrid_retriever,
                self.config
            )
            await self.orchestrator_agent.initialize()
    
    async def _initialize_cache(self) -> None:
        """Initialize cache manager"""
        self.cache_manager = CacheManager(self.config.cache)
        await self.cache_manager.initialize()
    
    async def _initialize_metrics(self) -> None:
        """Initialize metrics collector"""
        self.metrics_collector = MetricsCollector(self.config.monitoring)
        await self.metrics_collector.initialize()
    
    async def query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> RAGResponse:
        """
        Process a query through the RAG system
        
        Args:
            query: The user query
            context: Additional context for the query
            use_cache: Whether to use cached responses
            
        Returns:
            RAGResponse with answer and metadata
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache and self.cache_manager:
                cached_response = await self.cache_manager.get_cached_response(query)
                if cached_response:
                    return RAGResponse(**cached_response)
            
            # Process query through orchestrator
            if self.orchestrator_agent:
                result = await self.orchestrator_agent.process_query(query, context)
            else:
                # Fallback to direct retrieval
                retrieval_result = await self.hybrid_retriever.retrieve(query)
                result = {
                    "answer": "Generated answer based on retrieved documents",
                    "sources": retrieval_result.documents,
                    "confidence": 0.8,
                    "metadata": {}
                }
            
            query_time = time.time() - start_time
            
            # Create response
            response = RAGResponse(
                answer=result["answer"],
                sources=result["sources"],
                confidence=result.get("confidence", 0.8),
                query_time=query_time,
                total_tokens=result.get("total_tokens", 0),
                metadata=result.get("metadata", {})
            )
            
            # Cache response
            if use_cache and self.cache_manager:
                await self.cache_manager.cache_response(query, response.__dict__)
            
            # Record metrics
            if self.metrics_collector:
                await self.metrics_collector.record_query_metrics(
                    query, query_time, len(result["sources"])
                )
            
            return response
            
        except Exception as e:
            # Record error
            if self.metrics_collector:
                await self.metrics_collector.record_error("query", str(e))
            raise
    
    async def ingest_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentIngestionResult:
        """
        Ingest a single document into the RAG system
        
        Args:
            file_path: Path to the document
            metadata: Additional metadata for the document
            
        Returns:
            DocumentIngestionResult with processing details
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Process document
            processed_doc = await self.document_processor.process_document(file_path)
            
            # Create embeddings and store
            document_id = await self.vector_store_manager.add_documents(
                [processed_doc], metadata
            )
            
            processing_time = time.time() - start_time
            
            # Record metrics
            if self.metrics_collector:
                await self.metrics_collector.record_ingestion_metrics(
                    file_path, processing_time, len(processed_doc.chunks)
                )
            
            return DocumentIngestionResult(
                document_id=document_id[0] if document_id else "unknown",
                success=True,
                chunks_created=len(processed_doc.chunks),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record error
            if self.metrics_collector:
                await self.metrics_collector.record_error("ingestion", str(e))
            
            return DocumentIngestionResult(
                document_id="",
                success=False,
                chunks_created=0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def ingest_documents_batch(
        self,
        file_paths: List[Union[str, Path]],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[DocumentIngestionResult]:
        """
        Ingest multiple documents in batch
        
        Args:
            file_paths: List of document paths
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of DocumentIngestionResult
        """
        if not self.initialized:
            await self.initialize()
        
        results = []
        
        # Process documents in parallel
        tasks = []
        for i, file_path in enumerate(file_paths):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            task = self.ingest_document(file_path, metadata)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(DocumentIngestionResult(
                    document_id="",
                    success=False,
                    chunks_created=0,
                    processing_time=0,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def update_document(
        self,
        document_id: str,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentIngestionResult:
        """Update an existing document"""
        # Delete old document
        await self.delete_document(document_id)
        
        # Ingest new document
        return await self.ingest_document(file_path, metadata)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the system"""
        if not self.initialized:
            await self.initialize()
        
        try:
            await self.vector_store_manager.delete_documents([document_id])
            return True
        except Exception as e:
            if self.metrics_collector:
                await self.metrics_collector.record_error("deletion", str(e))
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information"""
        if not self.initialized:
            await self.initialize()
        
        stats = {
            "system_status": "initialized" if self.initialized else "not_initialized",
            "vector_store_stats": await self.vector_store_manager.get_stats(),
            "embedding_stats": await self.embedding_manager.get_stats(),
            "cache_stats": await self.cache_manager.get_cache_stats() if self.cache_manager else {},
            "metrics_summary": await self.metrics_collector.get_performance_summary() if self.metrics_collector else {}
        }
        
        return stats
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Optimize system performance"""
        if not self.initialized:
            await self.initialize()
        
        optimizations = {}
        
        # Optimize vector store
        if self.vector_store_manager:
            optimizations["vector_store"] = await self.vector_store_manager.optimize()
        
        # Optimize embedding cache
        if self.embedding_manager:
            optimizations["embedding_cache"] = await self.embedding_manager.optimize_cache()
        
        # Clean up expired cache entries
        if self.cache_manager:
            optimizations["cache_cleanup"] = await self.cache_manager.cleanup_expired()
        
        return optimizations
    
    async def shutdown(self) -> None:
        """Shutdown the RAG system gracefully"""
        if not self.initialized:
            return
        
        print("Shutting down Advanced RAG System...")
        
        # Shutdown components in reverse order
        if self.metrics_collector:
            await self.metrics_collector.close()
        
        if self.cache_manager:
            await self.cache_manager.close()
        
        if self.orchestrator_agent:
            await self.orchestrator_agent.shutdown()
        
        if self.hybrid_retriever:
            await self.hybrid_retriever.shutdown()
        
        if self.vector_store_manager:
            await self.vector_store_manager.shutdown()
        
        if self.embedding_manager:
            await self.embedding_manager.shutdown()
        
        self.initialized = False
        print("Advanced RAG System shutdown complete!") 