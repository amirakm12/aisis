"""
Hybrid Retriever - Combines semantic and keyword search with reranking
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..utils.config import RetrieverConfig
from ..vector_stores.vector_store_manager import VectorStoreManager


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval"""
    documents: List[Dict[str, Any]]
    confidence: float
    retrieval_strategy: str
    metadata: Dict[str, Any]


class HybridRetriever:
    """
    Hybrid retriever that combines semantic and keyword search
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager, config: RetrieverConfig):
        self.vector_store_manager = vector_store_manager
        self.config = config
        self.bm25_index = None
        self.reranker = None
    
    async def initialize(self) -> None:
        """Initialize the hybrid retriever"""
        # Initialize BM25 index for keyword search
        await self._initialize_bm25_index()
        
        # Initialize reranker if enabled
        if self.config.use_reranking:
            await self._initialize_reranker()
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve documents using hybrid search
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use reranking
            filter_metadata: Metadata filters
            
        Returns:
            RetrievalResult with documents and metadata
        """
        top_k = top_k or self.config.top_k
        use_reranking = use_reranking if use_reranking is not None else self.config.use_reranking
        
        # Step 1: Semantic retrieval
        semantic_results = await self._semantic_retrieval(
            query, top_k, filter_metadata
        )
        
        # Step 2: Keyword retrieval
        keyword_results = await self._keyword_retrieval(
            query, top_k, filter_metadata
        )
        
        # Step 3: Hybrid fusion
        fused_results = await self._hybrid_retrieval(
            query, semantic_results, keyword_results, top_k
        )
        
        # Step 4: Reranking (if enabled)
        if use_reranking and self.reranker:
            final_results = await self._rerank_results(
                query, fused_results, top_k
            )
        else:
            final_results = fused_results
        
        # Calculate confidence
        confidence = self._calculate_confidence(final_results)
        
        return RetrievalResult(
            documents=final_results,
            confidence=confidence,
            retrieval_strategy="hybrid",
            metadata={
                "semantic_results": len(semantic_results),
                "keyword_results": len(keyword_results),
                "reranked": use_reranking,
                "total_retrieved": len(final_results)
            }
        )
    
    async def _semantic_retrieval(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity"""
        search_results = await self.vector_store_manager.search(
            query, top_k, filter_metadata
        )
        
        # Convert to standard format
        documents = []
        for result in search_results:
            documents.append({
                "id": result.document_id,
                "content": result.content,
                "metadata": result.metadata,
                "score": result.similarity_score,
                "source": "semantic"
            })
        
        return documents
    
    async def _keyword_retrieval(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform keyword search using BM25"""
        if not self.bm25_index:
            # Fallback to semantic search
            return await self._semantic_retrieval(query, top_k, filter_metadata)
        
        # Simple keyword matching (in production, use proper BM25)
        # For now, return empty results
        return []
    
    async def _hybrid_retrieval(
        self,
        query: str,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fuse semantic and keyword results"""
        # Combine results
        all_results = semantic_results + keyword_results
        
        # Remove duplicates based on document ID
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                unique_results.append(result)
        
        # Score fusion
        fused_results = await self._fuse_results(unique_results)
        
        # Sort by combined score
        fused_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return fused_results[:top_k]
    
    async def _fuse_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fuse scores from different retrieval methods"""
        for result in results:
            semantic_score = result.get("score", 0) if result.get("source") == "semantic" else 0
            keyword_score = result.get("score", 0) if result.get("source") == "keyword" else 0
            
            # Weighted combination
            combined_score = (
                self.config.hybrid_weight * semantic_score +
                (1 - self.config.hybrid_weight) * keyword_score
            )
            
            result["combined_score"] = combined_score
            result["semantic_score"] = semantic_score
            result["keyword_score"] = keyword_score
        
        return results
    
    async def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder"""
        if not self.reranker or not results:
            return results
        
        # Prepare pairs for reranking
        pairs = [(query, result["content"]) for result in results]
        
        # Get reranking scores
        scores = await self._get_reranking_scores(pairs)
        
        # Update results with reranking scores
        for i, result in enumerate(results):
            result["reranking_score"] = scores[i] if i < len(scores) else 0
        
        # Sort by reranking score
        results.sort(key=lambda x: x.get("reranking_score", 0), reverse=True)
        
        return results[:top_k]
    
    async def _get_reranking_scores(self, pairs: List[tuple]) -> List[float]:
        """Get reranking scores for query-document pairs"""
        # Placeholder for cross-encoder reranking
        # In production, use models like cross-encoder/ms-marco-MiniLM-L-6-v2
        return [0.8] * len(pairs)  # Default scores
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for retrieval results"""
        if not results:
            return 0.0
        
        # Calculate average score
        scores = [result.get("combined_score", 0) for result in results]
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range
        confidence = min(1.0, max(0.0, avg_score))
        
        return confidence
    
    async def _initialize_bm25_index(self) -> None:
        """Initialize BM25 index for keyword search"""
        # Placeholder for BM25 initialization
        # In production, use libraries like rank_bm25
        self.bm25_index = None
    
    async def _initialize_reranker(self) -> None:
        """Initialize cross-encoder reranker"""
        # Placeholder for reranker initialization
        # In production, use sentence-transformers cross-encoders
        self.reranker = None
    
    async def shutdown(self) -> None:
        """Shutdown the retriever"""
        self.bm25_index = None
        self.reranker = None 