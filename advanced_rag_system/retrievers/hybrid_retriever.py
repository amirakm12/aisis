"""
Hybrid Retriever - Combines multiple retrieval strategies for optimal results
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from loguru import logger

from ..utils.config import RetrieverConfig
from ..vector_stores.vector_store_manager import VectorStoreManager
from ..embeddings.embedding_manager import EmbeddingManager

@dataclass
class RetrievalResult:
    """Result from a retrieval operation"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str
    rank: int

class HybridRetriever:
    """
    Hybrid Retriever that combines multiple retrieval strategies:
    - Semantic search using vector embeddings
    - Keyword search using BM25/TF-IDF
    - Hybrid fusion of both approaches
    - Advanced reranking using cross-encoders
    """
    
    def __init__(
        self,
        config: RetrieverConfig,
        vector_store_manager: VectorStoreManager,
        embedding_manager: EmbeddingManager
    ):
        self.config = config
        self.vector_store_manager = vector_store_manager
        self.embedding_manager = embedding_manager
        
        # Initialize keyword search components
        self.keyword_index = {}
        self.document_corpus = []
        self.bm25_index = None
        
        # Reranker model
        self.reranker = None
        self.reranker_initialized = False
        
        logger.info("Hybrid Retriever initialized")
    
    async def initialize_reranker(self):
        """Initialize the reranking model"""
        if self.reranker_initialized or not self.config.enable_reranking:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            self.reranker = CrossEncoder(self.config.reranker_model)
            self.reranker_initialized = True
            logger.info(f"Reranker model '{self.config.reranker_model}' initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {str(e)}")
            self.config.enable_reranking = False
    
    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        strategy: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method that coordinates different strategies
        
        Args:
            query: Search query
            top_k: Number of results to return
            strategy: Retrieval strategy ('semantic', 'keyword', 'hybrid')
            filters: Additional filters for search
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or self.config.top_k
        confidence_threshold = confidence_threshold or self.config.similarity_threshold
        
        try:
            if strategy == "semantic":
                results = await self._semantic_retrieval(query, top_k * 2, filters)
            elif strategy == "keyword":
                results = await self._keyword_retrieval(query, top_k * 2, filters)
            elif strategy == "hybrid":
                results = await self._hybrid_retrieval(query, top_k, filters)
            else:
                raise ValueError(f"Unsupported retrieval strategy: {strategy}")
            
            # Filter by confidence threshold
            filtered_results = [
                result for result in results
                if result.get("score", 0) >= confidence_threshold
            ]
            
            # Apply reranking if enabled
            if self.config.enable_reranking and len(filtered_results) > 1:
                filtered_results = await self._rerank_results(query, filtered_results)
            
            # Limit to top_k results
            final_results = filtered_results[:top_k]
            
            logger.info(f"Retrieved {len(final_results)} documents using {strategy} strategy")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []
    
    async def _semantic_retrieval(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform semantic retrieval using vector similarity"""
        try:
            results = await self.vector_store_manager.search(
                query=query,
                top_k=top_k,
                filters=filters,
                similarity_threshold=0.0  # Apply threshold later
            )
            
            # Add retrieval method metadata
            for result in results:
                result["retrieval_method"] = "semantic"
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {str(e)}")
            return []
    
    async def _keyword_retrieval(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based retrieval using BM25"""
        try:
            # Initialize BM25 if not already done
            if self.bm25_index is None:
                await self._initialize_bm25_index()
            
            if self.bm25_index is None:
                logger.warning("BM25 index not available, falling back to semantic search")
                return await self._semantic_retrieval(query, top_k, filters)
            
            # Tokenize query
            query_tokens = self._tokenize(query)
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0:  # Only include non-zero scores
                    doc_info = self.document_corpus[idx]
                    results.append({
                        "id": doc_info.get("id", f"doc_{idx}"),
                        "content": doc_info.get("content", ""),
                        "metadata": doc_info.get("metadata", {}),
                        "score": float(scores[idx]),
                        "retrieval_method": "keyword"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {str(e)}")
            return []
    
    async def _hybrid_retrieval(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval combining semantic and keyword approaches"""
        try:
            # Get results from both methods
            semantic_results = await self._semantic_retrieval(query, top_k * 2, filters)
            keyword_results = await self._keyword_retrieval(query, top_k * 2, filters)
            
            # Combine and deduplicate results
            combined_results = await self._fuse_results(
                semantic_results,
                keyword_results,
                alpha=self.config.hybrid_search_alpha
            )
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            # Fallback to semantic search
            return await self._semantic_retrieval(query, top_k, filters)
    
    async def _fuse_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Fuse results from semantic and keyword retrieval"""
        try:
            # Create lookup dictionaries
            semantic_lookup = {result["id"]: result for result in semantic_results}
            keyword_lookup = {result["id"]: result for result in keyword_results}
            
            # Get all unique document IDs
            all_ids = set(semantic_lookup.keys()) | set(keyword_lookup.keys())
            
            # Normalize scores to [0, 1] range
            if semantic_results:
                max_semantic_score = max(r.get("score", 0) for r in semantic_results)
                min_semantic_score = min(r.get("score", 0) for r in semantic_results)
            else:
                max_semantic_score = min_semantic_score = 0
            
            if keyword_results:
                max_keyword_score = max(r.get("score", 0) for r in keyword_results)
                min_keyword_score = min(r.get("score", 0) for r in keyword_results)
            else:
                max_keyword_score = min_keyword_score = 0
            
            fused_results = []
            
            for doc_id in all_ids:
                semantic_result = semantic_lookup.get(doc_id)
                keyword_result = keyword_lookup.get(doc_id)
                
                # Normalize scores
                semantic_score = 0.0
                if semantic_result and max_semantic_score > min_semantic_score:
                    semantic_score = (semantic_result["score"] - min_semantic_score) / (max_semantic_score - min_semantic_score)
                
                keyword_score = 0.0
                if keyword_result and max_keyword_score > min_keyword_score:
                    keyword_score = (keyword_result["score"] - min_keyword_score) / (max_keyword_score - min_keyword_score)
                
                # Combine scores using weighted average
                combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
                
                # Use the result with content (prefer semantic if both available)
                base_result = semantic_result or keyword_result
                if base_result:
                    fused_result = base_result.copy()
                    fused_result["score"] = combined_score
                    fused_result["retrieval_method"] = "hybrid"
                    fused_result["fusion_details"] = {
                        "semantic_score": semantic_score,
                        "keyword_score": keyword_score,
                        "alpha": alpha
                    }
                    fused_results.append(fused_result)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Error fusing results: {str(e)}")
            # Return semantic results as fallback
            return semantic_results
    
    async def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results using a cross-encoder model"""
        try:
            if not self.reranker_initialized:
                await self.initialize_reranker()
            
            if not self.reranker:
                return results
            
            # Prepare query-document pairs
            pairs = []
            for result in results:
                content = result.get("content", "")[:512]  # Limit content length
                pairs.append([query, content])
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update results with reranking scores
            reranked_results = []
            for i, result in enumerate(results):
                result_copy = result.copy()
                result_copy["rerank_score"] = float(rerank_scores[i])
                result_copy["original_score"] = result.get("score", 0)
                result_copy["score"] = float(rerank_scores[i])  # Use rerank score as primary
                reranked_results.append(result_copy)
            
            # Sort by reranking score
            reranked_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            logger.info(f"Reranked {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error reranking results: {str(e)}")
            return results
    
    async def _initialize_bm25_index(self):
        """Initialize BM25 index from vector store documents"""
        try:
            # This is a simplified implementation
            # In practice, you'd maintain the BM25 index alongside the vector store
            
            # For now, we'll skip BM25 initialization and use semantic search
            logger.warning("BM25 index initialization not implemented, using semantic search only")
            self.bm25_index = None
            
        except Exception as e:
            logger.error(f"Error initializing BM25 index: {str(e)}")
            self.bm25_index = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        import re
        
        # Simple tokenization - could be enhanced with proper NLP libraries
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    async def add_documents_to_keyword_index(
        self,
        documents: List[Dict[str, Any]]
    ):
        """Add documents to the keyword search index"""
        try:
            # Add to document corpus
            self.document_corpus.extend(documents)
            
            # Reinitialize BM25 index if we have documents
            if len(self.document_corpus) > 0:
                await self._initialize_bm25_index()
            
            logger.info(f"Added {len(documents)} documents to keyword index")
            
        except Exception as e:
            logger.error(f"Error adding documents to keyword index: {str(e)}")
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "config": {
                "top_k": self.config.top_k,
                "similarity_threshold": self.config.similarity_threshold,
                "enable_reranking": self.config.enable_reranking,
                "reranker_model": self.config.reranker_model,
                "hybrid_search_alpha": self.config.hybrid_search_alpha
            },
            "corpus_stats": {
                "document_count": len(self.document_corpus),
                "bm25_initialized": self.bm25_index is not None,
                "reranker_initialized": self.reranker_initialized
            }
        }
    
    async def optimize_retrieval(self) -> Dict[str, Any]:
        """Optimize retrieval performance"""
        try:
            optimization_results = {
                "optimizations_performed": [],
                "performance_improvements": {}
            }
            
            # Optimize keyword index if needed
            if len(self.document_corpus) > 1000 and self.bm25_index is None:
                await self._initialize_bm25_index()
                optimization_results["optimizations_performed"].append("bm25_index_creation")
            
            # Initialize reranker if not done
            if self.config.enable_reranking and not self.reranker_initialized:
                await self.initialize_reranker()
                if self.reranker_initialized:
                    optimization_results["optimizations_performed"].append("reranker_initialization")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing retrieval: {str(e)}")
            return {"error": str(e)}