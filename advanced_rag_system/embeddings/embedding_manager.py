"""
Embedding Manager - Handles different embedding models and strategies
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from loguru import logger

from ..utils.config import EmbeddingConfig

class BaseEmbeddingModel(ABC):
    """Base class for embedding models"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self):
        """Initialize the embedding model"""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text"""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the model"""
        pass

class SentenceTransformerModel(BaseEmbeddingModel):
    """Sentence Transformers implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model = None
    
    async def initialize(self):
        """Initialize Sentence Transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model
            self.model = SentenceTransformer(self.config.model_name)
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.is_initialized = True
            logger.info(f"SentenceTransformer model '{self.config.model_name}' initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {str(e)}")
            raise
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Truncate text if too long
            if len(text) > self.config.max_length:
                text = text[:self.config.max_length]
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Truncate texts if too long
            processed_texts = []
            for text in texts:
                if len(text) > self.config.max_length:
                    text = text[:self.config.max_length]
                processed_texts.append(text)
            
            # Generate embeddings in batches
            embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                
                batch_embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True,
                    batch_size=len(batch)
                )
                
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 0
    
    async def close(self):
        """Close the model"""
        try:
            # SentenceTransformers doesn't require explicit closing
            self.is_initialized = False
            logger.info("SentenceTransformer model closed")
            
        except Exception as e:
            logger.error(f"Error closing SentenceTransformer: {str(e)}")

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI Embeddings implementation"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.client = None
    
    async def initialize(self):
        """Initialize OpenAI client"""
        try:
            import openai
            import os
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            
            self.client = openai.AsyncOpenAI(api_key=api_key)
            
            self.is_initialized = True
            logger.info(f"OpenAI embedding model '{self.config.model_name}' initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
            raise
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using OpenAI"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Truncate text if too long
            if len(text) > self.config.max_length:
                text = text[:self.config.max_length]
            
            # Generate embedding
            response = await self.client.embeddings.create(
                model=self.config.model_name,
                input=text
            )
            
            embedding = np.array(response.data[0].embedding)
            
            if self.config.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding text with OpenAI: {str(e)}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using OpenAI"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Process texts in batches to respect API limits
            embeddings = []
            batch_size = min(self.config.batch_size, 100)  # OpenAI limit
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Truncate texts if too long
                processed_batch = []
                for text in batch:
                    if len(text) > self.config.max_length:
                        text = text[:self.config.max_length]
                    processed_batch.append(text)
                
                # Generate embeddings for batch
                response = await self.client.embeddings.create(
                    model=self.config.model_name,
                    input=processed_batch
                )
                
                # Extract embeddings
                for embedding_data in response.data:
                    embedding = np.array(embedding_data.embedding)
                    
                    if self.config.normalize_embeddings:
                        embedding = embedding / np.linalg.norm(embedding)
                    
                    embeddings.append(embedding)
                
                # Add delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding texts with OpenAI: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension for OpenAI models"""
        # Common OpenAI embedding dimensions
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return model_dimensions.get(self.config.model_name, 1536)
    
    async def close(self):
        """Close OpenAI client"""
        try:
            # OpenAI client doesn't require explicit closing
            self.is_initialized = False
            logger.info("OpenAI embedding model closed")
            
        except Exception as e:
            logger.error(f"Error closing OpenAI embeddings: {str(e)}")

class EmbeddingManager:
    """
    Manager class for embeddings with multiple model support
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedding_model = None
        self.is_initialized = False
        self.embedding_cache = {}  # Simple in-memory cache
        
        # Model factory
        self.model_factories = {
            "sentence_transformers": SentenceTransformerModel,
            "openai": OpenAIEmbeddingModel,
        }
    
    async def initialize(self):
        """Initialize the embedding manager"""
        if self.is_initialized:
            return
        
        try:
            model_type = self.config.model_type.lower()
            
            if model_type not in self.model_factories:
                raise ValueError(f"Unsupported embedding model type: {model_type}")
            
            # Create and initialize embedding model
            model_class = self.model_factories[model_type]
            self.embedding_model = model_class(self.config)
            await self.embedding_model.initialize()
            
            self.is_initialized = True
            logger.info(f"Embedding Manager initialized with {model_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Embedding Manager: {str(e)}")
            raise
    
    async def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Embed a single text"""
        if not self.is_initialized:
            await self.initialize()
        
        # Check cache first
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = await self.embedding_model.embed_text(text)
            
            # Cache the result
            if use_cache:
                self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    async def embed_texts(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """Embed multiple texts"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Check cache for existing embeddings
            embeddings = []
            texts_to_embed = []
            indices_to_embed = []
            
            for i, text in enumerate(texts):
                if use_cache and text in self.embedding_cache:
                    embeddings.append(self.embedding_cache[text])
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
            
            # Embed texts not in cache
            if texts_to_embed:
                new_embeddings = await self.embedding_model.embed_texts(texts_to_embed)
                
                # Fill in the placeholders and update cache
                for i, (idx, embedding) in enumerate(zip(indices_to_embed, new_embeddings)):
                    embeddings[idx] = embedding
                    
                    if use_cache:
                        self.embedding_cache[texts_to_embed[i]] = embedding
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between two embeddings"""
        try:
            if metric == "cosine":
                # Cosine similarity
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return dot_product / (norm1 * norm2)
            
            elif metric == "euclidean":
                # Euclidean distance (converted to similarity)
                distance = np.linalg.norm(embedding1 - embedding2)
                return 1.0 / (1.0 + distance)
            
            elif metric == "dot_product":
                # Dot product similarity
                return np.dot(embedding1, embedding2)
            
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")
                
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    async def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5,
        metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """Find most similar embeddings"""
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = await self.compute_similarity(
                    query_embedding, candidate, metric
                )
                similarities.append({"index": i, "similarity": similarity})
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding most similar: {str(e)}")
            return []
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.embedding_model:
            return self.embedding_model.get_dimension()
        return 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "dimension": self.get_dimension(),
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "normalize_embeddings": self.config.normalize_embeddings,
            "initialized": self.is_initialized
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics"""
        return {
            **self.get_model_info(),
            "cache_size": len(self.embedding_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        # This is a simplified implementation
        # In a real system, you'd track hits and misses
        return 0.0 if not self.embedding_cache else 0.5
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimize embedding cache"""
        try:
            initial_size = len(self.embedding_cache)
            
            # Simple cache optimization: remove old entries if cache is too large
            max_cache_size = 1000  # Configurable
            
            if len(self.embedding_cache) > max_cache_size:
                # Keep only the most recent entries (simplified approach)
                # In a real implementation, you'd use LRU or similar
                items = list(self.embedding_cache.items())
                self.embedding_cache = dict(items[-max_cache_size:])
            
            final_size = len(self.embedding_cache)
            
            return {
                "optimization_performed": True,
                "initial_cache_size": initial_size,
                "final_cache_size": final_size,
                "entries_removed": initial_size - final_size
            }
            
        except Exception as e:
            logger.error(f"Error optimizing cache: {str(e)}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the embedding manager"""
        try:
            if self.embedding_model:
                await self.embedding_model.close()
            
            # Clear cache
            self.embedding_cache.clear()
            
            self.is_initialized = False
            logger.info("Embedding Manager closed")
            
        except Exception as e:
            logger.error(f"Error closing Embedding Manager: {str(e)}")
            raise