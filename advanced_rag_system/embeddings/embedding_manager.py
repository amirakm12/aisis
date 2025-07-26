"""
Embedding Manager - Manages different embedding models
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..utils.config import EmbeddingConfig


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations"""
    total_embeddings: int = 0
    total_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_embedding_time: float = 0.0


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding model"""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the embedding model"""
        pass


class SentenceTransformerModel(BaseEmbeddingModel):
    """Sentence Transformers embedding model"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    async def initialize(self) -> None:
        """Initialize Sentence Transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(self.config.model_name)
            
        except ImportError:
            raise ImportError(
                "Sentence Transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Encode text
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize,
            max_length=self.config.max_length
        )
        
        return embedding.tolist()
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Encode texts in batch
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.config.normalize,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size
        )
        
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 768  # Default dimension
    
    async def shutdown(self) -> None:
        """Shutdown the model"""
        self.model = None


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embedding model"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize OpenAI client"""
        try:
            import openai
            
            self.client = openai.AsyncOpenAI(api_key=self.config.api_key)
            
        except ImportError:
            raise ImportError("OpenAI is not installed. Install with: pip install openai")
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        response = await self.client.embeddings.create(
            model=self.config.model_name,
            input=text
        )
        
        return response.data[0].embedding
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        # Process in batches
        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            response = await self.client.embeddings.create(
                model=self.config.model_name,
                input=batch
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        # OpenAI text-embedding-ada-002 has 1536 dimensions
        if "ada" in self.config.model_name:
            return 1536
        return 1536  # Default
    
    async def shutdown(self) -> None:
        """Shutdown the client"""
        self.client = None


class EmbeddingManager:
    """Manager for embedding operations"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model: Optional[BaseEmbeddingModel] = None
        self.cache: Dict[str, List[float]] = {}
        self.stats = EmbeddingStats()
    
    async def initialize(self) -> None:
        """Initialize the appropriate embedding model"""
        if self.config.type.value == "sentence_transformers":
            self.model = SentenceTransformerModel(self.config)
        elif self.config.type.value == "openai":
            self.model = OpenAIEmbeddingModel(self.config)
        else:
            # Default to Sentence Transformers
            self.model = SentenceTransformerModel(self.config)
        
        await self.model.initialize()
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        # Check cache first
        cache_key = f"single_{hash(text)}"
        if cache_key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[cache_key]
        
        self.stats.cache_misses += 1
        
        # Generate embedding
        start_time = asyncio.get_event_loop().time()
        embedding = await self.model.embed_text(text)
        
        # Update stats
        self.stats.total_embeddings += 1
        self.stats.total_tokens += len(text.split())
        
        # Cache result
        self.cache[cache_key] = embedding
        
        # Update average time
        processing_time = asyncio.get_event_loop().time() - start_time
        self.stats.average_embedding_time = (
            (self.stats.average_embedding_time * (self.stats.total_embeddings - 1) + processing_time) /
            self.stats.total_embeddings
        )
        
        return embedding
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        # Check cache for all texts
        cache_keys = [f"batch_{hash(text)}" for text in texts]
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
            if cache_key in self.cache:
                cached_embeddings.append(self.cache[cache_key])
                self.stats.cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.stats.cache_misses += 1
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            start_time = asyncio.get_event_loop().time()
            new_embeddings = await self.model.embed_texts(uncached_texts)
            
            # Update stats
            self.stats.total_embeddings += len(uncached_texts)
            self.stats.total_tokens += sum(len(text.split()) for text in uncached_texts)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = f"batch_{hash(text)}"
                self.cache[cache_key] = embedding
            
            # Update average time
            processing_time = asyncio.get_event_loop().time() - start_time
            self.stats.average_embedding_time = (
                (self.stats.average_embedding_time * (self.stats.total_embeddings - len(uncached_texts)) + processing_time) /
                self.stats.total_embeddings
            )
        else:
            new_embeddings = []
        
        # Combine cached and new embeddings in correct order
        result = [None] * len(texts)
        for i, embedding in enumerate(cached_embeddings):
            result[i] = embedding
        
        for i, embedding in zip(uncached_indices, new_embeddings):
            result[i] = embedding
        
        return result
    
    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute similarity between two embeddings"""
        import numpy as np
        
        # Cosine similarity
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
    async def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """Find most similar embeddings"""
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model:
            return self.model.get_dimension()
        return 768  # Default
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": self.config.type.value,
            "model_name": self.config.model_name,
            "dimension": self.get_dimension(),
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "normalize": self.config.normalize
        }
    
    def get_stats(self) -> EmbeddingStats:
        """Get embedding statistics"""
        return self.stats
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimize the embedding cache"""
        # Simple cache optimization: limit size
        max_cache_size = 1000
        
        if len(self.cache) > max_cache_size:
            # Remove oldest entries (simplified)
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - max_cache_size]
            for key in keys_to_remove:
                del self.cache[key]
        
        return {
            "cache_size": len(self.cache),
            "cache_hit_rate": self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses) if (self.stats.cache_hits + self.stats.cache_misses) > 0 else 0,
            "total_embeddings": self.stats.total_embeddings,
            "average_time": self.stats.average_embedding_time
        }
    
    async def shutdown(self) -> None:
        """Shutdown the embedding manager"""
        if self.model:
            await self.model.shutdown()
        self.cache.clear() 