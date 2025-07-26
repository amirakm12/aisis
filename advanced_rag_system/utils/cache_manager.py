"""
Cache Manager - Handles caching for RAG system responses and embeddings
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from ..utils.config import CacheConfig


@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    memory_usage: float = 0.0


class CacheManager:
    """Manager for caching operations"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client = None
        self.memory_cache: Dict[str, Any] = {}
        self.stats = CacheStats()
    
    async def initialize(self) -> None:
        """Initialize cache manager"""
        if self.config.enable_cache:
            await self._initialize_redis()
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            import redis.asyncio as redis
            
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
        except ImportError:
            print("Redis not available, using in-memory cache only")
            self.redis_client = None
        except Exception as e:
            print(f"Redis connection failed: {e}, using in-memory cache only")
            self.redis_client = None
    
    async def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response for a query"""
        cache_key = self._generate_cache_key("response", query)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    self.stats.hits += 1
                    return json.loads(cached_data)
            except Exception:
                pass
        
        # Fallback to memory cache
        if cache_key in self.memory_cache:
            cached_data = self.memory_cache[cache_key]
            if self._is_cache_valid(cached_data):
                self.stats.hits += 1
                return cached_data["data"]
        
        self.stats.misses += 1
        return None
    
    async def cache_response(self, query: str, response: Dict[str, Any]) -> None:
        """Cache a response"""
        cache_key = self._generate_cache_key("response", query)
        cache_data = {
            "data": response,
            "timestamp": time.time(),
            "ttl": self.config.cache_ttl
        }
        
        # Store in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.config.cache_ttl,
                    json.dumps(cache_data)
                )
            except Exception:
                pass
        
        # Store in memory cache
        self.memory_cache[cache_key] = cache_data
        self.stats.total_entries += 1
    
    async def get_cached_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding for text"""
        cache_key = self._generate_cache_key("embedding", text)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    self.stats.hits += 1
                    return json.loads(cached_data)
            except Exception:
                pass
        
        # Fallback to memory cache
        if cache_key in self.memory_cache:
            cached_data = self.memory_cache[cache_key]
            if self._is_cache_valid(cached_data):
                self.stats.hits += 1
                return cached_data["data"]
        
        self.stats.misses += 1
        return None
    
    async def cache_embedding(self, text: str, embedding: list) -> None:
        """Cache an embedding"""
        cache_key = self._generate_cache_key("embedding", text)
        cache_data = {
            "data": embedding,
            "timestamp": time.time(),
            "ttl": self.config.cache_ttl
        }
        
        # Store in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.config.cache_ttl,
                    json.dumps(cache_data)
                )
            except Exception:
                pass
        
        # Store in memory cache
        self.memory_cache[cache_key] = cache_data
        self.stats.total_entries += 1
    
    def _generate_cache_key(self, prefix: str, content: str) -> str:
        """Generate cache key"""
        # Create hash of content
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}:{content_hash}"
    
    def _is_cache_valid(self, cache_data: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if not cache_data:
            return False
        
        timestamp = cache_data.get("timestamp", 0)
        ttl = cache_data.get("ttl", self.config.cache_ttl)
        
        return time.time() - timestamp < ttl
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, data in self.memory_cache.items():
            if not self._is_cache_valid(data):
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Update stats
        self.stats.total_entries = len(self.memory_cache)
        
        return len(expired_keys)
    
    async def clear_cache(self) -> None:
        """Clear all cache entries"""
        # Clear Redis cache
        if self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception:
                pass
        
        # Clear memory cache
        self.memory_cache.clear()
        self.stats.total_entries = 0
    
    async def get_cache_stats(self) -> CacheStats:
        """Get cache statistics"""
        # Update hit rate
        total_requests = self.stats.hits + self.stats.misses
        self.stats.hit_rate = self.stats.hits / total_requests if total_requests > 0 else 0.0
        
        # Estimate memory usage (simplified)
        self.stats.memory_usage = len(self.memory_cache) * 1024  # Rough estimate
        
        return self.stats
    
    async def close(self) -> None:
        """Close cache manager"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        
        self.memory_cache.clear() 