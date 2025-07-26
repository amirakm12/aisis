"""
Cache Manager - Handles caching for improved performance
"""

import asyncio
import json
import hashlib
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from loguru import logger

from .config import CacheConfig

class CacheManager:
    """
    Cache Manager for RAG system responses and embeddings
    
    Supports:
    - Response caching
    - Embedding caching
    - TTL-based expiration
    - Multiple backend support (Redis, in-memory)
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_backend = None
        self.is_initialized = False
        
        # In-memory fallback cache
        self.memory_cache = {}
        self.cache_timestamps = {}
    
    async def initialize(self):
        """Initialize the cache manager"""
        if self.is_initialized:
            return
        
        try:
            if self.config.cache_provider == "redis":
                await self._initialize_redis()
            else:
                # Use in-memory cache
                logger.info("Using in-memory cache")
            
            self.is_initialized = True
            logger.info(f"Cache Manager initialized with {self.config.cache_provider}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cache Manager: {str(e)}")
            # Fall back to in-memory cache
            self.config.cache_provider = "memory"
            self.is_initialized = True
    
    async def _initialize_redis(self):
        """Initialize Redis cache backend"""
        try:
            import redis.asyncio as redis
            
            connection_params = self.config.connection_params
            redis_url = connection_params.get("url", "redis://localhost:6379")
            
            self.cache_backend = redis.from_url(redis_url)
            
            # Test connection
            await self.cache_backend.ping()
            logger.info("Redis cache backend initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            raise
    
    async def get_cached_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Get cached response for a query"""
        if not self.config.enable_cache:
            return None
        
        try:
            cache_key = self._generate_cache_key("response", query, context, filters)
            
            if self.config.cache_provider == "redis" and self.cache_backend:
                cached_data = await self.cache_backend.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                # Use in-memory cache
                if cache_key in self.memory_cache:
                    if self._is_cache_valid(cache_key):
                        return self.memory_cache[cache_key]
                    else:
                        # Remove expired entry
                        del self.memory_cache[cache_key]
                        del self.cache_timestamps[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached response: {str(e)}")
            return None
    
    async def cache_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
        response: Any
    ) -> bool:
        """Cache a response"""
        if not self.config.enable_cache:
            return False
        
        try:
            cache_key = self._generate_cache_key("response", query, context, filters)
            
            if self.config.cache_provider == "redis" and self.cache_backend:
                # Store in Redis with TTL
                cached_data = json.dumps(response, default=str)
                await self.cache_backend.setex(
                    cache_key, 
                    self.config.cache_ttl, 
                    cached_data
                )
            else:
                # Store in memory cache
                self.memory_cache[cache_key] = response
                self.cache_timestamps[cache_key] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")
            return False
    
    async def get_cached_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding for text"""
        if not self.config.enable_cache:
            return None
        
        try:
            cache_key = self._generate_cache_key("embedding", text)
            
            if self.config.cache_provider == "redis" and self.cache_backend:
                cached_data = await self.cache_backend.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                # Use in-memory cache
                if cache_key in self.memory_cache:
                    if self._is_cache_valid(cache_key):
                        return self.memory_cache[cache_key]
                    else:
                        # Remove expired entry
                        del self.memory_cache[cache_key]
                        del self.cache_timestamps[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached embedding: {str(e)}")
            return None
    
    async def cache_embedding(self, text: str, embedding: list) -> bool:
        """Cache an embedding"""
        if not self.config.enable_cache:
            return False
        
        try:
            cache_key = self._generate_cache_key("embedding", text)
            
            if self.config.cache_provider == "redis" and self.cache_backend:
                # Store in Redis with TTL
                cached_data = json.dumps(embedding)
                await self.cache_backend.setex(
                    cache_key, 
                    self.config.cache_ttl, 
                    cached_data
                )
            else:
                # Store in memory cache
                self.memory_cache[cache_key] = embedding
                self.cache_timestamps[cache_key] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching embedding: {str(e)}")
            return False
    
    def _generate_cache_key(
        self, 
        cache_type: str, 
        primary_key: str, 
        *args
    ) -> str:
        """Generate a cache key"""
        # Create a hash of all parameters
        key_data = {
            "type": cache_type,
            "primary": primary_key,
            "args": args
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        return f"rag:{cache_type}:{key_hash[:16]}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        timestamp = self.cache_timestamps[cache_key]
        expiry_time = timestamp + timedelta(seconds=self.config.cache_ttl)
        
        return datetime.now() < expiry_time
    
    async def cleanup_expired(self) -> Dict[str, Any]:
        """Clean up expired cache entries"""
        try:
            cleanup_stats = {
                "entries_removed": 0,
                "entries_remaining": 0
            }
            
            if self.config.cache_provider == "redis" and self.cache_backend:
                # Redis handles TTL automatically
                cleanup_stats["entries_remaining"] = await self.cache_backend.dbsize()
            else:
                # Clean up memory cache
                current_time = datetime.now()
                expired_keys = []
                
                for key, timestamp in self.cache_timestamps.items():
                    expiry_time = timestamp + timedelta(seconds=self.config.cache_ttl)
                    if current_time >= expiry_time:
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    if key in self.memory_cache:
                        del self.memory_cache[key]
                    if key in self.cache_timestamps:
                        del self.cache_timestamps[key]
                
                cleanup_stats["entries_removed"] = len(expired_keys)
                cleanup_stats["entries_remaining"] = len(self.memory_cache)
            
            logger.info(f"Cache cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")
            return {"error": str(e)}
    
    async def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """Clear all or specific type of cache entries"""
        try:
            if self.config.cache_provider == "redis" and self.cache_backend:
                if cache_type:
                    # Clear specific type
                    pattern = f"rag:{cache_type}:*"
                    keys = await self.cache_backend.keys(pattern)
                    if keys:
                        await self.cache_backend.delete(*keys)
                else:
                    # Clear all RAG cache entries
                    pattern = "rag:*"
                    keys = await self.cache_backend.keys(pattern)
                    if keys:
                        await self.cache_backend.delete(*keys)
            else:
                # Clear memory cache
                if cache_type:
                    keys_to_remove = [
                        key for key in self.memory_cache.keys()
                        if key.startswith(f"rag:{cache_type}:")
                    ]
                    for key in keys_to_remove:
                        del self.memory_cache[key]
                        if key in self.cache_timestamps:
                            del self.cache_timestamps[key]
                else:
                    self.memory_cache.clear()
                    self.cache_timestamps.clear()
            
            logger.info(f"Cache cleared: {cache_type or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = {
                "provider": self.config.cache_provider,
                "enabled": self.config.enable_cache,
                "ttl": self.config.cache_ttl,
                "initialized": self.is_initialized
            }
            
            if self.config.cache_provider == "redis" and self.cache_backend:
                # Get Redis stats
                info = await self.cache_backend.info()
                stats.update({
                    "total_entries": info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0),
                    "memory_usage": info.get("used_memory_human", "unknown"),
                    "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                })
            else:
                # Memory cache stats
                stats.update({
                    "total_entries": len(self.memory_cache),
                    "memory_usage": "unknown",
                    "hit_rate": "unknown"
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the cache manager"""
        try:
            if self.cache_backend:
                await self.cache_backend.close()
            
            # Clear memory cache
            self.memory_cache.clear()
            self.cache_timestamps.clear()
            
            self.is_initialized = False
            logger.info("Cache Manager closed")
            
        except Exception as e:
            logger.error(f"Error closing Cache Manager: {str(e)}")
            raise