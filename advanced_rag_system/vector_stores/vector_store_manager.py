"""
Vector Store Manager - Handles multiple vector database backends
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from loguru import logger

from ..utils.config import VectorStoreConfig, VECTOR_STORE_CONFIGS
from ..embeddings.embedding_manager import EmbeddingManager

class BaseVectorStore(ABC):
    """Base class for all vector store implementations"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self):
        """Initialize the vector store"""
        pass
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Add documents with embeddings"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the vector store connection"""
        pass

class ChromaDBStore(BaseVectorStore):
    """ChromaDB implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.client = None
        self.collection = None
    
    async def initialize(self):
        """Initialize ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Get ChromaDB specific config
            chroma_config = VECTOR_STORE_CONFIGS.get("chromadb", {})
            
            # Create client
            self.client = chromadb.PersistentClient(
                path=chroma_config.get("persist_directory", "./chroma_db"),
                settings=Settings(**chroma_config.get("client_settings", {}))
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.similarity_metric}
            )
            
            self.is_initialized = True
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Add documents to ChromaDB"""
        try:
            ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
            documents_text = [doc.get("content", "") for doc in documents]
            
            self.collection.add(
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadata,
                ids=ids
            )
            
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB"""
        try:
            where_clause = filters if filters else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        try:
            self.collection.delete(ids=document_ids)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        try:
            count = self.collection.count()
            return {
                "provider": "chromadb",
                "collection_name": self.config.collection_name,
                "document_count": count,
                "initialized": self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"Error getting ChromaDB stats: {str(e)}")
            return {"provider": "chromadb", "error": str(e)}
    
    async def close(self):
        """Close ChromaDB connection"""
        try:
            # ChromaDB doesn't require explicit closing
            self.is_initialized = False
            logger.info("ChromaDB connection closed")
            
        except Exception as e:
            logger.error(f"Error closing ChromaDB: {str(e)}")

class PineconeStore(BaseVectorStore):
    """Pinecone implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.index = None
    
    async def initialize(self):
        """Initialize Pinecone"""
        try:
            import pinecone
            
            pinecone_config = VECTOR_STORE_CONFIGS.get("pinecone", {})
            api_key = pinecone_config.get("api_key")
            environment = pinecone_config.get("environment")
            
            if not api_key:
                raise ValueError("Pinecone API key not provided")
            
            pinecone.init(api_key=api_key, environment=environment)
            
            # Get or create index
            if self.config.collection_name not in pinecone.list_indexes():
                pinecone.create_index(
                    self.config.collection_name,
                    dimension=self.config.embedding_dimension,
                    metric=self.config.similarity_metric
                )
            
            self.index = pinecone.Index(self.config.collection_name)
            self.is_initialized = True
            logger.info("Pinecone initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Add documents to Pinecone"""
        try:
            vectors = []
            ids = []
            
            for i, (doc, embedding, meta) in enumerate(zip(documents, embeddings, metadata)):
                doc_id = doc.get("id", f"doc_{i}")
                ids.append(doc_id)
                
                # Combine content with metadata for Pinecone
                vector_metadata = {
                    **meta,
                    "content": doc.get("content", "")
                }
                
                vectors.append((doc_id, embedding, vector_metadata))
            
            self.index.upsert(vectors=vectors)
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            raise
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search Pinecone"""
        try:
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filters,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in query_response["matches"]:
                formatted_results.append({
                    "id": match["id"],
                    "content": match["metadata"].get("content", ""),
                    "metadata": match["metadata"],
                    "score": match["score"]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Pinecone"""
        try:
            self.index.delete(ids=document_ids)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from Pinecone: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "provider": "pinecone",
                "collection_name": self.config.collection_name,
                "document_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "initialized": self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {str(e)}")
            return {"provider": "pinecone", "error": str(e)}
    
    async def close(self):
        """Close Pinecone connection"""
        try:
            # Pinecone doesn't require explicit closing
            self.is_initialized = False
            logger.info("Pinecone connection closed")
            
        except Exception as e:
            logger.error(f"Error closing Pinecone: {str(e)}")

class VectorStoreManager:
    """
    Manager class for vector stores with multiple backend support
    """
    
    def __init__(
        self,
        config: VectorStoreConfig,
        embedding_manager: EmbeddingManager
    ):
        self.config = config
        self.embedding_manager = embedding_manager
        self.vector_store = None
        self.is_initialized = False
        
        # Store factory
        self.store_factories = {
            "chromadb": ChromaDBStore,
            "pinecone": PineconeStore,
            # Add more stores as needed
        }
    
    async def initialize(self):
        """Initialize the vector store manager"""
        if self.is_initialized:
            return
        
        try:
            provider = self.config.provider.lower()
            
            if provider not in self.store_factories:
                raise ValueError(f"Unsupported vector store provider: {provider}")
            
            # Create and initialize vector store
            store_class = self.store_factories[provider]
            self.vector_store = store_class(self.config)
            await self.vector_store.initialize()
            
            self.is_initialized = True
            logger.info(f"Vector Store Manager initialized with {provider}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store Manager: {str(e)}")
            raise
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add documents to the vector store"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Extract text content for embedding
            texts = [doc.get("content", "") for doc in documents]
            
            # Generate embeddings
            embeddings = await self.embedding_manager.embed_texts(texts)
            
            # Use provided metadata or create default
            if metadata is None:
                metadata = [doc.get("metadata", {}) for doc in documents]
            
            # Add to vector store
            document_ids = await self.vector_store.add_documents(
                documents, embeddings, metadata
            )
            
            logger.info(f"Added {len(document_ids)} documents to vector store")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search the vector store"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_manager.embed_text(query)
            
            # Search vector store
            results = await self.vector_store.search(
                query_embedding, top_k, filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results
                if result.get("score", 0) >= similarity_threshold
            ]
            
            logger.info(f"Found {len(filtered_results)} documents above threshold")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    async def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search using pre-computed embedding"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            results = await self.vector_store.search(
                query_embedding, top_k, filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results
                if result.get("score", 0) >= similarity_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a single document"""
        return await self.delete_documents([document_id])
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete multiple documents"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            success = await self.vector_store.delete_documents(document_ids)
            if success:
                logger.info(f"Deleted {len(document_ids)} documents")
            return success
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.is_initialized:
            return {"initialized": False}
        
        try:
            stats = await self.vector_store.get_stats()
            stats.update({
                "embedding_model": self.embedding_manager.get_model_info(),
                "manager_initialized": self.is_initialized
            })
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
    
    async def optimize(self) -> Dict[str, Any]:
        """Optimize the vector store (if supported)"""
        try:
            # Implementation depends on the vector store
            # For now, return basic optimization info
            stats = await self.get_stats()
            
            optimization_results = {
                "optimization_performed": False,
                "current_stats": stats,
                "recommendations": []
            }
            
            # Add recommendations based on stats
            doc_count = stats.get("document_count", 0)
            if doc_count > 10000:
                optimization_results["recommendations"].append(
                    "Consider partitioning data for better performance"
                )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing vector store: {str(e)}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the vector store manager"""
        try:
            if self.vector_store:
                await self.vector_store.close()
            
            self.is_initialized = False
            logger.info("Vector Store Manager closed")
            
        except Exception as e:
            logger.error(f"Error closing Vector Store Manager: {str(e)}")
            raise