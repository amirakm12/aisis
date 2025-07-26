"""
Vector Store Manager - Manages different vector database backends
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from ..utils.config import VectorStoreConfig
from ..embeddings.embedding_manager import EmbeddingManager


@dataclass
class Document:
    """Document representation for vector storage"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Search result from vector store"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store"""
        pass
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass
    
    @abstractmethod
    async def optimize(self) -> Dict[str, Any]:
        """Optimize the vector store"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the vector store"""
        pass


class ChromaDBStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig, embedding_manager: EmbeddingManager):
        self.config = config
        self.embedding_manager = embedding_manager
        self.client = None
        self.collection = None
    
    async def initialize(self) -> None:
        """Initialize ChromaDB connection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            
        except ImportError:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
    
    async def add_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents to ChromaDB"""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = await self.embedding_manager.embed_texts(texts)
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search ChromaDB for similar documents"""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")
        
        # Generate query embedding
        query_embedding = await self.embedding_manager.embed_text(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Convert to SearchResult format
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                document_id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                similarity_score=results['distances'][0][i]
            ))
        
        return search_results
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")
        
        try:
            self.collection.delete(ids=document_ids)
            return True
        except Exception:
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        if not self.collection:
            return {"status": "not_initialized"}
        
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.config.collection_name,
                "status": "active"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def optimize(self) -> Dict[str, Any]:
        """Optimize ChromaDB collection"""
        # ChromaDB handles optimization automatically
        return {"status": "optimized", "message": "ChromaDB auto-optimized"}
    
    async def shutdown(self) -> None:
        """Shutdown ChromaDB connection"""
        self.client = None
        self.collection = None


class PineconeStore(BaseVectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig, embedding_manager: EmbeddingManager):
        self.config = config
        self.embedding_manager = embedding_manager
        self.index = None
    
    async def initialize(self) -> None:
        """Initialize Pinecone connection"""
        try:
            import pinecone
            
            pinecone.init(
                api_key=self.config.api_key,
                environment=self.config.environment
            )
            
            # Get or create index
            index_name = self.config.index_name or "documents"
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=self.config.dimension,
                    metric=self.config.distance_metric
                )
            
            self.index = pinecone.Index(index_name)
            
        except ImportError:
            raise ImportError("Pinecone is not installed. Install with: pip install pinecone-client")
    
    async def add_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents to Pinecone"""
        if not self.index:
            raise RuntimeError("Pinecone not initialized")
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = await self.embedding_manager.embed_texts(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, doc in enumerate(documents):
            vectors.append({
                "id": doc.id,
                "values": embeddings[i],
                "metadata": {**doc.metadata, **(metadata or {})}
            })
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        
        return [doc.id for doc in documents]
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Pinecone for similar documents"""
        if not self.index:
            raise RuntimeError("Pinecone not initialized")
        
        # Generate query embedding
        query_embedding = await self.embedding_manager.embed_text(query)
        
        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_metadata
        )
        
        # Convert to SearchResult format
        search_results = []
        for match in results.matches:
            search_results.append(SearchResult(
                document_id=match.id,
                content=match.metadata.get("content", ""),
                metadata=match.metadata,
                similarity_score=match.score
            ))
        
        return search_results
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Pinecone"""
        if not self.index:
            raise RuntimeError("Pinecone not initialized")
        
        try:
            self.index.delete(ids=document_ids)
            return True
        except Exception:
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics"""
        if not self.index:
            return {"status": "not_initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_name": self.config.index_name,
                "status": "active"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def optimize(self) -> Dict[str, Any]:
        """Optimize Pinecone index"""
        # Pinecone handles optimization automatically
        return {"status": "optimized", "message": "Pinecone auto-optimized"}
    
    async def shutdown(self) -> None:
        """Shutdown Pinecone connection"""
        self.index = None


class VectorStoreManager:
    """Manager for vector store operations"""
    
    def __init__(self, config: VectorStoreConfig, embedding_manager: EmbeddingManager):
        self.config = config
        self.embedding_manager = embedding_manager
        self.vector_store: Optional[BaseVectorStore] = None
    
    async def initialize(self) -> None:
        """Initialize the appropriate vector store"""
        if self.config.type.value == "chroma":
            self.vector_store = ChromaDBStore(self.config, self.embedding_manager)
        elif self.config.type.value == "pinecone":
            self.vector_store = PineconeStore(self.config, self.embedding_manager)
        else:
            # Default to ChromaDB
            self.vector_store = ChromaDBStore(self.config, self.embedding_manager)
        
        await self.vector_store.initialize()
    
    async def add_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents to the vector store"""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        return await self.vector_store.add_documents(documents, metadata)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search the vector store"""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        return await self.vector_store.search(query, top_k, filter_metadata)
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store"""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        return await self.vector_store.delete_documents(document_ids)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        return await self.vector_store.get_stats()
    
    async def optimize(self) -> Dict[str, Any]:
        """Optimize the vector store"""
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        return await self.vector_store.optimize()
    
    async def shutdown(self) -> None:
        """Shutdown the vector store"""
        if self.vector_store:
            await self.vector_store.shutdown() 