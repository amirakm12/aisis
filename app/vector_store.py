import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document as LangchainDocument
from langchain.vectorstores import Chroma
import numpy as np
from app.config import settings
from app.database import CacheManager


class VectorStoreManager:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Cache for frequently accessed embeddings
        self.cache_manager = CacheManager()
    
    async def create_collection(self, collection_name: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new collection for storing document embeddings"""
        try:
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            return collection_name
        except Exception as e:
            if "already exists" in str(e).lower():
                return collection_name
            raise ValueError(f"Error creating collection: {str(e)}")
    
    async def add_documents(
        self, 
        collection_name: str, 
        documents: List[LangchainDocument],
        document_id: int
    ) -> List[str]:
        """Add documents to the vector store"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Generate unique IDs for each chunk
            chunk_ids = [f"{document_id}_{i}" for i in range(len(documents))]
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = []
            
            for i, doc in enumerate(documents):
                metadata = doc.metadata.copy()
                metadata.update({
                    'document_id': document_id,
                    'chunk_id': chunk_ids[i],
                    'chunk_index': i
                })
                metadatas.append(metadata)
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            # Cache embeddings for quick access
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                cache_key = f"embedding:{chunk_id}"
                self.cache_manager.set_cache(
                    cache_key, 
                    json.dumps(embedding), 
                    expire=86400  # 24 hours
                )
            
            return chunk_ids
            
        except Exception as e:
            raise ValueError(f"Error adding documents to vector store: {str(e)}")
    
    async def similarity_search(
        self, 
        collection_name: str,
        query: str, 
        k: int = None,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search in the vector store"""
        try:
            k = k or settings.max_retrieval_docs
            
            # Check cache first
            cache_key = f"search:{hash(query)}:{collection_name}:{k}"
            cached_result = self.cache_manager.get_cache(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            collection = self.chroma_client.get_collection(collection_name)
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
            
            # Cache results
            self.cache_manager.set_cache(
                cache_key, 
                json.dumps(formatted_results, default=str), 
                expire=3600  # 1 hour
            )
            
            return formatted_results
            
        except Exception as e:
            raise ValueError(f"Error performing similarity search: {str(e)}")
    
    async def hybrid_search(
        self,
        collection_name: str,
        query: str,
        k: int = None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining keyword and semantic search"""
        try:
            k = k or settings.max_retrieval_docs
            
            # Semantic search
            semantic_results = await self.similarity_search(
                collection_name, query, k * 2, filter_metadata
            )
            
            # Simple keyword search (in a production system, you might use BM25)
            keyword_results = await self._keyword_search(
                collection_name, query, k * 2, filter_metadata
            )
            
            # Combine and rerank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, semantic_weight, keyword_weight
            )
            
            return combined_results[:k]
            
        except Exception as e:
            raise ValueError(f"Error performing hybrid search: {str(e)}")
    
    async def _keyword_search(
        self,
        collection_name: str,
        query: str,
        k: int,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Simple keyword-based search"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Get all documents (in production, you'd want to optimize this)
            all_results = collection.get(
                where=filter_metadata,
                include=['documents', 'metadatas']
            )
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            scored_results = []
            
            for i, doc in enumerate(all_results['documents']):
                doc_words = set(doc.lower().split())
                keyword_score = len(query_words.intersection(doc_words)) / len(query_words)
                
                if keyword_score > 0:
                    result = {
                        'id': all_results['ids'][i],
                        'content': doc,
                        'metadata': all_results['metadatas'][i],
                        'keyword_score': keyword_score,
                        'similarity_score': keyword_score
                    }
                    scored_results.append(result)
            
            # Sort by keyword score
            scored_results.sort(key=lambda x: x['keyword_score'], reverse=True)
            return scored_results[:k]
            
        except Exception as e:
            return []
    
    def _combine_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine and rerank search results from different methods"""
        
        # Create a map of all unique results
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            result_id = result['id']
            result_map[result_id] = result.copy()
            result_map[result_id]['combined_score'] = result['similarity_score'] * semantic_weight
        
        # Add keyword results
        for result in keyword_results:
            result_id = result['id']
            if result_id in result_map:
                # Combine scores
                result_map[result_id]['combined_score'] += result.get('keyword_score', 0) * keyword_weight
            else:
                result_map[result_id] = result.copy()
                result_map[result_id]['combined_score'] = result.get('keyword_score', 0) * keyword_weight
        
        # Sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_results
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            count = collection.count()
            
            return {
                'name': collection_name,
                'document_count': count,
                'embedding_model': settings.embedding_model,
                'chunk_size': settings.chunk_size,
                'chunk_overlap': settings.chunk_overlap
            }
        except Exception as e:
            raise ValueError(f"Error getting collection stats: {str(e)}")
    
    async def delete_document(self, collection_name: str, document_id: int) -> bool:
        """Delete all chunks of a document from the vector store"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Find all chunks for this document
            results = collection.get(
                where={"document_id": document_id},
                include=['ids']
            )
            
            if results['ids']:
                collection.delete(ids=results['ids'])
                
                # Clear cache for these chunks
                for chunk_id in results['ids']:
                    cache_key = f"embedding:{chunk_id}"
                    self.cache_manager.delete_cache(cache_key)
            
            return True
        except Exception as e:
            raise ValueError(f"Error deleting document from vector store: {str(e)}")
    
    async def get_similar_documents(
        self,
        collection_name: str,
        document_id: int,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Get the first chunk of the target document
            target_results = collection.get(
                where={"document_id": document_id},
                limit=1,
                include=['embeddings', 'documents', 'metadatas']
            )
            
            if not target_results['embeddings']:
                return []
            
            # Use the first chunk's embedding for similarity search
            target_embedding = target_results['embeddings'][0]
            
            # Search for similar documents (excluding the target document)
            results = collection.query(
                query_embeddings=[target_embedding],
                n_results=k + 10,  # Get more to filter out the target document
                include=['documents', 'metadatas', 'distances']
            )
            
            # Filter out chunks from the same document and group by document_id
            similar_docs = {}
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                doc_id = metadata.get('document_id')
                
                if doc_id != document_id and doc_id not in similar_docs:
                    similar_docs[doc_id] = {
                        'document_id': doc_id,
                        'similarity_score': 1 - results['distances'][0][i],
                        'sample_content': results['documents'][0][i][:200] + "...",
                        'metadata': metadata
                    }
                
                if len(similar_docs) >= k:
                    break
            
            return list(similar_docs.values())
            
        except Exception as e:
            raise ValueError(f"Error finding similar documents: {str(e)}")


# Global vector store manager instance
vector_store_manager = VectorStoreManager()