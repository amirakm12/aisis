import os
import json
import uuid
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from app.database import Document, DocumentChunk, get_db
from app.document_processor import document_processor
from app.vector_store import vector_store_manager
from app.config import settings


class DocumentService:
    def __init__(self):
        self.default_collection = "documents"
    
    async def upload_document(
        self,
        file: UploadFile,
        db: Session,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """Upload and process a document"""
        collection_name = collection_name or self.default_collection
        
        try:
            # Validate file size
            if file.size > settings.max_file_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds maximum allowed size of {settings.max_file_size} bytes"
                )
            
            # Generate unique filename
            file_extension = Path(file.filename).suffix.lower()
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(settings.upload_dir, unique_filename)
            
            # Save file to disk
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract text content
            content = await document_processor.extract_text_from_file(
                file_path, file_extension[1:]  # Remove the dot
            )
            
            # Calculate content hash to detect duplicates
            content_hash = document_processor.calculate_content_hash(content)
            
            # Check for duplicate content
            existing_doc = db.query(Document).filter(Document.content_hash == content_hash).first()
            if existing_doc:
                # Remove the uploaded file since it's a duplicate
                os.remove(file_path)
                return {
                    "message": "Document with identical content already exists",
                    "document_id": existing_doc.id,
                    "is_duplicate": True
                }
            
            # Extract metadata
            metadata = document_processor.extract_metadata(file_path, content)
            
            # Analyze content
            content_analysis = await document_processor.analyze_content_type(content)
            metadata.update(content_analysis)
            
            # Create content preview
            content_preview = document_processor.create_content_preview(content)
            
            # Create document record
            document = Document(
                filename=unique_filename,
                original_filename=file.filename,
                file_path=file_path,
                file_type=file_extension[1:],
                file_size=file.size,
                content_hash=content_hash,
                content_preview=content_preview,
                metadata=json.dumps(metadata),
                vector_store_id=collection_name
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Process document for vector storage (background task)
            await self._process_document_for_rag(document, content, collection_name, db)
            
            return {
                "message": "Document uploaded and processed successfully",
                "document_id": document.id,
                "filename": document.original_filename,
                "file_size": document.file_size,
                "content_preview": content_preview,
                "metadata": metadata,
                "is_duplicate": False
            }
            
        except Exception as e:
            # Clean up file if error occurs
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    async def _process_document_for_rag(
        self,
        document: Document,
        content: str,
        collection_name: str,
        db: Session
    ):
        """Process document for RAG (vector storage)"""
        try:
            # Create or get collection
            await vector_store_manager.create_collection(
                collection_name,
                metadata={"description": "Document storage collection"}
            )
            
            # Create document chunks
            doc_metadata = {
                "document_id": document.id,
                "filename": document.original_filename,
                "file_type": document.file_type,
                "upload_time": document.upload_time.isoformat()
            }
            
            chunks = document_processor.create_chunks(content, doc_metadata)
            
            if chunks:
                # Add chunks to vector store
                chunk_ids = await vector_store_manager.add_documents(
                    collection_name, chunks, document.id
                )
                
                # Store chunk information in database
                for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
                    chunk_record = DocumentChunk(
                        document_id=document.id,
                        chunk_index=i,
                        content=chunk.page_content,
                        content_hash=document_processor.calculate_content_hash(chunk.page_content),
                        embedding_model=settings.embedding_model,
                        chunk_metadata=json.dumps(chunk.metadata)
                    )
                    db.add(chunk_record)
                
                # Mark document as processed
                document.is_processed = True
                db.commit()
            
        except Exception as e:
            print(f"Error processing document for RAG: {str(e)}")
            # Don't raise exception here to avoid failing the upload
    
    async def get_document(self, document_id: int, db: Session) -> Dict[str, Any]:
        """Get document information"""
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update access statistics
        document.access_count += 1
        document.last_accessed = datetime.now()
        db.commit()
        
        # Get chunk count
        chunk_count = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).count()
        
        return {
            "id": document.id,
            "filename": document.original_filename,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "upload_time": document.upload_time,
            "last_accessed": document.last_accessed,
            "access_count": document.access_count,
            "content_preview": document.content_preview,
            "metadata": json.loads(document.metadata) if document.metadata else {},
            "is_processed": document.is_processed,
            "chunk_count": chunk_count
        }
    
    async def list_documents(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        file_type: str = None
    ) -> Dict[str, Any]:
        """List documents with pagination"""
        query = db.query(Document)
        
        if file_type:
            query = query.filter(Document.file_type == file_type)
        
        total = query.count()
        documents = query.offset(skip).limit(limit).all()
        
        document_list = []
        for doc in documents:
            document_list.append({
                "id": doc.id,
                "filename": doc.original_filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "upload_time": doc.upload_time,
                "access_count": doc.access_count,
                "content_preview": doc.content_preview[:200] + "..." if len(doc.content_preview) > 200 else doc.content_preview,
                "is_processed": doc.is_processed
            })
        
        return {
            "documents": document_list,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    
    async def delete_document(self, document_id: int, db: Session) -> Dict[str, Any]:
        """Delete a document and its associated data"""
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        try:
            # Delete from vector store
            if document.vector_store_id:
                await vector_store_manager.delete_document(
                    document.vector_store_id, document_id
                )
            
            # Delete chunks from database
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            
            # Delete file from disk
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
            
            # Delete document record
            db.delete(document)
            db.commit()
            
            return {"message": "Document deleted successfully"}
            
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
    
    async def get_document_content(self, document_id: int, db: Session) -> str:
        """Get full document content"""
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        try:
            # Get content from file
            content = await document_processor.extract_text_from_file(
                document.file_path, document.file_type
            )
            return content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading document content: {str(e)}")
    
    async def get_similar_documents(
        self,
        document_id: int,
        db: Session,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find documents similar to the given document"""
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        try:
            # Get similar documents from vector store
            similar_docs = await vector_store_manager.get_similar_documents(
                document.vector_store_id or self.default_collection,
                document_id,
                k
            )
            
            # Enrich with database information
            enriched_docs = []
            for sim_doc in similar_docs:
                doc_id = sim_doc['document_id']
                db_doc = db.query(Document).filter(Document.id == doc_id).first()
                if db_doc:
                    enriched_docs.append({
                        "id": db_doc.id,
                        "filename": db_doc.original_filename,
                        "file_type": db_doc.file_type,
                        "similarity_score": sim_doc['similarity_score'],
                        "content_preview": sim_doc['sample_content'],
                        "upload_time": db_doc.upload_time
                    })
            
            return enriched_docs
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error finding similar documents: {str(e)}")
    
    async def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        collection_name = collection_name or self.default_collection
        
        try:
            # Get vector store stats
            vector_stats = await vector_store_manager.get_collection_stats(collection_name)
            
            # Get database stats
            db = next(get_db())
            total_docs = db.query(Document).count()
            processed_docs = db.query(Document).filter(Document.is_processed == True).count()
            total_chunks = db.query(DocumentChunk).count()
            
            # File type distribution
            file_types = db.query(Document.file_type, db.func.count(Document.id)).group_by(Document.file_type).all()
            
            return {
                "collection_name": collection_name,
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "total_chunks": total_chunks,
                "vector_store_stats": vector_stats,
                "file_type_distribution": dict(file_types)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting collection stats: {str(e)}")


# Global service instance
document_service = DocumentService()