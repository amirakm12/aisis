import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uvicorn

from app.database import get_db
from app.services import document_service
from app.rag_engine import rag_engine
from app.config import settings

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None
    search_type: Optional[str] = "hybrid"
    k: Optional[int] = None
    session_id: Optional[str] = None

class ConversationRequest(BaseModel):
    question: str
    session_id: str
    collection_name: Optional[str] = None
    k: Optional[int] = None

class SummaryRequest(BaseModel):
    document_id: int
    collection_name: Optional[str] = None
    summary_type: Optional[str] = "comprehensive"

class QuestionsRequest(BaseModel):
    document_id: int
    collection_name: Optional[str] = None
    num_questions: Optional[int] = 5

# Initialize FastAPI app
app = FastAPI(
    title="AI Content-Aware Storage with RAG",
    description="Advanced AI-powered document storage and retrieval system with RAG capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
except:
    # Create directories if they don't exist
    import os
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    templates = None

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Content-Aware Storage with RAG API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Storage with RAG"}

# Document Management Endpoints
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a document and process it for RAG"""
    return await document_service.upload_document(file, db, collection_name)

@app.get("/documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    file_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """List all documents with pagination"""
    return await document_service.list_documents(db, skip, limit, file_type)

@app.get("/documents/{document_id}")
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get document information by ID"""
    return await document_service.get_document(document_id, db)

@app.get("/documents/{document_id}/content")
async def get_document_content(document_id: int, db: Session = Depends(get_db)):
    """Get full document content"""
    content = await document_service.get_document_content(document_id, db)
    return {"document_id": document_id, "content": content}

@app.get("/documents/{document_id}/similar")
async def get_similar_documents(
    document_id: int,
    k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """Find documents similar to the given document"""
    return await document_service.get_similar_documents(document_id, db, k)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document and all associated data"""
    return await document_service.delete_document(document_id, db)

# RAG Endpoints
@app.post("/rag/question")
async def ask_question(request: QuestionRequest):
    """Ask a question using RAG"""
    return await rag_engine.answer_question(
        question=request.question,
        collection_name=request.collection_name,
        search_type=request.search_type,
        k=request.k,
        session_id=request.session_id
    )

@app.post("/rag/conversation")
async def conversational_qa(request: ConversationRequest):
    """Have a conversational Q&A with memory"""
    return await rag_engine.conversational_qa(
        question=request.question,
        session_id=request.session_id,
        collection_name=request.collection_name,
        k=request.k
    )

@app.post("/rag/summarize")
async def summarize_document(request: SummaryRequest):
    """Generate a summary of a document"""
    return await rag_engine.summarize_document(
        document_id=request.document_id,
        collection_name=request.collection_name,
        summary_type=request.summary_type
    )

@app.post("/rag/generate-questions")
async def generate_questions(request: QuestionsRequest):
    """Generate potential questions for a document"""
    return await rag_engine.generate_questions(
        document_id=request.document_id,
        collection_name=request.collection_name,
        num_questions=request.num_questions
    )

# Search Endpoints
@app.get("/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    collection_name: Optional[str] = Query(None),
    search_type: str = Query("hybrid", regex="^(semantic|hybrid)$"),
    k: int = Query(5, ge=1, le=20)
):
    """Search documents using semantic or hybrid search"""
    from app.vector_store import vector_store_manager
    
    collection_name = collection_name or "documents"
    
    if search_type == "hybrid":
        results = await vector_store_manager.hybrid_search(collection_name, query, k)
    else:
        results = await vector_store_manager.similarity_search(collection_name, query, k)
    
    return {
        "query": query,
        "search_type": search_type,
        "results": results,
        "total_results": len(results)
    }

# Analytics and Statistics Endpoints
@app.get("/stats/collection")
async def get_collection_stats(collection_name: Optional[str] = Query(None)):
    """Get collection statistics"""
    return await document_service.get_collection_stats(collection_name)

@app.get("/stats/search")
async def get_search_stats(db: Session = Depends(get_db)):
    """Get search analytics"""
    from app.database import SearchQuery
    from sqlalchemy import func
    
    # Get search statistics
    total_queries = db.query(SearchQuery).count()
    avg_response_time = db.query(func.avg(SearchQuery.response_time)).scalar() or 0
    avg_results = db.query(func.avg(SearchQuery.results_count)).scalar() or 0
    
    # Popular queries
    popular_queries = db.query(
        SearchQuery.query_text,
        func.count(SearchQuery.id).label('count')
    ).group_by(SearchQuery.query_text).order_by(
        func.count(SearchQuery.id).desc()
    ).limit(10).all()
    
    return {
        "total_queries": total_queries,
        "average_response_time": round(avg_response_time, 3),
        "average_results_count": round(avg_results, 1),
        "popular_queries": [{"query": q[0], "count": q[1]} for q in popular_queries]
    }

# Session Management
@app.post("/sessions/new")
async def create_session():
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get session information"""
    from app.database import RAGSession
    import json
    
    session = db.query(RAGSession).filter(RAGSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "last_interaction": session.last_interaction,
        "total_queries": session.total_queries,
        "conversation_history": json.loads(session.conversation_history) if session.conversation_history else [],
        "context_documents": json.loads(session.context_documents) if session.context_documents else []
    }

# Web Interface (if templates exist)
@app.get("/ui", response_class=HTMLResponse)
async def web_interface():
    """Simple web interface for the RAG system"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Content-Aware Storage with RAG</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .section h2 { color: #555; margin-top: 0; }
            input, textarea, select, button { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; }
            button { background-color: #007bff; color: white; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px; }
            .upload-area { border: 2px dashed #ddd; padding: 20px; text-align: center; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Content-Aware Storage with RAG</h1>
            
            <div class="section">
                <h2>📁 Upload Document</h2>
                <div class="upload-area">
                    <input type="file" id="fileInput" accept=".pdf,.docx,.txt,.md,.csv,.xlsx">
                    <button onclick="uploadDocument()">Upload Document</button>
                </div>
                <div id="uploadResult" class="result" style="display:none;"></div>
            </div>
            
            <div class="section">
                <h2>❓ Ask Questions</h2>
                <textarea id="questionInput" placeholder="Ask a question about your documents..." rows="3" style="width: 70%;"></textarea>
                <br>
                <select id="searchType">
                    <option value="hybrid">Hybrid Search</option>
                    <option value="semantic">Semantic Search</option>
                </select>
                <button onclick="askQuestion()">Ask Question</button>
                <div id="questionResult" class="result" style="display:none;"></div>
            </div>
            
            <div class="section">
                <h2>🔍 Search Documents</h2>
                <input type="text" id="searchInput" placeholder="Search query..." style="width: 60%;">
                <button onclick="searchDocuments()">Search</button>
                <div id="searchResult" class="result" style="display:none;"></div>
            </div>
            
            <div class="section">
                <h2>📊 Statistics</h2>
                <button onclick="getStats()">Get Collection Stats</button>
                <div id="statsResult" class="result" style="display:none;"></div>
            </div>
        </div>
        
        <script>
            async function uploadDocument() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) {
                    alert('Please select a file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/documents/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    document.getElementById('uploadResult').style.display = 'block';
                    document.getElementById('uploadResult').innerHTML = `
                        <strong>Upload Result:</strong><br>
                        ${result.message}<br>
                        Document ID: ${result.document_id}<br>
                        File Size: ${result.file_size} bytes
                    `;
                } catch (error) {
                    document.getElementById('uploadResult').style.display = 'block';
                    document.getElementById('uploadResult').innerHTML = `<strong>Error:</strong> ${error.message}`;
                }
            }
            
            async function askQuestion() {
                const question = document.getElementById('questionInput').value;
                const searchType = document.getElementById('searchType').value;
                
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                try {
                    const response = await fetch('/rag/question', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: question,
                            search_type: searchType
                        })
                    });
                    const result = await response.json();
                    document.getElementById('questionResult').style.display = 'block';
                    document.getElementById('questionResult').innerHTML = `
                        <strong>Answer:</strong><br>
                        ${result.answer}<br><br>
                        <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%<br>
                        <strong>Sources:</strong> ${result.sources.length} documents<br>
                        <strong>Response Time:</strong> ${result.response_time.toFixed(3)}s
                    `;
                } catch (error) {
                    document.getElementById('questionResult').style.display = 'block';
                    document.getElementById('questionResult').innerHTML = `<strong>Error:</strong> ${error.message}`;
                }
            }
            
            async function searchDocuments() {
                const query = document.getElementById('searchInput').value;
                
                if (!query) {
                    alert('Please enter a search query');
                    return;
                }
                
                try {
                    const response = await fetch(`/search?query=${encodeURIComponent(query)}&search_type=hybrid`);
                    const result = await response.json();
                    
                    let resultsHtml = `<strong>Search Results (${result.total_results}):</strong><br><br>`;
                    result.results.forEach((item, index) => {
                        resultsHtml += `
                            <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #007bff;">
                                <strong>Result ${index + 1}</strong> (Score: ${(item.similarity_score * 100).toFixed(1)}%)<br>
                                ${item.content.substring(0, 200)}...
                            </div>
                        `;
                    });
                    
                    document.getElementById('searchResult').style.display = 'block';
                    document.getElementById('searchResult').innerHTML = resultsHtml;
                } catch (error) {
                    document.getElementById('searchResult').style.display = 'block';
                    document.getElementById('searchResult').innerHTML = `<strong>Error:</strong> ${error.message}`;
                }
            }
            
            async function getStats() {
                try {
                    const response = await fetch('/stats/collection');
                    const result = await response.json();
                    
                    document.getElementById('statsResult').style.display = 'block';
                    document.getElementById('statsResult').innerHTML = `
                        <strong>Collection Statistics:</strong><br>
                        Total Documents: ${result.total_documents}<br>
                        Processed Documents: ${result.processed_documents}<br>
                        Total Chunks: ${result.total_chunks}<br>
                        Embedding Model: ${result.vector_store_stats.embedding_model}<br>
                        File Types: ${JSON.stringify(result.file_type_distribution)}
                    `;
                } catch (error) {
                    document.getElementById('statsResult').style.display = 'block';
                    document.getElementById('statsResult').innerHTML = `<strong>Error:</strong> ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )