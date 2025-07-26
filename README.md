# 🤖 AI Content-Aware Storage with RAG

An advanced AI-powered document storage and retrieval system with Retrieval-Augmented Generation (RAG) capabilities. This system allows you to upload documents, process them for semantic search, and ask intelligent questions about your content.

## ✨ Features

### 🔧 Core Capabilities
- **Multi-format Document Processing**: Supports PDF, DOCX, TXT, MD, CSV, XLSX, HTML, and more
- **Advanced RAG System**: Question-answering with context from your documents
- **Hybrid Search**: Combines semantic and keyword-based search for optimal results
- **Vector Storage**: Uses ChromaDB for efficient similarity search
- **Content-Aware Processing**: Intelligent document chunking and metadata extraction
- **Conversation Memory**: Maintains context across multiple interactions

### 🚀 Advanced Features
- **Duplicate Detection**: Prevents storing identical content multiple times
- **Document Similarity**: Find documents similar to a given document
- **Auto-summarization**: Generate summaries of documents
- **Question Generation**: Automatically generate potential questions from documents
- **Analytics Dashboard**: Track usage patterns and search performance
- **Caching System**: Redis-based caching for improved performance
- **Web Interface**: Built-in web UI for easy interaction

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PostgreSQL (for document metadata)
- Redis (for caching)
- OpenAI API key

### Quick Setup

1. **Clone and Setup**
```bash
git clone <repository-url>
cd ai-storage-rag
pip install -r requirements.txt
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Required Environment Variables**
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/ai_storage_db
REDIS_URL=redis://localhost:6379/0

# Vector Database Configuration
CHROMA_PERSIST_DIR=./chroma_db

# Application Configuration
SECRET_KEY=your_super_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Storage Configuration
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=50000000  # 50MB in bytes

# RAG Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_DOCS=5

# Celery Configuration (for background processing)
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

4. **Database Setup**
```bash
# Create PostgreSQL database
createdb ai_storage_db

# The application will automatically create tables on first run
```

5. **Start the Application**
```bash
python main.py
```

The application will be available at `http://localhost:8000`

## 🎯 Usage

### Web Interface
Visit `http://localhost:8000/ui` for the built-in web interface.

### API Endpoints

#### Document Management
```bash
# Upload a document
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# List documents
curl "http://localhost:8000/documents?skip=0&limit=10"

# Get document details
curl "http://localhost:8000/documents/1"

# Delete document
curl -X DELETE "http://localhost:8000/documents/1"
```

#### RAG Operations
```bash
# Ask a question
curl -X POST "http://localhost:8000/rag/question" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "search_type": "hybrid"}'

# Start a conversation
curl -X POST "http://localhost:8000/sessions/new"

curl -X POST "http://localhost:8000/rag/conversation" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello, what can you tell me?", "session_id": "your-session-id"}'

# Summarize a document
curl -X POST "http://localhost:8000/rag/summarize" \
  -H "Content-Type: application/json" \
  -d '{"document_id": 1, "summary_type": "comprehensive"}'
```

#### Search Operations
```bash
# Search documents
curl "http://localhost:8000/search?query=artificial%20intelligence&search_type=hybrid&k=5"

# Find similar documents
curl "http://localhost:8000/documents/1/similar?k=5"
```

#### Analytics
```bash
# Get collection statistics
curl "http://localhost:8000/stats/collection"

# Get search analytics
curl "http://localhost:8000/stats/search"
```

## 🏗️ Architecture

### System Components

1. **Document Processor** (`app/document_processor.py`)
   - Extracts text from various file formats
   - Creates intelligent document chunks
   - Generates metadata and content analysis

2. **Vector Store Manager** (`app/vector_store.py`)
   - Manages ChromaDB collections
   - Handles embedding generation
   - Implements hybrid search algorithms

3. **RAG Engine** (`app/rag_engine.py`)
   - Orchestrates question-answering
   - Manages conversation context
   - Generates summaries and questions

4. **Service Layer** (`app/services.py`)
   - Business logic for document operations
   - Handles file uploads and processing
   - Manages document lifecycle

5. **Database Models** (`app/database.py`)
   - PostgreSQL models for metadata
   - Redis caching utilities
   - Session management

### Data Flow

1. **Document Upload**: File → Text Extraction → Chunking → Vector Storage → Database
2. **Question Processing**: Query → Vector Search → Context Retrieval → LLM Generation → Response
3. **Conversation**: Session Management → Context Preservation → Multi-turn Dialogue

## 🔧 Configuration

### Embedding Models
The system uses HuggingFace sentence transformers. You can change the model in `.env`:
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast, good quality
# EMBEDDING_MODEL=all-mpnet-base-v2  # Better quality, slower
# EMBEDDING_MODEL=multi-qa-MiniLM-L6-cos-v1  # Optimized for Q&A
```

### Chunking Strategy
Adjust document chunking parameters:
```env
CHUNK_SIZE=1000        # Characters per chunk
CHUNK_OVERLAP=200      # Overlap between chunks
```

### Search Configuration
```env
MAX_RETRIEVAL_DOCS=5   # Maximum documents to retrieve for RAG
```

## 📊 Supported File Types

- **Text**: `.txt`, `.md`, `.py`, `.js`, `.css`, `.json`, `.xml`
- **Documents**: `.pdf`, `.docx`, `.doc`
- **Spreadsheets**: `.xlsx`, `.xls`, `.csv`
- **Web**: `.html`

## 🔍 Search Types

1. **Semantic Search**: Uses vector similarity for conceptual matching
2. **Hybrid Search**: Combines semantic and keyword search for optimal results

## 💡 Advanced Usage

### Custom Collections
Organize documents into collections:
```python
# Upload to specific collection
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf" \
  -F "collection_name=research_papers"
```

### Conversation Sessions
Maintain context across multiple questions:
```python
# Create session
session_response = requests.post("http://localhost:8000/sessions/new")
session_id = session_response.json()["session_id"]

# Use session for questions
requests.post("http://localhost:8000/rag/conversation", json={
    "question": "What is machine learning?",
    "session_id": session_id
})
```

### Document Analysis
```python
# Generate summary
requests.post("http://localhost:8000/rag/summarize", json={
    "document_id": 1,
    "summary_type": "brief"  # or "detailed", "comprehensive"
})

# Generate questions
requests.post("http://localhost:8000/rag/generate-questions", json={
    "document_id": 1,
    "num_questions": 10
})
```

## 🚀 Performance Optimization

### Caching
- Vector embeddings are cached in Redis
- Search results are cached for faster repeated queries
- Document content is cached after processing

### Background Processing
- Document processing happens asynchronously
- Large files are processed in chunks
- Vector storage is optimized for batch operations

## 🔒 Security Considerations

- File uploads are validated for size and type
- Content hashing prevents duplicate storage
- API endpoints include proper error handling
- Database queries use parameterized statements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **ChromaDB Errors**: Ensure the `chroma_db` directory is writable
2. **OpenAI API Errors**: Check your API key and rate limits
3. **Database Connection**: Verify PostgreSQL is running and accessible
4. **Redis Connection**: Ensure Redis server is running
5. **File Upload Errors**: Check file size limits and upload directory permissions

### Performance Tips

1. Use appropriate chunk sizes for your document types
2. Consider using faster embedding models for large collections
3. Implement document preprocessing for better text extraction
4. Monitor Redis memory usage for caching
5. Use database indexing for better query performance

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation at `http://localhost:8000/docs`
- Create an issue in the repository

---

**Built with ❤️ using FastAPI, LangChain, ChromaDB, and OpenAI**