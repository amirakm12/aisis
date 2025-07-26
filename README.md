# Advanced RAG System with AI Agents & Vector Databases

A comprehensive, production-ready Retrieval-Augmented Generation (RAG) system featuring intelligent AI agents, multiple vector database support, and advanced optimization capabilities.

## 🚀 Features

### Core Capabilities
- **🤖 AI Agents**: Intelligent query processing, document analysis, and orchestration
- **🗄️ Multiple Vector Databases**: ChromaDB, Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch
- **🔍 Hybrid Retrieval**: Combines semantic search, keyword search, and advanced reranking
- **📄 Multi-Modal Processing**: PDF, DOCX, TXT, HTML, Markdown, JSON, XML support
- **⚡ Performance Optimization**: Caching, metrics collection, and system optimization
- **🔧 Flexible Configuration**: Environment-based and file-based configuration

### Advanced Features
- **Query Expansion & Routing**: Intelligent query understanding and enhancement
- **Smart Chunking**: Context-aware document segmentation
- **Reranking**: Cross-encoder models for improved relevance
- **Real-time Monitoring**: Comprehensive metrics and performance tracking
- **Caching Layer**: Redis and in-memory caching for improved performance
- **Batch Processing**: Concurrent document ingestion and processing

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for LLM functionality)
- Optional: Redis (for caching), Vector database credentials

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your configuration:

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo

# Vector Store Configuration
VECTOR_STORE_PROVIDER=chromadb
VECTOR_STORE_COLLECTION=rag_documents

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: External Services
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-west1-gcp
REDIS_URL=redis://localhost:6379
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from advanced_rag_system import AdvancedRAGEngine, RAGConfig

async def main():
    # Initialize with default configuration
    config = RAGConfig.from_env()
    rag_engine = AdvancedRAGEngine(config)
    
    # Initialize the system
    await rag_engine.initialize()
    
    # Ingest a document
    result = await rag_engine.ingest_document(
        "path/to/document.pdf",
        metadata={"category": "research", "author": "John Doe"}
    )
    
    # Query the system
    response = await rag_engine.query(
        "What are the main findings in the document?",
        context={"user_id": "user123"}
    )
    
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Sources: {len(response.sources)}")
    
    # Cleanup
    await rag_engine.shutdown()

asyncio.run(main())
```

### Advanced Configuration

```python
from advanced_rag_system import (
    RAGConfig, VectorStoreConfig, EmbeddingConfig, 
    LLMConfig, AgentConfig, RetrieverConfig
)

# Custom configuration
config = RAGConfig(
    vector_store=VectorStoreConfig(
        provider="pinecone",
        collection_name="my_knowledge_base",
        embedding_dimension=1536
    ),
    embedding=EmbeddingConfig(
        model_name="text-embedding-ada-002",
        model_type="openai",
        batch_size=100
    ),
    llm=LLMConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7
    ),
    agent=AgentConfig(
        enable_query_expansion=True,
        enable_query_routing=True,
        confidence_threshold=0.8
    ),
    retriever=RetrieverConfig(
        top_k=10,
        enable_reranking=True,
        hybrid_search_alpha=0.7
    )
)

rag_engine = AdvancedRAGEngine(config)
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Query Agent   │    │ Document Agent   │    │Orchestrator     │
│                 │    │                  │    │Agent            │
│ • Intent        │    │ • Analysis       │    │ • Coordination  │
│   Detection     │    │ • Chunking       │    │ • Planning      │
│ • Expansion     │    │ • Metadata       │    │ • Optimization  │
│ • Routing       │    │   Extraction     │    │ • Reranking      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │    RAG Engine Core      │
                    │                         │
                    │ • Query Processing      │
                    │ • Document Ingestion    │
                    │ • Response Generation   │
                    │ • System Orchestration  │
                    └─────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                       │                        │
┌───────────────┐    ┌─────────────────┐    ┌──────────────────┐
│Vector Store   │    │Embedding        │    │Hybrid Retriever  │
│Manager        │    │Manager          │    │                  │
│               │    │                 │    │ • Semantic       │
│ • ChromaDB    │    │ • Transformers  │    │ • Keyword        │
│ • Pinecone    │    │ • OpenAI        │    │ • Fusion         │
│ • Weaviate    │    │ • Caching       │    │ • Reranking      │
│ • Qdrant      │    │ • Optimization  │    │ • Filtering      │
└───────────────┘    └─────────────────┘    └──────────────────┘
```

### AI Agents

1. **Query Agent**: Handles query understanding, expansion, and routing
2. **Document Agent**: Manages intelligent document processing and analysis
3. **Orchestrator Agent**: Coordinates the entire RAG workflow and decision-making

### Vector Database Support

| Database | Status | Features |
|----------|--------|----------|
| ChromaDB | ✅ | Local, persistent, easy setup |
| Pinecone | ✅ | Cloud, scalable, high performance |
| Weaviate | 🔄 | Graph-based, semantic search |
| Qdrant | 🔄 | High performance, filtering |
| Milvus | 🔄 | Scalable, enterprise-ready |
| Elasticsearch | 🔄 | Full-text + vector search |

## 📚 Examples

### Batch Document Processing

```python
# Process multiple documents concurrently
document_paths = [
    "docs/paper1.pdf",
    "docs/paper2.pdf", 
    "docs/report.docx"
]

results = await rag_engine.ingest_documents_batch(
    document_paths,
    metadata_list=[
        {"category": "research", "year": 2023},
        {"category": "research", "year": 2024},
        {"category": "report", "department": "AI"}
    ],
    max_concurrent=3
)
```

### Advanced Querying

```python
# Query with context and filters
response = await rag_engine.query(
    query="How does transformer architecture work?",
    context={
        "user_expertise": "intermediate",
        "domain": "machine_learning",
        "conversation_history": ["previous", "questions"]
    },
    filters={
        "category": "research",
        "year": {"$gte": 2020}
    }
)
```

### System Monitoring

```python
# Get comprehensive system statistics
stats = await rag_engine.get_system_stats()
print(f"Total documents: {stats['vector_store_stats']['document_count']}")
print(f"Average response time: {stats['metrics']['performance']['avg_response_time']:.2f}s")
print(f"Cache hit rate: {stats['embedding_stats']['cache_hit_rate']:.1%}")

# Optimize system performance
optimization_results = await rag_engine.optimize_system()
```

## 🔧 Configuration

### Configuration Options

The system supports extensive configuration through environment variables or configuration files:

#### Vector Store Configuration
- `VECTOR_STORE_PROVIDER`: chromadb, pinecone, weaviate, qdrant
- `VECTOR_STORE_COLLECTION`: Collection/index name
- `PINECONE_API_KEY`, `WEAVIATE_URL`, etc.

#### Embedding Configuration
- `EMBEDDING_MODEL`: Model name or path
- `EMBEDDING_MODEL_TYPE`: sentence_transformers, openai
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation

#### LLM Configuration
- `LLM_PROVIDER`: openai, anthropic, huggingface
- `LLM_MODEL`: Model name
- `LLM_TEMPERATURE`: Generation temperature

#### Agent Configuration
- `ENABLE_QUERY_EXPANSION`: Enable query expansion
- `ENABLE_QUERY_ROUTING`: Enable intelligent routing
- `AGENT_CONFIDENCE_THRESHOLD`: Minimum confidence for agent decisions

### Configuration File

```json
{
  "vector_store": {
    "provider": "chromadb",
    "collection_name": "my_rag_system",
    "embedding_dimension": 384
  },
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "model_type": "sentence_transformers",
    "batch_size": 32
  },
  "llm": {
    "provider": "openai",
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.7
  },
  "agent": {
    "enable_query_expansion": true,
    "enable_query_routing": true,
    "confidence_threshold": 0.8
  }
}
```

## 🎯 Use Cases

### Enterprise Knowledge Management
- Internal documentation search
- Policy and procedure queries
- Technical support automation
- Training material organization

### Research & Academia
- Literature review assistance
- Paper summarization
- Research question answering
- Citation and reference management

### Customer Support
- FAQ automation
- Product documentation search
- Troubleshooting assistance
- Multi-language support

### Content Creation
- Research assistance
- Fact checking
- Content ideation
- Source verification

## 🔍 Performance Optimization

### Caching Strategies
- **Response Caching**: Cache frequent query results
- **Embedding Caching**: Cache computed embeddings
- **Multi-level Caching**: Redis + in-memory caching

### Retrieval Optimization
- **Hybrid Search**: Combine semantic and keyword search
- **Reranking**: Use cross-encoders for relevance refinement
- **Smart Filtering**: Context-aware result filtering
- **Batch Processing**: Concurrent document processing

### System Monitoring
- **Real-time Metrics**: Query performance, success rates
- **Resource Monitoring**: Memory, CPU, storage usage
- **Error Tracking**: Detailed error analysis and reporting
- **Performance Analytics**: Response time distribution, bottleneck identification

## 🧪 Testing

Run the example demo:

```bash
python examples/basic_usage.py
```

Run unit tests:

```bash
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models and embeddings
- Hugging Face for transformer models
- ChromaDB, Pinecone, and other vector database providers
- The open-source community for various libraries and tools

## 📞 Support

- 📧 Email: support@advanced-rag-system.com
- 💬 Discord: [Join our community](https://discord.gg/advanced-rag)
- 📖 Documentation: [Full documentation](https://docs.advanced-rag-system.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/advanced-rag-system/issues)

---

**Built with ❤️ for the AI community**