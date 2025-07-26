"""
Basic Usage Example for Advanced RAG System

This example demonstrates how to use the Advanced RAG System
with AI Agents & Vector Databases.
"""

import asyncio
import os
from pathlib import Path
from typing import List

from advanced_rag_system import AdvancedRAGEngine, RAGConfig


async def basic_example():
    """Basic example of RAG system usage"""
    print("🚀 Starting Basic RAG System Example")
    
    # Create configuration
    config = RAGConfig()
    
    # Initialize RAG engine
    rag_engine = AdvancedRAGEngine(config)
    await rag_engine.initialize()
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Ingest documents
    print("\n📚 Ingesting documents...")
    for doc_path, content in documents:
        with open(doc_path, 'w') as f:
            f.write(content)
        
        result = await rag_engine.ingest_document(doc_path)
        print(f"✅ Ingested {doc_path}: {result.chunks_created} chunks")
    
    # Query the system
    print("\n🔍 Querying the system...")
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the benefits of AI?",
        "Explain neural networks"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = await rag_engine.query(query)
        print(f"Answer: {response.answer[:200]}...")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Sources: {len(response.sources)}")
        print(f"Query time: {response.query_time:.2f}s")
    
    # Get system stats
    print("\n📊 System Statistics:")
    stats = await rag_engine.get_system_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Cleanup
    for doc_path, _ in documents:
        if os.path.exists(doc_path):
            os.remove(doc_path)
    
    await rag_engine.shutdown()
    print("\n✅ Basic example completed!")


async def advanced_example():
    """Advanced example with custom configuration"""
    print("\n🚀 Starting Advanced RAG System Example")
    
    # Create custom configuration
    config = RAGConfig()
    config.vector_store.type = "chroma"
    config.embedding.model_name = "all-MiniLM-L6-v2"
    config.retriever.top_k = 15
    config.retriever.use_reranking = True
    config.cache.enable_cache = True
    config.monitoring.enable_metrics = True
    
    # Initialize RAG engine
    rag_engine = AdvancedRAGEngine(config)
    await rag_engine.initialize()
    
    # Create and ingest documents
    documents = create_sample_documents()
    
    print("\n📚 Batch ingesting documents...")
    doc_paths = [doc_path for doc_path, _ in documents]
    
    for doc_path, content in documents:
        with open(doc_path, 'w') as f:
            f.write(content)
    
    # Batch ingestion
    results = await rag_engine.ingest_documents_batch(doc_paths)
    
    for i, result in enumerate(results):
        status = "✅" if result.success else "❌"
        print(f"{status} Document {i+1}: {result.chunks_created} chunks, {result.processing_time:.2f}s")
    
    # Advanced queries with context
    print("\n🔍 Advanced queries...")
    
    # Query with context
    context = {"user_type": "researcher", "domain": "AI"}
    response = await rag_engine.query(
        "Explain the latest developments in AI",
        context=context,
        use_cache=True
    )
    
    print(f"Query with context: {response.answer[:300]}...")
    print(f"Confidence: {response.confidence:.2f}")
    
    # Query with different strategies
    queries_with_strategies = [
        ("What is deep learning?", {"strategy": "semantic"}),
        ("How do neural networks work?", {"strategy": "hybrid"}),
        ("Explain AI applications", {"strategy": "keyword"})
    ]
    
    for query, strategy in queries_with_strategies:
        response = await rag_engine.query(query, context=strategy)
        print(f"\nQuery: {query}")
        print(f"Strategy: {strategy}")
        print(f"Answer: {response.answer[:200]}...")
    
    # System optimization
    print("\n⚡ Optimizing system...")
    optimizations = await rag_engine.optimize_system()
    for component, result in optimizations.items():
        print(f"{component}: {result}")
    
    # Export metrics
    print("\n📈 Performance metrics:")
    metrics = await rag_engine.get_system_stats()
    print(f"Vector store: {metrics['vector_store_stats']}")
    print(f"Embedding stats: {metrics['embedding_stats']}")
    print(f"Cache stats: {metrics['cache_stats']}")
    print(f"Performance summary: {metrics['metrics_summary']}")
    
    # Cleanup
    for doc_path, _ in documents:
        if os.path.exists(doc_path):
            os.remove(doc_path)
    
    await rag_engine.shutdown()
    print("\n✅ Advanced example completed!")


def create_sample_documents() -> List[tuple]:
    """Create sample documents for testing"""
    documents = [
        ("ai_introduction.txt", """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

AI can be categorized into two main types: Narrow AI and General AI. Narrow AI, also known as Weak AI, is designed to perform specific tasks, such as facial recognition or language translation. General AI, or Strong AI, refers to systems that possess human-like intelligence and can perform any intellectual task that a human can do.

Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns.
        """),
        
        ("machine_learning.txt", """
Machine Learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience.

There are three main types of machine learning:
1. Supervised Learning: The algorithm learns from labeled training data to make predictions on new, unseen data.
2. Unsupervised Learning: The algorithm finds hidden patterns in unlabeled data without any guidance.
3. Reinforcement Learning: The algorithm learns by interacting with an environment and receiving rewards or penalties.

Neural networks are a key component of modern machine learning, particularly in deep learning. They are inspired by the human brain and consist of interconnected nodes (neurons) that process information and learn patterns.
        """),
        
        ("neural_networks.txt", """
Neural Networks are computational models inspired by biological neural networks in the human brain. They are the foundation of deep learning and have revolutionized many fields including computer vision, natural language processing, and speech recognition.

A neural network consists of layers of interconnected nodes (neurons). Each connection has a weight that determines the strength of the signal. The network learns by adjusting these weights based on the training data.

Key components of neural networks:
- Input Layer: Receives the initial data
- Hidden Layers: Process the information through weighted connections
- Output Layer: Produces the final result
- Activation Functions: Introduce non-linearity to the network

Deep learning refers to neural networks with multiple hidden layers, enabling them to learn complex patterns and representations from data.
        """),
        
        ("ai_applications.txt", """
Artificial Intelligence has found applications across numerous industries and domains. Some of the most prominent applications include:

Healthcare: AI is used for medical diagnosis, drug discovery, personalized medicine, and patient care management. Machine learning algorithms can analyze medical images, predict disease progression, and assist in treatment planning.

Finance: AI powers algorithmic trading, fraud detection, credit scoring, and risk assessment. Banks and financial institutions use AI to automate processes and make data-driven decisions.

Transportation: Self-driving cars, traffic prediction, and route optimization are all powered by AI. Companies like Tesla and Waymo are leading the development of autonomous vehicles.

Entertainment: AI is used in recommendation systems (Netflix, Spotify), content generation, and gaming. It helps personalize user experiences and create engaging content.

Education: AI enables personalized learning, automated grading, and intelligent tutoring systems that adapt to individual student needs.
        """)
    ]
    
    return documents


async def setup_environment():
    """Setup environment for the examples"""
    print("🔧 Setting up environment...")
    
    # Create examples directory if it doesn't exist
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Set environment variables for testing
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("VECTOR_STORE_TYPE", "chroma")
    os.environ.setdefault("EMBEDDING_TYPE", "sentence_transformers")
    
    print("✅ Environment setup completed!")


async def main():
    """Main function to run examples"""
    print("🎯 Advanced RAG System with AI Agents & Vector Databases")
    print("=" * 60)
    
    # Setup environment
    await setup_environment()
    
    # Run basic example
    await basic_example()
    
    # Run advanced example
    await advanced_example()
    
    print("\n🎉 All examples completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Multi-format document processing")
    print("✅ Intelligent chunking and embedding")
    print("✅ Hybrid retrieval (semantic + keyword)")
    print("✅ AI agents for query processing")
    print("✅ Caching and performance optimization")
    print("✅ Comprehensive metrics and monitoring")
    print("✅ Multi-vector database support")


if __name__ == "__main__":
    asyncio.run(main()) 