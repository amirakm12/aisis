"""
Basic Usage Example for Advanced RAG System with AI Agents & Vector Databases

This example demonstrates:
1. Setting up the RAG system with different configurations
2. Ingesting documents into the system
3. Performing queries with different strategies
4. Using AI agents for intelligent processing
5. Monitoring system performance
"""

import asyncio
import os
from pathlib import Path

# Import the Advanced RAG System
from advanced_rag_system import (
    AdvancedRAGEngine,
    RAGConfig,
    VectorStoreConfig,
    EmbeddingConfig,
    LLMConfig,
    AgentConfig,
    ProcessingConfig,
    RetrieverConfig,
    CacheConfig,
    MonitoringConfig
)

async def basic_example():
    """Basic example of using the Advanced RAG System"""
    print("🚀 Starting Advanced RAG System Demo")
    
    # 1. Create configuration
    config = RAGConfig(
        vector_store=VectorStoreConfig(
            provider="chromadb",
            collection_name="demo_collection",
            embedding_dimension=384
        ),
        embedding=EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_type="sentence_transformers",
            batch_size=16
        ),
        llm=LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.7
        ),
        agent=AgentConfig(
            enable_query_expansion=True,
            enable_query_routing=True,
            enable_document_analysis=True
        ),
        retriever=RetrieverConfig(
            top_k=5,
            enable_reranking=True,
            hybrid_search_alpha=0.7
        ),
        cache=CacheConfig(
            enable_cache=True,
            cache_provider="memory",
            cache_ttl=3600
        ),
        monitoring=MonitoringConfig(
            enable_metrics=True,
            log_level="INFO"
        )
    )
    
    # 2. Initialize the RAG engine
    rag_engine = AdvancedRAGEngine(config)
    await rag_engine.initialize()
    
    print("✅ RAG Engine initialized successfully")
    
    try:
        # 3. Create sample documents for ingestion
        await create_sample_documents()
        
        # 4. Ingest documents
        print("\n📄 Ingesting sample documents...")
        document_paths = [
            "sample_docs/artificial_intelligence.txt",
            "sample_docs/machine_learning.txt",
            "sample_docs/deep_learning.txt"
        ]
        
        ingestion_results = await rag_engine.ingest_documents_batch(
            document_paths,
            max_concurrent=2
        )
        
        successful_ingestions = sum(1 for result in ingestion_results if result.success)
        print(f"✅ Successfully ingested {successful_ingestions}/{len(document_paths)} documents")
        
        # 5. Perform various types of queries
        print("\n🔍 Performing queries with different strategies...")
        
        queries = [
            "What is artificial intelligence?",
            "How does machine learning differ from deep learning?",
            "What are the applications of neural networks?",
            "Compare supervised and unsupervised learning"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            # Query with hybrid strategy (default)
            response = await rag_engine.query(
                query=query,
                context={"user_id": "demo_user", "session_id": "demo_session"}
            )
            
            print(f"Answer: {response.answer[:200]}...")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Sources: {len(response.sources)}")
            print(f"Processing time: {response.processing_time:.2f}s")
        
        # 6. Demonstrate different retrieval strategies
        print("\n🎯 Testing different retrieval strategies...")
        
        test_query = "What are neural networks?"
        
        strategies = ["semantic", "hybrid"]
        for strategy in strategies:
            print(f"\n--- Strategy: {strategy} ---")
            response = await rag_engine.query(
                query=test_query,
                context={"strategy": strategy}
            )
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Sources: {len(response.sources)}")
        
        # 7. Get system statistics
        print("\n📊 System Statistics:")
        stats = await rag_engine.get_system_stats()
        
        print(f"Vector Store: {stats['vector_store_stats']['provider']}")
        print(f"Documents: {stats['vector_store_stats']['document_count']}")
        print(f"Embedding Model: {stats['embedding_stats']['model_name']}")
        
        if 'metrics' in stats:
            metrics = stats['metrics']
            print(f"Total Queries: {metrics['system']['total_queries']}")
            print(f"Average Response Time: {metrics['performance']['avg_response_time']:.2f}s")
            print(f"Average Confidence: {metrics['performance']['avg_confidence']:.2f}")
        
        # 8. Optimize system performance
        print("\n⚡ Optimizing system performance...")
        optimization_results = await rag_engine.optimize_system()
        print(f"Optimization completed: {optimization_results}")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
    
    finally:
        # 9. Shutdown the system
        print("\n🔄 Shutting down RAG engine...")
        await rag_engine.shutdown()
        print("✅ Demo completed successfully")

async def create_sample_documents():
    """Create sample documents for the demo"""
    sample_docs_dir = Path("sample_docs")
    sample_docs_dir.mkdir(exist_ok=True)
    
    documents = {
        "artificial_intelligence.txt": """
        Artificial Intelligence (AI) Overview
        
        Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
        that are programmed to think and learn like humans. The term may also be applied to any 
        machine that exhibits traits associated with a human mind such as learning and problem-solving.
        
        Key Characteristics of AI:
        - Learning: The ability to improve performance based on experience
        - Reasoning: The ability to draw conclusions from available information
        - Problem-solving: The ability to find solutions to complex challenges
        - Perception: The ability to interpret sensory data
        - Language understanding: The ability to comprehend and generate human language
        
        Types of AI:
        1. Narrow AI (Weak AI): AI that is designed for specific tasks
        2. General AI (Strong AI): AI that can understand, learn, and apply knowledge across various domains
        3. Superintelligent AI: AI that surpasses human intelligence in all aspects
        
        Applications:
        - Healthcare: Diagnosis, drug discovery, personalized treatment
        - Transportation: Autonomous vehicles, traffic optimization
        - Finance: Fraud detection, algorithmic trading, risk assessment
        - Education: Personalized learning, intelligent tutoring systems
        - Entertainment: Game AI, content recommendation systems
        """,
        
        "machine_learning.txt": """
        Machine Learning Fundamentals
        
        Machine Learning (ML) is a subset of artificial intelligence that enables computers 
        to learn and improve from experience without being explicitly programmed. It focuses 
        on the development of algorithms that can access data and use it to learn for themselves.
        
        Types of Machine Learning:
        
        1. Supervised Learning:
        - Uses labeled training data to learn a mapping function
        - Examples: Classification, Regression
        - Algorithms: Linear Regression, Decision Trees, Support Vector Machines, Random Forest
        
        2. Unsupervised Learning:
        - Finds hidden patterns in data without labeled examples
        - Examples: Clustering, Dimensionality Reduction, Association Rules
        - Algorithms: K-Means, Hierarchical Clustering, PCA, DBSCAN
        
        3. Reinforcement Learning:
        - Learns through interaction with an environment
        - Uses rewards and penalties to improve decision-making
        - Examples: Game playing, Robotics, Autonomous systems
        - Algorithms: Q-Learning, Policy Gradient, Actor-Critic
        
        Key Concepts:
        - Training Data: Dataset used to train the model
        - Features: Individual measurable properties of observed phenomena
        - Model: Mathematical representation of a real-world process
        - Overfitting: Model performs well on training data but poorly on new data
        - Cross-validation: Technique to assess model performance and generalization
        
        Popular ML Libraries:
        - Python: scikit-learn, TensorFlow, PyTorch, Keras
        - R: caret, randomForest, e1071
        - Java: Weka, MOA, Deeplearning4j
        """,
        
        "deep_learning.txt": """
        Deep Learning and Neural Networks
        
        Deep Learning is a subset of machine learning that uses artificial neural networks 
        with multiple layers (hence "deep") to model and understand complex patterns in data. 
        It's inspired by the structure and function of the human brain.
        
        Neural Network Basics:
        - Neurons (Nodes): Basic processing units that receive inputs and produce outputs
        - Layers: Collections of neurons that process information at different levels
        - Weights and Biases: Parameters that determine the strength of connections
        - Activation Functions: Functions that determine neuron output (ReLU, Sigmoid, Tanh)
        
        Types of Neural Networks:
        
        1. Feedforward Neural Networks:
        - Information flows in one direction from input to output
        - Used for: Basic classification and regression tasks
        
        2. Convolutional Neural Networks (CNNs):
        - Specialized for processing grid-like data (images)
        - Key components: Convolutional layers, Pooling layers, Fully connected layers
        - Applications: Image recognition, Computer vision, Medical imaging
        
        3. Recurrent Neural Networks (RNNs):
        - Can process sequences of data with memory
        - Variants: LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit)
        - Applications: Natural Language Processing, Time series prediction, Speech recognition
        
        4. Transformer Networks:
        - Use attention mechanisms to process sequences
        - Highly parallelizable and effective for NLP tasks
        - Examples: BERT, GPT, T5
        
        5. Generative Adversarial Networks (GANs):
        - Two networks competing against each other
        - Generator creates fake data, Discriminator tries to detect fake data
        - Applications: Image generation, Data augmentation, Style transfer
        
        Deep Learning Applications:
        - Computer Vision: Object detection, Image classification, Facial recognition
        - Natural Language Processing: Machine translation, Sentiment analysis, Chatbots
        - Speech Recognition: Voice assistants, Transcription services
        - Recommendation Systems: Content recommendation, Personalization
        - Game AI: Chess, Go, Video games
        - Autonomous Vehicles: Object detection, Path planning, Decision making
        
        Popular Deep Learning Frameworks:
        - TensorFlow: Google's open-source platform
        - PyTorch: Facebook's dynamic neural network library
        - Keras: High-level API for neural networks
        - JAX: Google's library for high-performance ML research
        """
    }
    
    for filename, content in documents.items():
        file_path = sample_docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    print(f"✅ Created {len(documents)} sample documents")

async def advanced_example():
    """Advanced example showing more sophisticated features"""
    print("\n🔬 Advanced RAG System Features Demo")
    
    # Configuration for advanced features
    config = RAGConfig(
        vector_store=VectorStoreConfig(
            provider="chromadb",
            collection_name="advanced_demo"
        ),
        agent=AgentConfig(
            enable_query_expansion=True,
            enable_query_routing=True,
            enable_document_analysis=True,
            max_iterations=3,
            confidence_threshold=0.8
        ),
        retriever=RetrieverConfig(
            top_k=10,
            enable_reranking=True,
            hybrid_search_alpha=0.6
        )
    )
    
    rag_engine = AdvancedRAGEngine(config)
    await rag_engine.initialize()
    
    try:
        # Advanced query with context and filters
        print("\n🎯 Advanced Query Processing...")
        
        response = await rag_engine.query(
            query="How can I implement a neural network for image classification?",
            context={
                "user_expertise": "intermediate",
                "domain": "computer_vision",
                "preferred_framework": "pytorch"
            },
            filters={
                "document_type": "tutorial",
                "difficulty": "intermediate"
            }
        )
        
        print(f"Answer: {response.answer[:300]}...")
        print(f"Query Analysis: {response.query_analysis}")
        print(f"Execution Plan: {response.retrieval_metadata}")
        
        # Batch document processing
        print("\n📚 Batch Document Processing...")
        
        # Simulate processing multiple documents with different metadata
        documents_with_metadata = [
            ("sample_docs/artificial_intelligence.txt", {"category": "overview", "difficulty": "beginner"}),
            ("sample_docs/machine_learning.txt", {"category": "fundamentals", "difficulty": "intermediate"}),
            ("sample_docs/deep_learning.txt", {"category": "advanced", "difficulty": "advanced"})
        ]
        
        results = []
        for doc_path, metadata in documents_with_metadata:
            result = await rag_engine.ingest_document(
                doc_path,
                metadata=metadata,
                processing_options={"enable_smart_chunking": True}
            )
            results.append(result)
        
        print(f"✅ Processed {len(results)} documents with metadata")
        
        # Performance monitoring
        print("\n📈 Performance Monitoring...")
        stats = await rag_engine.get_system_stats()
        
        if 'metrics' in stats:
            metrics = stats['metrics']
            print("System Performance:")
            print(f"  - Total Operations: {metrics['system']['total_queries'] + metrics['system']['total_ingestions']}")
            print(f"  - Average Response Time: {metrics['performance']['avg_response_time']:.3f}s")
            print(f"  - Error Rate: {metrics['system']['error_rate']:.1%}")
            print(f"  - Cache Hit Rate: {stats['embedding_stats'].get('cache_hit_rate', 'N/A')}")
        
    finally:
        await rag_engine.shutdown()

def setup_environment():
    """Setup environment variables and requirements"""
    print("🔧 Setting up environment...")
    
    # Check for required environment variables
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the demo:")
        for var in missing_vars:
            print(f"  export {var}=your_api_key_here")
        return False
    
    print("✅ Environment setup complete")
    return True

async def main():
    """Main demo function"""
    print("=" * 60)
    print("🤖 Advanced RAG System with AI Agents & Vector Databases")
    print("=" * 60)
    
    if not setup_environment():
        return
    
    try:
        # Run basic example
        await basic_example()
        
        # Run advanced example
        await advanced_example()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())