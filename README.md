# Vector Database System

A comprehensive C++ vector database implementation with embedding generation, multiple indexing strategies, and persistent storage capabilities.

## Features

### Core Functionality
- **Vector Storage & Retrieval**: Efficient storage and retrieval of high-dimensional vectors
- **Multiple Distance Metrics**: Euclidean, Cosine, Manhattan, and Dot Product similarity
- **Embedding Generation**: Multiple embedding strategies for text and feature vectors
- **Indexing Strategies**: Flat, IVF, HNSW, and LSH indexing for fast similarity search
- **Persistent Storage**: File-based and in-memory storage backends
- **Thread-Safe Operations**: Concurrent access support with mutex protection

### Embedding Methods
- **Hash Embedding**: Simple hash-based vector generation
- **Random Projection**: Dimensionality reduction using random matrices
- **TF-IDF**: Term frequency-inverse document frequency for text
- **N-gram**: Character n-gram based embeddings
- **Average Word**: Word embedding averaging for sentences

### Index Types
- **Flat Index**: Brute force search (exact results)
- **IVF Index**: Inverted File index with k-means clustering
- **HNSW Index**: Hierarchical Navigable Small World graphs
- **LSH Index**: Locality Sensitive Hashing for approximate search

## Project Structure

```
├── include/           # Header files
│   ├── types.h        # Core data types and structures
│   ├── similarity.h   # Distance calculation functions
│   ├── embedding.h    # Embedding generation interfaces
│   ├── index.h        # Indexing system interfaces
│   ├── storage.h      # Storage backend interfaces
│   └── vector_db.h    # Main database interface
├── src/               # Implementation files
│   ├── similarity.cpp # Distance calculations
│   ├── embedding.cpp  # Embedding generators
│   ├── index.cpp      # Index implementations
│   ├── storage.cpp    # Storage backends
│   └── vector_db.cpp  # Main database implementation
├── examples/          # Example programs
│   └── main.cpp       # Comprehensive demo
├── tests/             # Test files
│   └── test_main.cpp  # Unit tests
└── CMakeLists.txt     # Build configuration
```

## Building

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.20+
- OpenMP (optional, for performance optimization)

### Build Instructions

```bash
# Create build directory
mkdir build && cd build

# Configure build
cmake ..

# Build the project
make -j$(nproc)

# Run example
./vectordb_example

# Run tests
./vectordb_test
```

### Build Options

```bash
# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..

# Debug build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## Usage Examples

### Basic Vector Operations

```cpp
#include "vector_db.h"
using namespace VectorDB;

// Create database configuration
Config config;
config.dimension = 128;
config.index_type = IndexType::FLAT;
config.distance_metric = DistanceMetric::COSINE;

// Create database
auto db = create_vector_database(config);

// Add vectors
Vector vec = {1.0f, 2.0f, 3.0f, /*... 128 dimensions */};
VectorId id = db->add_vector(vec, {{"label", "example"}});

// Search for similar vectors
Vector query = {1.1f, 2.1f, 3.1f, /*... 128 dimensions */};
auto results = db->search(query, 5); // Top 5 results

// Print results
for (const auto& result : results) {
    std::cout << "ID: " << result.id 
              << ", Distance: " << result.distance << std::endl;
}
```

### Text Embedding and Search

```cpp
// Set up TF-IDF embedding
auto tfidf = create_embedding_generator(EmbeddingType::TFIDF, 256);
auto* tfidf_ptr = dynamic_cast<TFIDFEmbedding*>(tfidf.get());

// Train on documents
std::vector<std::string> documents = {
    "machine learning algorithms",
    "neural network architectures",
    "vector database systems"
};
tfidf_ptr->fit(documents);
db->set_embedding_generator(std::move(tfidf));

// Add text documents
std::vector<std::string> texts = {
    "deep learning with neural networks",
    "similarity search in vector spaces",
    "indexing strategies for databases"
};
auto ids = db->add_texts_batch(texts);

// Search with text query
auto results = db->search_text("machine learning neural networks", 3);
```

### Different Index Types

```cpp
// Flat index (exact search)
config.index_type = IndexType::FLAT;
auto flat_db = create_vector_database(config);

// IVF index (approximate search)
config.index_type = IndexType::IVF;
config.index_params.nlist = 100;    // Number of clusters
config.index_params.nprobe = 10;    // Clusters to search
auto ivf_db = create_vector_database(config);

// HNSW index (graph-based)
config.index_type = IndexType::HNSW;
config.index_params.M = 16;              // Connections per layer
config.index_params.ef_construction = 200; // Build-time parameter
config.index_params.ef_search = 50;      // Search-time parameter
auto hnsw_db = create_vector_database(config);
```

### Persistence

```cpp
// Save database
db->save("./my_vectordb");

// Load database
auto loaded_db = load_vector_database("./my_vectordb");

// Database automatically loads configuration and rebuilds index
std::cout << "Loaded " << loaded_db->size() << " vectors" << std::endl;
```

### Filtered Search

```cpp
// Search with metadata filter
Vector query = {/* query vector */};
auto filtered_results = db->search_with_filter(
    query, 10,
    [](const Metadata& meta) {
        auto it = meta.find("category");
        return it != meta.end() && it->second == "technical";
    }
);
```

## Configuration Options

### Distance Metrics
- `DistanceMetric::EUCLIDEAN`: L2 distance
- `DistanceMetric::COSINE`: Cosine similarity (1 - cos(θ))
- `DistanceMetric::MANHATTAN`: L1 distance
- `DistanceMetric::DOT_PRODUCT`: Negative dot product

### Index Parameters

#### IVF Index
- `nlist`: Number of clusters (default: 100)
- `nprobe`: Number of clusters to search (default: 10)

#### HNSW Index
- `M`: Number of connections per layer (default: 16)
- `ef_construction`: Size of dynamic candidate list during construction (default: 200)
- `ef_search`: Size of dynamic candidate list during search (default: 50)

#### LSH Index
- `num_bits`: Number of hash bits per table (default: 8)

## Performance Characteristics

### Index Comparison

| Index Type | Build Time | Search Time | Memory Usage | Accuracy |
|------------|------------|-------------|--------------|----------|
| Flat       | O(1)       | O(n)        | O(n)         | 100%     |
| IVF        | O(n log k) | O(n/k)      | O(n)         | ~95%     |
| HNSW       | O(n log n) | O(log n)    | O(n)         | ~99%     |
| LSH        | O(n)       | O(1)        | O(n)         | ~80%     |

### Scalability
- **Small datasets** (<10K vectors): Use Flat index
- **Medium datasets** (10K-1M vectors): Use IVF or HNSW
- **Large datasets** (>1M vectors): Use HNSW or LSH

## Advanced Features

### Batch Operations
```cpp
// Batch insert
std::vector<VectorEntry> entries = {/* vector entries */};
auto ids = db->add_vectors_batch(entries);

// Batch retrieval
std::vector<VectorId> ids_to_get = {1, 2, 3, 4, 5};
auto vectors = db->get_vectors(ids_to_get);

// Batch removal
std::vector<VectorId> ids_to_remove = {1, 3, 5};
db->remove_vectors_batch(ids_to_remove);
```

### Database Maintenance
```cpp
// Rebuild index for optimization
db->rebuild_index();

// Compact storage (remove deleted entries)
db->compact();

// Full optimization
db->optimize();

// Get database statistics
db->print_stats();
std::string info = db->get_info();
```

### Custom Embedding
```cpp
// Implement custom embedding generator
class CustomEmbedding : public EmbeddingGenerator {
public:
    Vector generate_embedding(const std::string& text) override {
        // Custom implementation
        return Vector(dimension_, 0.0f);
    }
    
    size_t get_dimension() const override { return dimension_; }
private:
    size_t dimension_;
};

// Use custom embedding
auto custom_emb = std::make_unique<CustomEmbedding>();
db->set_embedding_generator(std::move(custom_emb));
```

## Thread Safety

The vector database is thread-safe for concurrent operations:

```cpp
// Multiple threads can safely perform operations
std::thread t1([&db]() { db->search(query1, 10); });
std::thread t2([&db]() { db->add_vector(vec2, {}); });
std::thread t3([&db]() { db->remove_vector(id3); });

t1.join(); t2.join(); t3.join();
```

## Error Handling

The library uses exceptions for error handling:

```cpp
try {
    auto db = create_vector_database(config);
    db->add_vector(wrong_dimension_vector);
} catch (const DimensionMismatchException& e) {
    std::cerr << "Dimension error: " << e.what() << std::endl;
} catch (const VectorDBException& e) {
    std::cerr << "Database error: " << e.what() << std::endl;
}
```

## Utilities

The library provides utility functions for vector operations:

```cpp
// Generate random vectors for testing
auto random_vecs = Utils::generate_random_vectors(1000, 128);

// Vector arithmetic
Vector a = {1, 2, 3}, b = {4, 5, 6};
Vector sum = Utils::add_vectors(a, b);
Vector diff = Utils::subtract_vectors(a, b);
Vector scaled = Utils::multiply_vector(a, 2.0f);

// Generate test data
auto test_data = Utils::generate_test_data(100, 64);
auto test_texts = Utils::generate_test_texts(50);
```

## License

This project is provided as-is for educational and research purposes.

## Contributing

1. Ensure code follows the existing style
2. Add tests for new functionality
3. Update documentation as needed
4. Test on multiple platforms

## Future Enhancements

- GPU acceleration support
- Distributed storage backends
- More embedding methods (Word2Vec, BERT, etc.)
- Query optimization
- Approximate nearest neighbor benchmarks
- Python bindings