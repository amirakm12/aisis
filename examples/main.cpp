#include "vector_db.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace VectorDB;

void print_search_results(const std::vector<SearchResult>& results, const std::string& query_desc) {
    std::cout << "\n=== Search Results for: " << query_desc << " ===" << std::endl;
    std::cout << std::setw(5) << "ID" << std::setw(12) << "Distance" << std::setw(20) << "Metadata" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(5) << result.id 
                  << std::setw(12) << std::fixed << std::setprecision(4) << result.distance;
        
        // Print first metadata entry
        if (!result.metadata.empty()) {
            auto it = result.metadata.begin();
            std::cout << std::setw(20) << (it->first + ":" + it->second);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void demonstrate_basic_operations() {
    std::cout << "\n=== Demonstrating Basic Vector Operations ===" << std::endl;
    
    // Create database configuration
    Config config;
    config.dimension = 64;
    config.index_type = IndexType::FLAT;
    config.distance_metric = DistanceMetric::COSINE;
    config.storage_path = "./demo_vectordb";
    
    // Create vector database
    auto db = create_vector_database(config);
    
    std::cout << "Created vector database with " << config.dimension << " dimensions" << std::endl;
    
    // Generate some test vectors
    auto test_vectors = Utils::generate_test_data(100, config.dimension);
    
    // Add vectors to database
    auto start = std::chrono::high_resolution_clock::now();
    auto ids = db->add_vectors_batch(test_vectors);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Added " << ids.size() << " vectors in " << duration.count() << "ms" << std::endl;
    
    // Print database statistics
    db->print_stats();
    
    // Perform searches
    Vector query = Utils::random_vector(config.dimension);
    
    start = std::chrono::high_resolution_clock::now();
    auto results = db->search(query, 5);
    end = std::chrono::high_resolution_clock::now();
    
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Search completed in " << duration_us.count() << "μs" << std::endl;
    
    print_search_results(results, "Random Query Vector");
    
    // Test different distance metrics
    std::cout << "\n=== Testing Different Distance Metrics ===" << std::endl;
    
    for (auto metric : {DistanceMetric::EUCLIDEAN, DistanceMetric::COSINE, DistanceMetric::MANHATTAN}) {
        db->set_distance_metric(metric);
        auto metric_results = db->search(query, 3);
        
        std::string metric_name;
        switch (metric) {
            case DistanceMetric::EUCLIDEAN: metric_name = "Euclidean"; break;
            case DistanceMetric::COSINE: metric_name = "Cosine"; break;
            case DistanceMetric::MANHATTAN: metric_name = "Manhattan"; break;
            default: metric_name = "Unknown"; break;
        }
        
        print_search_results(metric_results, metric_name + " Distance");
    }
    
    // Save database
    db->save();
    std::cout << "Database saved to: " << config.storage_path << std::endl;
}

void demonstrate_text_operations() {
    std::cout << "\n=== Demonstrating Text Embedding Operations ===" << std::endl;
    
    Config config;
    config.dimension = 128;
    config.index_type = IndexType::FLAT;
    config.distance_metric = DistanceMetric::COSINE;
    config.storage_path = "./demo_text_vectordb";
    
    auto db = create_vector_database(config);
    
    // Set TF-IDF embedding generator
    auto tfidf_generator = create_embedding_generator(EmbeddingType::TFIDF, config.dimension);
    
    // Generate training texts for TF-IDF
    auto training_texts = Utils::generate_test_texts(200);
    
    // Cast to TFIDFEmbedding to call fit method
    auto* tfidf_ptr = dynamic_cast<TFIDFEmbedding*>(tfidf_generator.get());
    if (tfidf_ptr) {
        tfidf_ptr->fit(training_texts);
        std::cout << "Fitted TF-IDF model on " << training_texts.size() << " documents" << std::endl;
    }
    
    db->set_embedding_generator(std::move(tfidf_generator));
    
    // Add some sample documents
    std::vector<std::string> documents = {
        "machine learning algorithms for vector databases",
        "neural networks and deep learning applications",
        "similarity search in high dimensional spaces",
        "indexing techniques for fast retrieval",
        "embedding methods for text processing",
        "artificial intelligence and natural language",
        "database systems and query optimization",
        "information retrieval and search engines"
    };
    
    std::vector<Metadata> metadata;
    for (size_t i = 0; i < documents.size(); ++i) {
        metadata.push_back({
            {"doc_id", std::to_string(i)},
            {"category", i < 4 ? "technical" : "general"},
            {"length", std::to_string(documents[i].length())}
        });
    }
    
    auto text_ids = db->add_texts_batch(documents, metadata);
    std::cout << "Added " << text_ids.size() << " text documents" << std::endl;
    
    // Search with text queries
    std::vector<std::string> queries = {
        "machine learning vector search",
        "neural network embedding",
        "database indexing methods"
    };
    
    for (const auto& query : queries) {
        auto results = db->search_text(query, 3);
        print_search_results(results, "\"" + query + "\"");
    }
    
    // Demonstrate filtered search
    std::cout << "\n=== Filtered Search (technical category only) ===" << std::endl;
    
    Vector query_vec = db->generate_embedding("machine learning database");
    auto filtered_results = db->search_with_filter(
        query_vec, 5,
        [](const Metadata& meta) {
            auto it = meta.find("category");
            return it != meta.end() && it->second == "technical";
        }
    );
    
    print_search_results(filtered_results, "Technical Documents Only");
}

void demonstrate_index_types() {
    std::cout << "\n=== Demonstrating Different Index Types ===" << std::endl;
    
    Config config;
    config.dimension = 32;
    config.distance_metric = DistanceMetric::COSINE;
    config.storage_path = "./demo_index_vectordb";
    
    // Generate test data
    auto test_data = Utils::generate_test_data(1000, config.dimension);
    Vector query = Utils::random_vector(config.dimension);
    
    // Test different index types
    std::vector<IndexType> index_types = {
        IndexType::FLAT,
        IndexType::IVF,
        IndexType::HNSW,
        IndexType::LSH
    };
    
    std::vector<std::string> index_names = {
        "Flat (Brute Force)",
        "IVF (Inverted File)",
        "HNSW (Hierarchical NSW)",
        "LSH (Locality Sensitive Hash)"
    };
    
    for (size_t i = 0; i < index_types.size(); ++i) {
        std::cout << "\n--- Testing " << index_names[i] << " ---" << std::endl;
        
        config.index_type = index_types[i];
        auto db = create_vector_database(config);
        
        // Add data
        auto start = std::chrono::high_resolution_clock::now();
        db->add_vectors_batch(test_data);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto add_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Data insertion: " << add_duration.count() << "ms" << std::endl;
        
        // Build index
        start = std::chrono::high_resolution_clock::now();
        db->build_index();
        end = std::chrono::high_resolution_clock::now();
        
        auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Index building: " << build_duration.count() << "ms" << std::endl;
        
        // Search
        start = std::chrono::high_resolution_clock::now();
        auto results = db->search(query, 10);
        end = std::chrono::high_resolution_clock::now();
        
        auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Search time: " << search_duration.count() << "μs" << std::endl;
        std::cout << "Results found: " << results.size() << std::endl;
        
        if (!results.empty()) {
            std::cout << "Best match distance: " << std::fixed << std::setprecision(4) 
                      << results[0].distance << std::endl;
        }
    }
}

void demonstrate_persistence() {
    std::cout << "\n=== Demonstrating Persistence and Loading ===" << std::endl;
    
    const std::string db_path = "./persistent_vectordb";
    
    // Create and populate database
    {
        Config config;
        config.dimension = 50;
        config.storage_path = db_path;
        
        auto db = create_vector_database(config);
        
        // Add some data
        auto test_data = Utils::generate_test_data(50, config.dimension);
        db->add_vectors_batch(test_data);
        
        std::cout << "Created database with " << db->size() << " vectors" << std::endl;
        
        // Save database
        db->save();
        std::cout << "Database saved to: " << db_path << std::endl;
    }
    
    // Load database
    {
        auto loaded_db = load_vector_database(db_path);
        std::cout << "Loaded database with " << loaded_db->size() << " vectors" << std::endl;
        
        loaded_db->print_stats();
        
        // Test search on loaded database
        Vector query = Utils::random_vector(50);
        auto results = loaded_db->search(query, 5);
        
        print_search_results(results, "Query on Loaded Database");
    }
}

void benchmark_performance() {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    
    Config config;
    config.dimension = 128;
    config.index_type = IndexType::FLAT;
    config.distance_metric = DistanceMetric::COSINE;
    config.storage_path = "./benchmark_vectordb";
    
    auto db = create_vector_database(config);
    
    // Benchmark data insertion
    std::vector<size_t> data_sizes = {100, 500, 1000, 5000};
    
    for (size_t size : data_sizes) {
        auto test_data = Utils::generate_test_data(size, config.dimension);
        
        auto start = std::chrono::high_resolution_clock::now();
        db->add_vectors_batch(test_data);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double throughput = static_cast<double>(size) / duration.count() * 1000.0;
        
        std::cout << "Size: " << std::setw(5) << size 
                  << " | Time: " << std::setw(6) << duration.count() << "ms"
                  << " | Throughput: " << std::setw(8) << std::fixed << std::setprecision(1) 
                  << throughput << " vectors/sec" << std::endl;
    }
    
    // Benchmark search performance
    std::cout << "\nSearch Performance (on " << db->size() << " vectors):" << std::endl;
    
    Vector query = Utils::random_vector(config.dimension);
    std::vector<size_t> k_values = {1, 5, 10, 50};
    
    for (size_t k : k_values) {
        auto start = std::chrono::high_resolution_clock::now();
        auto results = db->search(query, k);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "k=" << std::setw(2) << k 
                  << " | Time: " << std::setw(6) << duration.count() << "μs"
                  << " | Found: " << results.size() << " results" << std::endl;
    }
}

int main() {
    std::cout << "=== Vector Database Demo ===" << std::endl;
    std::cout << "This demo showcases the vector database functionality including:" << std::endl;
    std::cout << "- Basic vector operations and indexing" << std::endl;
    std::cout << "- Text embedding and similarity search" << std::endl;
    std::cout << "- Different index types and their performance" << std::endl;
    std::cout << "- Database persistence and loading" << std::endl;
    std::cout << "- Performance benchmarking" << std::endl;
    
    try {
        demonstrate_basic_operations();
        demonstrate_text_operations();
        demonstrate_index_types();
        demonstrate_persistence();
        benchmark_performance();
        
        std::cout << "\n=== Demo Completed Successfully ===" << std::endl;
        std::cout << "Check the created directories for saved database files." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during demo: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}