#include "vector_db.h"
#include <iostream>

using namespace VectorDB;

int main() {
    std::cout << "=== Quick Vector Database Demo ===" << std::endl;
    
    try {
        // Create database configuration
        Config config;
        config.dimension = 4;
        config.index_type = IndexType::FLAT;
        config.distance_metric = DistanceMetric::COSINE;
        config.storage_path = "./quick_demo_db";
        
        // Create vector database
        auto db = create_vector_database(config);
        std::cout << "✓ Created vector database with " << config.dimension << " dimensions" << std::endl;
        
        // Add some vectors
        Vector v1 = {1.0f, 0.0f, 0.0f, 0.0f};
        Vector v2 = {0.0f, 1.0f, 0.0f, 0.0f};
        Vector v3 = {0.0f, 0.0f, 1.0f, 0.0f};
        Vector v4 = {0.5f, 0.5f, 0.0f, 0.0f};
        
        VectorId id1 = db->add_vector(v1, {{"label", "x-axis"}});
        VectorId id2 = db->add_vector(v2, {{"label", "y-axis"}});
        VectorId id3 = db->add_vector(v3, {{"label", "z-axis"}});
        VectorId id4 = db->add_vector(v4, {{"label", "xy-diagonal"}});
        
        std::cout << "✓ Added 4 vectors to database" << std::endl;
        std::cout << "  Database size: " << db->size() << std::endl;
        
        // Search for similar vectors
        Vector query = {0.9f, 0.1f, 0.0f, 0.0f}; // Similar to v1
        auto results = db->search(query, 3);
        
        std::cout << "✓ Search results for query [0.9, 0.1, 0.0, 0.0]:" << std::endl;
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "  " << (i+1) << ". ID: " << results[i].id 
                      << ", Distance: " << results[i].distance;
            if (!results[i].metadata.empty()) {
                std::cout << ", Label: " << results[i].metadata.at("label");
            }
            std::cout << std::endl;
        }
        
        // Test text embedding
        std::cout << "\n✓ Testing text embedding:" << std::endl;
        VectorId text_id = db->add_text("hello world", {{"type", "text"}});
        std::cout << "  Added text 'hello world' with ID: " << text_id << std::endl;
        
        auto text_results = db->search_text("hello", 2);
        std::cout << "  Search for 'hello' found " << text_results.size() << " results" << std::endl;
        
        // Test different distance metrics
        std::cout << "\n✓ Testing different distance metrics:" << std::endl;
        
        db->set_distance_metric(DistanceMetric::EUCLIDEAN);
        auto euclidean_results = db->search(query, 1);
        std::cout << "  Euclidean distance to closest: " << euclidean_results[0].distance << std::endl;
        
        db->set_distance_metric(DistanceMetric::COSINE);
        auto cosine_results = db->search(query, 1);
        std::cout << "  Cosine distance to closest: " << cosine_results[0].distance << std::endl;
        
        std::cout << "\n=== Demo completed successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}