#include "vector_db.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace VectorDB;

void test_similarity_calculations() {
    std::cout << "Testing similarity calculations..." << std::endl;
    
    Vector a = {1.0f, 2.0f, 3.0f};
    Vector b = {4.0f, 5.0f, 6.0f};
    
    // Test dot product
    float dot = SimilarityCalculator::dot_product(a, b);
    assert(std::abs(dot - 32.0f) < 1e-6);
    
    // Test euclidean distance
    float euclidean = SimilarityCalculator::euclidean_distance(a, b);
    assert(std::abs(euclidean - std::sqrt(27.0f)) < 1e-6);
    
    // Test vector magnitude
    float mag_a = SimilarityCalculator::vector_magnitude(a);
    assert(std::abs(mag_a - std::sqrt(14.0f)) < 1e-6);
    
    std::cout << "✓ Similarity calculations passed" << std::endl;
}

void test_vector_operations() {
    std::cout << "Testing vector operations..." << std::endl;
    
    Vector a = {1.0f, 2.0f, 3.0f};
    Vector b = {4.0f, 5.0f, 6.0f};
    
    // Test vector addition
    Vector sum = Utils::add_vectors(a, b);
    assert(sum.size() == 3);
    assert(std::abs(sum[0] - 5.0f) < 1e-6);
    assert(std::abs(sum[1] - 7.0f) < 1e-6);
    assert(std::abs(sum[2] - 9.0f) < 1e-6);
    
    // Test vector subtraction
    Vector diff = Utils::subtract_vectors(b, a);
    assert(diff.size() == 3);
    assert(std::abs(diff[0] - 3.0f) < 1e-6);
    assert(std::abs(diff[1] - 3.0f) < 1e-6);
    assert(std::abs(diff[2] - 3.0f) < 1e-6);
    
    // Test scalar multiplication
    Vector scaled = Utils::multiply_vector(a, 2.0f);
    assert(scaled.size() == 3);
    assert(std::abs(scaled[0] - 2.0f) < 1e-6);
    assert(std::abs(scaled[1] - 4.0f) < 1e-6);
    assert(std::abs(scaled[2] - 6.0f) < 1e-6);
    
    std::cout << "✓ Vector operations passed" << std::endl;
}

void test_embedding_generators() {
    std::cout << "Testing embedding generators..." << std::endl;
    
    // Test hash embedding
    auto hash_emb = create_embedding_generator(EmbeddingType::HASH, 64);
    Vector emb1 = hash_emb->generate_embedding("test text");
    Vector emb2 = hash_emb->generate_embedding("test text");
    Vector emb3 = hash_emb->generate_embedding("different text");
    
    assert(emb1.size() == 64);
    assert(emb2.size() == 64);
    assert(emb3.size() == 64);
    
    // Same text should produce same embedding
    bool same = true;
    for (size_t i = 0; i < emb1.size(); ++i) {
        if (std::abs(emb1[i] - emb2[i]) > 1e-6) {
            same = false;
            break;
        }
    }
    assert(same);
    
    // Test N-gram embedding
    auto ngram_emb = create_embedding_generator(EmbeddingType::NGRAM, 128);
    Vector ngram_vec = ngram_emb->generate_embedding("hello world");
    assert(ngram_vec.size() == 128);
    
    std::cout << "✓ Embedding generators passed" << std::endl;
}

void test_flat_index() {
    std::cout << "Testing flat index..." << std::endl;
    
    FlatIndex index(DistanceMetric::COSINE);
    
    // Add some vectors
    Vector v1 = {1.0f, 0.0f, 0.0f};
    Vector v2 = {0.0f, 1.0f, 0.0f};
    Vector v3 = {0.0f, 0.0f, 1.0f};
    
    VectorEntry e1(1, v1, {{"label", "x"}});
    VectorEntry e2(2, v2, {{"label", "y"}});
    VectorEntry e3(3, v3, {{"label", "z"}});
    
    index.add_vector(e1);
    index.add_vector(e2);
    index.add_vector(e3);
    
    assert(index.size() == 3);
    
    // Test search
    Vector query = {1.0f, 0.1f, 0.0f}; // Closest to v1
    auto results = index.search(query, 2);
    
    assert(results.size() == 2);
    assert(results[0].id == 1); // Should be closest to v1
    
    // Test removal
    index.remove_vector(2);
    assert(index.size() == 2);
    
    std::cout << "✓ Flat index passed" << std::endl;
}

void test_vector_database() {
    std::cout << "Testing vector database..." << std::endl;
    
    Config config;
    config.dimension = 32;
    config.index_type = IndexType::FLAT;
    config.distance_metric = DistanceMetric::COSINE;
    config.storage_path = "./test_vectordb";
    
    auto db = create_vector_database(config);
    
    // Test adding vectors
    auto test_data = Utils::generate_test_data(10, config.dimension);
    auto ids = db->add_vectors_batch(test_data);
    
    assert(ids.size() == 10);
    assert(db->size() == 10);
    
    // Test search
    Vector query = Utils::random_vector(config.dimension);
    auto results = db->search(query, 5);
    
    assert(results.size() <= 5);
    assert(results.size() <= db->size());
    
    // Test individual vector operations
    Vector new_vec = Utils::random_vector(config.dimension);
    VectorId new_id = db->add_vector(new_vec, {{"type", "test"}});
    
    VectorEntry retrieved;
    bool found = db->get_vector(new_id, retrieved);
    assert(found);
    assert(retrieved.id == new_id);
    assert(retrieved.metadata.at("type") == "test");
    
    // Test removal
    bool removed = db->remove_vector(new_id);
    assert(removed);
    assert(db->size() == 10); // Back to original size
    
    std::cout << "✓ Vector database passed" << std::endl;
}

void test_text_operations() {
    std::cout << "Testing text operations..." << std::endl;
    
    Config config;
    config.dimension = 64;
    config.storage_path = "./test_text_vectordb";
    
    auto db = create_vector_database(config);
    
    // Add some text documents
    std::vector<std::string> texts = {
        "machine learning vector database",
        "artificial intelligence neural networks",
        "similarity search algorithms"
    };
    
    auto ids = db->add_texts_batch(texts);
    assert(ids.size() == 3);
    assert(db->size() == 3);
    
    // Test text search
    auto results = db->search_text("machine learning", 2);
    assert(results.size() <= 2);
    
    std::cout << "✓ Text operations passed" << std::endl;
}

void test_persistence() {
    std::cout << "Testing persistence..." << std::endl;
    
    const std::string db_path = "./test_persistent_db";
    
    // Create and save database
    {
        Config config;
        config.dimension = 16;
        config.storage_path = db_path;
        
        auto db = create_vector_database(config);
        
        auto test_data = Utils::generate_test_data(5, config.dimension);
        db->add_vectors_batch(test_data);
        
        assert(db->size() == 5);
        db->save();
    }
    
    // Load database
    {
        auto loaded_db = load_vector_database(db_path);
        assert(loaded_db->size() == 5);
        
        // Test that we can search the loaded database
        Vector query = Utils::random_vector(16);
        auto results = loaded_db->search(query, 3);
        assert(results.size() <= 3);
    }
    
    std::cout << "✓ Persistence passed" << std::endl;
}

void run_all_tests() {
    std::cout << "=== Running Vector Database Tests ===" << std::endl;
    
    try {
        test_similarity_calculations();
        test_vector_operations();
        test_embedding_generators();
        test_flat_index();
        test_vector_database();
        test_text_operations();
        test_persistence();
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    try {
        run_all_tests();
        std::cout << "\n=== Test Suite Completed Successfully ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n=== Test Suite Failed ===" << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}