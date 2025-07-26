#pragma once

#include "types.h"
#include "index.h"
#include "embedding.h"
#include "storage.h"
#include "similarity.h"
#include <memory>
#include <atomic>
#include <mutex>
#include <functional>

namespace VectorDB {
    // Main VectorDB class that provides the high-level interface
    class VectorDatabase {
    private:
        Config config_;
        std::unique_ptr<Index> index_;
        std::unique_ptr<EmbeddingGenerator> embedding_generator_;
        std::unique_ptr<StorageManager> storage_manager_;
        
        std::atomic<VectorId> next_id_;
        mutable std::mutex db_mutex_;
        bool is_loaded_;
        
        VectorId generate_id();
        void validate_vector_dimension(const Vector& vec) const;
        
    public:
        // Constructors
        explicit VectorDatabase(const Config& config = Config{});
        VectorDatabase(const VectorDatabase&) = delete;
        VectorDatabase& operator=(const VectorDatabase&) = delete;
        VectorDatabase(VectorDatabase&&) = delete;
        VectorDatabase& operator=(VectorDatabase&&) = delete;
        ~VectorDatabase();
        
        // Configuration
        void set_config(const Config& config);
        const Config& get_config() const { return config_; }
        
        // Index management
        void build_index();
        void rebuild_index();
        void clear_index();
        
        // Vector operations
        VectorId add_vector(const Vector& vector, const Metadata& metadata = {});
        VectorId add_text(const std::string& text, const Metadata& metadata = {});
        VectorId add_features(const std::vector<float>& features, const Metadata& metadata = {});
        
        bool remove_vector(VectorId id);
        bool update_vector(VectorId id, const Vector& vector, const Metadata& metadata = {});
        bool update_metadata(VectorId id, const Metadata& metadata);
        
        bool get_vector(VectorId id, VectorEntry& entry) const;
        std::vector<VectorEntry> get_vectors(const std::vector<VectorId>& ids) const;
        
        // Search operations
        std::vector<SearchResult> search(const Vector& query, size_t k = 10, float threshold = 0.0f) const;
        std::vector<SearchResult> search_text(const std::string& text, size_t k = 10, float threshold = 0.0f) const;
        std::vector<SearchResult> search_features(const std::vector<float>& features, size_t k = 10, float threshold = 0.0f) const;
        
        // Advanced search with filters
        std::vector<SearchResult> search_with_filter(
            const Vector& query, 
            size_t k, 
            const std::function<bool(const Metadata&)>& filter,
            float threshold = 0.0f
        ) const;
        
        // Batch operations
        std::vector<VectorId> add_vectors_batch(const std::vector<VectorEntry>& entries);
        std::vector<VectorId> add_texts_batch(const std::vector<std::string>& texts, 
                                              const std::vector<Metadata>& metadata = {});
        bool remove_vectors_batch(const std::vector<VectorId>& ids);
        
        // Statistics and information
        size_t size() const;
        bool empty() const;
        std::vector<VectorId> get_all_ids() const;
        
        // Persistence
        void save(const std::string& path = "");
        void load(const std::string& path = "");
        void flush();
        
        // Maintenance
        void compact();
        void optimize();
        
        // Embedding management
        void set_embedding_generator(std::unique_ptr<EmbeddingGenerator> generator);
        EmbeddingGenerator* get_embedding_generator() const { return embedding_generator_.get(); }
        
        // Index management
        void set_index_type(IndexType type);
        IndexType get_index_type() const { return config_.index_type; }
        
        // Distance metric
        void set_distance_metric(DistanceMetric metric);
        DistanceMetric get_distance_metric() const { return config_.distance_metric; }
        
        // Utility functions
        Vector generate_embedding(const std::string& text) const;
        Vector generate_embedding(const std::vector<float>& features) const;
        float calculate_distance(const Vector& a, const Vector& b) const;
        
        // Debug and diagnostics
        void print_stats() const;
        std::string get_info() const;
    };
    
    // Convenience factory functions
    std::unique_ptr<VectorDatabase> create_vector_database(const Config& config = Config{});
    std::unique_ptr<VectorDatabase> load_vector_database(const std::string& path);
    
    // Utility functions for working with vectors
    namespace Utils {
        Vector random_vector(size_t dimension, float min_val = -1.0f, float max_val = 1.0f);
        Vector normalize_vector(const Vector& vec);
        std::vector<Vector> generate_random_vectors(size_t count, size_t dimension);
        
        // Vector operations
        Vector add_vectors(const Vector& a, const Vector& b);
        Vector subtract_vectors(const Vector& a, const Vector& b);
        Vector multiply_vector(const Vector& vec, float scalar);
        float dot_product(const Vector& a, const Vector& b);
        
        // Data generation for testing
        std::vector<VectorEntry> generate_test_data(size_t count, size_t dimension);
        std::vector<std::string> generate_test_texts(size_t count);
    }
}