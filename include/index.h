#pragma once

#include "types.h"
#include "similarity.h"
#include <memory>
#include <queue>
#include <random>
#include <unordered_set>
#include <limits>

namespace VectorDB {
    // Abstract base class for all index types
    class Index {
    public:
        virtual ~Index() = default;
        
        virtual void add_vector(const VectorEntry& entry) = 0;
        virtual void remove_vector(VectorId id) = 0;
        virtual std::vector<SearchResult> search(const Vector& query, size_t k, float threshold = 0.0f) = 0;
        virtual void build() = 0;
        virtual void clear() = 0;
        virtual size_t size() const = 0;
        
        virtual void save(const std::string& path) = 0;
        virtual void load(const std::string& path) = 0;
    };
    
    // Flat index (brute force search)
    class FlatIndex : public Index {
    private:
        std::vector<VectorEntry> vectors_;
        DistanceMetric metric_;
        
    public:
        explicit FlatIndex(DistanceMetric metric = DistanceMetric::COSINE);
        
        void add_vector(const VectorEntry& entry) override;
        void remove_vector(VectorId id) override;
        std::vector<SearchResult> search(const Vector& query, size_t k, float threshold = 0.0f) override;
        void build() override;
        void clear() override;
        size_t size() const override;
        
        void save(const std::string& path) override;
        void load(const std::string& path) override;
    };
    
    // IVF (Inverted File) Index
    class IVFIndex : public Index {
    private:
        struct Cluster {
            Vector centroid;
            std::vector<VectorEntry> vectors;
        };
        
        std::vector<Cluster> clusters_;
        size_t nlist_;
        size_t nprobe_;
        DistanceMetric metric_;
        bool is_trained_;
        
        void train_centroids(const std::vector<VectorEntry>& training_data);
        size_t find_nearest_cluster(const Vector& query) const;
        std::vector<size_t> find_nearest_clusters(const Vector& query, size_t nprobe) const;
        
    public:
        IVFIndex(size_t nlist, size_t nprobe, DistanceMetric metric = DistanceMetric::COSINE);
        
        void add_vector(const VectorEntry& entry) override;
        void remove_vector(VectorId id) override;
        std::vector<SearchResult> search(const Vector& query, size_t k, float threshold = 0.0f) override;
        void build() override;
        void clear() override;
        size_t size() const override;
        
        void save(const std::string& path) override;
        void load(const std::string& path) override;
    };
    
    // HNSW (Hierarchical Navigable Small World) Index
    class HNSWIndex : public Index {
    private:
        struct Node {
            VectorId id;
            Vector vector;
            Metadata metadata;
            std::vector<std::vector<VectorId>> connections; // connections per layer
        };
        
        std::unordered_map<VectorId, std::unique_ptr<Node>> nodes_;
        std::vector<std::vector<VectorId>> layers_;
        size_t M_;
        size_t ef_construction_;
        size_t ef_search_;
        DistanceMetric metric_;
        VectorId entry_point_;
        std::mt19937 rng_;
        
        int get_random_level();
        std::vector<VectorId> search_layer(const Vector& query, VectorId entry_point, 
                                           size_t ef, int layer) const;
        void select_neighbors(std::vector<VectorId>& candidates, size_t M) const;
        
    public:
        HNSWIndex(size_t M, size_t ef_construction, size_t ef_search, 
                  DistanceMetric metric = DistanceMetric::COSINE);
        
        void add_vector(const VectorEntry& entry) override;
        void remove_vector(VectorId id) override;
        std::vector<SearchResult> search(const Vector& query, size_t k, float threshold = 0.0f) override;
        void build() override;
        void clear() override;
        size_t size() const override;
        
        void save(const std::string& path) override;
        void load(const std::string& path) override;
        
        void set_ef_search(size_t ef) { ef_search_ = ef; }
    };
    
    // LSH (Locality Sensitive Hashing) Index
    class LSHIndex : public Index {
    private:
        struct HashTable {
            std::unordered_map<std::string, std::vector<VectorEntry>> buckets;
        };
        
        std::vector<HashTable> hash_tables_;
        std::vector<std::vector<float>> hash_functions_;
        size_t num_tables_;
        size_t num_bits_;
        size_t dimension_;
        DistanceMetric metric_;
        std::mt19937 rng_;
        
        void generate_hash_functions();
        std::string compute_hash(const Vector& vec, size_t table_idx) const;
        
    public:
        LSHIndex(size_t num_tables, size_t num_bits, size_t dimension,
                 DistanceMetric metric = DistanceMetric::COSINE);
        
        void add_vector(const VectorEntry& entry) override;
        void remove_vector(VectorId id) override;
        std::vector<SearchResult> search(const Vector& query, size_t k, float threshold = 0.0f) override;
        void build() override;
        void clear() override;
        size_t size() const override;
        
        void save(const std::string& path) override;
        void load(const std::string& path) override;
    };
    
    // Factory function to create indices
    std::unique_ptr<Index> create_index(IndexType type, const Config& config);
}