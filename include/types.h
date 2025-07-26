#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace VectorDB {
    // Type aliases
    using VectorId = uint64_t;
    using Vector = std::vector<float>;
    using Metadata = std::unordered_map<std::string, std::string>;
    
    // Distance metrics
    enum class DistanceMetric {
        EUCLIDEAN,
        COSINE,
        DOT_PRODUCT,
        MANHATTAN
    };
    
    // Index types
    enum class IndexType {
        FLAT,          // Brute force search
        IVF,           // Inverted file index
        HNSW,          // Hierarchical Navigable Small World
        LSH            // Locality Sensitive Hashing
    };
    
    // Vector entry structure
    struct VectorEntry {
        VectorId id;
        Vector vector;
        Metadata metadata;
        
        VectorEntry() = default;
        VectorEntry(VectorId id, const Vector& vec, const Metadata& meta = {})
            : id(id), vector(vec), metadata(meta) {}
    };
    
    // Search result structure
    struct SearchResult {
        VectorId id;
        float distance;
        Metadata metadata;
        
        SearchResult() = default;
        SearchResult(VectorId id, float dist, const Metadata& meta = {})
            : id(id), distance(dist), metadata(meta) {}
    };
    
    // Configuration structure
    struct Config {
        size_t dimension = 128;
        IndexType index_type = IndexType::FLAT;
        DistanceMetric distance_metric = DistanceMetric::COSINE;
        size_t max_vectors = 1000000;
        std::string storage_path = "./vectordb_data";
        
        // Index-specific parameters
        struct {
            size_t nlist = 100;        // For IVF
            size_t nprobe = 10;        // For IVF
            size_t M = 16;             // For HNSW
            size_t ef_construction = 200; // For HNSW
            size_t ef_search = 50;     // For HNSW
            size_t num_bits = 8;       // For LSH
        } index_params;
    };
    
    // Exception types
    class VectorDBException : public std::exception {
    private:
        std::string message;
    public:
        explicit VectorDBException(const std::string& msg) : message(msg) {}
        const char* what() const noexcept override { return message.c_str(); }
    };
    
    class DimensionMismatchException : public VectorDBException {
    public:
        explicit DimensionMismatchException(const std::string& msg) 
            : VectorDBException("Dimension mismatch: " + msg) {}
    };
    
    class IndexNotFoundException : public VectorDBException {
    public:
        explicit IndexNotFoundException(const std::string& msg)
            : VectorDBException("Index not found: " + msg) {}
    };
}