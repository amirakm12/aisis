#include "index.h"
#include <algorithm>
#include <fstream>
#include <iostream>

namespace VectorDB {
    
    // FlatIndex implementation
    FlatIndex::FlatIndex(DistanceMetric metric) : metric_(metric) {}
    
    void FlatIndex::add_vector(const VectorEntry& entry) {
        // Check if vector already exists
        auto it = std::find_if(vectors_.begin(), vectors_.end(),
                               [&entry](const VectorEntry& v) { return v.id == entry.id; });
        
        if (it != vectors_.end()) {
            *it = entry; // Update existing vector
        } else {
            vectors_.push_back(entry);
        }
    }
    
    void FlatIndex::remove_vector(VectorId id) {
        vectors_.erase(
            std::remove_if(vectors_.begin(), vectors_.end(),
                           [id](const VectorEntry& v) { return v.id == id; }),
            vectors_.end()
        );
    }
    
    std::vector<SearchResult> FlatIndex::search(const Vector& query, size_t k, float threshold) {
        std::vector<SearchResult> results;
        results.reserve(vectors_.size());
        
        // Calculate distances to all vectors
        for (const auto& vec_entry : vectors_) {
            float distance = SimilarityCalculator::calculate_distance(query, vec_entry.vector, metric_);
            
            if (threshold == 0.0f || distance <= threshold) {
                results.emplace_back(vec_entry.id, distance, vec_entry.metadata);
            }
        }
        
        // Sort by distance (ascending)
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.distance < b.distance;
                  });
        
        // Return top k results
        if (results.size() > k) {
            results.resize(k);
        }
        
        return results;
    }
    
    void FlatIndex::build() {
        // No building required for flat index
    }
    
    void FlatIndex::clear() {
        vectors_.clear();
    }
    
    size_t FlatIndex::size() const {
        return vectors_.size();
    }
    
    void FlatIndex::save(const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for saving index: " + path);
        }
        
        // Write number of vectors
        size_t count = vectors_.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
        
        // Write metric type
        int metric_int = static_cast<int>(metric_);
        file.write(reinterpret_cast<const char*>(&metric_int), sizeof(int));
        
        // Write vectors
        for (const auto& entry : vectors_) {
            // Write ID
            file.write(reinterpret_cast<const char*>(&entry.id), sizeof(VectorId));
            
            // Write vector
            size_t vec_size = entry.vector.size();
            file.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(entry.vector.data()), vec_size * sizeof(float));
            
            // Write metadata
            size_t meta_count = entry.metadata.size();
            file.write(reinterpret_cast<const char*>(&meta_count), sizeof(size_t));
            
            for (const auto& meta_pair : entry.metadata) {
                size_t key_len = meta_pair.first.length();
                size_t val_len = meta_pair.second.length();
                
                file.write(reinterpret_cast<const char*>(&key_len), sizeof(size_t));
                file.write(meta_pair.first.c_str(), key_len);
                file.write(reinterpret_cast<const char*>(&val_len), sizeof(size_t));
                file.write(meta_pair.second.c_str(), val_len);
            }
        }
    }
    
    void FlatIndex::load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for loading index: " + path);
        }
        
        vectors_.clear();
        
        // Read number of vectors
        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(size_t));
        
        // Read metric type
        int metric_int;
        file.read(reinterpret_cast<char*>(&metric_int), sizeof(int));
        metric_ = static_cast<DistanceMetric>(metric_int);
        
        vectors_.reserve(count);
        
        // Read vectors
        for (size_t i = 0; i < count; ++i) {
            VectorEntry entry;
            
            // Read ID
            file.read(reinterpret_cast<char*>(&entry.id), sizeof(VectorId));
            
            // Read vector
            size_t vec_size;
            file.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
            entry.vector.resize(vec_size);
            file.read(reinterpret_cast<char*>(entry.vector.data()), vec_size * sizeof(float));
            
            // Read metadata
            size_t meta_count;
            file.read(reinterpret_cast<char*>(&meta_count), sizeof(size_t));
            
            for (size_t j = 0; j < meta_count; ++j) {
                size_t key_len, val_len;
                
                file.read(reinterpret_cast<char*>(&key_len), sizeof(size_t));
                std::string key(key_len, '\0');
                file.read(&key[0], key_len);
                
                file.read(reinterpret_cast<char*>(&val_len), sizeof(size_t));
                std::string value(val_len, '\0');
                file.read(&value[0], val_len);
                
                entry.metadata[key] = value;
            }
            
            vectors_.push_back(entry);
        }
    }
    
    // IVFIndex implementation (basic structure)
    IVFIndex::IVFIndex(size_t nlist, size_t nprobe, DistanceMetric metric)
        : nlist_(nlist), nprobe_(nprobe), metric_(metric), is_trained_(false) {
        clusters_.resize(nlist_);
    }
    
    void IVFIndex::add_vector(const VectorEntry& entry) {
        if (!is_trained_) {
            // Store vectors for training
            size_t cluster_idx = entry.id % nlist_; // Simple assignment
            clusters_[cluster_idx].vectors.push_back(entry);
        } else {
            // Find nearest cluster and add vector
            size_t cluster_idx = find_nearest_cluster(entry.vector);
            clusters_[cluster_idx].vectors.push_back(entry);
        }
    }
    
    void IVFIndex::remove_vector(VectorId id) {
        for (auto& cluster : clusters_) {
            cluster.vectors.erase(
                std::remove_if(cluster.vectors.begin(), cluster.vectors.end(),
                               [id](const VectorEntry& v) { return v.id == id; }),
                cluster.vectors.end()
            );
        }
    }
    
    std::vector<SearchResult> IVFIndex::search(const Vector& query, size_t k, float threshold) {
        if (!is_trained_) {
            throw std::runtime_error("IVF index not trained");
        }
        
        std::vector<SearchResult> results;
        
        // Find nearest clusters to search
        auto cluster_indices = find_nearest_clusters(query, nprobe_);
        
        // Search in selected clusters
        for (size_t cluster_idx : cluster_indices) {
            const auto& cluster = clusters_[cluster_idx];
            
            for (const auto& vec_entry : cluster.vectors) {
                float distance = SimilarityCalculator::calculate_distance(query, vec_entry.vector, metric_);
                
                if (threshold == 0.0f || distance <= threshold) {
                    results.emplace_back(vec_entry.id, distance, vec_entry.metadata);
                }
            }
        }
        
        // Sort and return top k
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.distance < b.distance;
                  });
        
        if (results.size() > k) {
            results.resize(k);
        }
        
        return results;
    }
    
    void IVFIndex::build() {
        if (clusters_.empty()) return;
        
        // Collect all vectors for training
        std::vector<VectorEntry> all_vectors;
        for (const auto& cluster : clusters_) {
            all_vectors.insert(all_vectors.end(), cluster.vectors.begin(), cluster.vectors.end());
        }
        
        if (all_vectors.empty()) return;
        
        train_centroids(all_vectors);
        
        // Reassign vectors to clusters based on trained centroids
        for (auto& cluster : clusters_) {
            cluster.vectors.clear();
        }
        
        for (const auto& entry : all_vectors) {
            size_t cluster_idx = find_nearest_cluster(entry.vector);
            clusters_[cluster_idx].vectors.push_back(entry);
        }
        
        is_trained_ = true;
    }
    
    void IVFIndex::clear() {
        for (auto& cluster : clusters_) {
            cluster.vectors.clear();
            cluster.centroid.clear();
        }
        is_trained_ = false;
    }
    
    size_t IVFIndex::size() const {
        size_t total = 0;
        for (const auto& cluster : clusters_) {
            total += cluster.vectors.size();
        }
        return total;
    }
    
    void IVFIndex::save(const std::string& path) {
        // Basic implementation - save to file
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot save IVF index to: " + path);
        }
        
        // Write basic parameters
        file.write(reinterpret_cast<const char*>(&nlist_), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&nprobe_), sizeof(size_t));
        int metric_int = static_cast<int>(metric_);
        file.write(reinterpret_cast<const char*>(&metric_int), sizeof(int));
        file.write(reinterpret_cast<const char*>(&is_trained_), sizeof(bool));
        
        // Save clusters (simplified)
        for (const auto& cluster : clusters_) {
            size_t centroid_size = cluster.centroid.size();
            file.write(reinterpret_cast<const char*>(&centroid_size), sizeof(size_t));
            if (centroid_size > 0) {
                file.write(reinterpret_cast<const char*>(cluster.centroid.data()), centroid_size * sizeof(float));
            }
            
            size_t vector_count = cluster.vectors.size();
            file.write(reinterpret_cast<const char*>(&vector_count), sizeof(size_t));
            // Note: Full vector serialization would be implemented here
        }
    }
    
    void IVFIndex::load(const std::string& path) {
        // Basic implementation - load from file
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot load IVF index from: " + path);
        }
        
        // Read basic parameters
        file.read(reinterpret_cast<char*>(&nlist_), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&nprobe_), sizeof(size_t));
        int metric_int;
        file.read(reinterpret_cast<char*>(&metric_int), sizeof(int));
        metric_ = static_cast<DistanceMetric>(metric_int);
        file.read(reinterpret_cast<char*>(&is_trained_), sizeof(bool));
        
        clusters_.resize(nlist_);
        // Load clusters (simplified)
        for (auto& cluster : clusters_) {
            size_t centroid_size;
            file.read(reinterpret_cast<char*>(&centroid_size), sizeof(size_t));
            if (centroid_size > 0) {
                cluster.centroid.resize(centroid_size);
                file.read(reinterpret_cast<char*>(cluster.centroid.data()), centroid_size * sizeof(float));
            }
            
            size_t vector_count;
            file.read(reinterpret_cast<char*>(&vector_count), sizeof(size_t));
            // Note: Full vector deserialization would be implemented here
        }
    }
    
    void IVFIndex::train_centroids(const std::vector<VectorEntry>& training_data) {
        if (training_data.empty()) return;
        
        size_t dimension = training_data[0].vector.size();
        
        // Simple k-means initialization: randomly assign vectors to clusters
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, nlist_ - 1);
        
        // Initialize centroids with random vectors
        for (size_t i = 0; i < nlist_ && i < training_data.size(); ++i) {
            clusters_[i].centroid = training_data[i].vector;
        }
        
        // Simple k-means (limited iterations)
        for (int iter = 0; iter < 10; ++iter) {
            // Clear cluster assignments
            for (auto& cluster : clusters_) {
                cluster.vectors.clear();
            }
            
            // Assign vectors to nearest centroids
            for (const auto& entry : training_data) {
                size_t nearest_cluster = find_nearest_cluster(entry.vector);
                clusters_[nearest_cluster].vectors.push_back(entry);
            }
            
            // Update centroids
            bool changed = false;
            for (auto& cluster : clusters_) {
                if (cluster.vectors.empty()) continue;
                
                Vector new_centroid(dimension, 0.0f);
                for (const auto& entry : cluster.vectors) {
                    for (size_t d = 0; d < dimension; ++d) {
                        new_centroid[d] += entry.vector[d];
                    }
                }
                
                for (size_t d = 0; d < dimension; ++d) {
                    new_centroid[d] /= cluster.vectors.size();
                }
                
                // Check if centroid changed significantly
                float change = SimilarityCalculator::euclidean_distance(cluster.centroid, new_centroid);
                if (change > 0.001f) {
                    changed = true;
                }
                
                cluster.centroid = new_centroid;
            }
            
            if (!changed) break;
        }
    }
    
    size_t IVFIndex::find_nearest_cluster(const Vector& query) const {
        if (!is_trained_ || clusters_.empty()) {
            return 0;
        }
        
        size_t best_cluster = 0;
        float best_distance = std::numeric_limits<float>::max();
        
        for (size_t i = 0; i < clusters_.size(); ++i) {
            if (clusters_[i].centroid.empty()) continue;
            
            float distance = SimilarityCalculator::calculate_distance(query, clusters_[i].centroid, metric_);
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = i;
            }
        }
        
        return best_cluster;
    }
    
    std::vector<size_t> IVFIndex::find_nearest_clusters(const Vector& query, size_t nprobe) const {
        std::vector<std::pair<float, size_t>> distances;
        
        for (size_t i = 0; i < clusters_.size(); ++i) {
            if (clusters_[i].centroid.empty()) continue;
            
            float distance = SimilarityCalculator::calculate_distance(query, clusters_[i].centroid, metric_);
            distances.emplace_back(distance, i);
        }
        
        std::sort(distances.begin(), distances.end());
        
        std::vector<size_t> result;
        size_t count = std::min(nprobe, distances.size());
        for (size_t i = 0; i < count; ++i) {
            result.push_back(distances[i].second);
        }
        
        return result;
    }
    
    // Placeholder implementations for other index types
    HNSWIndex::HNSWIndex(size_t M, size_t ef_construction, size_t ef_search, DistanceMetric metric)
        : M_(M), ef_construction_(ef_construction), ef_search_(ef_search), metric_(metric), entry_point_(0), rng_(std::random_device{}()) {}
    
    void HNSWIndex::add_vector(const VectorEntry& entry) {
        // Placeholder implementation
        auto node = std::make_unique<Node>();
        node->id = entry.id;
        node->vector = entry.vector;
        node->metadata = entry.metadata;
        nodes_[entry.id] = std::move(node);
    }
    
    void HNSWIndex::remove_vector(VectorId id) {
        nodes_.erase(id);
    }
    
    std::vector<SearchResult> HNSWIndex::search(const Vector& query, size_t k, float threshold) {
        // Simplified implementation - just brute force for now
        std::vector<SearchResult> results;
        
        for (const auto& pair : nodes_) {
            float distance = SimilarityCalculator::calculate_distance(query, pair.second->vector, metric_);
            if (threshold == 0.0f || distance <= threshold) {
                results.emplace_back(pair.first, distance, pair.second->metadata);
            }
        }
        
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.distance < b.distance;
                  });
        
        if (results.size() > k) {
            results.resize(k);
        }
        
        return results;
    }
    
    void HNSWIndex::build() {
        // Placeholder
    }
    
    void HNSWIndex::clear() {
        nodes_.clear();
        layers_.clear();
    }
    
    size_t HNSWIndex::size() const {
        return nodes_.size();
    }
    
    void HNSWIndex::save(const std::string& path) {
        // Placeholder
    }
    
    void HNSWIndex::load(const std::string& path) {
        // Placeholder
    }
    
    // LSHIndex placeholder implementations
    LSHIndex::LSHIndex(size_t num_tables, size_t num_bits, size_t dimension, DistanceMetric metric)
        : num_tables_(num_tables), num_bits_(num_bits), dimension_(dimension), metric_(metric), rng_(std::random_device{}()) {
        hash_tables_.resize(num_tables_);
        generate_hash_functions();
    }
    
    void LSHIndex::generate_hash_functions() {
        hash_functions_.resize(num_tables_);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < num_tables_; ++i) {
            hash_functions_[i].resize(dimension_);
            for (size_t j = 0; j < dimension_; ++j) {
                hash_functions_[i][j] = dist(rng_);
            }
        }
    }
    
    std::string LSHIndex::compute_hash(const Vector& vec, size_t table_idx) const {
        std::string hash_str;
        hash_str.reserve(num_bits_);
        
        for (size_t bit = 0; bit < num_bits_; ++bit) {
            float dot_product = 0.0f;
            size_t func_idx = (table_idx * num_bits_ + bit) % hash_functions_[table_idx].size();
            
            for (size_t i = 0; i < vec.size() && i < dimension_; ++i) {
                dot_product += vec[i] * hash_functions_[table_idx][i];
            }
            
            hash_str += (dot_product >= 0) ? '1' : '0';
        }
        
        return hash_str;
    }
    
    void LSHIndex::add_vector(const VectorEntry& entry) {
        for (size_t i = 0; i < num_tables_; ++i) {
            std::string hash_val = compute_hash(entry.vector, i);
            hash_tables_[i].buckets[hash_val].push_back(entry);
        }
    }
    
    void LSHIndex::remove_vector(VectorId id) {
        for (auto& table : hash_tables_) {
            for (auto& bucket_pair : table.buckets) {
                auto& bucket = bucket_pair.second;
                bucket.erase(
                    std::remove_if(bucket.begin(), bucket.end(),
                                   [id](const VectorEntry& v) { return v.id == id; }),
                    bucket.end()
                );
            }
        }
    }
    
    std::vector<SearchResult> LSHIndex::search(const Vector& query, size_t k, float threshold) {
        std::unordered_set<VectorId> candidates;
        
        // Collect candidates from all hash tables
        for (size_t i = 0; i < num_tables_; ++i) {
            std::string hash_val = compute_hash(query, i);
            auto it = hash_tables_[i].buckets.find(hash_val);
            if (it != hash_tables_[i].buckets.end()) {
                for (const auto& entry : it->second) {
                    candidates.insert(entry.id);
                }
            }
        }
        
        // Calculate distances for candidates
        std::vector<SearchResult> results;
        for (VectorId candidate_id : candidates) {
            // Find the vector entry (simplified - in practice would be more efficient)
            for (const auto& table : hash_tables_) {
                for (const auto& bucket_pair : table.buckets) {
                    for (const auto& entry : bucket_pair.second) {
                        if (entry.id == candidate_id) {
                            float distance = SimilarityCalculator::calculate_distance(query, entry.vector, metric_);
                            if (threshold == 0.0f || distance <= threshold) {
                                results.emplace_back(entry.id, distance, entry.metadata);
                            }
                            goto next_candidate;
                        }
                    }
                }
                next_candidate:;
            }
        }
        
        // Sort and return top k
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.distance < b.distance;
                  });
        
        if (results.size() > k) {
            results.resize(k);
        }
        
        return results;
    }
    
    void LSHIndex::build() {
        // No additional building required for LSH
    }
    
    void LSHIndex::clear() {
        for (auto& table : hash_tables_) {
            table.buckets.clear();
        }
    }
    
    size_t LSHIndex::size() const {
        size_t total = 0;
        for (const auto& table : hash_tables_) {
            for (const auto& bucket_pair : table.buckets) {
                total += bucket_pair.second.size();
            }
        }
        return total / num_tables_; // Approximate, since vectors are duplicated across tables
    }
    
    void LSHIndex::save(const std::string& path) {
        // Placeholder
    }
    
    void LSHIndex::load(const std::string& path) {
        // Placeholder
    }
    
    // Factory function
    std::unique_ptr<Index> create_index(IndexType type, const Config& config) {
        switch (type) {
            case IndexType::FLAT:
                return std::make_unique<FlatIndex>(config.distance_metric);
            case IndexType::IVF:
                return std::make_unique<IVFIndex>(config.index_params.nlist, config.index_params.nprobe, config.distance_metric);
            case IndexType::HNSW:
                return std::make_unique<HNSWIndex>(config.index_params.M, config.index_params.ef_construction, 
                                                   config.index_params.ef_search, config.distance_metric);
            case IndexType::LSH:
                return std::make_unique<LSHIndex>(10, config.index_params.num_bits, config.dimension, config.distance_metric);
            default:
                throw std::invalid_argument("Unknown index type");
        }
    }
}