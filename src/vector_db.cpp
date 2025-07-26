#include "vector_db.h"
#include <iostream>
#include <sstream>
#include <random>

namespace VectorDB {
    
    // VectorDatabase implementation
    VectorDatabase::VectorDatabase(const Config& config) 
        : config_(config), next_id_(1), is_loaded_(false) {
        
        // Initialize index
        index_ = create_index(config_.index_type, config_);
        
        // Initialize embedding generator
        embedding_generator_ = create_embedding_generator(EmbeddingType::HASH, config_.dimension);
        
        // Initialize storage manager
        storage_manager_ = create_storage_manager(StorageType::FILE, config_.storage_path);
    }
    
    VectorDatabase::~VectorDatabase() {
        if (storage_manager_) {
            storage_manager_->close();
        }
    }
    
    void VectorDatabase::set_config(const Config& config) {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        config_ = config;
        
        // Recreate components if necessary
        if (config_.index_type != get_index_type()) {
            index_ = create_index(config_.index_type, config_);
        }
        
        // Save configuration
        if (storage_manager_) {
            storage_manager_->save_config(config_);
        }
    }
    
    void VectorDatabase::build_index() {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        if (!index_) return;
        
        // Load all vectors from storage and add to index
        auto vectors = storage_manager_->load_all_vectors();
        for (const auto& entry : vectors) {
            index_->add_vector(entry);
        }
        
        // Build the index
        index_->build();
    }
    
    void VectorDatabase::rebuild_index() {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        if (!index_) return;
        
        index_->clear();
        build_index();
    }
    
    void VectorDatabase::clear_index() {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        if (index_) {
            index_->clear();
        }
    }
    
    VectorId VectorDatabase::add_vector(const Vector& vector, const Metadata& metadata) {
        validate_vector_dimension(vector);
        
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        VectorId id = generate_id();
        VectorEntry entry(id, vector, metadata);
        
        // Add to storage
        storage_manager_->save_vector(entry);
        
        // Add to index
        if (index_) {
            index_->add_vector(entry);
        }
        
        return id;
    }
    
    VectorId VectorDatabase::add_text(const std::string& text, const Metadata& metadata) {
        if (!embedding_generator_) {
            throw std::runtime_error("No embedding generator available");
        }
        
        Vector embedding = embedding_generator_->generate_embedding(text);
        return add_vector(embedding, metadata);
    }
    
    VectorId VectorDatabase::add_features(const std::vector<float>& features, const Metadata& metadata) {
        if (!embedding_generator_) {
            throw std::runtime_error("No embedding generator available");
        }
        
        Vector embedding = embedding_generator_->generate_embedding(features);
        return add_vector(embedding, metadata);
    }
    
    bool VectorDatabase::remove_vector(VectorId id) {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        // Remove from storage
        storage_manager_->delete_vector(id);
        
        // Remove from index
        if (index_) {
            index_->remove_vector(id);
        }
        
        return true;
    }
    
    bool VectorDatabase::update_vector(VectorId id, const Vector& vector, const Metadata& metadata) {
        validate_vector_dimension(vector);
        
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        VectorEntry entry(id, vector, metadata);
        
        // Update in storage
        storage_manager_->save_vector(entry);
        
        // Update in index
        if (index_) {
            index_->add_vector(entry); // This will update existing entry
        }
        
        return true;
    }
    
    bool VectorDatabase::update_metadata(VectorId id, const Metadata& metadata) {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        // Load existing vector
        VectorEntry entry;
        if (!storage_manager_->load_vector(id, entry)) {
            return false;
        }
        
        // Update metadata
        entry.metadata = metadata;
        
        // Save back
        storage_manager_->save_vector(entry);
        
        // Update in index
        if (index_) {
            index_->add_vector(entry);
        }
        
        return true;
    }
    
    bool VectorDatabase::get_vector(VectorId id, VectorEntry& entry) const {
        std::lock_guard<std::mutex> lock(db_mutex_);
        return storage_manager_->load_vector(id, entry);
    }
    
    std::vector<VectorEntry> VectorDatabase::get_vectors(const std::vector<VectorId>& ids) const {
        std::lock_guard<std::mutex> lock(db_mutex_);
        return storage_manager_->load_vectors_batch(ids);
    }
    
    std::vector<SearchResult> VectorDatabase::search(const Vector& query, size_t k, float threshold) const {
        validate_vector_dimension(query);
        
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        if (!index_) {
            return {};
        }
        
        return index_->search(query, k, threshold);
    }
    
    std::vector<SearchResult> VectorDatabase::search_text(const std::string& text, size_t k, float threshold) const {
        if (!embedding_generator_) {
            throw std::runtime_error("No embedding generator available");
        }
        
        Vector query = embedding_generator_->generate_embedding(text);
        return search(query, k, threshold);
    }
    
    std::vector<SearchResult> VectorDatabase::search_features(const std::vector<float>& features, size_t k, float threshold) const {
        if (!embedding_generator_) {
            throw std::runtime_error("No embedding generator available");
        }
        
        Vector query = embedding_generator_->generate_embedding(features);
        return search(query, k, threshold);
    }
    
    std::vector<SearchResult> VectorDatabase::search_with_filter(
        const Vector& query, 
        size_t k, 
        const std::function<bool(const Metadata&)>& filter,
        float threshold) const {
        
        // Get all results and then filter
        auto all_results = search(query, 0, threshold); // Get all results
        
        std::vector<SearchResult> filtered_results;
        for (const auto& result : all_results) {
            if (filter(result.metadata)) {
                filtered_results.push_back(result);
                if (filtered_results.size() >= k) {
                    break;
                }
            }
        }
        
        return filtered_results;
    }
    
    std::vector<VectorId> VectorDatabase::add_vectors_batch(const std::vector<VectorEntry>& entries) {
        std::vector<VectorId> ids;
        ids.reserve(entries.size());
        
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        for (const auto& entry : entries) {
            validate_vector_dimension(entry.vector);
            
            VectorId id = generate_id();
            VectorEntry new_entry(id, entry.vector, entry.metadata);
            
            // Add to storage
            storage_manager_->save_vector(new_entry);
            
            // Add to index
            if (index_) {
                index_->add_vector(new_entry);
            }
            
            ids.push_back(id);
        }
        
        return ids;
    }
    
    std::vector<VectorId> VectorDatabase::add_texts_batch(const std::vector<std::string>& texts, 
                                                          const std::vector<Metadata>& metadata) {
        if (!embedding_generator_) {
            throw std::runtime_error("No embedding generator available");
        }
        
        std::vector<VectorEntry> entries;
        entries.reserve(texts.size());
        
        for (size_t i = 0; i < texts.size(); ++i) {
            Vector embedding = embedding_generator_->generate_embedding(texts[i]);
            Metadata meta = (i < metadata.size()) ? metadata[i] : Metadata{};
            entries.emplace_back(0, embedding, meta); // ID will be generated in add_vectors_batch
        }
        
        return add_vectors_batch(entries);
    }
    
    bool VectorDatabase::remove_vectors_batch(const std::vector<VectorId>& ids) {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        for (VectorId id : ids) {
            storage_manager_->delete_vector(id);
            if (index_) {
                index_->remove_vector(id);
            }
        }
        
        return true;
    }
    
    size_t VectorDatabase::size() const {
        std::lock_guard<std::mutex> lock(db_mutex_);
        return storage_manager_->vector_count();
    }
    
    bool VectorDatabase::empty() const {
        return size() == 0;
    }
    
    std::vector<VectorId> VectorDatabase::get_all_ids() const {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        auto vectors = storage_manager_->load_all_vectors();
        std::vector<VectorId> ids;
        ids.reserve(vectors.size());
        
        for (const auto& entry : vectors) {
            ids.push_back(entry.id);
        }
        
        return ids;
    }
    
    void VectorDatabase::save(const std::string& path) {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        std::string save_path = path.empty() ? config_.storage_path : path;
        
        // Save configuration
        storage_manager_->save_config(config_);
        
        // Save index if needed
        if (index_) {
            std::string index_path = save_path + "/index.bin";
            try {
                index_->save(index_path);
                storage_manager_->save_index_metadata("saved", "true");
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not save index: " << e.what() << std::endl;
            }
        }
        
        // Flush storage
        storage_manager_->flush();
    }
    
    void VectorDatabase::load(const std::string& path) {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        std::string load_path = path.empty() ? config_.storage_path : path;
        
        // Create new storage manager for the path
        if (!path.empty()) {
            storage_manager_ = create_storage_manager(StorageType::FILE, load_path);
        }
        
        // Load configuration
        Config loaded_config;
        if (storage_manager_->load_config(loaded_config)) {
            config_ = loaded_config;
            
            // Recreate index with loaded config
            index_ = create_index(config_.index_type, config_);
        }
        
        // Load index if it exists
        std::string index_metadata;
        if (storage_manager_->load_index_metadata("saved", index_metadata) && index_metadata == "true") {
            std::string index_path = load_path + "/index.bin";
            try {
                index_->load(index_path);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not load index, rebuilding: " << e.what() << std::endl;
                build_index();
            }
        } else {
            // Build index from stored vectors
            build_index();
        }
        
        is_loaded_ = true;
    }
    
    void VectorDatabase::flush() {
        std::lock_guard<std::mutex> lock(db_mutex_);
        storage_manager_->flush();
    }
    
    void VectorDatabase::compact() {
        std::lock_guard<std::mutex> lock(db_mutex_);
        storage_manager_->compact();
    }
    
    void VectorDatabase::optimize() {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        // Rebuild index for optimization
        if (index_) {
            index_->build();
        }
        
        // Compact storage
        storage_manager_->compact();
    }
    
    void VectorDatabase::set_embedding_generator(std::unique_ptr<EmbeddingGenerator> generator) {
        std::lock_guard<std::mutex> lock(db_mutex_);
        embedding_generator_ = std::move(generator);
    }
    
    void VectorDatabase::set_index_type(IndexType type) {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        config_.index_type = type;
        index_ = create_index(type, config_);
        
        // Rebuild index with new type
        build_index();
    }
    
    void VectorDatabase::set_distance_metric(DistanceMetric metric) {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        config_.distance_metric = metric;
        index_ = create_index(config_.index_type, config_);
        
        // Rebuild index with new metric
        build_index();
    }
    
    Vector VectorDatabase::generate_embedding(const std::string& text) const {
        if (!embedding_generator_) {
            throw std::runtime_error("No embedding generator available");
        }
        return embedding_generator_->generate_embedding(text);
    }
    
    Vector VectorDatabase::generate_embedding(const std::vector<float>& features) const {
        if (!embedding_generator_) {
            throw std::runtime_error("No embedding generator available");
        }
        return embedding_generator_->generate_embedding(features);
    }
    
    float VectorDatabase::calculate_distance(const Vector& a, const Vector& b) const {
        return SimilarityCalculator::calculate_distance(a, b, config_.distance_metric);
    }
    
    void VectorDatabase::print_stats() const {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        std::cout << "=== Vector Database Statistics ===" << std::endl;
        std::cout << "Total vectors: " << size() << std::endl;
        std::cout << "Dimension: " << config_.dimension << std::endl;
        std::cout << "Index type: " << static_cast<int>(config_.index_type) << std::endl;
        std::cout << "Distance metric: " << static_cast<int>(config_.distance_metric) << std::endl;
        std::cout << "Storage path: " << config_.storage_path << std::endl;
        
        if (index_) {
            std::cout << "Index size: " << index_->size() << std::endl;
        }
        
        std::cout << "=================================" << std::endl;
    }
    
    std::string VectorDatabase::get_info() const {
        std::lock_guard<std::mutex> lock(db_mutex_);
        
        std::ostringstream oss;
        oss << "VectorDB Info:\n";
        oss << "  Vectors: " << size() << "\n";
        oss << "  Dimension: " << config_.dimension << "\n";
        oss << "  Index: " << static_cast<int>(config_.index_type) << "\n";
        oss << "  Metric: " << static_cast<int>(config_.distance_metric) << "\n";
        oss << "  Storage: " << config_.storage_path;
        
        return oss.str();
    }
    
    VectorId VectorDatabase::generate_id() {
        return next_id_++;
    }
    
    void VectorDatabase::validate_vector_dimension(const Vector& vec) const {
        if (vec.size() != config_.dimension) {
            throw DimensionMismatchException(
                "Vector dimension " + std::to_string(vec.size()) + 
                " doesn't match configured dimension " + std::to_string(config_.dimension)
            );
        }
    }
    
    // Factory functions
    std::unique_ptr<VectorDatabase> create_vector_database(const Config& config) {
        return std::make_unique<VectorDatabase>(config);
    }
    
    std::unique_ptr<VectorDatabase> load_vector_database(const std::string& path) {
        auto db = std::make_unique<VectorDatabase>();
        db->load(path);
        return db;
    }
    
    // Utility functions
    namespace Utils {
        Vector random_vector(size_t dimension, float min_val, float max_val) {
            Vector vec(dimension);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(min_val, max_val);
            
            for (size_t i = 0; i < dimension; ++i) {
                vec[i] = dis(gen);
            }
            
            return vec;
        }
        
        Vector normalize_vector(const Vector& vec) {
            return SimilarityCalculator::normalize_vector(vec);
        }
        
        std::vector<Vector> generate_random_vectors(size_t count, size_t dimension) {
            std::vector<Vector> vectors;
            vectors.reserve(count);
            
            for (size_t i = 0; i < count; ++i) {
                vectors.push_back(random_vector(dimension));
            }
            
            return vectors;
        }
        
        Vector add_vectors(const Vector& a, const Vector& b) {
            if (a.size() != b.size()) {
                throw DimensionMismatchException("Vector dimensions don't match for addition");
            }
            
            Vector result(a.size());
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] + b[i];
            }
            
            return result;
        }
        
        Vector subtract_vectors(const Vector& a, const Vector& b) {
            if (a.size() != b.size()) {
                throw DimensionMismatchException("Vector dimensions don't match for subtraction");
            }
            
            Vector result(a.size());
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] - b[i];
            }
            
            return result;
        }
        
        Vector multiply_vector(const Vector& vec, float scalar) {
            Vector result(vec.size());
            for (size_t i = 0; i < vec.size(); ++i) {
                result[i] = vec[i] * scalar;
            }
            
            return result;
        }
        
        float dot_product(const Vector& a, const Vector& b) {
            return SimilarityCalculator::dot_product(a, b);
        }
        
        std::vector<VectorEntry> generate_test_data(size_t count, size_t dimension) {
            std::vector<VectorEntry> entries;
            entries.reserve(count);
            
            for (size_t i = 0; i < count; ++i) {
                Vector vec = random_vector(dimension);
                Metadata meta = {{"id", std::to_string(i)}, {"type", "test"}};
                entries.emplace_back(i + 1, vec, meta);
            }
            
            return entries;
        }
        
        std::vector<std::string> generate_test_texts(size_t count) {
            std::vector<std::string> texts;
            texts.reserve(count);
            
            std::vector<std::string> words = {
                "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                "machine", "learning", "vector", "database", "search", "similarity",
                "artificial", "intelligence", "neural", "network", "embedding", "index"
            };
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> word_dis(0, words.size() - 1);
            std::uniform_int_distribution<> length_dis(3, 10);
            
            for (size_t i = 0; i < count; ++i) {
                std::string text;
                size_t length = length_dis(gen);
                
                for (size_t j = 0; j < length; ++j) {
                    if (j > 0) text += " ";
                    text += words[word_dis(gen)];
                }
                
                texts.push_back(text);
            }
            
            return texts;
        }
    }
}