#include "storage.h"
#include <filesystem>
#include <iostream>
#include <sstream>

namespace VectorDB {
    
    // FileStorage implementation
    FileStorage::FileStorage(const std::string& base_path) 
        : base_path_(base_path), offsets_loaded_(false) {
        ensure_directory_exists(base_path_);
        vectors_file_ = base_path_ + "/vectors.bin";
        metadata_file_ = base_path_ + "/metadata.txt";
        index_file_ = base_path_ + "/index.bin";
    }
    
    FileStorage::~FileStorage() {
        close();
    }
    
    void FileStorage::ensure_directory_exists(const std::string& path) {
        std::filesystem::create_directories(path);
    }
    
    void FileStorage::save_vector(const VectorEntry& entry) {
        // Append to vectors file
        std::ofstream file(vectors_file_, std::ios::binary | std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open vectors file for writing");
        }
        
        size_t offset = file.tellp();
        
        // Write vector ID
        file.write(reinterpret_cast<const char*>(&entry.id), sizeof(VectorId));
        
        // Write vector
        write_vector(file, entry.vector);
        
        // Write metadata
        write_metadata(file, entry.metadata);
        
        file.close();
        
        // Update offset cache
        vector_offsets_[entry.id] = offset;
        save_offsets();
    }
    
    bool FileStorage::load_vector(VectorId id, VectorEntry& entry) {
        if (!offsets_loaded_) {
            load_offsets();
        }
        
        auto it = vector_offsets_.find(id);
        if (it == vector_offsets_.end()) {
            return false;
        }
        
        std::ifstream file(vectors_file_, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        file.seekg(it->second);
        
        // Read vector ID
        VectorId stored_id;
        file.read(reinterpret_cast<char*>(&stored_id), sizeof(VectorId));
        if (stored_id != id) {
            return false;
        }
        
        entry.id = id;
        
        // Read vector
        read_vector(file, entry.vector);
        
        // Read metadata
        read_metadata(file, entry.metadata);
        
        return true;
    }
    
    void FileStorage::delete_vector(VectorId id) {
        vector_offsets_.erase(id);
        save_offsets();
    }
    
    std::vector<VectorEntry> FileStorage::load_all_vectors() {
        if (!offsets_loaded_) {
            load_offsets();
        }
        
        std::vector<VectorEntry> vectors;
        vectors.reserve(vector_offsets_.size());
        
        for (const auto& pair : vector_offsets_) {
            VectorEntry entry;
            if (load_vector(pair.first, entry)) {
                vectors.push_back(entry);
            }
        }
        
        return vectors;
    }
    
    void FileStorage::clear() {
        vector_offsets_.clear();
        metadata_cache_.clear();
        
        // Remove files
        std::filesystem::remove(vectors_file_);
        std::filesystem::remove(metadata_file_);
        std::filesystem::remove(index_file_);
    }
    
    size_t FileStorage::count() const {
        if (!offsets_loaded_) {
            load_offsets();
        }
        return vector_offsets_.size();
    }
    
    void FileStorage::save_metadata(const std::string& key, const std::string& value) {
        metadata_cache_[key] = value;
        
        // Append to metadata file
        std::ofstream file(metadata_file_, std::ios::app);
        if (file.is_open()) {
            file << key << "=" << value << std::endl;
        }
    }
    
    bool FileStorage::load_metadata(const std::string& key, std::string& value) {
        // First check cache
        auto it = metadata_cache_.find(key);
        if (it != metadata_cache_.end()) {
            value = it->second;
            return true;
        }
        
        // Load from file
        std::ifstream file(metadata_file_);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string file_key = line.substr(0, pos);
                std::string file_value = line.substr(pos + 1);
                metadata_cache_[file_key] = file_value;
                
                if (file_key == key) {
                    value = file_value;
                    return true;
                }
            }
        }
        
        return false;
    }
    
    void FileStorage::flush() {
        save_offsets();
    }
    
    void FileStorage::close() {
        flush();
    }
    
    void FileStorage::load_offsets() const {
        std::ifstream file(index_file_, std::ios::binary);
        if (!file.is_open()) {
            offsets_loaded_ = true;
            return;
        }
        
        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(size_t));
        
        for (size_t i = 0; i < count; ++i) {
            VectorId id;
            size_t offset;
            file.read(reinterpret_cast<char*>(&id), sizeof(VectorId));
            file.read(reinterpret_cast<char*>(&offset), sizeof(size_t));
            vector_offsets_[id] = offset;
        }
        
        offsets_loaded_ = true;
    }
    
    void FileStorage::save_offsets() {
        std::ofstream file(index_file_, std::ios::binary);
        if (!file.is_open()) {
            return;
        }
        
        size_t count = vector_offsets_.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
        
        for (const auto& pair : vector_offsets_) {
            file.write(reinterpret_cast<const char*>(&pair.first), sizeof(VectorId));
            file.write(reinterpret_cast<const char*>(&pair.second), sizeof(size_t));
        }
    }
    
    void FileStorage::write_vector(std::ofstream& file, const Vector& vec) {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
    }
    
    void FileStorage::read_vector(std::ifstream& file, Vector& vec) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    }
    
    void FileStorage::write_metadata(std::ofstream& file, const Metadata& meta) {
        size_t count = meta.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
        
        for (const auto& pair : meta) {
            write_string(file, pair.first);
            write_string(file, pair.second);
        }
    }
    
    void FileStorage::read_metadata(std::ifstream& file, Metadata& meta) {
        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(size_t));
        
        meta.clear();
        for (size_t i = 0; i < count; ++i) {
            std::string key, value;
            read_string(file, key);
            read_string(file, value);
            meta[key] = value;
        }
    }
    
    void FileStorage::write_string(std::ofstream& file, const std::string& str) {
        size_t length = str.length();
        file.write(reinterpret_cast<const char*>(&length), sizeof(size_t));
        file.write(str.c_str(), length);
    }
    
    void FileStorage::read_string(std::ifstream& file, std::string& str) {
        size_t length;
        file.read(reinterpret_cast<char*>(&length), sizeof(size_t));
        str.resize(length);
        file.read(&str[0], length);
    }
    
    // MemoryStorage implementation
    void MemoryStorage::save_vector(const VectorEntry& entry) {
        vectors_[entry.id] = entry;
    }
    
    bool MemoryStorage::load_vector(VectorId id, VectorEntry& entry) {
        auto it = vectors_.find(id);
        if (it != vectors_.end()) {
            entry = it->second;
            return true;
        }
        return false;
    }
    
    void MemoryStorage::delete_vector(VectorId id) {
        vectors_.erase(id);
    }
    
    std::vector<VectorEntry> MemoryStorage::load_all_vectors() {
        std::vector<VectorEntry> result;
        result.reserve(vectors_.size());
        
        for (const auto& pair : vectors_) {
            result.push_back(pair.second);
        }
        
        return result;
    }
    
    void MemoryStorage::clear() {
        vectors_.clear();
        metadata_.clear();
    }
    
    size_t MemoryStorage::count() const {
        return vectors_.size();
    }
    
    void MemoryStorage::save_metadata(const std::string& key, const std::string& value) {
        metadata_[key] = value;
    }
    
    bool MemoryStorage::load_metadata(const std::string& key, std::string& value) {
        auto it = metadata_.find(key);
        if (it != metadata_.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
    
    // StorageManager implementation
    StorageManager::StorageManager(std::unique_ptr<Storage> storage) 
        : storage_(std::move(storage)) {}
    
    void StorageManager::save_vector(const VectorEntry& entry) {
        storage_->save_vector(entry);
    }
    
    bool StorageManager::load_vector(VectorId id, VectorEntry& entry) {
        return storage_->load_vector(id, entry);
    }
    
    void StorageManager::delete_vector(VectorId id) {
        storage_->delete_vector(id);
    }
    
    std::vector<VectorEntry> StorageManager::load_all_vectors() {
        return storage_->load_all_vectors();
    }
    
    void StorageManager::clear_vectors() {
        storage_->clear();
    }
    
    size_t StorageManager::vector_count() const {
        return storage_->count();
    }
    
    void StorageManager::save_config(const Config& config) {
        std::ostringstream oss;
        oss << config.dimension << "," << static_cast<int>(config.index_type) << ","
            << static_cast<int>(config.distance_metric) << "," << config.max_vectors;
        storage_->save_metadata("config", oss.str());
    }
    
    bool StorageManager::load_config(Config& config) {
        std::string config_str;
        if (!storage_->load_metadata("config", config_str)) {
            return false;
        }
        
        std::istringstream iss(config_str);
        std::string token;
        
        if (std::getline(iss, token, ',')) config.dimension = std::stoul(token);
        if (std::getline(iss, token, ',')) config.index_type = static_cast<IndexType>(std::stoi(token));
        if (std::getline(iss, token, ',')) config.distance_metric = static_cast<DistanceMetric>(std::stoi(token));
        if (std::getline(iss, token, ',')) config.max_vectors = std::stoul(token);
        
        return true;
    }
    
    void StorageManager::save_index_metadata(const std::string& index_type, const std::string& data) {
        storage_->save_metadata("index_" + index_type, data);
    }
    
    bool StorageManager::load_index_metadata(const std::string& index_type, std::string& data) {
        return storage_->load_metadata("index_" + index_type, data);
    }
    
    void StorageManager::save_vectors_batch(const std::vector<VectorEntry>& entries) {
        for (const auto& entry : entries) {
            storage_->save_vector(entry);
        }
    }
    
    std::vector<VectorEntry> StorageManager::load_vectors_batch(const std::vector<VectorId>& ids) {
        std::vector<VectorEntry> entries;
        entries.reserve(ids.size());
        
        for (VectorId id : ids) {
            VectorEntry entry;
            if (storage_->load_vector(id, entry)) {
                entries.push_back(entry);
            }
        }
        
        return entries;
    }
    
    void StorageManager::flush() {
        storage_->flush();
    }
    
    void StorageManager::close() {
        storage_->close();
    }
    
    void StorageManager::compact() {
        // For now, just flush
        flush();
    }
    
    // Factory functions
    std::unique_ptr<Storage> create_storage(StorageType type, const std::string& path) {
        switch (type) {
            case StorageType::FILE:
                return std::make_unique<FileStorage>(path.empty() ? "./vectordb_data" : path);
            case StorageType::MEMORY:
                return std::make_unique<MemoryStorage>();
            default:
                throw std::invalid_argument("Unknown storage type");
        }
    }
    
    std::unique_ptr<StorageManager> create_storage_manager(StorageType type, const std::string& path) {
        return std::make_unique<StorageManager>(create_storage(type, path));
    }
}