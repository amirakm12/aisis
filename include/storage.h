#pragma once

#include "types.h"
#include <fstream>
#include <memory>

namespace VectorDB {
    // Abstract base class for storage backends
    class Storage {
    public:
        virtual ~Storage() = default;
        
        virtual void save_vector(const VectorEntry& entry) = 0;
        virtual bool load_vector(VectorId id, VectorEntry& entry) = 0;
        virtual void delete_vector(VectorId id) = 0;
        virtual std::vector<VectorEntry> load_all_vectors() = 0;
        virtual void clear() = 0;
        virtual size_t count() const = 0;
        
        virtual void save_metadata(const std::string& key, const std::string& value) = 0;
        virtual bool load_metadata(const std::string& key, std::string& value) = 0;
        
        virtual void flush() = 0;
        virtual void close() = 0;
    };
    
    // File-based storage implementation
    class FileStorage : public Storage {
    private:
        std::string base_path_;
        std::string vectors_file_;
        std::string metadata_file_;
        std::string index_file_;
        
        mutable std::unordered_map<VectorId, size_t> vector_offsets_;
        mutable std::unordered_map<std::string, std::string> metadata_cache_;
        mutable bool offsets_loaded_;
        
        void ensure_directory_exists(const std::string& path);
        void load_offsets() const;
        void save_offsets();
        
        // Binary serialization helpers
        void write_vector(std::ofstream& file, const Vector& vec);
        void read_vector(std::ifstream& file, Vector& vec);
        void write_metadata(std::ofstream& file, const Metadata& meta);
        void read_metadata(std::ifstream& file, Metadata& meta);
        void write_string(std::ofstream& file, const std::string& str);
        void read_string(std::ifstream& file, std::string& str);
        
    public:
        explicit FileStorage(const std::string& base_path);
        ~FileStorage();
        
        void save_vector(const VectorEntry& entry) override;
        bool load_vector(VectorId id, VectorEntry& entry) override;
        void delete_vector(VectorId id) override;
        std::vector<VectorEntry> load_all_vectors() override;
        void clear() override;
        size_t count() const override;
        
        void save_metadata(const std::string& key, const std::string& value) override;
        bool load_metadata(const std::string& key, std::string& value) override;
        
        void flush() override;
        void close() override;
    };
    
    // Memory-based storage implementation (for testing/temporary use)
    class MemoryStorage : public Storage {
    private:
        std::unordered_map<VectorId, VectorEntry> vectors_;
        std::unordered_map<std::string, std::string> metadata_;
        
    public:
        MemoryStorage() = default;
        
        void save_vector(const VectorEntry& entry) override;
        bool load_vector(VectorId id, VectorEntry& entry) override;
        void delete_vector(VectorId id) override;
        std::vector<VectorEntry> load_all_vectors() override;
        void clear() override;
        size_t count() const override;
        
        void save_metadata(const std::string& key, const std::string& value) override;
        bool load_metadata(const std::string& key, std::string& value) override;
        
        void flush() override {}
        void close() override {}
    };
    
    // Storage manager for handling different storage backends
    class StorageManager {
    private:
        std::unique_ptr<Storage> storage_;
        
    public:
        explicit StorageManager(std::unique_ptr<Storage> storage);
        
        // Vector operations
        void save_vector(const VectorEntry& entry);
        bool load_vector(VectorId id, VectorEntry& entry);
        void delete_vector(VectorId id);
        std::vector<VectorEntry> load_all_vectors();
        void clear_vectors();
        size_t vector_count() const;
        
        // Metadata operations
        void save_config(const Config& config);
        bool load_config(Config& config);
        void save_index_metadata(const std::string& index_type, const std::string& data);
        bool load_index_metadata(const std::string& index_type, std::string& data);
        
        // Batch operations
        void save_vectors_batch(const std::vector<VectorEntry>& entries);
        std::vector<VectorEntry> load_vectors_batch(const std::vector<VectorId>& ids);
        
        // Maintenance
        void flush();
        void close();
        void compact(); // Remove deleted entries and optimize storage
    };
    
    // Factory functions
    enum class StorageType {
        FILE,
        MEMORY
    };
    
    std::unique_ptr<Storage> create_storage(StorageType type, const std::string& path = "");
    std::unique_ptr<StorageManager> create_storage_manager(StorageType type, const std::string& path = "");
}