#include "ultimate_memory.h"
#include "ultimate_errors.h"
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <atomic>

namespace ultimate {

struct MemoryBlock {
    void* ptr;
    size_t size;
    const char* file;
    int line;
    std::chrono::steady_clock::time_point allocated_time;
};

class MemoryManager {
private:
    static std::unique_ptr<MemoryManager> instance_;
    static std::mutex instance_mutex_;
    
    std::mutex memory_mutex_;
    std::unordered_map<void*, MemoryBlock> allocated_blocks_;
    std::atomic<size_t> total_allocated_;
    std::atomic<size_t> peak_allocated_;
    std::atomic<size_t> allocation_count_;
    size_t heap_size_;
    bool initialized_;
    
    // Memory pools
    struct MemoryPool {
        std::vector<void*> free_blocks;
        size_t block_size;
        size_t total_blocks;
        std::mutex pool_mutex;
    };
    
    std::unordered_map<size_t, std::unique_ptr<MemoryPool>> memory_pools_;
    
    MemoryManager() : total_allocated_(0), peak_allocated_(0), allocation_count_(0), 
                     heap_size_(0), initialized_(false) {}

public:
    static MemoryManager* getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::unique_ptr<MemoryManager>(new MemoryManager());
        }
        return instance_.get();
    }
    
    ultimate_error_t initialize(size_t heap_size) {
        if (initialized_) {
            return ULTIMATE_ERROR_ALREADY_INITIALIZED;
        }
        
        heap_size_ = heap_size;
        initialized_ = true;
        
        // Initialize common memory pools
        createPool(16, 1000);   // Small objects
        createPool(64, 500);    // Medium objects
        createPool(256, 200);   // Large objects
        createPool(1024, 50);   // Extra large objects
        
        return ULTIMATE_OK;
    }
    
    void* allocate(size_t size, const char* file = nullptr, int line = 0) {
        if (!initialized_ || size == 0) {
            return nullptr;
        }
        
        // Try to get from pool first
        void* ptr = allocateFromPool(size);
        if (!ptr) {
            ptr = std::malloc(size);
        }
        
        if (ptr) {
            std::lock_guard<std::mutex> lock(memory_mutex_);
            
            MemoryBlock block;
            block.ptr = ptr;
            block.size = size;
            block.file = file;
            block.line = line;
            block.allocated_time = std::chrono::steady_clock::now();
            
            allocated_blocks_[ptr] = block;
            
            size_t current_total = total_allocated_.fetch_add(size) + size;
            size_t current_peak = peak_allocated_.load();
            while (current_total > current_peak && 
                   !peak_allocated_.compare_exchange_weak(current_peak, current_total)) {
                current_peak = peak_allocated_.load();
            }
            
            allocation_count_.fetch_add(1);
        }
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr || !initialized_) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(memory_mutex_);
        
        auto it = allocated_blocks_.find(ptr);
        if (it != allocated_blocks_.end()) {
            size_t size = it->second.size;
            allocated_blocks_.erase(it);
            total_allocated_.fetch_sub(size);
            
            // Try to return to pool
            if (!returnToPool(ptr, size)) {
                std::free(ptr);
            }
        }
    }
    
    void* reallocate(void* ptr, size_t new_size, const char* file = nullptr, int line = 0) {
        if (!ptr) {
            return allocate(new_size, file, line);
        }
        
        if (new_size == 0) {
            deallocate(ptr);
            return nullptr;
        }
        
        std::lock_guard<std::mutex> lock(memory_mutex_);
        
        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            return nullptr; // Invalid pointer
        }
        
        size_t old_size = it->second.size;
        void* new_ptr = std::realloc(ptr, new_size);
        
        if (new_ptr) {
            // Update tracking
            allocated_blocks_.erase(it);
            
            MemoryBlock block;
            block.ptr = new_ptr;
            block.size = new_size;
            block.file = file;
            block.line = line;
            block.allocated_time = std::chrono::steady_clock::now();
            
            allocated_blocks_[new_ptr] = block;
            
            total_allocated_.fetch_add(new_size - old_size);
            
            size_t current_total = total_allocated_.load();
            size_t current_peak = peak_allocated_.load();
            while (current_total > current_peak && 
                   !peak_allocated_.compare_exchange_weak(current_peak, current_total)) {
                current_peak = peak_allocated_.load();
            }
        }
        
        return new_ptr;
    }
    
    ultimate_memory_stats_t getStats() {
        ultimate_memory_stats_t stats = {};
        
        stats.total_allocated = total_allocated_.load();
        stats.peak_allocated = peak_allocated_.load();
        stats.allocation_count = allocation_count_.load();
        stats.heap_size = heap_size_;
        
        std::lock_guard<std::mutex> lock(memory_mutex_);
        stats.active_allocations = allocated_blocks_.size();
        
        return stats;
    }
    
    std::vector<ultimate_memory_leak_t> getLeaks() {
        std::vector<ultimate_memory_leak_t> leaks;
        
        std::lock_guard<std::mutex> lock(memory_mutex_);
        
        auto now = std::chrono::steady_clock::now();
        
        for (const auto& pair : allocated_blocks_) {
            const MemoryBlock& block = pair.second;
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - block.allocated_time);
            
            // Consider blocks older than 1 minute as potential leaks
            if (duration.count() > 60000) {
                ultimate_memory_leak_t leak = {};
                leak.ptr = block.ptr;
                leak.size = block.size;
                leak.file = block.file;
                leak.line = block.line;
                leak.age_ms = duration.count();
                
                leaks.push_back(leak);
            }
        }
        
        return leaks;
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        
        // Free all remaining allocations
        for (const auto& pair : allocated_blocks_) {
            std::free(pair.first);
        }
        
        allocated_blocks_.clear();
        total_allocated_.store(0);
        allocation_count_.store(0);
        
        // Cleanup memory pools
        for (auto& pool_pair : memory_pools_) {
            auto& pool = pool_pair.second;
            std::lock_guard<std::mutex> pool_lock(pool->pool_mutex);
            
            for (void* block : pool->free_blocks) {
                std::free(block);
            }
            pool->free_blocks.clear();
        }
        
        memory_pools_.clear();
        initialized_ = false;
    }

private:
    void createPool(size_t block_size, size_t initial_blocks) {
        auto pool = std::make_unique<MemoryPool>();
        pool->block_size = block_size;
        pool->total_blocks = initial_blocks;
        
        // Pre-allocate blocks
        for (size_t i = 0; i < initial_blocks; ++i) {
            void* block = std::malloc(block_size);
            if (block) {
                pool->free_blocks.push_back(block);
            }
        }
        
        memory_pools_[block_size] = std::move(pool);
    }
    
    void* allocateFromPool(size_t size) {
        // Find the best fitting pool
        size_t best_size = SIZE_MAX;
        MemoryPool* best_pool = nullptr;
        
        for (auto& pool_pair : memory_pools_) {
            if (pool_pair.first >= size && pool_pair.first < best_size) {
                best_size = pool_pair.first;
                best_pool = pool_pair.second.get();
            }
        }
        
        if (best_pool) {
            std::lock_guard<std::mutex> lock(best_pool->pool_mutex);
            if (!best_pool->free_blocks.empty()) {
                void* ptr = best_pool->free_blocks.back();
                best_pool->free_blocks.pop_back();
                return ptr;
            }
        }
        
        return nullptr;
    }
    
    bool returnToPool(void* ptr, size_t size) {
        auto it = memory_pools_.find(size);
        if (it != memory_pools_.end()) {
            std::lock_guard<std::mutex> lock(it->second->pool_mutex);
            it->second->free_blocks.push_back(ptr);
            return true;
        }
        
        return false;
    }
};

std::unique_ptr<MemoryManager> MemoryManager::instance_ = nullptr;
std::mutex MemoryManager::instance_mutex_;

} // namespace ultimate

// C API Implementation
extern "C" {

ultimate_error_t ultimate_memory_init(size_t heap_size) {
    return ultimate::MemoryManager::getInstance()->initialize(heap_size);
}

void* ultimate_malloc(size_t size) {
    return ultimate::MemoryManager::getInstance()->allocate(size);
}

void* ultimate_malloc_debug(size_t size, const char* file, int line) {
    return ultimate::MemoryManager::getInstance()->allocate(size, file, line);
}

void ultimate_free(void* ptr) {
    ultimate::MemoryManager::getInstance()->deallocate(ptr);
}

void* ultimate_realloc(void* ptr, size_t size) {
    return ultimate::MemoryManager::getInstance()->reallocate(ptr, size);
}

void* ultimate_realloc_debug(void* ptr, size_t size, const char* file, int line) {
    return ultimate::MemoryManager::getInstance()->reallocate(ptr, size, file, line);
}

void* ultimate_calloc(size_t num, size_t size) {
    size_t total_size = num * size;
    void* ptr = ultimate::MemoryManager::getInstance()->allocate(total_size);
    if (ptr) {
        std::memset(ptr, 0, total_size);
    }
    return ptr;
}

ultimate_memory_stats_t ultimate_memory_get_stats(void) {
    return ultimate::MemoryManager::getInstance()->getStats();
}

size_t ultimate_memory_check_leaks(ultimate_memory_leak_t* leaks, size_t max_leaks) {
    auto leak_list = ultimate::MemoryManager::getInstance()->getLeaks();
    
    size_t count = std::min(leak_list.size(), max_leaks);
    
    if (leaks && count > 0) {
        std::copy(leak_list.begin(), leak_list.begin() + count, leaks);
    }
    
    return leak_list.size();
}

void ultimate_memory_deinit(void) {
    ultimate::MemoryManager::getInstance()->cleanup();
}

} // extern "C"