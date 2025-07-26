#include "memory_pool.h"
#include <string.h>
#include <stdio.h>

// Global memory pools registry
static memory_pool_t* g_memory_pools[MAX_MEMORY_POOLS] = {0};
static uint8_t g_pool_count = 0;

// Helper function to align memory
size_t memory_align(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Create a memory pool
memory_pool_t* memory_pool_create(const char* name, void* buffer, size_t pool_size, size_t block_size) {
    if (!buffer || pool_size == 0 || block_size == 0 || g_pool_count >= MAX_MEMORY_POOLS) {
        return NULL;
    }
    
    // Align block size
    block_size = memory_align(block_size + sizeof(memory_block_t), MEMORY_ALIGNMENT);
    
    // Calculate number of blocks
    size_t total_blocks = pool_size / block_size;
    if (total_blocks == 0) {
        return NULL;
    }
    
    // Allocate pool structure
    memory_pool_t* pool = (memory_pool_t*)((uint8_t*)buffer + pool_size - sizeof(memory_pool_t));
    pool->pool_start = buffer;
    pool->pool_size = pool_size - sizeof(memory_pool_t);
    pool->block_size = block_size;
    pool->total_blocks = total_blocks;
    pool->free_blocks = total_blocks;
    pool->magic = POOL_MAGIC_NUMBER;
    
    // Copy name
    strncpy(pool->name, name ? name : "unnamed", sizeof(pool->name) - 1);
    pool->name[sizeof(pool->name) - 1] = '\0';
    
    // Initialize free list
    uint8_t* current_ptr = (uint8_t*)buffer;
    memory_block_t* prev_block = NULL;
    
    for (size_t i = 0; i < total_blocks; i++) {
        memory_block_t* block = (memory_block_t*)current_ptr;
        block->magic = POOL_MAGIC_NUMBER;
        block->size = block_size - sizeof(memory_block_t);
        block->is_free = true;
        block->next = NULL;
        
        if (prev_block) {
            prev_block->next = block;
        } else {
            pool->free_list = block;
        }
        
        prev_block = block;
        current_ptr += block_size;
    }
    
    // Register pool
    g_memory_pools[g_pool_count++] = pool;
    
    return pool;
}

// Allocate memory from pool
void* memory_pool_alloc(memory_pool_t* pool) {
    if (!pool || pool->magic != POOL_MAGIC_NUMBER || !pool->free_list) {
        return NULL;
    }
    
    // Get first free block
    memory_block_t* block = pool->free_list;
    pool->free_list = block->next;
    
    block->is_free = false;
    block->next = NULL;
    pool->free_blocks--;
    
    // Return pointer after block header
    return (uint8_t*)block + sizeof(memory_block_t);
}

// Free memory back to pool
bool memory_pool_free(memory_pool_t* pool, void* ptr) {
    if (!pool || !ptr || pool->magic != POOL_MAGIC_NUMBER) {
        return false;
    }
    
    // Get block header
    memory_block_t* block = (memory_block_t*)((uint8_t*)ptr - sizeof(memory_block_t));
    
    // Validate block
    if (block->magic != POOL_MAGIC_NUMBER || block->is_free) {
        return false;
    }
    
    // Validate pointer is within pool bounds
    if ((uint8_t*)block < (uint8_t*)pool->pool_start || 
        (uint8_t*)block >= (uint8_t*)pool->pool_start + pool->pool_size) {
        return false;
    }
    
    // Mark as free and add to free list
    block->is_free = true;
    block->next = pool->free_list;
    pool->free_list = block;
    pool->free_blocks++;
    
    return true;
}

// Get memory pool statistics
memory_stats_t memory_pool_get_stats(memory_pool_t* pool) {
    memory_stats_t stats = {0};
    
    if (!pool || pool->magic != POOL_MAGIC_NUMBER) {
        return stats;
    }
    
    stats.total_allocated = (pool->total_blocks - pool->free_blocks) * pool->block_size;
    stats.current_usage = stats.total_allocated;
    stats.peak_usage = stats.total_allocated; // Simplified for this implementation
    stats.fragmentation_ratio = pool->free_blocks > 0 ? 
        (pool->free_blocks * 100) / pool->total_blocks : 0;
    
    return stats;
}

// Validate memory pool integrity
bool memory_pool_validate(memory_pool_t* pool) {
    if (!pool || pool->magic != POOL_MAGIC_NUMBER) {
        return false;
    }
    
    size_t free_count = 0;
    memory_block_t* current = pool->free_list;
    
    while (current) {
        if (current->magic != POOL_MAGIC_NUMBER || !current->is_free) {
            return false;
        }
        free_count++;
        current = current->next;
    }
    
    return free_count == pool->free_blocks;
}

// Defragment memory pool (simplified coalescing)
void memory_pool_defragment(memory_pool_t* pool) {
    if (!pool || pool->magic != POOL_MAGIC_NUMBER) {
        return;
    }
    
    // For fixed-size block pools, defragmentation is mainly
    // about reorganizing the free list for better cache locality
    // This is a simplified implementation
}

// Print memory pool statistics
void memory_pool_print_stats(memory_pool_t* pool) {
    if (!pool || pool->magic != POOL_MAGIC_NUMBER) {
        printf("Invalid memory pool\n");
        return;
    }
    
    memory_stats_t stats = memory_pool_get_stats(pool);
    
    printf("Memory Pool '%s' Statistics:\n", pool->name);
    printf("  Total blocks: %zu\n", pool->total_blocks);
    printf("  Free blocks: %zu\n", pool->free_blocks);
    printf("  Block size: %zu bytes\n", pool->block_size);
    printf("  Current usage: %zu bytes\n", stats.current_usage);
    printf("  Free space: %zu%%\n", stats.fragmentation_ratio);
    printf("  Pool utilization: %.2f%%\n", 
           ((float)(pool->total_blocks - pool->free_blocks) / pool->total_blocks) * 100.0f);
}

// Destroy memory pool
void memory_pool_destroy(memory_pool_t* pool) {
    if (!pool || pool->magic != POOL_MAGIC_NUMBER) {
        return;
    }
    
    // Remove from registry
    for (uint8_t i = 0; i < g_pool_count; i++) {
        if (g_memory_pools[i] == pool) {
            // Shift remaining pools
            for (uint8_t j = i; j < g_pool_count - 1; j++) {
                g_memory_pools[j] = g_memory_pools[j + 1];
            }
            g_pool_count--;
            break;
        }
    }
    
    // Clear magic number
    pool->magic = 0;
}