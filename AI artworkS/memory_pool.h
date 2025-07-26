#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Memory pool configuration
#define MAX_MEMORY_POOLS 8
#define MEMORY_ALIGNMENT 8
#define POOL_MAGIC_NUMBER 0xDEADBEEF

// Memory pool block header
typedef struct memory_block {
    struct memory_block* next;
    uint32_t magic;
    size_t size;
    bool is_free;
} memory_block_t;

// Memory pool structure
typedef struct memory_pool {
    void* pool_start;
    size_t pool_size;
    size_t block_size;
    size_t total_blocks;
    size_t free_blocks;
    memory_block_t* free_list;
    uint32_t magic;
    char name[16];
} memory_pool_t;

// Memory statistics
typedef struct memory_stats {
    size_t total_allocated;
    size_t total_freed;
    size_t peak_usage;
    size_t current_usage;
    size_t fragmentation_ratio;
    uint32_t allocation_count;
    uint32_t deallocation_count;
} memory_stats_t;

// Function prototypes
memory_pool_t* memory_pool_create(const char* name, void* buffer, size_t pool_size, size_t block_size);
void* memory_pool_alloc(memory_pool_t* pool);
bool memory_pool_free(memory_pool_t* pool, void* ptr);
void memory_pool_destroy(memory_pool_t* pool);
memory_stats_t memory_pool_get_stats(memory_pool_t* pool);
void memory_pool_defragment(memory_pool_t* pool);
bool memory_pool_validate(memory_pool_t* pool);

// Utility functions
size_t memory_align(size_t size, size_t alignment);
void memory_pool_print_stats(memory_pool_t* pool);

#ifdef __cplusplus
}
#endif

#endif // MEMORY_POOL_H