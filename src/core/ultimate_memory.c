#include "ultimate_memory.h"
#include "ultimate_errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

/* Memory configuration */
static ultimate_memory_config_t g_memory_config = {
    .heap_size = 256 * 1024 * 1024,  /* 256MB default */
    .alignment = 16,
    .enable_corruption_check = true,
    .enable_leak_detection = false,
    .enable_stack_monitoring = true,
    .check_interval_ms = 1000
};

/* Memory statistics */
static ultimate_memory_stats_t g_memory_stats = {0};

/* Memory pools */
static ultimate_pool_handle_t g_memory_pools[ULTIMATE_MAX_POOLS];
static uint32_t g_pool_count = 0;

/* Leak detection */
#if ULTIMATE_DEBUG_ENABLED
static ultimate_alloc_info_t g_alloc_table[1024];
static uint32_t g_alloc_count = 0;
static bool g_leak_detection_enabled = false;
#endif

/* Memory initialization */
ultimate_error_t ultimate_memory_init(void) {
    memset(&g_memory_stats, 0, sizeof(g_memory_stats));
    g_memory_stats.total_size = g_memory_config.heap_size;
    g_memory_stats.free_size = g_memory_config.heap_size;
    
#if ULTIMATE_DEBUG_ENABLED
    memset(g_alloc_table, 0, sizeof(g_alloc_table));
    g_alloc_count = 0;
#endif
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_memory_shutdown(void) {
#if ULTIMATE_DEBUG_ENABLED
    if (g_leak_detection_enabled && g_alloc_count > 0) {
        ultimate_memory_dump_leaks();
    }
#endif
    
    /* Cleanup all pools */
    for (uint32_t i = 0; i < g_pool_count; i++) {
        ultimate_pool_destroy(g_memory_pools[i]);
    }
    g_pool_count = 0;
    
    return ULTIMATE_OK;
}

/* Basic memory allocation */
void* ultimate_malloc(size_t size) {
    if (size == 0) return NULL;
    
    void* ptr = malloc(size);
    if (ptr) {
        g_memory_stats.allocated_size += size;
        g_memory_stats.free_size -= size;
        g_memory_stats.allocation_count++;
    }
    
    return ptr;
}

void* ultimate_calloc(size_t count, size_t size) {
    size_t total_size = count * size;
    void* ptr = ultimate_malloc(total_size);
    if (ptr) {
        memset(ptr, 0, total_size);
    }
    return ptr;
}

void* ultimate_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (new_ptr && ptr != new_ptr) {
        /* Update statistics would require tracking original size */
        g_memory_stats.allocation_count++;
    }
    return new_ptr;
}

void ultimate_free(void* ptr) {
    if (ptr) {
        free(ptr);
        g_memory_stats.free_count++;
    }
}

/* Aligned memory allocation */
void* ultimate_aligned_malloc(size_t size, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return NULL;  /* Alignment must be power of 2 */
    }
    
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    return NULL;
#endif
}

void ultimate_aligned_free(void* ptr) {
    if (ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

/* Memory pool management */
ultimate_error_t ultimate_pool_create(const ultimate_pool_config_t* config, 
                                     ultimate_pool_handle_t* handle) {
    if (!config || !handle || g_pool_count >= ULTIMATE_MAX_POOLS) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_pool_t* pool = (ultimate_pool_t*)ultimate_malloc(sizeof(ultimate_pool_t));
    if (!pool) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    memset(pool, 0, sizeof(ultimate_pool_t));
    pool->config = *config;
    pool->memory = ultimate_malloc(config->total_size);
    
    if (!pool->memory) {
        ultimate_free(pool);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Initialize free blocks */
    pool->free_blocks = (void**)ultimate_malloc(config->max_blocks * sizeof(void*));
    if (!pool->free_blocks) {
        ultimate_free(pool->memory);
        ultimate_free(pool);
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Setup free block list */
    uint8_t* block_ptr = (uint8_t*)pool->memory;
    for (uint32_t i = 0; i < config->max_blocks; i++) {
        pool->free_blocks[i] = block_ptr;
        block_ptr += config->block_size;
    }
    pool->free_count = config->max_blocks;
    
    *handle = (ultimate_pool_handle_t)pool;
    g_memory_pools[g_pool_count++] = *handle;
    
    return ULTIMATE_OK;
}

void* ultimate_pool_alloc(ultimate_pool_handle_t handle) {
    ultimate_pool_t* pool = (ultimate_pool_t*)handle;
    if (!pool || pool->free_count == 0) {
        return NULL;
    }
    
    void* block = pool->free_blocks[--pool->free_count];
    pool->used_count++;
    
    return block;
}

ultimate_error_t ultimate_pool_free(ultimate_pool_handle_t handle, void* ptr) {
    ultimate_pool_t* pool = (ultimate_pool_t*)handle;
    if (!pool || !ptr) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    /* Add block back to free list */
    if (pool->free_count < pool->config.max_blocks) {
        pool->free_blocks[pool->free_count++] = ptr;
        pool->used_count--;
        return ULTIMATE_OK;
    }
    
    return ULTIMATE_ERROR_POOL_FULL;
}

ultimate_error_t ultimate_pool_destroy(ultimate_pool_handle_t handle) {
    ultimate_pool_t* pool = (ultimate_pool_t*)handle;
    if (!pool) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_free(pool->free_blocks);
    ultimate_free(pool->memory);
    ultimate_free(pool);
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_pool_get_stats(ultimate_pool_handle_t handle, 
                                        ultimate_pool_stats_t* stats) {
    ultimate_pool_t* pool = (ultimate_pool_t*)handle;
    if (!pool || !stats) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    stats->total_blocks = pool->config.max_blocks;
    stats->used_blocks = pool->used_count;
    stats->free_blocks = pool->free_count;
    stats->block_size = pool->config.block_size;
    stats->total_size = pool->config.total_size;
    
    return ULTIMATE_OK;
}

/* Memory statistics */
ultimate_error_t ultimate_memory_get_stats(ultimate_memory_stats_t* stats) {
    if (!stats) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    *stats = g_memory_stats;
    return ULTIMATE_OK;
}

/* Memory barriers */
void ultimate_memory_barrier(void) {
#ifdef _WIN32
    MemoryBarrier();
#else
    __sync_synchronize();
#endif
}

void ultimate_memory_read_barrier(void) {
#ifdef _WIN32
    MemoryBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

void ultimate_memory_write_barrier(void) {
#ifdef _WIN32
    MemoryBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

/* Fast memory operations */
void ultimate_memory_copy_fast(void* dest, const void* src, size_t count) {
    memcpy(dest, src, count);
}

void ultimate_memory_set_fast(void* dest, int value, size_t count) {
    memset(dest, value, count);
}

/* Memory protection */
ultimate_error_t ultimate_memory_protect(void* addr, size_t size, 
                                        ultimate_memory_protection_t protection) {
#ifdef _WIN32
    DWORD win_protection;
    switch (protection) {
        case ULTIMATE_MEMORY_PROTECTION_READ:
            win_protection = PAGE_READONLY;
            break;
        case ULTIMATE_MEMORY_PROTECTION_WRITE:
            win_protection = PAGE_READWRITE;
            break;
        case ULTIMATE_MEMORY_PROTECTION_EXECUTE:
            win_protection = PAGE_EXECUTE_READ;
            break;
        case ULTIMATE_MEMORY_PROTECTION_NONE:
            win_protection = PAGE_NOACCESS;
            break;
        default:
            return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    DWORD old_protection;
    if (!VirtualProtect(addr, size, win_protection, &old_protection)) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#else
    int prot = 0;
    if (protection & ULTIMATE_MEMORY_PROTECTION_READ) prot |= PROT_READ;
    if (protection & ULTIMATE_MEMORY_PROTECTION_WRITE) prot |= PROT_WRITE;
    if (protection & ULTIMATE_MEMORY_PROTECTION_EXECUTE) prot |= PROT_EXEC;
    
    if (mprotect(addr, size, prot) != 0) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#endif
    
    return ULTIMATE_OK;
}

#ifdef _WIN32
/* Windows-specific memory functions */
ultimate_error_t ultimate_memory_virtual_alloc(size_t size, void** ptr) {
    if (!ptr) return ULTIMATE_ERROR_INVALID_PARAMETER;
    
    *ptr = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    return *ptr ? ULTIMATE_OK : ULTIMATE_ERROR_OUT_OF_MEMORY;
}

ultimate_error_t ultimate_memory_virtual_free(void* ptr) {
    if (!ptr) return ULTIMATE_ERROR_INVALID_PARAMETER;
    
    return VirtualFree(ptr, 0, MEM_RELEASE) ? ULTIMATE_OK : ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
}

ultimate_error_t ultimate_memory_commit(void* ptr, size_t size) {
    if (!ptr) return ULTIMATE_ERROR_INVALID_PARAMETER;
    
    return VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) ? 
           ULTIMATE_OK : ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
}

ultimate_error_t ultimate_memory_decommit(void* ptr, size_t size) {
    if (!ptr) return ULTIMATE_ERROR_INVALID_PARAMETER;
    
    return VirtualFree(ptr, size, MEM_DECOMMIT) ? 
           ULTIMATE_OK : ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
}
#endif

/* Memory configuration */
ultimate_error_t ultimate_memory_configure(const ultimate_memory_config_t* config) {
    if (!config) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    g_memory_config = *config;
    return ULTIMATE_OK;
}

#if ULTIMATE_DEBUG_ENABLED
/* Debug allocation functions */
void* ultimate_malloc_debug_impl(size_t size, const char* file, 
                                uint32_t line, const char* function) {
    void* ptr = ultimate_malloc(size);
    
    if (ptr && g_leak_detection_enabled && g_alloc_count < 1024) {
        ultimate_alloc_info_t* info = &g_alloc_table[g_alloc_count++];
        info->ptr = ptr;
        info->size = size;
        info->file = file;
        info->line = line;
        info->function = function;
        info->timestamp = ultimate_get_tick_count();
    }
    
    return ptr;
}

void ultimate_free_debug_impl(void* ptr, const char* file, 
                             uint32_t line, const char* function) {
    if (!ptr) return;
    
    /* Remove from allocation table */
    if (g_leak_detection_enabled) {
        for (uint32_t i = 0; i < g_alloc_count; i++) {
            if (g_alloc_table[i].ptr == ptr) {
                /* Move last entry to this position */
                g_alloc_table[i] = g_alloc_table[--g_alloc_count];
                break;
            }
        }
    }
    
    ultimate_free(ptr);
}

void ultimate_memory_leak_check_enable(bool enable) {
    g_leak_detection_enabled = enable;
}

ultimate_error_t ultimate_memory_get_leaks(ultimate_alloc_info_t* leaks, 
                                          uint32_t max_leaks, 
                                          uint32_t* leak_count) {
    if (!leaks || !leak_count) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    uint32_t count = g_alloc_count < max_leaks ? g_alloc_count : max_leaks;
    memcpy(leaks, g_alloc_table, count * sizeof(ultimate_alloc_info_t));
    *leak_count = count;
    
    return ULTIMATE_OK;
}

void ultimate_memory_dump_leaks(void) {
    printf("Memory Leak Report:\n");
    printf("==================\n");
    
    for (uint32_t i = 0; i < g_alloc_count; i++) {
        ultimate_alloc_info_t* info = &g_alloc_table[i];
        printf("Leak %u: %zu bytes at %p\n", i + 1, info->size, info->ptr);
        printf("  File: %s:%u\n", info->file ? info->file : "unknown", info->line);
        printf("  Function: %s\n", info->function ? info->function : "unknown");
        printf("  Timestamp: %u\n", info->timestamp);
        printf("\n");
    }
    
    printf("Total leaks: %u\n", g_alloc_count);
}
#endif