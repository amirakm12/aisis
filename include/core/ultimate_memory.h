#ifndef ULTIMATE_MEMORY_H
#define ULTIMATE_MEMORY_H

/**
 * @file ultimate_memory.h
 * @brief ULTIMATE System Memory Management
 * @version 1.0.0
 * @date 2024
 * 
 * Memory management functionality for the ULTIMATE Windows system.
 * Includes heap management, memory pools, and safety features.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "ultimate_types.h"
#include "ultimate_errors.h"

/* Memory alignment */
#define ULTIMATE_MEMORY_ALIGNMENT           8
#define ULTIMATE_MEMORY_ALIGN(size)         (((size) + ULTIMATE_MEMORY_ALIGNMENT - 1) & ~(ULTIMATE_MEMORY_ALIGNMENT - 1))

/* Forward declarations - types defined in ultimate_types.h */

/* Memory pool configuration - use type from ultimate_types.h */

/* Memory statistics */
typedef struct {
    size_t total_heap_size;
    size_t used_heap_size;
    size_t free_heap_size;
    size_t largest_free_block;
    uint32_t allocation_count;
    uint32_t free_count;
    uint32_t fragmentation_percent;
    uint32_t peak_usage;
} ultimate_heap_stats_t;

typedef struct {
    size_t pool_size;
    size_t block_size;
    uint32_t total_blocks;
    uint32_t used_blocks;
    uint32_t free_blocks;
    uint32_t peak_usage;
    const char* name;
} ultimate_pool_stats_t;

/* Core memory management API */
ultimate_error_t ultimate_memory_init(void* heap_start, size_t heap_size);
ultimate_error_t ultimate_memory_deinit(void);

/* Dynamic memory allocation */
void* ultimate_malloc(size_t size);
void* ultimate_calloc(size_t count, size_t size);
void* ultimate_realloc(void* ptr, size_t new_size);
void ultimate_free(void* ptr);

/* Aligned memory allocation */
void* ultimate_aligned_malloc(size_t size, size_t alignment);
void ultimate_aligned_free(void* ptr);

/* Memory pool management */
ultimate_error_t ultimate_pool_create(const ultimate_pool_config_t* config, 
                                     ultimate_pool_handle_t* pool);
ultimate_error_t ultimate_pool_delete(ultimate_pool_handle_t pool);

/* Pool allocation/deallocation */
void* ultimate_pool_alloc(ultimate_pool_handle_t pool);
ultimate_error_t ultimate_pool_free(ultimate_pool_handle_t pool, void* ptr);

/* Memory utilities */
void* ultimate_memcpy(void* dest, const void* src, size_t count);
void* ultimate_memset(void* dest, int value, size_t count);
int ultimate_memcmp(const void* ptr1, const void* ptr2, size_t count);
void* ultimate_memmove(void* dest, const void* src, size_t count);

/* Memory safety and debugging */
bool ultimate_memory_is_valid_ptr(const void* ptr);
bool ultimate_memory_is_heap_ptr(const void* ptr);
bool ultimate_memory_is_pool_ptr(ultimate_pool_handle_t pool, const void* ptr);

/* Memory statistics and monitoring */
ultimate_error_t ultimate_memory_get_heap_stats(ultimate_heap_stats_t* stats);
ultimate_error_t ultimate_memory_get_pool_stats(ultimate_pool_handle_t pool, 
                                               ultimate_pool_stats_t* stats);

/* Memory corruption detection */
ultimate_error_t ultimate_memory_check_integrity(void);
ultimate_error_t ultimate_memory_check_pool_integrity(ultimate_pool_handle_t pool);

/* Stack monitoring */
typedef struct {
    void* stack_start;
    void* stack_end;
    size_t stack_size;
    size_t stack_used;
    size_t stack_free;
    size_t stack_peak;
    bool overflow_detected;
} ultimate_stack_info_t;

ultimate_error_t ultimate_stack_get_info(ultimate_stack_info_t* info);
bool ultimate_stack_check_overflow(void);
void ultimate_stack_monitor_enable(bool enable);

/* Memory protection */
typedef enum {
    ULTIMATE_MEM_PROT_NONE = 0,
    ULTIMATE_MEM_PROT_READ = 1,
    ULTIMATE_MEM_PROT_WRITE = 2,
    ULTIMATE_MEM_PROT_EXEC = 4,
    ULTIMATE_MEM_PROT_RW = 3,
    ULTIMATE_MEM_PROT_RX = 5,
    ULTIMATE_MEM_PROT_RWX = 7
} ultimate_memory_protection_t;

ultimate_error_t ultimate_memory_protect(void* addr, size_t size, 
                                        ultimate_memory_protection_t protection);

/* Windows-specific memory functions */
#ifdef _WIN32
ultimate_error_t ultimate_memory_virtual_alloc(size_t size, void** ptr);
ultimate_error_t ultimate_memory_virtual_free(void* ptr);
ultimate_error_t ultimate_memory_commit(void* ptr, size_t size);
ultimate_error_t ultimate_memory_decommit(void* ptr, size_t size);
#endif

/* Memory barriers */
void ultimate_memory_barrier(void);
void ultimate_memory_read_barrier(void);
void ultimate_memory_write_barrier(void);

/* Low-level memory operations */
void ultimate_memory_copy_fast(void* dest, const void* src, size_t count);
void ultimate_memory_set_fast(void* dest, int value, size_t count);

/* Memory leak detection */
#if ULTIMATE_DEBUG_ENABLED
typedef struct {
    void* ptr;
    size_t size;
    const char* file;
    uint32_t line;
    const char* function;
    uint32_t timestamp;
} ultimate_alloc_info_t;

void ultimate_memory_leak_check_enable(bool enable);
ultimate_error_t ultimate_memory_get_leaks(ultimate_alloc_info_t* leaks, 
                                          uint32_t max_leaks, 
                                          uint32_t* leak_count);
void ultimate_memory_dump_leaks(void);

/* Debug allocation macros */
#define ultimate_malloc_debug(size) \
    ultimate_malloc_debug_impl(size, __FILE__, __LINE__, __FUNCTION__)
#define ultimate_free_debug(ptr) \
    ultimate_free_debug_impl(ptr, __FILE__, __LINE__, __FUNCTION__)

void* ultimate_malloc_debug_impl(size_t size, const char* file, 
                                uint32_t line, const char* function);
void ultimate_free_debug_impl(void* ptr, const char* file, 
                             uint32_t line, const char* function);
#endif

/* Memory configuration */
typedef struct {
    size_t heap_size;
    size_t alignment;
    bool enable_corruption_check;
    bool enable_leak_detection;
    bool enable_stack_monitoring;
    uint32_t check_interval_ms;
} ultimate_memory_config_t;

ultimate_error_t ultimate_memory_configure(const ultimate_memory_config_t* config);

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_MEMORY_H */ 