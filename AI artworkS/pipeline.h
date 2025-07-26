#ifndef PIPELINE_H
#define PIPELINE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "memory_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pipeline configuration
#define MAX_PIPELINE_STAGES 16
#define MAX_BUFFER_POOLS 4
#define PIPELINE_MAGIC 0xFEEDFACE

// Pipeline data types
typedef enum {
    PIPELINE_DATA_UINT8,
    PIPELINE_DATA_UINT16,
    PIPELINE_DATA_UINT32,
    PIPELINE_DATA_FLOAT32,
    PIPELINE_DATA_CUSTOM
} pipeline_data_type_t;

// Pipeline buffer
typedef struct pipeline_buffer {
    void* data;
    size_t size;
    size_t capacity;
    pipeline_data_type_t type;
    uint32_t ref_count;
    memory_pool_t* pool;
} pipeline_buffer_t;

// Pipeline stage function type
typedef bool (*pipeline_stage_func_t)(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context);

// Pipeline stage
typedef struct pipeline_stage {
    pipeline_stage_func_t process_func;
    void* context;
    char name[32];
    bool enabled;
    uint32_t processed_count;
    uint64_t total_time_us;
} pipeline_stage_t;

// Pipeline statistics
typedef struct pipeline_stats {
    uint32_t total_processed;
    uint32_t total_errors;
    uint64_t total_time_us;
    uint64_t avg_time_per_item_us;
    size_t memory_usage;
    float throughput_items_per_sec;
} pipeline_stats_t;

// Main pipeline structure
typedef struct data_pipeline {
    pipeline_stage_t stages[MAX_PIPELINE_STAGES];
    uint8_t stage_count;
    memory_pool_t* buffer_pools[MAX_BUFFER_POOLS];
    uint8_t pool_count;
    pipeline_stats_t stats;
    uint32_t magic;
    char name[32];
    bool is_running;
} data_pipeline_t;

// Pipeline creation and management
data_pipeline_t* pipeline_create(const char* name);
bool pipeline_add_stage(data_pipeline_t* pipeline, const char* stage_name, 
                       pipeline_stage_func_t func, void* context);
bool pipeline_remove_stage(data_pipeline_t* pipeline, uint8_t stage_index);
bool pipeline_enable_stage(data_pipeline_t* pipeline, uint8_t stage_index, bool enable);
void pipeline_destroy(data_pipeline_t* pipeline);

// Buffer management
pipeline_buffer_t* pipeline_buffer_create(data_pipeline_t* pipeline, size_t size, 
                                         pipeline_data_type_t type);
void pipeline_buffer_retain(pipeline_buffer_t* buffer);
void pipeline_buffer_release(pipeline_buffer_t* buffer);
bool pipeline_buffer_resize(pipeline_buffer_t* buffer, size_t new_size);

// Pipeline execution
bool pipeline_process(data_pipeline_t* pipeline, pipeline_buffer_t* input, 
                     pipeline_buffer_t** output);
bool pipeline_process_batch(data_pipeline_t* pipeline, pipeline_buffer_t** inputs, 
                           pipeline_buffer_t** outputs, size_t count);

// Memory pool integration
bool pipeline_add_memory_pool(data_pipeline_t* pipeline, memory_pool_t* pool);
memory_pool_t* pipeline_get_optimal_pool(data_pipeline_t* pipeline, size_t size);

// Performance monitoring
pipeline_stats_t pipeline_get_stats(data_pipeline_t* pipeline);
void pipeline_reset_stats(data_pipeline_t* pipeline);
void pipeline_print_performance(data_pipeline_t* pipeline);

// Optimization functions
void pipeline_optimize_memory_layout(data_pipeline_t* pipeline);
bool pipeline_validate_integrity(data_pipeline_t* pipeline);

// Built-in processing stages
bool pipeline_stage_copy(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context);
bool pipeline_stage_scale(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context);
bool pipeline_stage_filter(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context);
bool pipeline_stage_compress(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context);

#ifdef __cplusplus
}
#endif

#endif // PIPELINE_H