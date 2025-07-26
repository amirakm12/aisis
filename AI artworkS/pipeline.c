#include "pipeline.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Global pipeline registry
static data_pipeline_t* g_pipelines[8] = {0};
static uint8_t g_pipeline_count = 0;

// Utility function to get current time in microseconds
static uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

// Create a new pipeline
data_pipeline_t* pipeline_create(const char* name) {
    if (g_pipeline_count >= 8) {
        return NULL;
    }
    
    data_pipeline_t* pipeline = malloc(sizeof(data_pipeline_t));
    if (!pipeline) {
        return NULL;
    }
    
    memset(pipeline, 0, sizeof(data_pipeline_t));
    pipeline->magic = PIPELINE_MAGIC;
    pipeline->is_running = false;
    
    // Copy name
    strncpy(pipeline->name, name ? name : "unnamed", sizeof(pipeline->name) - 1);
    pipeline->name[sizeof(pipeline->name) - 1] = '\0';
    
    // Register pipeline
    g_pipelines[g_pipeline_count++] = pipeline;
    
    return pipeline;
}

// Add a processing stage to the pipeline
bool pipeline_add_stage(data_pipeline_t* pipeline, const char* stage_name, 
                       pipeline_stage_func_t func, void* context) {
    if (!pipeline || !func || pipeline->stage_count >= MAX_PIPELINE_STAGES) {
        return false;
    }
    
    pipeline_stage_t* stage = &pipeline->stages[pipeline->stage_count];
    stage->process_func = func;
    stage->context = context;
    stage->enabled = true;
    stage->processed_count = 0;
    stage->total_time_us = 0;
    
    strncpy(stage->name, stage_name ? stage_name : "unnamed", sizeof(stage->name) - 1);
    stage->name[sizeof(stage->name) - 1] = '\0';
    
    pipeline->stage_count++;
    return true;
}

// Remove a stage from the pipeline
bool pipeline_remove_stage(data_pipeline_t* pipeline, uint8_t stage_index) {
    if (!pipeline || stage_index >= pipeline->stage_count) {
        return false;
    }
    
    // Shift remaining stages
    for (uint8_t i = stage_index; i < pipeline->stage_count - 1; i++) {
        pipeline->stages[i] = pipeline->stages[i + 1];
    }
    
    pipeline->stage_count--;
    return true;
}

// Enable/disable a pipeline stage
bool pipeline_enable_stage(data_pipeline_t* pipeline, uint8_t stage_index, bool enable) {
    if (!pipeline || stage_index >= pipeline->stage_count) {
        return false;
    }
    
    pipeline->stages[stage_index].enabled = enable;
    return true;
}

// Create a pipeline buffer
pipeline_buffer_t* pipeline_buffer_create(data_pipeline_t* pipeline, size_t size, 
                                         pipeline_data_type_t type) {
    if (!pipeline || size == 0) {
        return NULL;
    }
    
    pipeline_buffer_t* buffer = malloc(sizeof(pipeline_buffer_t));
    if (!buffer) {
        return NULL;
    }
    
    // Try to allocate from memory pool first
    memory_pool_t* pool = pipeline_get_optimal_pool(pipeline, size);
    if (pool) {
        buffer->data = memory_pool_alloc(pool);
        buffer->pool = pool;
    } else {
        buffer->data = malloc(size);
        buffer->pool = NULL;
    }
    
    if (!buffer->data) {
        free(buffer);
        return NULL;
    }
    
    buffer->size = 0;
    buffer->capacity = size;
    buffer->type = type;
    buffer->ref_count = 1;
    
    return buffer;
}

// Retain a buffer (increase reference count)
void pipeline_buffer_retain(pipeline_buffer_t* buffer) {
    if (buffer) {
        buffer->ref_count++;
    }
}

// Release a buffer (decrease reference count and free if needed)
void pipeline_buffer_release(pipeline_buffer_t* buffer) {
    if (!buffer) {
        return;
    }
    
    buffer->ref_count--;
    if (buffer->ref_count == 0) {
        if (buffer->pool) {
            memory_pool_free(buffer->pool, buffer->data);
        } else {
            free(buffer->data);
        }
        free(buffer);
    }
}

// Resize a pipeline buffer
bool pipeline_buffer_resize(pipeline_buffer_t* buffer, size_t new_size) {
    if (!buffer || new_size == 0) {
        return false;
    }
    
    if (new_size <= buffer->capacity) {
        buffer->size = new_size;
        return true;
    }
    
    // Need to reallocate
    void* new_data;
    if (buffer->pool) {
        // Can't resize pool-allocated memory, fall back to malloc
        new_data = malloc(new_size);
        if (new_data) {
            memcpy(new_data, buffer->data, buffer->size);
            memory_pool_free(buffer->pool, buffer->data);
            buffer->pool = NULL;
        }
    } else {
        new_data = realloc(buffer->data, new_size);
    }
    
    if (!new_data) {
        return false;
    }
    
    buffer->data = new_data;
    buffer->capacity = new_size;
    buffer->size = new_size;
    
    return true;
}

// Add a memory pool to the pipeline
bool pipeline_add_memory_pool(data_pipeline_t* pipeline, memory_pool_t* pool) {
    if (!pipeline || !pool || pipeline->pool_count >= MAX_BUFFER_POOLS) {
        return false;
    }
    
    pipeline->buffer_pools[pipeline->pool_count++] = pool;
    return true;
}

// Get the optimal memory pool for a given size
memory_pool_t* pipeline_get_optimal_pool(data_pipeline_t* pipeline, size_t size) {
    if (!pipeline) {
        return NULL;
    }
    
    memory_pool_t* best_pool = NULL;
    size_t best_block_size = SIZE_MAX;
    
    for (uint8_t i = 0; i < pipeline->pool_count; i++) {
        memory_pool_t* pool = pipeline->buffer_pools[i];
        if (pool && pool->block_size >= size && pool->free_blocks > 0) {
            if (pool->block_size < best_block_size) {
                best_pool = pool;
                best_block_size = pool->block_size;
            }
        }
    }
    
    return best_pool;
}

// Process data through the pipeline
bool pipeline_process(data_pipeline_t* pipeline, pipeline_buffer_t* input, 
                     pipeline_buffer_t** output) {
    if (!pipeline || !input || !output || pipeline->magic != PIPELINE_MAGIC) {
        return false;
    }
    
    pipeline->is_running = true;
    uint64_t start_time = get_time_us();
    
    pipeline_buffer_t* current_input = input;
    pipeline_buffer_t* current_output = NULL;
    
    // Retain input buffer
    pipeline_buffer_retain(current_input);
    
    for (uint8_t i = 0; i < pipeline->stage_count; i++) {
        pipeline_stage_t* stage = &pipeline->stages[i];
        
        if (!stage->enabled) {
            continue;
        }
        
        // Create output buffer for this stage
        current_output = pipeline_buffer_create(pipeline, current_input->capacity, 
                                               current_input->type);
        if (!current_output) {
            pipeline_buffer_release(current_input);
            pipeline->stats.total_errors++;
            pipeline->is_running = false;
            return false;
        }
        
        // Process through this stage
        uint64_t stage_start = get_time_us();
        bool success = stage->process_func(current_input, current_output, stage->context);
        uint64_t stage_end = get_time_us();
        
        // Update stage statistics
        stage->total_time_us += (stage_end - stage_start);
        stage->processed_count++;
        
        if (!success) {
            pipeline_buffer_release(current_input);
            pipeline_buffer_release(current_output);
            pipeline->stats.total_errors++;
            pipeline->is_running = false;
            return false;
        }
        
        // Release previous input and use current output as next input
        pipeline_buffer_release(current_input);
        current_input = current_output;
    }
    
    // Update pipeline statistics
    uint64_t end_time = get_time_us();
    pipeline->stats.total_processed++;
    pipeline->stats.total_time_us += (end_time - start_time);
    pipeline->stats.avg_time_per_item_us = pipeline->stats.total_time_us / pipeline->stats.total_processed;
    
    *output = current_output;
    pipeline->is_running = false;
    
    return true;
}

// Process multiple items in batch
bool pipeline_process_batch(data_pipeline_t* pipeline, pipeline_buffer_t** inputs, 
                           pipeline_buffer_t** outputs, size_t count) {
    if (!pipeline || !inputs || !outputs || count == 0) {
        return false;
    }
    
    bool all_success = true;
    
    for (size_t i = 0; i < count; i++) {
        if (!pipeline_process(pipeline, inputs[i], &outputs[i])) {
            all_success = false;
        }
    }
    
    return all_success;
}

// Get pipeline statistics
pipeline_stats_t pipeline_get_stats(data_pipeline_t* pipeline) {
    pipeline_stats_t stats = {0};
    
    if (!pipeline) {
        return stats;
    }
    
    stats = pipeline->stats;
    
    // Calculate throughput
    if (stats.total_time_us > 0) {
        stats.throughput_items_per_sec = (float)stats.total_processed * 1000000.0f / stats.total_time_us;
    }
    
    // Calculate memory usage
    for (uint8_t i = 0; i < pipeline->pool_count; i++) {
        memory_pool_t* pool = pipeline->buffer_pools[i];
        if (pool) {
            memory_stats_t pool_stats = memory_pool_get_stats(pool);
            stats.memory_usage += pool_stats.current_usage;
        }
    }
    
    return stats;
}

// Reset pipeline statistics
void pipeline_reset_stats(data_pipeline_t* pipeline) {
    if (!pipeline) {
        return;
    }
    
    memset(&pipeline->stats, 0, sizeof(pipeline_stats_t));
    
    for (uint8_t i = 0; i < pipeline->stage_count; i++) {
        pipeline->stages[i].processed_count = 0;
        pipeline->stages[i].total_time_us = 0;
    }
}

// Print pipeline performance information
void pipeline_print_performance(data_pipeline_t* pipeline) {
    if (!pipeline) {
        return;
    }
    
    pipeline_stats_t stats = pipeline_get_stats(pipeline);
    
    printf("Pipeline '%s' Performance:\n", pipeline->name);
    printf("  Total processed: %u items\n", stats.total_processed);
    printf("  Total errors: %u\n", stats.total_errors);
    printf("  Average time per item: %lu us\n", stats.avg_time_per_item_us);
    printf("  Throughput: %.2f items/sec\n", stats.throughput_items_per_sec);
    printf("  Memory usage: %zu bytes\n", stats.memory_usage);
    
    printf("  Stage performance:\n");
    for (uint8_t i = 0; i < pipeline->stage_count; i++) {
        pipeline_stage_t* stage = &pipeline->stages[i];
        uint64_t avg_time = stage->processed_count > 0 ? 
                           stage->total_time_us / stage->processed_count : 0;
        printf("    %s: %u items, %lu us avg\n", 
               stage->name, stage->processed_count, avg_time);
    }
}

// Validate pipeline integrity
bool pipeline_validate_integrity(data_pipeline_t* pipeline) {
    if (!pipeline || pipeline->magic != PIPELINE_MAGIC) {
        return false;
    }
    
    // Validate memory pools
    for (uint8_t i = 0; i < pipeline->pool_count; i++) {
        if (!memory_pool_validate(pipeline->buffer_pools[i])) {
            return false;
        }
    }
    
    return true;
}

// Destroy pipeline
void pipeline_destroy(data_pipeline_t* pipeline) {
    if (!pipeline) {
        return;
    }
    
    // Remove from registry
    for (uint8_t i = 0; i < g_pipeline_count; i++) {
        if (g_pipelines[i] == pipeline) {
            for (uint8_t j = i; j < g_pipeline_count - 1; j++) {
                g_pipelines[j] = g_pipelines[j + 1];
            }
            g_pipeline_count--;
            break;
        }
    }
    
    pipeline->magic = 0;
    free(pipeline);
}

// Built-in pipeline stages
bool pipeline_stage_copy(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context) {
    (void)context; // Unused
    
    if (!input || !output || input->size == 0) {
        return false;
    }
    
    if (output->capacity < input->size) {
        if (!pipeline_buffer_resize(output, input->size)) {
            return false;
        }
    }
    
    memcpy(output->data, input->data, input->size);
    output->size = input->size;
    output->type = input->type;
    
    return true;
}

bool pipeline_stage_scale(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context) {
    float* scale_factor = (float*)context;
    
    if (!input || !output || !scale_factor || input->type != PIPELINE_DATA_FLOAT32) {
        return false;
    }
    
    if (output->capacity < input->size) {
        if (!pipeline_buffer_resize(output, input->size)) {
            return false;
        }
    }
    
    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;
    size_t count = input->size / sizeof(float);
    
    for (size_t i = 0; i < count; i++) {
        out_data[i] = in_data[i] * (*scale_factor);
    }
    
    output->size = input->size;
    output->type = input->type;
    
    return true;
}

bool pipeline_stage_filter(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context) {
    // Simple low-pass filter implementation
    (void)context; // Unused for now
    
    if (!input || !output || input->type != PIPELINE_DATA_FLOAT32 || input->size < sizeof(float) * 2) {
        return false;
    }
    
    if (output->capacity < input->size) {
        if (!pipeline_buffer_resize(output, input->size)) {
            return false;
        }
    }
    
    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;
    size_t count = input->size / sizeof(float);
    
    // Simple moving average filter
    out_data[0] = in_data[0];
    for (size_t i = 1; i < count; i++) {
        out_data[i] = (in_data[i] + in_data[i-1]) * 0.5f;
    }
    
    output->size = input->size;
    output->type = input->type;
    
    return true;
}

bool pipeline_stage_compress(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context) {
    // Simple run-length encoding for demonstration
    (void)context; // Unused
    
    if (!input || !output || input->type != PIPELINE_DATA_UINT8 || input->size == 0) {
        return false;
    }
    
    // Worst case: every byte is different, so we need 2x space
    size_t max_output_size = input->size * 2;
    if (output->capacity < max_output_size) {
        if (!pipeline_buffer_resize(output, max_output_size)) {
            return false;
        }
    }
    
    uint8_t* in_data = (uint8_t*)input->data;
    uint8_t* out_data = (uint8_t*)output->data;
    size_t out_pos = 0;
    
    for (size_t i = 0; i < input->size; ) {
        uint8_t current = in_data[i];
        uint8_t count = 1;
        
        // Count consecutive identical bytes
        while (i + count < input->size && in_data[i + count] == current && count < 255) {
            count++;
        }
        
        out_data[out_pos++] = count;
        out_data[out_pos++] = current;
        i += count;
    }
    
    output->size = out_pos;
    output->type = PIPELINE_DATA_UINT8;
    
    return true;
}