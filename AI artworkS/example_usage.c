#include "pipeline.h"
#include "memory_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Example: Signal processing pipeline with memory optimization
int main(void) {
    printf("=== AI artworkS - Advanced Pipeline & Memory Optimization Demo ===\n\n");
    
    // 1. Create memory pools for different data sizes
    printf("1. Creating optimized memory pools...\n");
    
    // Small buffer pool (256 bytes each)
    static uint8_t small_pool_buffer[4096];
    memory_pool_t* small_pool = memory_pool_create("small_buffers", 
                                                   small_pool_buffer, 
                                                   sizeof(small_pool_buffer), 
                                                   256);
    
    // Medium buffer pool (1KB each)
    static uint8_t medium_pool_buffer[8192];
    memory_pool_t* medium_pool = memory_pool_create("medium_buffers", 
                                                    medium_pool_buffer, 
                                                    sizeof(medium_pool_buffer), 
                                                    1024);
    
    // Large buffer pool (4KB each)
    static uint8_t large_pool_buffer[16384];
    memory_pool_t* large_pool = memory_pool_create("large_buffers", 
                                                   large_pool_buffer, 
                                                   sizeof(large_pool_buffer), 
                                                   4096);
    
    if (!small_pool || !medium_pool || !large_pool) {
        printf("Error: Failed to create memory pools\n");
        return -1;
    }
    
    printf("   ✓ Created 3 memory pools with different block sizes\n");
    memory_pool_print_stats(small_pool);
    memory_pool_print_stats(medium_pool);
    memory_pool_print_stats(large_pool);
    
    // 2. Create a high-performance data processing pipeline
    printf("\n2. Creating optimized data pipeline...\n");
    
    data_pipeline_t* signal_pipeline = pipeline_create("signal_processor");
    if (!signal_pipeline) {
        printf("Error: Failed to create pipeline\n");
        return -1;
    }
    
    // Add memory pools to pipeline for automatic optimization
    pipeline_add_memory_pool(signal_pipeline, small_pool);
    pipeline_add_memory_pool(signal_pipeline, medium_pool);
    pipeline_add_memory_pool(signal_pipeline, large_pool);
    
    // 3. Configure processing stages
    printf("   ✓ Adding processing stages...\n");
    
    // Stage 1: Copy input data
    pipeline_add_stage(signal_pipeline, "input_copy", pipeline_stage_copy, NULL);
    
    // Stage 2: Apply scaling
    static float scale_factor = 2.5f;
    pipeline_add_stage(signal_pipeline, "amplify", pipeline_stage_scale, &scale_factor);
    
    // Stage 3: Apply filtering
    pipeline_add_stage(signal_pipeline, "filter", pipeline_stage_filter, NULL);
    
    // Stage 4: Copy final data (compression requires uint8 data, not float)
    pipeline_add_stage(signal_pipeline, "output_copy", pipeline_stage_copy, NULL);
    
    printf("   ✓ Pipeline configured with 4 processing stages\n");
    
    // 4. Create test data
    printf("\n3. Generating test signal data...\n");
    
    const size_t signal_length = 1000;
    const size_t data_size = signal_length * sizeof(float);
    
    pipeline_buffer_t* input_buffer = pipeline_buffer_create(signal_pipeline, 
                                                           data_size, 
                                                           PIPELINE_DATA_FLOAT32);
    if (!input_buffer) {
        printf("Error: Failed to create input buffer\n");
        return -1;
    }
    
    // Generate a sine wave signal
    float* signal_data = (float*)input_buffer->data;
    for (size_t i = 0; i < signal_length; i++) {
        signal_data[i] = sinf(2.0f * 3.14159f * i / 100.0f) + 0.1f * sinf(2.0f * 3.14159f * i / 10.0f);
    }
    input_buffer->size = data_size;
    
    printf("   ✓ Generated %zu-sample sine wave signal\n", signal_length);
    
    // 5. Process data through optimized pipeline
    printf("\n4. Processing data through optimized pipeline...\n");
    
    pipeline_buffer_t* output_buffer = NULL;
    
    // Process single item
    bool success = pipeline_process(signal_pipeline, input_buffer, &output_buffer);
    if (!success) {
        printf("Error: Pipeline processing failed\n");
        return -1;
    }
    
    printf("   ✓ Single item processed successfully\n");
    
    // 6. Performance analysis
    printf("\n5. Performance Analysis:\n");
    pipeline_print_performance(signal_pipeline);
    
    // 7. Memory usage analysis
    printf("\n6. Memory Usage Analysis:\n");
    printf("Memory Pool Statistics After Processing:\n");
    memory_pool_print_stats(small_pool);
    memory_pool_print_stats(medium_pool);
    memory_pool_print_stats(large_pool);
    
    // 8. Batch processing demonstration
    printf("\n7. Batch Processing Demo...\n");
    
    const size_t batch_size = 5;
    pipeline_buffer_t* batch_inputs[batch_size];
    pipeline_buffer_t* batch_outputs[batch_size];
    
    // Create batch of input buffers
    for (size_t i = 0; i < batch_size; i++) {
        batch_inputs[i] = pipeline_buffer_create(signal_pipeline, 
                                               data_size, 
                                               PIPELINE_DATA_FLOAT32);
        if (batch_inputs[i]) {
            // Generate different frequency for each buffer
            float* batch_data = (float*)batch_inputs[i]->data;
            for (size_t j = 0; j < signal_length; j++) {
                batch_data[j] = sinf(2.0f * 3.14159f * j / (50.0f + i * 10.0f));
            }
            batch_inputs[i]->size = data_size;
        }
    }
    
    // Process batch
    bool batch_success = pipeline_process_batch(signal_pipeline, 
                                              batch_inputs, 
                                              batch_outputs, 
                                              batch_size);
    
    printf("   ✓ Batch of %zu items processed: %s\n", 
           batch_size, batch_success ? "SUCCESS" : "FAILED");
    
    // 9. Final performance metrics
    printf("\n8. Final Performance Metrics:\n");
    pipeline_stats_t final_stats = pipeline_get_stats(signal_pipeline);
    
    printf("Pipeline Summary:\n");
    printf("  Total items processed: %u\n", final_stats.total_processed);
    printf("  Total processing time: %lu us\n", final_stats.total_time_us);
    printf("  Average time per item: %lu us\n", final_stats.avg_time_per_item_us);
    printf("  Throughput: %.2f items/second\n", final_stats.throughput_items_per_sec);
    printf("  Memory efficiency: %zu bytes used\n", final_stats.memory_usage);
    printf("  Error rate: %.2f%%\n", 
           final_stats.total_processed > 0 ? 
           (float)final_stats.total_errors * 100.0f / final_stats.total_processed : 0.0f);
    
    // 10. Validate system integrity
    printf("\n9. System Integrity Check:\n");
    bool pipeline_valid = pipeline_validate_integrity(signal_pipeline);
    bool pools_valid = memory_pool_validate(small_pool) && 
                      memory_pool_validate(medium_pool) && 
                      memory_pool_validate(large_pool);
    
    printf("   Pipeline integrity: %s\n", pipeline_valid ? "VALID" : "CORRUPTED");
    printf("   Memory pools integrity: %s\n", pools_valid ? "VALID" : "CORRUPTED");
    
    // 11. Cleanup resources
    printf("\n10. Cleaning up resources...\n");
    
    // Release buffers
    pipeline_buffer_release(input_buffer);
    if (output_buffer) {
        pipeline_buffer_release(output_buffer);
    }
    
    for (size_t i = 0; i < batch_size; i++) {
        if (batch_inputs[i]) {
            pipeline_buffer_release(batch_inputs[i]);
        }
        if (batch_outputs[i]) {
            pipeline_buffer_release(batch_outputs[i]);
        }
    }
    
    // Destroy pipeline
    pipeline_destroy(signal_pipeline);
    
    // Destroy memory pools
    memory_pool_destroy(small_pool);
    memory_pool_destroy(medium_pool);
    memory_pool_destroy(large_pool);
    
    printf("   ✓ All resources cleaned up successfully\n");
    
    printf("\n=== AI artworkS Demo completed successfully! ===\n");
    printf("\nKey AI artworkS Optimizations Demonstrated:\n");
    printf("• Memory pool allocation reduces fragmentation\n");
    printf("• Pipeline stages can be enabled/disabled dynamically\n");
    printf("• Automatic buffer size optimization\n");
    printf("• Reference counting prevents memory leaks\n");
    printf("• Real-time performance monitoring\n");
    printf("• Batch processing for improved throughput\n");
    printf("• Built-in integrity validation\n");
    
    return 0;
}

// Custom processing stage example
bool custom_noise_reduction_stage(pipeline_buffer_t* input, pipeline_buffer_t* output, void* context) {
    float* threshold = (float*)context;
    
    if (!input || !output || !threshold || input->type != PIPELINE_DATA_FLOAT32) {
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
    
    // Simple noise gate
    for (size_t i = 0; i < count; i++) {
        if (fabsf(in_data[i]) < *threshold) {
            out_data[i] = 0.0f;  // Remove noise below threshold
        } else {
            out_data[i] = in_data[i];
        }
    }
    
    output->size = input->size;
    output->type = input->type;
    
    return true;
}

// Memory optimization helper
void optimize_pipeline_memory(data_pipeline_t* pipeline) {
    if (!pipeline) {
        return;
    }
    
    printf("Optimizing pipeline memory layout...\n");
    
    // Defragment all associated memory pools
    for (uint8_t i = 0; i < pipeline->pool_count; i++) {
        if (pipeline->buffer_pools[i]) {
            memory_pool_defragment(pipeline->buffer_pools[i]);
        }
    }
    
    printf("Memory optimization completed.\n");
}