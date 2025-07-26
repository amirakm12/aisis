#include "ultimate_core.h"
#include "ultimate_memory.h"
#include "ultimate_system.h"
#include "ultimate_neural.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Test counters */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, message) \
    do { \
        tests_run++; \
        if (condition) { \
            tests_passed++; \
            printf("✓ %s\n", message); \
        } else { \
            tests_failed++; \
            printf("✗ %s\n", message); \
        } \
    } while(0)

/* Test task function */
void test_task_function(void* params) {
    int* counter = (int*)params;
    (*counter)++;
    ultimate_task_sleep(100);
}

/* Timer callback function */
void test_timer_callback(ultimate_timer_handle_t timer, void* user_data) {
    int* counter = (int*)user_data;
    (*counter)++;
}

/* Test core system functions */
void test_core_system(void) {
    printf("\n=== Testing Core System Functions ===\n");
    
    ultimate_init_config_t config = {
        .cpu_frequency = 0,
        .tick_frequency = 1000,
        .max_tasks = 32,
        .max_queues = 16,
        .enable_watchdog = true,
        .enable_debug = true
    };
    
    /* Test system initialization */
    ultimate_error_t error = ultimate_init(&config);
    TEST_ASSERT(error == ULTIMATE_OK, "System initialization");
    
    /* Test system state */
    ultimate_state_t state = ultimate_get_state();
    TEST_ASSERT(state == ULTIMATE_STATE_READY, "System state after init");
    
    /* Test system start */
    error = ultimate_start();
    TEST_ASSERT(error == ULTIMATE_OK, "System start");
    
    state = ultimate_get_state();
    TEST_ASSERT(state == ULTIMATE_STATE_RUNNING, "System state after start");
    
    /* Test version functions */
    uint32_t version = ultimate_get_version();
    TEST_ASSERT(version > 0, "Get version number");
    
    const char* version_string = ultimate_get_version_string();
    TEST_ASSERT(version_string != NULL && strlen(version_string) > 0, "Get version string");
    
    /* Test timing functions */
    uint32_t tick1 = ultimate_get_tick_count();
    ultimate_delay_ms(10);
    uint32_t tick2 = ultimate_get_tick_count();
    TEST_ASSERT(tick2 > tick1, "Tick count increases");
    
    uint32_t time_ms = ultimate_get_time_ms();
    TEST_ASSERT(time_ms >= 0, "Get system time");
    
    /* Test critical sections */
    ultimate_enter_critical();
    ultimate_exit_critical();
    TEST_ASSERT(true, "Critical section management");
    
    /* Test system stop */
    error = ultimate_stop();
    TEST_ASSERT(error == ULTIMATE_OK, "System stop");
    
    /* Test system shutdown */
    error = ultimate_shutdown();
    TEST_ASSERT(error == ULTIMATE_OK, "System shutdown");
}

/* Test memory management functions */
void test_memory_management(void) {
    printf("\n=== Testing Memory Management Functions ===\n");
    
    /* Test basic allocation */
    void* ptr1 = ultimate_malloc(1024);
    TEST_ASSERT(ptr1 != NULL, "Memory allocation");
    
    void* ptr2 = ultimate_calloc(10, 100);
    TEST_ASSERT(ptr2 != NULL, "Calloc allocation");
    
    /* Test reallocation */
    ptr1 = ultimate_realloc(ptr1, 2048);
    TEST_ASSERT(ptr1 != NULL, "Memory reallocation");
    
    /* Test aligned allocation */
    void* aligned_ptr = ultimate_aligned_malloc(1024, 64);
    TEST_ASSERT(aligned_ptr != NULL, "Aligned memory allocation");
    TEST_ASSERT(((uintptr_t)aligned_ptr % 64) == 0, "Memory alignment check");
    
    /* Test memory pool */
    ultimate_pool_config_t pool_config = {
        .total_size = 4096,
        .block_size = 64,
        .max_blocks = 64,
        .type = ULTIMATE_POOL_TYPE_FIXED,
        .name = "TestPool",
        .thread_safe = true
    };
    
    ultimate_pool_handle_t pool;
    ultimate_error_t error = ultimate_pool_create(&pool_config, &pool);
    TEST_ASSERT(error == ULTIMATE_OK, "Memory pool creation");
    
    void* block1 = ultimate_pool_alloc(pool);
    TEST_ASSERT(block1 != NULL, "Pool block allocation");
    
    void* block2 = ultimate_pool_alloc(pool);
    TEST_ASSERT(block2 != NULL, "Second pool block allocation");
    
    error = ultimate_pool_free(pool, block1);
    TEST_ASSERT(error == ULTIMATE_OK, "Pool block deallocation");
    
    ultimate_pool_stats_t pool_stats;
    error = ultimate_pool_get_stats(pool, &pool_stats);
    TEST_ASSERT(error == ULTIMATE_OK, "Pool statistics");
    TEST_ASSERT(pool_stats.total_blocks == 64, "Pool total blocks");
    
    error = ultimate_pool_destroy(pool);
    TEST_ASSERT(error == ULTIMATE_OK, "Pool destruction");
    
    /* Test memory statistics */
    ultimate_memory_stats_t stats;
    error = ultimate_memory_get_stats(&stats);
    TEST_ASSERT(error == ULTIMATE_OK, "Memory statistics");
    
    /* Test memory barriers */
    ultimate_memory_barrier();
    ultimate_memory_read_barrier();
    ultimate_memory_write_barrier();
    TEST_ASSERT(true, "Memory barriers");
    
    /* Test fast memory operations */
    char src[100], dst[100];
    memset(src, 0xAA, sizeof(src));
    ultimate_memory_copy_fast(dst, src, sizeof(src));
    TEST_ASSERT(memcmp(src, dst, sizeof(src)) == 0, "Fast memory copy");
    
    ultimate_memory_set_fast(dst, 0x55, sizeof(dst));
    TEST_ASSERT(dst[0] == 0x55 && dst[99] == 0x55, "Fast memory set");
    
    /* Cleanup */
    ultimate_free(ptr1);
    ultimate_free(ptr2);
    ultimate_aligned_free(aligned_ptr);
    ultimate_pool_free(pool, block2);
}

/* Test task management functions */
void test_task_management(void) {
    printf("\n=== Testing Task Management Functions ===\n");
    
    /* Initialize system first */
    ultimate_init_config_t config = {
        .cpu_frequency = 0,
        .tick_frequency = 1000,
        .max_tasks = 32,
        .max_queues = 16,
        .enable_watchdog = true,
        .enable_debug = true
    };
    ultimate_init(&config);
    ultimate_start();
    
    int task_counter = 0;
    ultimate_task_config_t task_config = {
        .priority = ULTIMATE_PRIORITY_NORMAL,
        .stack_size = 4096,
        .name = "TestTask",
        .auto_start = false,
        .watchdog_timeout = 5000
    };
    
    ultimate_task_handle_t task;
    ultimate_error_t error = ultimate_task_create(test_task_function, &task_counter, &task_config, &task);
    TEST_ASSERT(error == ULTIMATE_OK, "Task creation");
    
    uint32_t task_id = ultimate_task_get_id(task);
    TEST_ASSERT(task_id > 0, "Task ID retrieval");
    
    ultimate_task_state_t state = ultimate_task_get_state(task);
    TEST_ASSERT(state == ULTIMATE_TASK_STATE_READY, "Initial task state");
    
    error = ultimate_task_start(task);
    TEST_ASSERT(error == ULTIMATE_OK, "Task start");
    
    ultimate_delay_ms(200);  /* Let task run */
    
    state = ultimate_task_get_state(task);
    TEST_ASSERT(state == ULTIMATE_TASK_STATE_RUNNING, "Running task state");
    
    error = ultimate_task_suspend(task);
    TEST_ASSERT(error == ULTIMATE_OK, "Task suspend");
    
    error = ultimate_task_resume(task);
    TEST_ASSERT(error == ULTIMATE_OK, "Task resume");
    
    error = ultimate_task_terminate(task);
    TEST_ASSERT(error == ULTIMATE_OK, "Task termination");
    
    TEST_ASSERT(task_counter > 0, "Task executed successfully");
    
    /* Test task utility functions */
    error = ultimate_task_sleep(10);
    TEST_ASSERT(error == ULTIMATE_OK, "Task sleep");
    
    error = ultimate_task_yield();
    TEST_ASSERT(error == ULTIMATE_OK, "Task yield");
    
    ultimate_shutdown();
}

/* Test queue management functions */
void test_queue_management(void) {
    printf("\n=== Testing Queue Management Functions ===\n");
    
    ultimate_queue_handle_t queue;
    ultimate_error_t error = ultimate_queue_create(10, sizeof(int), "TestQueue", &queue);
    TEST_ASSERT(error == ULTIMATE_OK, "Queue creation");
    
    uint32_t count = ultimate_queue_get_count(queue);
    TEST_ASSERT(count == 0, "Initial queue count");
    
    int test_data = 42;
    error = ultimate_queue_send(queue, &test_data, 1000);
    TEST_ASSERT(error == ULTIMATE_OK, "Queue send");
    
    count = ultimate_queue_get_count(queue);
    TEST_ASSERT(count == 1, "Queue count after send");
    
    int received_data = 0;
    error = ultimate_queue_receive(queue, &received_data, 1000);
    TEST_ASSERT(error == ULTIMATE_OK, "Queue receive");
    TEST_ASSERT(received_data == test_data, "Queue data integrity");
    
    count = ultimate_queue_get_count(queue);
    TEST_ASSERT(count == 0, "Queue count after receive");
    
    error = ultimate_queue_destroy(queue);
    TEST_ASSERT(error == ULTIMATE_OK, "Queue destruction");
}

/* Test timer management functions */
void test_timer_management(void) {
    printf("\n=== Testing Timer Management Functions ===\n");
    
    int timer_counter = 0;
    ultimate_timer_handle_t timer;
    ultimate_error_t error = ultimate_timer_create(100, false, test_timer_callback, &timer_counter, "TestTimer", &timer);
    TEST_ASSERT(error == ULTIMATE_OK, "Timer creation");
    
    bool is_active = ultimate_timer_is_active(timer);
    TEST_ASSERT(!is_active, "Timer initially inactive");
    
    error = ultimate_timer_start(timer);
    TEST_ASSERT(error == ULTIMATE_OK, "Timer start");
    
    is_active = ultimate_timer_is_active(timer);
    TEST_ASSERT(is_active, "Timer active after start");
    
    uint32_t remaining = ultimate_timer_get_remaining(timer);
    TEST_ASSERT(remaining <= 100, "Timer remaining time");
    
    error = ultimate_timer_reset(timer);
    TEST_ASSERT(error == ULTIMATE_OK, "Timer reset");
    
    error = ultimate_timer_stop(timer);
    TEST_ASSERT(error == ULTIMATE_OK, "Timer stop");
    
    is_active = ultimate_timer_is_active(timer);
    TEST_ASSERT(!is_active, "Timer inactive after stop");
}

/* Test neural network functions */
void test_neural_network(void) {
    printf("\n=== Testing Neural Network Functions ===\n");
    
    ultimate_error_t error = ultimate_neural_init();
    TEST_ASSERT(error == ULTIMATE_OK, "Neural system initialization");
    
    ultimate_neural_config_t config = {
        .model_type = ULTIMATE_NEURAL_TYPE_FEEDFORWARD,
        .precision = ULTIMATE_NEURAL_PRECISION_FLOAT32,
        .memory_limit = 32768,
        .name = "TestModel"
    };
    
    ultimate_neural_model_t model;
    error = ultimate_neural_model_create(&config, &model);
    TEST_ASSERT(error == ULTIMATE_OK, "Neural model creation");
    
    error = ultimate_neural_model_add_layer(model, 4);  /* Input layer */
    TEST_ASSERT(error == ULTIMATE_OK, "Add input layer");
    
    error = ultimate_neural_model_add_layer(model, 8);  /* Hidden layer */
    TEST_ASSERT(error == ULTIMATE_OK, "Add hidden layer");
    
    error = ultimate_neural_model_add_layer(model, 2);  /* Output layer */
    TEST_ASSERT(error == ULTIMATE_OK, "Add output layer");
    
    uint32_t layer_count = ultimate_neural_model_get_layer_count(model);
    TEST_ASSERT(layer_count == 3, "Layer count verification");
    
    error = ultimate_neural_model_compile(model);
    TEST_ASSERT(error == ULTIMATE_OK, "Model compilation");
    
    uint32_t param_count = ultimate_neural_model_get_parameter_count(model);
    TEST_ASSERT(param_count > 0, "Parameter count");
    
    /* Test tensor operations */
    uint32_t input_shape[] = {4};
    ultimate_neural_tensor_t input_tensor;
    error = ultimate_neural_tensor_create(input_shape, 1, ULTIMATE_NEURAL_PRECISION_FLOAT32, &input_tensor);
    TEST_ASSERT(error == ULTIMATE_OK, "Input tensor creation");
    
    error = ultimate_neural_tensor_fill(&input_tensor, 1.0f);
    TEST_ASSERT(error == ULTIMATE_OK, "Tensor fill");
    
    uint32_t output_shape[] = {2};
    ultimate_neural_tensor_t output_tensor;
    error = ultimate_neural_tensor_create(output_shape, 1, ULTIMATE_NEURAL_PRECISION_FLOAT32, &output_tensor);
    TEST_ASSERT(error == ULTIMATE_OK, "Output tensor creation");
    
    /* Test inference */
    error = ultimate_neural_inference(model, input_tensor, output_tensor);
    TEST_ASSERT(error == ULTIMATE_OK, "Neural inference");
    
    /* Test model properties */
    bool is_trained = ultimate_neural_model_is_trained(model);
    TEST_ASSERT(!is_trained, "Model initially untrained");
    
    float learning_rate = ultimate_neural_model_get_learning_rate(model);
    TEST_ASSERT(learning_rate > 0.0f, "Default learning rate");
    
    error = ultimate_neural_model_set_learning_rate(model, 0.01f);
    TEST_ASSERT(error == ULTIMATE_OK, "Set learning rate");
    
    learning_rate = ultimate_neural_model_get_learning_rate(model);
    TEST_ASSERT(learning_rate == 0.01f, "Learning rate updated");
    
    /* Test model save/load */
    error = ultimate_neural_model_save(model, "test_model.bin");
    TEST_ASSERT(error == ULTIMATE_OK, "Model save");
    
    ultimate_neural_model_t loaded_model;
    error = ultimate_neural_model_load("test_model.bin", &loaded_model);
    TEST_ASSERT(error == ULTIMATE_OK, "Model load");
    
    uint32_t loaded_layers = ultimate_neural_model_get_layer_count(loaded_model);
    TEST_ASSERT(loaded_layers == layer_count, "Loaded model layer count");
    
    /* Cleanup */
    error = ultimate_neural_tensor_destroy(&input_tensor);
    TEST_ASSERT(error == ULTIMATE_OK, "Input tensor cleanup");
    
    error = ultimate_neural_tensor_destroy(&output_tensor);
    TEST_ASSERT(error == ULTIMATE_OK, "Output tensor cleanup");
    
    error = ultimate_neural_model_destroy(model);
    TEST_ASSERT(error == ULTIMATE_OK, "Model cleanup");
    
    error = ultimate_neural_model_destroy(loaded_model);
    TEST_ASSERT(error == ULTIMATE_OK, "Loaded model cleanup");
    
    error = ultimate_neural_shutdown();
    TEST_ASSERT(error == ULTIMATE_OK, "Neural system shutdown");
}

/* Test file I/O functions */
void test_file_io(void) {
    printf("\n=== Testing File I/O Functions ===\n");
    
    const char* test_filename = "test_file.txt";
    const char* test_data = "Hello, ULTIMATE System!";
    
    /* Test file creation and writing */
    ultimate_file_handle_t file;
    ultimate_error_t error = ultimate_file_open(test_filename, ULTIMATE_FILE_MODE_WRITE, &file);
    TEST_ASSERT(error == ULTIMATE_OK, "File creation");
    
    size_t bytes_written;
    error = ultimate_file_write(file, test_data, strlen(test_data), &bytes_written);
    TEST_ASSERT(error == ULTIMATE_OK, "File write");
    TEST_ASSERT(bytes_written == strlen(test_data), "Write byte count");
    
    error = ultimate_file_flush(file);
    TEST_ASSERT(error == ULTIMATE_OK, "File flush");
    
    error = ultimate_file_close(file);
    TEST_ASSERT(error == ULTIMATE_OK, "File close");
    
    /* Test file existence */
    bool exists = ultimate_file_exists(test_filename);
    TEST_ASSERT(exists, "File exists check");
    
    /* Test file reading */
    error = ultimate_file_open(test_filename, ULTIMATE_FILE_MODE_READ, &file);
    TEST_ASSERT(error == ULTIMATE_OK, "File open for reading");
    
    uint64_t file_size;
    error = ultimate_file_size(file, &file_size);
    TEST_ASSERT(error == ULTIMATE_OK, "File size query");
    TEST_ASSERT(file_size == strlen(test_data), "File size correct");
    
    char read_buffer[100];
    size_t bytes_read;
    error = ultimate_file_read(file, read_buffer, sizeof(read_buffer), &bytes_read);
    TEST_ASSERT(error == ULTIMATE_OK, "File read");
    TEST_ASSERT(bytes_read == strlen(test_data), "Read byte count");
    
    read_buffer[bytes_read] = '\0';
    TEST_ASSERT(strcmp(read_buffer, test_data) == 0, "File data integrity");
    
    /* Test file seeking */
    error = ultimate_file_seek(file, 0, 0);  /* SEEK_SET */
    TEST_ASSERT(error == ULTIMATE_OK, "File seek");
    
    int64_t position;
    error = ultimate_file_tell(file, &position);
    TEST_ASSERT(error == ULTIMATE_OK, "File tell");
    TEST_ASSERT(position == 0, "File position after seek");
    
    error = ultimate_file_close(file);
    TEST_ASSERT(error == ULTIMATE_OK, "File close after reading");
    
    /* Test file operations */
    const char* copy_filename = "test_file_copy.txt";
    error = ultimate_file_copy(test_filename, copy_filename);
    TEST_ASSERT(error == ULTIMATE_OK, "File copy");
    
    exists = ultimate_file_exists(copy_filename);
    TEST_ASSERT(exists, "Copied file exists");
    
    const char* move_filename = "test_file_moved.txt";
    error = ultimate_file_move(copy_filename, move_filename);
    TEST_ASSERT(error == ULTIMATE_OK, "File move");
    
    exists = ultimate_file_exists(move_filename);
    TEST_ASSERT(exists, "Moved file exists");
    
    exists = ultimate_file_exists(copy_filename);
    TEST_ASSERT(!exists, "Original copy file removed");
    
    /* Cleanup */
    error = ultimate_file_delete(test_filename);
    TEST_ASSERT(error == ULTIMATE_OK, "File deletion");
    
    error = ultimate_file_delete(move_filename);
    TEST_ASSERT(error == ULTIMATE_OK, "Moved file deletion");
}

/* Test system statistics */
void test_system_statistics(void) {
    printf("\n=== Testing System Statistics Functions ===\n");
    
    /* Initialize system */
    ultimate_init_config_t config = {
        .cpu_frequency = 0,
        .tick_frequency = 1000,
        .max_tasks = 32,
        .max_queues = 16,
        .enable_watchdog = true,
        .enable_debug = true
    };
    ultimate_init(&config);
    ultimate_start();
    
    ultimate_system_stats_t stats;
    ultimate_error_t error = ultimate_system_get_stats(&stats);
    TEST_ASSERT(error == ULTIMATE_OK, "System statistics retrieval");
    
    TEST_ASSERT(stats.uptime_ms >= 0, "System uptime");
    TEST_ASSERT(stats.active_tasks >= 0, "Active task count");
    TEST_ASSERT(stats.cpu_usage_percent <= 100, "CPU usage percentage");
    TEST_ASSERT(stats.memory_info.total_size > 0, "Memory total size");
    
    ultimate_shutdown();
}

/* Main test runner */
int main(void) {
    printf("ULTIMATE System Test Suite\n");
    printf("==========================\n");
    
    /* Run all test suites */
    test_core_system();
    test_memory_management();
    test_task_management();
    test_queue_management();
    test_timer_management();
    test_neural_network();
    test_file_io();
    test_system_statistics();
    
    /* Print summary */
    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    
    if (tests_failed == 0) {
        printf("🎉 All tests passed!\n");
        return 0;
    } else {
        printf("❌ %d tests failed\n", tests_failed);
        return 1;
    }
}