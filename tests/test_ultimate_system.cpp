#include <gtest/gtest.h>
#include "ultimate_core.h"
#include "ultimate_memory.h"
#include "ultimate_system.h"
#include "ultimate_neural.h"
#include "ultimate_hardware.h"
#include <thread>
#include <chrono>

class UltimateSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the system for each test
        ultimate_init_config_t config = {};
        config.cpu_frequency = 0;
        config.tick_frequency = 1000;
        config.max_tasks = 32;
        config.max_queues = 16;
        config.heap_size = 1024 * 1024; // 1MB
        config.enable_watchdog = true;
        config.enable_debug = true;
        
        ASSERT_EQ(ultimate_init(&config), ULTIMATE_OK);
        ASSERT_EQ(ultimate_start(), ULTIMATE_OK);
    }
    
    void TearDown() override {
        ultimate_stop();
        ultimate_shutdown();
    }
};

// Core System Tests
TEST_F(UltimateSystemTest, SystemInitialization) {
    EXPECT_TRUE(ultimate_is_initialized());
    EXPECT_TRUE(ultimate_is_running());
    
    // Test version string
    const char* version = ultimate_get_version();
    EXPECT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0);
    
    // Test uptime
    ultimate_delay_ms(100);
    uint64_t uptime = ultimate_get_uptime_ms();
    EXPECT_GE(uptime, 100);
}

TEST_F(UltimateSystemTest, SystemStats) {
    ultimate_system_stats_t stats = ultimate_system_get_stats();
    
    EXPECT_GE(stats.uptime_ms, 0);
    EXPECT_LE(stats.active_tasks, 32); // Max tasks configured
    EXPECT_GE(stats.memory_info.total_size, 1024 * 1024); // At least 1MB
    EXPECT_LE(stats.memory_info.used_size, stats.memory_info.total_size);
}

// Memory Management Tests
TEST_F(UltimateSystemTest, MemoryAllocation) {
    // Test basic allocation
    void* ptr1 = ultimate_malloc(1024);
    ASSERT_NE(ptr1, nullptr);
    
    void* ptr2 = ultimate_malloc(2048);
    ASSERT_NE(ptr2, nullptr);
    
    // Test memory stats
    ultimate_memory_stats_t stats = ultimate_memory_get_stats();
    EXPECT_GE(stats.total_allocated, 1024 + 2048);
    EXPECT_GE(stats.allocation_count, 2);
    
    // Test reallocation
    ptr1 = ultimate_realloc(ptr1, 4096);
    ASSERT_NE(ptr1, nullptr);
    
    // Test calloc
    void* ptr3 = ultimate_calloc(100, sizeof(int));
    ASSERT_NE(ptr3, nullptr);
    
    // Verify calloc zeroed memory
    int* int_ptr = static_cast<int*>(ptr3);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(int_ptr[i], 0);
    }
    
    // Free memory
    ultimate_free(ptr1);
    ultimate_free(ptr2);
    ultimate_free(ptr3);
}

TEST_F(UltimateSystemTest, MemoryLeakDetection) {
    // Allocate some memory without freeing
    void* leak1 = ultimate_malloc(512);
    void* leak2 = ultimate_malloc(1024);
    
    // Wait a bit to ensure timestamps are different
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Check for leaks
    ultimate_memory_leak_t leaks[10];
    size_t leak_count = ultimate_memory_check_leaks(leaks, 10);
    
    EXPECT_GE(leak_count, 2);
    
    // Cleanup
    ultimate_free(leak1);
    ultimate_free(leak2);
}

// Task Management Tests
void test_task_function(void* params) {
    int* counter = static_cast<int*>(params);
    for (int i = 0; i < 10; ++i) {
        (*counter)++;
        ultimate_task_sleep(10);
    }
}

TEST_F(UltimateSystemTest, TaskCreationAndExecution) {
    int counter = 0;
    
    ultimate_task_config_t task_config = {};
    task_config.priority = ULTIMATE_PRIORITY_NORMAL;
    task_config.stack_size = 4096;
    strcpy(task_config.name, "TestTask");
    task_config.auto_start = true;
    task_config.watchdog_timeout = 5000;
    
    ultimate_task_handle_t task_handle;
    ASSERT_EQ(ultimate_task_create(test_task_function, &counter, &task_config, &task_handle), ULTIMATE_OK);
    
    // Wait for task to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    EXPECT_EQ(counter, 10);
    
    // Clean up task
    ultimate_task_delete(task_handle);
}

TEST_F(UltimateSystemTest, TaskControlOperations) {
    int counter = 0;
    
    ultimate_task_config_t task_config = {};
    task_config.priority = ULTIMATE_PRIORITY_NORMAL;
    task_config.stack_size = 4096;
    strcpy(task_config.name, "ControlTask");
    task_config.auto_start = false; // Don't auto-start
    task_config.watchdog_timeout = 5000;
    
    ultimate_task_handle_t task_handle;
    ASSERT_EQ(ultimate_task_create(test_task_function, &counter, &task_config, &task_handle), ULTIMATE_OK);
    
    // Task shouldn't run yet
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(counter, 0);
    
    // Start the task
    ASSERT_EQ(ultimate_task_start(task_handle), ULTIMATE_OK);
    
    // Let it run a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_GT(counter, 0);
    
    // Stop the task
    ASSERT_EQ(ultimate_task_stop(task_handle), ULTIMATE_OK);
    
    int stopped_counter = counter;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(counter, stopped_counter); // Should not have changed
    
    // Clean up
    ultimate_task_delete(task_handle);
}

// Power Management Tests
TEST_F(UltimateSystemTest, PowerManagement) {
    // Test power mode setting
    ASSERT_EQ(ultimate_power_set_mode(ULTIMATE_POWER_MODE_HIGH_PERFORMANCE), ULTIMATE_OK);
    EXPECT_EQ(ultimate_power_get_mode(), ULTIMATE_POWER_MODE_HIGH_PERFORMANCE);
    
    ASSERT_EQ(ultimate_power_set_mode(ULTIMATE_POWER_MODE_LOW_POWER), ULTIMATE_OK);
    EXPECT_EQ(ultimate_power_get_mode(), ULTIMATE_POWER_MODE_LOW_POWER);
    
    ASSERT_EQ(ultimate_power_set_mode(ULTIMATE_POWER_MODE_NORMAL), ULTIMATE_OK);
    EXPECT_EQ(ultimate_power_get_mode(), ULTIMATE_POWER_MODE_NORMAL);
}

// Neural Processing Tests
class NeuralProcessingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(ultimate_neural_init(), ULTIMATE_OK);
    }
    
    void TearDown() override {
        ultimate_neural_deinit();
    }
};

TEST_F(NeuralProcessingTest, ModelCreationAndInference) {
    // Create a neural model
    ultimate_neural_config_t neural_config = {};
    neural_config.model_type = ULTIMATE_NEURAL_TYPE_FEEDFORWARD;
    neural_config.precision = ULTIMATE_NEURAL_PRECISION_FLOAT32;
    neural_config.memory_limit = 32768;
    strcpy(neural_config.name, "TestModel");
    
    ultimate_neural_model_handle_t model_handle;
    ASSERT_EQ(ultimate_neural_model_create(&neural_config, &model_handle), ULTIMATE_OK);
    
    // Create input tensor
    size_t input_shape[] = {784}; // 28x28 image
    ultimate_neural_tensor_handle_t input_tensor;
    ASSERT_EQ(ultimate_neural_tensor_create(input_shape, 1, ULTIMATE_NEURAL_PRECISION_FLOAT32, &input_tensor), ULTIMATE_OK);
    
    // Create output tensor
    size_t output_shape[] = {10}; // 10 classes
    ultimate_neural_tensor_handle_t output_tensor;
    ASSERT_EQ(ultimate_neural_tensor_create(output_shape, 1, ULTIMATE_NEURAL_PRECISION_FLOAT32, &output_tensor), ULTIMATE_OK);
    
    // Set input data
    std::vector<float> input_data(784, 0.5f); // Simple test data
    ASSERT_EQ(ultimate_neural_tensor_set_data(input_tensor, input_data.data(), input_data.size() * sizeof(float)), ULTIMATE_OK);
    
    // Perform inference
    ASSERT_EQ(ultimate_neural_inference(model_handle, input_tensor, output_tensor), ULTIMATE_OK);
    
    // Get output data
    std::vector<float> output_data(10);
    ASSERT_EQ(ultimate_neural_tensor_get_data(output_tensor, output_data.data(), output_data.size() * sizeof(float)), ULTIMATE_OK);
    
    // Verify output (should be valid probabilities for softmax)
    float sum = 0.0f;
    for (float val : output_data) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0f, 0.01f); // Softmax output should sum to 1
}

TEST_F(NeuralProcessingTest, ModelSaveAndLoad) {
    // Create and configure a model
    ultimate_neural_config_t neural_config = {};
    neural_config.model_type = ULTIMATE_NEURAL_TYPE_FEEDFORWARD;
    neural_config.precision = ULTIMATE_NEURAL_PRECISION_FLOAT32;
    neural_config.memory_limit = 32768;
    strcpy(neural_config.name, "SaveLoadTest");
    
    ultimate_neural_model_handle_t original_model;
    ASSERT_EQ(ultimate_neural_model_create(&neural_config, &original_model), ULTIMATE_OK);
    
    // Save the model
    const char* model_file = "test_model.bin";
    ASSERT_EQ(ultimate_neural_model_save(original_model, model_file), ULTIMATE_OK);
    
    // Load the model
    ultimate_neural_model_handle_t loaded_model;
    ASSERT_EQ(ultimate_neural_model_load(model_file, &loaded_model), ULTIMATE_OK);
    
    // Test that loaded model works
    size_t input_shape[] = {784};
    ultimate_neural_tensor_handle_t input_tensor;
    ASSERT_EQ(ultimate_neural_tensor_create(input_shape, 1, ULTIMATE_NEURAL_PRECISION_FLOAT32, &input_tensor), ULTIMATE_OK);
    
    size_t output_shape[] = {10};
    ultimate_neural_tensor_handle_t output_tensor;
    ASSERT_EQ(ultimate_neural_tensor_create(output_shape, 1, ULTIMATE_NEURAL_PRECISION_FLOAT32, &output_tensor), ULTIMATE_OK);
    
    std::vector<float> input_data(784, 0.3f);
    ASSERT_EQ(ultimate_neural_tensor_set_data(input_tensor, input_data.data(), input_data.size() * sizeof(float)), ULTIMATE_OK);
    
    ASSERT_EQ(ultimate_neural_inference(loaded_model, input_tensor, output_tensor), ULTIMATE_OK);
    
    // Clean up
    remove(model_file);
}

// Hardware Integration Tests
class HardwareTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(ultimate_hardware_init(), ULTIMATE_OK);
    }
    
    void TearDown() override {
        ultimate_hardware_deinit();
    }
};

TEST_F(HardwareTest, FileOperations) {
    const char* test_file = "test_file.txt";
    const char* test_data = "Hello, ULTIMATE System!";
    size_t test_data_len = strlen(test_data);
    
    // Test file writing
    ultimate_file_handle_t write_handle;
    ASSERT_EQ(ultimate_file_open(test_file, ULTIMATE_FILE_MODE_WRITE, &write_handle), ULTIMATE_OK);
    
    size_t bytes_written;
    ASSERT_EQ(ultimate_file_write(write_handle, test_data, test_data_len, &bytes_written), ULTIMATE_OK);
    EXPECT_EQ(bytes_written, test_data_len);
    
    ASSERT_EQ(ultimate_file_close(write_handle), ULTIMATE_OK);
    
    // Test file reading
    ultimate_file_handle_t read_handle;
    ASSERT_EQ(ultimate_file_open(test_file, ULTIMATE_FILE_MODE_READ, &read_handle), ULTIMATE_OK);
    
    char read_buffer[256];
    size_t bytes_read;
    ASSERT_EQ(ultimate_file_read(read_handle, read_buffer, sizeof(read_buffer), &bytes_read), ULTIMATE_OK);
    EXPECT_EQ(bytes_read, test_data_len);
    
    read_buffer[bytes_read] = '\0';
    EXPECT_STREQ(read_buffer, test_data);
    
    ASSERT_EQ(ultimate_file_close(read_handle), ULTIMATE_OK);
    
    // Clean up
    remove(test_file);
}

TEST_F(HardwareTest, WindowCreation) {
    ultimate_window_config_t window_config = {};
    window_config.width = 800;
    window_config.height = 600;
    strcpy(window_config.title, "Test Window");
    window_config.resizable = true;
    window_config.fullscreen = false;
    
    ultimate_window_handle_t window_handle;
    ASSERT_EQ(ultimate_window_create(&window_config, &window_handle), ULTIMATE_OK);
    
    // Test showing and hiding window
    ASSERT_EQ(ultimate_window_show(window_handle), ULTIMATE_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    ASSERT_EQ(ultimate_window_hide(window_handle), ULTIMATE_OK);
}

// Socket operations test (simplified - would need actual server for full test)
TEST_F(HardwareTest, SocketCreation) {
    ultimate_socket_handle_t tcp_socket;
    ASSERT_EQ(ultimate_socket_create(ULTIMATE_SOCKET_TYPE_TCP, &tcp_socket), ULTIMATE_OK);
    
    ultimate_socket_handle_t udp_socket;
    ASSERT_EQ(ultimate_socket_create(ULTIMATE_SOCKET_TYPE_UDP, &udp_socket), ULTIMATE_OK);
    
    // Clean up
    ASSERT_EQ(ultimate_socket_close(tcp_socket), ULTIMATE_OK);
    ASSERT_EQ(ultimate_socket_close(udp_socket), ULTIMATE_OK);
}

// Performance Tests
TEST_F(UltimateSystemTest, MemoryPerformance) {
    const int num_allocations = 1000;
    const size_t allocation_size = 1024;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<void*> pointers;
    pointers.reserve(num_allocations);
    
    // Allocation performance
    for (int i = 0; i < num_allocations; ++i) {
        void* ptr = ultimate_malloc(allocation_size);
        ASSERT_NE(ptr, nullptr);
        pointers.push_back(ptr);
    }
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    // Deallocation performance
    for (void* ptr : pointers) {
        ultimate_free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto free_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
    
    // Performance expectations (adjust based on system)
    EXPECT_LT(alloc_time.count(), 10000); // Less than 10ms for 1000 allocations
    EXPECT_LT(free_time.count(), 5000);   // Less than 5ms for 1000 deallocations
    
    std::cout << "Allocation time: " << alloc_time.count() << " microseconds" << std::endl;
    std::cout << "Deallocation time: " << free_time.count() << " microseconds" << std::endl;
}

// Error Handling Tests
TEST_F(UltimateSystemTest, ErrorHandling) {
    // Test invalid parameters
    EXPECT_EQ(ultimate_task_create(nullptr, nullptr, nullptr, nullptr), ULTIMATE_ERROR_INVALID_PARAMETER);
    
    // Test invalid handles
    EXPECT_EQ(ultimate_task_start(9999), ULTIMATE_ERROR_TASK_NOT_FOUND);
    EXPECT_EQ(ultimate_task_stop(9999), ULTIMATE_ERROR_TASK_NOT_FOUND);
    EXPECT_EQ(ultimate_task_delete(9999), ULTIMATE_ERROR_TASK_NOT_FOUND);
    
    // Test file operations with invalid handles
    size_t bytes_read;
    char buffer[100];
    EXPECT_EQ(ultimate_file_read(9999, buffer, sizeof(buffer), &bytes_read), ULTIMATE_ERROR_FILE_NOT_FOUND);
    
    // Test neural operations without initialization
    ultimate_neural_deinit(); // Ensure it's not initialized
    
    ultimate_neural_config_t config = {};
    ultimate_neural_model_handle_t model;
    EXPECT_EQ(ultimate_neural_model_create(&config, &model), ULTIMATE_ERROR_NOT_INITIALIZED);
    
    // Re-initialize for cleanup
    ultimate_neural_init();
}

// Integration Tests
TEST_F(UltimateSystemTest, FullSystemIntegration) {
    // Test that all subsystems work together
    
    // 1. Memory allocation
    void* mem = ultimate_malloc(4096);
    ASSERT_NE(mem, nullptr);
    
    // 2. Task creation
    int task_counter = 0;
    ultimate_task_config_t task_config = {};
    task_config.priority = ULTIMATE_PRIORITY_NORMAL;
    task_config.stack_size = 4096;
    strcpy(task_config.name, "IntegrationTask");
    task_config.auto_start = true;
    task_config.watchdog_timeout = 5000;
    
    ultimate_task_handle_t task;
    ASSERT_EQ(ultimate_task_create(test_task_function, &task_counter, &task_config, &task), ULTIMATE_OK);
    
    // 3. Neural processing
    ASSERT_EQ(ultimate_neural_init(), ULTIMATE_OK);
    
    ultimate_neural_config_t neural_config = {};
    neural_config.model_type = ULTIMATE_NEURAL_TYPE_FEEDFORWARD;
    neural_config.precision = ULTIMATE_NEURAL_PRECISION_FLOAT32;
    neural_config.memory_limit = 32768;
    strcpy(neural_config.name, "IntegrationModel");
    
    ultimate_neural_model_handle_t model;
    ASSERT_EQ(ultimate_neural_model_create(&neural_config, &model), ULTIMATE_OK);
    
    // 4. Hardware operations
    ASSERT_EQ(ultimate_hardware_init(), ULTIMATE_OK);
    
    const char* test_file = "integration_test.txt";
    ultimate_file_handle_t file;
    ASSERT_EQ(ultimate_file_open(test_file, ULTIMATE_FILE_MODE_WRITE, &file), ULTIMATE_OK);
    
    const char* data = "Integration test successful!";
    size_t bytes_written;
    ASSERT_EQ(ultimate_file_write(file, data, strlen(data), &bytes_written), ULTIMATE_OK);
    ASSERT_EQ(ultimate_file_close(file), ULTIMATE_OK);
    
    // 5. Wait for task to complete and verify all systems worked
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(task_counter, 10);
    
    // 6. Check system stats
    ultimate_system_stats_t stats = ultimate_system_get_stats();
    EXPECT_GT(stats.uptime_ms, 0);
    EXPECT_GT(stats.memory_info.used_size, 0);
    
    // Cleanup
    ultimate_free(mem);
    ultimate_task_delete(task);
    ultimate_neural_deinit();
    ultimate_hardware_deinit();
    remove(test_file);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}