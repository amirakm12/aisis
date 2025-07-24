#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/aisis.h"

// Test result counters
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

// Test macros
#define TEST_ASSERT(condition, message) \
    do { \
        tests_run++; \
        if (condition) { \
            tests_passed++; \
            printf("✓ PASS: %s\n", message); \
        } else { \
            tests_failed++; \
            printf("✗ FAIL: %s\n", message); \
        } \
    } while(0)

#define TEST_ASSERT_EQUAL(expected, actual, message) \
    TEST_ASSERT((expected) == (actual), message)

#define TEST_ASSERT_NOT_NULL(ptr, message) \
    TEST_ASSERT((ptr) != NULL, message)

// Test functions
void test_aisis_init(void) {
    printf("\n=== Testing AISIS Initialization ===\n");
    
    // Test basic initialization
    int result = aisis_init();
    TEST_ASSERT_EQUAL(AISIS_SUCCESS, result, "Basic initialization should succeed");
    
    // Test double initialization (should succeed)
    result = aisis_init();
    TEST_ASSERT_EQUAL(AISIS_SUCCESS, result, "Double initialization should succeed");
    
    // Test status after initialization
    aisis_status_t status = aisis_get_status();
    TEST_ASSERT_EQUAL(AISIS_STATUS_IDLE, status, "Status should be IDLE after init");
    
    // Cleanup for next tests
    aisis_cleanup();
}

void test_aisis_init_with_config(void) {
    printf("\n=== Testing AISIS Configuration ===\n");
    
    // Test with valid configuration
    aisis_config_t config = {
        .buffer_size = 512,
        .timeout_ms = 3000,
        .debug_mode = true,
        .device_name = "TestDevice"
    };
    
    int result = aisis_init_with_config(&config);
    TEST_ASSERT_EQUAL(AISIS_SUCCESS, result, "Init with valid config should succeed");
    
    // Cleanup
    aisis_cleanup();
    
    // Test with invalid configuration (NULL)
    result = aisis_init_with_config(NULL);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Init with NULL config should fail");
    
    // Test with invalid buffer size
    config.buffer_size = 0;
    result = aisis_init_with_config(&config);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Init with zero buffer size should fail");
    
    // Test with buffer size too large
    config.buffer_size = AISIS_MAX_BUFFER_SIZE + 1;
    result = aisis_init_with_config(&config);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Init with oversized buffer should fail");
    
    // Test with invalid timeout
    config.buffer_size = 512;
    config.timeout_ms = 0;
    result = aisis_init_with_config(&config);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Init with zero timeout should fail");
}

void test_aisis_version(void) {
    printf("\n=== Testing AISIS Version ===\n");
    
    const char* version = aisis_get_version();
    TEST_ASSERT_NOT_NULL(version, "Version string should not be NULL");
    TEST_ASSERT(strlen(version) > 0, "Version string should not be empty");
    printf("Library version: %s\n", version);
}

void test_aisis_run(void) {
    printf("\n=== Testing AISIS Run ===\n");
    
    // Test run without initialization
    int result = aisis_run();
    TEST_ASSERT_EQUAL(AISIS_ERROR_INIT, result, "Run without init should fail");
    
    // Test run with initialization
    aisis_init();
    result = aisis_run();
    TEST_ASSERT_EQUAL(AISIS_SUCCESS, result, "Run after init should succeed");
    
    aisis_cleanup();
}

void test_device_management(void) {
    printf("\n=== Testing Device Management ===\n");
    
    aisis_init();
    
    // Test device scanning
    aisis_device_info_t devices[AISIS_MAX_DEVICES];
    int device_count = aisis_scan_devices(devices, AISIS_MAX_DEVICES);
    TEST_ASSERT(device_count > 0, "Should find at least one device");
    TEST_ASSERT(device_count <= AISIS_MAX_DEVICES, "Should not exceed max devices");
    
    printf("Found %d devices\n", device_count);
    
    // Test device scanning with invalid parameters
    int result = aisis_scan_devices(NULL, 10);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Scan with NULL buffer should fail");
    
    result = aisis_scan_devices(devices, 0);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Scan with zero max devices should fail");
    
    // Test device connection
    if (device_count > 0) {
        result = aisis_connect_device(devices[0].device_id);
        TEST_ASSERT_EQUAL(AISIS_SUCCESS, result, "Connect to valid device should succeed");
        
        // Test disconnect
        result = aisis_disconnect_device();
        TEST_ASSERT_EQUAL(AISIS_SUCCESS, result, "Disconnect should succeed");
    }
    
    // Test connection to invalid device
    result = aisis_connect_device(9999);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Connect to invalid device should fail");
    
    aisis_cleanup();
}

void test_data_processing(void) {
    printf("\n=== Testing Data Processing ===\n");
    
    aisis_init();
    
    // Test data processing with valid data
    uint8_t input[] = {0x01, 0x02, 0x03, 0x04, 0x05};
    uint8_t output[16];
    
    int processed = aisis_process_data(input, sizeof(input), output, sizeof(output));
    TEST_ASSERT_EQUAL(sizeof(input), processed, "Should process all input bytes");
    
    // Verify data transformation
    bool data_transformed = false;
    for (size_t i = 0; i < sizeof(input); i++) {
        if (output[i] != input[i]) {
            data_transformed = true;
            break;
        }
    }
    TEST_ASSERT(data_transformed, "Data should be transformed during processing");
    
    // Test with invalid parameters
    int result = aisis_process_data(NULL, sizeof(input), output, sizeof(output));
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Process with NULL input should fail");
    
    result = aisis_process_data(input, sizeof(input), NULL, sizeof(output));
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Process with NULL output should fail");
    
    result = aisis_process_data(input, 0, output, sizeof(output));
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Process with zero input size should fail");
    
    result = aisis_process_data(input, sizeof(input), output, 0);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Process with zero output size should fail");
    
    aisis_cleanup();
}

void test_command_interface(void) {
    printf("\n=== Testing Command Interface ===\n");
    
    aisis_init();
    
    // Test command sending
    uint8_t command[] = {0xAA, 0xBB, 0xCC, 0xDD};
    int result = aisis_send_command(command, sizeof(command));
    TEST_ASSERT_EQUAL(AISIS_SUCCESS, result, "Send valid command should succeed");
    
    // Test response reading
    uint8_t response[16];
    int bytes_read = aisis_read_response(response, sizeof(response));
    TEST_ASSERT(bytes_read > 0, "Should read response bytes");
    TEST_ASSERT(bytes_read <= sizeof(response), "Should not exceed buffer size");
    
    // Test with invalid parameters
    result = aisis_send_command(NULL, sizeof(command));
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Send NULL command should fail");
    
    result = aisis_send_command(command, 0);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, result, "Send zero-length command should fail");
    
    bytes_read = aisis_read_response(NULL, sizeof(response));
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, bytes_read, "Read to NULL buffer should fail");
    
    bytes_read = aisis_read_response(response, 0);
    TEST_ASSERT_EQUAL(AISIS_ERROR_INVALID, bytes_read, "Read with zero buffer size should fail");
    
    aisis_cleanup();
}

void test_error_conditions(void) {
    printf("\n=== Testing Error Conditions ===\n");
    
    // Test operations without initialization
    aisis_status_t status = aisis_get_status();
    TEST_ASSERT_EQUAL(AISIS_STATUS_UNKNOWN, status, "Status without init should be UNKNOWN");
    
    // Test cleanup without initialization (should not crash)
    aisis_cleanup();
    TEST_ASSERT(1, "Cleanup without init should not crash");
    
    // Test large buffer handling
    aisis_init();
    
    uint8_t large_buffer[AISIS_MAX_BUFFER_SIZE + 100];
    memset(large_buffer, 0xAA, sizeof(large_buffer));
    
    int result = aisis_send_command(large_buffer, sizeof(large_buffer));
    TEST_ASSERT_EQUAL(AISIS_ERROR_MEMORY, result, "Send oversized command should fail");
    
    uint8_t output[16];
    result = aisis_process_data(large_buffer, sizeof(large_buffer), output, sizeof(output));
    TEST_ASSERT_EQUAL(AISIS_ERROR_MEMORY, result, "Process oversized data should fail");
    
    aisis_cleanup();
}

void print_test_summary(void) {
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("TEST SUMMARY\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Total tests: %d\n", tests_run);
    printf("Passed:      %d\n", tests_passed);
    printf("Failed:      %d\n", tests_failed);
    printf("Success rate: %.1f%%\n", tests_run > 0 ? (100.0 * tests_passed / tests_run) : 0.0);
    
    if (tests_failed == 0) {
        printf("\n🎉 ALL TESTS PASSED! 🎉\n");
    } else {
        printf("\n❌ %d TESTS FAILED ❌\n", tests_failed);
    }
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
}

int main(void) {
    printf("AISIS Library Test Suite\n");
    printf("========================\n");
    
    // Run all tests
    test_aisis_version();
    test_aisis_init();
    test_aisis_init_with_config();
    test_aisis_run();
    test_device_management();
    test_data_processing();
    test_command_interface();
    test_error_conditions();
    
    // Print summary
    print_test_summary();
    
    // Return appropriate exit code
    return (tests_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}