#ifndef AISIS_H
#define AISIS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/**
 * @file aisis.h
 * @brief Main AISIS library header file
 * @version 1.0.0
 */

// Version information
#define AISIS_VERSION_MAJOR 1
#define AISIS_VERSION_MINOR 0
#define AISIS_VERSION_PATCH 0

// Error codes
#define AISIS_SUCCESS           0
#define AISIS_ERROR_INIT       -1
#define AISIS_ERROR_INVALID    -2
#define AISIS_ERROR_MEMORY     -3
#define AISIS_ERROR_IO         -4
#define AISIS_ERROR_TIMEOUT    -5

// Configuration constants
#define AISIS_MAX_BUFFER_SIZE  1024
#define AISIS_MAX_DEVICES      16
#define AISIS_TIMEOUT_MS       5000

/**
 * @brief AISIS device status enumeration
 */
typedef enum {
    AISIS_STATUS_UNKNOWN = 0,
    AISIS_STATUS_IDLE,
    AISIS_STATUS_RUNNING,
    AISIS_STATUS_ERROR,
    AISIS_STATUS_STOPPED
} aisis_status_t;

/**
 * @brief AISIS configuration structure
 */
typedef struct {
    uint32_t buffer_size;
    uint32_t timeout_ms;
    bool debug_mode;
    char device_name[64];
} aisis_config_t;

/**
 * @brief AISIS device information structure
 */
typedef struct {
    uint32_t device_id;
    aisis_status_t status;
    char name[32];
    uint32_t version;
} aisis_device_info_t;

// Core functions
/**
 * @brief Initialize the AISIS library
 * @return 0 on success, negative error code on failure
 */
int aisis_init(void);

/**
 * @brief Initialize AISIS with custom configuration
 * @param config Pointer to configuration structure
 * @return 0 on success, negative error code on failure
 */
int aisis_init_with_config(const aisis_config_t *config);

/**
 * @brief Run the main AISIS process
 * @return 0 on success, negative error code on failure
 */
int aisis_run(void);

/**
 * @brief Cleanup and shutdown AISIS library
 */
void aisis_cleanup(void);

/**
 * @brief Get AISIS library version
 * @return Version string
 */
const char* aisis_get_version(void);

/**
 * @brief Get current AISIS status
 * @return Current status
 */
aisis_status_t aisis_get_status(void);

// Device management functions
/**
 * @brief Scan for available devices
 * @param devices Array to store device information
 * @param max_devices Maximum number of devices to scan
 * @return Number of devices found, or negative error code
 */
int aisis_scan_devices(aisis_device_info_t *devices, uint32_t max_devices);

/**
 * @brief Connect to a specific device
 * @param device_id Device ID to connect to
 * @return 0 on success, negative error code on failure
 */
int aisis_connect_device(uint32_t device_id);

/**
 * @brief Disconnect from current device
 * @return 0 on success, negative error code on failure
 */
int aisis_disconnect_device(void);

// Data processing functions
/**
 * @brief Process data buffer
 * @param input Input data buffer
 * @param input_size Size of input data
 * @param output Output data buffer
 * @param output_size Size of output buffer
 * @return Number of bytes processed, or negative error code
 */
int aisis_process_data(const uint8_t *input, uint32_t input_size, 
                       uint8_t *output, uint32_t output_size);

/**
 * @brief Send command to device
 * @param command Command buffer
 * @param command_size Size of command
 * @return 0 on success, negative error code on failure
 */
int aisis_send_command(const uint8_t *command, uint32_t command_size);

/**
 * @brief Read response from device
 * @param response Response buffer
 * @param response_size Size of response buffer
 * @return Number of bytes read, or negative error code
 */
int aisis_read_response(uint8_t *response, uint32_t response_size);

#ifdef __cplusplus
}
#endif

#endif // AISIS_H