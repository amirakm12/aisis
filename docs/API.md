# AISIS Library API Documentation

## Overview

The AISIS library provides a comprehensive API for embedded device management, data processing, and communication. This document describes all available functions, data structures, and usage patterns.

## Table of Contents

1. [Core Functions](#core-functions)
2. [Device Management](#device-management)
3. [Data Processing](#data-processing)
4. [Data Structures](#data-structures)
5. [Error Codes](#error-codes)
6. [Constants](#constants)
7. [Usage Examples](#usage-examples)

## Core Functions

### aisis_init()

Initialize the AISIS library with default configuration.

**Syntax:**
```c
int aisis_init(void);
```

**Returns:**
- `AISIS_SUCCESS` (0) on success
- `AISIS_ERROR_INIT` (-1) on initialization failure

**Example:**
```c
if (aisis_init() != AISIS_SUCCESS) {
    fprintf(stderr, "Failed to initialize AISIS library\n");
    return -1;
}
```

### aisis_init_with_config()

Initialize the AISIS library with custom configuration.

**Syntax:**
```c
int aisis_init_with_config(const aisis_config_t *config);
```

**Parameters:**
- `config`: Pointer to configuration structure

**Returns:**
- `AISIS_SUCCESS` (0) on success
- `AISIS_ERROR_INVALID` (-2) if config is NULL or invalid
- `AISIS_ERROR_INIT` (-1) on initialization failure

**Example:**
```c
aisis_config_t config = {
    .buffer_size = 2048,
    .timeout_ms = 10000,
    .debug_mode = true,
    .device_name = "MyDevice"
};

if (aisis_init_with_config(&config) != AISIS_SUCCESS) {
    fprintf(stderr, "Failed to initialize AISIS with custom config\n");
    return -1;
}
```

### aisis_run()

Execute the main AISIS processing loop.

**Syntax:**
```c
int aisis_run(void);
```

**Returns:**
- `AISIS_SUCCESS` (0) on successful completion
- `AISIS_ERROR_INIT` (-1) if library not initialized

**Example:**
```c
int result = aisis_run();
if (result != AISIS_SUCCESS) {
    fprintf(stderr, "AISIS execution failed: %d\n", result);
}
```

### aisis_cleanup()

Cleanup and shutdown the AISIS library.

**Syntax:**
```c
void aisis_cleanup(void);
```

**Example:**
```c
aisis_cleanup(); // Always call before program exit
```

### aisis_get_version()

Get the AISIS library version string.

**Syntax:**
```c
const char* aisis_get_version(void);
```

**Returns:**
- Pointer to version string (e.g., "1.0.0")

**Example:**
```c
printf("AISIS Library Version: %s\n", aisis_get_version());
```

### aisis_get_status()

Get the current AISIS library status.

**Syntax:**
```c
aisis_status_t aisis_get_status(void);
```

**Returns:**
- Current status (see `aisis_status_t` enum)

**Example:**
```c
aisis_status_t status = aisis_get_status();
switch (status) {
    case AISIS_STATUS_IDLE:
        printf("AISIS is idle\n");
        break;
    case AISIS_STATUS_RUNNING:
        printf("AISIS is running\n");
        break;
    // ... handle other statuses
}
```

## Device Management

### aisis_scan_devices()

Scan for available devices.

**Syntax:**
```c
int aisis_scan_devices(aisis_device_info_t *devices, uint32_t max_devices);
```

**Parameters:**
- `devices`: Array to store device information
- `max_devices`: Maximum number of devices to scan

**Returns:**
- Number of devices found (>= 0)
- `AISIS_ERROR_INVALID` (-2) if parameters are invalid

**Example:**
```c
aisis_device_info_t devices[AISIS_MAX_DEVICES];
int device_count = aisis_scan_devices(devices, AISIS_MAX_DEVICES);

if (device_count > 0) {
    printf("Found %d devices:\n", device_count);
    for (int i = 0; i < device_count; i++) {
        printf("  Device %u: %s\n", devices[i].device_id, devices[i].name);
    }
}
```

### aisis_connect_device()

Connect to a specific device.

**Syntax:**
```c
int aisis_connect_device(uint32_t device_id);
```

**Parameters:**
- `device_id`: ID of the device to connect to

**Returns:**
- `AISIS_SUCCESS` (0) on successful connection
- `AISIS_ERROR_INVALID` (-2) if device ID is invalid
- `AISIS_ERROR_IO` (-4) if device is in error state

**Example:**
```c
if (aisis_connect_device(devices[0].device_id) == AISIS_SUCCESS) {
    printf("Connected to device %u\n", devices[0].device_id);
}
```

### aisis_disconnect_device()

Disconnect from the currently connected device.

**Syntax:**
```c
int aisis_disconnect_device(void);
```

**Returns:**
- `AISIS_SUCCESS` (0) on successful disconnection

**Example:**
```c
aisis_disconnect_device();
printf("Disconnected from device\n");
```

## Data Processing

### aisis_process_data()

Process data through the AISIS pipeline.

**Syntax:**
```c
int aisis_process_data(const uint8_t *input, uint32_t input_size, 
                       uint8_t *output, uint32_t output_size);
```

**Parameters:**
- `input`: Input data buffer
- `input_size`: Size of input data in bytes
- `output`: Output data buffer
- `output_size`: Size of output buffer in bytes

**Returns:**
- Number of bytes processed (>= 0)
- `AISIS_ERROR_INVALID` (-2) if parameters are invalid
- `AISIS_ERROR_MEMORY` (-3) if buffer sizes exceed limits

**Example:**
```c
uint8_t input_data[] = {0x01, 0x02, 0x03, 0x04};
uint8_t output_data[16];

int processed = aisis_process_data(input_data, sizeof(input_data), 
                                  output_data, sizeof(output_data));
if (processed > 0) {
    printf("Processed %d bytes\n", processed);
}
```

### aisis_send_command()

Send a command to the connected device.

**Syntax:**
```c
int aisis_send_command(const uint8_t *command, uint32_t command_size);
```

**Parameters:**
- `command`: Command data buffer
- `command_size`: Size of command in bytes

**Returns:**
- `AISIS_SUCCESS` (0) on successful transmission
- `AISIS_ERROR_INVALID` (-2) if parameters are invalid
- `AISIS_ERROR_MEMORY` (-3) if command size exceeds buffer limit

**Example:**
```c
uint8_t command[] = {0xAA, 0xBB, 0xCC, 0xDD};
if (aisis_send_command(command, sizeof(command)) == AISIS_SUCCESS) {
    printf("Command sent successfully\n");
}
```

### aisis_read_response()

Read response from the connected device.

**Syntax:**
```c
int aisis_read_response(uint8_t *response, uint32_t response_size);
```

**Parameters:**
- `response`: Response data buffer
- `response_size`: Size of response buffer in bytes

**Returns:**
- Number of bytes read (>= 0)
- `AISIS_ERROR_INVALID` (-2) if parameters are invalid

**Example:**
```c
uint8_t response[64];
int bytes_read = aisis_read_response(response, sizeof(response));
if (bytes_read > 0) {
    printf("Received %d bytes\n", bytes_read);
}
```

## Data Structures

### aisis_config_t

Configuration structure for AISIS initialization.

```c
typedef struct {
    uint32_t buffer_size;    // Buffer size (1 to AISIS_MAX_BUFFER_SIZE)
    uint32_t timeout_ms;     // Timeout in milliseconds (> 0)
    bool debug_mode;         // Enable debug output
    char device_name[64];    // Device name string
} aisis_config_t;
```

### aisis_device_info_t

Device information structure.

```c
typedef struct {
    uint32_t device_id;      // Unique device identifier
    aisis_status_t status;   // Current device status
    char name[32];           // Device name
    uint32_t version;        // Device firmware version
} aisis_device_info_t;
```

### aisis_status_t

Status enumeration for AISIS library and devices.

```c
typedef enum {
    AISIS_STATUS_UNKNOWN = 0,  // Status unknown
    AISIS_STATUS_IDLE,         // Idle state
    AISIS_STATUS_RUNNING,      // Running/active state
    AISIS_STATUS_ERROR,        // Error state
    AISIS_STATUS_STOPPED       // Stopped state
} aisis_status_t;
```

## Error Codes

| Code | Name | Value | Description |
|------|------|-------|-------------|
| `AISIS_SUCCESS` | Success | 0 | Operation completed successfully |
| `AISIS_ERROR_INIT` | Initialization Error | -1 | Library initialization failed |
| `AISIS_ERROR_INVALID` | Invalid Parameter | -2 | Invalid parameter provided |
| `AISIS_ERROR_MEMORY` | Memory Error | -3 | Memory allocation or buffer error |
| `AISIS_ERROR_IO` | I/O Error | -4 | Input/output operation failed |
| `AISIS_ERROR_TIMEOUT` | Timeout Error | -5 | Operation timed out |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `AISIS_VERSION_MAJOR` | 1 | Major version number |
| `AISIS_VERSION_MINOR` | 0 | Minor version number |
| `AISIS_VERSION_PATCH` | 0 | Patch version number |
| `AISIS_MAX_BUFFER_SIZE` | 1024 | Maximum buffer size in bytes |
| `AISIS_MAX_DEVICES` | 16 | Maximum number of devices |
| `AISIS_TIMEOUT_MS` | 5000 | Default timeout in milliseconds |

## Usage Examples

### Basic Usage Pattern

```c
#include "aisis.h"

int main() {
    // Initialize library
    if (aisis_init() != AISIS_SUCCESS) {
        return -1;
    }
    
    // Scan for devices
    aisis_device_info_t devices[AISIS_MAX_DEVICES];
    int count = aisis_scan_devices(devices, AISIS_MAX_DEVICES);
    
    if (count > 0) {
        // Connect to first device
        aisis_connect_device(devices[0].device_id);
        
        // Send command and read response
        uint8_t cmd[] = {0x01, 0x02};
        uint8_t resp[32];
        
        aisis_send_command(cmd, sizeof(cmd));
        int bytes = aisis_read_response(resp, sizeof(resp));
        
        // Process some data
        uint8_t input[] = {0x10, 0x20, 0x30};
        uint8_t output[16];
        aisis_process_data(input, sizeof(input), output, sizeof(output));
        
        // Disconnect
        aisis_disconnect_device();
    }
    
    // Cleanup
    aisis_cleanup();
    return 0;
}
```

### Advanced Configuration

```c
#include "aisis.h"

int main() {
    // Custom configuration
    aisis_config_t config = {
        .buffer_size = 2048,
        .timeout_ms = 10000,
        .debug_mode = true,
        .device_name = "AdvancedDevice"
    };
    
    // Initialize with custom config
    if (aisis_init_with_config(&config) != AISIS_SUCCESS) {
        fprintf(stderr, "Initialization failed\n");
        return -1;
    }
    
    // Check status
    aisis_status_t status = aisis_get_status();
    printf("AISIS Status: %d\n", status);
    
    // Run main processing
    int result = aisis_run();
    if (result != AISIS_SUCCESS) {
        fprintf(stderr, "Processing failed: %d\n", result);
    }
    
    aisis_cleanup();
    return result;
}
```

### Error Handling

```c
#include "aisis.h"

int handle_aisis_error(int error_code) {
    switch (error_code) {
        case AISIS_SUCCESS:
            return 0;
        case AISIS_ERROR_INIT:
            fprintf(stderr, "AISIS initialization error\n");
            break;
        case AISIS_ERROR_INVALID:
            fprintf(stderr, "Invalid parameter error\n");
            break;
        case AISIS_ERROR_MEMORY:
            fprintf(stderr, "Memory/buffer error\n");
            break;
        case AISIS_ERROR_IO:
            fprintf(stderr, "I/O operation error\n");
            break;
        case AISIS_ERROR_TIMEOUT:
            fprintf(stderr, "Operation timeout\n");
            break;
        default:
            fprintf(stderr, "Unknown error: %d\n", error_code);
            break;
    }
    return error_code;
}

int main() {
    int result = aisis_init();
    if (result != AISIS_SUCCESS) {
        return handle_aisis_error(result);
    }
    
    // ... rest of application
    
    aisis_cleanup();
    return 0;
}
```

## Thread Safety

The AISIS library is **not thread-safe** by default. If you need to use it in a multi-threaded environment, you must provide your own synchronization mechanisms (mutexes, semaphores, etc.).

## Memory Management

The AISIS library manages its own internal memory. Users are responsible for:
- Allocating buffers for data processing
- Managing device information arrays
- Ensuring buffer sizes don't exceed limits

The library does not perform dynamic memory allocation during normal operation, making it suitable for embedded environments with limited memory.