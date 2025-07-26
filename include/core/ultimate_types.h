#ifndef ULTIMATE_TYPES_H
#define ULTIMATE_TYPES_H

/**
 * @file ultimate_types.h
 * @brief ULTIMATE System Type Definitions
 * @version 1.0.0
 * @date 2024
 * 
 * Core type definitions and constants for the ULTIMATE Windows system.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Standard includes */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Forward declarations */
typedef struct ultimate_task ultimate_task_t;
typedef struct ultimate_queue ultimate_queue_t;
typedef struct ultimate_timer ultimate_timer_t;
typedef struct ultimate_pool ultimate_pool_t;

/* Handle types */
typedef void* ultimate_task_handle_t;
typedef void* ultimate_queue_handle_t;
typedef void* ultimate_timer_handle_t;
typedef void* ultimate_pool_handle_t;
typedef void* ultimate_mutex_handle_t;
typedef void* ultimate_semaphore_handle_t;
typedef void* ultimate_window_handle_t;
typedef void* ultimate_socket_handle_t;
typedef void* ultimate_file_handle_t;
typedef void* ultimate_neural_model_t;

/* System constants */
#define ULTIMATE_MAX_POOLS            16
#define ULTIMATE_MAX_MUTEXES          32
#define ULTIMATE_MAX_SEMAPHORES       32
#define ULTIMATE_MAX_WINDOWS          16
#define ULTIMATE_MAX_SOCKETS          64
#define ULTIMATE_MAX_FILES            128
#define ULTIMATE_MAX_NEURAL_MODELS    8

/* Task priority levels */
typedef enum {
    ULTIMATE_PRIORITY_IDLE = 0,
    ULTIMATE_PRIORITY_LOW,
    ULTIMATE_PRIORITY_NORMAL,
    ULTIMATE_PRIORITY_HIGH,
    ULTIMATE_PRIORITY_REALTIME,
    ULTIMATE_PRIORITY_MAX
} ultimate_priority_t;

/* Task states */
typedef enum {
    ULTIMATE_TASK_STATE_READY = 0,
    ULTIMATE_TASK_STATE_RUNNING,
    ULTIMATE_TASK_STATE_BLOCKED,
    ULTIMATE_TASK_STATE_SUSPENDED,
    ULTIMATE_TASK_STATE_TERMINATED
} ultimate_task_state_t;

/* Power modes */
typedef enum {
    ULTIMATE_POWER_MODE_FULL = 0,
    ULTIMATE_POWER_MODE_BALANCED,
    ULTIMATE_POWER_MODE_POWER_SAVER,
    ULTIMATE_POWER_MODE_CUSTOM
} ultimate_power_mode_t;

/* Memory protection types */
typedef enum {
    ULTIMATE_MEMORY_PROTECTION_NONE = 0,
    ULTIMATE_MEMORY_PROTECTION_READ = 1,
    ULTIMATE_MEMORY_PROTECTION_WRITE = 2,
    ULTIMATE_MEMORY_PROTECTION_EXECUTE = 4
} ultimate_memory_protection_t;

/* Pool types */
typedef enum {
    ULTIMATE_POOL_TYPE_FIXED = 0,
    ULTIMATE_POOL_TYPE_VARIABLE,
    ULTIMATE_POOL_TYPE_RING_BUFFER
} ultimate_pool_type_t;

/* File modes */
typedef enum {
    ULTIMATE_FILE_MODE_READ = 0,
    ULTIMATE_FILE_MODE_WRITE,
    ULTIMATE_FILE_MODE_APPEND,
    ULTIMATE_FILE_MODE_READ_WRITE
} ultimate_file_mode_t;

/* Socket types */
typedef enum {
    ULTIMATE_SOCKET_TYPE_TCP = 0,
    ULTIMATE_SOCKET_TYPE_UDP,
    ULTIMATE_SOCKET_TYPE_RAW
} ultimate_socket_type_t;

/* Input event types */
typedef enum {
    ULTIMATE_INPUT_TYPE_KEYBOARD = 0,
    ULTIMATE_INPUT_TYPE_MOUSE,
    ULTIMATE_INPUT_TYPE_TOUCH,
    ULTIMATE_INPUT_TYPE_GAMEPAD
} ultimate_input_type_t;

/* Neural network types */
typedef enum {
    ULTIMATE_NEURAL_TYPE_FEEDFORWARD = 0,
    ULTIMATE_NEURAL_TYPE_CNN,
    ULTIMATE_NEURAL_TYPE_RNN,
    ULTIMATE_NEURAL_TYPE_LSTM,
    ULTIMATE_NEURAL_TYPE_TRANSFORMER
} ultimate_neural_type_t;

/* Neural precision types */
typedef enum {
    ULTIMATE_NEURAL_PRECISION_FLOAT32 = 0,
    ULTIMATE_NEURAL_PRECISION_FLOAT16,
    ULTIMATE_NEURAL_PRECISION_INT8,
    ULTIMATE_NEURAL_PRECISION_INT16
} ultimate_neural_precision_t;

/* Task configuration */
typedef struct {
    ultimate_priority_t priority;
    uint32_t stack_size;
    const char* name;
    bool auto_start;
    uint32_t watchdog_timeout;
} ultimate_task_config_t;

/* Pool configuration */
typedef struct {
    uint32_t total_size;
    uint32_t block_size;
    uint32_t max_blocks;
    ultimate_pool_type_t type;
    const char* name;
    bool thread_safe;
} ultimate_pool_config_t;

/* Pool statistics */
typedef struct {
    uint32_t total_blocks;
    uint32_t used_blocks;
    uint32_t free_blocks;
    uint32_t block_size;
    uint32_t total_size;
} ultimate_pool_stats_t;

/* Memory statistics */
typedef struct {
    uint64_t total_size;
    uint64_t allocated_size;
    uint64_t free_size;
    uint32_t allocation_count;
    uint32_t free_count;
    uint32_t peak_usage;
} ultimate_memory_stats_t;

/* System statistics */
typedef struct {
    uint32_t uptime_ms;
    uint32_t active_tasks;
    uint32_t cpu_usage_percent;
    ultimate_memory_stats_t memory_info;
    uint32_t interrupt_count;
    uint32_t context_switches;
} ultimate_system_stats_t;

/* Window configuration */
typedef struct {
    uint32_t width;
    uint32_t height;
    const char* title;
    bool resizable;
    bool fullscreen;
} ultimate_window_config_t;

/* Input event */
typedef struct {
    ultimate_input_type_t type;
    uint32_t timestamp;
    union {
        struct {
            uint32_t key_code;
            bool pressed;
        } keyboard;
        struct {
            int32_t x, y;
            uint32_t buttons;
        } mouse;
        struct {
            int32_t x, y;
            uint32_t id;
        } touch;
    };
} ultimate_input_event_t;

/* Neural configuration */
typedef struct {
    ultimate_neural_type_t model_type;
    ultimate_neural_precision_t precision;
    uint32_t memory_limit;
    const char* name;
} ultimate_neural_config_t;

/* Neural tensor */
typedef struct {
    void* data;
    uint32_t* shape;
    uint32_t ndim;
    ultimate_neural_precision_t precision;
    uint32_t size;
} ultimate_neural_tensor_t;

/* Callback function types */
typedef void (*ultimate_task_function_t)(void* params);
typedef void (*ultimate_timer_callback_t)(ultimate_timer_handle_t timer, void* user_data);
typedef void (*ultimate_input_callback_t)(ultimate_window_handle_t window, 
                                         const ultimate_input_event_t* event, 
                                         void* user_data);

/* Pool structure */
struct ultimate_pool {
    ultimate_pool_config_t config;
    void* memory;
    void** free_blocks;
    uint32_t free_count;
    uint32_t used_count;
};

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_TYPES_H */ 