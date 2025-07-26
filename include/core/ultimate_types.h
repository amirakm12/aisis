#ifndef ULTIMATE_TYPES_H
#define ULTIMATE_TYPES_H

/**
 * @file ultimate_types.h
 * @brief ULTIMATE System Core Types
 * @version 1.0.0
 * @date 2024
 * 
 * Core type definitions for the ULTIMATE embedded system.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Forward declaration of error type */
typedef int32_t ultimate_error_t;

/* Basic type aliases */
typedef uint8_t   u8;
typedef uint16_t  u16;
typedef uint32_t  u32;
typedef uint64_t  u64;
typedef int8_t    s8;
typedef int16_t   s16;
typedef int32_t   s32;
typedef int64_t   s64;
typedef float     f32;
typedef double    f64;

/* System handle types */
typedef void* ultimate_handle_t;
typedef uint32_t ultimate_id_t;
typedef uint32_t ultimate_priority_t;
typedef uint32_t ultimate_timeout_t;

/* Task and thread types */
typedef struct ultimate_task* ultimate_task_handle_t;
typedef void (*ultimate_task_func_t)(void* params);

typedef enum {
    ULTIMATE_PRIORITY_IDLE = 0,
    ULTIMATE_PRIORITY_LOW = 1,
    ULTIMATE_PRIORITY_NORMAL = 2,
    ULTIMATE_PRIORITY_HIGH = 3,
    ULTIMATE_PRIORITY_CRITICAL = 4,
    ULTIMATE_PRIORITY_MAX = 5
} ultimate_priority_level_t;

/* Memory and buffer types */
typedef struct {
    void* data;
    size_t size;
    size_t capacity;
    bool is_allocated;
} ultimate_buffer_t;

typedef struct {
    void* start_addr;
    size_t total_size;
    size_t used_size;
    size_t free_size;
    uint32_t block_count;
} ultimate_memory_info_t;

/* Communication types */
typedef struct ultimate_queue* ultimate_queue_handle_t;
typedef struct ultimate_mutex* ultimate_mutex_handle_t;
typedef struct ultimate_semaphore* ultimate_semaphore_handle_t;

typedef enum {
    ULTIMATE_MSG_TYPE_DATA = 0,
    ULTIMATE_MSG_TYPE_COMMAND,
    ULTIMATE_MSG_TYPE_EVENT,
    ULTIMATE_MSG_TYPE_ERROR,
    ULTIMATE_MSG_TYPE_STATUS
} ultimate_message_type_t;

typedef struct {
    ultimate_message_type_t type;
    ultimate_id_t sender_id;
    ultimate_id_t receiver_id;
    uint32_t timestamp;
    size_t data_size;
    void* data;
} ultimate_message_t;

/* Timer types */
typedef struct ultimate_timer* ultimate_timer_handle_t;
typedef void (*ultimate_timer_callback_t)(ultimate_timer_handle_t timer, void* user_data);

typedef enum {
    ULTIMATE_TIMER_ONE_SHOT = 0,
    ULTIMATE_TIMER_PERIODIC
} ultimate_timer_type_t;

/* GPIO and hardware types */
typedef enum {
    ULTIMATE_GPIO_INPUT = 0,
    ULTIMATE_GPIO_OUTPUT,
    ULTIMATE_GPIO_ALTERNATE,
    ULTIMATE_GPIO_ANALOG
} ultimate_gpio_mode_t;

typedef enum {
    ULTIMATE_GPIO_LOW = 0,
    ULTIMATE_GPIO_HIGH = 1
} ultimate_gpio_state_t;

typedef struct {
    uint32_t port;
    uint32_t pin;
    ultimate_gpio_mode_t mode;
    ultimate_gpio_state_t initial_state;
} ultimate_gpio_config_t;

/* Callback function types */
typedef void (*ultimate_callback_t)(void* user_data);
typedef void (*ultimate_error_callback_t)(ultimate_error_t error, const char* message, void* user_data);
typedef bool (*ultimate_event_handler_t)(uint32_t event_id, void* event_data, void* user_data);

/* System statistics */
typedef struct {
    uint32_t uptime_ms;
    uint32_t total_tasks;
    uint32_t active_tasks;
    uint32_t cpu_usage_percent;
    ultimate_memory_info_t memory_info;
    uint32_t context_switches;
    uint32_t interrupts_handled;
} ultimate_system_stats_t;

/* Configuration structures */
typedef struct {
    ultimate_priority_level_t priority;
    size_t stack_size;
    const char* name;
    bool auto_start;
    ultimate_timeout_t watchdog_timeout;
} ultimate_task_config_t;

typedef struct {
    size_t max_messages;
    size_t message_size;
    ultimate_timeout_t timeout;
    const char* name;
} ultimate_queue_config_t;

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_TYPES_H */