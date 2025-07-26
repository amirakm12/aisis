#ifndef ULTIMATE_SYSTEM_H
#define ULTIMATE_SYSTEM_H

/**
 * @file ultimate_system.h
 * @brief ULTIMATE System Management
 * @version 1.0.0
 * @date 2024
 * 
 * System-level functionality including task management, scheduling,
 * inter-task communication, and system services for Windows.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "ultimate_config.h"
#include "ultimate_types.h"
#include "ultimate_errors.h"

/* Task management API */
ultimate_error_t ultimate_task_create(ultimate_task_func_t task_func,
                                     void* params,
                                     const ultimate_task_config_t* config,
                                     ultimate_task_handle_t* task_handle);

ultimate_error_t ultimate_task_delete(ultimate_task_handle_t task);
ultimate_error_t ultimate_task_start(ultimate_task_handle_t task);
ultimate_error_t ultimate_task_stop(ultimate_task_handle_t task);
ultimate_error_t ultimate_task_suspend(ultimate_task_handle_t task);
ultimate_error_t ultimate_task_resume(ultimate_task_handle_t task);

/* Task information and control */
ultimate_task_handle_t ultimate_task_get_current(void);
ultimate_id_t ultimate_task_get_id(ultimate_task_handle_t task);
const char* ultimate_task_get_name(ultimate_task_handle_t task);
ultimate_priority_level_t ultimate_task_get_priority(ultimate_task_handle_t task);
ultimate_error_t ultimate_task_set_priority(ultimate_task_handle_t task, 
                                           ultimate_priority_level_t priority);

/* Task synchronization */
void ultimate_task_yield(void);
void ultimate_task_sleep(uint32_t ms);
void ultimate_task_sleep_until(uint32_t wake_time);

/* Queue management */
ultimate_error_t ultimate_queue_create(const ultimate_queue_config_t* config,
                                      ultimate_queue_handle_t* queue);
ultimate_error_t ultimate_queue_delete(ultimate_queue_handle_t queue);

ultimate_error_t ultimate_queue_send(ultimate_queue_handle_t queue,
                                    const void* message,
                                    size_t message_size,
                                    ultimate_timeout_t timeout);

ultimate_error_t ultimate_queue_receive(ultimate_queue_handle_t queue,
                                       void* message,
                                       size_t max_size,
                                       size_t* received_size,
                                       ultimate_timeout_t timeout);

ultimate_error_t ultimate_queue_peek(ultimate_queue_handle_t queue,
                                    void* message,
                                    size_t max_size,
                                    size_t* message_size);

uint32_t ultimate_queue_get_count(ultimate_queue_handle_t queue);
bool ultimate_queue_is_full(ultimate_queue_handle_t queue);
bool ultimate_queue_is_empty(ultimate_queue_handle_t queue);

/* Mutex management */
ultimate_error_t ultimate_mutex_create(ultimate_mutex_handle_t* mutex);
ultimate_error_t ultimate_mutex_delete(ultimate_mutex_handle_t mutex);
ultimate_error_t ultimate_mutex_lock(ultimate_mutex_handle_t mutex, 
                                    ultimate_timeout_t timeout);
ultimate_error_t ultimate_mutex_unlock(ultimate_mutex_handle_t mutex);
bool ultimate_mutex_is_locked(ultimate_mutex_handle_t mutex);

/* Semaphore management */
ultimate_error_t ultimate_semaphore_create(uint32_t initial_count,
                                          uint32_t max_count,
                                          ultimate_semaphore_handle_t* semaphore);
ultimate_error_t ultimate_semaphore_delete(ultimate_semaphore_handle_t semaphore);
ultimate_error_t ultimate_semaphore_wait(ultimate_semaphore_handle_t semaphore,
                                        ultimate_timeout_t timeout);
ultimate_error_t ultimate_semaphore_post(ultimate_semaphore_handle_t semaphore);
uint32_t ultimate_semaphore_get_count(ultimate_semaphore_handle_t semaphore);

/* Timer management */
ultimate_error_t ultimate_timer_create(ultimate_timer_type_t type,
                                      uint32_t period_ms,
                                      ultimate_timer_callback_t callback,
                                      void* user_data,
                                      ultimate_timer_handle_t* timer);

ultimate_error_t ultimate_timer_delete(ultimate_timer_handle_t timer);
ultimate_error_t ultimate_timer_start(ultimate_timer_handle_t timer);
ultimate_error_t ultimate_timer_stop(ultimate_timer_handle_t timer);
ultimate_error_t ultimate_timer_reset(ultimate_timer_handle_t timer);
ultimate_error_t ultimate_timer_set_period(ultimate_timer_handle_t timer, 
                                          uint32_t period_ms);
bool ultimate_timer_is_active(ultimate_timer_handle_t timer);

/* Event system */
typedef uint32_t ultimate_event_mask_t;

ultimate_error_t ultimate_event_wait(ultimate_event_mask_t event_mask,
                                    bool wait_all,
                                    ultimate_timeout_t timeout,
                                    ultimate_event_mask_t* received_events);

ultimate_error_t ultimate_event_set(ultimate_task_handle_t task,
                                   ultimate_event_mask_t events);

ultimate_error_t ultimate_event_clear(ultimate_event_mask_t events);

/* System information */
typedef enum {
    ULTIMATE_TASK_STATE_READY = 0,
    ULTIMATE_TASK_STATE_RUNNING,
    ULTIMATE_TASK_STATE_BLOCKED,
    ULTIMATE_TASK_STATE_SUSPENDED,
    ULTIMATE_TASK_STATE_DELETED
} ultimate_task_state_t;

typedef struct {
    ultimate_id_t task_id;
    char name[ULTIMATE_TASK_NAME_MAX_LEN];
    ultimate_task_state_t state;
    ultimate_priority_level_t priority;
    size_t stack_size;
    size_t stack_used;
    uint32_t cpu_usage_percent;
    uint32_t runtime_ms;
} ultimate_task_info_t;

ultimate_error_t ultimate_system_get_task_list(ultimate_task_info_t* task_list,
                                              uint32_t max_tasks,
                                              uint32_t* task_count);

ultimate_error_t ultimate_system_get_task_info(ultimate_task_handle_t task,
                                              ultimate_task_info_t* info);

ultimate_error_t ultimate_system_get_stats(ultimate_system_stats_t* stats);

/* System control */
ultimate_error_t ultimate_system_suspend_all(void);
ultimate_error_t ultimate_system_resume_all(void);
ultimate_error_t ultimate_system_set_tick_hook(ultimate_callback_t hook);

/* Windows-specific system functions */
#ifdef _WIN32
ultimate_error_t ultimate_system_get_windows_info(void);
ultimate_error_t ultimate_system_set_process_priority(int priority);
ultimate_error_t ultimate_system_get_process_memory_info(void);
ultimate_error_t ultimate_system_set_affinity_mask(uint64_t mask);
#endif

/* Power management */
typedef enum {
    ULTIMATE_POWER_MODE_ACTIVE = 0,
    ULTIMATE_POWER_MODE_IDLE,
    ULTIMATE_POWER_MODE_SLEEP,
    ULTIMATE_POWER_MODE_DEEP_SLEEP,
    ULTIMATE_POWER_MODE_STANDBY
} ultimate_power_mode_t;

ultimate_error_t ultimate_power_set_mode(ultimate_power_mode_t mode);
ultimate_power_mode_t ultimate_power_get_mode(void);
ultimate_error_t ultimate_power_register_callback(ultimate_callback_t callback);

/* Watchdog management */
ultimate_error_t ultimate_watchdog_init(uint32_t timeout_ms);
ultimate_error_t ultimate_watchdog_start(void);
ultimate_error_t ultimate_watchdog_stop(void);
ultimate_error_t ultimate_watchdog_refresh(void);
ultimate_error_t ultimate_watchdog_set_timeout(uint32_t timeout_ms);

/* Critical section management */
typedef uint32_t ultimate_critical_state_t;

ultimate_critical_state_t ultimate_critical_enter(void);
void ultimate_critical_exit(ultimate_critical_state_t state);

/* System debugging and profiling */
typedef struct {
    uint32_t context_switches;
    uint32_t interrupts_handled;
    uint32_t tick_count;
    uint32_t idle_time_percent;
    uint32_t max_interrupt_latency_us;
    uint32_t avg_interrupt_latency_us;
} ultimate_system_debug_info_t;

ultimate_error_t ultimate_system_get_debug_info(ultimate_system_debug_info_t* info);
void ultimate_system_enable_profiling(bool enable);

/* System hooks and callbacks */
typedef enum {
    ULTIMATE_HOOK_IDLE = 0,
    ULTIMATE_HOOK_TICK,
    ULTIMATE_HOOK_TASK_SWITCH,
    ULTIMATE_HOOK_MALLOC,
    ULTIMATE_HOOK_FREE,
    ULTIMATE_HOOK_STACK_OVERFLOW,
    ULTIMATE_HOOK_MAX
} ultimate_hook_type_t;

ultimate_error_t ultimate_system_set_hook(ultimate_hook_type_t hook_type,
                                         ultimate_callback_t callback);

/* System configuration */
typedef struct {
    uint32_t max_tasks;
    uint32_t max_queues;
    uint32_t max_timers;
    uint32_t max_mutexes;
    uint32_t max_semaphores;
    uint32_t tick_rate_hz;
    bool enable_profiling;
    bool enable_stack_checking;
    bool enable_runtime_stats;
} ultimate_system_config_t;

ultimate_error_t ultimate_system_configure(const ultimate_system_config_t* config);

/* System utilities */
uint32_t ultimate_system_get_free_heap_size(void);
uint32_t ultimate_system_get_min_free_heap_size(void);
uint32_t ultimate_system_get_uptime_ms(void);
uint32_t ultimate_system_get_cpu_usage_percent(void);

/* Boot and initialization */
typedef struct {
    const char* version;
    uint32_t build_date;
    uint32_t build_time;
    const char* compiler;
    uint32_t heap_size;
    uint32_t stack_size;
} ultimate_system_info_t;

ultimate_error_t ultimate_system_get_info(ultimate_system_info_t* info);

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_SYSTEM_H */ 