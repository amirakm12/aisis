#ifndef ULTIMATE_CORE_H
#define ULTIMATE_CORE_H

/**
 * @file ultimate_core.h
 * @brief ULTIMATE System Core Header
 * @version 1.0.0
 * @date 2024
 * 
 * Main core header for the ULTIMATE Windows system.
 * Provides essential system definitions, types, and core functionality.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Standard includes */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Core system includes */
#include "ultimate_types.h"
#include "ultimate_config.h"
#include "ultimate_errors.h"
#include "ultimate_memory.h"
#include "ultimate_system.h"

/* Version information */
#define ULTIMATE_VERSION_MAJOR    1
#define ULTIMATE_VERSION_MINOR    0
#define ULTIMATE_VERSION_PATCH    0
#define ULTIMATE_VERSION_STRING   "1.0.0"

/* System constants */
#define ULTIMATE_MAX_TASKS        32
#define ULTIMATE_MAX_QUEUES       16
#define ULTIMATE_MAX_TIMERS       64
#define ULTIMATE_STACK_SIZE       4096

/* Core system states */
typedef enum {
    ULTIMATE_STATE_UNINITIALIZED = 0,
    ULTIMATE_STATE_INITIALIZING,
    ULTIMATE_STATE_READY,
    ULTIMATE_STATE_RUNNING,
    ULTIMATE_STATE_ERROR,
    ULTIMATE_STATE_SHUTDOWN
} ultimate_state_t;

/* System initialization structure */
typedef struct {
    uint32_t cpu_frequency;
    uint32_t tick_frequency;
    uint16_t max_tasks;
    uint16_t max_queues;
    bool enable_watchdog;
    bool enable_debug;
} ultimate_init_config_t;

/* Core API functions */
ultimate_error_t ultimate_init(const ultimate_init_config_t* config);
ultimate_error_t ultimate_start(void);
ultimate_error_t ultimate_stop(void);
ultimate_error_t ultimate_shutdown(void);
ultimate_state_t ultimate_get_state(void);
uint32_t ultimate_get_version(void);
const char* ultimate_get_version_string(void);

/* System tick and timing */
uint32_t ultimate_get_tick_count(void);
uint32_t ultimate_get_time_ms(void);
void ultimate_delay_ms(uint32_t ms);
void ultimate_delay_us(uint32_t us);

/* Critical section management */
void ultimate_enter_critical(void);
void ultimate_exit_critical(void);

/* System reset and recovery */
void ultimate_system_reset(void);
void ultimate_system_recovery(void);

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_CORE_H */ 