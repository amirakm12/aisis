#ifndef ULTIMATE_ERRORS_H
#define ULTIMATE_ERRORS_H

/**
 * @file ultimate_errors.h
 * @brief ULTIMATE System Error Codes and Handling
 * @version 1.0.0
 * @date 2024
 * 
 * Comprehensive error code definitions and error handling functionality for Windows.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* Error code type */
typedef int32_t ultimate_error_t;

/* Success code */
#define ULTIMATE_OK                         0

/* General error codes (1-99) */
#define ULTIMATE_ERROR_GENERIC              1
#define ULTIMATE_ERROR_INVALID_PARAMETER    2
#define ULTIMATE_ERROR_NULL_POINTER         3
#define ULTIMATE_ERROR_OUT_OF_MEMORY        4
#define ULTIMATE_ERROR_BUFFER_TOO_SMALL     5
#define ULTIMATE_ERROR_BUFFER_OVERFLOW      6
#define ULTIMATE_ERROR_NOT_INITIALIZED      7
#define ULTIMATE_ERROR_ALREADY_INITIALIZED  8
#define ULTIMATE_ERROR_NOT_SUPPORTED        9
#define ULTIMATE_ERROR_PERMISSION_DENIED    10
#define ULTIMATE_ERROR_RESOURCE_BUSY        11
#define ULTIMATE_ERROR_RESOURCE_UNAVAILABLE 12
#define ULTIMATE_ERROR_OPERATION_FAILED     13
#define ULTIMATE_ERROR_OPERATION_TIMEOUT    14
#define ULTIMATE_ERROR_OPERATION_CANCELLED  15
#define ULTIMATE_ERROR_INVALID_STATE        16
#define ULTIMATE_ERROR_INVALID_CONFIGURATION 17
#define ULTIMATE_ERROR_CHECKSUM_MISMATCH    18
#define ULTIMATE_ERROR_VERSION_MISMATCH     19
#define ULTIMATE_ERROR_CORRUPTED_DATA       20

/* System error codes (100-199) */
#define ULTIMATE_ERROR_SYSTEM_FAILURE       100
#define ULTIMATE_ERROR_SYSTEM_OVERLOAD      101
#define ULTIMATE_ERROR_SYSTEM_CALL_FAILED   102
#define ULTIMATE_ERROR_POOL_FULL            103
#define ULTIMATE_ERROR_TIMEOUT              104
#define ULTIMATE_ERROR_IO_ERROR             105
#define ULTIMATE_ERROR_FILE_NOT_FOUND       106
#define ULTIMATE_ERROR_SYSTEM_SHUTDOWN      102
#define ULTIMATE_ERROR_SYSTEM_RESET         103
#define ULTIMATE_ERROR_WATCHDOG_TIMEOUT     104
#define ULTIMATE_ERROR_STACK_OVERFLOW       105
#define ULTIMATE_ERROR_HEAP_CORRUPTION      106
#define ULTIMATE_ERROR_CRITICAL_SECTION     107
#define ULTIMATE_ERROR_INTERRUPT_FAILURE    108
#define ULTIMATE_ERROR_CLOCK_FAILURE        109
#define ULTIMATE_ERROR_POWER_FAILURE        110

/* Task and scheduling error codes (200-299) */
#define ULTIMATE_ERROR_TASK_CREATE_FAILED   200
#define ULTIMATE_ERROR_TASK_DELETE_FAILED   201
#define ULTIMATE_ERROR_TASK_START_FAILED    202
#define ULTIMATE_ERROR_TASK_STOP_FAILED     203
#define ULTIMATE_ERROR_TASK_SUSPEND_FAILED  204
#define ULTIMATE_ERROR_TASK_RESUME_FAILED   205
#define ULTIMATE_ERROR_TASK_NOT_FOUND       206
#define ULTIMATE_ERROR_TASK_PRIORITY_INVALID 207
#define ULTIMATE_ERROR_TASK_STACK_TOO_SMALL 208
#define ULTIMATE_ERROR_SCHEDULER_ERROR      209
#define ULTIMATE_ERROR_CONTEXT_SWITCH_FAILED 210

/* Communication error codes (300-399) */
#define ULTIMATE_ERROR_QUEUE_FULL           300
#define ULTIMATE_ERROR_QUEUE_EMPTY          301
#define ULTIMATE_ERROR_QUEUE_CREATE_FAILED  302
#define ULTIMATE_ERROR_QUEUE_DELETE_FAILED  303
#define ULTIMATE_ERROR_MESSAGE_TOO_LARGE    304
#define ULTIMATE_ERROR_MESSAGE_INVALID      305
#define ULTIMATE_ERROR_MUTEX_CREATE_FAILED  306
#define ULTIMATE_ERROR_MUTEX_LOCK_FAILED    307
#define ULTIMATE_ERROR_MUTEX_UNLOCK_FAILED  308
#define ULTIMATE_ERROR_SEMAPHORE_CREATE_FAILED 309
#define ULTIMATE_ERROR_SEMAPHORE_WAIT_FAILED 310
#define ULTIMATE_ERROR_SEMAPHORE_POST_FAILED 311

/* Timer error codes (400-499) */
#define ULTIMATE_ERROR_TIMER_CREATE_FAILED  400
#define ULTIMATE_ERROR_TIMER_DELETE_FAILED  401
#define ULTIMATE_ERROR_TIMER_START_FAILED   402
#define ULTIMATE_ERROR_TIMER_STOP_FAILED    403
#define ULTIMATE_ERROR_TIMER_NOT_FOUND      404
#define ULTIMATE_ERROR_TIMER_INVALID_PERIOD 405
#define ULTIMATE_ERROR_TIMER_CALLBACK_NULL  406

/* Windows-specific error codes (500-599) */
#define ULTIMATE_ERROR_WINDOW_CREATE_FAILED 500
#define ULTIMATE_ERROR_WINDOW_DESTROY_FAILED 501
#define ULTIMATE_ERROR_WINDOW_INVALID_HANDLE 502
#define ULTIMATE_ERROR_WINDOW_MESSAGE_FAILED 503
#define ULTIMATE_ERROR_WINDOW_REGISTER_FAILED 504
#define ULTIMATE_ERROR_WINDOW_CLASS_FAILED  505
#define ULTIMATE_ERROR_WINDOW_PAINT_FAILED  506
#define ULTIMATE_ERROR_WINDOW_RESIZE_FAILED 507
#define ULTIMATE_ERROR_WINDOW_FOCUS_FAILED  508
#define ULTIMATE_ERROR_WINDOW_SHOW_FAILED   509
#define ULTIMATE_ERROR_WINDOW_HIDE_FAILED   510

/* File system error codes (600-699) */
#define ULTIMATE_ERROR_FS_NOT_MOUNTED       600
#define ULTIMATE_ERROR_FS_MOUNT_FAILED      601
#define ULTIMATE_ERROR_FS_FILE_NOT_FOUND    602
#define ULTIMATE_ERROR_FS_FILE_CREATE_FAILED 603
#define ULTIMATE_ERROR_FS_FILE_READ_FAILED  604
#define ULTIMATE_ERROR_FS_FILE_WRITE_FAILED 605
#define ULTIMATE_ERROR_FS_DISK_FULL         606
#define ULTIMATE_ERROR_FS_PERMISSION_DENIED 607
#define ULTIMATE_ERROR_FS_PATH_TOO_LONG     608
#define ULTIMATE_ERROR_FS_ACCESS_DENIED     609

/* Network error codes (700-799) */
#define ULTIMATE_ERROR_NET_NOT_CONNECTED    700
#define ULTIMATE_ERROR_NET_CONNECTION_FAILED 701
#define ULTIMATE_ERROR_NET_SEND_FAILED      702
#define ULTIMATE_ERROR_NET_RECEIVE_FAILED   703
#define ULTIMATE_ERROR_NET_TIMEOUT          704
#define ULTIMATE_ERROR_NET_INVALID_ADDRESS  705
#define ULTIMATE_ERROR_NET_SOCKET_FAILED    706
#define ULTIMATE_ERROR_NET_BIND_FAILED      707
#define ULTIMATE_ERROR_NET_LISTEN_FAILED    708
#define ULTIMATE_ERROR_NET_ACCEPT_FAILED    709

/* Neural system error codes (800-899) */
#define ULTIMATE_ERROR_NEURAL_INIT_FAILED   800
#define ULTIMATE_ERROR_NEURAL_MODEL_INVALID 801
#define ULTIMATE_ERROR_NEURAL_INPUT_INVALID 802
#define ULTIMATE_ERROR_NEURAL_OUTPUT_INVALID 803
#define ULTIMATE_ERROR_NEURAL_COMPUTE_FAILED 804
#define ULTIMATE_ERROR_NEURAL_MEMORY_ERROR  805
#define ULTIMATE_ERROR_NEURAL_CALIBRATION_FAILED 806

/* Custom application error codes (900-999) */
#define ULTIMATE_ERROR_APP_CUSTOM_BASE      900

/* Error severity levels */
typedef enum {
    ULTIMATE_SEVERITY_INFO = 0,
    ULTIMATE_SEVERITY_WARNING,
    ULTIMATE_SEVERITY_ERROR,
    ULTIMATE_SEVERITY_CRITICAL,
    ULTIMATE_SEVERITY_FATAL
} ultimate_error_severity_t;

/* Error information structure */
typedef struct {
    ultimate_error_t code;
    ultimate_error_severity_t severity;
    uint32_t timestamp;
    const char* file;
    uint32_t line;
    const char* function;
    const char* message;
} ultimate_error_info_t;

/* Error callback function type */
typedef void (*ultimate_error_handler_t)(const ultimate_error_info_t* error_info);

/* Error handling API */
const char* ultimate_error_to_string(ultimate_error_t error);
ultimate_error_severity_t ultimate_error_get_severity(ultimate_error_t error);
bool ultimate_error_is_success(ultimate_error_t error);
bool ultimate_error_is_critical(ultimate_error_t error);

/* Error reporting and logging */
void ultimate_error_report(ultimate_error_t error, 
                          ultimate_error_severity_t severity,
                          const char* file, 
                          uint32_t line, 
                          const char* function, 
                          const char* message);

void ultimate_error_set_handler(ultimate_error_handler_t handler);
ultimate_error_handler_t ultimate_error_get_handler(void);

/* Error statistics */
typedef struct {
    uint32_t total_errors;
    uint32_t info_count;
    uint32_t warning_count;
    uint32_t error_count;
    uint32_t critical_count;
    uint32_t fatal_count;
    ultimate_error_t last_error;
    uint32_t last_error_timestamp;
} ultimate_error_stats_t;

void ultimate_error_get_stats(ultimate_error_stats_t* stats);
void ultimate_error_clear_stats(void);

/* Debug and assert functions */
void ultimate_assert_failed(const char* file, uint32_t line, const char* expression);
void ultimate_debug_printf(const char* format, ...);

/* Error macros for convenience */
#define ULTIMATE_ERROR_REPORT(error, severity, message) \
    ultimate_error_report(error, severity, __FILE__, __LINE__, __FUNCTION__, message)

#define ULTIMATE_ERROR_RETURN_IF_FAILED(expr) \
    do { \
        ultimate_error_t _err = (expr); \
        if (_err != ULTIMATE_OK) { \
            ULTIMATE_ERROR_REPORT(_err, ULTIMATE_SEVERITY_ERROR, #expr " failed"); \
            return _err; \
        } \
    } while(0)

#define ULTIMATE_ERROR_CHECK(expr) \
    do { \
        ultimate_error_t _err = (expr); \
        if (_err != ULTIMATE_OK) { \
            ULTIMATE_ERROR_REPORT(_err, ULTIMATE_SEVERITY_WARNING, #expr " failed"); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_ERRORS_H */ 