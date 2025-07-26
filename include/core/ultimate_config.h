#ifndef ULTIMATE_CONFIG_H
#define ULTIMATE_CONFIG_H

/**
 * @file ultimate_config.h
 * @brief ULTIMATE System Configuration
 * @version 1.0.0
 * @date 2024
 * 
 * System-wide configuration options and compile-time settings.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* System configuration */
#ifndef ULTIMATE_CPU_FREQUENCY_HZ
#define ULTIMATE_CPU_FREQUENCY_HZ           180000000UL  /* 180 MHz */
#endif

#ifndef ULTIMATE_TICK_FREQUENCY_HZ
#define ULTIMATE_TICK_FREQUENCY_HZ          1000UL       /* 1 kHz (1ms tick) */
#endif

#ifndef ULTIMATE_MAX_TASKS
#define ULTIMATE_MAX_TASKS                  16
#endif

#ifndef ULTIMATE_MAX_QUEUES
#define ULTIMATE_MAX_QUEUES                 8
#endif

#ifndef ULTIMATE_MAX_TIMERS
#define ULTIMATE_MAX_TIMERS                 32
#endif

#ifndef ULTIMATE_MAX_MUTEXES
#define ULTIMATE_MAX_MUTEXES                16
#endif

#ifndef ULTIMATE_MAX_SEMAPHORES
#define ULTIMATE_MAX_SEMAPHORES             16
#endif

/* Memory configuration */
#ifndef ULTIMATE_HEAP_SIZE
#define ULTIMATE_HEAP_SIZE                  (64 * 1024)  /* 64KB heap */
#endif

#ifndef ULTIMATE_STACK_SIZE_DEFAULT
#define ULTIMATE_STACK_SIZE_DEFAULT         2048         /* 2KB default stack */
#endif

#ifndef ULTIMATE_STACK_SIZE_MIN
#define ULTIMATE_STACK_SIZE_MIN             512          /* 512B minimum stack */
#endif

#ifndef ULTIMATE_STACK_SIZE_MAX
#define ULTIMATE_STACK_SIZE_MAX             8192         /* 8KB maximum stack */
#endif

/* Task configuration */
#ifndef ULTIMATE_TASK_NAME_MAX_LEN
#define ULTIMATE_TASK_NAME_MAX_LEN          16
#endif

#ifndef ULTIMATE_TASK_PRIORITY_LEVELS
#define ULTIMATE_TASK_PRIORITY_LEVELS       5
#endif

#ifndef ULTIMATE_TASK_QUANTUM_MS
#define ULTIMATE_TASK_QUANTUM_MS            10           /* 10ms time slice */
#endif

/* Communication configuration */
#ifndef ULTIMATE_QUEUE_MAX_MESSAGES
#define ULTIMATE_QUEUE_MAX_MESSAGES         32
#endif

#ifndef ULTIMATE_MESSAGE_MAX_SIZE
#define ULTIMATE_MESSAGE_MAX_SIZE           256
#endif

#ifndef ULTIMATE_TIMEOUT_INFINITE
#define ULTIMATE_TIMEOUT_INFINITE           0xFFFFFFFFUL
#endif

#ifndef ULTIMATE_TIMEOUT_IMMEDIATE
#define ULTIMATE_TIMEOUT_IMMEDIATE          0UL
#endif

/* Debug and logging configuration */
#ifndef ULTIMATE_DEBUG_ENABLED
#define ULTIMATE_DEBUG_ENABLED              1
#endif

#ifndef ULTIMATE_LOG_LEVEL
#define ULTIMATE_LOG_LEVEL                  3            /* 0=None, 1=Error, 2=Warning, 3=Info, 4=Debug */
#endif

#ifndef ULTIMATE_LOG_BUFFER_SIZE
#define ULTIMATE_LOG_BUFFER_SIZE            1024
#endif

/* Hardware configuration */
#ifndef ULTIMATE_GPIO_PORTS_MAX
#define ULTIMATE_GPIO_PORTS_MAX             8
#endif

#ifndef ULTIMATE_GPIO_PINS_PER_PORT
#define ULTIMATE_GPIO_PINS_PER_PORT         16
#endif

#ifndef ULTIMATE_UART_INSTANCES_MAX
#define ULTIMATE_UART_INSTANCES_MAX         4
#endif

#ifndef ULTIMATE_SPI_INSTANCES_MAX
#define ULTIMATE_SPI_INSTANCES_MAX          3
#endif

#ifndef ULTIMATE_I2C_INSTANCES_MAX
#define ULTIMATE_I2C_INSTANCES_MAX          3
#endif

/* Watchdog configuration */
#ifndef ULTIMATE_WATCHDOG_ENABLED
#define ULTIMATE_WATCHDOG_ENABLED           1
#endif

#ifndef ULTIMATE_WATCHDOG_TIMEOUT_MS
#define ULTIMATE_WATCHDOG_TIMEOUT_MS        5000         /* 5 second timeout */
#endif

/* Performance monitoring */
#ifndef ULTIMATE_PROFILING_ENABLED
#define ULTIMATE_PROFILING_ENABLED          1
#endif

#ifndef ULTIMATE_STATS_UPDATE_INTERVAL_MS
#define ULTIMATE_STATS_UPDATE_INTERVAL_MS   1000         /* Update stats every 1 second */
#endif

/* Power management */
#ifndef ULTIMATE_POWER_MANAGEMENT_ENABLED
#define ULTIMATE_POWER_MANAGEMENT_ENABLED   1
#endif

#ifndef ULTIMATE_IDLE_POWER_MODE
#define ULTIMATE_IDLE_POWER_MODE            1            /* 0=None, 1=Sleep, 2=Deep Sleep */
#endif

/* Safety and reliability */
#ifndef ULTIMATE_STACK_OVERFLOW_CHECK
#define ULTIMATE_STACK_OVERFLOW_CHECK       1
#endif

#ifndef ULTIMATE_MEMORY_CORRUPTION_CHECK
#define ULTIMATE_MEMORY_CORRUPTION_CHECK    1
#endif

#ifndef ULTIMATE_ASSERT_ENABLED
#define ULTIMATE_ASSERT_ENABLED             1
#endif

/* Feature flags */
#ifndef ULTIMATE_NEURAL_ENABLED
#define ULTIMATE_NEURAL_ENABLED             1
#endif

#ifndef ULTIMATE_FILESYSTEM_ENABLED
#define ULTIMATE_FILESYSTEM_ENABLED         0
#endif

#ifndef ULTIMATE_NETWORK_ENABLED
#define ULTIMATE_NETWORK_ENABLED            0
#endif

#ifndef ULTIMATE_USB_ENABLED
#define ULTIMATE_USB_ENABLED                0
#endif

/* Compiler and platform specific */
#if defined(__GNUC__)
    #define ULTIMATE_COMPILER_GCC           1
    #define ULTIMATE_INLINE                 __inline__
    #define ULTIMATE_FORCE_INLINE           __attribute__((always_inline))
    #define ULTIMATE_NO_INLINE              __attribute__((noinline))
    #define ULTIMATE_PACKED                 __attribute__((packed))
    #define ULTIMATE_ALIGNED(x)             __attribute__((aligned(x)))
#elif defined(__ARMCC_VERSION)
    #define ULTIMATE_COMPILER_ARMCC         1
    #define ULTIMATE_INLINE                 __inline
    #define ULTIMATE_FORCE_INLINE           __forceinline
    #define ULTIMATE_NO_INLINE              __attribute__((noinline))
    #define ULTIMATE_PACKED                 __packed
    #define ULTIMATE_ALIGNED(x)             __attribute__((aligned(x)))
#else
    #define ULTIMATE_INLINE                 inline
    #define ULTIMATE_FORCE_INLINE           inline
    #define ULTIMATE_NO_INLINE              
    #define ULTIMATE_PACKED                 
    #define ULTIMATE_ALIGNED(x)             
#endif

/* Assert macro */
#if ULTIMATE_ASSERT_ENABLED
    #define ULTIMATE_ASSERT(expr) \
        do { \
            if (!(expr)) { \
                ultimate_assert_failed(__FILE__, __LINE__, #expr); \
            } \
        } while(0)
#else
    #define ULTIMATE_ASSERT(expr) ((void)0)
#endif

/* Debug print macro */
#if ULTIMATE_DEBUG_ENABLED
    #define ULTIMATE_DEBUG_PRINT(fmt, ...) ultimate_debug_printf(fmt, ##__VA_ARGS__)
#else
    #define ULTIMATE_DEBUG_PRINT(fmt, ...) ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* ULTIMATE_CONFIG_H */