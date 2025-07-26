#include "ultimate_core.h"
#include "ultimate_system.h"
#include "ultimate_memory.h"
#include "ultimate_errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#else
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#endif

/* Global system state */
static ultimate_state_t g_system_state = ULTIMATE_STATE_UNINITIALIZED;
static ultimate_init_config_t g_system_config;
static uint32_t g_system_start_time = 0;
static bool g_critical_section_entered = false;

/* System initialization */
ultimate_error_t ultimate_init(const ultimate_init_config_t* config) {
    if (config == NULL) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    if (g_system_state != ULTIMATE_STATE_UNINITIALIZED) {
        return ULTIMATE_ERROR_ALREADY_INITIALIZED;
    }
    
    g_system_state = ULTIMATE_STATE_INITIALIZING;
    
    /* Copy configuration */
    memcpy(&g_system_config, config, sizeof(ultimate_init_config_t));
    
    /* Initialize memory subsystem */
    ultimate_error_t error = ultimate_memory_init();
    if (error != ULTIMATE_OK) {
        g_system_state = ULTIMATE_STATE_ERROR;
        return error;
    }
    
    /* Initialize system subsystem */
    error = ultimate_system_init();
    if (error != ULTIMATE_OK) {
        g_system_state = ULTIMATE_STATE_ERROR;
        return error;
    }
    
    /* Initialize timing */
#ifdef _WIN32
    timeBeginPeriod(1);
#endif
    g_system_start_time = ultimate_get_tick_count();
    
    g_system_state = ULTIMATE_STATE_READY;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_start(void) {
    if (g_system_state != ULTIMATE_STATE_READY) {
        return ULTIMATE_ERROR_INVALID_STATE;
    }
    
    g_system_state = ULTIMATE_STATE_RUNNING;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_stop(void) {
    if (g_system_state != ULTIMATE_STATE_RUNNING) {
        return ULTIMATE_ERROR_INVALID_STATE;
    }
    
    g_system_state = ULTIMATE_STATE_READY;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_shutdown(void) {
    if (g_system_state == ULTIMATE_STATE_UNINITIALIZED) {
        return ULTIMATE_ERROR_INVALID_STATE;
    }
    
    /* Cleanup subsystems */
    ultimate_system_shutdown();
    ultimate_memory_shutdown();
    
#ifdef _WIN32
    timeEndPeriod(1);
#endif
    
    g_system_state = ULTIMATE_STATE_UNINITIALIZED;
    return ULTIMATE_OK;
}

ultimate_state_t ultimate_get_state(void) {
    return g_system_state;
}

uint32_t ultimate_get_version(void) {
    return (ULTIMATE_VERSION_MAJOR << 16) | 
           (ULTIMATE_VERSION_MINOR << 8) | 
           ULTIMATE_VERSION_PATCH;
}

const char* ultimate_get_version_string(void) {
    return ULTIMATE_VERSION_STRING;
}

/* System timing functions */
uint32_t ultimate_get_tick_count(void) {
#ifdef _WIN32
    return GetTickCount();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
#endif
}

uint32_t ultimate_get_time_ms(void) {
    return ultimate_get_tick_count() - g_system_start_time;
}

void ultimate_delay_ms(uint32_t ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
}

void ultimate_delay_us(uint32_t us) {
#ifdef _WIN32
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    while (1) {
        QueryPerformanceCounter(&end);
        uint64_t elapsed = ((end.QuadPart - start.QuadPart) * 1000000) / frequency.QuadPart;
        if (elapsed >= us) break;
    }
#else
    usleep(us);
#endif
}

/* Critical section management */
void ultimate_enter_critical(void) {
#ifdef _WIN32
    /* On Windows, we would use a critical section or mutex */
    /* For now, using a simple flag */
#endif
    g_critical_section_entered = true;
}

void ultimate_exit_critical(void) {
#ifdef _WIN32
    /* On Windows, we would leave the critical section */
#endif
    g_critical_section_entered = false;
}

/* System reset and recovery */
void ultimate_system_reset(void) {
    ultimate_shutdown();
    ultimate_init(&g_system_config);
    ultimate_start();
}

void ultimate_system_recovery(void) {
    if (g_system_state == ULTIMATE_STATE_ERROR) {
        ultimate_system_reset();
    }
}