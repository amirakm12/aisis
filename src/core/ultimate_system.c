#include "ultimate_system.h"
#include "ultimate_core.h"
#include "ultimate_memory.h"
#include "ultimate_errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#pragma comment(lib, "psapi.lib")
#else
#include <pthread.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#endif

/* Task management structures */
typedef struct ultimate_task {
    ultimate_task_config_t config;
    ultimate_task_function_t function;
    void* params;
    ultimate_task_state_t state;
    uint32_t id;
    uint32_t stack_usage;
    uint32_t cpu_time;
    uint32_t last_run_time;
#ifdef _WIN32
    HANDLE thread_handle;
    DWORD thread_id;
#else
    pthread_t thread;
#endif
} ultimate_task_t;

/* Queue management structures */
typedef struct ultimate_queue {
    void** items;
    uint32_t max_items;
    uint32_t item_size;
    uint32_t head;
    uint32_t tail;
    uint32_t count;
    const char* name;
#ifdef _WIN32
    CRITICAL_SECTION lock;
    HANDLE not_empty;
    HANDLE not_full;
#else
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
#endif
} ultimate_queue_t;

/* Timer management structures */
typedef struct ultimate_timer {
    uint32_t id;
    uint32_t interval_ms;
    uint32_t remaining_ms;
    bool periodic;
    bool active;
    ultimate_timer_callback_t callback;
    void* user_data;
    const char* name;
} ultimate_timer_t;

/* Global system state */
static ultimate_task_t g_tasks[ULTIMATE_MAX_TASKS];
static uint32_t g_task_count = 0;
static uint32_t g_next_task_id = 1;

static ultimate_queue_t g_queues[ULTIMATE_MAX_QUEUES];
static uint32_t g_queue_count = 0;

static ultimate_timer_t g_timers[ULTIMATE_MAX_TIMERS];
static uint32_t g_timer_count = 0;
static uint32_t g_next_timer_id = 1;

static ultimate_system_stats_t g_system_stats = {0};
static bool g_system_initialized = false;

/* Forward declarations */
#ifdef _WIN32
static DWORD WINAPI task_thread_proc(LPVOID lpParameter);
#else
static void* task_thread_proc(void* arg);
#endif

/* System initialization */
ultimate_error_t ultimate_system_init(void) {
    if (g_system_initialized) {
        return ULTIMATE_ERROR_ALREADY_INITIALIZED;
    }
    
    /* Initialize task array */
    memset(g_tasks, 0, sizeof(g_tasks));
    g_task_count = 0;
    g_next_task_id = 1;
    
    /* Initialize queue array */
    memset(g_queues, 0, sizeof(g_queues));
    g_queue_count = 0;
    
    /* Initialize timer array */
    memset(g_timers, 0, sizeof(g_timers));
    g_timer_count = 0;
    g_next_timer_id = 1;
    
    /* Initialize system statistics */
    memset(&g_system_stats, 0, sizeof(g_system_stats));
    
    g_system_initialized = true;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_system_shutdown(void) {
    if (!g_system_initialized) {
        return ULTIMATE_ERROR_NOT_INITIALIZED;
    }
    
    /* Stop all tasks */
    for (uint32_t i = 0; i < g_task_count; i++) {
        if (g_tasks[i].state != ULTIMATE_TASK_STATE_TERMINATED) {
            ultimate_task_terminate(g_tasks[i].id);
        }
    }
    
    /* Destroy all queues */
    for (uint32_t i = 0; i < g_queue_count; i++) {
        ultimate_queue_destroy(&g_queues[i]);
    }
    
    /* Stop all timers */
    for (uint32_t i = 0; i < g_timer_count; i++) {
        g_timers[i].active = false;
    }
    
    g_system_initialized = false;
    return ULTIMATE_OK;
}

/* Task management functions */
ultimate_error_t ultimate_task_create(ultimate_task_function_t function,
                                     void* params,
                                     const ultimate_task_config_t* config,
                                     ultimate_task_handle_t* handle) {
    if (!function || !config || !handle || g_task_count >= ULTIMATE_MAX_TASKS) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_task_t* task = &g_tasks[g_task_count];
    memset(task, 0, sizeof(ultimate_task_t));
    
    task->config = *config;
    task->function = function;
    task->params = params;
    task->state = ULTIMATE_TASK_STATE_READY;
    task->id = g_next_task_id++;
    
#ifdef _WIN32
    task->thread_handle = CreateThread(
        NULL,
        config->stack_size,
        task_thread_proc,
        task,
        CREATE_SUSPENDED,
        &task->thread_id
    );
    
    if (!task->thread_handle) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    /* Set thread priority */
    int win_priority = THREAD_PRIORITY_NORMAL;
    switch (config->priority) {
        case ULTIMATE_PRIORITY_IDLE: win_priority = THREAD_PRIORITY_IDLE; break;
        case ULTIMATE_PRIORITY_LOW: win_priority = THREAD_PRIORITY_BELOW_NORMAL; break;
        case ULTIMATE_PRIORITY_NORMAL: win_priority = THREAD_PRIORITY_NORMAL; break;
        case ULTIMATE_PRIORITY_HIGH: win_priority = THREAD_PRIORITY_ABOVE_NORMAL; break;
        case ULTIMATE_PRIORITY_REALTIME: win_priority = THREAD_PRIORITY_TIME_CRITICAL; break;
    }
    SetThreadPriority(task->thread_handle, win_priority);
    
    if (config->auto_start) {
        ResumeThread(task->thread_handle);
        task->state = ULTIMATE_TASK_STATE_RUNNING;
    }
#else
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, config->stack_size);
    
    int result = pthread_create(&task->thread, &attr, task_thread_proc, task);
    pthread_attr_destroy(&attr);
    
    if (result != 0) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    if (config->auto_start) {
        task->state = ULTIMATE_TASK_STATE_RUNNING;
    }
#endif
    
    *handle = (ultimate_task_handle_t)task;
    g_task_count++;
    g_system_stats.active_tasks++;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_task_start(ultimate_task_handle_t handle) {
    ultimate_task_t* task = (ultimate_task_t*)handle;
    if (!task || task->state != ULTIMATE_TASK_STATE_READY) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    ResumeThread(task->thread_handle);
#endif
    
    task->state = ULTIMATE_TASK_STATE_RUNNING;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_task_suspend(ultimate_task_handle_t handle) {
    ultimate_task_t* task = (ultimate_task_t*)handle;
    if (!task || task->state != ULTIMATE_TASK_STATE_RUNNING) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    SuspendThread(task->thread_handle);
#endif
    
    task->state = ULTIMATE_TASK_STATE_SUSPENDED;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_task_resume(ultimate_task_handle_t handle) {
    ultimate_task_t* task = (ultimate_task_t*)handle;
    if (!task || task->state != ULTIMATE_TASK_STATE_SUSPENDED) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    ResumeThread(task->thread_handle);
#endif
    
    task->state = ULTIMATE_TASK_STATE_RUNNING;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_task_terminate(ultimate_task_handle_t handle) {
    ultimate_task_t* task = (ultimate_task_t*)handle;
    if (!task) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    TerminateThread(task->thread_handle, 0);
    CloseHandle(task->thread_handle);
#else
    pthread_cancel(task->thread);
    pthread_join(task->thread, NULL);
#endif
    
    task->state = ULTIMATE_TASK_STATE_TERMINATED;
    g_system_stats.active_tasks--;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_task_sleep(uint32_t ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_task_yield(void) {
#ifdef _WIN32
    Sleep(0);
#else
    sched_yield();
#endif
    return ULTIMATE_OK;
}

ultimate_task_state_t ultimate_task_get_state(ultimate_task_handle_t handle) {
    ultimate_task_t* task = (ultimate_task_t*)handle;
    return task ? task->state : ULTIMATE_TASK_STATE_TERMINATED;
}

uint32_t ultimate_task_get_id(ultimate_task_handle_t handle) {
    ultimate_task_t* task = (ultimate_task_t*)handle;
    return task ? task->id : 0;
}

/* Queue management functions */
ultimate_error_t ultimate_queue_create(uint32_t max_items, uint32_t item_size,
                                      const char* name, ultimate_queue_handle_t* handle) {
    if (!handle || g_queue_count >= ULTIMATE_MAX_QUEUES) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_queue_t* queue = &g_queues[g_queue_count];
    memset(queue, 0, sizeof(ultimate_queue_t));
    
    queue->items = (void**)ultimate_malloc(max_items * sizeof(void*));
    if (!queue->items) {
        return ULTIMATE_ERROR_OUT_OF_MEMORY;
    }
    
    for (uint32_t i = 0; i < max_items; i++) {
        queue->items[i] = ultimate_malloc(item_size);
        if (!queue->items[i]) {
            /* Cleanup on failure */
            for (uint32_t j = 0; j < i; j++) {
                ultimate_free(queue->items[j]);
            }
            ultimate_free(queue->items);
            return ULTIMATE_ERROR_OUT_OF_MEMORY;
        }
    }
    
    queue->max_items = max_items;
    queue->item_size = item_size;
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    queue->name = name;
    
#ifdef _WIN32
    InitializeCriticalSection(&queue->lock);
    queue->not_empty = CreateEvent(NULL, FALSE, FALSE, NULL);
    queue->not_full = CreateEvent(NULL, FALSE, TRUE, NULL);
#else
    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
#endif
    
    *handle = (ultimate_queue_handle_t)queue;
    g_queue_count++;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_queue_send(ultimate_queue_handle_t handle, const void* item,
                                    uint32_t timeout_ms) {
    ultimate_queue_t* queue = (ultimate_queue_t*)handle;
    if (!queue || !item) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    EnterCriticalSection(&queue->lock);
    
    if (queue->count >= queue->max_items) {
        LeaveCriticalSection(&queue->lock);
        DWORD result = WaitForSingleObject(queue->not_full, timeout_ms);
        if (result != WAIT_OBJECT_0) {
            return ULTIMATE_ERROR_TIMEOUT;
        }
        EnterCriticalSection(&queue->lock);
    }
    
    memcpy(queue->items[queue->tail], item, queue->item_size);
    queue->tail = (queue->tail + 1) % queue->max_items;
    queue->count++;
    
    SetEvent(queue->not_empty);
    LeaveCriticalSection(&queue->lock);
#else
    pthread_mutex_lock(&queue->lock);
    
    while (queue->count >= queue->max_items) {
        pthread_cond_wait(&queue->not_full, &queue->lock);
    }
    
    memcpy(queue->items[queue->tail], item, queue->item_size);
    queue->tail = (queue->tail + 1) % queue->max_items;
    queue->count++;
    
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->lock);
#endif
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_queue_receive(ultimate_queue_handle_t handle, void* item,
                                       uint32_t timeout_ms) {
    ultimate_queue_t* queue = (ultimate_queue_t*)handle;
    if (!queue || !item) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    EnterCriticalSection(&queue->lock);
    
    if (queue->count == 0) {
        LeaveCriticalSection(&queue->lock);
        DWORD result = WaitForSingleObject(queue->not_empty, timeout_ms);
        if (result != WAIT_OBJECT_0) {
            return ULTIMATE_ERROR_TIMEOUT;
        }
        EnterCriticalSection(&queue->lock);
    }
    
    memcpy(item, queue->items[queue->head], queue->item_size);
    queue->head = (queue->head + 1) % queue->max_items;
    queue->count--;
    
    SetEvent(queue->not_full);
    LeaveCriticalSection(&queue->lock);
#else
    pthread_mutex_lock(&queue->lock);
    
    while (queue->count == 0) {
        pthread_cond_wait(&queue->not_empty, &queue->lock);
    }
    
    memcpy(item, queue->items[queue->head], queue->item_size);
    queue->head = (queue->head + 1) % queue->max_items;
    queue->count--;
    
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->lock);
#endif
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_queue_destroy(ultimate_queue_handle_t handle) {
    ultimate_queue_t* queue = (ultimate_queue_t*)handle;
    if (!queue) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    DeleteCriticalSection(&queue->lock);
    CloseHandle(queue->not_empty);
    CloseHandle(queue->not_full);
#else
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
#endif
    
    for (uint32_t i = 0; i < queue->max_items; i++) {
        ultimate_free(queue->items[i]);
    }
    ultimate_free(queue->items);
    
    return ULTIMATE_OK;
}

uint32_t ultimate_queue_get_count(ultimate_queue_handle_t handle) {
    ultimate_queue_t* queue = (ultimate_queue_t*)handle;
    return queue ? queue->count : 0;
}

/* Timer management functions */
ultimate_error_t ultimate_timer_create(uint32_t interval_ms, bool periodic,
                                      ultimate_timer_callback_t callback,
                                      void* user_data, const char* name,
                                      ultimate_timer_handle_t* handle) {
    if (!callback || !handle || g_timer_count >= ULTIMATE_MAX_TIMERS) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_timer_t* timer = &g_timers[g_timer_count];
    memset(timer, 0, sizeof(ultimate_timer_t));
    
    timer->id = g_next_timer_id++;
    timer->interval_ms = interval_ms;
    timer->remaining_ms = interval_ms;
    timer->periodic = periodic;
    timer->active = false;
    timer->callback = callback;
    timer->user_data = user_data;
    timer->name = name;
    
    *handle = (ultimate_timer_handle_t)timer;
    g_timer_count++;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_timer_start(ultimate_timer_handle_t handle) {
    ultimate_timer_t* timer = (ultimate_timer_t*)handle;
    if (!timer) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    timer->active = true;
    timer->remaining_ms = timer->interval_ms;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_timer_stop(ultimate_timer_handle_t handle) {
    ultimate_timer_t* timer = (ultimate_timer_t*)handle;
    if (!timer) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    timer->active = false;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_timer_reset(ultimate_timer_handle_t handle) {
    ultimate_timer_t* timer = (ultimate_timer_t*)handle;
    if (!timer) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    timer->remaining_ms = timer->interval_ms;
    return ULTIMATE_OK;
}

uint32_t ultimate_timer_get_remaining(ultimate_timer_handle_t handle) {
    ultimate_timer_t* timer = (ultimate_timer_t*)handle;
    return timer ? timer->remaining_ms : 0;
}

bool ultimate_timer_is_active(ultimate_timer_handle_t handle) {
    ultimate_timer_t* timer = (ultimate_timer_t*)handle;
    return timer ? timer->active : false;
}

/* System statistics */
ultimate_error_t ultimate_system_get_stats(ultimate_system_stats_t* stats) {
    if (!stats) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    g_system_stats.uptime_ms = ultimate_get_time_ms();
    g_system_stats.active_tasks = g_task_count;
    
    /* Get memory statistics */
    ultimate_memory_get_stats(&g_system_stats.memory_info);
    
#ifdef _WIN32
    /* Get CPU usage on Windows */
    FILETIME idle_time, kernel_time, user_time;
    if (GetSystemTimes(&idle_time, &kernel_time, &user_time)) {
        /* Calculate CPU usage percentage */
        static ULARGE_INTEGER last_idle = {0}, last_kernel = {0}, last_user = {0};
        
        ULARGE_INTEGER idle, kernel, user;
        idle.LowPart = idle_time.dwLowDateTime;
        idle.HighPart = idle_time.dwHighDateTime;
        kernel.LowPart = kernel_time.dwLowDateTime;
        kernel.HighPart = kernel_time.dwHighDateTime;
        user.LowPart = user_time.dwLowDateTime;
        user.HighPart = user_time.dwHighDateTime;
        
        if (last_idle.QuadPart != 0) {
            ULONGLONG idle_diff = idle.QuadPart - last_idle.QuadPart;
            ULONGLONG kernel_diff = kernel.QuadPart - last_kernel.QuadPart;
            ULONGLONG user_diff = user.QuadPart - last_user.QuadPart;
            ULONGLONG total_diff = kernel_diff + user_diff;
            
            if (total_diff > 0) {
                g_system_stats.cpu_usage_percent = (uint32_t)(100 - (idle_diff * 100) / total_diff);
            }
        }
        
        last_idle = idle;
        last_kernel = kernel;
        last_user = user;
    }
#else
    /* Get CPU usage on Linux */
    FILE* stat_file = fopen("/proc/stat", "r");
    if (stat_file) {
        unsigned long user, nice, system, idle;
        if (fscanf(stat_file, "cpu %lu %lu %lu %lu", &user, &nice, &system, &idle) == 4) {
            static unsigned long last_total = 0, last_idle = 0;
            unsigned long total = user + nice + system + idle;
            
            if (last_total != 0) {
                unsigned long total_diff = total - last_total;
                unsigned long idle_diff = idle - last_idle;
                
                if (total_diff > 0) {
                    g_system_stats.cpu_usage_percent = (uint32_t)(100 - (idle_diff * 100) / total_diff);
                }
            }
            
            last_total = total;
            last_idle = idle;
        }
        fclose(stat_file);
    }
#endif
    
    *stats = g_system_stats;
    return ULTIMATE_OK;
}

/* Process and service management */
ultimate_error_t ultimate_process_create(const char* executable, const char* args,
                                        uint32_t* process_id) {
    if (!executable || !process_id) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);
    
    char command_line[1024];
    snprintf(command_line, sizeof(command_line), "%s %s", executable, args ? args : "");
    
    if (!CreateProcessA(NULL, command_line, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    *process_id = pi.dwProcessId;
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
#else
    pid_t pid = fork();
    if (pid == 0) {
        /* Child process */
        if (args) {
            execl(executable, executable, args, (char*)NULL);
        } else {
            execl(executable, executable, (char*)NULL);
        }
        exit(1);
    } else if (pid > 0) {
        *process_id = (uint32_t)pid;
    } else {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#endif
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_process_terminate(uint32_t process_id) {
#ifdef _WIN32
    HANDLE process = OpenProcess(PROCESS_TERMINATE, FALSE, process_id);
    if (!process) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    BOOL result = TerminateProcess(process, 0);
    CloseHandle(process);
    
    return result ? ULTIMATE_OK : ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
#else
    return (kill((pid_t)process_id, SIGTERM) == 0) ? ULTIMATE_OK : ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
#endif
}

bool ultimate_process_is_running(uint32_t process_id) {
#ifdef _WIN32
    HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, process_id);
    if (!process) {
        return false;
    }
    
    DWORD exit_code;
    BOOL result = GetExitCodeProcess(process, &exit_code);
    CloseHandle(process);
    
    return result && (exit_code == STILL_ACTIVE);
#else
    return (kill((pid_t)process_id, 0) == 0);
#endif
}

#ifdef _WIN32
/* Windows service management */
ultimate_error_t ultimate_service_install(const char* service_name,
                                         const char* display_name,
                                         const char* executable_path) {
    if (!service_name || !display_name || !executable_path) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    SC_HANDLE scm = OpenSCManager(NULL, NULL, SC_MANAGER_CREATE_SERVICE);
    if (!scm) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    SC_HANDLE service = CreateServiceA(
        scm,
        service_name,
        display_name,
        SERVICE_ALL_ACCESS,
        SERVICE_WIN32_OWN_PROCESS,
        SERVICE_DEMAND_START,
        SERVICE_ERROR_NORMAL,
        executable_path,
        NULL, NULL, NULL, NULL, NULL
    );
    
    CloseServiceHandle(scm);
    
    if (!service) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    CloseServiceHandle(service);
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_service_start(const char* service_name) {
    if (!service_name) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    SC_HANDLE scm = OpenSCManager(NULL, NULL, SC_MANAGER_CONNECT);
    if (!scm) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    SC_HANDLE service = OpenServiceA(scm, service_name, SERVICE_START);
    if (!service) {
        CloseServiceHandle(scm);
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    BOOL result = StartService(service, 0, NULL);
    
    CloseServiceHandle(service);
    CloseServiceHandle(scm);
    
    return result ? ULTIMATE_OK : ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
}

ultimate_error_t ultimate_service_stop(const char* service_name) {
    if (!service_name) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    SC_HANDLE scm = OpenSCManager(NULL, NULL, SC_MANAGER_CONNECT);
    if (!scm) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    SC_HANDLE service = OpenServiceA(scm, service_name, SERVICE_STOP);
    if (!service) {
        CloseServiceHandle(scm);
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    
    SERVICE_STATUS status;
    BOOL result = ControlService(service, SERVICE_CONTROL_STOP, &status);
    
    CloseServiceHandle(service);
    CloseServiceHandle(scm);
    
    return result ? ULTIMATE_OK : ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
}
#endif

/* Thread procedure implementations */
#ifdef _WIN32
static DWORD WINAPI task_thread_proc(LPVOID lpParameter) {
    ultimate_task_t* task = (ultimate_task_t*)lpParameter;
    if (task && task->function) {
        task->function(task->params);
    }
    return 0;
}
#else
static void* task_thread_proc(void* arg) {
    ultimate_task_t* task = (ultimate_task_t*)arg;
    if (task && task->function) {
        task->function(task->params);
    }
    return NULL;
}
#endif