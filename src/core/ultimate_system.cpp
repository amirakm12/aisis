#include "ultimate_system.h"
#include "ultimate_errors.h"
#include "ultimate_memory.h"
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace ultimate {

struct Task {
    ultimate_task_handle_t handle;
    ultimate_task_func_t function;
    void* params;
    ultimate_task_config_t config;
    std::thread thread;
    std::atomic<bool> running;
    std::atomic<bool> should_stop;
    std::chrono::steady_clock::time_point last_activity;
    
    Task() : running(false), should_stop(false) {}
};

class SystemManager {
private:
    static std::unique_ptr<SystemManager> instance_;
    static std::mutex instance_mutex_;
    
    std::mutex system_mutex_;
    std::vector<std::unique_ptr<Task>> tasks_;
    std::atomic<ultimate_task_handle_t> next_task_handle_;
    
    bool initialized_;
    bool running_;
    ultimate_init_config_t config_;
    
    // Power management
    ultimate_power_mode_t current_power_mode_;
    std::atomic<uint32_t> cpu_usage_percent_;
    
    // System monitoring
    std::thread monitor_thread_;
    std::atomic<bool> monitor_running_;
    
    SystemManager() : next_task_handle_(1), initialized_(false), running_(false),
                     current_power_mode_(ULTIMATE_POWER_MODE_NORMAL),
                     cpu_usage_percent_(0), monitor_running_(false) {}

public:
    static SystemManager* getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::unique_ptr<SystemManager>(new SystemManager());
        }
        return instance_.get();
    }
    
    ultimate_error_t initialize(const ultimate_init_config_t* config) {
        if (initialized_) {
            return ULTIMATE_ERROR_ALREADY_INITIALIZED;
        }
        
        if (!config) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        config_ = *config;
        initialized_ = true;
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t start() {
        if (!initialized_) {
            return ULTIMATE_ERROR_NOT_INITIALIZED;
        }
        
        if (running_) {
            return ULTIMATE_ERROR_ALREADY_RUNNING;
        }
        
        running_ = true;
        
        // Start system monitor
        monitor_running_ = true;
        monitor_thread_ = std::thread(&SystemManager::monitorLoop, this);
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t stop() {
        if (!running_) {
            return ULTIMATE_ERROR_NOT_RUNNING;
        }
        
        // Stop all tasks
        std::lock_guard<std::mutex> lock(system_mutex_);
        
        for (auto& task : tasks_) {
            if (task->running.load()) {
                task->should_stop.store(true);
                if (task->thread.joinable()) {
                    task->thread.join();
                }
            }
        }
        
        // Stop monitor
        monitor_running_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
        
        running_ = false;
        return ULTIMATE_OK;
    }
    
    ultimate_error_t createTask(ultimate_task_func_t function, void* params,
                               const ultimate_task_config_t* config,
                               ultimate_task_handle_t* handle) {
        if (!initialized_ || !function || !config || !handle) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        std::lock_guard<std::mutex> lock(system_mutex_);
        
        if (tasks_.size() >= config_.max_tasks) {
            return ULTIMATE_ERROR_RESOURCE_EXHAUSTED;
        }
        
        auto task = std::make_unique<Task>();
        task->handle = next_task_handle_.fetch_add(1);
        task->function = function;
        task->params = params;
        task->config = *config;
        task->last_activity = std::chrono::steady_clock::now();
        
        *handle = task->handle;
        
        if (config->auto_start) {
            task->running.store(true);
            task->thread = std::thread(&SystemManager::taskWrapper, this, task.get());
        }
        
        tasks_.push_back(std::move(task));
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t startTask(ultimate_task_handle_t handle) {
        std::lock_guard<std::mutex> lock(system_mutex_);
        
        auto task = findTask(handle);
        if (!task) {
            return ULTIMATE_ERROR_TASK_NOT_FOUND;
        }
        
        if (task->running.load()) {
            return ULTIMATE_ERROR_TASK_ALREADY_RUNNING;
        }
        
        task->running.store(true);
        task->should_stop.store(false);
        task->thread = std::thread(&SystemManager::taskWrapper, this, task);
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t stopTask(ultimate_task_handle_t handle) {
        std::lock_guard<std::mutex> lock(system_mutex_);
        
        auto task = findTask(handle);
        if (!task) {
            return ULTIMATE_ERROR_TASK_NOT_FOUND;
        }
        
        if (!task->running.load()) {
            return ULTIMATE_ERROR_TASK_NOT_RUNNING;
        }
        
        task->should_stop.store(true);
        if (task->thread.joinable()) {
            task->thread.join();
        }
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t deleteTask(ultimate_task_handle_t handle) {
        std::lock_guard<std::mutex> lock(system_mutex_);
        
        auto it = std::find_if(tasks_.begin(), tasks_.end(),
            [handle](const std::unique_ptr<Task>& task) {
                return task->handle == handle;
            });
        
        if (it == tasks_.end()) {
            return ULTIMATE_ERROR_TASK_NOT_FOUND;
        }
        
        auto& task = *it;
        if (task->running.load()) {
            task->should_stop.store(true);
            if (task->thread.joinable()) {
                task->thread.join();
            }
        }
        
        tasks_.erase(it);
        return ULTIMATE_OK;
    }
    
    ultimate_error_t setPowerMode(ultimate_power_mode_t mode) {
        current_power_mode_ = mode;
        
#ifdef _WIN32
        SYSTEM_POWER_STATUS power_status;
        if (GetSystemPowerStatus(&power_status)) {
            // Adjust system behavior based on power mode
            switch (mode) {
                case ULTIMATE_POWER_MODE_LOW_POWER:
                    SetThreadExecutionState(ES_CONTINUOUS);
                    break;
                case ULTIMATE_POWER_MODE_HIGH_PERFORMANCE:
                    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
                    break;
                default:
                    SetThreadExecutionState(ES_CONTINUOUS);
                    break;
            }
        }
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_power_mode_t getPowerMode() const {
        return current_power_mode_;
    }
    
    ultimate_system_stats_t getStats() {
        ultimate_system_stats_t stats = {};
        
        std::lock_guard<std::mutex> lock(system_mutex_);
        
        stats.active_tasks = 0;
        for (const auto& task : tasks_) {
            if (task->running.load()) {
                stats.active_tasks++;
            }
        }
        
        stats.total_tasks = tasks_.size();
        stats.cpu_usage_percent = cpu_usage_percent_.load();
        stats.power_mode = current_power_mode_;
        
        // Get memory stats
        auto memory_stats = ultimate_memory_get_stats();
        stats.memory_info.total_size = memory_stats.heap_size;
        stats.memory_info.used_size = memory_stats.total_allocated;
        stats.memory_info.free_size = memory_stats.heap_size - memory_stats.total_allocated;
        stats.memory_info.peak_used = memory_stats.peak_allocated;
        
        // Calculate uptime
        static auto start_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        stats.uptime_ms = duration.count();
        
        return stats;
    }
    
    void deinitialize() {
        if (running_) {
            stop();
        }
        
        std::lock_guard<std::mutex> lock(system_mutex_);
        tasks_.clear();
        initialized_ = false;
    }

private:
    Task* findTask(ultimate_task_handle_t handle) {
        auto it = std::find_if(tasks_.begin(), tasks_.end(),
            [handle](const std::unique_ptr<Task>& task) {
                return task->handle == handle;
            });
        
        return (it != tasks_.end()) ? it->get() : nullptr;
    }
    
    void taskWrapper(Task* task) {
        if (!task) return;
        
        try {
            // Set thread priority based on task priority
#ifdef _WIN32
            int win_priority = THREAD_PRIORITY_NORMAL;
            switch (task->config.priority) {
                case ULTIMATE_PRIORITY_LOW:
                    win_priority = THREAD_PRIORITY_BELOW_NORMAL;
                    break;
                case ULTIMATE_PRIORITY_HIGH:
                    win_priority = THREAD_PRIORITY_ABOVE_NORMAL;
                    break;
                case ULTIMATE_PRIORITY_CRITICAL:
                    win_priority = THREAD_PRIORITY_TIME_CRITICAL;
                    break;
                default:
                    win_priority = THREAD_PRIORITY_NORMAL;
                    break;
            }
            SetThreadPriority(GetCurrentThread(), win_priority);
#endif
            
            // Run the task function
            task->function(task->params);
            
        } catch (...) {
            // Handle task exceptions
        }
        
        task->running.store(false);
    }
    
    void monitorLoop() {
        while (monitor_running_.load()) {
            updateSystemMetrics();
            
            // Check for task timeouts
            checkTaskWatchdogs();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    
    void updateSystemMetrics() {
#ifdef _WIN32
        static PDH_HQUERY query = nullptr;
        static PDH_HCOUNTER counter = nullptr;
        
        if (!query) {
            PdhOpenQuery(nullptr, 0, &query);
            PdhAddCounter(query, L"\\Processor(_Total)\\% Processor Time", 0, &counter);
        }
        
        if (query && counter) {
            PdhCollectQueryData(query);
            PDH_FMT_COUNTERVALUE value;
            if (PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, nullptr, &value) == ERROR_SUCCESS) {
                cpu_usage_percent_.store(static_cast<uint32_t>(value.doubleValue));
            }
        }
#else
        // Linux implementation would go here
        cpu_usage_percent_.store(0);
#endif
    }
    
    void checkTaskWatchdogs() {
        std::lock_guard<std::mutex> lock(system_mutex_);
        
        auto now = std::chrono::steady_clock::now();
        
        for (auto& task : tasks_) {
            if (task->running.load() && task->config.watchdog_timeout > 0) {
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - task->last_activity);
                
                if (duration.count() > task->config.watchdog_timeout) {
                    // Task timeout detected - could restart or report error
                    task->should_stop.store(true);
                }
            }
        }
    }
};

std::unique_ptr<SystemManager> SystemManager::instance_ = nullptr;
std::mutex SystemManager::instance_mutex_;

} // namespace ultimate

// C API Implementation
extern "C" {

ultimate_error_t ultimate_system_init(const ultimate_init_config_t* config) {
    return ultimate::SystemManager::getInstance()->initialize(config);
}

ultimate_error_t ultimate_system_start(void) {
    return ultimate::SystemManager::getInstance()->start();
}

ultimate_error_t ultimate_system_stop(void) {
    return ultimate::SystemManager::getInstance()->stop();
}

ultimate_error_t ultimate_task_create(ultimate_task_func_t function, void* params,
                                     const ultimate_task_config_t* config,
                                     ultimate_task_handle_t* handle) {
    return ultimate::SystemManager::getInstance()->createTask(function, params, config, handle);
}

ultimate_error_t ultimate_task_start(ultimate_task_handle_t handle) {
    return ultimate::SystemManager::getInstance()->startTask(handle);
}

ultimate_error_t ultimate_task_stop(ultimate_task_handle_t handle) {
    return ultimate::SystemManager::getInstance()->stopTask(handle);
}

ultimate_error_t ultimate_task_delete(ultimate_task_handle_t handle) {
    return ultimate::SystemManager::getInstance()->deleteTask(handle);
}

void ultimate_task_sleep(uint32_t milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

ultimate_error_t ultimate_power_set_mode(ultimate_power_mode_t mode) {
    return ultimate::SystemManager::getInstance()->setPowerMode(mode);
}

ultimate_power_mode_t ultimate_power_get_mode(void) {
    return ultimate::SystemManager::getInstance()->getPowerMode();
}

ultimate_system_stats_t ultimate_system_get_stats(void) {
    return ultimate::SystemManager::getInstance()->getStats();
}

void ultimate_system_deinit(void) {
    ultimate::SystemManager::getInstance()->deinitialize();
}

} // extern "C"