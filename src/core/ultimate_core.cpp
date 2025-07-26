#include "ultimate_core.h"
#include "ultimate_system.h"
#include "ultimate_memory.h"
#include "ultimate_errors.h"
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>

namespace ultimate {

class UltimateSystemImpl {
private:
    static std::unique_ptr<UltimateSystemImpl> instance_;
    static std::mutex instance_mutex_;
    
    ultimate_init_config_t config_;
    bool initialized_;
    bool running_;
    std::chrono::steady_clock::time_point start_time_;
    
    UltimateSystemImpl() : initialized_(false), running_(false) {}

public:
    static UltimateSystemImpl* getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::unique_ptr<UltimateSystemImpl>(new UltimateSystemImpl());
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
        
        // Copy configuration
        config_ = *config;
        
        // Initialize memory management
        ultimate_error_t error = ultimate_memory_init(config_.heap_size);
        if (error != ULTIMATE_OK) {
            return error;
        }
        
        // Initialize system components
        error = ultimate_system_init(&config_);
        if (error != ULTIMATE_OK) {
            ultimate_memory_deinit();
            return error;
        }
        
        initialized_ = true;
        start_time_ = std::chrono::steady_clock::now();
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t start() {
        if (!initialized_) {
            return ULTIMATE_ERROR_NOT_INITIALIZED;
        }
        
        if (running_) {
            return ULTIMATE_ERROR_ALREADY_RUNNING;
        }
        
        // Start system scheduler
        ultimate_error_t error = ultimate_system_start();
        if (error != ULTIMATE_OK) {
            return error;
        }
        
        running_ = true;
        return ULTIMATE_OK;
    }
    
    ultimate_error_t stop() {
        if (!running_) {
            return ULTIMATE_ERROR_NOT_RUNNING;
        }
        
        ultimate_error_t error = ultimate_system_stop();
        running_ = false;
        return error;
    }
    
    ultimate_error_t shutdown() {
        if (running_) {
            stop();
        }
        
        if (!initialized_) {
            return ULTIMATE_ERROR_NOT_INITIALIZED;
        }
        
        ultimate_system_deinit();
        ultimate_memory_deinit();
        
        initialized_ = false;
        return ULTIMATE_OK;
    }
    
    bool isInitialized() const { return initialized_; }
    bool isRunning() const { return running_; }
    
    uint64_t getUptime() const {
        if (!initialized_) return 0;
        
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
        return duration.count();
    }
};

std::unique_ptr<UltimateSystemImpl> UltimateSystemImpl::instance_ = nullptr;
std::mutex UltimateSystemImpl::instance_mutex_;

} // namespace ultimate

// C API Implementation
extern "C" {

ultimate_error_t ultimate_init(const ultimate_init_config_t* config) {
    return ultimate::UltimateSystemImpl::getInstance()->initialize(config);
}

ultimate_error_t ultimate_start(void) {
    return ultimate::UltimateSystemImpl::getInstance()->start();
}

ultimate_error_t ultimate_stop(void) {
    return ultimate::UltimateSystemImpl::getInstance()->stop();
}

ultimate_error_t ultimate_shutdown(void) {
    return ultimate::UltimateSystemImpl::getInstance()->shutdown();
}

bool ultimate_is_initialized(void) {
    return ultimate::UltimateSystemImpl::getInstance()->isInitialized();
}

bool ultimate_is_running(void) {
    return ultimate::UltimateSystemImpl::getInstance()->isRunning();
}

uint64_t ultimate_get_uptime_ms(void) {
    return ultimate::UltimateSystemImpl::getInstance()->getUptime();
}

void ultimate_delay_ms(uint32_t milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

const char* ultimate_get_version(void) {
    return ULTIMATE_VERSION_STRING;
}

} // extern "C"