#pragma once

#include <memory>
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>
#include <atomic>

namespace Ultimate {
namespace Core {

struct PerformanceMetrics {
    double frameTime = 0.0;
    double cpuUsage = 0.0;
    double memoryUsage = 0.0;
    double gpuUsage = 0.0;
    int activeThreads = 0;
    int allocatedObjects = 0;
};

class HyperPerformanceEngine {
public:
    static HyperPerformanceEngine& getInstance();
    
    // Initialization and cleanup
    bool initialize();
    void shutdown();
    
    // Performance monitoring
    void beginFrame();
    void endFrame();
    
    void beginProfileBlock(const std::string& name);
    void endProfileBlock(const std::string& name);
    
    // Metrics collection
    const PerformanceMetrics& getCurrentMetrics() const;
    const PerformanceMetrics& getAverageMetrics() const;
    
    // Performance optimization
    void enableHyperMode(bool enable);
    bool isHyperModeEnabled() const;
    
    void setPerformanceLevel(int level); // 1-10, 10 being maximum
    int getPerformanceLevel() const;
    
    // Resource optimization
    void optimizeMemoryUsage();
    void optimizeCPUUsage();
    void optimizeGPUUsage();
    
    // Auto-scaling
    void enableAutoScaling(bool enable);
    bool isAutoScalingEnabled() const;
    
    // Thermal management
    void enableThermalThrottling(bool enable);
    bool isThermalThrottlingEnabled() const;
    
    // Profiling data
    struct ProfileData {
        std::string name;
        double totalTime = 0.0;
        double averageTime = 0.0;
        double minTime = std::numeric_limits<double>::max();
        double maxTime = 0.0;
        int callCount = 0;
    };
    
    std::vector<ProfileData> getProfilingData() const;
    void clearProfilingData();
    
    // Performance targets
    void setTargetFrameRate(double fps);
    void setTargetCPUUsage(double percentage);
    void setTargetMemoryUsage(size_t bytes);
    
    // Callbacks
    using PerformanceCallback = std::function<void(const PerformanceMetrics&)>;
    void setPerformanceCallback(PerformanceCallback callback);

private:
    HyperPerformanceEngine() = default;
    ~HyperPerformanceEngine() = default;
    HyperPerformanceEngine(const HyperPerformanceEngine&) = delete;
    HyperPerformanceEngine& operator=(const HyperPerformanceEngine&) = delete;
    
    // Internal state
    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_hyperModeEnabled{false};
    std::atomic<int> m_performanceLevel{5};
    std::atomic<bool> m_autoScalingEnabled{true};
    std::atomic<bool> m_thermalThrottlingEnabled{true};
    
    // Metrics
    PerformanceMetrics m_currentMetrics;
    PerformanceMetrics m_averageMetrics;
    
    // Profiling
    std::unordered_map<std::string, ProfileData> m_profileData;
    std::unordered_map<std::thread::id, std::string> m_activeBlocks;
    
    // Timing
    std::chrono::high_resolution_clock::time_point m_frameStartTime;
    std::chrono::high_resolution_clock::time_point m_lastUpdateTime;
    
    // Targets
    double m_targetFrameRate = 60.0;
    double m_targetCPUUsage = 80.0;
    size_t m_targetMemoryUsage = 1024 * 1024 * 1024; // 1GB
    
    // Callback
    PerformanceCallback m_performanceCallback;
    
    // Internal methods
    void updateMetrics();
    void applyOptimizations();
    void checkThermalLimits();
    double getCurrentCPUUsage();
    double getCurrentMemoryUsage();
    double getCurrentGPUUsage();
};

// RAII profiler helper
class ScopedProfiler {
public:
    explicit ScopedProfiler(const std::string& name);
    ~ScopedProfiler();
    
private:
    std::string m_name;
};

// Convenience macro for profiling
#define ULTIMATE_PROFILE(name) Ultimate::Core::ScopedProfiler _prof(name)

} // namespace Core
} // namespace Ultimate