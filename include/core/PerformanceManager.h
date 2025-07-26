#pragma once

#include <memory>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <atomic>

namespace Ultimate {
namespace Core {

// Performance levels
enum class PerformanceLevel {
    PowerSaver = 1,
    Balanced = 2,
    Performance = 3,
    HighPerformance = 4,
    Maximum = 5
};

// Resource types
enum class ResourceType {
    CPU,
    Memory,
    GPU,
    Disk,
    Network,
    Battery
};

// Performance metrics
struct SystemMetrics {
    // CPU metrics
    double cpuUsage = 0.0;
    double cpuTemperature = 0.0;
    int cpuCores = 0;
    int cpuThreads = 0;
    double cpuFrequency = 0.0;
    
    // Memory metrics
    size_t totalMemory = 0;
    size_t usedMemory = 0;
    size_t availableMemory = 0;
    double memoryUsagePercent = 0.0;
    
    // GPU metrics
    double gpuUsage = 0.0;
    double gpuTemperature = 0.0;
    size_t gpuMemoryTotal = 0;
    size_t gpuMemoryUsed = 0;
    double gpuMemoryUsagePercent = 0.0;
    
    // Disk metrics
    size_t diskSpaceTotal = 0;
    size_t diskSpaceUsed = 0;
    size_t diskSpaceAvailable = 0;
    double diskUsagePercent = 0.0;
    double diskReadSpeed = 0.0;
    double diskWriteSpeed = 0.0;
    
    // Network metrics
    double networkUploadSpeed = 0.0;
    double networkDownloadSpeed = 0.0;
    size_t networkBytesUploaded = 0;
    size_t networkBytesDownloaded = 0;
    
    // Battery metrics (for laptops/mobile)
    double batteryLevel = 100.0;
    bool batteryCharging = false;
    double batteryTimeRemaining = 0.0;
    
    // System metrics
    double frameRate = 0.0;
    double frameTime = 0.0;
    int activeProcesses = 0;
    int activeThreads = 0;
    double systemUptime = 0.0;
};

// Performance targets
struct PerformanceTargets {
    double targetFrameRate = 60.0;
    double maxCpuUsage = 80.0;
    double maxMemoryUsage = 80.0;
    double maxGpuUsage = 90.0;
    double maxTemperature = 85.0;
    double minBatteryLevel = 20.0;
};

// Resource allocation
struct ResourceAllocation {
    int cpuCores = -1; // -1 = auto
    size_t memoryLimit = 0; // 0 = no limit
    int gpuMemoryLimit = 0; // 0 = no limit
    int networkBandwidthLimit = 0; // 0 = no limit
    int diskIOLimit = 0; // 0 = no limit
};

class PerformanceManager {
public:
    static PerformanceManager& getInstance();
    
    // Initialization and cleanup
    bool initialize();
    void shutdown();
    
    // Performance monitoring
    void startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const;
    
    const SystemMetrics& getCurrentMetrics() const;
    const SystemMetrics& getAverageMetrics() const;
    const SystemMetrics& getPeakMetrics() const;
    
    // Performance control
    void setPerformanceLevel(PerformanceLevel level);
    PerformanceLevel getPerformanceLevel() const;
    
    void setPerformanceTargets(const PerformanceTargets& targets);
    const PerformanceTargets& getPerformanceTargets() const;
    
    // Resource management
    void setResourceAllocation(const ResourceAllocation& allocation);
    const ResourceAllocation& getResourceAllocation() const;
    
    void limitResource(ResourceType type, double percentage);
    void unlimitResource(ResourceType type);
    double getResourceLimit(ResourceType type) const;
    
    // Automatic optimization
    void enableAutoOptimization(bool enable);
    bool isAutoOptimizationEnabled() const;
    
    void enableThermalThrottling(bool enable);
    bool isThermalThrottlingEnabled() const;
    
    void enablePowerManagement(bool enable);
    bool isPowerManagementEnabled() const;
    
    // Process management
    void setProcessPriority(int processId, int priority);
    int getProcessPriority(int processId) const;
    
    void setThreadAffinity(int threadId, const std::vector<int>& cpuCores);
    std::vector<int> getThreadAffinity(int threadId) const;
    
    // Memory management
    void enableMemoryCompression(bool enable);
    bool isMemoryCompressionEnabled() const;
    
    void enableSwapFile(bool enable);
    bool isSwapFileEnabled() const;
    
    void setSwapFileSize(size_t sizeBytes);
    size_t getSwapFileSize() const;
    
    void optimizeMemoryUsage();
    void clearMemoryCache();
    
    // GPU management
    void setGPUPowerLimit(int watts);
    int getGPUPowerLimit() const;
    
    void setGPUMemoryClockSpeed(int mhz);
    int getGPUMemoryClockSpeed() const;
    
    void setGPUCoreClockSpeed(int mhz);
    int getGPUCoreClockSpeed() const;
    
    // Thermal management
    void setThermalTarget(double temperature);
    double getThermalTarget() const;
    
    void setFanCurve(const std::vector<std::pair<double, double>>& curve);
    std::vector<std::pair<double, double>> getFanCurve() const;
    
    // Power management
    void setPowerProfile(const std::string& profile);
    std::string getPowerProfile() const;
    
    std::vector<std::string> getAvailablePowerProfiles() const;
    
    void enableCPUBoost(bool enable);
    bool isCPUBoostEnabled() const;
    
    void enableGPUBoost(bool enable);
    bool isGPUBoostEnabled() const;
    
    // Performance profiling
    void startProfiling(const std::string& name);
    void endProfiling(const std::string& name);
    
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
    
    // System information
    std::string getCPUName() const;
    std::string getGPUName() const;
    size_t getTotalSystemMemory() const;
    std::string getOperatingSystem() const;
    
    // Benchmarking
    void runBenchmark(const std::string& benchmarkName);
    double getBenchmarkScore(const std::string& benchmarkName) const;
    std::unordered_map<std::string, double> getAllBenchmarkScores() const;
    
    // Callbacks
    using MetricsCallback = std::function<void(const SystemMetrics&)>;
    using ThresholdCallback = std::function<void(ResourceType, double)>;
    using OptimizationCallback = std::function<void(const std::string&)>;
    
    void setMetricsCallback(MetricsCallback callback);
    void setThresholdCallback(ThresholdCallback callback);
    void setOptimizationCallback(OptimizationCallback callback);
    
    // Alerts and warnings
    void setThreshold(ResourceType type, double threshold);
    double getThreshold(ResourceType type) const;
    
    bool isThresholdExceeded(ResourceType type) const;
    std::vector<ResourceType> getExceededThresholds() const;
    
    // Performance history
    void enableHistoryTracking(bool enable);
    bool isHistoryTrackingEnabled() const;
    
    void setHistoryDuration(int seconds);
    int getHistoryDuration() const;
    
    std::vector<SystemMetrics> getMetricsHistory() const;
    void clearMetricsHistory();

private:
    PerformanceManager() = default;
    ~PerformanceManager() = default;
    PerformanceManager(const PerformanceManager&) = delete;
    PerformanceManager& operator=(const PerformanceManager&) = delete;
    
    // Internal state
    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_monitoring{false};
    PerformanceLevel m_performanceLevel = PerformanceLevel::Balanced;
    
    // Metrics
    SystemMetrics m_currentMetrics;
    SystemMetrics m_averageMetrics;
    SystemMetrics m_peakMetrics;
    std::vector<SystemMetrics> m_metricsHistory;
    
    // Configuration
    PerformanceTargets m_targets;
    ResourceAllocation m_allocation;
    std::unordered_map<ResourceType, double> m_resourceLimits;
    std::unordered_map<ResourceType, double> m_thresholds;
    
    // Auto optimization
    std::atomic<bool> m_autoOptimizationEnabled{true};
    std::atomic<bool> m_thermalThrottlingEnabled{true};
    std::atomic<bool> m_powerManagementEnabled{true};
    
    // Memory management
    bool m_memoryCompressionEnabled = false;
    bool m_swapFileEnabled = true;
    size_t m_swapFileSize = 2ULL * 1024 * 1024 * 1024; // 2GB
    
    // GPU settings
    int m_gpuPowerLimit = 0;
    int m_gpuMemoryClockSpeed = 0;
    int m_gpuCoreClockSpeed = 0;
    
    // Thermal settings
    double m_thermalTarget = 80.0;
    std::vector<std::pair<double, double>> m_fanCurve;
    
    // Power settings
    std::string m_powerProfile = "Balanced";
    bool m_cpuBoostEnabled = true;
    bool m_gpuBoostEnabled = true;
    
    // Profiling
    std::unordered_map<std::string, ProfileData> m_profileData;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> m_activeProfiles;
    
    // Benchmarks
    std::unordered_map<std::string, double> m_benchmarkScores;
    
    // History tracking
    bool m_historyTrackingEnabled = true;
    int m_historyDuration = 3600; // 1 hour
    
    // Callbacks
    MetricsCallback m_metricsCallback;
    ThresholdCallback m_thresholdCallback;
    OptimizationCallback m_optimizationCallback;
    
    // Internal methods
    void updateMetrics();
    void applyPerformanceLevel();
    void checkThresholds();
    void performAutoOptimization();
    void updateHistory();
    void initializeSystemInfo();
};

// RAII profiler helper
class ScopedPerformanceProfiler {
public:
    explicit ScopedPerformanceProfiler(const std::string& name);
    ~ScopedPerformanceProfiler();
    
private:
    std::string m_name;
};

// Convenience macro for profiling
#define ULTIMATE_PROFILE_PERFORMANCE(name) Ultimate::Core::ScopedPerformanceProfiler _perf_prof(name)

} // namespace Core
} // namespace Ultimate