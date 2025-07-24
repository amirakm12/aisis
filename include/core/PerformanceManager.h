#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <functional>
#include <string>

namespace aisis {

struct SystemMetrics {
    float cpuUsage{0.0f};
    float memoryUsage{0.0f};
    float gpuUsage{0.0f};
    float diskUsage{0.0f};
    float networkUsage{0.0f};
    float temperature{0.0f};
    int activeThreads{0};
    size_t memoryAllocated{0};
    size_t memoryAvailable{0};
    float frameRate{0.0f};
    float renderTime{0.0f};
    float audioLatency{0.0f};
    std::chrono::high_resolution_clock::time_point timestamp;
};

struct PerformanceProfile {
    std::string name;
    int cpuPriority{0}; // -20 to 19 (Linux), THREAD_PRIORITY_* (Windows)
    int threadCount{0}; // 0 = auto-detect
    bool gpuAcceleration{true};
    int renderQuality{8}; // 1-10 scale
    int audioQuality{8}; // 1-10 scale
    bool memoryOptimization{true};
    bool powerSaving{false};
    float targetFrameRate{60.0f};
    std::unordered_map<std::string, float> customSettings;
};

struct OptimizationRule {
    std::string name;
    std::function<bool(const SystemMetrics&)> condition;
    std::function<void()> action;
    int priority{0};
    bool enabled{true};
    std::chrono::milliseconds cooldown{1000};
    std::chrono::high_resolution_clock::time_point lastTriggered;
};

class PerformanceManager {
public:
    PerformanceManager();
    ~PerformanceManager();
    
    // Initialization
    bool initialize();
    void shutdown();
    
    // Monitoring
    void startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const { return m_monitoring; }
    
    // Metrics
    SystemMetrics getCurrentMetrics() const;
    std::vector<SystemMetrics> getMetricsHistory(int seconds = 60) const;
    float getAverageMetric(const std::string& metricName, int seconds = 10) const;
    float getPeakMetric(const std::string& metricName, int seconds = 60) const;
    
    // Performance Profiles
    void createProfile(const std::string& name, const PerformanceProfile& profile);
    void deleteProfile(const std::string& name);
    void activateProfile(const std::string& name);
    std::string getActiveProfile() const { return m_activeProfile; }
    std::vector<std::string> getAvailableProfiles() const;
    void saveProfilesToFile(const std::string& filePath) const;
    bool loadProfilesFromFile(const std::string& filePath);
    
    // Automatic Optimization
    void enableAutoOptimization(bool enabled);
    bool isAutoOptimizationEnabled() const { return m_autoOptimizationEnabled; }
    void addOptimizationRule(const OptimizationRule& rule);
    void removeOptimizationRule(const std::string& name);
    void enableOptimizationRule(const std::string& name, bool enabled);
    
    // System Control
    void setCPUAffinity(const std::vector<int>& cores);
    void setProcessPriority(int priority);
    void setMemoryLimit(size_t limitMB);
    void enablePowerSaving(bool enabled);
    void setThermalThrottling(float maxTemp);
    
    // Resource Management
    void optimizeMemoryUsage();
    void clearCache();
    void defragmentMemory();
    void preloadResources(const std::vector<std::string>& resources);
    void unloadUnusedResources();
    
    // GPU Management
    void enableGPUScheduling(bool enabled);
    void setGPUMemoryLimit(size_t limitMB);
    void optimizeGPUPerformance();
    std::vector<std::string> getAvailableGPUs() const;
    void selectGPU(int gpuIndex);
    
    // Threading Optimization
    void optimizeThreadPool();
    void setMaxThreads(int maxThreads);
    void balanceThreadLoad();
    void enableHyperThreading(bool enabled);
    
    // Real-time Adjustments
    void enableDynamicQuality(bool enabled);
    void setQualityTarget(float targetFrameRate);
    void enableAdaptiveRendering(bool enabled);
    void setPerformanceTarget(const std::string& target); // "performance", "quality", "balanced"
    
    // Benchmarking
    void runBenchmark(const std::string& benchmarkName);
    float getBenchmarkScore(const std::string& benchmarkName) const;
    std::unordered_map<std::string, float> getAllBenchmarkScores() const;
    void compareBenchmarks(const std::string& baseline, const std::string& current);
    
    // Profiling
    void startProfiling(const std::string& sessionName);
    void stopProfiling();
    void saveProfilingResults(const std::string& filePath) const;
    std::vector<std::string> getPerformanceBottlenecks() const;
    
    // Event System
    using PerformanceCallback = std::function<void(const std::string&, const SystemMetrics&)>;
    void registerCallback(const std::string& event, PerformanceCallback callback);
    void unregisterCallback(const std::string& event);
    
    // System Information
    std::string getCPUInfo() const;
    std::string getGPUInfo() const;
    size_t getTotalMemory() const;
    size_t getAvailableMemory() const;
    std::string getSystemInfo() const;
    
    // Advanced Features
    void enablePredictiveOptimization(bool enabled);
    void setOptimizationAggressiveness(int level); // 1-10 scale
    void enableCloudOptimization(bool enabled);
    void syncOptimizationSettings(const std::string& cloudProvider);
    
    // Debugging and Diagnostics
    void enableDebugMode(bool enabled);
    void generatePerformanceReport(const std::string& filePath) const;
    std::vector<std::string> getDiagnosticInfo() const;
    void runSystemDiagnostics();
    
private:
    // Core monitoring
    std::atomic<bool> m_monitoring{false};
    std::atomic<bool> m_initialized{false};
    std::thread m_monitoringThread;
    std::mutex m_metricsMutex;
    
    // Metrics storage
    std::vector<SystemMetrics> m_metricsHistory;
    size_t m_maxHistorySize{3600}; // 1 hour at 1Hz
    SystemMetrics m_currentMetrics;
    
    // Performance profiles
    std::unordered_map<std::string, PerformanceProfile> m_profiles;
    std::string m_activeProfile{"default"};
    std::mutex m_profilesMutex;
    
    // Optimization
    std::atomic<bool> m_autoOptimizationEnabled{true};
    std::vector<OptimizationRule> m_optimizationRules;
    std::thread m_optimizationThread;
    std::atomic<bool> m_optimizationRunning{false};
    
    // System state
    std::vector<int> m_cpuAffinity;
    int m_processPriority{0};
    size_t m_memoryLimit{0}; // 0 = no limit
    bool m_powerSavingEnabled{false};
    float m_thermalThreshold{85.0f}; // Celsius
    
    // GPU management
    int m_selectedGPU{0};
    size_t m_gpuMemoryLimit{0};
    bool m_gpuSchedulingEnabled{true};
    
    // Threading
    int m_maxThreads{0}; // 0 = auto
    bool m_hyperThreadingEnabled{true};
    
    // Dynamic adjustments
    bool m_dynamicQualityEnabled{true};
    float m_targetFrameRate{60.0f};
    bool m_adaptiveRenderingEnabled{true};
    std::string m_performanceTarget{"balanced"};
    
    // Benchmarking
    std::unordered_map<std::string, float> m_benchmarkScores;
    std::mutex m_benchmarkMutex;
    
    // Profiling
    bool m_profilingActive{false};
    std::string m_currentProfilingSession;
    std::chrono::high_resolution_clock::time_point m_profilingStartTime;
    std::vector<std::pair<std::string, std::chrono::microseconds>> m_profilingData;
    
    // Event system
    std::unordered_map<std::string, std::vector<PerformanceCallback>> m_callbacks;
    std::mutex m_callbacksMutex;
    
    // Advanced features
    bool m_predictiveOptimizationEnabled{false};
    int m_optimizationAggressiveness{5};
    bool m_cloudOptimizationEnabled{false};
    bool m_debugMode{false};
    
    // Thread functions
    void monitoringThreadFunction();
    void optimizationThreadFunction();
    
    // Metrics collection
    void updateSystemMetrics();
    float getCPUUsage();
    float getMemoryUsage();
    float getGPUUsage();
    float getDiskUsage();
    float getNetworkUsage();
    float getSystemTemperature();
    
    // Optimization logic
    void checkOptimizationRules();
    void applyPerformanceProfile(const PerformanceProfile& profile);
    void adjustQualitySettings();
    void optimizeForTarget();
    
    // System control implementation
    void applyCPUAffinity();
    void applyProcessPriority();
    void applyMemoryLimit();
    void applyThermalThrottling();
    
    // Utility methods
    void triggerEvent(const std::string& event, const SystemMetrics& metrics);
    void addMetricsToHistory(const SystemMetrics& metrics);
    void cleanupOldMetrics();
    std::string formatSystemInfo() const;
    
    // Platform-specific implementations
    #ifdef _WIN32
    void initializeWindowsPerformance();
    #elif __linux__
    void initializeLinuxPerformance();
    #endif
};

} // namespace aisis