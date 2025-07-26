#pragma once

#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <functional>
#include <chrono>

namespace aisis {

/**
 * @brief ULTIMATE Performance Manager - Performance optimization and monitoring
 * 
 * This engine provides:
 * - ⚡ Real-time performance monitoring and optimization
 * - ⚡ Quantum-enhanced performance prediction
 * - ⚡ Adaptive resource allocation
 * - ⚡ Reality-bending performance tuning
 * - ⚡ Parallel universe performance coordination
 * - ⚡ Transcendent performance beyond human limits
 */
class PerformanceManager {
public:
    /**
     * @brief Performance modes for different optimization levels
     */
    enum class PerformanceMode {
        NORMAL,              // Normal performance
        ENHANCED,            // Enhanced performance
        ULTIMATE,            // Ultimate performance (default)
        QUANTUM_OPTIMIZED,   // Quantum optimized
        REALITY_BENDING      // Reality-bending performance
    };

    /**
     * @brief Constructor
     */
    PerformanceManager();

    /**
     * @brief Destructor
     */
    ~PerformanceManager();

    /**
     * @brief Initialize the performance manager
     * @return true if initialization successful
     */
    bool initialize();

    /**
     * @brief Set performance mode
     * @param mode Target performance mode
     */
    void setPerformanceMode(PerformanceMode mode);

    /**
     * @brief Get current performance mode
     * @return Current performance mode
     */
    PerformanceMode getPerformanceMode() const { return m_performanceMode; }

    /**
     * @brief Enable quantum performance optimization
     * @param enabled Whether to enable quantum features
     */
    void enableQuantumPerformanceOptimization(bool enabled = true);

    /**
     * @brief Enable adaptive resource allocation
     * @param enabled Whether to enable adaptive features
     */
    void enableAdaptiveResourceAllocation(bool enabled = true);

    /**
     * @brief Enable reality-bending tuning
     * @param enabled Whether to enable reality-bending features
     */
    void enableRealityBendingTuning(bool enabled = true);

    /**
     * @brief Enable parallel universe coordination
     * @param enabled Whether to enable parallel universe features
     */
    void enableParallelUniverseCoordination(bool enabled = true);

    /**
     * @brief Monitor system performance
     * @return Performance metrics
     */
    struct PerformanceMetrics {
        float cpuUsage;
        float memoryUsage;
        float gpuUsage;
        float networkUsage;
        float diskUsage;
        float quantumCoherence;
        float realityStability;
        float overallPerformance;
    };
    PerformanceMetrics monitorSystemPerformance();

    /**
     * @brief Optimize quantum performance
     * @param parameters Performance parameters
     */
    void optimizeQuantumPerformance(const std::vector<float>& parameters);

    /**
     * @brief Allocate adaptive resources
     * @param resourceType Resource type
     * @param amount Amount to allocate
     * @return Allocation success
     */
    bool allocateAdaptiveResources(const std::string& resourceType, float amount);

    /**
     * @brief Tune reality-bending performance
     * @param tuningParameters Tuning parameters
     */
    void tuneRealityBendingPerformance(const std::vector<float>& tuningParameters);

    /**
     * @brief Coordinate parallel universe performance
     * @param universeIds Vector of universe IDs
     * @param performanceData Vector of performance data from each universe
     * @param coordinatedResult Coordinated performance result
     */
    void coordinateParallelUniversePerformance(const std::vector<uint32_t>& universeIds,
                                             const std::vector<PerformanceMetrics>& performanceData,
                                             PerformanceMetrics& coordinatedResult);

    /**
     * @brief Get performance efficiency
     * @return Performance efficiency score (0.0 to 1.0)
     */
    float getPerformanceEfficiency() const;

    /**
     * @brief Get quantum coherence
     * @return Quantum coherence score (0.0 to 1.0)
     */
    float getQuantumCoherence() const;

    /**
     * @brief Get reality stability
     * @return Reality stability score (0.0 to 1.0)
     */
    float getRealityStability() const;

    /**
     * @brief Run performance benchmark suite
     * @return Benchmark results
     */
    struct PerformanceBenchmarkResults {
        float monitoringSpeed;
        float optimizationEfficiency;
        float resourceAllocation;
        float tuningAccuracy;
        float coordinationEffectiveness;
        float overallPerformance;
        float quantumCoherence;
        float realityStability;
    };
    PerformanceBenchmarkResults runPerformanceBenchmark();

private:
    // ULTIMATE State management
    std::atomic<PerformanceMode> m_performanceMode{PerformanceMode::ULTIMATE};
    std::atomic<bool> m_quantumPerformanceOptimizationEnabled{true};
    std::atomic<bool> m_adaptiveResourceAllocationEnabled{true};
    std::atomic<bool> m_realityBendingTuningEnabled{true};
    std::atomic<bool> m_parallelUniverseCoordinationEnabled{true};

    // ULTIMATE Performance tracking
    std::atomic<float> m_performanceEfficiency{1.0f};
    std::atomic<float> m_quantumCoherence{1.0f};
    std::atomic<float> m_realityStability{1.0f};
    std::chrono::high_resolution_clock::time_point m_lastMonitor;
};

} // namespace aisis 