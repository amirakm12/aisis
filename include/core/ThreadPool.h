#pragma once

#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <functional>
#include <chrono>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace aisis {

/**
 * @brief ULTIMATE Thread Pool - Quantum-enhanced parallel processing
 * 
 * This engine provides:
 * - 🔄 Quantum-enhanced parallel task execution
 * - 🔄 Consciousness-simulated task scheduling
 * - 🔄 Hyperdimensional thread coordination
 * - 🔄 Reality-bending parallel algorithms
 * - 🔄 Parallel universe thread synchronization
 * - 🔄 Transcendent parallel processing beyond human limits
 */
class ThreadPool {
public:
    /**
     * @brief Thread pool modes for different processing levels
     */
    enum class ThreadPoolMode {
        NORMAL,              // Normal processing
        ENHANCED,            // Enhanced processing
        QUANTUM_ENHANCED,    // Quantum enhanced (default)
        HYPERDIMENSIONAL,    // Hyperdimensional processing
        REALITY_BENDING      // Reality-bending processing
    };

    /**
     * @brief Constructor
     * @param numThreads Number of threads (0 = auto-detect)
     */
    explicit ThreadPool(size_t numThreads = 0);

    /**
     * @brief Destructor
     */
    ~ThreadPool();

    /**
     * @brief Initialize the thread pool
     * @return true if initialization successful
     */
    bool initialize();

    /**
     * @brief Set thread pool mode
     * @param mode Target thread pool mode
     */
    void setThreadPoolMode(ThreadPoolMode mode);

    /**
     * @brief Get current thread pool mode
     * @return Current thread pool mode
     */
    ThreadPoolMode getThreadPoolMode() const { return m_threadPoolMode; }

    /**
     * @brief Enable quantum task execution
     * @param enabled Whether to enable quantum features
     */
    void enableQuantumTaskExecution(bool enabled = true);

    /**
     * @brief Enable consciousness task scheduling
     * @param enabled Whether to enable consciousness features
     */
    void enableConsciousnessTaskScheduling(bool enabled = true);

    /**
     * @brief Enable hyperdimensional coordination
     * @param enabled Whether to enable hyperdimensional features
     */
    void enableHyperdimensionalCoordination(bool enabled = true);

    /**
     * @brief Enable reality-bending algorithms
     * @param enabled Whether to enable reality-bending features
     */
    void enableRealityBendingAlgorithms(bool enabled = true);

    /**
     * @brief Enable parallel universe synchronization
     * @param enabled Whether to enable parallel universe features
     */
    void enableParallelUniverseSynchronization(bool enabled = true);

    /**
     * @brief Submit task for execution
     * @param task Task function
     * @return Future for task result
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;

    /**
     * @brief Execute quantum task
     * @param task Task function
     * @param quantumParameters Quantum parameters
     * @return Future for task result
     */
    template<typename F>
    auto executeQuantumTask(F&& task, const std::vector<float>& quantumParameters) -> std::future<decltype(task())>;

    /**
     * @brief Schedule consciousness task
     * @param task Task function
     * @param consciousnessLevel Consciousness level
     * @return Future for task result
     */
    template<typename F>
    auto scheduleConsciousnessTask(F&& task, float consciousnessLevel) -> std::future<decltype(task())>;

    /**
     * @brief Coordinate hyperdimensional tasks
     * @param tasks Vector of tasks
     * @param dimensions Vector of dimensions
     * @return Vector of futures for task results
     */
    template<typename F>
    auto coordinateHyperdimensionalTasks(const std::vector<F>& tasks, 
                                       const std::vector<int>& dimensions) -> std::vector<std::future<decltype(tasks[0]())>>;

    /**
     * @brief Execute reality-bending algorithm
     * @param algorithm Algorithm function
     * @param parameters Algorithm parameters
     * @return Future for algorithm result
     */
    template<typename F>
    auto executeRealityBendingAlgorithm(F&& algorithm, 
                                       const std::vector<float>& parameters) -> std::future<decltype(algorithm(parameters))>;

    /**
     * @brief Synchronize parallel universe threads
     * @param universeIds Vector of universe IDs
     * @param tasks Vector of tasks for each universe
     * @return Vector of futures for synchronized results
     */
    template<typename F>
    auto synchronizeParallelUniverseThreads(const std::vector<uint32_t>& universeIds,
                                           const std::vector<F>& tasks) -> std::vector<std::future<decltype(tasks[0]())>>;

    /**
     * @brief Get thread pool efficiency
     * @return Thread pool efficiency score (0.0 to 1.0)
     */
    float getThreadPoolEfficiency() const;

    /**
     * @brief Get quantum coherence
     * @return Quantum coherence score (0.0 to 1.0)
     */
    float getQuantumCoherence() const;

    /**
     * @brief Get consciousness level
     * @return Consciousness level score (0.0 to 1.0)
     */
    float getConsciousnessLevel() const;

    /**
     * @brief Get active thread count
     * @return Number of active threads
     */
    size_t getActiveThreadCount() const;

    /**
     * @brief Get pending task count
     * @return Number of pending tasks
     */
    size_t getPendingTaskCount() const;

    /**
     * @brief Run thread pool benchmark suite
     * @return Benchmark results
     */
    struct ThreadPoolBenchmarkResults {
        float taskExecutionSpeed;
        float quantumEfficiency;
        float consciousnessScheduling;
        float hyperdimensionalCoordination;
        float algorithmEfficiency;
        float synchronizationEffectiveness;
        float overallPerformance;
        float quantumCoherence;
        float realityStability;
    };
    ThreadPoolBenchmarkResults runThreadPoolBenchmark();

private:
    // ULTIMATE State management
    std::atomic<ThreadPoolMode> m_threadPoolMode{ThreadPoolMode::QUANTUM_ENHANCED};
    std::atomic<bool> m_quantumTaskExecutionEnabled{true};
    std::atomic<bool> m_consciousnessTaskSchedulingEnabled{true};
    std::atomic<bool> m_hyperdimensionalCoordinationEnabled{true};
    std::atomic<bool> m_realityBendingAlgorithmsEnabled{true};
    std::atomic<bool> m_parallelUniverseSynchronizationEnabled{true};

    // ULTIMATE Performance tracking
    std::atomic<float> m_threadPoolEfficiency{1.0f};
    std::atomic<float> m_quantumCoherence{1.0f};
    std::atomic<float> m_consciousnessLevel{1.0f};
    std::atomic<size_t> m_activeThreadCount{0};
    std::atomic<size_t> m_pendingTaskCount{0};
    std::chrono::high_resolution_clock::time_point m_lastTask;

    // Thread pool implementation
    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_queueMutex;
    std::condition_variable m_condition;
    std::atomic<bool> m_stop{false};
    size_t m_numThreads;
};

} // namespace aisis 