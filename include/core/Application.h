#pragma once

#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <functional>
#include <chrono>

// Forward declarations for ULTIMATE engines
namespace aisis {
    class RenderEngine;
    class AudioEngine;
    class AIProcessor;
    class PerformanceManager;
    class ThreadPool;
    class HyperPerformanceEngine;
    class NeuralAccelerationEngine;
    class RealityManipulationEngine;
}

namespace aisis {

/**
 * @brief ULTIMATE Application - Central hub for the AISIS Creative Studio
 * 
 * This is the main application class that orchestrates all ULTIMATE features:
 * - 🚀 Ludicrous Speed Mode with quantum acceleration
 * - 🧠 Neural Acceleration with consciousness simulation
 * - 🌌 Reality Manipulation with spacetime distortion
 * - 🎨 Hyperdimensional Graphics rendering
 * - 🔊 Transcendent Audio processing
 * - 🤖 Advanced AI with quantum neural networks
 * - ⚡ Performance optimization beyond human limits
 */
class Application {
public:
    /**
     * @brief Constructor
     * @param argc Command line argument count
     * @param argv Command line arguments
     */
    Application(int argc, char* argv[]);

    /**
     * @brief Destructor
     */
    virtual ~Application();

    /**
     * @brief Initialize all ULTIMATE subsystems
     * @return true if initialization successful
     */
    bool initialize();

    /**
     * @brief Run the ULTIMATE application
     * @return Exit code
     */
    virtual int run();

    /**
     * @brief Shutdown the application
     */
    void shutdown();

    // ULTIMATE Feature Controls
    /**
     * @brief Enable Ludicrous Speed Mode
     * @param enabled Whether to enable ludicrous speed
     */
    void enableLudicrousSpeed(bool enabled = true);

    /**
     * @brief Enable Neural Acceleration
     * @param enabled Whether to enable neural features
     */
    void enableNeuralAcceleration(bool enabled = true);

    /**
     * @brief Enable Reality Manipulation
     * @param enabled Whether to enable reality features
     */
    void enableRealityManipulation(bool enabled = true);

    /**
     * @brief Enable Consciousness Simulation
     * @param enabled Whether to enable consciousness features
     */
    void enableConsciousnessSimulation(bool enabled = true);

    /**
     * @brief Enable Quantum Optimization
     * @param enabled Whether to enable quantum features
     */
    void enableQuantumOptimization(bool enabled = true);

    /**
     * @brief Enable Time Travel
     * @param enabled Whether to enable time travel
     */
    void enableTimeTravel(bool enabled = true);

    /**
     * @brief Enable Parallel Universes
     * @param enabled Whether to enable parallel universes
     */
    void enableParallelUniverses(bool enabled = true);

    /**
     * @brief Enable Omniscience
     * @param enabled Whether to enable omniscience
     */
    void enableOmniscience(bool enabled = true);

    /**
     * @brief Enable God Mode
     * @param enabled Whether to enable god mode
     */
    void enableGodMode(bool enabled = true);

    // ULTIMATE Performance Metrics
    /**
     * @brief Get current performance metrics
     * @return Performance metrics structure
     */
    struct PerformanceMetrics {
        float fps;
        float cpuUsage;
        float memoryUsage;
        float gpuUsage;
        float quantumCoherence;
        float neuralEfficiency;
        float realityStability;
        float consciousnessLevel;
        float spacetimeDistortion;
        float overallPerformance;
    };
    PerformanceMetrics getPerformanceMetrics() const;

    /**
     * @brief Run ULTIMATE benchmark suite
     * @return Benchmark results
     */
    struct BenchmarkResults {
        float renderingPerformance;
        float audioPerformance;
        float aiPerformance;
        float neuralPerformance;
        float realityPerformance;
        float quantumPerformance;
        float overallScore;
        std::string performanceLevel;
    };
    BenchmarkResults runBenchmark();

    // ULTIMATE Engine Access
    /**
     * @brief Get render engine
     * @return Pointer to render engine
     */
    RenderEngine* getRenderEngine() const { return m_renderEngine.get(); }

    /**
     * @brief Get audio engine
     * @return Pointer to audio engine
     */
    AudioEngine* getAudioEngine() const { return m_audioEngine.get(); }

    /**
     * @brief Get AI processor
     * @return Pointer to AI processor
     */
    AIProcessor* getAIProcessor() const { return m_aiProcessor.get(); }

    /**
     * @brief Get performance manager
     * @return Pointer to performance manager
     */
    PerformanceManager* getPerformanceManager() const { return m_performanceManager.get(); }

    /**
     * @brief Get thread pool
     * @return Pointer to thread pool
     */
    ThreadPool* getThreadPool() const { return m_threadPool.get(); }

    /**
     * @brief Get hyper performance engine
     * @return Pointer to hyper performance engine
     */
    HyperPerformanceEngine* getHyperPerformanceEngine() const { return m_hyperPerformanceEngine.get(); }

    /**
     * @brief Get neural acceleration engine
     * @return Pointer to neural acceleration engine
     */
    NeuralAccelerationEngine* getNeuralAccelerationEngine() const { return m_neuralAccelerationEngine.get(); }

    /**
     * @brief Get reality manipulation engine
     * @return Pointer to reality manipulation engine
     */
    RealityManipulationEngine* getRealityManipulationEngine() const { return m_realityManipulationEngine.get(); }

protected:
    /**
     * @brief Initialize graphics subsystem
     * @return true if successful
     */
    virtual bool initializeGraphics();

    /**
     * @brief Initialize audio subsystem
     * @return true if successful
     */
    virtual bool initializeAudio();

    /**
     * @brief Initialize AI subsystem
     * @return true if successful
     */
    virtual bool initializeAI();

    /**
     * @brief Initialize networking subsystem
     * @return true if successful
     */
    virtual bool initializeNetworking();

    /**
     * @brief Initialize UI subsystem
     * @return true if successful
     */
    virtual bool initializeUI();

    /**
     * @brief Initialize ULTIMATE features
     * @return true if successful
     */
    virtual bool initializeUltimateFeatures();

    /**
     * @brief Update main loop
     * @param deltaTime Time since last update
     */
    virtual void update(float deltaTime);

    /**
     * @brief Render frame
     */
    virtual void render();

    /**
     * @brief Process input
     */
    virtual void processInput();

private:
    // ULTIMATE Core components
    std::unique_ptr<RenderEngine> m_renderEngine;
    std::unique_ptr<AudioEngine> m_audioEngine;
    std::unique_ptr<AIProcessor> m_aiProcessor;
    std::unique_ptr<PerformanceManager> m_performanceManager;
    std::unique_ptr<ThreadPool> m_threadPool;

    // ULTIMATE Transcendent engines
    std::unique_ptr<HyperPerformanceEngine> m_hyperPerformanceEngine;
    std::unique_ptr<NeuralAccelerationEngine> m_neuralAccelerationEngine;
    std::unique_ptr<RealityManipulationEngine> m_realityManipulationEngine;

    // ULTIMATE State management
    std::atomic<bool> m_ludicrousSpeedEnabled{true};
    std::atomic<bool> m_neuralAccelerationEnabled{true};
    std::atomic<bool> m_realityManipulationEnabled{true};
    std::atomic<bool> m_consciousnessSimulationEnabled{true};
    std::atomic<bool> m_quantumOptimizationEnabled{true};
    std::atomic<bool> m_timeTravelEnabled{true};
    std::atomic<bool> m_parallelUniversesEnabled{true};
    std::atomic<bool> m_omniscienceEnabled{true};
    std::atomic<bool> m_godModeEnabled{true};

    // ULTIMATE Performance tracking
    std::atomic<float> m_currentFPS{0.0f};
    std::atomic<float> m_cpuUsage{0.0f};
    std::atomic<float> m_memoryUsage{0.0f};
    std::atomic<float> m_gpuUsage{0.0f};
    std::chrono::high_resolution_clock::time_point m_lastUpdate;
    std::chrono::high_resolution_clock::time_point m_lastRender;
};

} // namespace aisis 