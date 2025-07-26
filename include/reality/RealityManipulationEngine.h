#pragma once

#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <functional>
#include <chrono>

namespace aisis {

// Forward declarations
class QuantumPhysicsEngine;
class DimensionalRenderingEngine;
class RealityDistortionField;

/**
 * @brief ULTIMATE Reality Manipulation Engine - Spacetime distortion and reality control
 * 
 * This engine provides:
 * - 🌌 Spacetime distortion for performance optimization
 * - 🌌 Dimensional rendering in 4D, 5D, and beyond
 * - 🌌 Parallel universe simulation running simultaneously
 * - 🌌 Quantum physics simulation at particle level
 * - 🌌 Causality manipulation for time travel effects
 * - 🌌 God mode with omnipresence and omniscience
 */
class RealityManipulationEngine {
public:
    /**
     * @brief Reality modes for different manipulation levels
     */
    enum class RealityMode {
        NORMAL,              // Normal reality
        ENHANCED,            // Enhanced reality
        TRANSCENDENT_REALITY, // Transcendent reality (default)
        GOD_MODE             // God mode reality control
    };

    /**
     * @brief Constructor
     */
    RealityManipulationEngine();

    /**
     * @brief Destructor
     */
    ~RealityManipulationEngine();

    /**
     * @brief Initialize the reality manipulation engine
     * @return true if initialization successful
     */
    bool initialize();

    /**
     * @brief Set reality mode
     * @param mode Target reality mode
     */
    void setRealityMode(RealityMode mode);

    /**
     * @brief Get current reality mode
     * @return Current reality mode
     */
    RealityMode getRealityMode() const { return m_realityMode; }

    /**
     * @brief Enable reality branching
     * @param enabled Whether to enable reality branching
     */
    void enableRealityBranching(bool enabled = true);

    /**
     * @brief Enable mind over matter
     * @param enabled Whether to enable mind-over-matter
     */
    void enableMindOverMatter(bool enabled = true);

    /**
     * @brief Enable telekinesis
     * @param enabled Whether to enable telekinesis
     */
    void enableTelekinesis(bool enabled = true);

    /**
     * @brief Enable psychic phenomena
     * @param enabled Whether to enable psychic features
     */
    void enablePsychicPhenomena(bool enabled = true);

    /**
     * @brief Enable god mode
     * @param enabled Whether to enable god mode
     */
    void enableGodMode(bool enabled = true);

    /**
     * @brief Enable omnipresence
     * @param enabled Whether to enable omnipresence
     */
    void enableOmnipresence(bool enabled = true);

    /**
     * @brief Enable omniscience
     * @param enabled Whether to enable omniscience
     */
    void enableOmniscience(bool enabled = true);

    /**
     * @brief Enable time travel
     * @param enabled Whether to enable time travel
     */
    void enableTimeTravel(bool enabled = true);

    /**
     * @brief Link to consciousness
     * @param consciousness Pointer to consciousness simulator
     */
    void linkToConsciousness(void* consciousness);

    /**
     * @brief Get reality coherence
     * @return Reality coherence score (0.0 to 1.0)
     */
    float getRealityCoherence() const;

    /**
     * @brief Check if reality is stable
     * @return true if reality is stable
     */
    bool isRealityStable() const;

    /**
     * @brief Detect reality anomalies
     * @return Vector of anomaly descriptions
     */
    std::vector<std::string> detectAnomalies();

    /**
     * @brief Emergency reality reset
     */
    void emergencyRealityReset();

    /**
     * @brief Get quantum physics engine
     * @return Pointer to quantum physics engine
     */
    QuantumPhysicsEngine* getPhysicsEngine() const { return m_physicsEngine.get(); }

    /**
     * @brief Get dimensional rendering engine
     * @return Pointer to dimensional rendering engine
     */
    DimensionalRenderingEngine* getRenderingEngine() const { return m_renderingEngine.get(); }

    /**
     * @brief Get reality distortion field
     * @return Pointer to reality distortion field
     */
    RealityDistortionField* getDistortionField() const { return m_distortionField.get(); }

    /**
     * @brief Run reality benchmark suite
     * @return Benchmark results
     */
    struct RealityBenchmarkResults {
        float physicsAccuracy;
        float renderingPerformance;
        float quantumCoherence;
        float realityStability;
        float dimensionalComplexity;
        float overallRealityScore;
        float spacetimeDistortion;
        float causalityIntegrity;
    };
    RealityBenchmarkResults runRealityBenchmark();

private:
    // ULTIMATE Reality components
    std::unique_ptr<QuantumPhysicsEngine> m_physicsEngine;
    std::unique_ptr<DimensionalRenderingEngine> m_renderingEngine;
    std::unique_ptr<RealityDistortionField> m_distortionField;

    // ULTIMATE State management
    std::atomic<RealityMode> m_realityMode{RealityMode::TRANSCENDENT_REALITY};
    std::atomic<bool> m_realityBranchingEnabled{true};
    std::atomic<bool> m_mindOverMatterEnabled{true};
    std::atomic<bool> m_telekinesisEnabled{true};
    std::atomic<bool> m_psychicPhenomenaEnabled{true};
    std::atomic<bool> m_godModeEnabled{true};
    std::atomic<bool> m_omnipresenceEnabled{true};
    std::atomic<bool> m_omniscienceEnabled{true};
    std::atomic<bool> m_timeTravelEnabled{true};

    // ULTIMATE Performance tracking
    std::atomic<float> m_realityCoherence{1.0f};
    std::atomic<bool> m_realityStable{true};
    std::chrono::high_resolution_clock::time_point m_lastRealityCheck;
};

/**
 * @brief Quantum Physics Engine - Quantum physics simulation at particle level
 * 
 * Features:
 * - Quantum particle simulation
 * - Wave function collapse
 * - Quantum entanglement simulation
 * - Uncertainty principle implementation
 * - Quantum tunneling effects
 * - Reality-bending physics laws
 */
class QuantumPhysicsEngine {
public:
    /**
     * @brief Constructor
     */
    QuantumPhysicsEngine();

    /**
     * @brief Destructor
     */
    ~QuantumPhysicsEngine();

    /**
     * @brief Initialize quantum physics engine
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Simulate quantum particle
     * @param particleType Type of particle
     * @param position Initial position
     * @param momentum Initial momentum
     * @return Particle simulation result
     */
    struct ParticleSimulation {
        std::vector<float> position;
        std::vector<float> momentum;
        float energy;
        float quantumState;
        bool entangled;
    };
    ParticleSimulation simulateParticle(const std::string& particleType, 
                                       const std::vector<float>& position,
                                       const std::vector<float>& momentum);

    /**
     * @brief Collapse wave function
     * @param waveFunction Wave function to collapse
     * @param measurementType Type of measurement
     * @return Collapsed state
     */
    std::vector<float> collapseWaveFunction(const std::vector<float>& waveFunction,
                                           const std::string& measurementType);

    /**
     * @brief Create quantum entanglement
     * @param particle1 First particle
     * @param particle2 Second particle
     * @return Entanglement strength
     */
    float createEntanglement(ParticleSimulation& particle1, ParticleSimulation& particle2);

    /**
     * @brief Apply uncertainty principle
     * @param position Position uncertainty
     * @param momentum Momentum uncertainty
     * @return Uncertainty product
     */
    float applyUncertaintyPrinciple(float position, float momentum);

    /**
     * @brief Enable quantum tunneling
     * @param enabled Whether to enable tunneling
     */
    void enableQuantumTunneling(bool enabled = true);

    /**
     * @brief Get simulation accuracy
     * @return Simulation accuracy (0.0 to 1.0)
     */
    float getSimulationAccuracy() const;

private:
    std::atomic<bool> m_tunnelingEnabled{true};
    std::atomic<float> m_simulationAccuracy{1.0f};
    std::vector<ParticleSimulation> m_activeParticles;
};

/**
 * @brief Dimensional Rendering Engine - Rendering in multiple dimensions
 * 
 * Features:
 * - 4D, 5D, and higher dimensional rendering
 * - Hyperbolic geometry rendering
 * - Fractal geometry with infinite detail
 * - Quantum superposition rendering
 * - Parallel universe visualization
 * - Reality-bending visual effects
 */
class DimensionalRenderingEngine {
public:
    /**
     * @brief Constructor
     */
    DimensionalRenderingEngine();

    /**
     * @brief Destructor
     */
    ~DimensionalRenderingEngine();

    /**
     * @brief Initialize dimensional rendering
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Set rendering dimension
     * @param dimension Target dimension (3, 4, 5, etc.)
     */
    void setDimension(int dimension);

    /**
     * @brief Enable GPU acceleration
     * @param enabled Whether to enable GPU acceleration
     */
    void enableGPUAcceleration(bool enabled = true);

    /**
     * @brief Set render quality
     * @param quality Quality level (1-10)
     */
    void setRenderQuality(int quality);

    /**
     * @brief Enable holographic projection
     * @param enabled Whether to enable holographic features
     */
    void enableHolographicProjection(bool enabled = true);

    /**
     * @brief Enable fractal rendering
     * @param enabled Whether to enable fractal features
     */
    void enableFractalRendering(bool enabled = true);

    /**
     * @brief Enable hyperbolic geometry
     * @param enabled Whether to enable hyperbolic features
     */
    void enableHyperbolicGeometry(bool enabled = true);

    /**
     * @brief Render parallel universes
     * @param universeIds Vector of universe IDs to render
     */
    void renderParallelUniverses(const std::vector<uint32_t>& universeIds);

    /**
     * @brief Get dimensional complexity
     * @return Dimensional complexity score
     */
    float getDimensionalComplexity() const;

    /**
     * @brief Render frame in multiple dimensions
     * @param scene Scene data
     * @param output Output buffer
     * @param width Image width
     * @param height Image height
     * @param dimensions Vector of dimensions to render
     */
    void renderMultiDimensionalFrame(const void* scene, void* output, 
                                   int width, int height,
                                   const std::vector<int>& dimensions);

private:
    std::atomic<int> m_currentDimension{4};
    std::atomic<bool> m_gpuAccelerationEnabled{true};
    std::atomic<int> m_renderQuality{10};
    std::atomic<bool> m_holographicEnabled{true};
    std::atomic<bool> m_fractalEnabled{true};
    std::atomic<bool> m_hyperbolicEnabled{true};
    std::atomic<float> m_dimensionalComplexity{1.0f};
    std::vector<uint32_t> m_activeUniverses;
};

/**
 * @brief Reality Distortion Field - Spacetime manipulation and reality control
 * 
 * Features:
 * - Spacetime warp and distortion
 * - Gravity well creation
 * - Time dilation effects
 * - Dimensional shifts
 * - Quantum fluctuations
 * - Reality-bending field effects
 */
class RealityDistortionField {
public:
    /**
     * @brief Constructor
     */
    RealityDistortionField();

    /**
     * @brief Destructor
     */
    ~RealityDistortionField();

    /**
     * @brief Initialize reality distortion field
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Create spacetime warp
     * @param position Warp position
     * @param intensity Warp intensity
     * @return Warp field ID
     */
    uint32_t createSpacetimeWarp(const std::vector<float>& position, float intensity);

    /**
     * @brief Create gravity well
     * @param position Well position
     * @param mass Well mass
     * @return Gravity well ID
     */
    uint32_t createGravityWell(const std::vector<float>& position, float mass);

    /**
     * @brief Apply time dilation
     * @param position Dilation position
     * @param factor Dilation factor (0.1 = 10x slower, 2.0 = 2x faster)
     */
    void applyTimeDilation(const std::vector<float>& position, float factor);

    /**
     * @brief Create dimensional shift
     * @param position Shift position
     * @param targetDimension Target dimension
     */
    void createDimensionalShift(const std::vector<float>& position, int targetDimension);

    /**
     * @brief Generate quantum fluctuations
     * @param position Fluctuation position
     * @param intensity Fluctuation intensity
     */
    void generateQuantumFluctuations(const std::vector<float>& position, float intensity);

    /**
     * @brief Alter probability field
     * @param position Field position
     * @param event Event to alter
     * @param newProbability New probability (0.0 to 1.0)
     */
    void alterProbabilityField(const std::vector<float>& position, 
                             const std::string& event, float newProbability);

    /**
     * @brief Create causality loop
     * @param startPosition Loop start position
     * @param endPosition Loop end position
     * @return Loop ID
     */
    uint32_t createCausalityLoop(const std::vector<float>& startPosition,
                                const std::vector<float>& endPosition);

    /**
     * @brief Get field strength
     * @return Field strength (0.0 to 1.0)
     */
    float getFieldStrength() const;

    /**
     * @brief Get distortion intensity
     * @return Distortion intensity (0.0 to 1.0)
     */
    float getDistortionIntensity() const;

private:
    struct DistortionField {
        uint32_t id;
        std::vector<float> position;
        float intensity;
        std::string type;
        std::atomic<bool> active{true};
    };

    std::vector<DistortionField> m_activeFields;
    std::atomic<float> m_fieldStrength{1.0f};
    std::atomic<float> m_distortionIntensity{1.0f};
    std::atomic<uint32_t> m_nextFieldId{1};
};

} // namespace aisis 