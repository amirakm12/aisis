#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <future>

namespace Ultimate {
namespace AI {

// Forward declarations
class NeuralNetwork;
class AIModel;
class DataSet;

// AI processing modes
enum class ProcessingMode {
    CPU,
    GPU,
    NPU,
    Hybrid,
    Distributed
};

// AI model types
enum class ModelType {
    Classification,
    Regression,
    NeuralNetwork,
    DeepLearning,
    ReinforcementLearning,
    NaturalLanguageProcessing,
    ComputerVision,
    GenerativeAI
};

// Optimization levels
enum class OptimizationLevel {
    None = 0,
    Basic = 1,
    Moderate = 2,
    Aggressive = 3,
    Extreme = 4
};

struct AIStats {
    int activeModels = 0;
    int totalInferences = 0;
    double averageInferenceTime = 0.0;
    double cpuUsage = 0.0;
    double gpuUsage = 0.0;
    size_t memoryUsage = 0;
    size_t vramUsage = 0;
    int queuedTasks = 0;
};

struct InferenceResult {
    bool success = false;
    std::vector<float> output;
    double confidence = 0.0;
    double processingTime = 0.0;
    std::string errorMessage;
};

class AIProcessor {
public:
    static AIProcessor& getInstance();
    
    // Initialization and cleanup
    bool initialize(ProcessingMode mode = ProcessingMode::Hybrid);
    void shutdown();
    
    // Model management
    std::shared_ptr<AIModel> loadModel(const std::string& filepath, ModelType type);
    std::shared_ptr<AIModel> createModel(ModelType type, const std::string& architecture);
    void unloadModel(const std::string& modelId);
    void unloadAllModels();
    
    std::shared_ptr<AIModel> getModel(const std::string& modelId) const;
    std::vector<std::string> getLoadedModels() const;
    
    // Inference
    InferenceResult runInference(const std::string& modelId, 
                               const std::vector<float>& input);
    
    std::future<InferenceResult> runInferenceAsync(const std::string& modelId,
                                                  const std::vector<float>& input);
    
    std::vector<InferenceResult> runBatchInference(const std::string& modelId,
                                                  const std::vector<std::vector<float>>& inputs);
    
    // Training
    bool startTraining(const std::string& modelId, 
                      std::shared_ptr<DataSet> trainingData,
                      std::shared_ptr<DataSet> validationData = nullptr);
    
    void stopTraining(const std::string& modelId);
    bool isTraining(const std::string& modelId) const;
    
    double getTrainingProgress(const std::string& modelId) const;
    double getTrainingLoss(const std::string& modelId) const;
    double getValidationAccuracy(const std::string& modelId) const;
    
    // Model optimization
    void optimizeModel(const std::string& modelId, OptimizationLevel level);
    void quantizeModel(const std::string& modelId, int bits = 8);
    void pruneModel(const std::string& modelId, float threshold = 0.1f);
    
    // Processing configuration
    void setProcessingMode(ProcessingMode mode);
    ProcessingMode getProcessingMode() const;
    
    void setMaxConcurrentInferences(int maxInferences);
    int getMaxConcurrentInferences() const;
    
    void setBatchSize(int batchSize);
    int getBatchSize() const;
    
    // Performance tuning
    void enableAutoOptimization(bool enable);
    bool isAutoOptimizationEnabled() const;
    
    void setMemoryLimit(size_t bytes);
    size_t getMemoryLimit() const;
    
    void enableModelCaching(bool enable);
    bool isModelCachingEnabled() const;
    
    // Advanced features
    void enableTensorRT(bool enable);
    bool isTensorRTEnabled() const;
    
    void enableCUDA(bool enable);
    bool isCUDAEnabled() const;
    
    void enableOpenVINO(bool enable);
    bool isOpenVINOEnabled() const;
    
    void enableDirectML(bool enable);
    bool isDirectMLEnabled() const;
    
    // Data preprocessing
    std::vector<float> preprocessImage(const std::string& imagePath, 
                                     int targetWidth, int targetHeight);
    
    std::vector<float> preprocessText(const std::string& text, 
                                    const std::string& tokenizer = "default");
    
    std::vector<float> preprocessAudio(const std::vector<float>& audioData,
                                     int sampleRate, int targetLength);
    
    // Model evaluation
    double evaluateModel(const std::string& modelId, 
                        std::shared_ptr<DataSet> testData);
    
    std::unordered_map<std::string, double> getModelMetrics(const std::string& modelId);
    
    // Distributed processing
    void enableDistributedProcessing(bool enable);
    bool isDistributedProcessingEnabled() const;
    
    void addWorkerNode(const std::string& address, int port);
    void removeWorkerNode(const std::string& address);
    std::vector<std::string> getWorkerNodes() const;
    
    // Statistics and monitoring
    const AIStats& getAIStats() const;
    void resetAIStats();
    
    // Callbacks
    using InferenceCallback = std::function<void(const std::string&, const InferenceResult&)>;
    using TrainingCallback = std::function<void(const std::string&, double, double)>;
    using ErrorCallback = std::function<void(const std::string&, const std::string&)>;
    
    void setInferenceCallback(InferenceCallback callback);
    void setTrainingCallback(TrainingCallback callback);
    void setErrorCallback(ErrorCallback callback);
    
    // Hardware detection
    std::vector<std::string> getAvailableDevices() const;
    void setPreferredDevice(const std::string& device);
    std::string getPreferredDevice() const;
    
    // Model serialization
    bool saveModel(const std::string& modelId, const std::string& filepath);
    bool exportModel(const std::string& modelId, const std::string& format,
                    const std::string& filepath);

private:
    AIProcessor() = default;
    ~AIProcessor() = default;
    AIProcessor(const AIProcessor&) = delete;
    AIProcessor& operator=(const AIProcessor&) = delete;
    
    // Internal state
    bool m_initialized = false;
    ProcessingMode m_processingMode = ProcessingMode::Hybrid;
    
    // Models
    std::unordered_map<std::string, std::shared_ptr<AIModel>> m_models;
    std::unordered_map<std::string, bool> m_trainingStatus;
    
    // Configuration
    int m_maxConcurrentInferences = 4;
    int m_batchSize = 1;
    bool m_autoOptimizationEnabled = true;
    size_t m_memoryLimit = 2ULL * 1024 * 1024 * 1024; // 2GB
    bool m_modelCachingEnabled = true;
    
    // Hardware acceleration
    bool m_tensorRTEnabled = false;
    bool m_cudaEnabled = false;
    bool m_openVINOEnabled = false;
    bool m_directMLEnabled = false;
    
    // Distributed processing
    bool m_distributedProcessingEnabled = false;
    std::vector<std::string> m_workerNodes;
    
    // Statistics
    AIStats m_aiStats;
    
    // Callbacks
    InferenceCallback m_inferenceCallback;
    TrainingCallback m_trainingCallback;
    ErrorCallback m_errorCallback;
    
    // Hardware
    std::string m_preferredDevice = "auto";
    
    // Internal methods
    void updateAIStats();
    void initializeHardwareAcceleration();
    void optimizeMemoryUsage();
    std::string generateModelId();
};

} // namespace AI
} // namespace Ultimate