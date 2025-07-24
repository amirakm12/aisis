#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <string>
#include <future>

namespace aisis {

struct AIModel {
    enum Type {
        OBJECT_DETECTION, FACE_RECOGNITION, STYLE_TRANSFER, SUPER_RESOLUTION,
        AUDIO_ENHANCEMENT, NOISE_REDUCTION, CONTENT_GENERATION, TEXT_TO_SPEECH,
        SPEECH_TO_TEXT, LANGUAGE_TRANSLATION, SENTIMENT_ANALYSIS, IMAGE_CLASSIFICATION
    };
    
    Type type;
    std::string modelPath;
    std::string configPath;
    cv::dnn::Net network;
    bool loaded{false};
    bool gpuAccelerated{false};
    float confidence{0.5f};
};

struct DetectionResult {
    int classId;
    std::string className;
    float confidence;
    cv::Rect boundingBox;
    std::vector<cv::Point> keypoints;
    cv::Mat mask;
};

struct ProcessingTask {
    enum Type { IMAGE, AUDIO, TEXT, VIDEO };
    Type type;
    std::string inputPath;
    std::string outputPath;
    std::unordered_map<std::string, float> parameters;
    std::function<void(bool, const std::string&)> callback;
    std::chrono::high_resolution_clock::time_point timestamp;
};

class AIProcessor {
public:
    AIProcessor();
    ~AIProcessor();
    
    // Initialization
    bool initialize();
    void shutdown();
    
    // Model management
    uint32_t loadModel(AIModel::Type type, const std::string& modelPath, const std::string& configPath = "");
    void unloadModel(uint32_t modelId);
    bool isModelLoaded(uint32_t modelId) const;
    void enableGPUAcceleration(uint32_t modelId, bool enabled);
    void setModelConfidence(uint32_t modelId, float confidence);
    
    // Computer Vision
    std::vector<DetectionResult> detectObjects(uint32_t modelId, const cv::Mat& image);
    std::vector<DetectionResult> recognizeFaces(uint32_t modelId, const cv::Mat& image);
    cv::Mat enhanceImage(uint32_t modelId, const cv::Mat& image);
    cv::Mat upscaleImage(uint32_t modelId, const cv::Mat& image, float scale = 2.0f);
    cv::Mat applyStyleTransfer(uint32_t modelId, const cv::Mat& content, const cv::Mat& style);
    cv::Mat removeBackground(uint32_t modelId, const cv::Mat& image);
    cv::Mat colorizeImage(uint32_t modelId, const cv::Mat& grayscaleImage);
    
    // Audio Processing
    std::vector<float> enhanceAudio(uint32_t modelId, const std::vector<float>& audio, int sampleRate);
    std::vector<float> reduceNoise(uint32_t modelId, const std::vector<float>& audio, int sampleRate);
    std::vector<float> separateVocals(uint32_t modelId, const std::vector<float>& audio, int sampleRate);
    std::string speechToText(uint32_t modelId, const std::vector<float>& audio, int sampleRate);
    std::vector<float> textToSpeech(uint32_t modelId, const std::string& text, const std::string& voice = "default");
    
    // Content Generation
    cv::Mat generateImage(uint32_t modelId, const std::string& prompt, int width = 512, int height = 512);
    std::string generateText(uint32_t modelId, const std::string& prompt, int maxLength = 100);
    std::vector<float> generateMusic(uint32_t modelId, const std::string& style, float duration = 30.0f);
    cv::Mat inpaintImage(uint32_t modelId, const cv::Mat& image, const cv::Mat& mask);
    
    // Natural Language Processing
    std::string translateText(uint32_t modelId, const std::string& text, const std::string& fromLang, const std::string& toLang);
    float analyzeSentiment(uint32_t modelId, const std::string& text);
    std::vector<std::string> extractKeywords(uint32_t modelId, const std::string& text);
    std::string summarizeText(uint32_t modelId, const std::string& text, float ratio = 0.3f);
    
    // Video Processing
    bool processVideo(uint32_t modelId, const std::string& inputPath, const std::string& outputPath, 
                     const std::function<cv::Mat(const cv::Mat&)>& frameProcessor);
    std::vector<DetectionResult> trackObjects(uint32_t modelId, const std::string& videoPath);
    cv::Mat stabilizeVideo(uint32_t modelId, const std::string& inputPath, const std::string& outputPath);
    
    // Intelligent Automation
    void enableSmartAssistant(bool enabled);
    void setAutoEnhancement(bool enabled);
    void enableContentSuggestions(bool enabled);
    void setWorkflowAutomation(bool enabled);
    std::vector<std::string> getContentSuggestions(const std::string& context);
    void automateWorkflow(const std::string& workflowName);
    
    // Real-time Processing
    void enableRealTimeProcessing(bool enabled);
    void startRealTimeCamera(int cameraId = 0);
    void stopRealTimeCamera();
    void setRealTimeCallback(std::function<void(const cv::Mat&, const std::vector<DetectionResult>&)> callback);
    
    // Batch Processing
    uint32_t submitBatchTask(const ProcessingTask& task);
    void cancelBatchTask(uint32_t taskId);
    bool isBatchTaskComplete(uint32_t taskId) const;
    float getBatchProgress(uint32_t taskId) const;
    void setBatchPriority(uint32_t taskId, int priority);
    
    // Performance and Optimization
    void enableMultiThreading(bool enabled);
    void setThreadCount(int threads);
    void enableMemoryOptimization(bool enabled);
    void setProcessingQuality(int quality); // 1-10 scale
    void enableModelCaching(bool enabled);
    
    // Training and Fine-tuning
    bool startTraining(uint32_t modelId, const std::string& datasetPath);
    void stopTraining(uint32_t modelId);
    float getTrainingProgress(uint32_t modelId) const;
    bool saveTrainedModel(uint32_t modelId, const std::string& outputPath);
    
    // Analytics and Monitoring
    float getProcessingTime(uint32_t modelId) const;
    size_t getMemoryUsage(uint32_t modelId) const;
    float getGPUUtilization() const;
    int getQueueSize() const;
    std::vector<std::string> getPerformanceMetrics() const;
    
    // Cloud Integration
    bool enableCloudProcessing(bool enabled);
    bool uploadModelToCloud(uint32_t modelId, const std::string& cloudProvider);
    bool downloadModelFromCloud(const std::string& modelId, const std::string& cloudProvider);
    void setCloudCredentials(const std::string& provider, const std::unordered_map<std::string, std::string>& credentials);
    
private:
    // Core components
    std::unordered_map<uint32_t, std::unique_ptr<AIModel>> m_models;
    std::queue<ProcessingTask> m_taskQueue;
    std::unordered_map<uint32_t, std::future<void>> m_runningTasks;
    
    // Threading
    std::vector<std::thread> m_workerThreads;
    std::thread m_realTimeThread;
    std::atomic<bool> m_processing{false};
    std::atomic<bool> m_realTimeEnabled{false};
    std::mutex m_queueMutex;
    std::condition_variable m_queueCondition;
    
    // Real-time processing
    cv::VideoCapture m_camera;
    std::function<void(const cv::Mat&, const std::vector<DetectionResult>&)> m_realTimeCallback;
    std::atomic<bool> m_cameraRunning{false};
    
    // Performance settings
    bool m_multiThreadingEnabled{true};
    int m_threadCount{4};
    bool m_memoryOptimizationEnabled{true};
    int m_processingQuality{8};
    bool m_modelCachingEnabled{true};
    
    // Smart features
    bool m_smartAssistantEnabled{true};
    bool m_autoEnhancementEnabled{false};
    bool m_contentSuggestionsEnabled{true};
    bool m_workflowAutomationEnabled{false};
    
    // Cloud settings
    bool m_cloudProcessingEnabled{false};
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> m_cloudCredentials;
    
    // Performance monitoring
    std::unordered_map<uint32_t, std::chrono::high_resolution_clock::time_point> m_processingStartTimes;
    std::unordered_map<uint32_t, float> m_processingTimes;
    std::unordered_map<uint32_t, size_t> m_memoryUsages;
    std::atomic<float> m_gpuUtilization{0.0f};
    
    // ID generation
    std::atomic<uint32_t> m_nextModelId{1};
    std::atomic<uint32_t> m_nextTaskId{1};
    
    // Worker methods
    void workerThreadFunction();
    void realTimeThreadFunction();
    void processTask(const ProcessingTask& task);
    
    // Model utilities
    bool loadDNNModel(AIModel& model);
    void preprocessImage(const cv::Mat& input, cv::Mat& output, const AIModel& model);
    void postprocessDetections(const cv::Mat& output, std::vector<DetectionResult>& results, const AIModel& model);
    
    // Performance optimization
    void optimizeModel(uint32_t modelId);
    void updatePerformanceMetrics();
    void cleanupResources();
    
    // Smart assistant
    void analyzeContext(const std::string& context);
    void generateSuggestions(const std::string& context);
    void executeAutomation(const std::string& action);
    
    // Cloud integration
    bool uploadToCloud(const std::string& data, const std::string& provider);
    std::string downloadFromCloud(const std::string& id, const std::string& provider);
    bool authenticateCloudProvider(const std::string& provider);
};

} // namespace aisis