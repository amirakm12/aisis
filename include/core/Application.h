#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <functional>

namespace aisis {

class PerformanceManager;
class ThreadPool;
class RenderEngine;
class AudioEngine;
class AIProcessor;
class NetworkManager;
class UIManager;
class Project;

class Application {
public:
    Application(int argc, char* argv[]);
    ~Application();
    
    // Initialization methods
    bool initializeGraphics();
    bool initializeAudio();
    bool initializeAI();
    bool initializeNetworking();
    bool initializeUI();
    
    // Core functionality
    int run();
    void shutdown();
    
    // Performance management
    void setPerformanceManager(std::unique_ptr<PerformanceManager> manager);
    void setThreadPool(std::unique_ptr<ThreadPool> pool);
    
    // Project management
    bool createProject(const std::string& name, const std::string& type);
    bool loadProject(const std::string& path);
    bool saveProject();
    bool exportProject(const std::string& format, const std::string& path);
    
    // Real-time collaboration
    bool startCollaborationSession();
    bool joinCollaborationSession(const std::string& sessionId);
    void endCollaborationSession();
    
    // Plugin system
    bool loadPlugin(const std::string& path);
    void unloadPlugin(const std::string& name);
    std::vector<std::string> getAvailablePlugins() const;
    
    // Performance features
    void enableGPUAcceleration(bool enable);
    void setRenderQuality(int quality); // 1-10 scale
    void enableMultiThreading(bool enable);
    void setMemoryOptimization(bool enable);
    
    // Advanced features
    bool enableAIAssistant(bool enable);
    void setAutoSaveInterval(int seconds);
    bool enableCloudSync(bool enable);
    void setWorkspaceLayout(const std::string& layout);
    
    // Event system
    using EventCallback = std::function<void(const std::string&, const std::unordered_map<std::string, std::string>&)>;
    void registerEventCallback(const std::string& eventType, EventCallback callback);
    void triggerEvent(const std::string& eventType, const std::unordered_map<std::string, std::string>& data);
    
    // Getters
    RenderEngine* getRenderEngine() const { return m_renderEngine.get(); }
    AudioEngine* getAudioEngine() const { return m_audioEngine.get(); }
    AIProcessor* getAIProcessor() const { return m_aiProcessor.get(); }
    NetworkManager* getNetworkManager() const { return m_networkManager.get(); }
    UIManager* getUIManager() const { return m_uiManager.get(); }
    
private:
    // Core systems
    std::unique_ptr<PerformanceManager> m_performanceManager;
    std::unique_ptr<ThreadPool> m_threadPool;
    std::unique_ptr<RenderEngine> m_renderEngine;
    std::unique_ptr<AudioEngine> m_audioEngine;
    std::unique_ptr<AIProcessor> m_aiProcessor;
    std::unique_ptr<NetworkManager> m_networkManager;
    std::unique_ptr<UIManager> m_uiManager;
    
    // Application state
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_initialized{false};
    std::unique_ptr<Project> m_currentProject;
    
    // Performance settings
    bool m_gpuAcceleration{true};
    int m_renderQuality{8};
    bool m_multiThreading{true};
    bool m_memoryOptimization{true};
    
    // Advanced features
    bool m_aiAssistantEnabled{true};
    int m_autoSaveInterval{30}; // seconds
    bool m_cloudSyncEnabled{false};
    std::string m_workspaceLayout{"default"};
    
    // Event system
    std::unordered_map<std::string, std::vector<EventCallback>> m_eventCallbacks;
    
    // Command line arguments
    std::vector<std::string> m_args;
    
    // Main loop
    void mainLoop();
    void update(double deltaTime);
    void render();
    
    // Initialization helpers
    bool parseCommandLineArgs();
    bool loadConfiguration();
    bool initializeSubsystems();
    
    // Performance optimization
    void optimizeMemoryUsage();
    void updatePerformanceMetrics();
    void adjustQualitySettings();
};

} // namespace aisis