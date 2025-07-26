#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace Ultimate {
namespace Core {

class Application {
public:
    // Singleton pattern
    static Application& getInstance();
    
    // Application lifecycle
    bool initialize(int argc, char* argv[]);
    void run();
    void shutdown();
    
    // Core functionality
    void setTitle(const std::string& title);
    const std::string& getTitle() const;
    
    bool isRunning() const;
    void requestExit();
    
    // Event handling
    using EventCallback = std::function<void()>;
    void setUpdateCallback(EventCallback callback);
    void setRenderCallback(EventCallback callback);
    
    // Time management
    double getDeltaTime() const;
    double getTotalTime() const;
    
    // Configuration
    void setTargetFPS(int fps);
    int getTargetFPS() const;
    
    // Resource management
    void addResourcePath(const std::string& path);
    std::string getResourcePath(const std::string& filename) const;

private:
    Application() = default;
    ~Application() = default;
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    
    // Internal state
    bool m_initialized = false;
    bool m_running = false;
    std::string m_title = "ULTIMATE System";
    int m_targetFPS = 60;
    
    // Timing
    double m_deltaTime = 0.0;
    double m_totalTime = 0.0;
    
    // Callbacks
    EventCallback m_updateCallback;
    EventCallback m_renderCallback;
    
    // Resource paths
    std::vector<std::string> m_resourcePaths;
    
    // Internal methods
    void updateTiming();
    void processEvents();
    void update();
    void render();
};

} // namespace Core
} // namespace Ultimate