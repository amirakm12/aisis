#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <future>
#include <omp.h>

#include "core/Application.h"
#include "core/PerformanceManager.h"
#include "core/ThreadPool.h"
#include "graphics/RenderEngine.h"
#include "audio/AudioEngine.h"
#include "ai/AIProcessor.h"
#include "networking/NetworkManager.h"
#include "ui/UIManager.h"

using namespace aisis;

int main(int argc, char* argv[]) {
    std::cout << "AISIS Creative Studio v2.0.0 - High Performance Edition" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    try {
        // Initialize performance monitoring
        auto perfManager = std::make_unique<PerformanceManager>();
        perfManager->startMonitoring();
        
        // Configure OpenMP for maximum performance
        int numThreads = std::thread::hardware_concurrency();
        omp_set_num_threads(numThreads);
        std::cout << "Configured OpenMP with " << numThreads << " threads" << std::endl;
        
        // Initialize thread pool for async operations
        auto threadPool = std::make_unique<ThreadPool>(numThreads * 2);
        
        // Create application instance with optimized settings
        Application app(argc, argv);
        app.setPerformanceManager(std::move(perfManager));
        app.setThreadPool(std::move(threadPool));
        
        // Initialize all subsystems in parallel for faster startup
        std::vector<std::future<bool>> initTasks;
        
        initTasks.push_back(std::async(std::launch::async, [&app]() {
            return app.initializeGraphics();
        }));
        
        initTasks.push_back(std::async(std::launch::async, [&app]() {
            return app.initializeAudio();
        }));
        
        initTasks.push_back(std::async(std::launch::async, [&app]() {
            return app.initializeAI();
        }));
        
        initTasks.push_back(std::async(std::launch::async, [&app]() {
            return app.initializeNetworking();
        }));
        
        initTasks.push_back(std::async(std::launch::async, [&app]() {
            return app.initializeUI();
        }));
        
        // Wait for all subsystems to initialize
        bool allInitialized = true;
        for (auto& task : initTasks) {
            if (!task.get()) {
                allInitialized = false;
            }
        }
        
        if (!allInitialized) {
            std::cerr << "Failed to initialize one or more subsystems!" << std::endl;
            return -1;
        }
        
        std::cout << "All subsystems initialized successfully!" << std::endl;
        
        // Start the main application loop
        return app.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred!" << std::endl;
        return -1;
    }
}