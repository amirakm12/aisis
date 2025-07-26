#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>
#include <chrono>

namespace Ultimate {
namespace Core {

// Task priority levels
enum class TaskPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
    Immediate = 5
};

// Thread pool modes
enum class PoolMode {
    Fixed,      // Fixed number of threads
    Dynamic,    // Threads created/destroyed based on load
    Adaptive    // Automatically adjusts based on system performance
};

// Task execution policies
enum class ExecutionPolicy {
    FIFO,       // First In, First Out
    LIFO,       // Last In, First Out
    Priority,   // Based on task priority
    WorkStealing // Work-stealing algorithm
};

// Task statistics
struct TaskStats {
    int totalTasks = 0;
    int completedTasks = 0;
    int failedTasks = 0;
    int queuedTasks = 0;
    int activeTasks = 0;
    double averageExecutionTime = 0.0;
    double totalExecutionTime = 0.0;
};

// Thread statistics
struct ThreadStats {
    int totalThreads = 0;
    int activeThreads = 0;
    int idleThreads = 0;
    double averageCpuUsage = 0.0;
    double totalCpuTime = 0.0;
};

// Task wrapper with metadata
template<typename T>
struct Task {
    std::function<T()> function;
    TaskPriority priority = TaskPriority::Normal;
    std::chrono::steady_clock::time_point submitTime;
    std::chrono::milliseconds timeout{0};
    std::string name;
    int retryCount = 0;
    int maxRetries = 0;
    
    Task(std::function<T()> f, TaskPriority p = TaskPriority::Normal, 
         const std::string& n = "", std::chrono::milliseconds t = std::chrono::milliseconds{0})
        : function(std::move(f)), priority(p), submitTime(std::chrono::steady_clock::now()), 
          timeout(t), name(n) {}
};

class ThreadPool {
public:
    // Constructor with configuration
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency(),
                       PoolMode mode = PoolMode::Fixed,
                       ExecutionPolicy policy = ExecutionPolicy::FIFO);
    
    // Destructor
    ~ThreadPool();
    
    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    
    // Task submission
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    template<class F, class... Args>
    auto enqueue(TaskPriority priority, F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    template<class F, class... Args>
    auto enqueue(TaskPriority priority, const std::string& name, F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    template<class F, class... Args>
    auto enqueue(TaskPriority priority, const std::string& name, 
                std::chrono::milliseconds timeout, F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // Batch task submission
    template<class F>
    std::vector<std::future<typename std::result_of<F()>::type>>
    enqueueBatch(const std::vector<F>& tasks, TaskPriority priority = TaskPriority::Normal);
    
    // Parallel algorithms
    template<class Iterator, class Function>
    void parallelFor(Iterator first, Iterator last, Function func, 
                    size_t chunkSize = 0);
    
    template<class Iterator, class Function, class Reduction>
    auto parallelReduce(Iterator first, Iterator last, Function func, 
                       Reduction reduction) -> decltype(reduction());
    
    // Pool management
    void resize(size_t numThreads);
    void setMode(PoolMode mode);
    void setExecutionPolicy(ExecutionPolicy policy);
    
    size_t size() const;
    size_t activeThreads() const;
    size_t queueSize() const;
    
    // Thread affinity
    void setThreadAffinity(size_t threadIndex, const std::vector<int>& cpuCores);
    void setAllThreadsAffinity(const std::vector<int>& cpuCores);
    
    // Priority and scheduling
    void setThreadPriority(int priority);
    int getThreadPriority() const;
    
    void enableWorkStealing(bool enable);
    bool isWorkStealingEnabled() const;
    
    // Performance tuning
    void setMaxQueueSize(size_t maxSize);
    size_t getMaxQueueSize() const;
    
    void setIdleTimeout(std::chrono::milliseconds timeout);
    std::chrono::milliseconds getIdleTimeout() const;
    
    void enableDynamicScaling(bool enable);
    bool isDynamicScalingEnabled() const;
    
    void setScalingThresholds(double scaleUpThreshold, double scaleDownThreshold);
    std::pair<double, double> getScalingThresholds() const;
    
    // Task management
    void cancelAllTasks();
    void cancelTasksByName(const std::string& name);
    void cancelTasksByPriority(TaskPriority priority);
    
    bool waitForCompletion(std::chrono::milliseconds timeout = std::chrono::milliseconds{0});
    void waitForIdle();
    
    // Statistics and monitoring
    TaskStats getTaskStats() const;
    ThreadStats getThreadStats() const;
    void resetStats();
    
    // Performance monitoring
    double getCpuUsage() const;
    double getMemoryUsage() const;
    double getThroughput() const; // Tasks per second
    
    // Thread pool state
    bool isRunning() const;
    bool isIdle() const;
    
    void pause();
    void resume();
    bool isPaused() const;
    
    void shutdown();
    void shutdownNow();
    
    // Callbacks
    using TaskCallback = std::function<void(const std::string&, bool)>;
    using ThreadCallback = std::function<void(size_t, const std::string&)>;
    
    void setTaskCompletionCallback(TaskCallback callback);
    void setThreadEventCallback(ThreadCallback callback);
    
    // Exception handling
    void setExceptionHandler(std::function<void(const std::exception&)> handler);
    
    // Debugging and profiling
    void enableProfiling(bool enable);
    bool isProfilingEnabled() const;
    
    std::vector<std::string> getActiveTaskNames() const;
    std::vector<std::string> getQueuedTaskNames() const;
    
    void dumpState() const;

private:
    // Internal task representation
    struct TaskBase {
        virtual ~TaskBase() = default;
        virtual void execute() = 0;
        TaskPriority priority = TaskPriority::Normal;
        std::chrono::steady_clock::time_point submitTime;
        std::chrono::milliseconds timeout{0};
        std::string name;
        int retryCount = 0;
        int maxRetries = 0;
    };
    
    template<typename T>
    struct TaskImpl : TaskBase {
        std::packaged_task<T()> task;
        
        TaskImpl(std::packaged_task<T()> t) : task(std::move(t)) {}
        
        void execute() override {
            task();
        }
    };
    
    // Task comparator for priority queue
    struct TaskComparator {
        bool operator()(const std::unique_ptr<TaskBase>& a, 
                       const std::unique_ptr<TaskBase>& b) const {
            if (a->priority != b->priority) {
                return static_cast<int>(a->priority) < static_cast<int>(b->priority);
            }
            return a->submitTime > b->submitTime; // Earlier tasks have higher priority
        }
    };
    
    // Thread worker function
    void worker(size_t threadIndex);
    void dynamicWorker(size_t threadIndex);
    
    // Task queue management
    std::unique_ptr<TaskBase> getNextTask(size_t threadIndex);
    void addTask(std::unique_ptr<TaskBase> task);
    
    // Dynamic scaling
    void checkScaling();
    void scaleUp();
    void scaleDown();
    
    // Statistics update
    void updateStats();
    
    // Member variables
    std::vector<std::thread> m_workers;
    std::priority_queue<std::unique_ptr<TaskBase>, 
                       std::vector<std::unique_ptr<TaskBase>>, 
                       TaskComparator> m_tasks;
    
    // Synchronization
    std::mutex m_queueMutex;
    std::condition_variable m_condition;
    std::condition_variable m_idleCondition;
    
    // State
    std::atomic<bool> m_stop{false};
    std::atomic<bool> m_paused{false};
    std::atomic<size_t> m_activeTasks{0};
    
    // Configuration
    PoolMode m_mode;
    ExecutionPolicy m_policy;
    size_t m_minThreads;
    size_t m_maxThreads;
    size_t m_maxQueueSize = 1000;
    std::chrono::milliseconds m_idleTimeout{30000}; // 30 seconds
    
    // Dynamic scaling
    bool m_dynamicScalingEnabled = false;
    double m_scaleUpThreshold = 0.8;
    double m_scaleDownThreshold = 0.2;
    std::chrono::steady_clock::time_point m_lastScaleCheck;
    
    // Work stealing
    bool m_workStealingEnabled = false;
    std::vector<std::queue<std::unique_ptr<TaskBase>>> m_localQueues;
    std::vector<std::mutex> m_localMutexes;
    
    // Thread management
    int m_threadPriority = 0;
    std::vector<std::vector<int>> m_threadAffinities;
    
    // Statistics
    mutable std::mutex m_statsMutex;
    TaskStats m_taskStats;
    ThreadStats m_threadStats;
    std::chrono::steady_clock::time_point m_startTime;
    
    // Callbacks
    TaskCallback m_taskCallback;
    ThreadCallback m_threadCallback;
    std::function<void(const std::exception&)> m_exceptionHandler;
    
    // Profiling
    bool m_profilingEnabled = false;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> m_taskStartTimes;
    mutable std::mutex m_profilingMutex;
};

// Template implementations
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    return enqueue(TaskPriority::Normal, std::forward<F>(f), std::forward<Args>(args)...);
}

template<class F, class... Args>
auto ThreadPool::enqueue(TaskPriority priority, F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    return enqueue(priority, "", std::forward<F>(f), std::forward<Args>(args)...);
}

template<class F, class... Args>
auto ThreadPool::enqueue(TaskPriority priority, const std::string& name, F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    return enqueue(priority, name, std::chrono::milliseconds{0}, 
                  std::forward<F>(f), std::forward<Args>(args)...);
}

template<class F, class... Args>
auto ThreadPool::enqueue(TaskPriority priority, const std::string& name, 
                        std::chrono::milliseconds timeout, F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    auto taskImpl = std::make_unique<TaskImpl<return_type>>(std::move(*task));
    taskImpl->priority = priority;
    taskImpl->name = name;
    taskImpl->timeout = timeout;
    taskImpl->submitTime = std::chrono::steady_clock::now();
    
    {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        
        if (m_stop) {
            throw std::runtime_error("ThreadPool is stopped");
        }
        
        if (m_tasks.size() >= m_maxQueueSize) {
            throw std::runtime_error("ThreadPool queue is full");
        }
        
        m_tasks.emplace(std::move(taskImpl));
        m_taskStats.totalTasks++;
        m_taskStats.queuedTasks++;
    }
    
    m_condition.notify_one();
    return result;
}

} // namespace Core
} // namespace Ultimate