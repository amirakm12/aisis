#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <chrono>

namespace aisis {

template<typename T>
class LockFreeQueue {
public:
    LockFreeQueue() : head_(new Node), tail_(head_.load()) {}
    
    ~LockFreeQueue() {
        while (Node* const old_head = head_.load()) {
            head_.store(old_head->next);
            delete old_head;
        }
    }
    
    void enqueue(T item) {
        Node* const new_node = new Node;
        Node* const prev_tail = tail_.exchange(new_node);
        prev_tail->data = std::move(item);
        prev_tail->next.store(new_node);
    }
    
    bool dequeue(T& result) {
        Node* head = head_.load();
        Node* const next = head->next.load();
        if (next == nullptr) {
            return false;
        }
        result = std::move(next->data);
        head_.store(next);
        delete head;
        return true;
    }
    
    bool empty() const {
        Node* head = head_.load();
        return head->next.load() == nullptr;
    }
    
private:
    struct Node {
        std::atomic<Node*> next{nullptr};
        T data;
    };
    
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
};

class WorkStealingQueue {
public:
    WorkStealingQueue() = default;
    
    void push(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_front(std::move(task));
    }
    
    bool try_pop(std::function<void()>& task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        task = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }
    
    bool try_steal(std::function<void()>& task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        task = std::move(queue_.back());
        queue_.pop_back();
        return true;
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
private:
    mutable std::mutex mutex_;
    std::deque<std::function<void()>> queue_;
};

struct TaskStats {
    std::atomic<uint64_t> tasksCompleted{0};
    std::atomic<uint64_t> tasksQueued{0};
    std::atomic<uint64_t> tasksStolen{0};
    std::atomic<uint64_t> totalExecutionTime{0}; // microseconds
    std::atomic<uint64_t> totalWaitTime{0}; // microseconds
    std::chrono::high_resolution_clock::time_point startTime;
};

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    ~ThreadPool();
    
    // Task submission
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    template<typename F, typename... Args>
    auto enqueue_high_priority(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // Bulk operations
    template<typename Iterator, typename Function>
    void parallel_for(Iterator first, Iterator last, Function func);
    
    template<typename Function>
    void parallel_for(size_t start, size_t end, Function func);
    
    template<typename Iterator, typename Function, typename Reducer>
    auto parallel_reduce(Iterator first, Iterator last, Function func, Reducer reducer)
        -> typename std::result_of<Function(*first)>::type;
    
    // Control methods
    void pause();
    void resume();
    void wait_for_all();
    void clear_queue();
    
    // Configuration
    void resize(size_t numThreads);
    void set_thread_affinity(const std::vector<std::vector<int>>& affinities);
    void enable_work_stealing(bool enabled);
    void set_queue_size_limit(size_t limit);
    
    // Statistics and monitoring
    TaskStats getStats() const { return stats_; }
    size_t active_threads() const { return activeThreads_.load(); }
    size_t queued_tasks() const;
    size_t get_thread_count() const { return workers_.size(); }
    float get_cpu_utilization() const;
    std::vector<size_t> get_per_thread_queue_sizes() const;
    
    // Advanced features
    void enable_priority_scheduling(bool enabled);
    void set_load_balancing_strategy(const std::string& strategy); // "round_robin", "least_loaded", "work_stealing"
    void enable_adaptive_scaling(bool enabled);
    void set_thread_local_storage_size(size_t bytes);
    
    // Performance profiling
    void enable_profiling(bool enabled);
    std::vector<std::pair<std::string, std::chrono::microseconds>> get_profiling_data() const;
    void reset_profiling_data();
    
    // Shutdown control
    void shutdown(bool graceful = true);
    bool is_shutdown() const { return shutdown_.load(); }
    
private:
    // Worker thread data
    struct WorkerData {
        std::thread thread;
        std::unique_ptr<WorkStealingQueue> localQueue;
        std::atomic<bool> active{false};
        std::atomic<uint64_t> tasksProcessed{0};
        std::atomic<uint64_t> tasksStolen{0};
        std::vector<int> cpuAffinity;
        std::chrono::high_resolution_clock::time_point lastActiveTime;
    };
    
    // Core components
    std::vector<std::unique_ptr<WorkerData>> workers_;
    LockFreeQueue<std::function<void()>> globalQueue_;
    std::queue<std::function<void()>> highPriorityQueue_;
    
    // Synchronization
    std::mutex globalMutex_;
    std::mutex highPriorityMutex_;
    std::condition_variable condition_;
    std::condition_variable allTasksComplete_;
    
    // State management
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> paused_{false};
    std::atomic<size_t> activeThreads_{0};
    std::atomic<size_t> pendingTasks_{0};
    
    // Configuration
    bool workStealingEnabled_{true};
    size_t queueSizeLimit_{10000};
    bool prioritySchedulingEnabled_{false};
    std::string loadBalancingStrategy_{"work_stealing"};
    bool adaptiveScalingEnabled_{false};
    size_t threadLocalStorageSize_{1024};
    
    // Statistics
    mutable TaskStats stats_;
    bool profilingEnabled_{false};
    mutable std::mutex profilingMutex_;
    std::vector<std::pair<std::string, std::chrono::microseconds>> profilingData_;
    
    // Worker thread functions
    void worker_thread(size_t threadId);
    bool try_get_task(size_t threadId, std::function<void()>& task);
    bool try_steal_task(size_t threadId, std::function<void()>& task);
    
    // Load balancing
    size_t select_worker_round_robin();
    size_t select_worker_least_loaded();
    size_t select_worker_random();
    
    // Adaptive scaling
    void monitor_performance();
    void adjust_thread_count();
    bool should_scale_up() const;
    bool should_scale_down() const;
    
    // Utility methods
    void set_thread_affinity(std::thread& thread, const std::vector<int>& affinity);
    void update_statistics(const std::chrono::microseconds& executionTime);
    void cleanup_completed_threads();
    
    // Round-robin counter for load balancing
    std::atomic<size_t> roundRobinCounter_{0};
};

// Template implementations
template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    if (shutdown_.load()) {
        throw std::runtime_error("ThreadPool is shutdown");
    }
    
    auto taskWrapper = [task, this]() {
        auto start = std::chrono::high_resolution_clock::now();
        (*task)();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (profilingEnabled_) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            update_statistics(duration);
        }
        
        pendingTasks_.fetch_sub(1);
    };
    
    // Try to assign to least loaded worker's local queue
    if (workStealingEnabled_ && !workers_.empty()) {
        size_t workerId = select_worker_least_loaded();
        if (workers_[workerId]->localQueue->size() < queueSizeLimit_ / workers_.size()) {
            workers_[workerId]->localQueue->push(taskWrapper);
            pendingTasks_.fetch_add(1);
            stats_.tasksQueued.fetch_add(1);
            condition_.notify_one();
            return result;
        }
    }
    
    // Fall back to global queue
    globalQueue_.enqueue(taskWrapper);
    pendingTasks_.fetch_add(1);
    stats_.tasksQueued.fetch_add(1);
    condition_.notify_one();
    
    return result;
}

template<typename F, typename... Args>
auto ThreadPool::enqueue_high_priority(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    if (shutdown_.load()) {
        throw std::runtime_error("ThreadPool is shutdown");
    }
    
    auto taskWrapper = [task, this]() {
        auto start = std::chrono::high_resolution_clock::now();
        (*task)();
        auto end = std::chrono::high_resolution_clock::now();
        
        if (profilingEnabled_) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            update_statistics(duration);
        }
        
        pendingTasks_.fetch_sub(1);
    };
    
    {
        std::lock_guard<std::mutex> lock(highPriorityMutex_);
        highPriorityQueue_.push(taskWrapper);
    }
    
    pendingTasks_.fetch_add(1);
    stats_.tasksQueued.fetch_add(1);
    condition_.notify_one();
    
    return result;
}

template<typename Iterator, typename Function>
void ThreadPool::parallel_for(Iterator first, Iterator last, Function func) {
    size_t distance = std::distance(first, last);
    if (distance == 0) return;
    
    size_t numThreads = std::min(distance, workers_.size());
    size_t chunkSize = distance / numThreads;
    
    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);
    
    Iterator current = first;
    for (size_t i = 0; i < numThreads; ++i) {
        Iterator chunkEnd = (i == numThreads - 1) ? last : std::next(current, chunkSize);
        
        futures.emplace_back(enqueue([current, chunkEnd, func]() {
            for (auto it = current; it != chunkEnd; ++it) {
                func(*it);
            }
        }));
        
        current = chunkEnd;
    }
    
    for (auto& future : futures) {
        future.wait();
    }
}

template<typename Function>
void ThreadPool::parallel_for(size_t start, size_t end, Function func) {
    if (start >= end) return;
    
    size_t distance = end - start;
    size_t numThreads = std::min(distance, workers_.size());
    size_t chunkSize = distance / numThreads;
    
    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);
    
    for (size_t i = 0; i < numThreads; ++i) {
        size_t chunkStart = start + i * chunkSize;
        size_t chunkEnd = (i == numThreads - 1) ? end : chunkStart + chunkSize;
        
        futures.emplace_back(enqueue([chunkStart, chunkEnd, func]() {
            for (size_t j = chunkStart; j < chunkEnd; ++j) {
                func(j);
            }
        }));
    }
    
    for (auto& future : futures) {
        future.wait();
    }
}

template<typename Iterator, typename Function, typename Reducer>
auto ThreadPool::parallel_reduce(Iterator first, Iterator last, Function func, Reducer reducer)
    -> typename std::result_of<Function(*first)>::type {
    
    using ResultType = typename std::result_of<Function(*first)>::type;
    
    size_t distance = std::distance(first, last);
    if (distance == 0) {
        throw std::invalid_argument("Empty range for parallel_reduce");
    }
    if (distance == 1) {
        return func(*first);
    }
    
    size_t numThreads = std::min(distance, workers_.size());
    size_t chunkSize = distance / numThreads;
    
    std::vector<std::future<ResultType>> futures;
    futures.reserve(numThreads);
    
    Iterator current = first;
    for (size_t i = 0; i < numThreads; ++i) {
        Iterator chunkEnd = (i == numThreads - 1) ? last : std::next(current, chunkSize);
        
        futures.emplace_back(enqueue([current, chunkEnd, func, reducer]() -> ResultType {
            if (current == chunkEnd) {
                throw std::runtime_error("Empty chunk in parallel_reduce");
            }
            
            ResultType result = func(*current);
            ++current;
            
            for (auto it = current; it != chunkEnd; ++it) {
                result = reducer(result, func(*it));
            }
            
            return result;
        }));
        
        current = chunkEnd;
    }
    
    // Collect and reduce results
    std::vector<ResultType> results;
    results.reserve(futures.size());
    
    for (auto& future : futures) {
        results.emplace_back(future.get());
    }
    
    ResultType finalResult = results[0];
    for (size_t i = 1; i < results.size(); ++i) {
        finalResult = reducer(finalResult, results[i]);
    }
    
    return finalResult;
}

} // namespace aisis