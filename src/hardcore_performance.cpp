#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <memory>
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
    #include <immintrin.h>  // For AVX intrinsics
    #include <processthreadsapi.h>
    #include <winbase.h>
#else
    #include <pthread.h>
    #include <sched.h>
    #include <unistd.h>
    #include <sys/mman.h>
    #include <immintrin.h>  // AVX support on Linux
#endif

class HardcorePerformanceEngine {
private:
    std::atomic<bool> running{true};
    size_t vector_size;
    float* aligned_a;
    float* aligned_b;
    float* aligned_result;
    int core_count;

public:
    HardcorePerformanceEngine(size_t vec_size = 1 << 20) : vector_size(vec_size) {
        // Allocate aligned memory for SIMD operations
        #ifdef _WIN32
            aligned_a = (float*)_aligned_malloc(vector_size * sizeof(float), 32);
            aligned_b = (float*)_aligned_malloc(vector_size * sizeof(float), 32);
            aligned_result = (float*)_aligned_malloc(vector_size * sizeof(float), 32);
            
            SYSTEM_INFO sysinfo;
            GetSystemInfo(&sysinfo);
            core_count = sysinfo.dwNumberOfProcessors;
        #else
            aligned_a = (float*)aligned_alloc(32, vector_size * sizeof(float));
            aligned_b = (float*)aligned_alloc(32, vector_size * sizeof(float));
            aligned_result = (float*)aligned_alloc(32, vector_size * sizeof(float));
            
            core_count = std::thread::hardware_concurrency();
        #endif

        if (!aligned_a || !aligned_b || !aligned_result) {
            std::cerr << "Failed to allocate aligned memory!\n";
            exit(1);
        }

        // Initialize vectors with test data
        initialize_vectors();
    }

    ~HardcorePerformanceEngine() {
        #ifdef _WIN32
            _aligned_free(aligned_a);
            _aligned_free(aligned_b);
            _aligned_free(aligned_result);
        #else
            free(aligned_a);
            free(aligned_b);
            free(aligned_result);
        #endif
    }

    void initialize_vectors() {
        for (size_t i = 0; i < vector_size; ++i) {
            aligned_a[i] = static_cast<float>(i * 0.5f);
            aligned_b[i] = static_cast<float>((vector_size - i) * 0.3f);
        }
    }

    // Windows-specific thread pinning and priority setting
    #ifdef _WIN32
    void pin_thread_to_core(HANDLE thread, DWORD core_id) {
        DWORD_PTR mask = 1ULL << core_id;
        if (!SetThreadAffinityMask(thread, mask)) {
            std::cerr << "Failed to set affinity for core " << core_id << "\n";
            // Don't exit - continue with degraded performance
        } else {
            std::cout << "Thread pinned to core " << core_id << "\n";
        }
    }

    void set_thread_realtime_priority(HANDLE thread) {
        if (!SetThreadPriority(thread, THREAD_PRIORITY_TIME_CRITICAL)) {
            std::cerr << "Failed to set thread priority (may need admin rights)\n";
            // Try highest available priority
            SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
        } else {
            std::cout << "Thread set to real-time priority\n";
        }
    }
    #else
    // Linux equivalent functions
    void pin_thread_to_core(pthread_t thread, int core_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        
        int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        if (result != 0) {
            std::cerr << "Failed to set affinity for core " << core_id << "\n";
        } else {
            std::cout << "Thread pinned to core " << core_id << "\n";
        }
    }

    void set_thread_realtime_priority(pthread_t thread) {
        struct sched_param param;
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
        
        int result = pthread_setschedparam(thread, SCHED_FIFO, &param);
        if (result != 0) {
            std::cerr << "Failed to set real-time priority (may need root)\n";
            // Try nice priority as fallback
            int nice_result = nice(-20);
            (void)nice_result; // Suppress unused variable warning
        } else {
            std::cout << "Thread set to real-time priority\n";
        }
    }
    #endif

    // SIMD-accelerated compute kernels
    void avx2_vector_add(float* a, float* b, float* result, size_t n) {
        size_t i = 0;
        // Process 8 floats at a time with AVX2
        for (; i + 7 < n; i += 8) {
            __m256 va = _mm256_load_ps(a + i);
            __m256 vb = _mm256_load_ps(b + i);
            __m256 vr = _mm256_add_ps(va, vb);
            _mm256_store_ps(result + i, vr);
        }
        // Handle remaining elements
        for (; i < n; ++i) {
            result[i] = a[i] + b[i];
        }
    }

    void avx2_vector_multiply_add(float* a, float* b, float* c, float* result, size_t n) {
        size_t i = 0;
        // Fused multiply-add operation: result = a * b + c
        for (; i + 7 < n; i += 8) {
            __m256 va = _mm256_load_ps(a + i);
            __m256 vb = _mm256_load_ps(b + i);
            __m256 vc = _mm256_load_ps(c + i);
            __m256 vr = _mm256_fmadd_ps(va, vb, vc);
            _mm256_store_ps(result + i, vr);
        }
        for (; i < n; ++i) {
            result[i] = a[i] * b[i] + c[i];
        }
    }

    void avx2_dot_product(float* a, float* b, size_t n, float& result) {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        
        for (; i + 7 < n; i += 8) {
            __m256 va = _mm256_load_ps(a + i);
            __m256 vb = _mm256_load_ps(b + i);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        
        // Horizontal sum
        __m128 low = _mm256_castps256_ps128(sum);
        __m128 high = _mm256_extractf128_ps(sum, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        
        result = _mm_cvtss_f32(sum128);
        
        // Handle remaining elements
        for (; i < n; ++i) {
            result += a[i] * b[i];
        }
    }

    // Hardcore worker thread implementation
    void hardcore_worker(int core_id) {
        #ifdef _WIN32
            HANDLE thread = GetCurrentThread();
            pin_thread_to_core(thread, core_id);
            set_thread_realtime_priority(thread);
        #else
            pthread_t thread = pthread_self();
            pin_thread_to_core(thread, core_id);
            set_thread_realtime_priority(thread);
        #endif

        std::cout << "Worker " << core_id << " starting hardcore computation...\n";

        // Performance counters
        uint64_t iterations = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Allocate per-thread working memory
        float* temp_result = nullptr;
        #ifdef _WIN32
            temp_result = (float*)_aligned_malloc(vector_size * sizeof(float), 32);
        #else
            temp_result = (float*)aligned_alloc(32, vector_size * sizeof(float));
        #endif

        while (running.load(std::memory_order_relaxed)) {
            // Intensive SIMD computations
            avx2_vector_add(aligned_a, aligned_b, temp_result, vector_size);
            avx2_vector_multiply_add(aligned_a, aligned_b, temp_result, aligned_result, vector_size);
            
            float dot_result;
            avx2_dot_product(aligned_a, aligned_b, vector_size, dot_result);
            
            // Additional intensive operations
            for (int j = 0; j < 10; ++j) {
                avx2_vector_add(temp_result, aligned_result, temp_result, vector_size);
            }
            
            iterations++;
            
            // Performance reporting every 1000 iterations
            if (iterations % 1000 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time).count();
                
                if (duration > 0) {
                    double ops_per_sec = (iterations * 1000.0) / duration;
                    std::cout << "Core " << core_id << ": " << ops_per_sec 
                              << " ops/sec, dot_product: " << dot_result << "\n";
                }
            }
        }

        #ifdef _WIN32
            _aligned_free(temp_result);
        #else
            free(temp_result);
        #endif

        std::cout << "Worker " << core_id << " completed " << iterations << " iterations\n";
    }

    void run_benchmark(int duration_seconds = 10) {
        std::cout << "=== HARDCORE PERFORMANCE ENGINE ===\n";
        std::cout << "Vector size: " << vector_size << " floats\n";
        std::cout << "CPU cores: " << core_count << "\n";
        std::cout << "Running for " << duration_seconds << " seconds...\n\n";

        // Enable high performance mode
        #ifdef _WIN32
            SetProcessPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
            std::cout << "Process priority set to HIGH\n";
        #endif

        // Launch worker threads
        std::vector<std::thread> workers;
        for (int i = 0; i < core_count; ++i) {
            workers.emplace_back(&HardcorePerformanceEngine::hardcore_worker, this, i);
        }

        // Run for specified duration
        std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
        
        // Signal shutdown
        running.store(false, std::memory_order_relaxed);
        
        // Wait for all workers to complete
        for (auto& worker : workers) {
            worker.join();
        }

        std::cout << "\n=== BENCHMARK COMPLETE ===\n";
    }

    void system_optimization_guide() {
        std::cout << "\n=== SYSTEM OPTIMIZATION GUIDE ===\n";
        
        #ifdef _WIN32
        std::cout << "Windows Optimizations:\n";
        std::cout << "1. Power Plan: powercfg /duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61\n";
        std::cout << "2. Set CPU Min/Max to 100% in power options\n";
        std::cout << "3. Disable Windows Defender real-time protection\n";
        std::cout << "4. Set process priority to Realtime in Task Manager\n";
        std::cout << "5. Disable CPU parking: powercfg /setacvalueindex scheme_current sub_processor PROCTHROTTLEMIN 100\n";
        std::cout << "6. Disable HPET: bcdedit /set useplatformclock false\n";
        std::cout << "7. Disable interrupt moderation in network adapter settings\n";
        #else
        std::cout << "Linux Optimizations:\n";
        std::cout << "1. Set CPU governor to performance: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor\n";
        std::cout << "2. Disable CPU frequency scaling: echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo\n";
        std::cout << "3. Set process priority: sudo nice -n -20 ./hardcore_performance\n";
        std::cout << "4. Use real-time kernel: sudo apt install linux-lowlatency\n";
        std::cout << "5. Disable swap: sudo swapoff -a\n";
        std::cout << "6. Set CPU isolation: isolcpus=1-N in kernel parameters\n";
        #endif
        
        std::cout << "\nHardware Recommendations:\n";
        std::cout << "- High-end cooling (liquid cooling recommended)\n";
        std::cout << "- High-quality PSU with stable power delivery\n";
        std::cout << "- Fast RAM with low latency (DDR4-3200+ or DDR5)\n";
        std::cout << "- Monitor temperatures with HWiNFO64 or similar\n";
        std::cout << "- Ensure adequate case ventilation\n";
        std::cout << "\n";
    }
};

int main(int argc, char* argv[]) {
    int duration = 10;
    size_t vector_size = 1 << 20; // 1M floats default
    
    if (argc > 1) {
        duration = std::atoi(argv[1]);
    }
    if (argc > 2) {
        vector_size = std::atoi(argv[2]);
    }

    HardcorePerformanceEngine engine(vector_size);
    engine.system_optimization_guide();
    engine.run_benchmark(duration);

    return 0;
}