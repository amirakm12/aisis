#include <windows.h>
#include <iostream>
#include <immintrin.h>  // For AVX intrinsics
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <memory>
#include <powrprof.h>
#include <winnt.h>

#pragma comment(lib, "powrprof.lib")

class WarlordPerformanceManager {
private:
    std::vector<std::thread> workers;
    std::atomic<bool> shutdown_flag{false};
    size_t core_count;
    
public:
    WarlordPerformanceManager() {
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        core_count = sysinfo.dwNumberOfProcessors;
        
        std::cout << "🔥 WARLORD MODE ACTIVATED - " << core_count << " cores detected\n";
        initialize_system_dominance();
    }
    
    ~WarlordPerformanceManager() {
        shutdown();
    }
    
    // BRUTAL SYSTEM INITIALIZATION - NO MERCY
    void initialize_system_dominance() {
        std::cout << "⚡ Setting ULTIMATE PERFORMANCE power scheme...\n";
        set_ultimate_performance_mode();
        
        std::cout << "🎯 Disabling processor parking...\n";
        disable_processor_parking();
        
        std::cout << "🚀 Setting process to REALTIME priority...\n";
        set_process_realtime_priority();
        
        std::cout << "💀 Disabling power throttling...\n";
        disable_power_throttling();
        
        std::cout << "🔥 System dominated successfully!\n";
    }
    
private:
    // Set thread affinity to specific cores aggressively
    void pin_thread_to_core(HANDLE thread, DWORD core_id) {
        DWORD_PTR mask = 1ULL << core_id;
        if (!SetThreadAffinityMask(thread, mask)) {
            std::cerr << "💥 CRITICAL: Failed to dominate core " << core_id << "\n";
            ExitProcess(1);
        }
        std::cout << "✅ Core " << core_id << " ENSLAVED\n";
    }
    
    // Set real-time priority for zero scheduling latency
    void set_thread_realtime_priority(HANDLE thread) {
        if (!SetThreadPriority(thread, THREAD_PRIORITY_TIME_CRITICAL)) {
            std::cerr << "💥 CRITICAL: Failed to achieve thread supremacy\n";
            ExitProcess(1);
        }
    }
    
    // Set process to realtime priority class
    void set_process_realtime_priority() {
        HANDLE process = GetCurrentProcess();
        if (!SetPriorityClass(process, REALTIME_PRIORITY_CLASS)) {
            std::cerr << "⚠️ WARNING: Could not set REALTIME priority, trying HIGH\n";
            if (!SetPriorityClass(process, HIGH_PRIORITY_CLASS)) {
                std::cerr << "💥 CRITICAL: Priority escalation failed\n";
            }
        }
    }
    
    // Ultimate Performance power scheme activation
    void set_ultimate_performance_mode() {
        // Ultimate Performance GUID: e9a42b02-d5df-448d-aa00-03f14749eb61
        GUID ultimate_guid = {0xe9a42b02, 0xd5df, 0x448d, {0xaa, 0x00, 0x03, 0xf1, 0x47, 0x49, 0xeb, 0x61}};
        
        if (PowerSetActiveScheme(NULL, &ultimate_guid) != ERROR_SUCCESS) {
            std::cerr << "⚠️ WARNING: Could not activate Ultimate Performance mode\n";
            std::cerr << "💡 Run: powercfg /duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61\n";
        }
    }
    
    // Disable processor parking via registry
    void disable_processor_parking() {
        HKEY hKey;
        DWORD value = 0;
        
        const char* parking_key = "SYSTEM\\CurrentControlSet\\Control\\Power\\PowerSettings\\54533251-82be-4824-96c1-47b60b740d00\\0cc5b647-c1df-4637-891a-dec35c318583";
        
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, parking_key, 0, KEY_SET_VALUE, &hKey) == ERROR_SUCCESS) {
            RegSetValueExA(hKey, "ValueMax", 0, REG_DWORD, (BYTE*)&value, sizeof(value));
            RegSetValueExA(hKey, "ValueMin", 0, REG_DWORD, (BYTE*)&value, sizeof(value));
            RegCloseKey(hKey);
        }
    }
    
    // Disable Windows power throttling
    void disable_power_throttling() {
        HANDLE process = GetCurrentProcess();
        PROCESS_POWER_THROTTLING_STATE PowerThrottling;
        RtlZeroMemory(&PowerThrottling, sizeof(PowerThrottling));
        PowerThrottling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        PowerThrottling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        PowerThrottling.StateMask = 0;
        
        SetProcessInformation(process, ProcessPowerThrottling, &PowerThrottling, sizeof(PowerThrottling));
    }
    
public:
    // SIMD-accelerated compute kernel using AVX2 - MAXIMUM BRUTALITY
    void avx2_vector_add(float* __restrict a, float* __restrict b, float* __restrict result, size_t n) {
        size_t i = 0;
        const size_t simd_end = n - (n % 8);
        
        // Main SIMD loop - 8 floats per iteration
        for (; i < simd_end; i += 8) {
            __m256 va = _mm256_load_ps(a + i);
            __m256 vb = _mm256_load_ps(b + i);
            __m256 vr = _mm256_add_ps(va, vb);
            _mm256_store_ps(result + i, vr);
        }
        
        // Tail loop for remainder
        for (; i < n; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    
    // Even more brutal: AVX2 FMA (Fused Multiply-Add)
    void avx2_vector_fma(float* __restrict a, float* __restrict b, float* __restrict c, float* __restrict result, size_t n) {
        size_t i = 0;
        const size_t simd_end = n - (n % 8);
        
        for (; i < simd_end; i += 8) {
            __m256 va = _mm256_load_ps(a + i);
            __m256 vb = _mm256_load_ps(b + i);
            __m256 vc = _mm256_load_ps(c + i);
            __m256 vr = _mm256_fmadd_ps(va, vb, vc);  // a * b + c
            _mm256_store_ps(result + i, vr);
        }
        
        for (; i < n; ++i) {
            result[i] = a[i] * b[i] + c[i];
        }
    }
    
    // Matrix multiplication with cache-friendly blocking
    void brutal_matrix_multiply(float* __restrict A, float* __restrict B, float* __restrict C, size_t N) {
        const size_t BLOCK_SIZE = 64;  // Cache-friendly block size
        
        for (size_t i = 0; i < N; i += BLOCK_SIZE) {
            for (size_t j = 0; j < N; j += BLOCK_SIZE) {
                for (size_t k = 0; k < N; k += BLOCK_SIZE) {
                    // Process block
                    size_t max_i = std::min(i + BLOCK_SIZE, N);
                    size_t max_j = std::min(j + BLOCK_SIZE, N);
                    size_t max_k = std::min(k + BLOCK_SIZE, N);
                    
                    for (size_t ii = i; ii < max_i; ++ii) {
                        for (size_t jj = j; jj < max_j; jj += 8) {
                            __m256 sum = _mm256_load_ps(&C[ii * N + jj]);
                            
                            for (size_t kk = k; kk < max_k; ++kk) {
                                __m256 a_vec = _mm256_broadcast_ss(&A[ii * N + kk]);
                                __m256 b_vec = _mm256_load_ps(&B[kk * N + jj]);
                                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                            }
                            
                            _mm256_store_ps(&C[ii * N + jj], sum);
                        }
                    }
                }
            }
        }
    }
    
    // Hardcore worker thread that dominates a core completely
    void hardcore_worker(int core_id, float* a, float* b, float* c, float* res, size_t n) {
        HANDLE thread = GetCurrentThread();
        pin_thread_to_core(thread, core_id);
        set_thread_realtime_priority(thread);
        
        std::cout << "💀 Core " << core_id << " worker UNLEASHED\n";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t iterations = 0;
        
        while (!shutdown_flag.load(std::memory_order_relaxed)) {
            // Rotate between different brutal operations
            switch (iterations % 3) {
                case 0:
                    avx2_vector_add(a, b, res, n);
                    break;
                case 1:
                    avx2_vector_fma(a, b, c, res, n);
                    break;
                case 2:
                    // Simulate some cache-intensive work
                    for (size_t i = 0; i < n; i += 64) {
                        _mm_prefetch((char*)(a + i + 64), _MM_HINT_T0);
                        avx2_vector_add(a + i, b + i, res + i, std::min(64ULL, n - i));
                    }
                    break;
            }
            
            iterations++;
            
            // Performance reporting every 10 million iterations
            if (iterations % 10000000 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                double ops_per_sec = (double)iterations * n / (duration.count() / 1000.0);
                
                std::cout << "🔥 Core " << core_id << ": " << iterations/1000000 << "M iterations, " 
                         << ops_per_sec/1e9 << " GFLOPS\n";
            }
        }
        
        std::cout << "💀 Core " << core_id << " worker TERMINATED after " << iterations << " iterations\n";
    }
    
    // Launch the army of worker threads
    void launch_warlord_army(size_t vector_size = 1 << 20) {
        std::cout << "🚀 Deploying " << core_count << " WARLORD threads...\n";
        
        // Allocate aligned memory for maximum performance
        float* a = (float*)_aligned_malloc(vector_size * sizeof(float), 32);
        float* b = (float*)_aligned_malloc(vector_size * sizeof(float), 32);
        float* c = (float*)_aligned_malloc(vector_size * sizeof(float), 32);
        float* res = (float*)_aligned_malloc(vector_size * sizeof(float), 32);
        
        if (!a || !b || !c || !res) {
            std::cerr << "💥 CRITICAL: Memory allocation failed\n";
            return;
        }
        
        // Initialize with performance-friendly patterns
        for (size_t i = 0; i < vector_size; ++i) {
            a[i] = static_cast<float>(i) * 0.001f;
            b[i] = static_cast<float>(vector_size - i) * 0.001f;
            c[i] = static_cast<float>(i * 2) * 0.001f;
        }
        
        // Deploy worker threads to each core
        for (size_t i = 0; i < core_count; ++i) {
            workers.emplace_back(&WarlordPerformanceManager::hardcore_worker, this, 
                               static_cast<int>(i), a, b, c, res, vector_size);
        }
        
        std::cout << "⚡ ALL CORES DOMINATED - MAXIMUM PERFORMANCE ACHIEVED\n";
        std::cout << "💡 Press Ctrl+C or call shutdown() to stop the mayhem\n";
        
        // Keep memory alive until shutdown
        // Note: In a real application, you'd want better memory management
        std::atexit([a, b, c, res]() {
            _aligned_free(a);
            _aligned_free(b);
            _aligned_free(c);
            _aligned_free(res);
        });
    }
    
    void shutdown() {
        if (!shutdown_flag.exchange(true)) {
            std::cout << "🛑 Initiating controlled shutdown...\n";
            
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            
            workers.clear();
            std::cout << "✅ All workers terminated cleanly\n";
        }
    }
    
    // Performance monitoring and system stats
    void print_system_status() {
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        
        std::cout << "\n📊 SYSTEM STATUS:\n";
        std::cout << "💾 Total RAM: " << memInfo.ullTotalPhys / (1024*1024*1024) << " GB\n";
        std::cout << "💾 Available RAM: " << memInfo.ullAvailPhys / (1024*1024*1024) << " GB\n";
        std::cout << "🔥 CPU Cores: " << core_count << "\n";
        
        // Get CPU frequency info if available
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        std::cout << "⚡ Processor Architecture: " << sysInfo.wProcessorArchitecture << "\n";
        std::cout << "🎯 Page Size: " << sysInfo.dwPageSize << " bytes\n";
    }
};

// Signal handler for graceful shutdown
WarlordPerformanceManager* g_warlord = nullptr;

BOOL WINAPI ConsoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT && g_warlord) {
        std::cout << "\n🛑 Shutdown signal received...\n";
        g_warlord->shutdown();
        return TRUE;
    }
    return FALSE;
}

int main() {
    std::cout << R"(
🔥🔥🔥 WARLORD PERFORMANCE MODE 🔥🔥🔥
💀 WARNING: MAXIMUM SYSTEM DOMINATION MODE 💀
⚡ This will push your system to absolute limits ⚡
🌡️ Ensure adequate cooling before proceeding 🌡️
)" << std::endl;
    
    try {
        WarlordPerformanceManager warlord;
        g_warlord = &warlord;
        
        // Set console handler for graceful shutdown
        SetConsoleCtrlHandler(ConsoleHandler, TRUE);
        
        warlord.print_system_status();
        warlord.launch_warlord_army();
        
        // Keep main thread alive
        std::cout << "🎮 Main thread entering monitoring mode...\n";
        std::cout << "💡 System will run until interrupted (Ctrl+C)\n";
        
        // Monitoring loop
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            warlord.print_system_status();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "💥 CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}