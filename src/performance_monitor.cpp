#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <atomic>
#include <sstream>

#ifdef _WIN32
    #include <windows.h>
    #include <pdh.h>
    #include <psapi.h>
    #pragma comment(lib, "pdh.lib")
    #pragma comment(lib, "psapi.lib")
#else
    #include <unistd.h>
    #include <sys/times.h>
    #include <sys/sysinfo.h>
    #include <fstream>
#endif

class HardcorePerformanceMonitor {
private:
    std::atomic<bool> monitoring{true};
    std::chrono::milliseconds update_interval{1000};
    
    struct SystemMetrics {
        double cpu_usage_percent = 0.0;
        double memory_usage_percent = 0.0;
        uint64_t memory_used_mb = 0;
        uint64_t memory_total_mb = 0;
        std::vector<double> per_core_usage;
        double cpu_temperature = 0.0;
        uint64_t context_switches = 0;
        uint64_t interrupts = 0;
    };

public:
    HardcorePerformanceMonitor(int interval_ms = 1000) 
        : update_interval(interval_ms) {}

    #ifdef _WIN32
    SystemMetrics get_system_metrics() {
        SystemMetrics metrics;
        
        // Get CPU usage
        static PDH_HQUERY cpu_query = nullptr;
        static PDH_HCOUNTER cpu_counter = nullptr;
        static bool initialized = false;
        
        if (!initialized) {
            PdhOpenQuery(nullptr, 0, &cpu_query);
            PdhAddCounter(cpu_query, L"\\Processor(_Total)\\% Processor Time", 0, &cpu_counter);
            PdhCollectQueryData(cpu_query);
            initialized = true;
        }
        
        PdhCollectQueryData(cpu_query);
        PDH_FMT_COUNTERVALUE counter_val;
        PdhGetFormattedCounterValue(cpu_counter, PDH_FMT_DOUBLE, nullptr, &counter_val);
        metrics.cpu_usage_percent = counter_val.dblValue;
        
        // Get memory info
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        GlobalMemoryStatusEx(&mem_status);
        
        metrics.memory_total_mb = mem_status.ullTotalPhys / (1024 * 1024);
        metrics.memory_used_mb = (mem_status.ullTotalPhys - mem_status.ullAvailPhys) / (1024 * 1024);
        metrics.memory_usage_percent = ((double)metrics.memory_used_mb / metrics.memory_total_mb) * 100.0;
        
        return metrics;
    }
    #else
    SystemMetrics get_system_metrics() {
        SystemMetrics metrics;
        
        // Get CPU usage from /proc/stat
        static uint64_t prev_idle = 0, prev_total = 0;
        std::ifstream stat_file("/proc/stat");
        std::string line;
        
        if (std::getline(stat_file, line)) {
            std::istringstream iss(line);
            std::string cpu_label;
            uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
            
            iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
            
            uint64_t total = user + nice + system + idle + iowait + irq + softirq + steal;
            uint64_t total_diff = total - prev_total;
            uint64_t idle_diff = idle - prev_idle;
            
            if (total_diff > 0) {
                metrics.cpu_usage_percent = ((double)(total_diff - idle_diff) / total_diff) * 100.0;
            }
            
            prev_total = total;
            prev_idle = idle;
        }
        
        // Get memory info from /proc/meminfo
        std::ifstream meminfo("/proc/meminfo");
        std::string mem_line;
        uint64_t mem_total = 0, mem_available = 0;
        
        while (std::getline(meminfo, mem_line)) {
            if (mem_line.find("MemTotal:") == 0) {
                std::istringstream iss(mem_line);
                std::string label, kb_label;
                iss >> label >> mem_total >> kb_label;
            } else if (mem_line.find("MemAvailable:") == 0) {
                std::istringstream iss(mem_line);
                std::string label, kb_label;
                iss >> label >> mem_available >> kb_label;
            }
        }
        
        metrics.memory_total_mb = mem_total / 1024;
        metrics.memory_used_mb = (mem_total - mem_available) / 1024;
        metrics.memory_usage_percent = ((double)metrics.memory_used_mb / metrics.memory_total_mb) * 100.0;
        
        // Get per-core usage
        stat_file.clear();
        stat_file.seekg(0);
        std::getline(stat_file, line); // Skip first line (total)
        
        while (std::getline(stat_file, line) && line.find("cpu") == 0) {
            std::istringstream iss(line);
            std::string cpu_label;
            uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
            
            iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
            
            uint64_t total = user + nice + system + idle + iowait + irq + softirq + steal;
            if (total > 0) {
                double core_usage = ((double)(total - idle) / total) * 100.0;
                metrics.per_core_usage.push_back(core_usage);
            }
        }
        
        // Try to get CPU temperature (if available)
        std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
        if (temp_file.is_open()) {
            int temp_millidegrees;
            temp_file >> temp_millidegrees;
            metrics.cpu_temperature = temp_millidegrees / 1000.0;
        }
        
        // Get context switches and interrupts from /proc/stat
        stat_file.clear();
        stat_file.seekg(0);
        while (std::getline(stat_file, line)) {
            if (line.find("ctxt") == 0) {
                std::istringstream iss(line);
                std::string label;
                iss >> label >> metrics.context_switches;
            } else if (line.find("intr") == 0) {
                std::istringstream iss(line);
                std::string label;
                iss >> label >> metrics.interrupts;
            }
        }
        
        return metrics;
    }
    #endif

    void print_system_info() {
        std::cout << "\n=== SYSTEM INFORMATION ===\n";
        
        #ifdef _WIN32
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        std::cout << "CPU Cores: " << sys_info.dwNumberOfProcessors << "\n";
        std::cout << "Page Size: " << sys_info.dwPageSize << " bytes\n";
        std::cout << "Processor Architecture: ";
        switch (sys_info.wProcessorArchitecture) {
            case PROCESSOR_ARCHITECTURE_AMD64: std::cout << "x64\n"; break;
            case PROCESSOR_ARCHITECTURE_INTEL: std::cout << "x86\n"; break;
            default: std::cout << "Unknown\n"; break;
        }
        #else
        long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        long page_size = sysconf(_SC_PAGESIZE);
        std::cout << "CPU Cores: " << cpu_count << "\n";
        std::cout << "Page Size: " << page_size << " bytes\n";
        
        // Try to get CPU info
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                std::cout << "CPU: " << line.substr(line.find(":") + 2) << "\n";
                break;
            }
        }
        #endif
        
        std::cout << "========================\n\n";
    }

    void start_monitoring() {
        print_system_info();
        
        std::cout << "Starting hardcore performance monitoring...\n";
        std::cout << "Press Ctrl+C to stop\n\n";
        
        // Print header
        std::cout << std::left << std::setw(10) << "Time"
                  << std::setw(12) << "CPU %"
                  << std::setw(12) << "Memory %"
                  << std::setw(12) << "Mem (MB)"
                  << std::setw(12) << "Temp (°C)"
                  << std::setw(15) << "Ctx Switches"
                  << std::setw(12) << "Interrupts"
                  << "Per-Core Usage\n";
        
        std::cout << std::string(100, '-') << "\n";
        
        auto start_time = std::chrono::steady_clock::now();
        
        while (monitoring.load()) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            
            SystemMetrics metrics = get_system_metrics();
            
            // Print metrics
            std::cout << std::left << std::setw(10) << elapsed.count() << "s"
                      << std::setw(12) << std::fixed << std::setprecision(1) << metrics.cpu_usage_percent
                      << std::setw(12) << std::fixed << std::setprecision(1) << metrics.memory_usage_percent
                      << std::setw(12) << metrics.memory_used_mb
                      << std::setw(12) << std::fixed << std::setprecision(1) << metrics.cpu_temperature
                      << std::setw(15) << metrics.context_switches
                      << std::setw(12) << metrics.interrupts;
            
            // Print per-core usage
            for (size_t i = 0; i < metrics.per_core_usage.size() && i < 8; ++i) {
                std::cout << "C" << i << ":" << std::fixed << std::setprecision(0) 
                          << metrics.per_core_usage[i] << "% ";
            }
            
            std::cout << "\n";
            
            // Check for thermal throttling warning
            if (metrics.cpu_temperature > 85.0) {
                std::cout << "⚠️  WARNING: High CPU temperature detected! (" 
                          << metrics.cpu_temperature << "°C)\n";
            }
            
            // Check for high memory usage
            if (metrics.memory_usage_percent > 90.0) {
                std::cout << "⚠️  WARNING: High memory usage! (" 
                          << metrics.memory_usage_percent << "%)\n";
            }
            
            std::this_thread::sleep_for(update_interval);
        }
    }
    
    void stop_monitoring() {
        monitoring.store(false);
    }
    
    void generate_performance_report() {
        std::cout << "\n=== PERFORMANCE OPTIMIZATION CHECKLIST ===\n";
        
        SystemMetrics metrics = get_system_metrics();
        
        std::cout << "Current System Status:\n";
        std::cout << "- CPU Usage: " << std::fixed << std::setprecision(1) 
                  << metrics.cpu_usage_percent << "%\n";
        std::cout << "- Memory Usage: " << std::fixed << std::setprecision(1) 
                  << metrics.memory_usage_percent << "%\n";
        std::cout << "- Temperature: " << std::fixed << std::setprecision(1) 
                  << metrics.cpu_temperature << "°C\n";
        
        std::cout << "\nOptimization Recommendations:\n";
        
        if (metrics.cpu_usage_percent < 80.0) {
            std::cout << "✓ CPU has headroom for more intensive workloads\n";
        } else {
            std::cout << "⚠️ CPU is under heavy load - monitor for throttling\n";
        }
        
        if (metrics.memory_usage_percent < 80.0) {
            std::cout << "✓ Memory usage is acceptable\n";
        } else {
            std::cout << "⚠️ High memory usage - consider increasing RAM or optimizing memory allocation\n";
        }
        
        if (metrics.cpu_temperature < 70.0) {
            std::cout << "✓ CPU temperature is optimal\n";
        } else if (metrics.cpu_temperature < 85.0) {
            std::cout << "⚠️ CPU temperature is elevated - improve cooling\n";
        } else {
            std::cout << "🔥 CPU temperature is critical - immediate cooling improvement needed!\n";
        }
        
        std::cout << "\nHardware Monitoring Tools:\n";
        #ifdef _WIN32
        std::cout << "- HWiNFO64: Real-time hardware monitoring\n";
        std::cout << "- MSI Afterburner: GPU monitoring and overclocking\n";
        std::cout << "- Intel XTU: CPU overclocking and monitoring\n";
        std::cout << "- Process Explorer: Advanced process monitoring\n";
        #else
        std::cout << "- htop: Advanced process monitoring\n";
        std::cout << "- sensors: Hardware temperature monitoring\n";
        std::cout << "- iotop: I/O monitoring\n";
        std::cout << "- perf: CPU profiling and analysis\n";
        #endif
        
        std::cout << "\n==========================================\n";
    }
};

int main(int argc, char* argv[]) {
    int interval = 1000; // Default 1 second
    
    if (argc > 1) {
        interval = std::atoi(argv[1]);
        if (interval < 100) interval = 100; // Minimum 100ms
    }
    
    HardcorePerformanceMonitor monitor(interval);
    
    // Set up signal handler for graceful shutdown
    std::thread monitor_thread([&monitor]() {
        monitor.start_monitoring();
    });
    
    std::cout << "Press Enter to stop monitoring and generate report...\n";
    std::cin.get();
    
    monitor.stop_monitoring();
    monitor_thread.join();
    
    monitor.generate_performance_report();
    
    return 0;
}