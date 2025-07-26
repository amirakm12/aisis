#include "ai/AIAccelerator.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>

using namespace aisis;

/**
 * @brief AI Acceleration Demo - Comprehensive demonstration of AI acceleration capabilities
 * 
 * This demo showcases:
 * - Multiple acceleration types (GPU, NPU, Quantum, Optical, Neuromorphic)
 * - Performance benchmarking
 * - Real-time monitoring
 * - Power management
 * - Edge AI acceleration
 * - Distributed acceleration
 */
class AIAccelerationDemo {
public:
    AIAccelerationDemo() : m_accelerator(std::make_unique<AIAccelerator>()) {
        std::cout << "🚀 AI Acceleration Demo Initialized" << std::endl;
    }

    void run() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "🚀 ULTIMATE AI ACCELERATION DEMO" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        // Initialize the accelerator
        if (!m_accelerator->initialize()) {
            std::cerr << "❌ Failed to initialize AI accelerator" << std::endl;
            return;
        }

        // Run comprehensive demo
        demonstrateHardwareCapabilities();
        demonstrateAccelerationTypes();
        demonstrateWorkloadOptimization();
        demonstratePerformanceBenchmarking();
        demonstratePowerManagement();
        demonstrateEdgeAcceleration();
        demonstrateDistributedAcceleration();
        demonstrateRealTimeMonitoring();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "✅ AI ACCELERATION DEMO COMPLETED SUCCESSFULLY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

private:
    std::unique_ptr<AIAccelerator> m_accelerator;

    void demonstrateHardwareCapabilities() {
        std::cout << "\n🔧 HARDWARE CAPABILITIES DEMONSTRATION" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        auto capabilities = m_accelerator->getHardwareCapabilities();
        
        std::cout << "📊 Detected Hardware:" << std::endl;
        std::cout << "   GPU Available: " << (capabilities.gpu_available ? "✅ Yes" : "❌ No") << std::endl;
        std::cout << "   NPU Available: " << (capabilities.npu_available ? "✅ Yes" : "❌ No") << std::endl;
        std::cout << "   Quantum Available: " << (capabilities.quantum_available ? "✅ Yes" : "❌ No") << std::endl;
        std::cout << "   Optical Available: " << (capabilities.optical_available ? "✅ Yes" : "❌ No") << std::endl;
        std::cout << "   Neuromorphic Available: " << (capabilities.neuromorphic_available ? "✅ Yes" : "❌ No") << std::endl;
        std::cout << "   GPU Memory: " << capabilities.gpu_memory_gb << " GB" << std::endl;
        std::cout << "   System Memory: " << capabilities.system_memory_gb << " GB" << std::endl;
        std::cout << "   CPU Cores: " << capabilities.cpu_cores << std::endl;
        std::cout << "   GPU Compute Capability: " << capabilities.gpu_compute_capability << std::endl;
    }

    void demonstrateAccelerationTypes() {
        std::cout << "\n🚀 ACCELERATION TYPES DEMONSTRATION" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        // Test different acceleration types
        std::vector<AIAccelerator::AccelerationType> types = {
            AIAccelerator::AccelerationType::GPU_ACCELERATED,
            AIAccelerator::AccelerationType::NPU_ACCELERATED,
            AIAccelerator::AccelerationType::QUANTUM_ACCELERATED,
            AIAccelerator::AccelerationType::OPTICAL_ACCELERATED,
            AIAccelerator::AccelerationType::NEUROMORPHIC,
            AIAccelerator::AccelerationType::HYBRID_ACCELERATED
        };

        std::vector<std::string> typeNames = {
            "GPU Acceleration",
            "NPU Acceleration", 
            "Quantum Acceleration",
            "Optical Acceleration",
            "Neuromorphic Acceleration",
            "Hybrid Acceleration"
        };

        for (size_t i = 0; i < types.size(); ++i) {
            std::cout << "\n🔧 Testing " << typeNames[i] << "..." << std::endl;
            
            m_accelerator->setAccelerationType(types[i]);
            
            // Load a demo model
            m_accelerator->loadModel("demo_model.onnx", "ONNX");
            
            // Perform inference test
            std::vector<float> input = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
            std::vector<float> output;
            
            auto start = std::chrono::high_resolution_clock::now();
            bool success = m_accelerator->performInference(input, output);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            if (success) {
                std::cout << "   ✅ Inference successful in " << duration.count() << " μs" << std::endl;
                std::cout << "   📊 Output: [";
                for (size_t j = 0; j < std::min(output.size(), size_t(5)); ++j) {
                    std::cout << std::fixed << std::setprecision(3) << output[j];
                    if (j < std::min(output.size(), size_t(5)) - 1) std::cout << ", ";
                }
                if (output.size() > 5) std::cout << ", ...";
                std::cout << "]" << std::endl;
            } else {
                std::cout << "   ❌ Inference failed" << std::endl;
            }
            
            // Get metrics
            auto metrics = m_accelerator->getAccelerationMetrics();
            std::cout << "   📈 Throughput: " << metrics.throughput_tokens_per_second << " tokens/sec" << std::endl;
            std::cout << "   ⚡ Latency: " << metrics.latency_milliseconds << " ms" << std::endl;
            std::cout << "   🔋 Power: " << metrics.power_consumption_watts << "W" << std::endl;
            std::cout << "   🌡️ Temperature: " << metrics.thermal_temperature_celsius << "°C" << std::endl;
        }
    }

    void demonstrateWorkloadOptimization() {
        std::cout << "\n⚙️ WORKLOAD OPTIMIZATION DEMONSTRATION" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        std::vector<AIAccelerator::WorkloadType> workloads = {
            AIAccelerator::WorkloadType::INFERENCE,
            AIAccelerator::WorkloadType::TRAINING,
            AIAccelerator::WorkloadType::TRANSFORMER,
            AIAccelerator::WorkloadType::CONVOLUTIONAL,
            AIAccelerator::WorkloadType::RECURRENT,
            AIAccelerator::WorkloadType::GENERATIVE,
            AIAccelerator::WorkloadType::REINFORCEMENT,
            AIAccelerator::WorkloadType::QUANTUM_ML
        };

        std::vector<std::string> workloadNames = {
            "Inference",
            "Training",
            "Transformer Models",
            "Convolutional Networks",
            "Recurrent Networks", 
            "Generative AI",
            "Reinforcement Learning",
            "Quantum Machine Learning"
        };

        for (size_t i = 0; i < workloads.size(); ++i) {
            std::cout << "\n🔧 Optimizing for " << workloadNames[i] << "..." << std::endl;
            
            m_accelerator->setWorkloadType(workloads[i]);
            
            // Simulate workload-specific processing
            std::vector<std::vector<float>> trainingData(100, std::vector<float>(10));
            std::vector<std::vector<float>> labels(100, std::vector<float>(1));
            
            // Generate random training data
            for (auto& sample : trainingData) {
                for (auto& val : sample) {
                    val = static_cast<float>(rand()) / RAND_MAX;
                }
            }
            
            for (auto& label : labels) {
                label[0] = static_cast<float>(rand()) / RAND_MAX;
            }
            
            std::cout << "   🚀 Training with " << trainingData.size() << " samples..." << std::endl;
            m_accelerator->trainModel(trainingData, labels, 10);
            
            auto metrics = m_accelerator->getAccelerationMetrics();
            std::cout << "   📊 Acceleration Factor: " << m_accelerator->getAccelerationFactor() << "x" << std::endl;
            std::cout << "   ⚡ Throughput: " << metrics.throughput_tokens_per_second << " tokens/sec" << std::endl;
        }
    }

    void demonstratePerformanceBenchmarking() {
        std::cout << "\n📊 PERFORMANCE BENCHMARKING DEMONSTRATION" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        std::cout << "🚀 Running comprehensive benchmark suite..." << std::endl;
        
        auto benchmarkResults = m_accelerator->runAccelerationBenchmark();
        
        std::cout << "\n📈 BENCHMARK RESULTS:" << std::endl;
        std::cout << "   🚀 Inference Speed: " << benchmarkResults.inference_speed_tokens_per_second << " tokens/sec" << std::endl;
        std::cout << "   🎯 Training Speed: " << benchmarkResults.training_speed_samples_per_second << " samples/sec" << std::endl;
        std::cout << "   🔋 Energy Efficiency: " << benchmarkResults.energy_efficiency_tokens_per_watt << " tokens/watt" << std::endl;
        std::cout << "   💾 Memory Efficiency: " << benchmarkResults.memory_efficiency_mb_per_token << " MB/token" << std::endl;
        std::cout << "   🌡️ Thermal Performance: " << benchmarkResults.thermal_performance_celsius << "°C" << std::endl;
        std::cout << "   🎯 Accuracy: " << benchmarkResults.accuracy_percentage << "%" << std::endl;
        std::cout << "   ⚡ Latency: " << benchmarkResults.latency_milliseconds << " ms" << std::endl;
        std::cout << "   📈 Throughput Multiplier: " << benchmarkResults.throughput_multiplier << "x" << std::endl;
    }

    void demonstratePowerManagement() {
        std::cout << "\n🔋 POWER MANAGEMENT DEMONSTRATION" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        std::vector<std::string> powerModes = {"Performance", "Balanced", "Power Saving"};
        
        for (int mode = 0; mode < 3; ++mode) {
            std::cout << "\n🔧 Testing " << powerModes[mode] << " mode..." << std::endl;
            
            m_accelerator->setPowerManagementMode(mode);
            
            // Simulate workload
            std::vector<float> input(1000);
            for (auto& val : input) {
                val = static_cast<float>(rand()) / RAND_MAX;
            }
            
            std::vector<float> output;
            m_accelerator->performInference(input, output);
            
            std::cout << "   🔋 Power Consumption: " << m_accelerator->getPowerConsumption() << "W" << std::endl;
            std::cout << "   🌡️ Thermal Status: " << m_accelerator->getThermalStatus() << "°C" << std::endl;
            
            // Wait a bit to see power changes
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void demonstrateEdgeAcceleration() {
        std::cout << "\n🌐 EDGE AI ACCELERATION DEMONSTRATION" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        std::cout << "🔧 Enabling Edge AI acceleration..." << std::endl;
        m_accelerator->enableEdgeAcceleration(true);
        
        // Simulate edge AI workload
        std::cout << "🚀 Running edge AI inference..." << std::endl;
        
        for (int i = 0; i < 5; ++i) {
            std::vector<float> input(100);
            for (auto& val : input) {
                val = static_cast<float>(rand()) / RAND_MAX;
            }
            
            std::vector<float> output;
            auto start = std::chrono::high_resolution_clock::now();
            m_accelerator->performInference(input, output);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "   📊 Edge inference " << (i + 1) << ": " << duration.count() << " μs" << std::endl;
        }
        
        m_accelerator->enableEdgeAcceleration(false);
    }

    void demonstrateDistributedAcceleration() {
        std::cout << "\n🌍 DISTRIBUTED ACCELERATION DEMONSTRATION" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        std::cout << "🔧 Enabling distributed acceleration with 4 nodes..." << std::endl;
        m_accelerator->enableDistributedAcceleration(true, 4);
        
        // Simulate distributed workload
        std::cout << "🚀 Running distributed AI processing..." << std::endl;
        
        std::vector<std::vector<float>> batchInputs(100, std::vector<float>(50));
        for (auto& input : batchInputs) {
            for (auto& val : input) {
                val = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        
        std::vector<std::vector<float>> batchOutputs;
        auto start = std::chrono::high_resolution_clock::now();
        m_accelerator->performBatchInference(batchInputs, batchOutputs);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "   📊 Distributed batch processing: " << batchInputs.size() << " samples in " << duration.count() << " ms" << std::endl;
        std::cout << "   🚀 Average time per sample: " << duration.count() / batchInputs.size() << " ms" << std::endl;
        
        m_accelerator->enableDistributedAcceleration(false, 1);
    }

    void demonstrateRealTimeMonitoring() {
        std::cout << "\n📊 REAL-TIME MONITORING DEMONSTRATION" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        std::cout << "🔧 Enabling real-time monitoring..." << std::endl;
        m_accelerator->enableRealTimeMonitoring(true);
        
        std::cout << "🚀 Running monitored workload for 10 seconds..." << std::endl;
        
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
            // Perform continuous inference
            std::vector<float> input(100);
            for (auto& val : input) {
                val = static_cast<float>(rand()) / RAND_MAX;
            }
            
            std::vector<float> output;
            m_accelerator->performInference(input, output);
            
            // Print status every 2 seconds
            static int counter = 0;
            if (++counter % 20 == 0) { // 20 * 100ms = 2 seconds
                std::cout << "   📊 Status - Power: " << m_accelerator->getPowerConsumption() 
                         << "W, Temp: " << m_accelerator->getThermalStatus() << "°C" << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        m_accelerator->enableRealTimeMonitoring(false);
        std::cout << "✅ Real-time monitoring completed" << std::endl;
    }
};

int main() {
    std::cout << "🚀 Starting ULTIMATE AI Acceleration Demo..." << std::endl;
    
    try {
        AIAccelerationDemo demo;
        demo.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ Demo failed with error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\n🎉 AI Acceleration Demo completed successfully!" << std::endl;
    return 0;
} 