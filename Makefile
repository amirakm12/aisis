# Hardcore Performance Computing Makefile
# Optimized for maximum performance with aggressive compiler flags

CXX = g++
CC = gcc

# Hardcore optimization flags
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -flto -ffast-math \
           -funroll-loops -fprefetch-loop-arrays -ftree-vectorize \
           -fomit-frame-pointer -DNDEBUG -mavx2 -mfma -msse4.2 \
           -Wall -Wextra -pthread

# Additional flags for maximum performance
PERFORMANCE_FLAGS = -fno-stack-protector -fno-rtti \
                   -finline-functions -fipa-pta -fgcse-after-reload \
                   -ftracer -fvect-cost-model=unlimited

# Linux specific flags
LINUX_FLAGS = -D_GNU_SOURCE -lrt -ldl

# Windows specific flags (when cross-compiling)
WINDOWS_FLAGS = -lpsapi -lpdh

# Directories
SRCDIR = src
SCRIPTDIR = scripts
BINDIR = bin
OBJDIR = obj

# Source files
HARDCORE_SRC = $(SRCDIR)/hardcore_performance.cpp
MONITOR_SRC = $(SRCDIR)/performance_monitor.cpp

# Object files
HARDCORE_OBJ = $(OBJDIR)/hardcore_performance.o
MONITOR_OBJ = $(OBJDIR)/performance_monitor.o

# Executables
HARDCORE_BIN = $(BINDIR)/hardcore_performance
MONITOR_BIN = $(BINDIR)/performance_monitor

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    PLATFORM_FLAGS = $(LINUX_FLAGS)
    OPTIMIZATION_SCRIPT = $(SCRIPTDIR)/linux_optimization.sh
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM_FLAGS = -framework CoreFoundation
    OPTIMIZATION_SCRIPT = $(SCRIPTDIR)/macos_optimization.sh
endif

# Default target
.PHONY: all clean install hardcore monitor optimize test benchmark help

all: $(HARDCORE_BIN) $(MONITOR_BIN)

# Create directories
$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(BINDIR):
	@mkdir -p $(BINDIR)

# Hardcore performance engine
$(HARDCORE_OBJ): $(HARDCORE_SRC) | $(OBJDIR)
	@echo "🔥 Compiling hardcore performance engine with maximum optimization..."
	$(CXX) $(CXXFLAGS) $(PERFORMANCE_FLAGS) $(PLATFORM_FLAGS) -c $< -o $@

$(HARDCORE_BIN): $(HARDCORE_OBJ) | $(BINDIR)
	@echo "🚀 Linking hardcore performance engine..."
	$(CXX) $(CXXFLAGS) $(PERFORMANCE_FLAGS) $(PLATFORM_FLAGS) $< -o $@
	@echo "✅ Hardcore performance engine built: $@"

# Performance monitor
$(MONITOR_OBJ): $(MONITOR_SRC) | $(OBJDIR)
	@echo "📊 Compiling performance monitor..."
	$(CXX) $(CXXFLAGS) $(PLATFORM_FLAGS) -c $< -o $@

$(MONITOR_BIN): $(MONITOR_OBJ) | $(BINDIR)
	@echo "🔗 Linking performance monitor..."
	$(CXX) $(CXXFLAGS) $(PLATFORM_FLAGS) $< -o $@
	@echo "✅ Performance monitor built: $@"

# Individual targets
hardcore: $(HARDCORE_BIN)

monitor: $(MONITOR_BIN)

# System optimization
optimize:
	@echo "⚡ Running system optimization..."
	@if [ -f "$(OPTIMIZATION_SCRIPT)" ]; then \
		chmod +x $(OPTIMIZATION_SCRIPT); \
		sudo $(OPTIMIZATION_SCRIPT); \
	else \
		echo "❌ Optimization script not found for this platform"; \
	fi

# Install binaries to system path
install: all
	@echo "📦 Installing binaries..."
	sudo cp $(HARDCORE_BIN) /usr/local/bin/
	sudo cp $(MONITOR_BIN) /usr/local/bin/
	@echo "✅ Binaries installed to /usr/local/bin/"

# Performance test
test: all
	@echo "🧪 Running performance tests..."
	@echo "Testing hardcore performance engine (10 seconds)..."
	$(HARDCORE_BIN) 10
	@echo "✅ Performance test completed"

# Comprehensive benchmark
benchmark: all
	@echo "🏁 Running comprehensive benchmark suite..."
	@echo "=== Small Vector Test (1K elements) ==="
	$(HARDCORE_BIN) 5 1024
	@echo "=== Medium Vector Test (1M elements) ==="
	$(HARDCORE_BIN) 10 1048576
	@echo "=== Large Vector Test (10M elements) ==="
	$(HARDCORE_BIN) 15 10485760
	@echo "✅ Benchmark suite completed"

# Monitor system during benchmark
monitor-benchmark: all
	@echo "📊 Starting system monitoring during benchmark..."
	$(MONITOR_BIN) 500 & \
	MONITOR_PID=$$!; \
	sleep 2; \
	$(HARDCORE_BIN) 30; \
	kill $$MONITOR_PID 2>/dev/null || true

# Debug build (for development)
debug: CXXFLAGS = -std=c++17 -O0 -g -Wall -Wextra -pthread -mavx2 -DDEBUG
debug: PERFORMANCE_FLAGS = 
debug: all
	@echo "🐛 Debug build completed"

# Profile build (for profiling with perf/gprof)
profile: CXXFLAGS += -pg -g -fno-omit-frame-pointer
profile: all
	@echo "📈 Profile build completed"

# Static analysis
analyze:
	@echo "🔍 Running static analysis..."
	@command -v cppcheck >/dev/null 2>&1 && cppcheck --enable=all $(SRCDIR)/*.cpp || echo "cppcheck not found"
	@command -v clang-tidy >/dev/null 2>&1 && clang-tidy $(SRCDIR)/*.cpp -- $(CXXFLAGS) || echo "clang-tidy not found"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(OBJDIR) $(BINDIR)
	@echo "✅ Clean completed"

# System information
sysinfo:
	@echo "💻 System Information:"
	@echo "OS: $(UNAME_S)"
	@echo "CPU: $$(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2 | xargs)"
	@echo "Cores: $$(nproc)"
	@echo "Memory: $$(free -h | grep Mem | awk '{print $$2}')"
	@echo "Compiler: $$($(CXX) --version | head -1)"
	@echo "AVX2 Support: $$(grep -q avx2 /proc/cpuinfo && echo 'Yes' || echo 'No')"
	@echo "FMA Support: $$(grep -q fma /proc/cpuinfo && echo 'Yes' || echo 'No')"

# Performance tips
tips:
	@echo "🎯 HARDCORE PERFORMANCE TIPS:"
	@echo
	@echo "🔥 Compilation:"
	@echo "   - Built with -O3 -march=native -mtune=native for your specific CPU"
	@echo "   - LTO (Link Time Optimization) enabled for cross-function optimization"
	@echo "   - AVX2/FMA instructions enabled for SIMD acceleration"
	@echo
	@echo "⚡ System Optimization:"
	@echo "   - Run 'make optimize' to configure system for maximum performance"
	@echo "   - Use 'sudo nice -n -20' to run with highest priority"
	@echo "   - Disable CPU frequency scaling: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
	@echo "   - Pin process to specific cores: taskset -c 0-7 ./hardcore_performance"
	@echo
	@echo "🌡️  Thermal Management:"
	@echo "   - Monitor temperatures: watch sensors"
	@echo "   - Ensure adequate cooling before running intensive workloads"
	@echo "   - Use 'make monitor-benchmark' to track system metrics during execution"
	@echo
	@echo "📊 Profiling:"
	@echo "   - Use 'make profile' to build with profiling support"
	@echo "   - Profile with: perf record -g ./hardcore_performance && perf report"
	@echo "   - Memory profiling: valgrind --tool=massif ./hardcore_performance"

# Help
help:
	@echo "🚀 HARDCORE PERFORMANCE COMPUTING BUILD SYSTEM"
	@echo
	@echo "Available targets:"
	@echo "  all              - Build all applications (default)"
	@echo "  hardcore         - Build hardcore performance engine only"
	@echo "  monitor          - Build performance monitor only"
	@echo "  optimize         - Run system optimization script (requires sudo)"
	@echo "  install          - Install binaries to /usr/local/bin (requires sudo)"
	@echo "  test             - Run quick performance test"
	@echo "  benchmark        - Run comprehensive benchmark suite"
	@echo "  monitor-benchmark- Run benchmark with system monitoring"
	@echo "  debug            - Build debug version"
	@echo "  profile          - Build with profiling support"
	@echo "  analyze          - Run static analysis tools"
	@echo "  clean            - Remove build artifacts"
	@echo "  sysinfo          - Display system information"
	@echo "  tips             - Show performance optimization tips"
	@echo "  help             - Show this help message"
	@echo
	@echo "Usage examples:"
	@echo "  make && make optimize    # Build and optimize system"
	@echo "  make benchmark           # Run performance benchmarks"
	@echo "  make monitor-benchmark   # Monitor system during benchmark"
	@echo "  sudo nice -n -20 make test  # Run with highest priority"

# Prevent make from deleting intermediate files
.PRECIOUS: $(HARDCORE_OBJ) $(MONITOR_OBJ)