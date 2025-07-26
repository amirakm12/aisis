#!/bin/bash

# ULTIMATE System Build Script
# Comprehensive build automation for all components

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"
ENABLE_TESTS="${ENABLE_TESTS:-ON}"
ENABLE_EXAMPLES="${ENABLE_EXAMPLES:-ON}"

echo -e "${BLUE}=== ULTIMATE System Build Script ===${NC}"
echo -e "${BLUE}Build Type: ${BUILD_TYPE}${NC}"
echo -e "${BLUE}Build Directory: ${BUILD_DIR}${NC}"
echo -e "${BLUE}Install Prefix: ${INSTALL_PREFIX}${NC}"
echo -e "${BLUE}Parallel Jobs: ${PARALLEL_JOBS}${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required tools
check_dependencies() {
    print_status "Checking build dependencies..."
    
    local missing_deps=()
    
    # Check for CMake
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi
    
    # Check for C++ compiler
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        missing_deps+=("g++ or clang++")
    fi
    
    # Check for Python (for advanced pipelines)
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check for pip
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("pip3")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_status "All dependencies found."
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt --user
        print_status "Python dependencies installed."
    else
        print_warning "requirements.txt not found, skipping Python dependencies."
    fi
}

# Clean build directory
clean_build() {
    if [ "$1" = "clean" ] || [ "$1" = "rebuild" ]; then
        print_status "Cleaning build directory..."
        rm -rf "${BUILD_DIR}"
        print_status "Build directory cleaned."
    fi
}

# Configure CMake
configure_cmake() {
    print_status "Configuring CMake..."
    
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    cmake .. \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
        -DBUILD_TESTS="${ENABLE_TESTS}" \
        -DBUILD_EXAMPLES="${ENABLE_EXAMPLES}" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
    cd ..
    print_status "CMake configuration completed."
}

# Build the project
build_project() {
    print_status "Building ULTIMATE System..."
    
    cd "${BUILD_DIR}"
    
    # Build with progress output
    cmake --build . --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}" --verbose
    
    cd ..
    print_status "Build completed successfully."
}

# Run tests
run_tests() {
    if [ "${ENABLE_TESTS}" = "ON" ]; then
        print_status "Running tests..."
        
        cd "${BUILD_DIR}"
        
        # Run CTest
        if ctest --output-on-failure --parallel "${PARALLEL_JOBS}"; then
            print_status "All tests passed."
        else
            print_error "Some tests failed."
            exit 1
        fi
        
        cd ..
    else
        print_warning "Tests are disabled."
    fi
}

# Install the project
install_project() {
    if [ "$1" = "install" ]; then
        print_status "Installing ULTIMATE System..."
        
        cd "${BUILD_DIR}"
        
        if [ "$(id -u)" -eq 0 ]; then
            cmake --install .
        else
            sudo cmake --install .
        fi
        
        cd ..
        print_status "Installation completed."
    fi
}

# Build Python components
build_python_components() {
    print_status "Building Python components..."
    
    # Build advanced pipelines
    if [ -d "advanced_pipelines" ]; then
        print_status "Building advanced pipelines..."
        cd advanced_pipelines
        python3 -m py_compile *.py
        cd ..
        print_status "Advanced pipelines built."
    fi
    
    # Build RAG system
    if [ -d "advanced_rag_system" ]; then
        print_status "Building RAG system..."
        cd advanced_rag_system
        find . -name "*.py" -exec python3 -m py_compile {} \;
        cd ..
        print_status "RAG system built."
    fi
    
    print_status "Python components built successfully."
}

# Package the system
package_system() {
    if [ "$1" = "package" ]; then
        print_status "Creating package..."
        
        cd "${BUILD_DIR}"
        
        # Create package using CPack
        cpack
        
        cd ..
        print_status "Package created successfully."
    fi
}

# Generate documentation
generate_docs() {
    if [ "$1" = "docs" ]; then
        print_status "Generating documentation..."
        
        if command -v doxygen &> /dev/null; then
            doxygen Doxyfile
            print_status "Documentation generated."
        else
            print_warning "Doxygen not found, skipping documentation generation."
        fi
    fi
}

# Run benchmarks
run_benchmarks() {
    if [ "$1" = "benchmark" ]; then
        print_status "Running benchmarks..."
        
        cd "${BUILD_DIR}"
        
        # Run benchmark executables
        if [ -f "benchmark_ultimate" ]; then
            ./benchmark_ultimate
        fi
        
        # Run Python benchmarks
        cd ..
        if [ -f "advanced_pipelines/comprehensive_benchmark.py" ]; then
            python3 advanced_pipelines/comprehensive_benchmark.py
        fi
        
        print_status "Benchmarks completed."
    fi
}

# Display build summary
build_summary() {
    print_status "=== Build Summary ==="
    echo "Build Type: ${BUILD_TYPE}"
    echo "Build Directory: ${BUILD_DIR}"
    echo "Tests Enabled: ${ENABLE_TESTS}"
    echo "Examples Enabled: ${ENABLE_EXAMPLES}"
    
    if [ -d "${BUILD_DIR}" ]; then
        echo "Build Size: $(du -sh ${BUILD_DIR} | cut -f1)"
    fi
    
    print_status "Build completed successfully!"
}

# Main build process
main() {
    local command="${1:-build}"
    
    case "${command}" in
        "clean")
            clean_build clean
            ;;
        "configure")
            check_dependencies
            install_python_deps
            configure_cmake
            ;;
        "build")
            check_dependencies
            install_python_deps
            clean_build "$@"
            configure_cmake
            build_project
            build_python_components
            run_tests
            build_summary
            ;;
        "rebuild")
            check_dependencies
            install_python_deps
            clean_build rebuild
            configure_cmake
            build_project
            build_python_components
            run_tests
            build_summary
            ;;
        "test")
            run_tests
            ;;
        "install")
            install_project install
            ;;
        "package")
            package_system package
            ;;
        "docs")
            generate_docs docs
            ;;
        "benchmark")
            run_benchmarks benchmark
            ;;
        "all")
            check_dependencies
            install_python_deps
            clean_build clean
            configure_cmake
            build_project
            build_python_components
            run_tests
            install_project install
            package_system package
            generate_docs docs
            build_summary
            ;;
        "help"|"-h"|"--help")
            echo "ULTIMATE System Build Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  build      - Build the system (default)"
            echo "  rebuild    - Clean and build"
            echo "  clean      - Clean build directory"
            echo "  configure  - Configure CMake only"
            echo "  test       - Run tests"
            echo "  install    - Install the system"
            echo "  package    - Create package"
            echo "  docs       - Generate documentation"
            echo "  benchmark  - Run benchmarks"
            echo "  all        - Do everything"
            echo "  help       - Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  BUILD_TYPE     - Build type (Debug/Release) [default: Release]"
            echo "  BUILD_DIR      - Build directory [default: build]"
            echo "  INSTALL_PREFIX - Install prefix [default: /usr/local]"
            echo "  PARALLEL_JOBS  - Number of parallel jobs [default: $(nproc)]"
            echo "  ENABLE_TESTS   - Enable tests (ON/OFF) [default: ON]"
            echo "  ENABLE_EXAMPLES- Enable examples (ON/OFF) [default: ON]"
            echo ""
            echo "Examples:"
            echo "  $0 build                    # Build with default settings"
            echo "  BUILD_TYPE=Debug $0 build   # Debug build"
            echo "  $0 rebuild                  # Clean and rebuild"
            echo "  $0 all                      # Complete build and install"
            ;;
        *)
            print_error "Unknown command: ${command}"
            print_error "Use '$0 help' for usage information."
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"