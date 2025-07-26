#!/bin/bash

# AISIS Creative Studio Build Script
# Enhanced with ARM optimization support

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="AISIS Creative Studio"
BUILD_DIR="build"
SOURCE_DIR="aisis"
EXECUTABLE_NAME="aisis_creative_studio"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    AISIS CREATIVE STUDIO                      ║"
    echo "║                      Build System v2.0                       ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Function to detect system architecture
detect_architecture() {
    local arch=$(uname -m)
    print_status "Detected architecture: $arch"
    
    case $arch in
        arm*|aarch64)
            print_success "ARM architecture detected - enabling ARM optimizations"
            export ARM_OPTIMIZATIONS=true
            ;;
        x86_64|amd64)
            print_status "x86_64 architecture detected"
            export ARM_OPTIMIZATIONS=false
            ;;
        *)
            print_warning "Unknown architecture: $arch"
            export ARM_OPTIMIZATIONS=false
            ;;
    esac
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking build dependencies..."
    
    # Check for C++ compiler
    if command -v g++ &> /dev/null; then
        print_success "g++ compiler found: $(g++ --version | head -n1)"
    elif command -v clang++ &> /dev/null; then
        print_success "clang++ compiler found: $(clang++ --version | head -n1)"
    else
        print_error "No C++ compiler found! Please install g++ or clang++"
        exit 1
    fi
    
    # Check for CMake
    if command -v cmake &> /dev/null; then
        print_success "CMake found: $(cmake --version | head -n1)"
        export USE_CMAKE=true
    else
        print_warning "CMake not found - using direct compilation"
        export USE_CMAKE=false
    fi
    
    # Check for make
    if command -v make &> /dev/null; then
        print_success "Make found: $(make --version | head -n1)"
    else
        print_warning "Make not found"
    fi
}

# Function to clean build directory
clean_build() {
    print_status "Cleaning build directory..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
    fi
    mkdir -p "$BUILD_DIR"
}

# Function to build with CMake
build_with_cmake() {
    print_status "Building with CMake..."
    
    cd "$BUILD_DIR"
    
    # Configure
    print_status "Configuring project..."
    cmake "../$SOURCE_DIR" -DCMAKE_BUILD_TYPE=Release
    
    # Build
    print_status "Compiling project..."
    make -j$(nproc)
    
    cd ..
    print_success "CMake build completed"
}

# Function to build directly with g++
build_direct() {
    print_status "Building directly with g++..."
    
    # Compiler flags
    local CXX_FLAGS="-std=c++17 -O3 -Wall -Wextra -pthread"
    
    # Add ARM optimizations if detected
    if [ "$ARM_OPTIMIZATIONS" = true ]; then
        CXX_FLAGS="$CXX_FLAGS -march=native -mfpu=neon -ftree-vectorize"
        print_status "ARM optimizations enabled"
    fi
    
    # Detect compiler
    local COMPILER="g++"
    if command -v clang++ &> /dev/null && [ ! command -v g++ &> /dev/null ]; then
        COMPILER="clang++"
    fi
    
    print_status "Using compiler: $COMPILER"
    print_status "Compiler flags: $CXX_FLAGS"
    
    # Compile
    $COMPILER $CXX_FLAGS -o "$BUILD_DIR/$EXECUTABLE_NAME" "$SOURCE_DIR/main.cpp"
    
    print_success "Direct compilation completed"
}

# Function to run the application
run_application() {
    print_status "Running $PROJECT_NAME..."
    echo -e "${CYAN}"
    
    if [ "$USE_CMAKE" = true ]; then
        "./$BUILD_DIR/bin/$EXECUTABLE_NAME"
    else
        "./$BUILD_DIR/$EXECUTABLE_NAME"
    fi
    
    echo -e "${NC}"
}

# Function to install the application
install_application() {
    print_status "Installing $PROJECT_NAME..."
    
    local install_dir="/usr/local/bin"
    local executable_path
    
    if [ "$USE_CMAKE" = true ]; then
        executable_path="$BUILD_DIR/bin/$EXECUTABLE_NAME"
    else
        executable_path="$BUILD_DIR/$EXECUTABLE_NAME"
    fi
    
    if [ -f "$executable_path" ]; then
        sudo cp "$executable_path" "$install_dir/"
        sudo chmod +x "$install_dir/$EXECUTABLE_NAME"
        print_success "Application installed to $install_dir/$EXECUTABLE_NAME"
    else
        print_error "Executable not found: $executable_path"
        exit 1
    fi
}

# Function to create package
create_package() {
    print_status "Creating distribution package..."
    
    local package_name="aisis-creative-studio-$(date +%Y%m%d)"
    local package_dir="$BUILD_DIR/$package_name"
    
    mkdir -p "$package_dir"
    
    # Copy executable
    if [ "$USE_CMAKE" = true ]; then
        cp "$BUILD_DIR/bin/$EXECUTABLE_NAME" "$package_dir/"
    else
        cp "$BUILD_DIR/$EXECUTABLE_NAME" "$package_dir/"
    fi
    
    # Create README for package
    cat > "$package_dir/README.txt" << EOF
AISIS Creative Studio v2.0
Advanced Multimedia Creative Platform

Installation:
1. Copy the executable to your desired location
2. Make it executable: chmod +x $EXECUTABLE_NAME
3. Run: ./$EXECUTABLE_NAME

Features:
- Audio Processing with ARM optimization
- Video Processing with advanced filters
- Image Processing with AI enhancement
- Project Management system
- Plugin architecture
- Performance profiling
- Resource monitoring
- Multiple themes

For support, visit: https://github.com/your-repo/aisis-creative-studio
EOF
    
    # Create archive
    cd "$BUILD_DIR"
    tar -czf "$package_name.tar.gz" "$package_name"
    cd ..
    
    print_success "Package created: $BUILD_DIR/$package_name.tar.gz"
}

# Function to show system info
show_system_info() {
    print_status "System Information:"
    echo "  OS: $(uname -s)"
    echo "  Architecture: $(uname -m)"
    echo "  Kernel: $(uname -r)"
    echo "  CPU Cores: $(nproc)"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    
    if [ "$ARM_OPTIMIZATIONS" = true ]; then
        echo "  ARM Optimizations: Enabled"
    else
        echo "  ARM Optimizations: Disabled"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build     Build the project (default)"
    echo "  clean     Clean build directory"
    echo "  run       Build and run the application"
    echo "  install   Install the application system-wide"
    echo "  package   Create distribution package"
    echo "  info      Show system information"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Build the project"
    echo "  $0 run          # Build and run"
    echo "  $0 clean build  # Clean and build"
    echo "  $0 package      # Create package"
}

# Main execution
main() {
    print_header
    
    # Parse command line arguments
    local action=${1:-build}
    
    case $action in
        clean)
            clean_build
            ;;
        build)
            detect_architecture
            check_dependencies
            clean_build
            
            if [ "$USE_CMAKE" = true ]; then
                build_with_cmake
            else
                build_direct
            fi
            
            print_success "$PROJECT_NAME built successfully!"
            ;;
        run)
            detect_architecture
            check_dependencies
            clean_build
            
            if [ "$USE_CMAKE" = true ]; then
                build_with_cmake
            else
                build_direct
            fi
            
            run_application
            ;;
        install)
            install_application
            ;;
        package)
            create_package
            ;;
        info)
            show_system_info
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $action"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"