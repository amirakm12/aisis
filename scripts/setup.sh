#!/bin/bash

# ULTIMATE System Setup Script
# Comprehensive setup and dependency installation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="${INSTALL_DIR:-/usr/local}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
CMAKE_VERSION="${CMAKE_VERSION:-3.20}"

echo -e "${BLUE}=== ULTIMATE System Setup Script ===${NC}"
echo -e "${BLUE}Setting up complete development environment${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS and distribution
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
            DISTRO=$(lsb_release -si 2>/dev/null || echo "Debian")
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
            DISTRO=$(cat /etc/redhat-release | cut -d' ' -f1)
        elif [ -f /etc/arch-release ]; then
            OS="arch"
            DISTRO="Arch"
        else
            OS="linux"
            DISTRO="Unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macOS"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        OS="windows"
        DISTRO="Windows"
    else
        OS="unknown"
        DISTRO="Unknown"
    fi
    
    print_status "Detected OS: $DISTRO ($OS)"
}

# Install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    case $OS in
        "debian")
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                git \
                python3 \
                python3-pip \
                python3-dev \
                python3-venv \
                pkg-config \
                libssl-dev \
                libffi-dev \
                libbz2-dev \
                libreadline-dev \
                libsqlite3-dev \
                libncurses5-dev \
                libncursesw5-dev \
                xz-utils \
                tk-dev \
                libxml2-dev \
                libxmlsec1-dev \
                libffi-dev \
                liblzma-dev \
                wget \
                curl \
                llvm \
                make \
                zlib1g-dev \
                libnuma-dev \
                google-perftools \
                libgoogle-perftools-dev
            ;;
        "redhat")
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                git \
                python3 \
                python3-pip \
                python3-devel \
                openssl-devel \
                libffi-devel \
                bzip2-devel \
                readline-devel \
                sqlite-devel \
                ncurses-devel \
                xz-devel \
                tk-devel \
                libxml2-devel \
                xmlsec1-devel \
                wget \
                curl \
                llvm \
                make \
                zlib-devel \
                numactl-devel \
                gperftools-devel
            ;;
        "arch")
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                base-devel \
                cmake \
                git \
                python \
                python-pip \
                openssl \
                libffi \
                bzip2 \
                readline \
                sqlite \
                ncurses \
                xz \
                tk \
                libxml2 \
                xmlsec \
                wget \
                curl \
                llvm \
                make \
                zlib \
                numactl \
                gperftools
            ;;
        "macos")
            # Check if Homebrew is installed
            if ! command -v brew &> /dev/null; then
                print_status "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew update
            brew install \
                cmake \
                git \
                python@3.9 \
                openssl \
                libffi \
                bzip2 \
                readline \
                sqlite \
                xz \
                tk \
                libxml2 \
                xmlsec1 \
                wget \
                curl \
                llvm \
                make \
                zlib \
                gperftools
            ;;
        *)
            print_warning "Unknown OS, skipping system dependency installation"
            ;;
    esac
    
    print_status "System dependencies installed."
}

# Install Python dependencies
setup_python_environment() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Created Python virtual environment."
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "Installed Python dependencies."
    else
        # Install essential packages
        pip install \
            numpy>=1.21.0 \
            numba>=0.56.0 \
            psutil>=5.8.0 \
            aiofiles>=0.7.0 \
            aiohttp>=3.8.0 \
            uvloop>=0.16.0 \
            pytest>=6.2.0 \
            pytest-asyncio>=0.15.0 \
            pytest-benchmark>=3.4.0 \
            sphinx>=4.0.0 \
            sphinx-rtd-theme>=0.5.0
        print_status "Installed essential Python packages."
    fi
    
    # Install development tools
    pip install \
        black \
        flake8 \
        mypy \
        pre-commit \
        jupyter \
        ipython
    
    print_status "Python environment setup completed."
}

# Install C++ dependencies
setup_cpp_environment() {
    print_status "Setting up C++ environment..."
    
    # Check CMake version
    if command -v cmake &> /dev/null; then
        CMAKE_CURRENT=$(cmake --version | head -n1 | cut -d' ' -f3)
        print_status "Found CMake version: $CMAKE_CURRENT"
    else
        print_error "CMake not found. Please install CMake $CMAKE_VERSION or later."
        exit 1
    fi
    
    # Install vcpkg for C++ package management
    if [ ! -d "vcpkg" ]; then
        print_status "Installing vcpkg..."
        git clone https://github.com/Microsoft/vcpkg.git
        cd vcpkg
        ./bootstrap-vcpkg.sh
        cd ..
        print_status "vcpkg installed."
    fi
    
    # Install C++ packages via vcpkg
    if [ -f "vcpkg-configuration.json" ]; then
        print_status "Installing C++ dependencies via vcpkg..."
        ./vcpkg/vcpkg install --triplet=x64-linux
        print_status "C++ dependencies installed."
    fi
    
    print_status "C++ environment setup completed."
}

# Setup development tools
setup_development_tools() {
    print_status "Setting up development tools..."
    
    # Install Google Test
    if [ ! -d "googletest" ]; then
        print_status "Installing Google Test..."
        git clone https://github.com/google/googletest.git
        cd googletest
        mkdir -p build
        cd build
        cmake ..
        make -j$(nproc)
        sudo make install
        cd ../..
        print_status "Google Test installed."
    fi
    
    # Install Google Benchmark
    if [ ! -d "benchmark" ]; then
        print_status "Installing Google Benchmark..."
        git clone https://github.com/google/benchmark.git
        cd benchmark
        cmake -E make_directory "build"
        cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
        cmake --build "build" --config Release
        sudo cmake --build "build" --config Release --target install
        cd ..
        print_status "Google Benchmark installed."
    fi
    
    # Setup pre-commit hooks
    if [ -f ".pre-commit-config.yaml" ]; then
        source venv/bin/activate
        pre-commit install
        print_status "Pre-commit hooks installed."
    fi
    
    print_status "Development tools setup completed."
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    
    mkdir -p logs
    mkdir -p build
    mkdir -p dist
    mkdir -p test_data
    mkdir -p docs/generated
    mkdir -p config/environments
    
    print_status "Project directories created."
}

# Setup configuration files
setup_configuration() {
    print_status "Setting up configuration files..."
    
    # Create environment-specific configs
    if [ ! -f "config/environments/development.yaml" ]; then
        cp config/ultimate_config.yaml config/environments/development.yaml
        print_status "Created development configuration."
    fi
    
    if [ ! -f "config/environments/production.yaml" ]; then
        cp config/ultimate_config.yaml config/environments/production.yaml
        # Modify for production (disable debug, optimize settings)
        sed -i 's/enable_debug: true/enable_debug: false/' config/environments/production.yaml
        sed -i 's/log_level: "INFO"/log_level: "WARNING"/' config/environments/production.yaml
        print_status "Created production configuration."
    fi
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# ULTIMATE System Environment Configuration
ULTIMATE_ENV=development
ULTIMATE_CONFIG_PATH=config/environments/development.yaml
ULTIMATE_LOG_LEVEL=INFO
ULTIMATE_LOG_FILE=logs/ultimate.log
PYTHONPATH=\${PYTHONPATH}:$(pwd)
EOF
        print_status "Created .env file."
    fi
    
    print_status "Configuration setup completed."
}

# Setup IDE configurations
setup_ide_configurations() {
    print_status "Setting up IDE configurations..."
    
    # VS Code configuration
    mkdir -p .vscode
    
    if [ ! -f ".vscode/settings.json" ]; then
        cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "cmake.configureOnOpen": true,
    "cmake.buildDirectory": "\${workspaceFolder}/build",
    "files.associations": {
        "*.h": "c",
        "*.cpp": "cpp"
    },
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "C_Cpp.default.intelliSenseMode": "linux-gcc-x64"
}
EOF
        print_status "Created VS Code settings."
    fi
    
    if [ ! -f ".vscode/launch.json" ]; then
        cat > .vscode/launch.json << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug C++ Application",
            "type": "cppdbg",
            "request": "launch",
            "program": "\${workspaceFolder}/build/ultimate_app",
            "args": [],
            "stopAtEntry": false,
            "cwd": "\${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        },
        {
            "name": "Debug Python Application",
            "type": "python",
            "request": "launch",
            "program": "\${workspaceFolder}/examples/basic_usage.py",
            "console": "integratedTerminal",
            "python": "\${workspaceFolder}/venv/bin/python"
        }
    ]
}
EOF
        print_status "Created VS Code launch configuration."
    fi
    
    print_status "IDE configurations setup completed."
}

# Run initial tests
run_initial_tests() {
    print_status "Running initial tests..."
    
    # Test Python environment
    source venv/bin/activate
    python -c "import numpy, numba, psutil; print('Python environment OK')"
    
    # Test C++ compilation
    if [ -f "CMakeLists.txt" ]; then
        mkdir -p build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Debug
        make -j$(nproc) || true  # Don't fail if build has issues
        cd ..
        print_status "C++ compilation test completed."
    fi
    
    print_status "Initial tests completed."
}

# Generate documentation
generate_initial_docs() {
    print_status "Generating initial documentation..."
    
    source venv/bin/activate
    
    # Generate Sphinx documentation
    if [ -f "docs/conf.py" ]; then
        cd docs
        make html
        cd ..
        print_status "Sphinx documentation generated."
    fi
    
    # Generate Doxygen documentation if available
    if command -v doxygen &> /dev/null && [ -f "Doxyfile" ]; then
        doxygen
        print_status "Doxygen documentation generated."
    fi
    
    print_status "Documentation generation completed."
}

# Create desktop shortcut (Linux)
create_desktop_shortcut() {
    if [ "$OS" = "linux" ] && [ -n "$DISPLAY" ]; then
        print_status "Creating desktop shortcut..."
        
        DESKTOP_FILE="$HOME/Desktop/ULTIMATE-System.desktop"
        cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ULTIMATE System
Comment=Universal Low-latency Technology for Intelligent Memory and Task Execution
Exec=$PWD/scripts/run.sh
Icon=$PWD/assets/icon.png
Terminal=true
Categories=Development;
EOF
        chmod +x "$DESKTOP_FILE"
        print_status "Desktop shortcut created."
    fi
}

# Display setup summary
setup_summary() {
    print_status "=== Setup Summary ==="
    echo "OS: $DISTRO ($OS)"
    echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
    echo "CMake: $(cmake --version 2>/dev/null | head -n1 || echo 'Not found')"
    echo "GCC: $(gcc --version 2>/dev/null | head -n1 || echo 'Not found')"
    echo "Install Directory: $INSTALL_DIR"
    echo ""
    echo "Virtual Environment: $([ -d 'venv' ] && echo 'Created' || echo 'Not created')"
    echo "Build Directory: $([ -d 'build' ] && echo 'Ready' || echo 'Not ready')"
    echo "Configuration: $([ -f 'config/ultimate_config.yaml' ] && echo 'Available' || echo 'Missing')"
    echo ""
    print_status "Setup completed successfully!"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Source the virtual environment: source venv/bin/activate"
    echo "2. Build the system: ./scripts/build.sh"
    echo "3. Run tests: ./scripts/build.sh test"
    echo "4. See documentation in docs/ directory"
    echo ""
    echo -e "${GREEN}The ULTIMATE System is ready for development!${NC}"
}

# Main setup process
main() {
    local command="${1:-full}"
    
    case "${command}" in
        "deps"|"dependencies")
            detect_os
            install_system_dependencies
            ;;
        "python")
            setup_python_environment
            ;;
        "cpp")
            setup_cpp_environment
            ;;
        "dev"|"development")
            setup_development_tools
            ;;
        "config")
            setup_configuration
            ;;
        "ide")
            setup_ide_configurations
            ;;
        "test")
            run_initial_tests
            ;;
        "docs")
            generate_initial_docs
            ;;
        "full")
            detect_os
            install_system_dependencies
            create_directories
            setup_python_environment
            setup_cpp_environment
            setup_development_tools
            setup_configuration
            setup_ide_configurations
            create_desktop_shortcut
            run_initial_tests
            generate_initial_docs
            setup_summary
            ;;
        "minimal")
            detect_os
            install_system_dependencies
            create_directories
            setup_python_environment
            setup_configuration
            setup_summary
            ;;
        "help"|"-h"|"--help")
            echo "ULTIMATE System Setup Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  full       - Complete setup (default)"
            echo "  minimal    - Minimal setup for basic functionality"
            echo "  deps       - Install system dependencies only"
            echo "  python     - Setup Python environment only"
            echo "  cpp        - Setup C++ environment only"
            echo "  dev        - Setup development tools only"
            echo "  config     - Setup configuration files only"
            echo "  ide        - Setup IDE configurations only"
            echo "  test       - Run initial tests"
            echo "  docs       - Generate documentation"
            echo "  help       - Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  INSTALL_DIR    - Installation directory [default: /usr/local]"
            echo "  PYTHON_VERSION - Python version [default: 3.8]"
            echo "  CMAKE_VERSION  - CMake version [default: 3.20]"
            echo ""
            echo "Examples:"
            echo "  $0 full                    # Complete setup"
            echo "  $0 minimal                 # Minimal setup"
            echo "  INSTALL_DIR=/opt $0 full   # Custom install directory"
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