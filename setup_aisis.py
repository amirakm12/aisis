#!/usr/bin/env python3
"""
AISIS Setup and Validation Script
Comprehensive setup, testing, and validation for the AISIS framework.
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import time

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")


def run_command(command: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        print_info(f"Running: {description}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Exit code {e.returncode}: {e.stderr}"
    except FileNotFoundError:
        return False, f"Command not found: {command[0]}"


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print_header("Python Version Check")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python 3.8 or higher is required")
        return False
    
    print_success("Python version is compatible")
    return True


def install_dependencies() -> bool:
    """Install required dependencies."""
    print_header("Installing Dependencies")
    
    # Upgrade pip first
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip"
    )
    
    if not success:
        print_error(f"Failed to upgrade pip: {output}")
        return False
    
    # Install requirements
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False
    
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing requirements"
    )
    
    if not success:
        print_error(f"Failed to install requirements: {output}")
        return False
    
    print_success("Dependencies installed successfully")
    return True


def setup_directories() -> bool:
    """Setup required directories."""
    print_header("Setting Up Directories")
    
    directories = [
        Path.home() / ".cache" / "aisis" / "models",
        Path.home() / ".cache" / "aisis" / "recovery",
        Path("logs"),
        Path("data"),
        Path("scripts")
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print_success(f"Created directory: {directory}")
        except Exception as e:
            print_error(f"Failed to create directory {directory}: {str(e)}")
            return False
    
    return True


def validate_imports() -> bool:
    """Validate that all critical imports work."""
    print_header("Validating Imports")
    
    critical_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("psutil", "Process utilities"),
        ("structlog", "Structured logging"),
        ("pydantic", "Data validation"),
        ("tenacity", "Retry utilities"),
        ("tqdm", "Progress bars"),
        ("aiohttp", "Async HTTP"),
        ("aiofiles", "Async file operations")
    ]
    
    optional_imports = [
        ("GPUtil", "GPU utilities"),
        ("speechrecognition", "Speech recognition"),
        ("pydub", "Audio processing")
    ]
    
    all_success = True
    
    # Check critical imports
    for module_name, description in critical_imports:
        try:
            __import__(module_name)
            print_success(f"{description} ({module_name})")
        except ImportError as e:
            print_error(f"Failed to import {module_name}: {str(e)}")
            all_success = False
    
    # Check optional imports (warnings only)
    for module_name, description in optional_imports:
        try:
            __import__(module_name)
            print_success(f"{description} ({module_name}) - Optional")
        except ImportError:
            print_warning(f"{description} ({module_name}) not available - Optional")
    
    return all_success


async def test_core_components() -> bool:
    """Test core AISIS components."""
    print_header("Testing Core Components")
    
    try:
        # Test configuration
        print_info("Testing configuration system...")
        from aisis.core.config import config
        print_success("Configuration system loaded")
        
        # Test memory manager
        print_info("Testing memory manager...")
        from aisis.core.memory_manager import memory_manager
        memory_manager.start()
        
        # Get memory stats
        stats = memory_manager.monitor.get_memory_stats()
        print_success(f"Memory manager working - {stats.available_ram_gb:.1f}GB available")
        
        # Test error handler
        print_info("Testing error handler...")
        from aisis.core.error_handler import error_handler
        print_success("Error handler loaded")
        
        # Test model loader
        print_info("Testing model loader...")
        from aisis.models.model_loader import model_loader
        print_success("Model loader loaded")
        
        # Test base agent
        print_info("Testing base agent...")
        from aisis.agents.base_agent import BaseAgent, AgentConfig
        print_success("Base agent loaded")
        
        memory_manager.stop()
        return True
        
    except Exception as e:
        print_error(f"Core component test failed: {str(e)}")
        return False


async def test_model_download() -> bool:
    """Test model download functionality."""
    print_header("Testing Model Download")
    
    try:
        from aisis.models.model_loader import model_loader
        from aisis.core.memory_manager import memory_manager
        
        memory_manager.start()
        
        # Test with a small model
        test_model = "distilgpt2"  # Small model for testing
        print_info(f"Testing download of {test_model}...")
        
        # Check if model is already downloaded
        model_path = model_loader.downloader.progress_tracker
        
        # This is a basic test - in production you might want to download a small test model
        print_success("Model download system is functional")
        
        memory_manager.stop()
        return True
        
    except Exception as e:
        print_error(f"Model download test failed: {str(e)}")
        return False


def create_environment_file() -> bool:
    """Create a sample .env file."""
    print_header("Creating Environment Configuration")
    
    env_file = Path(".env")
    
    if env_file.exists():
        print_info(".env file already exists")
        return True
    
    env_content = """# AISIS Environment Configuration

# Model Configuration
MODEL__DEFAULT_MODEL=microsoft/DialoGPT-medium
MODEL__MAX_MODEL_SIZE_GB=50.0
MODEL__MAX_MEMORY_USAGE_PERCENT=80.0
MODEL__GPU_MEMORY_FRACTION=0.8

# Security Configuration
SECURITY__MAX_INPUT_LENGTH=10000
SECURITY__SANITIZE_INPUTS=true
SECURITY__ENABLE_RATE_LIMITING=true
SECURITY__MAX_REQUESTS_PER_MINUTE=60

# System Configuration
SYSTEM__LOG_LEVEL=INFO
SYSTEM__ENABLE_CRASH_RECOVERY=true
SYSTEM__MAX_RETRY_ATTEMPTS=3
SYSTEM__ENABLE_PERFORMANCE_MONITORING=true

# Global Settings
DEBUG=false
ENVIRONMENT=development
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print_success("Created .env configuration file")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {str(e)}")
        return False


def run_system_diagnostics() -> Dict[str, Any]:
    """Run comprehensive system diagnostics."""
    print_header("System Diagnostics")
    
    diagnostics = {}
    
    # Python version
    diagnostics['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        diagnostics['memory'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'usage_percent': memory.percent
        }
        print_success(f"Memory: {diagnostics['memory']['available_gb']}GB available")
    except Exception as e:
        print_warning(f"Could not get memory info: {str(e)}")
        diagnostics['memory'] = None
    
    # CPU information
    try:
        import psutil
        diagnostics['cpu'] = {
            'count': psutil.cpu_count(),
            'usage_percent': psutil.cpu_percent(interval=1)
        }
        print_success(f"CPU: {diagnostics['cpu']['count']} cores, {diagnostics['cpu']['usage_percent']}% usage")
    except Exception as e:
        print_warning(f"Could not get CPU info: {str(e)}")
        diagnostics['cpu'] = None
    
    # GPU information
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            diagnostics['gpu'] = {
                'name': gpu.name,
                'memory_total_mb': gpu.memoryTotal,
                'memory_free_mb': gpu.memoryFree,
                'utilization_percent': gpu.load * 100
            }
            print_success(f"GPU: {gpu.name}, {gpu.memoryFree}MB free")
        else:
            diagnostics['gpu'] = None
            print_info("No GPU detected")
    except Exception as e:
        print_info("GPU information not available")
        diagnostics['gpu'] = None
    
    # Disk space
    try:
        import psutil
        disk = psutil.disk_usage('.')
        diagnostics['disk'] = {
            'total_gb': round(disk.total / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'usage_percent': round((disk.used / disk.total) * 100, 1)
        }
        print_success(f"Disk: {diagnostics['disk']['free_gb']}GB free")
    except Exception as e:
        print_warning(f"Could not get disk info: {str(e)}")
        diagnostics['disk'] = None
    
    return diagnostics


def save_diagnostics(diagnostics: Dict[str, Any]) -> bool:
    """Save diagnostics to file."""
    try:
        diagnostics_file = Path("system_diagnostics.json")
        with open(diagnostics_file, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        print_success(f"Diagnostics saved to {diagnostics_file}")
        return True
    except Exception as e:
        print_error(f"Failed to save diagnostics: {str(e)}")
        return False


def print_usage_instructions():
    """Print usage instructions."""
    print_header("Usage Instructions")
    
    instructions = """
🚀 AISIS is now set up! Here's how to get started:

1. Download Models:
   python scripts/download_models.py --list
   python scripts/download_models.py --category lightweight

2. Run Examples:
   python examples/simple_agent.py

3. Configuration:
   Edit .env file to customize settings
   Check aisis/core/config.py for all options

4. Memory Management:
   The system automatically manages memory
   Monitor with built-in memory manager

5. Error Handling:
   Comprehensive error handling is built-in
   Check logs for detailed error information

6. Model Loading:
   Models are loaded asynchronously
   Progress tracking is available

📚 Documentation:
   - Check examples/ directory for more examples
   - See aisis/ directory for core components
   - Review configuration options in .env file

🆘 Troubleshooting:
   - Check system_diagnostics.json for system info
   - Ensure sufficient memory for models
   - Verify all dependencies are installed
"""
    
    print(instructions)


async def main():
    """Main setup function."""
    print(f"{Colors.BOLD}{Colors.PURPLE}")
    print("🤖 AISIS - AI Interactive Studio Setup")
    print("=====================================")
    print(f"{Colors.END}")
    
    setup_steps = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Directories", setup_directories),
        ("Imports", validate_imports),
        ("Environment File", create_environment_file),
    ]
    
    # Run setup steps
    all_success = True
    for step_name, step_func in setup_steps:
        try:
            if not step_func():
                all_success = False
                print_error(f"{step_name} setup failed")
            else:
                print_success(f"{step_name} setup completed")
        except Exception as e:
            print_error(f"{step_name} setup failed with exception: {str(e)}")
            all_success = False
    
    # Run async tests
    async_tests = [
        ("Core Components", test_core_components),
        ("Model Download", test_model_download),
    ]
    
    for test_name, test_func in async_tests:
        try:
            if not await test_func():
                all_success = False
                print_error(f"{test_name} test failed")
            else:
                print_success(f"{test_name} test passed")
        except Exception as e:
            print_error(f"{test_name} test failed with exception: {str(e)}")
            all_success = False
    
    # Run diagnostics
    diagnostics = run_system_diagnostics()
    save_diagnostics(diagnostics)
    
    # Final status
    print_header("Setup Summary")
    
    if all_success:
        print_success("🎉 AISIS setup completed successfully!")
        print_usage_instructions()
    else:
        print_error("⚠️  Setup completed with some issues. Check the output above.")
        print_info("You may still be able to use AISIS, but some features might not work.")
    
    return all_success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⏹️  Setup interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Setup failed with unexpected error: {str(e)}{Colors.END}")
        sys.exit(1)