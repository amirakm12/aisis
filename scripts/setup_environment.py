"""
AISIS Environment Setup Script
Automates the setup of development environment and dependencies
"""

import os
import subprocess
import sys
from pathlib import Path

import torch
from loguru import logger

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        logger.info(f"Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        logger.error(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.10+")
        return False

def check_cuda_availability() -> bool:
    """Check CUDA availability and version"""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        
        logger.info(f"CUDA {cuda_version} available")
        logger.info(f"GPU Count: {device_count}")
        logger.info(f"Primary GPU: {device_name}")
        return True
    else:
        logger.warning("CUDA not available - will use CPU (performance limited)")
        return False

def create_directories() -> None:
    """Create necessary project directories"""
    directories = [
        "models",
        "cache", 
        "logs",
        "config",
        "plugins",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_dependencies() -> bool:
    """Install Python dependencies"""
    try:
        logger.info("Installing dependencies...")
        
        # Install PyTorch with CUDA support
        if torch.cuda.is_available():
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True)
        
        # Install other dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        logger.info("Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def download_initial_models() -> None:
    """Download initial AI models"""
    logger.info("Downloading initial models...")
    
    models_dir = Path("models")
    
    # Create model subdirectories
    (models_dir / "whisper").mkdir(exist_ok=True)
    (models_dir / "bark").mkdir(exist_ok=True)
    (models_dir / "llm").mkdir(exist_ok=True)
    (models_dir / "diffusion").mkdir(exist_ok=True)
    
    logger.info("Model directories created")
    logger.info("Note: Models will be downloaded automatically on first use")

def create_config_files() -> None:
    """Create initial configuration files"""
    config_dir = Path("config")
    
    # Create voice presets config
    voice_presets = {
        "professional": {
            "name": "professional",
            "description": "Professional, clear voice",
            "voice_id": "v2/en_speaker_6"
        },
        "friendly": {
            "name": "friendly", 
            "description": "Warm, friendly voice",
            "voice_id": "v2/en_speaker_9"
        }
    }
    
    import json
    with open(config_dir / "voice_presets.json", "w") as f:
        json.dump(voice_presets, f, indent=2)
    
    logger.info("Configuration files created")

def setup_git_hooks() -> None:
    """Setup Git pre-commit hooks"""
    try:
        subprocess.run([sys.executable, "-m", "pre_commit", "install"], check=True)
        logger.info("Git hooks installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to install Git hooks (optional)")

def main() -> int:
    """Main setup function"""
    logger.info("Setting up AISIS development environment...")
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    # Check CUDA
    cuda_available = check_cuda_availability()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Download models
    download_initial_models()
    
    # Create config files
    create_config_files()
    
    # Setup Git hooks
    setup_git_hooks()
    
    logger.info("AISIS environment setup complete!")
    
    if cuda_available:
        logger.info("GPU acceleration is available and ready")
    else:
        logger.warning("Running in CPU mode - consider installing CUDA for better performance")
    
    logger.info("Run 'python main.py' to start AISIS")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
