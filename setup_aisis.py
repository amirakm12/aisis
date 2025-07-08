#!/usr/bin/env python3
"""
AISIS Setup Script
Helps users install dependencies and validate their AISIS installation
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", f"{version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    core_packages = [
        ("loguru", "loguru"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("huggingface_hub", "huggingface_hub"),
    ]
    
    # Optional UI dependencies
    ui_packages = [
        ("PySide6", "PySide6"),
        ("psutil", "psutil"),
    ]
    
    # Install core packages
    print("\nInstalling core packages...")
    for package, import_name in core_packages:
        if not check_package(package, import_name):
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
    
    # Install UI packages
    print("\nInstalling UI packages...")
    for package, import_name in ui_packages:
        if not check_package(package, import_name):
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"⚠️  Failed to install {package} (UI features may be limited)")

def check_gpu_support():
    """Check GPU support"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU available: {device_name}")
            return True
        else:
            print("⚠️  No CUDA GPU detected - will use CPU mode")
            return False
    except ImportError:
        print("❌ PyTorch not available - cannot check GPU status")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = ["models", "cache", "logs", "outputs", "temp"]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        print(f"✅ Directory created: {directory}")

def validate_installation():
    """Validate the AISIS installation"""
    print("\n🔍 Validating installation...")
    
    try:
        # Test model manager
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from src.core.enhanced_model_manager import enhanced_model_manager
        
        # List models
        models = enhanced_model_manager.list_models()
        print(f"✅ Model manager works - {len(models)} models available")
        
        # Test model integration
        from src.core.model_integration import model_integration
        capabilities = model_integration.get_model_capabilities()
        print(f"✅ Model integration works")
        
        # Test agent
        from src.agents.semantic_editing import SemanticEditingAgent
        agent = SemanticEditingAgent()
        print(f"✅ Semantic editing agent created")
        
        # Test main AISIS
        from src import AISIS
        aisis = AISIS()
        print(f"✅ AISIS system available")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🧪 Running quick test...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test image creation and processing
        from PIL import Image
        import numpy as np
        
        # Create test image
        array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_image = Image.fromarray(array)
        print("✅ Test image created")
        
        # Test agent capabilities
        from src.agents.semantic_editing import SemanticEditingAgent
        agent = SemanticEditingAgent()
        capabilities = agent.capabilities
        print(f"✅ Agent capabilities: {capabilities['tasks']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AISIS Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU support
    check_gpu_support()
    
    # Setup directories
    setup_directories()
    
    # Validate installation
    if not validate_installation():
        print("\n❌ Installation validation failed")
        return False
    
    # Run quick test
    if not run_quick_test():
        print("\n⚠️  Quick test had issues, but basic installation seems OK")
    
    print("\n✨ AISIS Setup Complete!")
    print("\nNext steps:")
    print("1. Run the demonstration: python demo_aisis_functionality.py")
    print("2. Launch GUI mode: python main.py gui")
    print("3. Launch CLI mode: python main.py cli")
    print("4. Read the documentation: IMPLEMENTATION_SUMMARY.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)