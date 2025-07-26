#!/usr/bin/env python3
"""
Test script for AI-ARTWORK Enhanced Installer
Verifies that all components can be imported and basic functionality works
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

def test_imports():
    """Test that all installer components can be imported"""
    print("🧪 Testing installer imports...")
    
    try:
        from enhanced_installer import (
            EnhancedInstaller, InstallationConfig, SystemInfo, 
            DependencyResolver, GPUDetector, ModelSelector,
            PathManager, ShortcutCreator, RollbackManager,
            FirstRunWizard, ProgressTracker, Colors
        )
        print("✅ All installer components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_system_detection():
    """Test system detection functionality"""
    print("\n🖥️  Testing system detection...")
    
    try:
        from enhanced_installer import GPUDetector, DependencyResolver
        
        # Test GPU detection
        gpu_info = GPUDetector.detect_gpu_info()
        print(f"GPU Available: {gpu_info['available']}")
        if gpu_info['available']:
            print(f"GPU Name: {gpu_info['primary_device']['name']}")
            print(f"CUDA Version: {gpu_info['cuda_version']}")
        
        # Test dependency resolver
        resolver = DependencyResolver()
        missing, outdated, conflicts = resolver.check_dependencies()
        print(f"Missing packages: {len(missing)}")
        print(f"Outdated packages: {len(outdated)}")
        print(f"Conflicts: {len(conflicts)}")
        
        print("✅ System detection working")
        return True
    except Exception as e:
        print(f"❌ System detection failed: {e}")
        return False

def test_model_selector():
    """Test model selection functionality"""
    print("\n🤖 Testing model selector...")
    
    try:
        from enhanced_installer import ModelSelector, SystemInfo
        
        selector = ModelSelector()
        available_models = selector.available_models
        print(f"Available models: {len(available_models)}")
        
        # Create dummy system info
        system_info = SystemInfo(
            os_type="Linux",
            architecture="x86_64", 
            python_version="3.10.0",
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            cpu_count=8,
            gpu_available=True,
            gpu_name="NVIDIA RTX 4090",
            cuda_version="12.0"
        )
        
        recommended = selector.get_recommended_models(system_info)
        compatible = selector.filter_compatible_models(system_info)
        
        print(f"Recommended models: {len(recommended)}")
        print(f"Compatible models: {len(compatible)}")
        
        print("✅ Model selector working")
        return True
    except Exception as e:
        print(f"❌ Model selector failed: {e}")
        return False

def test_progress_tracker():
    """Test progress tracking functionality"""
    print("\n📊 Testing progress tracker...")
    
    try:
        from enhanced_installer import ProgressTracker
        
        tracker = ProgressTracker(5)
        
        # Test progress updates
        tracker.update_step("Test step 1", 25.0)
        tracker.complete_step()
        
        tracker.update_step("Test step 2", 50.0)
        tracker.complete_step()
        
        print("✅ Progress tracker working")
        return True
    except Exception as e:
        print(f"❌ Progress tracker failed: {e}")
        return False

def test_configuration():
    """Test installation configuration"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from enhanced_installer import InstallationConfig
        
        config = InstallationConfig(
            install_path=Path("/tmp/test-install"),
            create_shortcuts=True,
            add_to_path=True,
            install_models=["whisper-base"],
            gpu_support=True
        )
        
        print(f"Install path: {config.install_path}")
        print(f"Models to install: {config.install_models}")
        print(f"GPU support: {config.gpu_support}")
        
        print("✅ Configuration working")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_wizard():
    """Test first-run wizard (basic functionality)"""
    print("\n🎯 Testing first-run wizard...")
    
    try:
        from enhanced_installer import FirstRunWizard
        
        wizard = FirstRunWizard(Path("/tmp/test-install"))
        
        # Test system info gathering
        system_info = wizard._get_system_info()
        print(f"Detected OS: {system_info.os_type}")
        print(f"Python version: {system_info.python_version}")
        print(f"CPU cores: {system_info.cpu_count}")
        
        print("✅ First-run wizard working")
        return True
    except Exception as e:
        print(f"❌ First-run wizard failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI-ARTWORK Enhanced Installer Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_system_detection,
        test_model_selector,
        test_progress_tracker,
        test_configuration,
        test_wizard
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installer is ready to use.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())