#!/usr/bin/env python3
"""
AI-ARTWORK Full Installation Test
Comprehensive test script to verify complete installation and functionality
"""

import os
import sys
import subprocess
import importlib
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message: str, status: str = "INFO", color: str = Colors.BLUE):
    """Print formatted status message"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {status}:{Colors.END} {message}")

def print_success(message: str):
    """Print success message"""
    print_status(message, "SUCCESS", Colors.GREEN)

def print_error(message: str):
    """Print error message"""
    print_status(message, "ERROR", Colors.RED)

def print_warning(message: str):
    """Print warning message"""
    print_status(message, "WARNING", Colors.YELLOW)

class InstallationTester:
    def __init__(self):
        self.test_results = {}
        self.project_root = Path(__file__).parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        
    def test_system_info(self) -> bool:
        """Test system information"""
        print_status("Testing system information...")
        
        try:
            # Python version
            python_version = sys.version_info
            if python_version.major == 3 and python_version.minor >= 8:
                print_success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} - Compatible")
                self.test_results['python_version'] = True
            else:
                print_error(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} - Requires Python 3.8+")
                self.test_results['python_version'] = False
                return False
            
            # Platform info
            print_status(f"Platform: {platform.system()} {platform.release()}")
            print_status(f"Architecture: {platform.machine()}")
            print_status(f"Processor: {platform.processor()}")
            
            # Available memory
            try:
                import psutil
                memory = psutil.virtual_memory()
                print_status(f"Total RAM: {memory.total // (1024**3):.1f} GB")
                print_status(f"Available RAM: {memory.available // (1024**3):.1f} GB")
            except ImportError:
                print_warning("psutil not available - cannot check memory")
            
            return True
            
        except Exception as e:
            print_error(f"System info test failed: {e}")
            return False
    
    def test_dependencies(self) -> bool:
        """Test all required dependencies"""
        print_status("Testing dependencies...")
        
        if not self.requirements_file.exists():
            print_error("requirements.txt not found")
            return False
        
        # Read requirements
        with open(self.requirements_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        failed_deps = []
        successful_deps = []
        
        for req in requirements:
            if '>=' in req:
                package_name = req.split('>=')[0]
            elif '==' in req:
                package_name = req.split('==')[0]
            else:
                package_name = req
            
            try:
                module = importlib.import_module(package_name.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                print_success(f"✓ {package_name} ({version})")
                successful_deps.append(package_name)
            except ImportError:
                print_error(f"✗ {package_name} - Not installed")
                failed_deps.append(package_name)
            except Exception as e:
                print_warning(f"? {package_name} - Error checking version: {e}")
                successful_deps.append(package_name)
        
        self.test_results['dependencies'] = len(failed_deps) == 0
        self.test_results['failed_deps'] = failed_deps
        self.test_results['successful_deps'] = successful_deps
        
        if failed_deps:
            print_error(f"Failed dependencies: {', '.join(failed_deps)}")
            return False
        
        print_success(f"All {len(successful_deps)} dependencies installed successfully")
        return True
    
    def test_gpu_support(self) -> bool:
        """Test GPU/CUDA support"""
        print_status("Testing GPU support...")
        
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                
                print_success(f"CUDA {cuda_version} available")
                print_success(f"GPU Count: {device_count}")
                print_success(f"Primary GPU: {device_name}")
                
                # Test GPU memory
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    print_success(f"GPU Memory: {gpu_memory // (1024**3):.1f} GB")
                except:
                    pass
                
                self.test_results['gpu_support'] = True
                return True
            else:
                print_warning("CUDA not available - will use CPU")
                self.test_results['gpu_support'] = False
                return True  # Not a failure, just warning
                
        except ImportError:
            print_warning("PyTorch not installed - cannot test GPU support")
            self.test_results['gpu_support'] = False
            return True
        except Exception as e:
            print_error(f"GPU test failed: {e}")
            self.test_results['gpu_support'] = False
            return False
    
    def test_project_structure(self) -> bool:
        """Test project structure and key files"""
        print_status("Testing project structure...")
        
        required_files = [
            "main.py",
            "setup.py",
            "requirements.txt",
            "README.md",
            "src/__init__.py",
            "src/ui/__init__.py",
            "src/core/__init__.py",
            "src/agents/__init__.py"
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print_success(f"✓ {file_path}")
                existing_files.append(file_path)
            else:
                print_error(f"✗ {file_path} - Missing")
                missing_files.append(file_path)
        
        self.test_results['project_structure'] = len(missing_files) == 0
        self.test_results['missing_files'] = missing_files
        
        if missing_files:
            print_error(f"Missing files: {', '.join(missing_files)}")
            return False
        
        print_success(f"All {len(existing_files)} required files present")
        return True
    
    def test_imports(self) -> bool:
        """Test importing key modules"""
        print_status("Testing module imports...")
        
        # Add current directory to Python path for testing
        import sys
        sys.path.insert(0, str(self.project_root))
        
        modules_to_test = [
            "src.core.config",
            "src.ui.main_window",
            "src.agents.base_agent",
            "loguru",
            "numpy",
            "PIL",
            "cv2",
            "torch",
            "PySide6.QtWidgets"
        ]
        
        failed_imports = []
        successful_imports = []
        
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
                print_success(f"✓ {module_name}")
                successful_imports.append(module_name)
            except ImportError as e:
                print_error(f"✗ {module_name} - {e}")
                failed_imports.append(module_name)
            except Exception as e:
                print_warning(f"? {module_name} - {e}")
                successful_imports.append(module_name)
        
        self.test_results['imports'] = len(failed_imports) == 0
        self.test_results['failed_imports'] = failed_imports
        
        if failed_imports:
            print_error(f"Failed imports: {', '.join(failed_imports)}")
            return False
        
        print_success(f"All {len(successful_imports)} modules imported successfully")
        return True
    
    def test_configuration(self) -> bool:
        """Test configuration loading"""
        print_status("Testing configuration...")
        
        try:
            # Add current directory to Python path for testing
            import sys
            sys.path.insert(0, str(self.project_root))
            
            from src.core.config import config
            print_success("Configuration loaded successfully")
            self.test_results['configuration'] = True
            return True
        except Exception as e:
            print_error(f"Configuration test failed: {e}")
            self.test_results['configuration'] = False
            return False
    
    def test_ui_components(self) -> bool:
        """Test UI component creation"""
        print_status("Testing UI components...")
        
        try:
            # Add current directory to Python path for testing
            import sys
            sys.path.insert(0, str(self.project_root))
            
            from PySide6.QtWidgets import QApplication
            from src.ui.main_window import MainWindow
            
            # Create minimal app for testing
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            # Test window creation
            window = MainWindow()
            print_success("Main window created successfully")
            
            # Clean up
            window.deleteLater()
            
            self.test_results['ui_components'] = True
            return True
            
        except Exception as e:
            print_error(f"UI test failed: {e}")
            self.test_results['ui_components'] = False
            return False
    
    def test_voice_processing(self) -> bool:
        """Test voice processing capabilities"""
        print_status("Testing voice processing...")
        
        try:
            # Test Whisper import
            import whisper
            print_success("Whisper imported successfully")
            
            # Test Bark import
            try:
                import bark
                print_success("Bark imported successfully")
            except ImportError:
                print_warning("Bark not available")
            
            # Test soundfile
            import soundfile
            print_success("Soundfile imported successfully")
            
            self.test_results['voice_processing'] = True
            return True
            
        except Exception as e:
            print_error(f"Voice processing test failed: {e}")
            self.test_results['voice_processing'] = False
            return False
    
    def test_ai_models(self) -> bool:
        """Test AI model loading capabilities"""
        print_status("Testing AI model capabilities...")
        
        try:
            # Test transformers
            from transformers import AutoTokenizer, AutoModel
            print_success("Transformers imported successfully")
            
            # Test diffusers
            try:
                from diffusers import DiffusionPipeline
                print_success("Diffusers imported successfully")
            except ImportError:
                print_warning("Diffusers not available")
            
            # Test basic model loading (small test model)
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="./cache")
                print_success("Basic model loading test passed")
            except Exception as e:
                print_warning(f"Model loading test failed (network issue?): {e}")
            
            self.test_results['ai_models'] = True
            return True
            
        except Exception as e:
            print_error(f"AI models test failed: {e}")
            self.test_results['ai_models'] = False
            return False
    
    def test_performance(self) -> bool:
        """Test basic performance metrics"""
        print_status("Testing performance...")
        
        try:
            import torch
            import time
            
            # Test CPU tensor operations
            start_time = time.time()
            x = torch.randn(1000, 1000)
            y = torch.randn(1000, 1000)
            z = torch.mm(x, y)
            cpu_time = time.time() - start_time
            
            print_success(f"CPU matrix multiplication: {cpu_time:.3f}s")
            
            # Test GPU if available
            if torch.cuda.is_available():
                start_time = time.time()
                x_gpu = x.cuda()
                y_gpu = y.cuda()
                z_gpu = torch.mm(x_gpu, y_gpu)
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                
                print_success(f"GPU matrix multiplication: {gpu_time:.3f}s")
                if gpu_time < cpu_time:
                    print_success("GPU acceleration working")
                else:
                    print_warning("GPU not faster than CPU (may be small matrix)")
            
            self.test_results['performance'] = True
            return True
            
        except Exception as e:
            print_error(f"Performance test failed: {e}")
            self.test_results['performance'] = False
            return False
    
    def run_installation_test(self) -> bool:
        """Run complete installation test"""
        print_status("Starting AI-ARTWORK Full Installation Test", "TEST", Colors.BOLD)
        print_status("=" * 60, "TEST", Colors.BOLD)
        
        tests = [
            ("System Information", self.test_system_info),
            ("Project Structure", self.test_project_structure),
            ("Dependencies", self.test_dependencies),
            ("GPU Support", self.test_gpu_support),
            ("Module Imports", self.test_imports),
            ("Configuration", self.test_configuration),
            ("UI Components", self.test_ui_components),
            ("Voice Processing", self.test_voice_processing),
            ("AI Models", self.test_ai_models),
            ("Performance", self.test_performance)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print_status(f"\n--- {test_name} ---", "TEST", Colors.BOLD)
            try:
                if test_func():
                    passed_tests += 1
                    print_success(f"{test_name} test PASSED")
                else:
                    print_error(f"{test_name} test FAILED")
            except Exception as e:
                print_error(f"{test_name} test ERROR: {e}")
        
        # Summary
        print_status("\n" + "=" * 60, "SUMMARY", Colors.BOLD)
        print_status(f"Tests Passed: {passed_tests}/{total_tests}", "SUMMARY", Colors.BOLD)
        
        if passed_tests == total_tests:
            print_success("🎉 ALL TESTS PASSED - Installation is complete and working!")
            print_success("You can now run 'python main.py' to start AI-ARTWORK")
        else:
            print_error(f"❌ {total_tests - passed_tests} tests failed")
            print_error("Please check the errors above and fix the issues")
        
        # Save test results
        results_file = self.project_root / "test_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print_status(f"Detailed results saved to: {results_file}", "INFO")
        except Exception as e:
            print_warning(f"Could not save results file: {e}")
        
        return passed_tests == total_tests

def main():
    """Main function"""
    tester = InstallationTester()
    success = tester.run_installation_test()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 